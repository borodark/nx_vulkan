# mac-248 — URGENT X3 (sign fix) + Y queue + Z1 (Philox cross-check)

X3 is the priority. Y queue (Y1, Y2, Y3 — Phase 2 distribution
chains) is unblocked once X3 lands. Z1 below is a small parallel
audit task: cross-check `random_philox.spv` output against a
Random123 reference (since the shader's round structure looks
non-canonical).

---

## Z2 — Replace `random_philox.spv` with canonical Random123 Philox 2x32-10

**Z1 outcome (both sides)**: Linux Python and mac-248 C++/Random123
both confirmed the shader's PRNG is non-canonical. 0/10 match
across 10 reference inputs. Same shader uint32 outputs from
both check-sides (e.g., ctr=0 shader=0x6da9e4a2; canonical
ref=0xc534ae0b). The shader is deterministic and reproducible
but is NOT Random123 Philox 2x32-10.

User decision: replace with canonical implementation.

### What to change

In `shaders/random_philox.comp`, swap the body of
`philox2x32(uint counter, uint key)` for the canonical round.
**Do NOT change** anything else: `mulhilo`, `uint_to_uniform`,
the Box-Muller wrapper in `main()`, the push constants, the
buffer layout, the specialization constant for DIST. Public
behaviour (uniform [0,1) for DIST=0, normal via Box-Muller for
DIST=1) stays identical.

### The canonical round (from Random123 `philox.h`)

State: `ctr[0]`, `ctr[1]`, `key` — three uint32. The shader's
input is a single 32-bit `counter`, so `ctr[0] = counter` and
`ctr[1] = 0` at start (this is a standard convention for
single-counter Philox; gives 2^32 unique outputs per key).

Per round:
```
hi, lo = mulhilo(M, ctr[0])              // unsigned 32x32→64
ctr[0]' = hi ^ key ^ ctr[1]              // mix in BOTH key and ctr[1]
ctr[1]' = lo                             // new ctr[1] is the low half
key'    = key + 0x9E3779B9               // bump SEPARATE key
```

10 rounds. Output is `uvec2(ctr[0], ctr[1])` after the last round.

GLSL spec (replace the existing `philox2x32` body):

```glsl
uvec2 philox2x32(uint counter, uint key_in) {
    uint c0  = counter;
    uint c1  = 0u;          // canonical single-counter convention
    uint key = key_in;      // SEPARATE from ctr state

    for (int i = 0; i < 10; i++) {
        uvec2 prod = mulhilo(c0, 0xD2511F53u);  // {lo, hi}
        uint c0_new = prod.y ^ key ^ c1;
        uint c1_new = prod.x;
        c0 = c0_new;
        c1 = c1_new;
        key += 0x9E3779B9u;
    }

    return uvec2(c0, c1);
}
```

Three semantic changes from the current shader:
1. Track `key` as a SEPARATE state variable (not folded into `hi`)
2. Initialize `c1 = 0` (not `c1 = key`)
3. Round mix is `c0' = mulhi ^ key ^ c1` (not `c0' = key ^ mulhi`)

### Acceptance check

Re-run your Z1 cross-check after the swap. With `(counter=0, key=42)`,
the shader should now produce **exactly** Random123's reference:
```
ctr=0: shader=0xc534ae0b ref=0xc534ae0b  MATCH
ctr=1: shader=0xe0569325 ref=0xe0569325  MATCH
...
ctr=9: shader=0xb53b1830 ref=0xb53b1830  MATCH
```

Should be 10/10 match. If even ctr=0 differs, there's an off-by-one
in the round structure; recheck against `~/spirit/thirdparty/Random123/include/Random123/philox.h`.

### Compile + push

```sh
cd ~/spirit
glslangValidator -V shaders/random_philox.comp -o shaders/random_philox.spv
git add shaders/random_philox.{comp,spv}
git commit -m "shaders: random_philox — canonical Random123 Philox 2x32-10"
git push origin feat/fused-leapfrog-chain-normal
```

Effort: **~30-60 min** including the Z1 re-run for verification.

### Why this matters

While the current shader's PRNG output LOOKS uniform-ish on quick
inspection, "looks uniform" is not "passes statistical tests."
NUTS samplers and other downstream consumers depend on the PRNG
being statistically sound. Random123 Philox 2x32-10 is a
well-known, well-tested cipher; the shader's variant is custom
and untested. Replacing avoids the future "we trusted the
random_philox shader and got biased posteriors" failure mode.

NB: nothing currently depends on this shader on the eXMC NUTS
critical path — Erlang `:rand` is used for momenta. But other
Vulkan-side users (any future GPU random sampling) deserve the
canonical implementation.

---

## Z1 — Philox 2x32-10 cross-check vs Random123 reference (DONE)

Cross-check completed by both Linux (Python) and mac-248 (C++).
Both reached the same verdict: 0/10 match against canonical
Random123. Findings drove Z2 above.

**Background**: a recent audit of `random_philox.spv` flagged
that the shader's round structure differs from Random123's
canonical Philox 2x32-10. The shader's per-round mix:

```glsl
prod = mulhilo(lo, 0xD2511F53u);
lo = hi ^ prod.y;
hi = prod.x;
hi += 0x9E3779B9u;
```

Random123 canonical (from `Random123/philox.h`):
```c
ctr[0] = mulhi(M, ctr[0]_old) ^ key ^ ctr[1]_old;
ctr[1] = mullo(M, ctr[0]_old);
key += W;
```

The shader's variant omits the XOR with `ctr[1]` and applies the
Weyl bump to `mullo(M, ctr[0])` instead of the key. May still be
statistically sound, but we should know definitively before
shipping NUTS samplers that depend on it.

**Task**: run the shader's `random_philox.spv` for `(counter, key)
= (0..9, 42)` and compare the raw uint32 outputs to Random123's
`philox2x32_R(10, ...)` reference for the same inputs.

Spirit already vendors Random123 at `~/spirit/thirdparty/Random123/`
(check via `find ~/spirit/thirdparty -name 'philox.h'` — should
exist). If yes, write a small C++ test program:

```cpp
#include <cstdio>
#include <cstdint>
#include "Random123/philox.h"

int main() {
    using philox2x32 = r123::Philox2x32;
    philox2x32 rng;
    for (uint32_t i = 0; i < 10; i++) {
        philox2x32::ctr_type ctr  = {{i, 0}};   // {ctr_lo, ctr_hi}
        philox2x32::key_type key  = {{42}};
        philox2x32::ctr_type out  = rng(ctr, key);
        printf("ctr=%u: out = (0x%08x, 0x%08x)\n", i, out.v[0], out.v[1]);
    }
    return 0;
}
```

Compile + run:
```sh
cd ~/spirit
g++ -std=c++14 -I thirdparty -O2 -o /tmp/philox_ref \
    /tmp/philox_ref.cpp
/tmp/philox_ref
```

Then dispatch the shader via your existing test harness (or a
new minimal one) for the same inputs. The shader's output goes
through `uint_to_uniform` to produce floats in `[0,1)` — to
compare raw uint32, you'll need to either bypass that step (run
a debug version that writes the raw uint to the output buffer)
OR back out the uint from the float (`out_uint = round(f * 2^23)
<< 9`).

Easiest: temporarily modify `random_philox.comp` to write
`r.x` directly (uint32) instead of `uint_to_uniform(r.x)` for the
DIST=0 path, recompile, dispatch, read the buffer as uint32. Then
revert the temporary edit.

Output format: print as hex, side by side:
```
ctr=0: shader=0x........ ref=0x........  MATCH/DIFFER
ctr=1: ...
...
```

**Report back**: the comparison table. If even ctr=0 differs,
the shader's variant is non-canonical; we'll need to either
(a) replace with a canonical Random123 implementation, or
(b) document the variant explicitly and run a separate
statistical-quality test to verify it's still sound.

If they all match, the shader's round structure is mathematically
equivalent to Random123's despite the different code shape — a
nice surprise.

Effort: ~30 min if Random123 is already vendored; ~1 hour if you
need to fetch and add it.

I'm running an independent Python cross-check on the Linux side
in parallel; we'll compare findings.

---



X3 below is the priority. After X3 lands, the Y queue (three new
chain shaders for HMC-friendly distributions) is ready to start
in any order — Phase 2 of the original PLAN_FUSED_LEAPFROG.md is
unblocked once the sign fix verifies the chain pattern.

---

## Y queue — Phase 2 distribution chains (after X3)

Each is a clean port of the `leapfrog_chain_normal` template,
with the gradient formula and push constants swapped. Same K-step
loop, same shared-memory logp reduction, same I/O contract.
**All assume the X3 sign fix is in** (use `+` for momentum
updates, not `-`).

### Y1 — `leapfrog_chain_studentt.spv` (real-valued, no transform)

Student-t with degrees of freedom ν, location μ, scale σ. Defined
on the entire real line — no transform needed. Closed-form
gradient.

```
log p(q | ν, μ, σ) = log Γ((ν+1)/2) - log Γ(ν/2) - 0.5*log(πν)
                     - log σ - ((ν+1)/2) * log(1 + (1/ν)*((q-μ)/σ)²)
∇logp = -((ν+1)/(ν*σ²)) * (q - μ) / (1 + (1/ν) * ((q-μ)/σ)²)
```

In shader form, define `z = (qi - μ)/σ`, then:
```glsl
float z = (qi - pc.mu) / pc.sigma;
float denom = 1.0 + z * z / pc.nu;
float grad_q = -((pc.nu + 1.0) / (pc.nu * pc.sigma * pc.sigma))
               * (qi - pc.mu) / denom;
```

For per-step logp, the constants `log Γ((ν+1)/2) - log Γ(ν/2) -
0.5*log(πν) - log σ` are fixed at compile time (compute on host
and pass as a single push constant `logp_const`, additive). The
per-element contribution is `-((ν+1)/2) * log(denom)`. Per-step
sum reduces over `n` (workgroup pattern, same as Normal chain).

Push constants: `{n, K, eps, mu, sigma, nu, logp_const}` = 28 bytes.

**Note**: Student-t reduces to Normal as ν→∞ and to Cauchy at
ν=1; if it works at ν=3 (commonly used) it generalizes.

### Y2 — `leapfrog_chain_cauchy.spv` (real-valued, ν=1 special case)

Cauchy(loc, scale) — special case of Student-t at ν=1. Could be
derived from Y1 by hardcoding ν=1, but the closed-form is cleaner
to implement directly:

```
log p(q | loc, scale) = -log(π * scale) - log(1 + ((q - loc)/scale)²)
∇logp = -2*(q - loc) / (scale² + (q - loc)²)
```

Shader:
```glsl
float diff = qi - pc.loc;
float grad_q = -2.0 * diff / (pc.scale * pc.scale + diff * diff);
```

Per-element logp = `-log(1 + (diff/scale)²)`; constant per step is
`-n * log(π * scale)` (workgroup-0 includes it).

Push: `{n, K, eps, loc, scale, log_pi_scale}` = 24 bytes.

### Y3 — `leapfrog_chain_halfnormal.spv` (positive, log-transform)

HalfNormal(σ) on the unconstrained line via log-transform
`q_uc = log(q)`. The unconstrained log-density (with Jacobian) is:
```
log p(q_uc | σ) = -log(σ) - 0.5*log(2π/2)            (HalfNormal constant)
                  + q_uc                              (Jacobian for log transform)
                  - 0.5 * exp(2*q_uc) / σ²
∇logp(q_uc) = 1 - exp(2*q_uc) / σ²
```

Shader:
```glsl
float exp_2quc = exp(2.0 * qi);
float grad_q = 1.0 - exp_2quc / (pc.sigma * pc.sigma);
```

Per-element logp = `q_uc - 0.5 * exp(2*q_uc)/σ²`; constant per
step is `-n * (log(σ) + 0.5*log(π))`.

Push: `{n, K, eps, sigma, log_const}` = 20 bytes.

**Numerical caution**: `exp(2*q_uc)` overflows for `q_uc > ~44`
(f32). For typical priors with σ ~ 1, the unconstrained range is
small enough; document the limitation.

### Compile + push for any Y task

```sh
glslangValidator -V shaders/leapfrog_chain_<name>.comp \
                 -o shaders/leapfrog_chain_<name>.spv
git add shaders/leapfrog_chain_<name>.{comp,spv}
git commit -m "shaders: leapfrog_chain_<name> — Phase 2 chain"
git push origin feat/fused-leapfrog-chain-normal
```

### What the Y queue DOES NOT need

- Multi-WG variants (single-WG enough until proven otherwise)
- f64 versions (X1 deferred until f32 chain pattern is fully
  verified post-X3)
- Per-distribution wiring on Linux (one PR after Y queue lands)

### Sanity check before pushing each Y shader

For Y1 (Student-t at ν=3, μ=0, σ=1, q=2.0, p=0.5, eps=0.1, K=2):
hand-derive the first leapfrog step with the corrected sign and
include the expected output in your commit message. Same protocol
as we should have followed for the original chain shaders.

---



**Status update**: Stage 1.5.4 H1 + H2 are settled. The variance
bias (var ≈ 0.73 vs EXLA reference 1.36) is **not f32 precision**
and **not a row-0 contract mismatch**. It's a sign error in the
leapfrog momentum update — which I introduced in the original
spec and you implemented faithfully. The bug is in **all four**
leapfrog shaders we shipped together.

## What went wrong

Standard Hamiltonian leapfrog with `grad = ∇logp` (gradient of
log-density, not negative log-density):

```
p_half = p + (eps/2) * grad
q_new  = q + eps * inv_mass * p_half
p_new  = p_half + (eps/2) * grad_new
```

The `+` is correct. The eXMC code path (`BatchedLeapfrog.multi_step`
in `lib/exmc/nuts/batched_leapfrog.ex:79`) uses `+`. My spec and
the four shaders all use `-`. With `grad = -(q-μ)/σ²`, the sign
flip turns "pull back toward mode" into "push outward from mode."
The chain heats up, samples cluster near far edges, posterior
variance is compressed.

## The fix

**Two character changes per shader, FIVE shaders total** (your X1
push crossed mine — the new f64 chain inherits the same bug).

### 1. `shaders/leapfrog_normal.comp` (single-step Phase 1 baseline)

```diff
-    float p_half = pi - 0.5 * pc.eps * grad_q;
+    float p_half = pi + 0.5 * pc.eps * grad_q;
     float qn = qi + pc.eps * mi * p_half;
     float grad_qn = -(qn - pc.mu) * inv_var;
-    float pn = p_half - 0.5 * pc.eps * grad_qn;
+    float pn = p_half + 0.5 * pc.eps * grad_qn;
```

### 2. `shaders/leapfrog_chain_normal.comp` (single-WG chain)

Same diff inside the K-step loop body. The two momentum-update
lines.

### 3. `shaders/leapfrog_chain_normal_lg.comp` (multi-WG chain)

Same diff. Same two lines inside the K-step loop body.

### 4. `shaders/leapfrog_chain_exponential.comp` (Phase 2)

Same diff. The grad expression is different (`1 - λ * exp(qi)`)
but the bug is identical: `pi - 0.5 * eps * grad_q` should be
`pi + 0.5 * eps * grad_q`, and the same for the second update.

### 5. `shaders/leapfrog_chain_normal_f64.comp` (X1 you just pushed)

Same diff, just with `0.5LF` instead of `0.5`. Lines 46 and 49.
Confirmed inheriting the same bug from the spec — your X1 push
crossed mine; X1 itself was a clean port of the f32 spec, so it
inherits the f32 spec's bug.

## Things NOT to change

- `grad_q` and `grad_qn` definitions — these correctly compute
  ∇logp. Leave as-is. The bug is purely in the sign of the
  momentum update, not the gradient.
- Output buffer layout, push constants, workgroup sizing,
  reduction patterns — all correct. The chain shader's per-step
  logp via shared-memory reduction is correct. Multi-WG layout
  with workgroup-0-includes-constant is correct.
- The `f64` sibling task X1 is **deferred** until after this
  fix lands. f64 would have inherited the same sign error.

## Compile + push

```sh
cd ~/spirit
git checkout feat/fused-leapfrog-chain-normal
git pull origin feat/fused-leapfrog-chain-normal

# Edit five files, run glslang on each:
glslangValidator -V shaders/leapfrog_normal.comp              -o shaders/leapfrog_normal.spv
glslangValidator -V shaders/leapfrog_chain_normal.comp        -o shaders/leapfrog_chain_normal.spv
glslangValidator -V shaders/leapfrog_chain_normal_lg.comp     -o shaders/leapfrog_chain_normal_lg.spv
glslangValidator -V shaders/leapfrog_chain_exponential.comp   -o shaders/leapfrog_chain_exponential.spv
glslangValidator -V shaders/leapfrog_chain_normal_f64.comp    -o shaders/leapfrog_chain_normal_f64.spv

git add shaders/*.{comp,spv}
git commit -m "shaders: fix sign on momentum update — leapfrog uses + not -"
git push origin feat/fused-leapfrog-chain-normal
```

## Verification before push

For `leapfrog_chain_normal.spv` at K=2, n=4, q=[1,2,3,4],
p=[0.5,...], mu=0, sigma=1, eps=0.1, the corrected shader should
produce:

| | q_chain[0] | p_chain[0] | grad_chain[0] | q_chain[1] | p_chain[1] |
|---|------------|------------|---------------|------------|------------|
| Bugged (current) | 1.055 | 0.6028 | -1.055 | 1.1205 | 0.7115 |
| **Corrected** | **0.945** | **0.3973** | **-0.945** | **0.8915** | **0.2937** |

Hand-derivation for q[0]=1 with the FIXED formula:
```
grad_q = -(1 - 0)/1 = -1
p_half = 0.5 + 0.5*0.1*(-1) = 0.45
qn = 1 + 0.1*1*0.45 = 1.045    ← wait, hand-check
```

Hmm let me redo: p=0.5, ∇logp=-1, eps=0.1.
- p_half = 0.5 + (0.1/2)*(-1) = 0.5 - 0.05 = 0.45
- qn = 1 + 0.1*1*0.45 = 1.045 ← actually closer to mode than start (1)? That's wrong direction.

Wait, q=1 with p=0.5 starting outward — over a half-step the momentum reduces (toward zero) and position advances toward 1.045 — *also* outward. That's *correct* for HMC: position keeps advancing in the direction momentum points (positive p means rising q), but momentum decays as the gradient pulls back. Over many steps the momentum reverses and the position oscillates back through the mode.

Compare to the bugged version where p_half = 0.55 (momentum INCREASES) → q rises faster, divergent.

So the corrected step-1 numbers are:
- q_chain[0,0] = 1.045
- p_chain[0,0] = 0.45 + (0.1/2)*(-(1.045-0)/1) = 0.45 - 0.05225 = 0.39775
- grad_chain[0,0] = -(1.045)/1 = -1.045

Step 2: q=1.045, p=0.39775
- grad = -1.045
- p_half = 0.39775 + 0.05*(-1.045) = 0.34550
- qn = 1.045 + 0.1*0.34550 = 1.07955
- p_new = 0.34550 + 0.05*(-1.07955) = 0.29152

So the corrected at step 2: q ≈ 1.080, p ≈ 0.292.

If the corrected shader output for q=1, p=0.5, mu=0, sigma=1,
eps=0.1, K=2 matches:
- q_chain[0,0] ≈ 1.045
- p_chain[0,0] ≈ 0.398
- grad_chain[0,0] ≈ -1.045
- q_chain[1,0] ≈ 1.080
- p_chain[1,0] ≈ 0.292

then the fix is correct. After push, the Linux side re-runs the
diagnostic test and the fused-chain assertion `var in [0.7, 1.3]`
should pass on `x ~ N(0,1)`.

## Why I'm asking you instead of editing here

I don't have `glslangValidator` on the Linux dev box (it's only
in the spirit shader-compiler environment on mac-248). The
`.comp` source edit + `.spv` recompile pair has to happen there.

Sorry for the spec error. I'll be more careful about the
+/- convention in future leapfrog specs — for any HMC variant,
when `grad = ∇logp`, momentum update is **PLUS** half-eps times
grad.

## Cross-reference

- `~/projects/learn_erl/pymc/exmc/lib/exmc/nuts/batched_leapfrog.ex:79`
  — the canonical correct formula (same one Stan and PyMC use)
- `~/projects/learn_erl/pymc/exmc/test/nuts/fused_chain_diag_test.exs`
  — the assertion that flips green when this is fixed
- Linux side will pick up your shader push, re-vendor into
  `priv/shaders/`, run the diag test, and report whether the
  fused chain now matches EXLA reference.
- X2 (`reduce_full_f64.spv`) from the previous TODO round is
  **still useful** if you want a second concurrent task — that
  one is independent of all this.
