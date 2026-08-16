# T10 — the "Ampere over-dispersion", diagnosed

**Date:** 2026-08-15 · **Investigator branch:** `invest/t10-ampere-dispersion`
**nx_vulkan commit:** `be3ba07` (worktree) · **eXMC commit:** `e8d3c55b4`
(`/home/io/projects/learn_erl/pymc/exmc`, branch `feat/news-signal-fold-in`)

**Hosts.** super-io — Linux, **NVIDIA GeForce RTX 3060 Ti (DiscreteGpu)**,
Ampere, confirmed real device via `Nx.Vulkan.NativeV.device_name/0`. mac-247 —
FreeBSD 15.0, **NVIDIA GeForce GT 650M (DiscreteGpu)**, Kepler, confirmed via
the vulkano NIF init line (`device_name/0` does not exist on that checkout,
`nx_vulkan@9f936b7`).

---

## The one-sentence answer

**There is no Ampere over-dispersion.** The synthesised chain shader produces
**bit-identical** trajectories on Ampere and Kepler for fixed inputs. The
over-dispersion is real, reproduces on super-io today, and is caused by a
**host-independent off-by-one in eXMC's synthesised GLSL**: `logp_chain[k]`
holds the log-density of the position *before* leapfrog step k, while
`q_chain[k]`, `p_chain[k]` and `grad_chain[k]` hold the state *after* it. The
Kepler fleet never caught it because on a host without EXLA the validator's
"EXLA reference" arm silently resolves to Vulkan and compares the chain shader
against itself.

The defect is in **eXMC**, not `nx_vulkan`:
`exmc/lib/exmc/nuts/custom_synth/multi_rv_custom_spec.ex`, `@template` (and the
same bug in `@batched_template`). `nx_vulkan`'s own
`Nx.Vulkan.ShaderTemplate` skeleton — the one eXMC's template was generalised
from — has the ordering **right**; the multi-RV generalisation moved the
log-prob body above the position update. (That `nx_vulkan` path is itself stale
for an unrelated reason — see the side finding below — so this is a statement
about provenance, not about a shader anyone currently dispatches.)

---

## What was measured

### 1. Fixed-input differential, chain shader vs host, on Ampere

One `leapfrog_chain_synth_f64` dispatch per row. Fixed
`q_i = 0.3 + 0.17i`, `p_i = 1.0 − 0.23i`, `inv_mass_i = 1.0 + 0.5i`,
`eps = 0.05`, `dir_sign = +1`. Reference: the same leapfrog composed in Elixir
from `Nx.Defn.grad` of `MultiRvCustomSpec.compose_logp_defn/1` on
`Nx.Defn.Evaluator` / `BinaryBackend` — i.e. the identical math, host-side.
Max elementwise absolute difference:

| model | d | K | Δq | Δp | Δgrad | Δlogp |
|---|--:|--:|--:|--:|--:|--:|
| Normal(0,1) | 1 | 32 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 4.4e-16 |
| Exponential(λ=2) | 1 | 32 | 5.6e-17 | 1.7e-17 | 0.0 | 2.2e-16 |
| Cauchy(0,1) | 1 | 32 | 1.1e-16 | 5.6e-17 | 2.2e-16 | 0.0 |
| HalfNormal(σ=1) | 1 | 32 | 1.1e-16 | 8.7e-19 | 0.0 | 2.2e-16 |
| Normal+Exponential+Normal | 3 | 32 | 2.2e-16 | 2.2e-16 | 2.2e-16 | 8.9e-16 |

K ∈ {1, 2, 4, 32} all behave the same; only K=32 is tabulated. **1–4 ulp at
f64.** The chain shader's arithmetic on Ampere is correct.

Note on the Δlogp column: the host reference was written to mirror the
shader's control flow *exactly*, including evaluating the log-density at the
pre-update position. So Δlogp ≈ 0 confirms the shader computes what its source
says it computes — it does **not** validate that the source is right. That is
§4's job.

### 2. Fixed-input differential, Ampere vs Kepler

The rendered GLSL and the exact `(q, p, extras, push, K)` byte-for-byte inputs
were shipped to mac-247, recompiled there with the local `glslangValidator`,
and dispatched through `Nx.Vulkan.NativeV.leapfrog_chain_synth_f64/6` — no eXMC
on that host, so the eXMC branch skew (mac-247 is on `feat/168-ssbo-obs`) is
irrelevant to the comparison.

| model | K | q | p | grad | logp |
|---|--:|---|---|---|---|
| Normal(0,1), d=1 | 1, 2, 4 | **identical** | identical | identical | identical |
| Normal(0,1), d=1 | 32 | **identical** | identical | identical | 2.2e-16 (1 ulp, one entry) |
| 3-RV mixed, d=3 | 1, 2, 4, 32 | **identical** | identical | identical | **identical** |

The trajectory — the thing that determines the posterior — is **bit-identical
across the two architectures** in all 8 cases. A single 1-ulp difference in one
of 32 log-density values cannot produce a 5.9× variance ratio.

*This closes T10's step 1 and step 2 with a negative result: the chain shader is
exonerated as an architecture-dependent defect, and the "Ampere is untrustworthy
for numerics" premise is false.*

### 3. The over-dispersion is real and reproduces — against a real reference

`Exmc.NUTS.Sampler.sample/3`, 300 warmup + 800 samples, seed 42, super-io.
Reference arm forced to `compiler: :none` (`Nx.Defn.Evaluator` + `BinaryBackend`,
f64); Vulkan arm is the synthesised f64 chain shader.

| model | arm | mean | variance | analytic |
|---|---|--:|--:|---|
| Normal(0,1) | CPU (`:none`) | −0.0353 | **1.4523** | 0.0 / 1.0 |
| Normal(0,1) | Vulkan chain | −0.1223 | **8.5467** | — |
| | | | **5.885× variance** | |
| Exponential(λ=2) | CPU (`:none`) | **0.5021** | 0.2600 | 0.5 / 0.25 |
| Exponential(λ=2) | Vulkan chain | **1.4477** | 2.8661 | — |
| | | **2.88× mean** | **11.02× variance** | |
| HalfNormal(σ=1) | CPU (`:none`) | 0.8056 | 0.3773 | 0.7979 / 0.3634 |
| HalfNormal(σ=1) | Vulkan chain | 1.3943 | 1.7459 | — |
| | | **1.73× mean** | **4.63× variance** | |

The Normal(0,1) numbers reproduce D90's report (variance **8.5** against
**1.45**) to three significant figures — but note that D90's 1.45 was never
EXLA either: it is what the CPU arm gives, and it is itself 45% above the
analytic 1.0. Exponential reproduces in the same direction (D90 reported mean
2.77 against 0.5; at this warmup/sample count it is 1.45 against 0.50, with the
CPU arm landing on the analytic value to 3 decimal places). The CPU arm is
close to analytic for Exponential and HalfNormal; only its Normal variance
(1.4523 against 1.0) is itself off, and that is the very number D90 quoted as
"EXLA's 1.45".

The over-dispersion is therefore **real, current, and reproducible on super-io
at HEAD** — but per §2 it is not caused by the GPU.

---

## The defect

### `logp_chain[k]` lags the rest of the chain by one leapfrog step

`exmc/lib/exmc/nuts/custom_synth/multi_rv_custom_spec.ex`, `@template`:

```glsl
for (uint k = 0u; k < pc.K; k++) {
    ...
    double grad_q = 0.0lf;
    double lp_i   = 0.0lf;
    if (in_bounds) {
        {{prior_grad_body_q}}      // grad at q_k
        {{prior_logp_body_q}}      // <-- lp_i evaluated at q_k
    }
    double p_half = pi + 0.5lf * pc.eps * grad_q;
    double qn = qi + pc.eps * mi * p_half;
    qi = qn;                       // qi is now q_{k+1}
    ...
    if (in_bounds) {
        q_chain[k * pc.d + tid]    = qi;         // q_{k+1}
        p_chain[k * pc.d + tid]    = pi;         // p_{k+1}
        grad_chain[k * pc.d + tid] = grad_qn;    // grad at q_{k+1}
    }
    partial[tid] = lp_i;           // <-- still q_k
    ...
    if (tid == 0u) logp_chain[k] = partial[0];
}
```

So `logp_chain[k] = log p(q_chain[k−1])`, and `logp_chain[0] = log p(q_init)`.

**Numeric proof** (Normal(0,1), `q_init = 0.3`, `eps = 0.05`, `inv_mass = 1`,
one dispatch, measured on super-io):

```
logp_chain[0] = -0.9639385175704956
log p(0.3)    = -0.5 * (1.8378770351409912 + 0.09) = -0.9639385175704956   ✔ equal
q_chain[0]    =  0.349625
log p(0.349625) = -0.9800573378829956 = logp_chain[1]                       ✔ lagged
```

### Why that over-disperses

`Exmc.NUTS.Tree.synth_chain_subtree/10` pairs them by index:

```elixir
q_new    = Nx.slice(all_q, [idx, 0], [1, d])      # q_{idx+1}
logp_new = Nx.tensor(elem(raw_logps, idx), ...)   # log p(q_idx)   <-- wrong point
jlp      = Nx.subtract(logp_new, ke)              # ke from p_{idx+1}
```

Every NUTS leaf therefore carries `joint_logp = log p(q_{k−1}) − KE(p_k)`. On
the outbound half of a trajectory the density is falling, so `log p(q_{k−1}) >
log p(q_k)` and **every distant leaf looks more probable than it is**. The
multinomial leaf sampler over-weights the far end of each trajectory — a
mis-scaled, not mis-signed, error, exactly the signature T10 named. The same
inflated joint log-probs feed the divergence check (so no divergences are
raised) and the dual-averaging acceptance statistic (so the adapted `eps` is
pushed *up*, compounding the effect). "Gradient too small or `eps` too large"
was the right instinct; the mechanism is a stale log-density.

`Exmc.NUTS.BatchedLeapfrog.multi_step/8` — the non-chain reference path — is
the contract the shader was supposed to meet, and it stores
`all_logp[i] = log p(q_new)`, i.e. the **post**-step density, alongside
`all_q[i] = q_new`. `nx_vulkan`'s `Nx.Vulkan.ShaderTemplate` also emits its
`{{logp_block}}` *after* the position update, so it satisfies the contract.
eXMC's multi-RV template is the only one of the three that does not.

### Why the Kepler fleet never saw it

`Exmc.NUTS.Vulkan.Validator.run_exla/2` produces its reference by *deleting*
`Application.get_env(:exmc, :compiler)` and letting `Exmc.JIT.detect_compiler/0`
auto-detect. `auto_detect/0` falls through `EXLA → Nx.Vulkan → nil`. On a host
where EXLA is not loadable it therefore returns **`Nx.Vulkan`** — the reference
arm and the arm under test are the same backend, and the comparison is vacuous.

Measured on super-io at this commit, `EXLA loadable? false`,
`detect_compiler` after the delete = `Nx.Vulkan`. `EXMC_COMPILER=vulkan mix test
test/exmc/nuts/vulkan/validator_test.exs --include requires_vulkan` returns
**16 tests, 0 failures** in 112 s — a green run that establishes nothing.

FreeBSD has no EXLA at all, so **mac-247's celebrated 16/0 is the same vacuous
self-comparison.** The fleet's "reference host for numerical correctness" has
never once compared the Vulkan sampler against a non-Vulkan reference. super-io
scored 8/16 precisely because it was, at the time, the only host with a working
EXLA — i.e. the only host actually running the test. The conclusion drawn from
that (disqualify the Ampere box, trust the 2012 Kepler) inverted the evidence.

---

## 4. Confirming causality — move the log-prob body below the update

The one-line change described under "Recommended fix" below was applied to
eXMC's `@template` and the §1 and §3 measurements re-run on super-io. The
change was **not committed to eXMC** — it was reverted afterwards, and this
repo's branch carries no eXMC code.

First, the fixed-input check. `q`, `p` and `grad` are unchanged (bit-identical
to §1, as expected — the fix moves only where the density is sampled), and
`logp_chain` shifts forward by exactly one index:

```
Normal(0,1), q_init = 0.3, eps = 0.05, K = 32
  before:  logp_chain[0..2] = [-0.9639385175704956, -0.9800573378829956, -0.9982902113599976]
  after:   logp_chain[0..2] = [-0.9800573378829956, -0.9982902113599976, -1.0184549232221907]
```

`logp_chain[0]` now equals `log p(q_chain[0])` instead of `log p(q_init)`.

Then the sampling comparison, identical settings to §3:

| model | arm | mean | variance | var ratio vk/cpu |
|---|---|--:|--:|--:|
| Normal(0,1) | CPU | −0.0353 | 1.4523 | |
| Normal(0,1) | Vulkan, **before** | −0.1223 | 8.5467 | **5.885** |
| Normal(0,1) | Vulkan, **after** | −0.0377 | 1.4481 | **0.997** |
| Exponential(λ=2) | CPU | 0.5021 | 0.2600 | |
| Exponential(λ=2) | Vulkan, **before** | 1.4477 | 2.8661 | **11.02** |
| Exponential(λ=2) | Vulkan, **after** | 0.5128 | 0.2295 | **0.883** |
| HalfNormal(σ=1) | CPU | 0.8056 | 0.3773 | |
| HalfNormal(σ=1) | Vulkan, **before** | 1.3943 | 1.7459 | **4.63** |
| HalfNormal(σ=1) | Vulkan, **after** | 0.9522 | 0.4505 | **1.194** |

**Normal and Exponential are closed.** Normal goes from 5.885× to 0.997×;
Exponential's mean goes from 1.4477 to 0.5128 against a 0.5021 CPU arm and a
0.5 analytic value. The one-line change accounts for essentially all of the
reported over-dispersion in both.

**HalfNormal is improved but not closed.** Variance ratio drops 4.63 → 1.19,
mean 1.3943 → 0.9522, but the CPU arm gives 0.8056 and the analytic mean is
0.7979. At 800 samples the standard error on that mean is ≈0.024, so a 0.147
gap is ~6 SE — a real residual, not Monte-Carlo noise. HalfNormal is also the
one family in this battery with a documented history of a *separate* bug (W7
Stage 2.5: a `:softplus` vs `:log` transform mismatch, `research/gpu_node/
WORKSTREAM_W7_linux_nvidia_drift.md`). That residual is a separate
investigation and this document does not claim it.

---

## Side finding: this repo's own Phase-1 synthesis path is stale

Not part of T10, recorded because it was found on the way and it explains why
`nx_vulkan` could not have caught the bug in its own test suite.

`Nx.Vulkan.ShaderTemplate`'s GLSL declares the push block as
`{uint n; uint K; float eps; …}` — offsets 0, 4, 8 — and
`Nx.Vulkan.ChainShaderSpecs.beta_push/6` packs exactly that. But
`parse_push_block/1` in `native/nx_vulkan_vulkano/src/lib.rs:293`, which backs
`leapfrog_chain_synth/6`, reads `{k_steps@0, n_obs@4, d@8, _pad@12, eps@16}` —
eXMC's `MultiRvCustomSpec` layout — and `PushBlockF64` (`:318`) uses the same
shape for the f64 entry point. The two are incompatible: fed a
`ChainShaderSpecs` push, the NIF would read `d` out of the float bits of `eps`.

Consequence: **the `ShaderTemplate` / `ChainShaderSpecs` / `Synthesis` trio
cannot currently be dispatched through either NIF.** `native_v.ex:329` already
says as much for the f32 entry point ("kept as stub only (unused)"). This is
consistent with the test suite: `test/nx_vulkan/synthesis_test.exs` only checks
that GLSL compiles to SPIR-V and caches, never dispatches, and renders its spec
with `logp_block: "float lp_i = 0.0;"` — a template that never computes a
log-density cannot detect a misaligned one.

Established from source reading only; no dispatch was attempted against it.

---

## Established vs inferred

**Established by measurement.**

- The chain shader agrees with a host-side reference to 1–4 ulp at f64 on
  Ampere, for 5 model shapes and K ∈ {1,2,4,32}.
- Ampere and Kepler produce bit-identical `q`/`p`/`grad` chains, and logp
  agreeing to ≤1 ulp, for identical fixed inputs and identical GLSL.
- `logp_chain[k] = log p(q_chain[k−1])` — demonstrated numerically on a single
  dispatch, not inferred from source alone.
- `Tree.synth_chain_subtree/10` pairs `q_chain[idx]` with `logp_chain[idx]`.
- `BatchedLeapfrog.multi_step/8` stores the post-step density; the shader
  stores the pre-step density.
- The validator's reference arm resolves to `Nx.Vulkan` on super-io, and the
  validator battery is 16/0 there under that (vacuous) configuration.
- Vulkan vs CPU sampling variance ratios before and after the fix (§3, §4).
- Moving the log-prob body below the position update removes the
  over-dispersion for Normal (5.885× → 0.997×) and Exponential
  (11.02× → 0.883×) and most of it for HalfNormal (4.63× → 1.19×).

**Inferred, not proven.**

- That mac-247 would show the same over-dispersion against a genuine CPU
  reference. It follows from bit-identical trajectories plus host-independent
  Elixir, but it was not run there.
- The dual-averaging feedback (inflated accept → larger adapted `eps` →
  further dispersion) is a mechanism argument, not a measurement.

**Not determined.**

- Whether the 8/16-vs-2/16 split reported in D90 decomposes exactly along this
  bug — that run's host configuration (which EXLA, which commit) is not
  recoverable from the doc.
- Whether `@batched_template` (Task #154) is exercised anywhere in the failing
  battery. It carries the identical off-by-one and should be fixed with it.
- The HalfNormal residual after the fix (mean 0.9522 against a 0.8056 CPU arm
  and 0.7979 analytic, ~6 SE). Not chased; W7 Stage 2.5 already records a
  transform mismatch for this family, which is the first place to look.
- Whether the Normal CPU arm's own variance (1.4523 against 1.0) is warmup
  length, the adapted step size, or something else. It is common to both arms
  so it does not affect any conclusion here, but it means the CPU arm is a
  *relative* reference, not a certified one.

---

## Recommended fix (eXMC, not this repo)

Move `{{prior_logp_body_q}}` out of the pre-update `if (in_bounds)` block and
into the post-update one, next to `{{prior_grad_body_qn}}`, in **both**
`@template` and `@batched_template`. The emitted fragments read `qi` and
`q_shared[]`, both of which hold the new position at that point (the template
already refreshes `q_shared` and barriers around it), and `lp_i` is declared at
loop scope, so no other change is needed. Verified in §4; **not committed** —
the eXMC working tree was restored after measuring.

```diff
         double grad_q = 0.0lf;
         double lp_i   = 0.0lf;
         if (in_bounds) {
 {{prior_grad_body_q}}
-{{prior_logp_body_q}}
         }
         double p_half = pi + 0.5lf * pc.eps * grad_q;
         double qn = qi + pc.eps * mi * p_half;
         qi = qn;
 
         barrier();
         if (in_bounds) q_shared[tid] = qi;
         barrier();
 
         double grad_qn = 0.0lf;
         if (in_bounds) {
 {{prior_grad_body_qn}}
+{{prior_logp_body_q}}
         }
```

Two related items eXMC should fix in the same pass:

1. **`Validator.run_exla/2` must fail loud when EXLA is unavailable** rather
   than silently self-comparing. One `Code.ensure_loaded?(EXLA)` guard turns a
   fleet-wide false green into a skip. Until it lands, no validator result from
   any FreeBSD host means anything.
2. **`@batched_template` carries the identical lag** and was not measured here.

Then add the missing contract test. Neither repo has one:
`nx_vulkan/test/nx_vulkan/synthesis_test.exs` renders its spec with
`logp_block: "float lp_i = 0.0;"`, so it cannot detect misalignment. A
fixed-input assertion that `logp_chain[k] == log p(q_chain[k])` for K > 1 would
have caught this at synthesis time and is cheap — the reproducer in this
document is that test.

---

## Reproducer

`Nx.Vulkan.NativeV.leapfrog_chain_synth_f64/6`, one dispatch, Normal(0,1),
d = 1, K = 4, `q_init = [0.3]`, `p_init = [1.0]`, `inv_mass = [1.0]`,
`eps = 0.05`, `dir_sign = +1`, GLSL from
`Exmc.NUTS.CustomSynth.MultiRvCustomSpec.render/1`:

| k | `q_chain[k]` | `logp_chain[k]` (actual) | `log p(q_chain[k])` (expected) |
|--:|---|---|---|
| 0 | 0.349625 | −0.9639385175704956 | −0.9800573378829956 |
| 1 | 0.39900... | −0.9800573378829956 | −0.9982902113599976 |

Actual equals expected shifted by one index; `logp_chain[0]` equals
`log p(0.3)`, the input. Identical on RTX 3060 Ti and GT 650M.

---

## What this retires

"super-io is not a valid host for Vulkan numerical validation" was the wrong
lesson. The correct one is the opposite: **super-io was the only host running a
real comparison, and the fleet's reference host has never run one at all.**
T10's `Done when` is met — the defect is localised to a repo (eXMC), a file
(`multi_rv_custom_spec.ex`), and a line, with a fixed-input reproducer that
involves no sampling. The standing exclusion should be lifted and replaced with
a validator that cannot pass vacuously.
