# mac-248 — Two parallel tasks (X1 + X2) while Linux side debugs Stage 1.5.4

The chain shaders (Normal single-WG, Normal multi-WG, Exponential)
are all wired and smoke-passing. Linux side is in the middle of
Stage 1.5.4 (variance-bias diagnosis): EXLA reference gives
var ≈ 1.0 on `x ~ N(0,1)`, the fused chain gives var ≈ 0.5–0.7.
Either f32 precision drift or a chain integration bug. The H1
unfused-Vulkan run is in flight to triangulate.

Two independent tasks for mac-248 to start NOW. Both are useful
**regardless** of how H1 lands.

## Layout

```
cd ~/spirit && git pull origin feat/fused-leapfrog-chain-normal
cd ~/nx_vulkan && git pull origin feat/fused-leapfrog-chain-normal
```

Branch off `feat/fused-leapfrog-chain-normal`. Push to the same
branch (or a sub-branch — your call).

---

## X1 — `leapfrog_chain_normal_f64.spv` (highest strategic value)

f64 version of the existing single-workgroup chain shader. Same
logic everywhere, types widened to `double`.

**Why this is the right pick now**:
- If Stage 1.5.4 concludes "f32 precision is the cause" — this
  is the immediate fix. Variance bias goes away.
- If Stage 1.5.4 concludes "chain has a real bug" — still
  useful: high-precision option for sensitive models, complements
  the bug fix.
- Same precedent as `reduce_axis_f64.spv` (already shipped):
  add the f64 sibling alongside the f32 version, Linux side
  picks based on tensor type at dispatch time.

**Shader spec**:

```glsl
#version 450
#extension GL_ARB_gpu_shader_fp64 : enable

// f64 sibling of leapfrog_chain_normal.comp. All buffers and
// arithmetic in f64. Output sizes double (8 bytes per element).
// Push constants {n, K, eps, mu, sigma} = 4 + 4 + 8 + 8 + 8 = 32 bytes.
// Single workgroup; n <= 256.

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint   n;
    uint   K;
    double eps;
    double mu;
    double sigma;
} pc;

layout (std430, binding = 0) readonly  buffer In_q     { double q_init[]; };
layout (std430, binding = 1) readonly  buffer In_p     { double p_init[]; };
layout (std430, binding = 2) readonly  buffer In_mass  { double inv_mass[]; };
layout (std430, binding = 3) writeonly buffer Out_q    { double q_chain[]; };    // K × n
layout (std430, binding = 4) writeonly buffer Out_p    { double p_chain[]; };    // K × n
layout (std430, binding = 5) writeonly buffer Out_grad { double grad_chain[]; }; // K × n
layout (std430, binding = 6) writeonly buffer Out_logp { double logp_chain[]; }; // K

shared double partial[256];

const double LOG_2PI_HALF_F64 = 0.91893853320467274LF;

void main() {
    uint i   = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationIndex;

    bool in_bounds = (i < pc.n);
    double inv_var  = 1.0LF / (pc.sigma * pc.sigma);
    double log_sigma = log(pc.sigma);
    double n_d = double(pc.n);

    double qi = in_bounds ? q_init[i] : 0.0LF;
    double pi = in_bounds ? p_init[i] : 0.0LF;
    double mi = in_bounds ? inv_mass[i] : 0.0LF;

    for (uint k = 0; k < pc.K; k++) {
        double grad_q = in_bounds ? -(qi - pc.mu) * inv_var : 0.0LF;
        double p_half = pi - 0.5LF * pc.eps * grad_q;
        qi = qi + pc.eps * mi * p_half;
        double grad_qn = in_bounds ? -(qi - pc.mu) * inv_var : 0.0LF;
        pi = p_half - 0.5LF * pc.eps * grad_qn;

        if (in_bounds) {
            q_chain[k * pc.n + i]    = qi;
            p_chain[k * pc.n + i]    = pi;
            grad_chain[k * pc.n + i] = grad_qn;
        }

        double zi = in_bounds ? (qi - pc.mu) / pc.sigma : 0.0LF;
        partial[tid] = zi * zi;
        barrier();

        for (uint s = 128u; s > 0u; s /= 2u) {
            if (tid < s) partial[tid] += partial[tid + s];
            barrier();
        }

        if (tid == 0u) {
            logp_chain[k] = -0.5LF * partial[0] - n_d * (log_sigma + LOG_2PI_HALF_F64);
        }

        barrier();
    }
}
```

**Compile + push**:
```sh
cd ~/spirit
glslangValidator -V shaders/leapfrog_chain_normal_f64.comp \
                 -o shaders/leapfrog_chain_normal_f64.spv
git add shaders/leapfrog_chain_normal_f64.{comp,spv}
git commit -m "shaders: leapfrog_chain_normal_f64 — f64 sibling of the chain"
git push origin feat/fused-leapfrog-chain-normal
```

**Sanity check**: at K=2, n=4, q=[1,2,3,4], p=[0.5...], mu=0,
sigma=1, eps=0.1, the f64 shader should give the same numbers as
the f32 shader to ~15 decimal places (well past f32's 7-digit
precision). Specifically `q_chain[0,0]` should be `1.054999999...`
(vs f32's `1.054999955...`).

**`#extension GL_ARB_gpu_shader_fp64 : enable`**: required for f64
in compute shaders. The GT 750M (Kepler) supports it; FreeBSD's
nvidia driver enables it. If glslang flags an issue, the
fallback is to compile with `--target-env vulkan1.1 --target-spv
spv1.3` to ensure the right SPIR-V capability bits are emitted.

**Push constants 32 bytes**: still well under Vulkan's 128-byte
spec floor.

Effort: **~1 hour** including the sanity check.

---

## X2 — `reduce_full_f64.spv` (the perennial leftover)

The f64 full-axis reduction shader. Has been deferred 4 times now
across previous TODO rounds. Closes the only remaining f64 GPU
gap on the gradient hot path (currently host-falls-back for f64
determinant + f64 mass-matrix Welford).

**Spec preserved verbatim** at `git show 09280e3:248_TODO.md`
(prior 248_TODO that introduced this task). Workgroup tree
reduction in f64; output one f64 per workgroup; op selector
matches the f32 shader (0=sum, 1=max, 2=min); single-pass
single-workgroup-per-dispatch.

**Compile + push**:
```sh
cd ~/spirit
glslangValidator -V shaders/reduce_full_f64.comp \
                 -o shaders/reduce_full_f64.spv
git add shaders/reduce_full_f64.{comp,spv}
git commit -m "shaders: reduce_full_f64 — f64 full-axis sum/max/min"
git push origin feat/fused-leapfrog-chain-normal
```

Effort: **~30-60 minutes**.

---

## What this DOES NOT need from you

- No Linux-side wiring (C++ shim, Rust NIF, Elixir wrapper, eXMC
  integration). All Linux side once .spv lands.
- No correctness against running NUTS — the smoke check at K=2
  against the f32 shader's numbers is sufficient.

## Order

X1 first. It's the strategic bet that resolves the variance
question if H1 turns out to be f32. X2 is the comfort task —
independent, well-spec'd, finite — and a good fallback if X1
hits a glslang surprise around f64 in compute shaders.

## What NOT to do (yet)

- More distribution families (HalfNormal, StudentT, etc.) — Phase 2
  is gated on Stage 1.5.4 success.
- Multi-workgroup f64 chain — single-WG f64 is enough for the
  variance-question, multi-WG can wait.
- Shader microbenchmarks — interesting but not actionable yet.

## Cross-reference

- `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md` — full
  plan, including the Phase 1 / 1.5 history
- `~/projects/learn_erl/nx_vulkan/CHECKLIST_FUSED_LEAPFROG.md` —
  Linux-side stage tracker
- `~/projects/learn_erl/pymc/exmc/test/nuts/fused_chain_diag_test.exs` —
  the diagnostic test that pins the variance question. Once X1
  ships and is wired, this test should flip from
  `var ≈ 0.5-0.7` to `var ≈ 1.0` under EXMC_COMPILER=vulkan if
  f32 precision was the cause.
