# Plan — Fused Vulkan `step_fn` kernel

**Status**:
- 2026-05-02: Plan greenlit (Approach A, Phase 1).
- 2026-05-03: Phase 1 (single-step `leapfrog_normal` shader)
  shipped Stages 1-4 — round-trip correct, math verified.
  Per-step bench measured **537 µs/step at d=8** = ~11×
  improvement, NOT the predicted ~120×. Root cause: the
  per-Vulkan-dispatch baseline is ~500 µs regardless of shader
  complexity. Single-step fusion hit a floor.
- 2026-05-03: Pivoted to **Phase 1.5 — `leapfrog_chain_normal`
  shader** below. Single-step infrastructure preserved on
  branch `feat/fused-leapfrog-normal` (commit `e794c5b`) as a
  baseline.

## Why this plan exists

`docs/VULKAN_KNOWN_ISSUES.md` (in `pymc/exmc/docs/`) issue #2
documents the structural problem: NUTS sampling under
`EXMC_COMPILER=vulkan` is pathologically slow because each
leapfrog step expands to ~12 elementwise dispatches, and each
dispatch costs ~280-700 µs of indirection on top of GPU work.
For ~10,000 leapfrog calls per warmup, that's ~60s of pure
overhead per warmup phase — and we have benchmark tests with
multiple warmup rounds. Result: tests that run in 2-10s under
EXLA take 30-90 minutes (or never complete) under Vulkan.

This isn't a bug. It's the predicted consequence of
`RESEARCH_FAST_KERNELS.md`'s break-even rule for `d ≤ 50` MCMC
workloads. The fix is to collapse the per-leapfrog dispatch
chain into a single shader.

## Goal & success criterion

**One Vulkan dispatch per NUTS leapfrog step**, replacing the
current ~12.

Concrete success: `EXMC_COMPILER=vulkan` runs the
Exponential-Poisson NUTS test in ≤ 30 seconds (currently
doesn't complete in 90 minutes). At that threshold, the
structural blocker dissolves and the full eXMC suite becomes
viable under Vulkan.

## The make-or-break design decision

A leapfrog step is straightforward:
1. `p_half = p - 0.5 * eps * grad_q logp(q)`
2. `q_new = q + eps * inv_mass * p_half`
3. `p_new = p_half - 0.5 * eps * grad_q logp(q_new)`

The hard part is **`grad_q logp(q)`** — model-specific, computed
today by `Nx.Defn.value_and_grad` after tracing through the
model's distributions. To run inside one shader, that gradient
has to live in the shader's body, not in a separate Elixir/Nx
call.

Three architectures in increasing scope:

| Approach | Cost | Coverage | Reversibility |
|----------|------|----------|---------------|
| **A. Hand-write fused shader per distribution family** (Normal, HalfNormal, Exponential, …) | 1-2 weeks for ~10 families | Single-RV models in those families | Easy — opt-in path, falls back to current dispatch |
| **B. Hand-write fused shader per *composed* model class** (Normal-Normal hierarchical, multi-RV with priors) | 1-3 months | Hand-picked benchmark models | Same — opt-in path |
| **C. Compile arbitrary eXMC IR → SPIR-V shader with autodiff** | 6+ months — research project, not engineering | All models | Requires in-shader autodiff infrastructure that doesn't exist yet |

Approach **A** is the right start. Phase 1 = one shader for
univariate Normal, with the criterion above as the go/no-go.

## Phase 1 — Normal-Normal POC

### Files to touch

```
spirit/                         (mac-248)
  shaders/leapfrog_normal.comp  NEW shader
  shaders/leapfrog_normal.spv   NEW compiled output

nx_vulkan/                      (Linux dev box)
  c_src/spirit/                 ← refresh from spirit checkout
  c_src/nx_vulkan_shim.{cpp,h}  + nxv_leapfrog_normal entry
  native/.../src/lib.rs         + leapfrog_normal NIF
  lib/nx_vulkan/native.ex       + leapfrog_normal stub
  lib/nx_vulkan.ex              + Nx.Vulkan.leapfrog_normal/N
  lib/nx_vulkan/fast.ex         + Fast.leapfrog_normal/N (named-kernel form)
  lib/nx_vulkan/backend.ex      + fast_leapfrog_normal callback
  test/                         + correctness + benchmark

pymc/exmc/                      (Linux dev box)
  lib/exmc/nuts/leapfrog.ex     + opt-in dispatch on detected single-Normal model
  lib/exmc/jit.ex               + helper to detect "fused-eligible" IR
  test/                         + assertion: same posterior with/without fused path
  config/test.exs               + EXMC_FUSED_LEAPFROG env var
```

### Shader sketch (univariate Normal observation)

```glsl
#version 450
layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint n;
    float eps;
    float mu;       // prior mean
    float sigma;    // prior sd
} pc;

layout (std430, binding = 0) readonly  buffer In_q     { float q[]; };
layout (std430, binding = 1) readonly  buffer In_p     { float p[]; };
layout (std430, binding = 2) readonly  buffer In_mass  { float inv_mass[]; };
layout (std430, binding = 3) writeonly buffer Out_q    { float q_new[]; };
layout (std430, binding = 4) writeonly buffer Out_p    { float p_new[]; };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;

    float inv_var = 1.0 / (pc.sigma * pc.sigma);

    // grad_q log N(q | mu, sigma) = -(q - mu) / sigma^2
    float grad_q  = -(q[i] - pc.mu) * inv_var;

    float p_half  = p[i] - 0.5 * pc.eps * grad_q;
    float qn      = q[i] + pc.eps * inv_mass[i] * p_half;
    float grad_qn = -(qn - pc.mu) * inv_var;
    float pn      = p_half - 0.5 * pc.eps * grad_qn;

    q_new[i] = qn;
    p_new[i] = pn;
}
```

~30 lines. Single dispatch. No log-density value emitted from
this shader (kinetic energy + new log-density use existing
`kinetic_energy.spv` + `normal_logpdf.spv` named kernels —
already shipped). Not generalizable beyond Normal, but cheap
to write.

### Wiring sketch

```elixir
def step(q, p, eps, mass, model_meta) do
  if model_meta.fused_eligible? and
       Application.get_env(:exmc, :fused_leapfrog, false) do
    Nx.Vulkan.Fast.leapfrog_normal(q, p, eps, mass,
                                    model_meta.mu, model_meta.sigma)
  else
    step_fn.(q, p, eps, mass)
  end
end
```

`fused_eligible?` is computed once at compile time by walking
the IR and checking: single RV, Normal distribution, no
constraints. Conservative pattern match; anything that doesn't
fit falls through to the existing path.

### Test + benchmark

- **Correctness**: same posterior (mean + variance within MCMC
  noise) with `EXMC_FUSED_LEAPFROG=true` vs default, on the
  simple Normal-Normal model.
- **Per-step bench**: `bench/leapfrog_bench.exs` extended with
  a fused-path comparison. Target: ≤ 50 µs per leapfrog body
  (vs current Vulkan ~6000 µs).
- **End-to-end**: a Normal-Normal posterior mean test currently
  taking 39 minutes under Vulkan should run in ≤ 30 seconds
  with the fused path enabled.

## Risks & open questions

1. **Push constants size** — Vulkan limits push constants to
   128 bytes typically. Scalars `mu` and `sigma` fit; if we
   extend to multi-dimensional `mu/sigma`, they become buffers.
   For Phase 1 univariate Normal this is fine.
2. **`inv_mass` as buffer vs scalar** — diagonal mass: vector.
   Identity mass: scalar. Phase 1 assumes diagonal vector (the
   eXMC default).
3. **The `n` dimension dispatch** — for `d ≤ 50` (the eXMC
   sweet spot), one workgroup of 256 threads is enough. Almost
   no parallelism, but that's fine; the win is *fewer
   dispatches*, not more parallelism.
4. **Phase 1 success doesn't generalize** — we only prove
   Normal works. Phase 2 expansion to ten families is real
   work; Phase 3 (hierarchical / composed models) is much
   harder. We may discover Phase 2 isn't worth doing if the
   eXMC user base doesn't actually run that many fixed-form
   single-RV models.
5. **mac-248 hand-off** — shader writing happens on mac-248
   (FreeBSD shader compiler box). Wiring happens on Linux.
   Coordinated work over 2-3 sessions.

## What we need before starting

- **Confirm Phase 1 scope** — POC for one shader (univariate
  Normal), with the explicit acceptance test of
  "Exponential-Poisson NUTS in ≤ 30s." If POC succeeds, plan
  Phase 2 then; if it doesn't, learn why before building more.
- **Confirm shader location** — same `~/spirit/shaders/`
  (mac-248 compiles, vendored copy lives in
  `nx_vulkan/priv/shaders/`).
- **mac-248 availability** — ready to receive a 248_TODO update
  for `leapfrog_normal.{comp,spv}`.
- **Approval to begin** — once given, draft a new 248_TODO
  with the shader spec (will replace the current Flavor A gaps
  TODO since this is now higher priority), and start the
  Linux-side wiring in parallel.

## Phase 1.5 — `leapfrog_chain_normal`

### Why the pivot

Phase 1's measured per-step cost (537 µs at d=8) revealed that
the Vulkan dispatch **baseline** — queue submission + fence
wait + Rustler decode — is ~500 µs *regardless of what the
shader does*. The shader compute itself is microseconds; the
overhead floor dominates. Single-step fusion got us 11×, but
no further single-step fusion can do better.

To go below the per-dispatch floor, multiple leapfrog steps
must share one dispatch.

### The math

For a chain of K leapfrog steps in one shader:
amortized per-step cost = `(500 µs + K × shader_compute) / K`.

| K | Per-step amortized | vs EXLA target (~50 µs) |
|---|--------------------|-------------------------|
| 1 (Phase 1) | 537 µs | 11× slower |
| 8 | ~63 µs | 1.3× slower |
| 16 | ~32 µs | **1.5× faster** |
| 32 | ~16 µs | **3× faster** |
| 64 | ~8 µs | **6× faster** |

K=32 is the sweet spot. Below the EXLA target, comfortably
above the speculative-waste threshold.

### The shader (sketch)

```glsl
#version 450
layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint  n;
    uint  K;
    float eps;
    float mu;
    float sigma;
} pc;

layout (std430, binding = 0) readonly  buffer In_q     { float q_init[]; };
layout (std430, binding = 1) readonly  buffer In_p     { float p_init[]; };
layout (std430, binding = 2) readonly  buffer In_mass  { float inv_mass[]; };
layout (std430, binding = 3) writeonly buffer Out_q    { float q_chain[]; };    // K × n
layout (std430, binding = 4) writeonly buffer Out_p    { float p_chain[]; };    // K × n
layout (std430, binding = 5) writeonly buffer Out_grad { float grad_chain[]; }; // K × n
layout (std430, binding = 6) writeonly buffer Out_logp { float logp_chain[]; }; // K (per-step scalar)

shared float partial[256];   // for per-step logp reduction

void main() {
    uint i   = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationIndex;

    float inv_var = 1.0 / (pc.sigma * pc.sigma);
    float qi = (i < pc.n) ? q_init[i] : 0.0;
    float pi = (i < pc.n) ? p_init[i] : 0.0;
    float mi = (i < pc.n) ? inv_mass[i] : 0.0;
    float log_2pi_half = 0.91893853;
    float log_sigma    = log(pc.sigma);

    for (uint k = 0; k < pc.K; k++) {
        // Half-step momentum, full-step position, half-step momentum
        float grad_q = (i < pc.n) ? -(qi - pc.mu) * inv_var : 0.0;
        float p_half = pi - 0.5 * pc.eps * grad_q;
        qi = qi + pc.eps * mi * p_half;
        float grad_qn = (i < pc.n) ? -(qi - pc.mu) * inv_var : 0.0;
        pi = p_half - 0.5 * pc.eps * grad_qn;

        if (i < pc.n) {
            q_chain[k * pc.n + i]    = qi;
            p_chain[k * pc.n + i]    = pi;
            grad_chain[k * pc.n + i] = grad_qn;
        }

        // Per-step logp = -0.5 * sum_i ((qi - mu)/sigma)² - n*log(sigma) - n*log(2π)/2
        // Reduce across workgroup using shared memory.
        float zi = (i < pc.n) ? (qi - pc.mu) / pc.sigma : 0.0;
        partial[tid] = (i < pc.n) ? zi * zi : 0.0;
        barrier();
        for (uint s = 128u; s > 0u; s /= 2u) {
            if (tid < s) partial[tid] += partial[tid + s];
            barrier();
        }
        if (tid == 0u) {
            float n_f = float(pc.n);
            logp_chain[k] = -0.5 * partial[0] - n_f * (log_sigma + log_2pi_half);
        }
        barrier();   // ensure all threads see logp_chain[k] before next iter
    }
}
```

~75 lines including the shared-memory reduction. Push
constants stay at 20 bytes.

### Wiring (much simpler than expected — see audit)

The eXMC NUTS speculative path already batches K leapfrog
steps via `BatchedLeapfrog.multi_step` (XLA while-loop). Only
**two call sites** (`lib/exmc/nuts/tree.ex:524` and `:572` in
`ensure_available/3`) need conditional dispatch:

```elixir
{all_q, all_p, all_logp, all_grad} =
  if vulkan_chain_eligible?(spec_buf) do
    Nx.Vulkan.Fast.leapfrog_chain_normal(
      q_init, p_init, inv_mass,
      n_steps, eps, mu, sigma
    )
  else
    spec_buf.multi_step_fn.(q, p, grad, eps, inv_mass, n_steps)
  end
```

No tree builder changes. No Rust NIF changes. No merge-logic
changes. The contract `(q, p, grad, eps, inv_mass, n_steps) →
{all_q, all_p, all_logp, all_grad}` is matched exactly by the
shader's output buffers.

### K policy

The audit confirms K is already adaptive in the speculative
path: `max(32, n_needed * 2)` where `n_needed = 2^depth`. So K
defaults to 32 and grows with observed tree depth. Perfect —
no new K-selection logic needed.

### Stages

This re-numbers from Phase 1's checklist:

- **Stage 1.5.1**: Mac-248 writes `leapfrog_chain_normal.{comp,spv}`
- **Stage 1.5.2**: Vendor shader, C++ shim, Rust NIF (returns
  4 buffers), Elixir wrapper, named-kernel form. Same pattern
  as Phase 1 Stages 1-5.
- **Stage 1.5.3**: eXMC `vulkan_chain_eligible?/1` predicate +
  conditional dispatch in tree.ex. **2 lines changed.**
- **Stage 1.5.4**: Correctness vs unfused at K=32, K=64.
- **Stage 1.5.5**: Per-step bench. Target: ≤ 16 µs/step at K=32
  (matching the math table above).
- **Stage 1.5.6**: **Acceptance test** — Exponential-Poisson
  NUTS in ≤ 30 seconds. Same bar as Phase 1's Stage 9, now
  achievable.

### Risks (real this time)

1. **Speculation waste**: K=32 means ~50% wasted compute on
   typical tree depth ~5 (= 32 leaves total, but average tree
   uses ~16). Acceptable — better than current path.
2. **f32 numerical drift over K=32 steps**: 32 leapfrog steps
   in f32 may accumulate more drift than EXLA's f64. Will need
   to compare posterior recovery on the Normal-Normal model.
3. **Per-step logp reduction in shader**: requires workgroup
   shared memory + barrier pattern. Shader compiler may flag
   the layout; mac-248 to fix in place if so.
4. **Output buffer growth**: K=32 × n=8 × 4 buffers × 4 bytes =
   4 KB. K=128 × n=100 × 4 buffers × 4 bytes = 200 KB. Both
   well under buffer pool limits.

### Audit summary (informs the wiring effort)

Per the eXMC speculative-path audit (2026-05-03), the refactor
to plug a Vulkan chain dispatch into the speculative path is
**low-invasiveness, 2-4 hours best case, 1-2 person-days
likely**. The tree builder is already leapfrog-agnostic; the
NIF boundary is already binary-blob-passing; the merge logic
already consumes pre-computed states. Only the dispatch site
(2 lines) needs to change.

## Cross-references

- `RESEARCH_FAST_KERNELS.md` — break-even rule that predicts
  this exact structural problem
- `pymc/exmc/docs/VULKAN_KNOWN_ISSUES.md` issue #2 — the
  empirical evidence (3 Vulkan test runs, all hung on NUTS)
- `lib/nx_vulkan/fast.ex` — the named-kernel module the new
  `leapfrog_normal` will live in
- `pymc/exmc/lib/exmc/nuts/leapfrog.ex` — the eXMC-side
  step_fn this fuses
- `pymc/www.dataalienist.com/blog-walkable-path.html` — the
  strategic context (Vulkan as the FreeBSD GPU path; this
  fusion is what makes that claim true for MCMC workloads)
