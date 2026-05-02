# Plan — Fused Vulkan `step_fn` kernel

**Status**: Planned, not started. Paused 2026-05-02 awaiting
go-ahead to execute Phase 1.

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
