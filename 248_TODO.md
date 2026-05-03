# mac-248 — Fused leapfrog Normal shader (Phase 1)

**Goal**: write `leapfrog_normal.{comp,spv}` — a single Vulkan
shader that performs one full NUTS leapfrog step (half-momentum
+ full-position + half-momentum) for a univariate Normal
log-density model. This is Phase 1 of the fused step_fn project
documented in `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md`.

**Why this exists**: NUTS sampling under `EXMC_COMPILER=vulkan`
is currently pathologically slow because each leapfrog expands
to ~12 elementwise dispatches × ~500 µs of indirection each.
For a typical warmup of 10,000 leapfrog steps, that's ~60s of
pure overhead per warmup phase, on top of the actual compute.
Tests that complete in 2-10s under EXLA take 30-90 minutes (or
never) under Vulkan. See
`~/projects/learn_erl/pymc/exmc/docs/VULKAN_KNOWN_ISSUES.md`
issue #2 for the full empirical evidence (3 test runs, all
hung on NUTS).

The fix predicted by `RESEARCH_FAST_KERNELS.md` is to collapse
the per-leapfrog dispatch chain into a single shader. This TODO
is the smallest defensible bet: ONE shader for the simplest
model class. The acceptance test is whether the Exponential-
Poisson NUTS test runs in ≤ 30 seconds (currently doesn't
complete in 90 minutes — so target is a ≥ 200× improvement).

Branch: `feat/fused-leapfrog-normal` off current main on each
repo (`spirit` for the shader, `nx_vulkan` for the wiring).

Linux side will pick up your push and add the C++ shim, Rust
NIF, backend dispatch, eXMC-side opt-in, and benchmark.

## Layout note

Mac-248 (FreeBSD 15, GT 750M) uses the flat layout: `~/spirit/`,
`~/nx_vulkan/`. Both at current main. Pull before branching:

```
cd ~/spirit && git pull origin main && git checkout -b feat/fused-leapfrog-normal
cd ~/nx_vulkan && git pull origin main
```

## The shader

A leapfrog step for an unconstrained univariate Normal model
(prior = `N(mu, sigma)`, no observation) is:

1. `p_half = p − 0.5 · eps · grad_q logp(q)`
2. `q_new  = q + eps · inv_mass · p_half`
3. `p_new  = p_half − 0.5 · eps · grad_q logp(q_new)`

The gradient `grad_q log N(q | mu, sigma) = −(q − mu) / sigma²`.
That's a closed form — no autodiff machinery needed inside the
shader.

```glsl
// shaders/leapfrog_normal.comp — fused NUTS leapfrog for a
// univariate Normal log-density model. One dispatch per step
// instead of ~12 elementwise dispatches via the IR walker.
//
// q, p, q_new, p_new all live as f32[n] (the parameter
// dimension, typically n ≤ 50 for typical Bayesian models).
// inv_mass is the diagonal mass-matrix inverse, also f32[n].
// mu, sigma, eps are scalars in push constants — model
// constants captured at compile time.

#version 450

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint  n;
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

    // Step 1: half-step momentum
    //   p_half = p − 0.5 · eps · grad_q logp(q)
    //   grad_q = −(q − mu) / sigma²
    float grad_q = -(q[i] - pc.mu) * inv_var;
    float p_half = p[i] - 0.5 * pc.eps * grad_q;

    // Step 2: full-step position
    //   q_new = q + eps · inv_mass · p_half
    float qn = q[i] + pc.eps * inv_mass[i] * p_half;

    // Step 3: half-step momentum at new position
    //   p_new = p_half − 0.5 · eps · grad_q logp(q_new)
    float grad_qn = -(qn - pc.mu) * inv_var;
    float pn = p_half - 0.5 * pc.eps * grad_qn;

    q_new[i] = qn;
    p_new[i] = pn;
}
```

~30 lines. Single dispatch. No log-density value emitted from
this shader — `kinetic_energy.spv` and `normal_logpdf.spv` (both
already shipped) handle the energy + new-logp computations
called from the surrounding NUTS tree code.

**Push constants budget check**: `uint + 3 × float = 16 bytes`,
well under Vulkan's 128-byte spec floor. Safe everywhere.

**Workgroup sizing**: at the typical eXMC dimension `d ≤ 50`,
one workgroup of 256 threads covers it with no need for
multi-workgroup dispatch. Larger `n` (rare for MCMC) is handled
by the standard `gl_GlobalInvocationID.x` bounds check above.

## Compile + push

```
cd ~/spirit
glslangValidator -V shaders/leapfrog_normal.comp \
                 -o shaders/leapfrog_normal.spv
git add shaders/leapfrog_normal.{comp,spv}
git commit -m "shaders: leapfrog_normal — fused univariate Normal step"
git push origin feat/fused-leapfrog-normal
```

If `glslangValidator` flags anything (validation layer warnings,
unused bindings, etc.), fix in place and re-push. The Linux side
treats the latest `.spv` on the branch as canonical.

## After your push

Linux side will:

1. Vendor the new shader into `nx_vulkan/priv/shaders/leapfrog_normal.spv`.
2. Add `nxv_leapfrog_normal` to `c_src/nx_vulkan_shim.{cpp,h}`.
3. Add `leapfrog_normal` Rust NIF in `native/.../src/lib.rs`.
4. Add `Nx.Vulkan.leapfrog_normal/N` and
   `Nx.Vulkan.Fast.leapfrog_normal/N` (named-kernel form).
5. Add the eXMC-side opt-in in `Exmc.NUTS.Leapfrog.step` gated
   on `Application.get_env(:exmc, :fused_leapfrog, false)` plus
   IR-eligibility detection (single Normal RV, no constraints).
6. Add the bench: `bench/leapfrog_fused_bench.exs` with the
   target ≤ 50 µs/body criterion.
7. Run the acceptance test: Exponential-Poisson under
   `EXMC_COMPILER=vulkan EXMC_FUSED_LEAPFROG=true mix test`,
   target ≤ 30 seconds.

Each of those steps lives in `nx_vulkan/CHECKLIST_FUSED_LEAPFROG.md`
(also on `feat/fused-leapfrog-normal`) so progress is visible.

## What this DOES NOT need from you

- No C++ shim, no Rust NIF, no Elixir wiring. All Linux side.
- No tests. Linux side writes correctness + benchmark tests
  using the actual NUTS sampler.
- No documentation. The PLAN_FUSED_LEAPFROG.md file already
  carries the architectural rationale.

## If the shader needs adjustments later

If the bench shows the shader is correct but slower than
predicted, common shader-side fixes:
- Reduce reads of `q[i]` and `p[i]` to one each (already done
  in the sketch — verify in your final).
- Move the `1.0 / (sigma * sigma)` inverse-variance outside the
  per-thread compute via a `specialization constant` if push
  constants prove too slow (they shouldn't at d ≤ 50).
- Try `local_size_x = 64` if the NVIDIA driver schedules better
  with smaller workgroups for tiny `n` — measure before changing.

If the bench shows the shader is *correct but the speedup is
under 100×*, that means a second bottleneck exists outside the
shader (HLO compile, queue submission, fence wait) and we need
to instrument the Linux-side dispatch path before going further.

## Optional parallel work

The `reduce_full_f64.spv` shader from the previous TODO (close
the only Vulkan f64 full-axis reduce gap) is still useful and
unblocked. If you want a quick win between leapfrog iterations,
pick that up — it's independent of the leapfrog work and was
already specified at the previous version of this file (see
`git show 09280e3:248_TODO.md` for the spec).

## Cross-reference

- `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md` — the
  full plan and rationale (this is its execution).
- `~/projects/learn_erl/nx_vulkan/RESEARCH_FAST_KERNELS.md` —
  the break-even rule that predicted this exact bottleneck.
- `~/projects/learn_erl/pymc/exmc/docs/VULKAN_KNOWN_ISSUES.md` —
  the empirical evidence (3 test runs, all hung on NUTS sampling).
- `~/projects/learn_erl/nx_vulkan/CHECKLIST_FUSED_LEAPFROG.md` —
  Linux-side work order, checked off as it lands.
