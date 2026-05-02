# mac-248 — Specialized Fast shaders (parallel work)

**Goal**: prototype 1-2 specialized shaders that don't fit the generic
4-input chain. The Linux side has shipped `Nx.Vulkan.Fast` with
named kernels that delegate to `fused_chain_4`; some MCMC hot paths
need shader changes the generic fused chain can't express.

Branch: `feat/fast-shaders` off `feature/vulkan-backend`.

Linux work in flight on `feat/fast-kernels` (nx_vulkan repo).

## Layout note

Mac-248 uses the flat layout: `~/spirit/`, `~/nx_vulkan/`.

## Two prototypes (pick whichever interests you)

### A. `kinetic_energy.spv` — fused (p² × inv_mass) reduction

NUTS computes `0.5 * sum(p² * inv_mass)` every leapfrog step. Today
this is 3 dispatches (multiply, multiply, sum) plus a host scalar
multiply by 0.5. Could collapse to 1 shader.

```glsl
#version 450

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint n;
} pc;

layout (std430, binding = 0) readonly  buffer In_p   { float p[]; };
layout (std430, binding = 1) readonly  buffer In_m   { float inv_mass[]; };
layout (std430, binding = 2) writeonly buffer Out_k  { float k_out[]; };

shared float partial[256];

void main() {
    uint tid = gl_LocalInvocationIndex;
    uint i = gl_GlobalInvocationID.x;

    float local_sum = 0.0;
    if (i < pc.n) {
        float pi = p[i];
        local_sum = pi * pi * inv_mass[i];
    }
    partial[tid] = local_sum;
    barrier();

    for (uint stride = 128u; stride > 0u; stride /= 2u) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        barrier();
    }

    if (tid == 0u) k_out[gl_WorkGroupID.x] = partial[0] * 0.5;
}
```

Output is one float per workgroup (caller does final reduction or we
allocate `out_k` of size = num_workgroups). The 0.5 multiplier is
baked into the shader so the kinetic-energy formula `0.5 * sum(p²*M⁻¹)`
becomes one dispatch instead of three.

Compile + push:

```
glslangValidator -V shaders/kinetic_energy.comp -o shaders/kinetic_energy.spv
git add shaders/kinetic_energy.{comp,spv}
git commit -m "shaders: kinetic_energy — fused 0.5*sum(p²*inv_mass)"
git push origin feat/fast-shaders
```

### B. `normal_logpdf.spv` — fused Gaussian log-density

`-0.5 * ((x-μ)/σ)² - log(σ) - 0.5*log(2π)`. Used everywhere in MCMC
distribution code. Currently 5+ dispatches. Could collapse to 1.

```glsl
#version 450

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint n;
} pc;

layout (std430, binding = 0) readonly  buffer In_x  { float x[]; };
layout (std430, binding = 1) readonly  buffer In_mu { float mu[]; };
layout (std430, binding = 2) readonly  buffer In_s  { float sigma[]; };
layout (std430, binding = 3) writeonly buffer Out_l { float logp[]; };

const float LOG_SQRT_2PI = 0.91893853320467274178;  // 0.5 * log(2π)

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;

    float z = (x[i] - mu[i]) / sigma[i];
    logp[i] = -0.5 * z * z - log(sigma[i]) - LOG_SQRT_2PI;
}
```

Output shape matches `x`. One dispatch instead of 5+.

Compile + push:

```
glslangValidator -V shaders/normal_logpdf.comp -o shaders/normal_logpdf.spv
git add shaders/normal_logpdf.{comp,spv}
git commit -m "shaders: normal_logpdf — fused Gaussian log-density"
git push origin feat/fast-shaders
```

## After your push

Linux side will:

1. Merge `feat/fast-shaders` into `feature/vulkan-backend`.
2. Add `nxv_kinetic_energy` and/or `nxv_normal_logpdf` C++ shims.
3. Add Rust NIFs.
4. Add Nx.Vulkan helper functions.
5. Add `Nx.Vulkan.Fast.kinetic_energy(p, inv_mass)` and/or
   `Nx.Vulkan.Fast.normal_logpdf(x, mu, sigma)` with defn fallbacks.
6. Tests + benchmarks.
7. Refactor exmc to use these named kernels (separate session).

## What this DOES NOT need from you

- No changes to existing shaders. fused_elementwise_4in is doing its
  job for `q + eps*p` and `p + half_eps*grad`.
- No backend C++ changes — these are pure new shaders + dispatches
  Linux side wires up.

## Cross-reference

- `~/projects/learn_erl/nx_vulkan/FAST_KERNELS_PLAN.md` — full
  approach plan + work split rationale
- `~/projects/learn_erl/emily/lib/emily/fast.ex` — reference impl
  showing the named-kernel pattern
- `~/projects/learn_erl/nx_vulkan/lib/nx_vulkan/fast.ex` — Linux
  side already shipping kernels 1-4 via `fused_chain_4`
