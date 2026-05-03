# mac-248 — Fused leapfrog **chain** shader (Phase 1.5)

**Goal**: write `leapfrog_chain_normal.{comp,spv}` — a single
Vulkan shader that performs **K consecutive** NUTS leapfrog
steps for a univariate Normal model in one dispatch, emitting
all K intermediate `(q, p, grad, logp)` states.

**Why this replaces the previous 248_TODO**: the single-step
`leapfrog_normal.spv` shipped on `feat/fused-leapfrog-normal`
(commit `e794c5b` on nx_vulkan main, your `4eac8c68` on spirit)
works correctly but only delivers ~11× per-step speedup. The
Vulkan dispatch baseline is ~500 µs *regardless of shader
complexity* — measured 537 µs/step at d=8. To go below that
floor, multiple steps must share one dispatch.

The chain shader's amortized per-step cost is `(500 µs +
K × shader_compute) / K`. At K=32 → ~16 µs/step (3× faster
than EXLA), at K=64 → ~8 µs/step (6× faster). The acceptance
test bar (Exponential-Poisson NUTS in ≤ 30s under
`EXMC_COMPILER=vulkan`) becomes hittable.

Branch: **`feat/fused-leapfrog-chain-normal`** off the previous
`feat/fused-leapfrog-normal` (preserves the single-step
infrastructure as a baseline).

## Layout note

Mac-248 (FreeBSD 15, GT 750M) flat layout. Pull and branch:

```
cd ~/spirit && git fetch origin && git checkout feat/fused-leapfrog-normal
git pull && git checkout -b feat/fused-leapfrog-chain-normal
cd ~/nx_vulkan && git fetch origin && git checkout feat/fused-leapfrog-normal
git pull
```

## The shader

The shader runs K leapfrog steps in a loop, writing all K
intermediate states. The eXMC NUTS speculative path consumes
exactly this format: `(all_q[K,n], all_p[K,n], all_grad[K,n],
all_logp[K])` — the audit confirmed the binding contract.

Per-step `logp` requires a workgroup reduction (sum across `n`
dimensions). Use shared memory + barrier; standard pattern.

```glsl
#version 450

// Fused chain of K NUTS leapfrog steps for a univariate Normal
// log-density model. Emits all K intermediate (q, p, grad, logp)
// states so the caller (Elixir tree builder) can do U-turn checks
// and divergence detection on the host without re-dispatching.
//
// Push constants: {n, K, eps, mu, sigma} = 20 bytes.
// Output buffers are K × n (q, p, grad) and K (logp) — at K=32,
// n=50 that's ~25 KB total, comfortably within buffer pool limits.

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
layout (std430, binding = 6) writeonly buffer Out_logp { float logp_chain[]; }; // K

shared float partial[256];

const float LOG_2PI_HALF = 0.91893853320467274;  // 0.5 * log(2π)

void main() {
    uint i   = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationIndex;

    bool in_bounds = (i < pc.n);
    float inv_var  = 1.0 / (pc.sigma * pc.sigma);
    float log_sigma = log(pc.sigma);
    float n_f = float(pc.n);

    // Each thread carries its own dimension's running state.
    float qi = in_bounds ? q_init[i] : 0.0;
    float pi = in_bounds ? p_init[i] : 0.0;
    float mi = in_bounds ? inv_mass[i] : 0.0;

    for (uint k = 0; k < pc.K; k++) {
        // Half-step momentum at q
        float grad_q = in_bounds ? -(qi - pc.mu) * inv_var : 0.0;
        float p_half = pi - 0.5 * pc.eps * grad_q;

        // Full-step position
        qi = qi + pc.eps * mi * p_half;

        // Half-step momentum at q_new
        float grad_qn = in_bounds ? -(qi - pc.mu) * inv_var : 0.0;
        pi = p_half - 0.5 * pc.eps * grad_qn;

        // Write per-dimension chains
        if (in_bounds) {
            q_chain[k * pc.n + i]    = qi;
            p_chain[k * pc.n + i]    = pi;
            grad_chain[k * pc.n + i] = grad_qn;
        }

        // Per-step logp = -0.5 * sum_i z_i² - n*(log(σ) + 0.5*log(2π))
        // where z_i = (q_i - μ) / σ. Reduce across workgroup.
        float zi = in_bounds ? (qi - pc.mu) / pc.sigma : 0.0;
        partial[tid] = zi * zi;
        barrier();

        for (uint s = 128u; s > 0u; s /= 2u) {
            if (tid < s) partial[tid] += partial[tid + s];
            barrier();
        }

        if (tid == 0u) {
            logp_chain[k] = -0.5 * partial[0] - n_f * (log_sigma + LOG_2PI_HALF);
        }

        // Ensure all threads have finished reading partial[] before
        // the next iteration overwrites it.
        barrier();
    }
}
```

~75 lines including the reduction. Single workgroup of 256
threads is sufficient for `n ≤ 256` (covers eXMC's typical
`d ≤ 50` sweet spot). For larger `n`, the per-step reduction
needs multiple workgroups → second-pass reduction → caller
handles. **Phase 1.5 assumes `n ≤ 256`** (single workgroup).
Document this clearly in the C++ shim.

**Push constants budget**: 20 bytes (well under 128).

**Workgroup-size note**: `local_size_x = 256` is intentional —
matches the `n_groups = (n + 255) / 256` dispatch math used by
the Linux side for all our reduction-style shaders. Don't
change without coordinating with the dispatch code.

## Compile + push

```
cd ~/spirit
glslangValidator -V shaders/leapfrog_chain_normal.comp \
                 -o shaders/leapfrog_chain_normal.spv
git add shaders/leapfrog_chain_normal.{comp,spv}
git commit -m "shaders: leapfrog_chain_normal — K-step fused chain for univariate Normal"
git push origin feat/fused-leapfrog-chain-normal
```

If glslang flags anything (likely candidates: shared-memory
size constraints, barrier placement warnings, or the
write-only buffer + partial workgroup edge case at `n < 256`),
fix in place and re-push.

## Sanity check before pushing

Validate that the shader outputs are correct for K=2 by
hand-stepping. Push constants `{n=4, K=2, eps=0.1, mu=0.0,
sigma=1.0}`, inputs `q=[1,2,3,4]`, `p=[0.5,0.5,0.5,0.5]`,
`inv_mass=[1,1,1,1]`:

```
Step 0:
  grad_q[i] = -q[i]   (since mu=0, sigma=1)
  p_half[i] = 0.5 - 0.5*0.1*(-q[i]) = 0.5 + 0.05*q[i]
  qn[i]     = q[i] + 0.1*p_half[i]
  grad_qn[i]= -qn[i]
  pn[i]     = p_half[i] - 0.5*0.1*(-qn[i])

For q[0]=1: p_half=0.55, qn=1.055, grad_qn=-1.055, pn=0.60275
For q[1]=2: p_half=0.60, qn=2.060, grad_qn=-2.060, pn=0.703
For q[2]=3: p_half=0.65, qn=3.065, grad_qn=-3.065, pn=0.80325
For q[3]=4: p_half=0.70, qn=4.070, grad_qn=-4.070, pn=0.9035

logp[0] = -0.5 * (1.055² + 2.060² + 3.065² + 4.070²) - 4 * 0.91893853
        = -0.5 * (1.113 + 4.244 + 9.394 + 16.564) - 3.6757541
        = -0.5 * 31.315 - 3.6758
        = -15.6575 - 3.6758
        = -19.3333

Step 1: feed q_chain[0,*] back as q, p_chain[0,*] as p, repeat.
For q[0]=1.055: grad=-1.055, p_half=0.60275-0.5*0.1*(-1.055)=0.65553,
                 qn=1.055+0.1*0.65553=1.12055, ...
```

These numbers exactly match the per-step sanity check
performed on the Phase 1 single-step shader (commit
`e794c5b`); the chain shader at K=2 must reproduce them across
two iterations. If not, the loop-body or shared-memory
reduction has a bug.

## After your push

Linux side will:

1. Vendor the new shader into `nx_vulkan/priv/shaders/leapfrog_chain_normal.spv`.
2. Add `nxv_leapfrog_chain_normal` to the C++ shim (7-buffer
   dispatch, 20-byte push constants).
3. Add `leapfrog_chain_normal` Rust NIF that allocates the four
   output buffers and returns a 4-tuple of `ResourceArc`s.
4. Add `Nx.Vulkan.leapfrog_chain_normal/N` and the named-kernel
   form `Nx.Vulkan.Fast.leapfrog_chain_normal/N`.
5. Wire the eXMC speculative path: 2-line conditional in
   `lib/exmc/nuts/tree.ex` `ensure_available/3` (lines 524 + 572).
6. Correctness test: same posterior recovery vs unfused path
   on the Normal-Normal model.
7. Run the acceptance test: Exponential-Poisson under
   `EXMC_COMPILER=vulkan EXMC_FUSED_LEAPFROG=true`, target
   ≤ 30 seconds.

The eXMC audit (preserved in PLAN_FUSED_LEAPFROG.md) confirmed
the speculative path is leapfrog-agnostic at the dispatch
boundary — no tree builder, NIF, or merge changes required.
Total Linux-side effort: 1-2 person-days.

## What this DOES NOT need from you

- No C++, Rust, or Elixir wiring.
- No tests of the eXMC integration.
- No K-selection logic (eXMC already chooses K = max(32,
  2 × tree-depth) per iteration).

## Optional parallel work

Same as the previous TODO: `reduce_full_f64.spv` is unblocked
and useful, picked from `git show 09280e3:248_TODO.md`. Pick
that up between leapfrog iterations if the chain shader needs
debug cycles.

## Cross-reference

- `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md` —
  full plan, including the Phase 1 lessons-learned and the
  Phase 1.5 design rationale.
- `~/projects/learn_erl/nx_vulkan/RESEARCH_FAST_KERNELS.md` —
  break-even rule that explains why we measured what we
  measured.
- `~/projects/learn_erl/pymc/exmc/lib/exmc/nuts/tree.ex` —
  speculative path (lines 524, 572 are the swap sites).
- `~/projects/learn_erl/pymc/exmc/lib/exmc/nuts/batched_leapfrog.ex` —
  the existing XLA batched-leapfrog whose contract the chain
  shader must match.
- Phase 1 baseline commit on nx_vulkan: `e794c5b` on
  `feat/fused-leapfrog-normal`.
- Spirit Phase 1 shader: commit `4eac8c68` on
  `feat/fused-leapfrog-normal`.
