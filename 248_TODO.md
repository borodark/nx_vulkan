# mac-248 — Three concurrent tasks (pick any order)

The chain shader (`leapfrog_chain_normal.spv` at spirit
`53438dff`) is shipped and working on Linux. Per-step bench at
K=32 hits 49.7 µs (matches EXLA target). The Linux side is
working through correctness diagnosis (Stage 1.5.4 — variance
bias under f32 chain integration).

While that diagnosis proceeds, mac-248 has three independent
useful tasks. **Pick any order; they don't depend on each
other and don't depend on the Linux-side correctness work
finishing.**

## Layout note

Mac-248 (FreeBSD 15, GT 750M) flat layout:
`~/spirit/`, `~/nx_vulkan/`. Both have `feat/fused-leapfrog-chain-normal`
checked out. Pull before each task:

```
cd ~/spirit && git pull origin feat/fused-leapfrog-chain-normal
cd ~/nx_vulkan && git pull origin feat/fused-leapfrog-chain-normal
```

---

## Task A — FreeBSD bring-up of the chain pipeline

**Goal**: prove the entire fused-chain path works on FreeBSD's
nvidia driver, not just on the Linux dev box. This is the
strategic milestone the *walkable-path* blog post listed as
not-yet-done; a passing run here flips the post's central claim
from promise to measurement.

**Steps**:

```sh
# 1. Vulkan loader + headers + glslang (skip if already installed)
sudo pkg install vulkan-loader vulkan-headers vulkan-tools \
                 vulkan-validation-layers glslang shaderc

# 2. Confirm the GT 750M is visible
vulkaninfo --summary | head -30

# 3. Build nx_vulkan against the FreeBSD Vulkan stack
cd ~/nx_vulkan
mix deps.get
mix compile
# If headers/libs aren't on the default path:
#   CPATH=/usr/local/include LIBRARY_PATH=/usr/local/lib mix compile
#   RUSTFLAGS="-L /usr/local/lib"

# 4. Run the nx_vulkan test suite — all 152 tests should pass
mix test

# 5. Run the new chain bench
mix run -e '
Nx.Vulkan.init()
n = 8
{:ok, q} = Nx.Vulkan.upload_f32(for i <- 1..n, do: i / 10.0)
{:ok, p} = Nx.Vulkan.upload_f32(for _ <- 1..n, do: 0.3)
{:ok, m} = Nx.Vulkan.upload_f32(for _ <- 1..n, do: 1.0)
for _ <- 1..50, do: Nx.Vulkan.leapfrog_chain_normal(q, p, m, 32, 0.05, 0.0, 1.0)
{us, _} = :timer.tc(fn ->
  for _ <- 1..200, do: Nx.Vulkan.leapfrog_chain_normal(q, p, m, 32, 0.05, 0.0, 1.0)
end)
IO.puts("K=32 on GT 750M FreeBSD: #{Float.round(us / 200, 1)} µs/dispatch = #{Float.round(us / 200 / 32, 2)} µs/step")
'

# 6. (Optional but valuable) run the K=1, 2, 8, 16, 32, 64, 128 sweep
#    so we have full per-step numbers on FreeBSD GT 750M to compare
#    against the Linux RTX 3060 Ti measurements.
```

**Report back**:

- `vulkaninfo --summary` first 20 lines
- `Nx.Vulkan.Native.device_name()` and `has_f64()` outputs
- Any package-install issues, build errors, or test failures
- The K=32 µs/step number (and the K-sweep if you ran it)
- `mix test` summary line

If something fails to build, capture the exact error and stop;
Linux side will adjust the build script. If the build is fine
but a specific shader fails to load, note which `.spv` and the
validation error.

**Expected outcome**: 152/0 on the test suite, K=32 on GT 750M
likely ~150-300 µs/step (Kepler is older than the RTX 3060 Ti
but Vulkan compute should be in the same order of magnitude).
A successful run validates that the *walkable-path* claim is
real.

**~30-60 minutes** including the package install.

---

## Task B — Multi-workgroup chain shader (`leapfrog_chain_normal_lg.comp`)

**Goal**: lift the `n ≤ 256` constraint of the current chain
shader. The single-workgroup version uses a workgroup-shared
reduction for the per-step `logp`. For `n > 256`, multiple
workgroups need to cooperate via either a second-pass reduction
or a different algorithm.

**Approach**: simplest correct version first — write the per-step
reduction as a *two-pass* shader pair:

1. `leapfrog_chain_normal_lg.comp` — per-workgroup partials. Each
   workgroup processes 256 dimensions and emits its own per-step
   partial sum. Output: `q_chain[K,n]`, `p_chain[K,n]`,
   `grad_chain[K,n]`, `partial_logp[K, num_workgroups]`.
2. The Linux side does a tiny second-pass sum over the
   `num_workgroups` partials to get final `logp_chain[K]`. (This
   is host-side cheap, ~µs.)

```glsl
#version 450
layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint n;
    uint K;
    uint num_workgroups;   // ceil(n / 256)
    float eps;
    float mu;
    float sigma;
} pc;

layout (std430, binding = 0) readonly  buffer In_q       { float q_init[]; };
layout (std430, binding = 1) readonly  buffer In_p       { float p_init[]; };
layout (std430, binding = 2) readonly  buffer In_mass    { float inv_mass[]; };
layout (std430, binding = 3) writeonly buffer Out_q      { float q_chain[]; };       // K × n
layout (std430, binding = 4) writeonly buffer Out_p      { float p_chain[]; };       // K × n
layout (std430, binding = 5) writeonly buffer Out_grad   { float grad_chain[]; };    // K × n
layout (std430, binding = 6) writeonly buffer Out_partial{ float partial_logp[]; };  // K × num_workgroups

shared float partial[256];

void main() {
    // Same per-thread carrying as the single-workgroup version:
    // each thread handles dimension i = gl_GlobalInvocationID.x
    // (across all workgroups). The K-loop is identical. The only
    // difference is the per-step logp reduction:
    //   - thread 0 of each workgroup writes its partial sum to
    //     partial_logp[k * num_workgroups + workgroup_id]
    //   - host sums those partials per K to get final logp_chain[k]
    //
    // ... (see leapfrog_chain_normal.comp for the per-step body) ...
}
```

The dispatch is `(num_workgroups, 1, 1)` instead of `(1, 1, 1)`.

**Expected outcome**: chain shader works for any `n`, with no
regression on the `n ≤ 256` fast path (you can keep the
single-workgroup shader as a separate file and the Linux side
picks based on `n`).

**~1-2 hours** of GLSL work + one-shot validation against the
existing single-workgroup shader at `n = 256` (must produce
identical output).

---

## Task C — `leapfrog_chain_exponential.spv` (Phase 2 stretch)

**Goal**: clone the chain pattern for one more distribution
family — Exponential (rate λ). Closed-form unconstrained
gradient: for `q ~ Exp(λ)` on the unconstrained line via
log-transform `q_uc = log(q)`, the unconstrained log-density is
`log p(q_uc) = q_uc - λ * exp(q_uc)` (includes Jacobian) and
`grad_q_uc = 1 - λ * exp(q_uc)`.

This is Phase 2 of the original PLAN_FUSED_LEAPFROG.md scope:
expand from "univariate Normal only" to common single-RV
distributions. Exponential is the easiest non-Normal because:

1. Closed-form gradient (no autodiff in the shader)
2. Single scalar parameter (`λ`) → push constants stay tiny
3. The transform makes it unconstrained → no constraint
   handling

```glsl
#version 450
layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint  n;
    uint  K;
    float eps;
    float lambda;   // rate parameter
} pc;

// Same buffer layout as leapfrog_chain_normal: q, p, inv_mass
// in; q_chain, p_chain, grad_chain, logp_chain out.

// Per-step:
//   grad_q = 1.0 - pc.lambda * exp(qi);    // unconstrained gradient
//   p_half = pi - 0.5 * pc.eps * grad_q;
//   qi     = qi + pc.eps * inv_mass[i] * p_half;
//   grad_qn = 1.0 - pc.lambda * exp(qi);
//   pn     = p_half - 0.5 * pc.eps * grad_qn;
//   logp[k] (per-step reduction over n):
//     -log(λ) terms cancel in the workgroup sum; the per-element
//     contribution is qi - lambda * exp(qi). Sum then add
//     n * log(λ) per step on the host (or here).
```

**The `exp()` call inside the per-step loop** is the only
substantive difference from the Normal shader. SPIR-V's `exp` is
fast on NVIDIA hardware. Numerical caution: `exp(qi)` overflows
for `qi > ~88` (f32). For typical Bayesian models that's not
reachable, but worth noting.

**Optional but recommended**: hand-verify K=2 against a Python
or Stan reference for one (q_init, p_init, eps, λ) tuple before
pushing.

**~3-4 hours** including the math derivation and sanity check.

**Why this is stretch**: only useful AFTER the Linux side
finishes Stage 1.5.4 (variance correctness). If the Normal
chain has a real semantic bug, the same bug would propagate to
this shader. Best to wait until Normal is verified, then port
the pattern.

---

## Optional ongoing — `reduce_full_f64.spv`

Still unblocked, still useful. Spec at `git show
09280e3:248_TODO.md`. Closes the only Vulkan f64 full-axis
reduce gap. Independent of everything above. Pick it up if any
of A/B/C is in a debug cycle.

## Cross-reference

- `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md` — full
  plan, including Phase 1 / Phase 1.5 history and Phase 2 scope.
- `~/projects/learn_erl/nx_vulkan/CHECKLIST_FUSED_LEAPFROG.md` —
  Linux-side stage tracker. Stage 1.5.3 (eXMC wiring) is
  shipped on `pymc/main` at `f78b42733`. Stage 1.5.4 in flight.
- `~/projects/learn_erl/pymc/www.dataalienist.com/blog-walkable-path.html`
  — strategic context. Task A is the milestone that makes the
  central claim measured.
