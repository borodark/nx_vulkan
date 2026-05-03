# Checklist — Fused leapfrog Normal (Phase 1)

Tracks the Linux-side work order for landing
`Nx.Vulkan.Fast.leapfrog_normal` and the eXMC opt-in. Mac-248
ships the shader (`leapfrog_normal.{comp,spv}`); each step
below is gated on the .spv being available in
`~/spirit/shaders/`.

Branch: `feat/fused-leapfrog-normal` on both `nx_vulkan` and
`pymc`. Mark each item `[x]` as it lands; cross-link the commit
SHA in the parenthetical.

## Gate 0 — Shader available

- [ ] `~/spirit/shaders/leapfrog_normal.spv` exists on
      `feat/fused-leapfrog-normal` (mac-248)
- [ ] `glslangValidator -V leapfrog_normal.comp` reports zero
      errors
- [ ] Commit SHA: ___________

## Stage 1 — Vendor the shader

- [ ] `cp ~/spirit/shaders/leapfrog_normal.spv nx_vulkan/priv/shaders/`
- [ ] Refresh `c_src/spirit/VENDOR.md` if Spirit's version moved
- [ ] Commit SHA: ___________

## Stage 2 — C++ shim

- [ ] Add `nxv_leapfrog_normal` declaration to
      `c_src/nx_vulkan_shim.h`
- [ ] Add implementation to `c_src/nx_vulkan_shim.cpp` —
      pattern-match on `nxv_kinetic_energy` (similar 5-buffer
      dispatch shape; push constants struct
      `{uint n; float eps; float mu; float sigma}`)
- [ ] `mix compile` — clean
- [ ] Commit SHA: ___________

## Stage 3 — Rust NIF

- [ ] Add `leapfrog_normal` `#[rustler::nif]` in
      `native/nx_vulkan_native/src/lib.rs`
- [ ] Signature:
      `(q, p, inv_mass, n: u32, eps: f32, mu: f32, sigma: f32)
       → NifResult<{ResourceArc<VulkanTensor>, ResourceArc<VulkanTensor>}>`
      (returns the two output tensors `(q_new, p_new)`)
- [ ] Allocate two output buffers via `nxv_buf_alloc`, dispatch,
      return both refs
- [ ] Add stub in `lib/nx_vulkan/native.ex`
- [ ] `mix compile` — Rustler builds clean
- [ ] Commit SHA: ___________

## Stage 4 — Elixir wrapper

- [ ] `Nx.Vulkan.leapfrog_normal/N` in `lib/nx_vulkan.ex` —
      raw NIF wrapper that takes Nx tensors, transfers, calls,
      returns Nx tensors (same pattern as
      `Nx.Vulkan.kinetic_energy/2`)
- [ ] Doctest with simple correctness check (one step,
      hand-computed expected output)
- [ ] Commit SHA: ___________

## Stage 5 — Named-kernel form (Fast)

- [ ] `Nx.Vulkan.Fast.leapfrog_normal/6` in
      `lib/nx_vulkan/fast.ex` — emits
      `Nx.Defn.Expr.optional(:fast_leapfrog_normal, …, fallback)`
- [ ] Defn fallback: composed `Nx.subtract`/`Nx.multiply`/etc.
      that any backend can run (so EXLA / EMLX / BinaryBackend
      get the same semantics from a non-Vulkan execution)
- [ ] `fast_leapfrog_normal` callback in `lib/nx_vulkan/backend.ex`
      that dispatches to `Nx.Vulkan.leapfrog_normal` when shapes
      / types fit, falls back to defn otherwise
- [ ] Test: same numerical output Vulkan vs defn-fallback
- [ ] Commit SHA: ___________

## Stage 6 — eXMC opt-in dispatch

In `pymc/exmc` repo, branch `feat/fused-leapfrog-normal`:

- [ ] Add `Exmc.NUTS.Leapfrog.fused_eligible?/1` — IR walker
      that returns `true` iff the model is exactly one Normal
      RV with no constraints, no observations, scalar mu &
      sigma. Conservative pattern match.
- [ ] Modify `Exmc.NUTS.Leapfrog.step/N` to dispatch to
      `Nx.Vulkan.Fast.leapfrog_normal` when `fused_eligible?`
      and `Application.get_env(:exmc, :fused_leapfrog, false)`
      and `Exmc.JIT.detect_compiler() == Nx.Vulkan`
- [ ] Default off; users opt in via
      `config :exmc, :fused_leapfrog, true` or
      `EXMC_FUSED_LEAPFROG=true`
- [ ] Commit SHA: ___________

## Stage 7 — Correctness test

- [ ] In `pymc/exmc/test/nuts/fused_leapfrog_test.exs`: run a
      simple Normal-Normal posterior with and without
      `EXMC_FUSED_LEAPFROG=true`, assert posterior mean &
      variance agree within MCMC noise (mean within ±0.1,
      variance within ±0.2 at N=1000 samples)
- [ ] Tag `@tag :requires_vulkan` so it only runs under
      `EXMC_COMPILER=vulkan`
- [ ] Commit SHA: ___________

## Stage 8 — Per-step benchmark

- [ ] Extend `nx_vulkan/bench/leapfrog_bench.exs` with the
      fused-path timing
- [ ] Target: ≤ 50 µs per leapfrog body (vs current Vulkan
      ~6000 µs → ~120× per-step)
- [ ] Commit SHA: ___________

## Stage 9 — Acceptance test

The make-or-break gate. If this fails, the project pauses for
diagnosis (probably HLO compile or queue-submission overhead
hiding under the dispatch overhead).

- [ ] `EXMC_COMPILER=vulkan EXMC_FUSED_LEAPFROG=true` running
      `mix test test/sampler_test.exs` (Exponential-Poisson and
      similar single-RV NUTS tests) completes in ≤ 30 seconds
- [ ] Wall-clock recorded: ___________
- [ ] Untag `:vulkan_known_failure` from the relevant tests in
      `pymc/exmc/test/` if they now pass
- [ ] Update `pymc/exmc/docs/VULKAN_KNOWN_ISSUES.md` issue #2
      with the resolution
- [ ] Commit SHA: ___________

## Stage 10 — Plan Phase 2 (or stop)

- [ ] If Stage 9 succeeded (the predicted ~200× speedup
      delivered): draft Phase 2 — same shader pattern for
      HalfNormal, Exponential, Beta, Gamma. Each as an
      independent shader, independent commit. List in
      PLAN_FUSED_LEAPFROG.md.
- [ ] If Stage 9 failed but stage 8 (per-step bench) showed the
      shader-only speedup: the bottleneck is in the surrounding
      tree code (Elixir-side bookkeeping, JIT compile,
      Nx.Vulkan dispatch overhead). Diagnose before further
      shader work.
- [ ] If Stage 8 also failed (shader is fast but the speedup
      doesn't show through to the test): instrumented profiling
      run to find the second bottleneck.
- [ ] Commit SHA: ___________

## Definition of done for Phase 1

All boxes above checked, both branches merged to their
respective `main`, the relevant `:vulkan_known_failure` tags
removed from the eXMC test suite, and the
`PLAN_FUSED_LEAPFROG.md` file updated to reflect the measured
outcome and inform whether Phase 2 is worth pursuing.
