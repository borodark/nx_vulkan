# Nx 0.12.1 upgrade verification — mac-248 (FreeBSD GT 750M)

Date: 2026-07-05

## nx_vulkan (`feat/nx-0.12-compat`)

| Suite | Result |
|---|---|
| `mix test test/nx_vulkan/` | **84/84 pass, 0 failures** |
| Nx version | 0.12.1 (upgraded from 0.11.0 via `mix deps.update nx`) |

The mix.lock shipped with the branch had Nx 0.11.0 pinned; ran
`mix deps.update nx` to pull 0.12.1. All 84 tests pass including
the f64 matmul tests added earlier on `feat/f64-matmul`.

## exmc (`feat/nx-0.12` on `origin`)

| Suite | Result |
|---|---|
| `mix test` | **222/236 pass, 14 failures** |
| Nx version | 0.12.1 |

### 14 failures — all in `Nx.Defn.Expr.optional/3` removal

All 14 failures are in the same describe block: `v0.1 phase 1.9 —
dense linalg` in `test/nx_vulkan_test.exs`. Root cause:

`Nx.Defn.Expr.optional/3` was removed in Nx 0.12. The `Nx.Vulkan.Fast`
module (`nx_vulkan/lib/nx_vulkan/fast.ex`) calls it at 6 sites:

- `leapfrog_position` (line 63)
- `leapfrog_momentum_half` (line 77)
- `momentum_step` (line 94)
- `inv_mass_apply` (line 109)
- `kinetic_energy` (line 128)
- `normal_logpdf` (line 148)

These are fused-kernel optional dispatch points. The `Nx.Vulkan.Backend`
has matching `optional/3` callbacks (line 1220+). Nx 0.12 removed the
`Expr.optional` mechanism — needs a replacement strategy.

**This is an nx_vulkan library code change**, not an exmc change. The
`feat/nx-0.12-compat` branch only bumped the mix.exs constraint but
did not update `fast.ex`.

### No other regressions

The remaining 222 tests pass cleanly. The `MassMatrix.finalize_dense`
fix (wrapping in `Nx.with_default_backend(Nx.BinaryBackend, ...)`) works
correctly — no backend leak crashes.

## Recommendation

1. Fix `Nx.Vulkan.Fast` to use whatever Nx 0.12 provides instead of
   `Expr.optional/3`. Likely `Nx.Defn.Kernel.optional/3` or a custom
   backend callback mechanism.
2. If `optional` has no replacement in Nx 0.12, the `Fast` module can
   be bypassed (all 6 kernels have `_fallback` functions that use
   standard Nx ops). The fused-kernel optimization would be lost but
   correctness is maintained.
3. Re-run after the fix; expect 236/236.
