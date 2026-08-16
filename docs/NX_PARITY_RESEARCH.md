# Nx Backend Parity Research — VulkanoBackend vs EXLA / EMLX

> ⚠️ **STALE — describes a world nx 0.13 dissolved.** Written against nx
> 0.12's 71-callback surface with 33 implemented. nx 0.13 restructured the
> backend behaviour and every callback is now implemented. `PARITY_STATUS.md`
> already labels this "stale, do not use"; this banner makes that visible
> without opening another file. For the real gap see `MISSION.md`.

**Date:** 2026-05-25
**Author:** parity research session
**Scope:** `Nx.Vulkan.VulkanoBackend` — gap analysis vs reference backends
**Status:** Research phase complete; implementation phase planned but not started

---

## TL;DR

`Nx.Vulkan.VulkanoBackend` implements **33 of 71** `Nx.Backend` callbacks. The 38-callback gap (≈54% of the API) spans 12 op families. **Exmc itself uses zero of the missing callbacks today** — the gap is downstream-facing (Scholar, Axon, Bumblebee, and future exmc features that would touch unimplemented ops).

Recommendation: close the gap in three tiers, starting with **easy host fallbacks for everything outside the `skip` families** (~16 callbacks, 1-2 days). Defer GPU-shader implementations until a workload requires them.

---

## The gap, by family

| family | count | EXMC usage | complexity | recommendation |
|---|---:|---:|---|---|
| reduction (all, any, all_close, product, reduce) | 5 | 0 | easy | **Tier 1 host fallback** |
| reduction-cumulative (sum/max/min/product) | 4 | 0 | easy-med | **Tier 1 host fallback** |
| shape (reverse, to_batched, bitcast) | 3 | 0 | easy | **Tier 1 host fallback** |
| sort (argsort, sort, top_k, take_along_axis) | 4 | 0 | medium | **Tier 1 host fallback** |
| window (max/min/product/sum/reduce/scatter_max/scatter_min) | 7 | 0 | medium | **Tier 1 host fallback** |
| logic (logical_not) | 1 | 0 | easy | **Tier 1 host fallback** |
| linalg (cholesky, det, eigh, lu, qr, svd, solve, triangular_solve) | 8 | 0 | medium | **Tier 1 host fallback via BinaryBackend's LinAlg** |
| system (optional) | 1 | n/a | n/a | **No impl needed** (optional callback) |
| fft (fft, fft2, ifft) | 2 | 0 | very hard | **Skip** (out of exmc scope) |
| conv | 1 | 0 | very hard | **Skip** (CNN territory; not exmc's domain) |
| pointer (from_pointer, to_pointer) | 2 | 0 | n/a | **Skip** (FFI surface) |
| complex (phase) | 1 | 0 | medium | **Skip** (no complex tensors in exmc) |

**Tier 1 candidates (host fallback):** 32 callbacks across 7 families → ~16 unique implementations (some share a pattern, e.g., the 7 window ops are nearly identical templates).

**Skip:** 7 callbacks across fft/conv/pointer/complex — explicitly leave unimplemented; document as "fall back to EXLA / BinaryBackend for these workloads."

**Already done:** 33 callbacks (init, from_binary, to_binary, backend_copy, backend_transfer, backend_deallocate, inspect, constant, iota, eye, broadcast, concatenate, gather, indexed_add, indexed_put, pad, put_slice, reduce_max, reduce_min, reshape, slice, select, squeeze, sum, take, transpose, as_type, dot, block, stack, argmax, argmin, clip).

---

## Why parity matters even when exmc doesn't need it

1. **Downstream libraries** — Scholar (ML algorithms), Axon (NN models), Bumblebee (transformer models) all depend on a full `Nx.Backend`. Today, running any of them on `VulkanoBackend` will crash on the first unimplemented op.
2. **Future exmc features** — diagnostics (`sort` for credible intervals), model comparison (`all_close` for posterior checks), batched sampling (`to_batched`), advanced linalg (`cholesky` for dense mass matrices) are all plausible near-term needs.
3. **Library-level claim** — "VulkanoBackend is a complete Nx backend on FreeBSD" is a much stronger pitch than "implements most of what one specific consumer needs."

---

## The 38 missing callbacks (full list, sorted by family)

### reduction (5)
- `all(out, tensor, keyword)` — reduce-AND over axis
- `all_close(out, tensor, tensor, keyword)` — pairwise tolerance check
- `any(out, tensor, keyword)` — reduce-OR over axis
- `product(out, tensor, keyword)` — multiplicative reduction
- `reduce(out, tensor, acc, keyword, fun)` — generic reduction with user function

### reduction-cumulative (4)
- `cumulative_max(out, t, keyword)`
- `cumulative_min(out, t, keyword)`
- `cumulative_product(out, t, keyword)`
- `cumulative_sum(out, t, keyword)`

### shape (3)
- `bitcast(out, tensor)` — reinterpret type without conversion
- `reverse(out, tensor, axes)` — reverse along axes
- `to_batched(out, tensor, opts)` — split leading axis into chunks

### sort (4)
- `argsort(out, tensor, keyword)`
- `sort(out, tensor, keyword)`
- `take_along_axis(out, tensor, indices, opts)`
- `top_k(out, tensor, opts)`

### window (7)
- `window_max(out, t, dimensions, opts)`
- `window_min(out, t, dimensions, opts)`
- `window_product(out, t, dimensions, opts)`
- `window_reduce(out, t, acc, dimensions, opts, fun)`
- `window_sum(out, t, dimensions, opts)`
- `window_scatter_max(out, t, source, init_value, dimensions, opts)`
- `window_scatter_min(out, t, source, init_value, dimensions, opts)`

### logic (1)
- `logical_not(out, tensor)`

### linalg (8) — all `Nx.LinAlg.*` callbacks
- `cholesky(out, tensor)`
- `determinant(out, tensor)`
- `eigh({eigenvals_out, eigenvecs_out}, tensor, keyword)`
- `lu({p_out, l_out, u_out}, tensor, keyword)`
- `qr({q_out, r_out}, tensor, keyword)`
- `solve(out, a, b)`
- `svd({u_out, s_out, vt_out}, tensor, keyword)`
- `triangular_solve(out, a, b, keyword)`

### skip (7) — explicitly not implementing
- `fft(out, tensor, opts)`, `fft2(out, tensor, opts)`, `ifft(out, tensor, opts)` — FFT (very-hard shaders; out of scope)
- `conv(out, tensor, kernel, opts)` — convolution (CNN-specific; out of scope)
- `from_pointer(...)`, `to_pointer(out, tensor)` — FFI (not exmc surface)
- `phase(out, tensor)` — complex-number argument (no complex tensors in exmc)

### system (1)
- `optional(out, tensor, fun)` — backend signaling that callbacks can fall through to a default. **No impl needed** — the optional-callback machinery in `Nx.Backend` handles this.

---

## Implementation tiers

### Tier 1 — Host fallback for all "not-skip" callbacks (~16 unique impls, 1-2 days)

Same pattern as our existing `stack`/`argmax`/`clip` host fallbacks: download to `Nx.BinaryBackend`, call `Nx.<op>` there, return on `Nx.BinaryBackend` per the Tier 1 contract.

```elixir
@impl true
def <op>(out, tensor, opts) do
  bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
  result = Nx.<op>(bin, opts)
  host_result(out, result)
end
```

The 7 window ops share a structure → a small macro or list-of-clauses pattern saves repetition.

The 8 linalg ops should delegate to `Nx.BinaryBackend.LinAlg.<op>` directly (which uses Erlang's `:numerl` or LAPACK).

**Expected effort:** 1-2 days for all of Tier 1 across the 32 callbacks. Tests for each (`assert Nx.equal(VulkanoBackend.<op>(...), Nx.BinaryBackend.<op>(...))`) add another day.

### Tier 2 — GPU shaders for hot ops (~1-2 weeks, opportunistic)

For ops where the host fallback's CPU + transfer cost dominates, implement a real Vulkan compute kernel. Candidates ranked by likely future hotness:

- **`cumulative_sum`** — prefix-scan kernel; well-known GPU pattern; useful for many ML algorithms
- **`sort` + `argsort`** — bitonic or radix sort on GPU; medium effort
- **`top_k`** — built on partial sort; medium effort
- **`reverse`** — trivial kernel
- **`all` / `any`** — reduce-AND/OR; small variant on existing reduce kernels

Each is a 200-500 LOC shader + Rust NIF + integration. Pick based on actual workload demand, not preemptively.

### Tier 3 — GPU linalg (months, defer)

`cholesky`, `qr`, `svd`, `eigh` are hard on GPU. The vulkan ecosystem doesn't have mature linalg shaders today. For exmc's "dense mass matrix" use case (8x8 max), host fallback is fine. Reconsider if a future model needs serious linalg perf on Kepler.

---

## Validation methodology — the parity test suite

### Design

`test/nx_vulkan/parity_test.exs` — programmatically:

1. For each `Nx.Backend` `@callback`, generate a fixture: representative input shape + dtype + values
2. Run the op on `EXLA.Backend` (reference)
3. Run the op on `VulkanoBackend`
4. Compare outputs:
   - f64: max abs error < 1e-10
   - f32: max abs error < 1e-6
   - integers: exact equality
5. Report per-op: PASS / FAIL / SKIP-UNIMPLEMENTED / NUMERIC-DIVERGE
6. Aggregate: parity score = passed / (passed + failed)

### Cross-host validation

Same test suite runs on:
- **super-io** (Linux + RTX 3060 Ti) — EXLA reference + VulkanoBackend target
- **mac-247** (FreeBSD + GT 650M) — VulkanoBackend only (no EXLA)
- **mac-248** (FreeBSD + GT 750M) — VulkanoBackend only

**Cross-host check:** same op + same seed + same input → same output across all three machines (within tolerance). Catches non-determinism in vulkano dispatch (reduction order, etc.).

### Fixture generation

```elixir
@fixtures %{
  add:   {%{a: {3, 4}, b: {3, 4}}, :f32},
  sort:  {%{tensor: {100}}, :s32},
  qr:    {%{tensor: {8, 8}}, :f64},
  conv:  :skip,  # documented out of scope
  ...
}
```

Seeded via `:rand.seed_s(:exsss, {1, 2, 3})` → reproducible across hosts.

### Output

JSON report per host:
```json
{
  "host": "mac-247",
  "device": "GT 650M (Kepler, f64=yes)",
  "vulkano_commit": "70bd017",
  "results": {
    "add": {"status": "PASS", "max_err": 1.2e-7},
    "cumulative_sum": {"status": "SKIP_UNIMPLEMENTED"},
    "qr": {"status": "PASS_HOST_FALLBACK", "max_err": 0.0},
    ...
  },
  "parity_score": 0.46
}
```

Diff across hosts → cross-host correctness gate.

### Estimated effort

- Test harness skeleton: 4-6 hours
- Per-callback fixtures: 5-10 min each × 71 callbacks = 6-12 hours
- First-pass run + result analysis: 2-3 hours
- **Total: 2-3 days** for a working parity suite (before any Tier 1 impl).

---

## Sequencing the work

| order | item | who | when |
|---|---|---|---|
| 1 | Build parity test harness (super-io) | dev | 1 day |
| 2 | Generate fixtures for all 71 callbacks | dev | 1 day |
| 3 | Baseline run on super-io → measure starting parity score | dev | 2 hours |
| 4 | Tier 1 host-fallback for the 16 unique callbacks | dev | 1-2 days |
| 5 | Re-run parity suite; verify score improves to ~88% (excluding skip set) | dev | 2 hours |
| 6 | Push to mac-247 + mac-248; run cross-host parity | dev | 3 hours |
| 7 | Diff cross-host outputs; investigate any drift | dev | 1-3 hours |
| 8 | Document parity score per host in `README.md` + this doc | dev | 1 hour |
| later | Tier 2 GPU shaders for hottest ops (data-driven) | dev | weeks |

**Total Tier 1 sprint:** ~5-6 days. After: VulkanoBackend is functionally complete for ~88% of the Nx API on three GPU types.

---

## Open questions

1. **Should `Nx.LinAlg.*` callbacks fall back through `:nx_extra` or BinaryBackend directly?** Need to check what `Nx.BinaryBackend.LinAlg` actually exports.
2. **What's the EXLA reference's exact reduction order?** If EXLA is non-deterministic, the cross-host diff might fail spuriously. May need `--xla_gpu_deterministic_ops` or similar.
3. **Window ops have an `acc` and `fun` argument** — does the GLSL emitter even need to be involved, or is host fallback always sufficient? Probably always-host (window kernels are rarely hot in exmc-style workloads).
4. **`to_batched` returns a STREAM**, not a tensor. Special handling needed at the backend-callback layer.
5. **What's the EXLA behavior on `optional` callback fallthrough?** Need to confirm we don't need to opt in via some attribute.

---

## References

- Raw gap inventory: `/tmp/parity_gap.csv` (39 missing callbacks with family + complexity metadata)
- Analysis script: `/tmp/parity_gap_analysis.exs`
- Nx callbacks source: `deps/nx/lib/nx/backend.ex`
- VulkanoBackend impl: `lib/nx_vulkan/vulkano_backend.ex`
- EXMC Vulkano DOs/DON'Ts: `docs/EXMC_VULKAN_DOS_AND_DONTS.md` (in exmc repo)
- Existing host-fallback callbacks (as Tier 1 pattern reference): `stack/3`, `argmax/3`, `argmin/3`, `clip/4` (commits `489f4b6`, `70bd017`)
