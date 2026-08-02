# Fleet conformance + race — VulkanoBackend across 3 GPUs

> ℹ️ **Dated snapshot (2026-07-30, @ `7ba0767`).** Test counts below
> (839 doctests / 174 tests) are from before the fusion-compiler + f32
> work landed. The current suite is **863 doctests, 361 tests, 0
> failures** on the same three GPUs.

Driven over SSH from mac-247, 2026-07-30, branch `f32-matmul-prototype` @
`7ba0767`. "Use all 3 hosts to implement test and race."

## Conformance (Nx's own doctest suite as default backend)

`test/nx_vulkan/nx_doctest_test.exs` runs `doctest Nx` with VulkanoBackend as the
default backend — the community-standard backend validation. **Identical on all
three GPUs:**

| Host | GPU | Arch | `mix test` |
|---|---|---|---|
| mac-247 | GeForce GT 650M | Kepler | **839 doctests, 174 tests, 0 failures** |
| mac-248 | GeForce GT 750M | Kepler | **839 doctests, 174 tests, 0 failures** |
| super-io (249) | GeForce RTX 3060 Ti | Ampere | **839 doctests, 174 tests, 0 failures** |

839 / 954 of Nx's doctests pass; 115 excepted (documented: native-shader
last-ULP inspect diffs, complex + f8/f16 dtypes, and a tracked real-bug backlog —
see `ROADMAP_NEXT_BEST_NX.md` thrust 0). The suite found + we fixed two real bugs
(slice dynamic indices, composed-fallback default-backend leak). **The backend is
validated identically across Kepler and Ampere.**

**Re-validated after thrust-2 (2026-07-31, all on-GPU changes) at the latest
branch commit:** GT 650M, GT 750M, and RTX 3060 Ti each pass **863 doctests, 199
tests, 0 failures**. The broadcast/select/compare/cast shaders, the `cast`/
`apply_{binary_broadcast,select,compare}` NIFs, `robust_buffer_access`, and the
u8-packing/byte-extract tricks all work identically across Kepler and Ampere.

## Race (f32 vs f64, per GPU) — headlines

Full per-host reports: `f32_race_*_*.json`, `MAC248_GT750M_RESULTS.md`,
`AMPERE_SUPER_IO_RESULTS*.md`.

| | GT 650M / GT 750M (Kepler) | RTX 3060 Ti (Ampere) |
|---|---|---|
| tiled `:f32acc` matmul | ~1.8–2.7× | 2.18→2.97× (512→2048) |
| conv `:f32acc` | 1.4–4.4× | 2.46–3.06× |
| bandwidth ops (add/tanh/sum) | 1.9–4.8× | ~2× |
| f64 matmul tiling | 1.35–1.4× | ~1.15× |
| register blocking (32×32) | regresses fast path | regresses fast path |

Consistent story across the fleet: dtype-dispatched f32 + the accumulator policy
+ 16×16 tiling deliver 1.8–3× on the compute-bound fast path and ~2–5× on
bandwidth-bound ops; register blocking (as implemented) isn't a win anywhere.
