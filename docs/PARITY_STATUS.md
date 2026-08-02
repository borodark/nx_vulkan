# PARITY_STATUS — regenerated Nx.Backend gap for VulkanoBackend

> ℹ️ **Dated snapshot (2026-07-28).** Its central claim — the name-only
> Nx.Backend gap is empty, every callback implemented — still holds. But
> it predates two later changes on `main`: native **f32** compute (this
> doc's op list shows f64-only) and the `Nx.Vulkan.Compiler` fusion
> compiler. Current suite: **863 doctests, 361 tests, 0 failures** (this
> doc's "130 tests" is stale).

**Date:** 2026-07-28
**Host:** mac.247 (FreeBSD 15.0 Linux-compat layer), Vulkan via **llvmpipe (LLVM
19.1.7)** — software rasteriser through the real vulkano stack. This is the
correctness reference the project validates against.
**nx:** 0.13.0 (note: `mix.lock` is gitignored; the committed `mix.exs` pins
`~> 0.13`, so `mix deps.get` resolves 0.13.0 fresh on each host).
**Supersedes:** `docs/NX_PARITY_RESEARCH.md` / `docs/nx_parity_gap.csv`
(2026-05-25) — both stale, do not use.

## Regenerated gap (path-independent)

```elixir
impl = Nx.Vulkan.VulkanoBackend.__info__(:functions) |> Enum.map(&elem(&1, 0)) |> MapSet.new()
cbs  = Nx.Backend.behaviour_info(:callbacks)          |> Enum.map(&elem(&1, 0)) |> MapSet.new()
MapSet.difference(cbs, impl) |> Enum.sort()
#=> []      (nx 0.13.0 declares 115 callbacks; every one is implemented)
```

**The name-only gap is empty.** Every `Nx.Backend` callback in nx 0.13 has an
implementation in `lib/nx_vulkan/vulkano_backend.ex`.

### What changed under nx 0.13 (why the old worklist looks "done")

nx 0.13 **removed** a batch of ops from the `Nx.Backend` behaviour. They are no
longer dispatched as backend callbacks — Nx composes them from lower-level
primitives or routes them through the `block/4` callback:

| op(s) | how nx 0.13 dispatches now |
|---|---|
| `cholesky` `determinant` `solve` `qr` `lu` `svd` `eigh` `top_k` `cumulative_max/min/product` `all_close` | via **`block/4`** (a `Nx.Block.*` struct) → our `block/4` transfers to `BinaryBackend` |
| `take` `take_along_axis` `logical_not` | composed from primitives (`gather` / `slice` / elementwise), which we implement |

`triangular_solve/4` **remains** a real callback and stays implemented.

The consequence: the module still *carries* `def cholesky/2`, `def take/4`,
etc., each annotated `@impl true`, but nx 0.13 never calls them — they are dead
clauses that emit `got "@impl true" ... but no behaviour specifies such
callback` warnings. Phase 1 removes them; correctness is preserved by `block/4`
+ primitive composition (verified vs `BinaryBackend`).

## Bucket classification (per the governing rule: incidental vs hot-kernel)

### 1. Native shader — accelerated on the GPU (already landed)
`add` `subtract` `multiply` `divide` `pow` `max` `min` (elementwise binary),
`exp` `log` `sqrt` `abs` `negate` `sigmoid` `tanh` `floor` `ceil` `sign`
(elementwise unary), `sum` `reduce_max` `reduce_min` (axis reductions),
`dot` (2-D f64 matmul), `transpose` (2-D), `reshape` `squeeze` (zero-copy),
`constant` `iota` `eye`, `concatenate` (outer axis), plus the fused leapfrog
chain family.

### 2. Native shader — IN SCOPE, not yet landed (Phase 2)
`conv` `fft` `ifft` (and `fft2` after 1-D `fft`). **Currently host-fallback**
(round-trip to `BinaryBackend`) — to be promoted to real f64 Vulkan compute
shaders. A host fallback here is the silent-performance-cliff the rule warns
against: `conv`/`fft` are hot kernels for Axon/Scholar workloads.

### 3. Host fallback — incidental, correct-but-CPU (landed)
Comparisons (`equal` … `greater_equal`), `select`, `all` `any`, bitwise/logical
families, trig (`sin` `cos` …), `product`, `window_*` (7), `reduce`,
`gather` `take` `take_along_axis`, `argmax` `argmin`, `sort` `argsort` `top_k`,
`cumulative_*`, `clip`, `pad`, `put_slice`, `indexed_put` `indexed_add`,
`broadcast`, `stack`, `slice`, `as_type`, `bitcast`, `reverse`, `to_batched`,
`all_close`, `logical_not`, and small linalg (`cholesky` `qr` `lu` `svd` `eigh`
`solve` `triangular_solve` `determinant`) via `block/4`.

### 4. Skip — permanent (documented in LIMITATIONS.md)
`from_pointer` `to_pointer` (FFI handles — nothing to compute; `BinaryBackend`
itself raises) and `phase` (complex — the shader ISA is f64-**real**, no complex
type). `fft`/`ifft`/`conv` are **no longer** in this bucket.

## Snapshot (2026-07-28, pre-cleanup)

**Compile warnings** (`mix compile 2>&1 | grep -i warning`): the 15 dead
`@impl true`-on-non-callback clauses listed above, one `to_batched/3` unused
`out` variable, and one pre-existing `native_v.ex` clause-grouping warning on
`leapfrog_chain_synth_f64/6` (in the fused-leapfrog path — guardrailed, left
untouched).

**Tests** (`mix test`): the orphaned legacy `test/nx_vulkan_test.exs` and the
`init/0`-based `test/nx_vulkan/pipeline_cache_test.exs` — both left broken by
commit `bb94217` ("Drop spirit C++ backend"), referencing the deleted
`Nx.Vulkan.Backend` / `Nx.Vulkan.Fuse` / `Nx.Vulkan.init/0` APIs — were removed.
After removal: **78 tests, 0 failures** on this host's Vulkan.

## Phase 1 outcome (2026-07-28)

- Removed the 15 dead `@impl true`-on-non-callback clauses (cholesky,
  determinant, solve, qr, lu, svd, eigh, take, take_along_axis, top_k,
  cumulative_max/min/product, all_close, logical_not) and the duplicate
  explicit `all/3`/`any/3` (the `for op <- [:all, :any]` loop remains).
- Fixed a **latent `to_batched/3` bug**: the fallback read `opts[:batch_size]`,
  but nx 0.13 encodes batch size in the `out` template's leading dim (opts
  carries only `:leftover`) — every `Nx.to_batched` call would have crashed.
  Now derived from `out.shape`.
- `vulkano_backend.ex` compiles **warning-free**. One pre-existing warning
  remains in `native_v.ex` (`leapfrog_chain_synth_f64/6` clause grouping) — it
  is in the guardrailed fused-leapfrog path and is not a parity op, so left
  untouched.
- New `test/nx_vulkan/parity_fallback_test.exs` verifies all removed ops (via
  `block/4` and via primitive composition) plus the retained fallbacks against
  a `BinaryBackend` reference in f64. Full suite: **106 tests, 0 failures**.

## Phase 2 outcome (2026-07-28) — native GPU shaders

**`fft` / `ifft`** (commit d644bc4): real f64 radix-2 Cooley-Tukey on the GPU —
bit-reversed complex load + log2(n) butterfly stages in one auto-synchronised
command buffer, twiddles precomputed in Rust f64. GPU-covered: last axis,
power-of-two length == axis size, real-f64 or complex-f64 input, c128 output.
Verified on-GPU (result stays on VulkanoBackend) and bit-identical to
BinaryBackend (maxerr 0.0; ifft∘fft roundtrip ~2e-16). fft2/mixed-radix/other
axes/padded lengths/f32→c64 host-fall-back, still correct.

**`conv`** (commit f1df6d1): real f64 im2col + GEMM on the GPU. im2col shader
unfolds the input (stride, low padding, input & kernel dilation folded into the
index math); a conv-GEMM shader multiplies by the flattened kernel and writes
canonical {N,Cout,O_total} output directly. GPU-covered: spatial rank 1..3,
feature/batch groups == 1, identity permutations, f64. Verified on-GPU and
bit-identical to BinaryBackend across strides/padding/dilation/multichannel/
batch/1d/2d/3d (maxerr 0.0). Groups>1, non-identity permutations, non-f64 and
rank>3 host-fall-back, still correct.

Both use `glslangValidator`-compiled shaders in `priv/shaders/` and dispatch via
new `NativeV` NIFs (`fft`, `conv_im2col`, `conv_gemm`) in the vulkano lib.rs.

## Definition of done (from PARITY_TASK.md)

- [x] Regenerated gap shows only permanent skips remaining (gap is empty; the
      skip set falls back rather than being "missing").
- [x] Phase 1 ops correct vs `BinaryBackend`; `mix test` green; the parity
      module compiles warning-free.
- [x] Phase 2 `conv` + `fft`/`ifft` run on the GPU (verified: result stays on
      VulkanoBackend for the covered case) and correct vs `BinaryBackend`.
      fft2 / mixed-radix fft / grouped & permuted conv remain correct via host
      fallback (documented follow-ons, `LIMITATIONS.md`).
- [x] `LIMITATIONS.md` lists the true skips.

Full suite after Phase 2: **130 tests, 0 failures** on this host's Vulkan.
