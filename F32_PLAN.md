# F32 compute path — plan + prototype

**Branch:** `f32-matmul-prototype` (off `parity-tier1`).
**Date:** 2026-07-28.

## Why

`VulkanoBackend` is f64-only compute (f32 inputs are cast to f64). f64 is
rate-limited on real GPUs — ~1/24 of f32 on the GT 650M (Kepler), ~1/32 on
consumer RTX — so an f32 path is a 2–30× win exactly where deep learning spends
its time. It is **zero win on this llvmpipe host** (both run on the CPU), so f32
must be an opt-in per-dtype path, not a default. The mechanism: keep data in f32
and dispatch f32 shaders on the tensor's real dtype, instead of upcasting.

## The governing test (per op)

f32 is worth it only where the op is **(a)** a compute/bandwidth-bound hot
kernel **and (b)** numerically tolerant of ~7 significant digits.

| Op family | f32? | Note |
|---|---|---|
| `dot`/matmul, `conv` | **yes — f32 I/O, f64 accumulator** | DL hot kernel; guard the accumulator |
| elementwise unary/binary, activations | yes | bandwidth-bound, DL-standard |
| reductions (sum/mean/max) | yes — f64 accumulator | bandwidth-bound; guard the sum |
| transpose/concat/slice/pad/gather | yes (free) | pure movement, no precision risk |
| fft/ifft | opt-in | fine for DSP/DL, keep f64 default |
| leapfrog chain (MCMC) | **no** | HMC needs f64 for energy conservation |
| linalg, comparisons | no | sensitive / host-fallback / u8 output |

## The accumulator decision (what this prototype demonstrates)

A naïve f32 matmul accumulates the K-length dot product in f32 and loses
precision as K grows (catastrophic cancellation, ~1e-3 relative at K~1e6). The
fix is **f32 I/O with an f64 accumulator**: read f32, accumulate `double`, store
f32. It is nearly free — the kernel is bound by f32 memory bandwidth, not the
accumulator — and it matches `Nx.BinaryBackend` (which itself accumulates in
Elixir f64) to f32 round-off. This prototype ships both shaders and measures the
difference to justify the choice (see `examples/f32_accumulator.exs`).

## Prototype scope (this branch)

- `dot`/matmul for `{:f, 32}` tensors → f32 shader with f64 accumulator, reusing
  the existing type-agnostic `matmul` NIF (buffers are raw bytes; the shader
  defines the element type). f32 output stays f32 on the GPU.
- Verified vs `BinaryBackend` (`test/nx_vulkan/matmul_f32_test.exs`) and an
  accuracy experiment vs the naïve-f32-accumulator baseline.

## Prototype results (2026-07-28, this host = llvmpipe/CPU Vulkan)

**Correctness** (`test/nx_vulkan/matmul_f32_test.exs`, 6 tests): f32 matmul
dispatches on the GPU, keeps the f32 dtype, and matches BinaryBackend to f32
round-off (< 1e-4); the f64 path is unchanged.

**Accumulator justification** (`examples/f32_accumulator.exs`):
- Ill-conditioned dot `[1e9, 1, …, 1, -1e9]`: naive f32 accumulator collapses to
  **0.0** (error = K−2, up to 1022); the **f64 accumulator is exact**.
- Well-conditioned random matmul: f64-acc error grows more slowly than naive —
  at K=4096, **2.7e-6 vs 4.7e-6**. The f64 accumulator is the right default.

**Race** (`examples/f32_vs_f64_race.exs`): on llvmpipe/CPU, f32 ≈ f64
(1.0–1.03×) — x86 does f64 natively and the shader still accumulates in f64, so
only load bandwidth differs (negligible at these sizes). **This host cannot show
the win**; on real GPU hardware f64 is rate-limited to ~1/24–1/32 of f32, where
the f32 path is a large speedup. f32 also halves memory footprint regardless of
device.

**Takeaway:** the mechanism (per-dtype shader dispatch + f64 accumulator) is
correct and cleanly extensible; the payoff is hardware-gated, so f32 should be an
opt-in path, benchmarked on a real GPU before flipping any default.

## Rollout status

- [x] **matmul** — `matmul_f32_f64acc.spv`, dispatched by dtype in `dot`.
- [x] **conv** — `conv_im2col_f32.spv` + `conv_gemm_f32.spv` (f64-acc GEMM);
      `conv_spvs/1` selects by dtype, buffers sized by `element_bytes/1`.
- [x] **elementwise** unary + binary — `elementwise_{unary,binary}_f32.spv`
      (same spec-constant op-code convention); filled the `binary_spv/unary_spv`
      `{:f,32}` arms.
- [x] **reductions** (sum/max/min) — `reduce_axis_f32.spv` with an f64
      accumulator for sum; `reduce_spv({:f,32})`.
- [x] **transpose** (2-D) — `transpose_f32.spv`; `transpose_spv/1`. (concat is a
      dtype-agnostic byte copy already; slice is host-fallback for all dtypes.)

All verified vs `BinaryBackend` (`test/nx_vulkan/{f32_ops,matmul_f32}_test.exs`):
algebraic/movement/reduction ops are exact, f32 transcendentals (exp/log/pow/
tanh/sigmoid) agree to ~f32 ulp (≤1.4e-6). Full suite: 171 tests, 0 failures.

### On item 5 (precision policy config)

Deliberately **not** adding a separate `default_precision` knob: the dispatch is
already keyed on the tensor's dtype, so the "f32 option" is simply *create f32
tensors* — the idiomatic Nx control (`Nx.tensor(.., type: {:f, 32})` /
`Nx.as_type(t, {:f, 32})`). Compute follows storage; f64 stays the default. A
forced-override (compute f32 even for f64 storage, or a wider accumulator) can be
layered on later via `init/1` opts if a real workload needs it, but it is not
required for the f32 path itself.

## Not converted (stay f64 / host, by design)

Leapfrog/MCMC chain (needs f64 for HMC stability), linalg + `block/4` family
(numerically sensitive, host-fallback), fft/ifft (f64 default; f32 spectral is a
future opt-in), comparisons (u8 output). Broadcast binary ops host-fall-back for
all dtypes (the broadcast shader is unwired).
