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

**Race** (`examples/f32_vs_f64_race.exs`, now covers matmul/conv/elementwise/
reductions), llvmpipe/CPU:

| op | f32 speedup | why |
|---|---|---|
| elementwise add 1M | **2.0×** | bandwidth-bound → halving bytes halves time |
| sum 1024² axis0 | **1.8×** | bandwidth-bound reduction |
| sum 1M full, tanh 1M | ~1.2× | partly compute-bound |
| matmul, conv | ~1.0× | compute-bound; f64 accumulator + native x86 f64 |

So even on CPU, **bandwidth-bound ops already win ~2×** (half the memory
traffic). The **compute-bound** ops (matmul/conv) show ~1× here only because x86
does f64 natively and the kernels accumulate in f64.

### Real GPU results — NVIDIA GeForce GT 650M (Kepler), 2026-07-29

Ran `scripts/race.sh` on the actual GT 650M (f64 rate-limited to ~1/24 of f32).
`bench_results/f32_race_mac_970cb1a.json`:

| op | f32 speedup | notes |
|---|---|---|
| elementwise add 1M | **4.14×** | bandwidth-bound; bigger than CPU |
| tanh 1M | **1.95×** | f32 transcendentals, no f64 |
| sum 1M / sum axis0 | **1.9× / 1.81×** | bandwidth-bound reductions |
| conv | 1.08–1.35× | im2col f32 helps; GEMM capped by f64 accumulator |
| **matmul** | **0.55–0.72× (SLOWER)** | the f64 accumulator is the bottleneck |

**Key finding — the f64 accumulator negates the compute-bound f32 win.** On a
device where f64 is rate-limited, `matmul_f32_f64acc` does the same slow f64 MACs
as `matmul_f64` (plus f32→f64 conversions), so it is *slower* than f64. Direct
3-way race (`examples/matmul_accumulator_race.exs`) on the GT 650M, 512³:

```
f64 = 21.1ms   f32/f64acc = 38.2ms (0.55x)   f32/f32acc = 12.7ms (1.67x)
```

A **pure f32 accumulator makes matmul 1.4–1.7× faster**; the accuracy-safe f64
accumulator makes it 0.55×. Conclusion, refined per family:

- **Bandwidth-bound** (elementwise, reductions): keep the f64 accumulator — it's
  cheap relative to memory and f32 still wins 1.8–4.1×. Ship as-is.
- **Compute-bound** (matmul, conv GEMM): the accumulator must be a **policy**.
  Default to f64 for accuracy, but offer a pure-f32 (or blocked/pairwise-f64)
  accumulator mode to actually get the speedup on f64-rate-limited GPUs. This is
  plan item 5 (precision policy), now justified by hardware data rather than
  omitted — the `matmul_f32_naive.spv` shader already exists as the fast variant.

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

### Item 5 — accumulator policy (implemented for matmul)

**Which dtype to compute in** stays keyed on the tensor's dtype — the "f32
option" is simply *create f32 tensors* (`Nx.tensor(.., type: {:f, 32})` /
`Nx.as_type`), compute follows storage, f64 is the default.

**Accumulator width**, however, is a genuine policy the GT 650M data forced (a
f64 accumulator makes compute-bound f32 matmul *slower* than f64 on rate-limited
GPUs). Implemented for matmul as a runtime setting, default `:f64`:

```elixir
Nx.Vulkan.VulkanoBackend.f32_matmul_accumulator()          #=> :f64  (default, accuracy-safe)
Nx.Vulkan.VulkanoBackend.put_f32_matmul_accumulator(:f32)  # fast on f64-starved GPUs
# or: config :nx_vulkan, :f32_matmul_accumulator, :f32
```

One knob governs **both** GEMM kernels — matmul and conv:
- matmul: `:f64` → `matmul_f32_f64acc.spv`, `:f32` → `matmul_f32_f32acc.spv`.
- conv GEMM: `:f64` → `conv_gemm_f32_f64acc.spv`, `:f32` → `conv_gemm_f32_f32acc.spv`
  (im2col is pure f32 movement, always exact).

Verified both dispatch on GPU under each policy and the ill-conditioned accuracy
gap (`test/nx_vulkan/{matmul_f32,conv}_test.exs`).

### Validated on Ampere (RTX 3060 Ti), decision: keep `:f64` default

super-io ran the brief on an RTX 3060 Ti (GA104, f64 ~1/32). Pattern confirmed
across two GPU generations — `:f64acc ≤ f64 ≤ :f32acc`. Through `Nx.dot`, 512³:
`:f32acc` is **1.5–2.0×** faster, `:f64acc` is 0.6–0.7× (slower); rel-err of
`:f32acc` ~1–3e-6, growing ~√K (textbook f32 GEMM). Full details:
`bench_results/AMPERE_SUPER_IO_RESULTS.md`.

**Decision (their recommendation, adopted): keep the default `:f64`; `:f32` stays
opt-in.** Measurement note: time with ≥5 warm-ups, ≥20 iters, and configs
interleaved — single-shot config-ordered runs mis-report by ~3×.

### 1024 cliff — fixed by tiling the f32 GEMM

The `:f32acc` win was narrow/size-dependent (1.09× at 1024 on Ampere) because the
naive one-thread-per-output kernel hit a bandwidth/occupancy wall. Both f32
matmul shaders now use **16×16 shared-memory tiling** (each workgroup stages
16×16 tiles of A and B through shared memory, reusing every global load 16×;
boundary-safe for non-multiple-of-16 shapes; block summation also improves
accuracy). Re-raced on the GT 650M via `Nx.dot`:

```
             before (naive)      after (tiled)
512³ :f32acc   1.72×               2.68×
1024³ :f32acc  ~1.1× (Ampere)      2.66×   ← cliff gone; scales flat
```

`:f32acc` now holds ~2.7× at both 512³ and 1024³ on Kepler (and is faster than
the naive kernel at 512 too). `:f64acc` also improved slightly (0.55→0.66× at
512, tiling removed its bandwidth component; the f64-ALU rate still caps it).
Correctness unchanged (err ~1e-7 across shapes incl. 30×17×23, 100×50×70). This
reopens the device-aware-default question with a much stronger, size-stable case
— **re-run `RACE_TODO_SUPER_IO.md` on Ampere to confirm the tiled kernel scales
there** before deciding.

**Conv GEMM tiled too.** All three conv-GEMM shaders (`conv_gemm_f64`,
`conv_gemm_f32_f64acc`, `conv_gemm_f32_f32acc`) now use the same 16×16 tiling —
conv's GEMM is `C = A·Wᵀ` (kernel `Cout×K`) written to the permuted
`{N,Cout,O_total}` layout, so the `conv_gemm` NIF now dispatches 2-D (Cout × M).
Boundary-safe (verified with Cout=7/13, M=25). On the GT 650M via `Nx.conv`,
larger convs: `{8,32,28,28}·{64,32,3,3}` → `:f32acc` **1.98×**;
`{4,64,16,16}·{128,64,3,3}` → **1.37×** (f64 exact, f32 err ≤7e-7).

**f64 matmul tiled too — every GEMM is now tiled.** `matmul_f64.spv` (the
f64-tensor path) also got 16×16 tiling. It stays f64-exact (err ~4e-16 incl.
30×17×23 / 100×50×70) and is ~1.35–1.4× faster from removing the bandwidth
component (GT 650M: 512³ 21→15.8ms, 1024³ 145→105ms); the f64-ALU rate is the
remaining ceiling. So the full GEMM inventory — matmul {f64, f32/f64acc,
f32/f32acc} and conv-GEMM {f64, f32/f64acc, f32/f32acc} — is tiled and
boundary-safe.

## Not converted (stay f64 / host, by design)

Leapfrog/MCMC chain (needs f64 for HMC stability), linalg + `block/4` family
(numerically sensitive, host-fallback), fft/ifft (f64 default; f32 spectral is a
future opt-in), comparisons (u8 output). Broadcast binary ops host-fall-back for
all dtypes (the broadcast shader is unwired).
