# Benchmarks

**Scope:** every measured number this project claims, with its method and its
caveats. Read the baseline note first — it is the difference between an honest
figure and a flattering one.

> **Read the baseline before the multipliers.** Most tables here compare against
> `Nx.BinaryBackend`, which is a pure-Elixir interpreter. That is the right
> reference for *correctness* — it is the host fallback, so it is what every
> result must match bit-for-bit — and it is the honest reference on the FreeBSD
> Keplers, where EXLA is not built and the interpreter genuinely is the
> alternative. It is **not** a performance reference on a machine that can run
> EXLA.
>
> Measured across a width sweep on the RTX 3060 Ti: **EXLA on the host CPU
> computes the same gradient 20× faster than this backend at small model sizes
> and 215× faster at 6×10⁶ elements**, with the gap widening. There is no
> reachable model width on that box where this backend is the faster choice
> ([`bench_results/MODEL_SCALING.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/MODEL_SCALING.md)).
>
> So a "436×" below means *436× a tree-walking interpreter*, not 436× a
> competent CPU. **This project's case is reach, not speed** — see
> [`STANDING.md`](STANDING.md). Where CUDA exists,
> use EXLA.

### Batched dispatch (August 2026)

One `value_and_grad` step of an MNIST MLP at batch 32, submit-per-dispatch
(`NXV_BATCH_MAX=0`) vs batched. Same graph, same commit, arms differing only in
the environment variable:

| host | GPU | before | after | |
|---|---|---:|---:|---|
| super-io | RTX 3060 Ti (Ampere, 2021) | 16.4 ms | **9.6 ms** | 1.71× |
| mac-247 | GT 650M (Kepler, 2012) | 14.6 ms | **8.8 ms** | 1.65× |
| mac-248 | GT 750M (Kepler, 2013) | 13.3 ms | **9.1 ms** | 1.45× |

The loss is identical in every arm on every host — two architectures, two
operating systems. Batching changes *when* work is submitted, never what is
computed. Reproduce with
[`examples/mnist_mlp_step_bench.exs`](https://github.com/borodark/nx_vulkan/blob/main/examples/mnist_mlp_step_bench.exs);
method and the full cap sweep in
[`bench_results/BATCHED_DISPATCH.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/BATCHED_DISPATCH.md).

> **The three tables below predate batched dispatch** and are therefore
> pessimistic by roughly the factors above. They have not been re-run because
> their harnesses were scratch projects rather than committed examples — so
> they are left at their measured values rather than adjusted by arithmetic.

### CNN training (August 2026, pre-batching)

One `value_and_grad` step, batch 32, versus `Nx.BinaryBackend`. Losses are
bit-identical to the host on every row.

| model | super-io (RTX 3060 Ti) | mac-247 (GT 650M, 2012) | mac-248 (GT 750M, 2013) |
|---|---|---|---|
| conv→conv→dense, strided | **31.0 ms** (436×) | 35.1 ms (477×) | **25.4 ms** (440×) |
| LeNet-style, max-pooled | **84.1 ms** (363×) | 77.6 ms (434×) | **64.3 ms** (334×) |
| inference, batch 256 | 17.5 ms (1107×) | 71.1 ms (274×) | 31.8 ms (407×) |

The LeNet step was **20.9 s** before the backward pass stopped falling back to
the host — the same measurement, same box. Read the absolute times rather than
the multipliers: a speedup here mostly measures how slow pure-Elixir
`BinaryBackend` is, which varies by host CPU. The absolute GPU times cluster in
25–85 ms across three cards spanning 2012–2021, because at this model size the
work is dispatch-bound rather than compute-bound — which is why a 2012 laptop
GPU is competitive with a 2021 desktop one.

### vs EXLA (August 2026, pre-batching)

The [Axon MNIST guide](https://axon.hexdocs.pm/mnist.html) model, one training
step at batch 32 on the RTX 3060 Ti — a dense-only MLP, which is the shape most
favourable to EXLA and least favourable here:

| backend | ms | vs BinaryBackend |
|---|---:|---:|
| Vulkan, eager | 14.1 | 485× |
| Vulkan, `Nx.Vulkan.Compiler` | 18.5 | 370× |
| EXLA (CUDA) | 0.715 | 9581× |

**EXLA is ~20× ahead, and fusion does not close it — it costs 24%** (0.76×
fused vs eager). On a graph that is almost all `dot`, there is no elementwise
work for fusion to amortise against, so the deficit is dispatch overhead and
GEMM quality rather than missing whole-graph compilation.

That diagnosis is what batched submission acts on, and it is why this race is
labelled pre-batching: the eager row above has since improved ~1.7× on this
box. The race has **not** been re-run — it needs a working EXLA, which this
repo deliberately does not depend on — so no combined figure is claimed here.
The remaining lever the measurement points at is GEMM quality.

On a 2×strided-conv CNN the gap is similar — 41.3 ms eager vs 1.45 ms, with
fusion neutral at 0.98× — so EXLA leads on both graph shapes tested. The
qualitative difference is availability: EXLA cannot be installed on the two
FreeBSD Keplers at all, where Vulkan runs the same CNN in 64–78 ms. Full
numbers, an XLA gradient-compile failure narrowed to one specific conv
configuration, and what it took to get EXLA running:
[`bench_results/MNIST_EXLA_RACE.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/MNIST_EXLA_RACE.md).

### f32 vs f64 per op

`sh scripts/race.sh` — f32 speedup over f64, same shapes, all on-GPU:

| op | super-io | mac-247 | mac-248 |
|---|---|---|---|
| matmul 512³ | 0.61× | 0.47× | 0.45× |
| conv 16→32ch | 1.7× | 1.21× | 3.48× |
| elementwise add 1M | 2.38× | 4.3× | **7.01×** |
| sum 1M | 1.97× | 1.9× | 1.9× |

**f32 matmul is slower than f64, and that is the accumulator policy working as
designed**: f32 matmul defaults to `matmul_f32_f64acc.spv`, paying an f32→f64
conversion on top of the same f64 MAC rate. f32 wins where it is meant to —
bandwidth-bound elementwise and reductions. Switch with
`Nx.Vulkan.VulkanoBackend.put_f32_matmul_accumulator(:f32)` if you want speed
over the f64-accumulated reference.

### Square matmul (May 2026)

Milliseconds per dispatch, median of 50–200 iterations:

| size | bin (super-io) | bin (mac-247) | vulkano (super-io) | vulkano (mac-247) |
|---|---|---|---|---|
| 16×16 | 2.76 | 2.51 | 1.18 | **1.06** |
| 64×64 | 130.76 | 158.45 | 7.07 | 7.92 |
| 256×256 | 20,097 | 13,891 | 149.19 | **136.10** |
| 1024×1024 | n/a (hours) | n/a (hours) | 2,323 | 2,843 |

Full bench: [`examples/full_bench.exs`](https://github.com/borodark/nx_vulkan/blob/main/examples/full_bench.exs).

## How these were measured

Two working documents carry the method behind the numbers above, and neither is
a summary — they are the raw investigations, kept because the wrong turns in
them are the useful part.

- [`ELEMENTWISE_PCIE_TAX.md`](https://github.com/borodark/nx_vulkan/blob/main/docs/ELEMENTWISE_PCIE_TAX.md) — how a bandwidth
  deficit was localised to the allocator rather than the shader. `Nx.multiply`
  on a 448 GB/s card was sustaining 16.4 GB/s, and the elimination is the
  method: not dispatch overhead (chaining N ops gives a flat marginal cost), not
  a fallback (`Fallback.count/1` returns an empty map), not the shader. It was
  the memory-type filter — `PREFER_DEVICE | HOST_*` pairs a preference with a
  requirement, and the requirement wins, so every output buffer lived in system
  RAM and every store the shader executed crossed PCIe. **Resolved since:** the
  fix landed and the same path now measures 431 GB/s of 448. The follow-on
  "27x is now ~14x, find the rest" headline was a **210 MHz DVFS reading** — the
  card's idle floor — and there was nothing left to find. The doc's closing
  "not yet checked" list is partly answered too: the 32 MiB cliff is vulkano's,
  per-allocation, agreed 6/6 across boxes, and the BAR1 cliff is a separate
  256 MiB cumulative whole-process budget. Do not conflate them.
- [`DTRACE_VULKAN_PROFILING.md`](https://github.com/borodark/nx_vulkan/blob/main/docs/DTRACE_VULKAN_PROFILING.md) — profiling the
  dispatch stack on FreeBSD. **Read the banner and the "probes that actually
  work" section; the rest is kept for technique, not symbol names.** The trap it
  documents is worth the visit on its own: `vkQueueSubmit` and `vkWaitForFences`
  exist in `libvulkan.so.1` and appear in `dtrace -l`, so they look probeable —
  but vulkano resolves driver pointers through `vkGetDeviceProcAddr` and calls
  the ICD directly, so probing them records **zero events**. A probe that fires
  never is indistinguishable from a code path that runs never.
