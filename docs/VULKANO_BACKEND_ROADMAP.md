# Nx.Vulkan.VulkanoBackend — Roadmap

**Primary objective (2026-05-20 onward):** make
`Nx.Vulkan.VulkanoBackend` a viable Nx backend for the three target
ecosystems — **exmc** (NUTS sampling on FreeBSD), **Axon** (neural
networks, with autograd), **Scholar** (classical ML, with linalg).

Previously: `Nx.Vulkan.Backend` (C++ spirit) was the Vulkan backend.
The C++ Elixir backend has since been **removed** (commit `bb94217`);
`Nx.Vulkan.VulkanoBackend` is the only backend. It was preferred because:

- Resource lifetimes are managed by Rust ownership
  (`Arc<Buffer>` + `Subbuffer<u8>`), eliminating the stale-handle
  bug class that bit the R4 cutover.
- vulkano builds + runs cleanly on FreeBSD 15.0 and Linux without
  vendor-specific shims.
- vulkano matches the C++ spirit path's dispatch latency within
  ~10% on the bench target (GT 650M).
- Per-op shaders (the existing SPV catalog under `priv/shaders/`)
  load and dispatch identically — no shader rewrite needed.

## Where we are

| Layer | Status |
|---|---|
| Buffer lifecycle NIFs (alloc/upload/download/byte_size) | ✓ |
| Chain shader dispatch (`leapfrog_chain_synth`) | ✓ |
| `VulkanoBackend` storage callbacks (`from_binary`, `to_binary`, transfer, constant, iota, eye) | ✓ |
| `VulkanoBackend` binary SPV ops (add/sub/mul/div/pow/max/min) | ✓ |
| `VulkanoBackend` unary SPV ops (exp/log/sqrt/abs/neg/sigmoid/tanh/floor/ceil/sign) | ✓ |
| `VulkanoBackend` reductions (sum/reduce_max/reduce_min) | ✓ |
| `VulkanoBackend` movement (reshape, squeeze, 2D transpose) | ✓ |
| `VulkanoBackend` matmul (rank-2 f64 fast path via SPV) | ✓ |
| `VulkanoBackend` comparison (host fallback) | ✓ |
| `VulkanoBackend` sampler-host ops (pad/put_slice/indexed_put/indexed_add/broadcast/concatenate/gather/take, all host fallback, Tier 1) | ✓ |
| Defn integration via Evaluator (works when global default = VulkanoBackend) | ✓ |
| Full Defn compiler (whole-graph fusion; generates + caches SPV) | ✓ (`Nx.Vulkan.Compiler`, thrust 3) |
| Autograd primitives (forward op coverage is gradient coverage via `Nx.Defn.grad`) | ✓ |
| Linalg ops (cholesky, solve, qr, svd) via host fallback through `block/4` | ✓ |
| Linalg ops — native SPV implementations | ✗ |
| Persistent buffer pool / `SubbufferAllocator` | ✗ |
| Pipeline cache persisted to disk | ✓ (UUID-validated, survives BEAM restarts) |
| Multi-device routing (Intel iGPU alongside NVIDIA on legacy MBP) | ✗ — planned in [`MULTI_DEVICE_PLAN.md`](https://github.com/borodark/nx_vulkan/blob/main/docs/MULTI_DEVICE_PLAN.md) |

Test coverage (2026-09): **833 doctests, 907 tests, 0 failures** on four boxes — GT 650M (Kepler, FreeBSD), GT 750M (mac-248), RTX 3060 Ti (Ampere, Linux) and a Tegra X1 Jetson Nano (unified memory, Ubuntu). The spirit C++ backend and its test suite were dropped. Bench coverage committed to `bench_results/`.

## Stage breakdown

Stages are sized to land in one focused session each.

### Stage 1 — Elementwise binary  *(DONE)*

**Ops:** `add`, `subtract`, `multiply`, `divide`, `pow`, `max`, `min`.

NIF: `apply_binary(out_ref, a_ref, b_ref, n, op_code, spv_path)` —
takes 3 buffer refs, dispatches `elementwise_binary.spv` (already in
`priv/shaders/`) with the op selected via specialization constant.
Push block: `uint n`. Workgroup 256, `ceil(n/256)` groups.

VulkanoBackend callbacks: 7 op handlers that allocate an output
buffer and call `apply_binary`. Validation: head-to-head against
`Nx.BinaryBackend` for each op on f32 tensors.

### Stage 2 — Elementwise unary  *(DONE)*

**Ops:** `exp`, `log`, `sqrt`, `abs`, `negate`, `sigmoid`, `tanh`,
`relu` (clamp to 0), `ceil`, `floor`, `sign`, `reciprocal`, `square`,
`erf`, `expm1`.

NIF: `apply_unary(out_ref, a_ref, n, op_code, spv_path)`. Same
pattern as binary, one input. SPV: `elementwise_unary.spv`.

### Stage 3 — Reductions  *(DONE — sum/reduce_max/reduce_min; non-trivial axis sets fall back to host)*

**Ops:** `sum`, `reduce_max`, `reduce_min` over all axes (full
reduction to scalar). Then per-axis via `reduce_axis.spv`.

### Stage 4 — Shape / movement  *(PARTIAL DONE — reshape, squeeze, 2D transpose [1,0] on GPU; broadcast/slice/pad/concatenate/gather/take on host fallback per Tier 1 of SHAPE_C_PLAN.md)*

**Ops:** `reshape` (zero-copy ref rewrap), `squeeze`, `broadcast`
(GPU-side broadcast shader for non-zero-stride cases), `transpose`,
`slice`, `gather`.

### Stage 5 — Linalg  *(PARTIAL — dot/matmul rank-2 f64 fast path on GPU; cholesky/solve/qr/svd via `block/4` host fallback; native SPV impls TODO)*

**Ops:** `dot/6` (matmul), `cholesky`, `solve`, `qr`, `svd`,
`determinant`. Some of these need new shaders; matmul has multiple
tilings already in `priv/shaders/`.

### Stage 6 — Random + comparison + select  *(PARTIAL — comparison, select, all, any on host fallback; Random TODO)*

**Ops:** `Nx.Random.*` (Philox-backed), `less`/`greater`/`equal`/
`not_equal`, `select`.

### Stage 7 — Defn integration  *(DONE for Evaluator path — pin global default to VulkanoBackend at boot (Application.start), route Exmc.JIT.jit through Nx.Defn.Evaluator instead of Nx.Vulkan.jit (which would force the spirit backend). Custom Defn compiler TODO.)*

So `defn` blocks targeting `Nx.Vulkan.VulkanoBackend` work end-to-
end. May require a custom Nx.Defn compiler or routing through the
existing Vulkan-aware compiler with vulkano backend.

### Stage 8 — Autograd primitives  *(DONE — forward op coverage IS gradient coverage. `Nx.Defn.grad` is a graph transformation; once forward ops exist, gradients automatic. Validated end-to-end on Axon training step on the spirit backend; vulkano path inherits via the same Defn substrate.)*

For Axon: implement gradients of all stage-1–6 ops. Most are
automatic via `Nx.Defn.grad/2` once forward-pass ops exist; some need
custom adjoint impls.

### Stage 9 — Axon parity  *(DONE — Axon training loop ran end-to-end on VulkanoBackend; matches BinaryBackend reference to 8.6e-8 on the dense_0 kernel gradient sum.)*

Run a small Axon model (MLP, small CNN) end-to-end on
`Nx.Vulkan.VulkanoBackend`. Compare loss + gradients against
`BinaryBackend` reference.

### Stage 10 — Scholar parity  *(DONE — Scholar LinearRegression smoke-test passed via the `block/4` host-fallback path. Native SVD / cholesky impl TODO before declaring full parity.)*

Run k-means or PCA on `Nx.Vulkan.VulkanoBackend`. The linalg ops
from stage 5 are the gate.

### Stage 11 — Performance pass  *(IN FLIGHT — Tier 1 of SHAPE_C_PLAN.md landed: host-fallback ops skip the upload-back round trip and return BinaryBackend tensors. Consumer bench shows median ~1.25-1.3x speedup when result is read via `to_flat_list`. Persistent buffer pool, disk pipeline cache, native shaders for the bandwidth-bound four (broadcast, pad, concatenate, put_slice) all TODO — see SHAPE_C_PLAN.md Tier 2.)*

Add persistent buffer pool, vulkano `SubbufferAllocator` integration,
pipeline cache to disk (vulkano's `PipelineCache::with_data`).
Compare to C++ spirit + EXLA on Axon training step / sec.

## Performance target

For exmc on GT 650M: regime-model NUTS sample ≤500 ms — **met**, via the
synthesised chain shader.

For Axon: "at least half of EXLA's throughput on the same hardware where
EXLA runs" was the original target. **It is not met, and the gap is
large enough that it should be stated rather than left implied.**

Measured on the Axon MNIST MLP, one training step, RTX 3060 Ti
([`MNIST_EXLA_RACE.md`](../bench_results/MNIST_EXLA_RACE.md)): EXLA
0.715 ms vs 14.1 ms eager — **EXLA ~20× ahead**. Batched submission has
since taken roughly 1.7× off that (`BATCHED_DISPATCH.md`), putting the
gap near **~12×**. The target implies a 2× gap. So the shortfall is
about **6×**, on the graph shape least favourable to this backend
(dense-only, almost all `dot`).

Two things that target got wrong, worth keeping in view:

- **It assumed the deficit was whole-graph compilation.** It is not:
  fusion *regresses* on this graph (0.76×). The measured levers are
  per-dispatch cost — now largely taken — and **GEMM quality**, which is
  untouched. A tiled 16×16 GEMM is not competitive with cuBLAS, and no
  amount of scheduling work substitutes for that.
- **"On the same hardware where EXLA runs" quietly excludes the fleet
  this project exists for.** EXLA does not run on the two FreeBSD
  Keplers at all, so on those the ratio is not 12× or 2× — it is
  undefined. A throughput target benchmarked only where the competitor
  can run is the wrong shape of goal for a portability-first backend.

Re-based: treat **closing the GEMM gap on Ampere** as the measurable
performance objective, and **op coverage + correctness on hardware EXLA
cannot reach** as the differentiating one. Do not re-assert a fraction-of-EXLA
number until a GEMM improvement has actually been raced.

## Non-goals

- ~~f64 compute~~ **(shipped)** and ~~f32 compute~~ **(shipped)** — the
  hot ops (elementwise, matmul, conv, reduce, transpose) have native f32
  **and** f64 shaders and dtype-dispatch on the tensor type. f64 is the
  default accumulator policy (correctness first; consumer GPUs are slower
  at f64), but f32 is native — no longer merely cast — and wins on
  bandwidth-bound ops.
- CUDA-specific features (tensor cores, mixed precision) — vulkano
  abstracts over them, but extracting them is out of scope until
  stages 1–10 are done.
- Multi-GPU. Single device per process for now — the engineering is scoped in
  [`MULTI_DEVICE_PLAN.md`](https://github.com/borodark/nx_vulkan/blob/main/docs/MULTI_DEVICE_PLAN.md), which is blocked on a
  *driver* gap rather than a code one: mac-247 has both an Intel HD 4000 and the
  GT 650M on the PCI bus, but `vulkaninfo` enumerates only the NVIDIA card and
  llvmpipe, because Mesa's `anv` is not loaded on the FreeBSD side. The code gap
  behind it — picking the first `DiscreteGpu` and ignoring the rest — depends on
  the ArcSwap refactor in
  [`CONTEXT_LIFECYCLE_PLAN.md`](https://github.com/borodark/nx_vulkan/blob/main/docs/CONTEXT_LIFECYCLE_PLAN.md), which is the
  same prerequisite for tearing a context down and rebuilding it.

## Open architectural questions

1. **Persistent buffer pool.** Per-call alloc/free works but hits
   the allocator on every op. A `SubbufferAllocator` keyed by size
   class would amortise this. Defer until stage 11. **Still open and
   still unmeasured** as of 2026-09 — but note that the *adjacent*
   experiment has been run and lost: a pool over the chain NIFs'
   function-local buffers was built and reverted (`190bf67`), worth
   2.2 µs at concurrency one and nothing from two concurrent callers up,
   because the single queue saturates before the allocator does. That
   says nothing directly about this item, whose buffers are GC-owned
   rather than function-local — but it does say to measure at realistic
   concurrency on a quiet box first. See `ROADMAP.md`.

2. **Pipeline cache.** vulkano supports `PipelineCache::with_data`
   for disk-persisted compiled pipelines. Plumb through after
   stage 5.

3. **Defn compiler.** EXLA has its own; we'd need either a
   `Nx.Defn.Compiler` impl that knows how to dispatch through
   `Nx.Vulkan.NativeV`, or rely on `Nx.Defn.Evaluator` driving the
   backend op-by-op. Stage 7 decides.

4. ~~**Hex publish strategy.**~~ **Answered.** Published as `nx_vulkan`
   itself — 0.1.0 (2026-05), 0.2.0 (2026-08), 0.3.0 (2026-08). There is
   no separate `nx_vulkan_vulkano` package and no C++ path to keep
   alongside it: the spirit backend was dropped in `bb94217`, so vulkano
   is not a parallel option but the only one.

5. **Multi-device on a single machine.** mac-247 (FreeBSD 15 +
   2013-era MacBook Pro) has the GT 650M Mac Edition AND an Intel
   HD Graphics 4000 (Ivy Bridge iGPU). `pciconf -lv` confirms both
   on the PCI bus:

       vgapci1: Intel HD Graphics 4000      (vendor 0x8086, dev 0x0166)
       vgapci0: NVIDIA GT 650M Mac Edition  (vendor 0x10de, dev 0x0fd5)

   Currently only NVIDIA is exposed to Vulkan. `vulkaninfo` shows
   `llvmpipe` as the second device (Mesa's software Vulkan, not
   the iGPU). To surface the Intel iGPU:

   - Load `i915kms` + `drm-kmod`
   - Confirm FreeBSD `graphics/mesa-libs` ships with the `anv`
     Intel Vulkan driver enabled for x86_64
   - Investigate Apple MUX state — early-2013 MBPs may hard-route
     the iGPU into low-power/standby when discrete is active

   Even with both surfaced, nx_vulkan's `ctx()` picks the first
   `DiscreteGpu` and ignores everything else. Multi-device routing
   would need:

   - Device selection in `ctx()` (env var, config, or runtime API)
   - Per-device pipeline cache + allocator
   - Either device-affinity tags on `Nx.Vulkan.VulkanoBackend`
     tensors, or a workload-router that chooses device per op

   Performance ceiling estimate: GT 650M = ~691 GFLOPS f32;
   HD 4000 = ~330 GFLOPS f32 (no f64). Theoretical +50% peak on
   mac-247; realistic +20-30% on well-partitioned workloads.

   Filed as long-term — not blocking M-II or W-stage work. Pick
   up if dual-device compute on the legacy MBP becomes interesting
   (e.g., for a "compute fabric from yesterday's hardware" demo).
