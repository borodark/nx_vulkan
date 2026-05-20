# Nx.Vulkan.VulkanoBackend — Roadmap

**Primary objective (2026-05-20 onward):** make
`Nx.Vulkan.VulkanoBackend` a viable Nx backend for the three target
ecosystems — **exmc** (NUTS sampling on FreeBSD), **Axon** (neural
networks, with autograd), **Scholar** (classical ML, with linalg).

Previously: `Nx.Vulkan.Backend` (C++ spirit) was the Vulkan backend.
The C++ path is now legacy; new work targets the vulkano path because:

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
| `VulkanoBackend` storage callbacks (`from_binary`, `to_binary`, transfer, constant) | ✓ |
| `VulkanoBackend` compute callbacks (add, sub, mul, sum, …) | ✗ |
| Defn integration | ✗ |
| Autograd primitives | ✗ |
| Linalg ops (cholesky, solve, …) | ✗ |

## Stage breakdown

Stages are sized to land in one focused session each.

### Stage 1 — Elementwise binary

**Ops:** `add`, `subtract`, `multiply`, `divide`, `pow`, `max`, `min`.

NIF: `apply_binary(out_ref, a_ref, b_ref, n, op_code, spv_path)` —
takes 3 buffer refs, dispatches `elementwise_binary.spv` (already in
`priv/shaders/`) with the op selected via specialization constant.
Push block: `uint n`. Workgroup 256, `ceil(n/256)` groups.

VulkanoBackend callbacks: 7 op handlers that allocate an output
buffer and call `apply_binary`. Validation: head-to-head against
`Nx.BinaryBackend` for each op on f32 tensors.

### Stage 2 — Elementwise unary

**Ops:** `exp`, `log`, `sqrt`, `abs`, `negate`, `sigmoid`, `tanh`,
`relu` (clamp to 0), `ceil`, `floor`, `sign`, `reciprocal`, `square`,
`erf`, `expm1`.

NIF: `apply_unary(out_ref, a_ref, n, op_code, spv_path)`. Same
pattern as binary, one input. SPV: `elementwise_unary.spv`.

### Stage 3 — Reductions

**Ops:** `sum`, `reduce_max`, `reduce_min` over all axes (full
reduction to scalar). Then per-axis via `reduce_axis.spv`.

### Stage 4 — Shape / movement

**Ops:** `reshape` (zero-copy ref rewrap), `squeeze`, `broadcast`
(GPU-side broadcast shader for non-zero-stride cases), `transpose`,
`slice`, `gather`.

### Stage 5 — Linalg

**Ops:** `dot/6` (matmul), `cholesky`, `solve`, `qr`, `svd`,
`determinant`. Some of these need new shaders; matmul has multiple
tilings already in `priv/shaders/`.

### Stage 6 — Random + comparison + select

**Ops:** `Nx.Random.*` (Philox-backed), `less`/`greater`/`equal`/
`not_equal`, `select`.

### Stage 7 — Defn integration

So `defn` blocks targeting `Nx.Vulkan.VulkanoBackend` work end-to-
end. May require a custom Nx.Defn compiler or routing through the
existing Vulkan-aware compiler with vulkano backend.

### Stage 8 — Autograd primitives

For Axon: implement gradients of all stage-1–6 ops. Most are
automatic via `Nx.Defn.Grad` once forward-pass ops exist; some need
custom adjoint impls.

### Stage 9 — Axon parity

Run a small Axon model (MLP, small CNN) end-to-end on
`Nx.Vulkan.VulkanoBackend`. Compare loss + gradients against
`BinaryBackend` reference.

### Stage 10 — Scholar parity

Run k-means or PCA on `Nx.Vulkan.VulkanoBackend`. The linalg ops
from stage 5 are the gate.

### Stage 11 — Performance pass

Add persistent buffer pool, vulkano `SubbufferAllocator` integration,
pipeline cache to disk (vulkano's `PipelineCache::with_data`).
Compare to C++ spirit + EXLA on Axon training step / sec.

## Performance target

For exmc on GT 650M: regime-model NUTS sample ≤500 ms (already met
via the synthesised chain shader). For Axon on FreeBSD: at least
half of EXLA's throughput on the same hardware where EXLA runs.

## Non-goals

- f64 compute (vulkano supports it but most consumer GPUs are
  ~32× slower at f64). Storage f64 is fine; compute defaults to
  f32 with `as_type` cast.
- CUDA-specific features (tensor cores, mixed precision) — vulkano
  abstracts over them, but extracting them is out of scope until
  stages 1–10 are done.
- Multi-GPU. Single device per process for now.

## Open architectural questions

1. **Persistent buffer pool.** Per-call alloc/free works but hits
   the allocator on every op. A `SubbufferAllocator` keyed by size
   class would amortise this. Defer until stage 11.

2. **Pipeline cache.** vulkano supports `PipelineCache::with_data`
   for disk-persisted compiled pipelines. Plumb through after
   stage 5.

3. **Defn compiler.** EXLA has its own; we'd need either a
   `Nx.Defn.Compiler` impl that knows how to dispatch through
   `Nx.Vulkan.NativeV`, or rely on `Nx.Defn.Evaluator` driving the
   backend op-by-op. Stage 7 decides.

4. **Hex publish strategy.** Once stages 1–6 land, publish a 0.1
   nx_vulkan_vulkano package. Existing `nx_vulkan` keeps the C++
   path until parity is comfortable.
