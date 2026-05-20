# Changelog

## 0.1.0 (2026-05-20)

First Hex release.

### Added — `Nx.Vulkan.VulkanoBackend` (pure-Rust path)

A new `Nx.Backend` implementation built on the [vulkano](https://github.com/vulkano-rs/vulkano)
Rust wrapper around Vulkan compute. Sibling to the existing
`Nx.Vulkan.Backend` (C++ spirit-backed); they share the SPV
catalog under `priv/shaders/` and the chain-shader synthesis
pipeline.

**Why a second backend.** A use-after-free in the C++ FFI layer
crashed the live trader three minutes after every restart —
`Nx.Vulkan.Native.byte_size` raising `:badarg` on a stale
`VkBuf*` pointer that had outlived its referent. Vulkano's
`Arc<Buffer>` ownership makes that bug class structurally
impossible: a `Subbuffer<u8>` cannot outlive its parent at the
Rust type level.

**What it ships.**

- Buffer lifecycle: `buf_upload`, `buf_alloc`, `buf_download`,
  `buf_byte_size`, `buf_upload_into`. Each wraps a vulkano
  `Subbuffer<[u8]>` in a Rustler resource; the BEAM GC's drop
  triggers vulkano's `vkDestroyBuffer + vkFreeMemory` chain.
- Compute ops (24 native through specialised SPVs):
  - **Elementwise binary** (f32 + f64): add, subtract, multiply,
    divide, pow, max, min.
  - **Elementwise unary** (f32 + f64): exp, log, sqrt, abs,
    negate, sigmoid, tanh, floor, ceil, sign.
  - **Reductions** (f32 + f64): sum, reduce_max, reduce_min;
    all-axes, leading-axis, trailing-axis.
  - **Shape / movement**: reshape (zero-copy), squeeze
    (zero-copy), 2D transpose.
  - **Matmul**: rank-2 × rank-2, f32 only.
- Host-fallback callbacks (correctness first; perf-native
  shaders pending): slice, as_type, comparison ops (equal,
  not_equal, less, less_equal, greater, greater_equal), select,
  all, any, dot (non-standard axis configs), `block/4`
  (routes `Nx.Block.LinAlg.SVD/QR/Cholesky/solve` through
  `BinaryBackend`).
- Pipeline cache keyed by `(spv_path, op_code)`. First call
  builds the layout + pipeline; subsequent calls reuse them.
  Required for long-running workloads (without it, vulkano's
  `StandardDescriptorSetAllocator` creates a fresh
  `DescriptorPool` per unique layout identity, eventually
  exhausting driver limits on FreeBSD).

**Validated workloads.**

- **Axon training step**: Dense → sigmoid → Dense + MSE +
  `Nx.Defn.value_and_grad`. Forward loss matches `BinaryBackend`
  byte-identical; gradient sum agrees to 1e-8. 100-step SGD
  trajectory matches at every step within 2e-6; final loss
  agrees to 4e-7 with both backends converging by 350×.
- **eXMC regime model log-posterior**: 8 free RVs, softmax-mixture
  custom likelihood over 200 observations. Matches `BinaryBackend`
  to 1e-7 at f64 precision. Roughly 2× faster than the C++ path
  on the bench target (GT 650M, FreeBSD 15.0).
- **Scholar linear regression** (normal equation + SVD):
  coefficients match `BinaryBackend` to 2e-6 on synthetic
  regression. SVD via host-fallback `block/4`.

**Autograd.** No backward callbacks were written. `Nx.Defn.grad`
is a graph transformation that expresses backward ops in terms
of forward ops — forward op coverage is therefore gradient
coverage when running through `Nx.Defn.Evaluator`. Validated
end-to-end via the Axon training step.

### Added — Mission II chain-shader synthesis

`Exmc.NUTS.CustomSynth`-style runtime synthesis of multi-RV
HMC/NUTS chain shaders. Take a multi-RV IR with a Custom
likelihood, trace via `Nx.Defn`, emit GLSL, compile to SPIR-V,
content-address cache, dispatch. Validated on the regime model
(8 RVs + 200-obs softmax-mixture) on GT 650M at 60 ms per K=32
leapfrog dispatch — 8.3× under the 500 ms/sample budget.

### Existing — `Nx.Vulkan.Backend` (C++ spirit path)

The legacy backend stays in this release. It runs the chain-shader
synthesis pipeline and the Mission II dispatch. The stale-handle
bug class that motivated the migration is still present; the
recommended path forward is `VulkanoBackend` for general Nx
work plus the spirit-backed chain dispatch (or vulkano's
chain-shader dispatch via `Nx.Vulkan.NativeV.leapfrog_chain_synth`)
for HMC.

### Build notes

- Rust 1.85 pinned via `rust-toolchain.toml`. See the comment in
  that file for the upstream rustler reason.
- Vulkan SDK + `glslangValidator` required:
  - Linux: `apt install libvulkan-dev vulkan-tools glslang-tools`
  - FreeBSD: `pkg install vulkan-loader vulkan-headers vulkan-tools glslang shaderc`
- vulkano 0.34 builds in ~30s on Linux, ~3:18 on FreeBSD 15.0.

### What's missing (the honest queue)

- Persistent buffer pool (per-call allocation works but costs
  a millisecond per dispatch).
- f64 matmul shader (regime model's `Nx.dot` falls back to host).
- Native linalg shaders (SVD, QR, Cholesky, solve) — Scholar
  currently routes these through host.
- Custom `Nx.Defn` compiler — today we run through
  `Nx.Defn.Evaluator` op-by-op; whole-graph optimisation is
  EXLA-style work.
- Convolutions, FFTs, sort, scatter — the long tail of Nx ops.
- R4 live-trader cutover — the production trader has not been
  switched to `VulkanoBackend` yet.

### Links

- Blog: [The Backend That Didn't Need to Know](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html)
- Roadmap: [`docs/VULKANO_BACKEND_ROADMAP.md`](docs/VULKANO_BACKEND_ROADMAP.md)
- 10-minute intro: [`livebooks/intro_10min.livemd`](livebooks/intro_10min.livemd)
- Examples: [`examples/axon_training_loop.exs`](examples/axon_training_loop.exs),
  [`examples/full_bench.exs`](examples/full_bench.exs)
