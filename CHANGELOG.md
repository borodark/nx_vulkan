# Changelog

## 0.2.0 (2026-08-02)

The fusion compiler release. First release since 0.1.0.

### Added

- **`Nx.Defn` fusion compiler (`Nx.Vulkan.Compiler`).** An
  `Nx.Defn.Compiler` that traces a `defn` to a stage schedule running
  on-device with GPU-resident intermediates and no interpreter fallback:
  elementwise fusion (one shader, one dispatch), parallel fused
  reductions (workgroup-per-slot tree reduce), and a multi-stage split at
  `dot` / `conv` / `reduce` / `transpose` boundaries with `reshape` /
  `squeeze` as zero-copy view boundaries and tuple/multi-output support.
  Whole dense/CNN layers, classifier heads, softmax, layernorm, and
  `x @ Wᵀ` fuse. Both f32 and f64. Cross-stage CSE exists but is
  default-off (raced across the fleet, never wins; opt-in `NXV_CSE=1`).
  Use: `Nx.Defn.jit(&fun/2, compiler: Nx.Vulkan.Compiler)`.
- **Native f32 compute.** The hot ops (elementwise, matmul, conv, reduce,
  transpose) now dtype-dispatch native **f32** shaders alongside f64,
  instead of casting f32 to f64. f64 remains the default accumulator
  policy (safe); f32 wins on bandwidth-bound ops and is available where a
  workload opts into it. This supersedes the "f64-only compute" note
  below for those ops.
- **conv** (im2col + GEMM) and **transpose** as native GPU ops, in f32
  and f64.

### Changed

- **f64-only compute** *(later superseded — see "Native f32 compute"
  under Added above; native f32 shaders were re-added for the hot ops).*
  Native shaders — elementwise binary/unary,
  reductions, rank-2 matmul (`matmul_f64.spv`), and the leapfrog chain
  synth — now run in **f64**; f32 inputs are accepted and cast to f64.
  This supersedes 0.1.0's "elementwise f32 + f64 / matmul f32-only"
  coverage: matmul is now native f64 (was host-fallback for f64), and
  the eXMC precision contract is `:f64` (EMLX, the f32-only backend, was
  dropped). Consumer GPUs are slower at f64, but correctness took
  priority over the f32 speed path.
- **Support Nx 0.13.** The `:nx` version constraint is now `{:nx, "~> 0.13"}`
  (was `~> 0.10 or ~> 0.11 or ~> 0.12`). `VulkanoBackend` and the f64
  chain-shader synthesis path run unchanged against nx 0.13's backend API.
  Required for consumers on nx 0.13 — e.g. eXMC's vulkan milestone, which
  references this repo from `mix` and could not resolve against the prior
  sub-0.13 constraint.

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
