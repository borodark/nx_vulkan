# Changelog

## Unreleased

The backward-pass release. 0.2.0 shipped conv, the fusion compiler and native
f32 — but a CNN *training* step still ran mostly on the CPU, and nothing in the
suite could see it.

### Fixed — the backward pass

A host fallback returns a bit-identical result, because the fallback *is* the
`Nx.BinaryBackend` reference every test compares against. So no assertion on
values can detect that an op silently left the GPU, and one had: conv's entire
backward pass ran in pure Elixir. Nx.Defn.Grad (hidden, so not linked) emits
ops nobody writes by
hand, and every GPU fast path had been gated on the shapes a *forward* pass
produces.

Eight ops moved back on-device. Seven were narrow gates refusing work the
existing shaders could already do; only `reverse` and `broadcast` needed new
kernels.

- **conv** — accepts non-identity input/kernel/output permutations (the
  gradient swaps the first two axes) by rotating into the native layout
  on-device, and coerces a mismatched operand dtype instead of refusing it.
- **dot** — accepts any rank-2 contraction orientation. `y = x·W` contracts
  `[1]/[0]` and always hit the shader; its gradients arrive as `[1]/[1]` and
  `[0]/[0]`, so **every dense layer** was paying two host matmuls per step.
- **reduce** — accepts a kept axis in the middle (`sum(axes: [0,2,3])`, the
  conv bias gradient) by rotating the kept axes to the front.
- **max/min pooling**, both directions. Forward is one thread per output;
  backward is one thread per *input*, which is what avoids float atomics and
  is why it requires non-overlapping windows. Ties go to the **last** maximum
  in row-major order, verified against `BinaryBackend` — `>` instead of `>=`
  is correct on random data and wrong on every relu-zero tie.
- **reverse** and **broadcast** — new index-remap shaders (rank ≤ 4). Both had
  no shader at all; `broadcast` produced the `{:s, 32}` zeros that in turn made
  `select` fall back.
- **integer literals** — `coerce_to/2` now converts rank-0 integer constants
  and, via new `cast_s32_to_f32/f64` shaders, integer tensors of any shape. Nx
  materialises literals as `{:s, 32}`: relu's `max(x, 0)`, a mean's divisor,
  `select`'s zeros and pooling's `init_value` were each dragging a whole tensor
  to the CPU behind a four-byte constant.

A CNN training step now performs exactly **one** host fallback: `pow` in f64,
which GLSL.std.450 does not provide and which should stay a fallback rather
than silently boundary-cast through f32.

### Added — verification that can see this class of bug

- **`Nx.Vulkan.Fallback`** — counts host fallbacks per process, attributing
  each to the callback at compile time. `assert Fallback.count_total(fun) == 0`
  turns an invisible performance cliff into a test failure. Off by default.
- **`test/nx_vulkan/grad_test.exs`** — gradient parity against
  `BinaryBackend`. The suite previously had **no** gradient tests at all, which
  is how the conv regression survived; "autograd for free" was the headline
  claim of the README and nothing exercised it.
- **`docs/BACKEND_VERIFICATION_GAP.md`** — what the Nx ecosystem does and does
  not verify about a third-party backend. `doctest Nx` contains zero gradient
  examples, `deps/nx` ships no `test/`, and upstream's own
  `Nx.Helpers.check_grads!` is behind the packaging wall.

### Performance

One `value_and_grad` step, batch 32, vs `Nx.BinaryBackend`, losses
bit-identical:

| model | RTX 3060 Ti | GT 650M (2012) | GT 750M (2013) |
|---|---|---|---|
| conv→conv→dense | 31.0 ms (436×) | 35.1 ms (477×) | 25.4 ms (440×) |
| LeNet-style | 84.1 ms (363×) | 77.6 ms (434×) | 64.3 ms (334×) |

The LeNet step was **20.9 s** before this work. Absolute GPU times cluster in
25–85 ms across cards spanning 2012–2021 — at this size the work is
dispatch-bound, not compute-bound.

### Notes

- `standard_deviation/2` and `covariance/3` joined the doctest `@rounding`
  bucket: both used to host-fall-back and match exactly, and now run natively
  1 ULP away. Excepting a function drops all of its doctests (863 → 851), so
  the bucket is worth watching rather than growing silently.
- Nx.BinaryBackend's `window_scatter_max` round-trips f64 through f32. For f64
  pooling gradients this backend is now *more* accurate than the reference it
  is tested against, so the pooling test asserts values are exact elements of
  the source rather than agreement with the host.

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
