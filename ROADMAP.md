# nx_vulkan roadmap

Moved from README on 2026-07-13 so the main page stays focused on
what already works. Milestones and forward-looking items live here.

Plan history: [`PLAN_GPU_NODE.md`](https://github.com/borodark/nx_vulkan/blob/main/PLAN_GPU_NODE.md) (Phase 1–2 era)
and [`docs/VULKANO_BACKEND_ROADMAP.md`](docs/VULKANO_BACKEND_ROADMAP.md)
(Phase 3+). Per-workstream notes in
[`research/gpu_node/`](https://github.com/borodark/nx_vulkan/tree/main/research/gpu_node).

## Status snapshot

**Fusion compiler shipped** (August 2026): on top of the eager backend
(roadmap stages 1–8) the `Nx.Defn` fusion compiler landed — whole-graph
fusion with a multi-stage split at dot/conv/reduce/transpose boundaries,
f32 and f64. Main branch is stable across Linux + Ampere (RTX 3060 Ti)
and FreeBSD + Kepler (GT 650M, GT 750M): **863 doctests, 361 tests, 0
failures** on the fleet. The vulkano-only architecture (C++ spirit
backend dropped) merged 2026-07-13.

| Feature | Status |
|---|---|
| Vulkano buffer lifecycle | ✓ |
| Native compute op set via specialised SPVs | ✓ |
| Native **f32 and f64** shader paths (elementwise/matmul/conv/reduce/transpose) | ✓ |
| Pipeline cache (correctness + perf) | ✓ |
| Cross-host validation (Linux + 2× FreeBSD) | ✓ |
| Axon training step end-to-end | ✓ |
| eXMC regime log_p (f64) byte-identical | ✓ |
| Autograd via `Nx.Defn.grad` | ✓ |
| Scholar linear regression (coefs match to 2e-6) | ✓ |
| Cross-Kepler bit-determinism (GT 650M ≡ GT 750M) | ✓ |
| Ampere `primary_buffer_count=128` cmd-buffer fix | ✓ |
| Persistent buffer pool | mid-2026 |
| f64 matmul (`matmul_f64.spv`) | ✓ |
| Scholar native linalg shaders (SVD/QR/cholesky/solve) | mid-2026 |
| Polynomial f64 log/exp (behind config) — exmc side | ✓ (default: f32-cast) |
| Custom `Nx.Defn` compiler (whole-graph fusion) | ✓ |
| Native f32 compute (elementwise/matmul/conv/reduce/transpose) | ✓ |
| Conv (im2col + GEMM) / FFT | ✓ |
| sort / scatter | 2026 Q4 |

## Open items

**Op coverage — the long tail.** Convolutions and FFTs now have native
GPU shaders (conv = im2col + GEMM, in f32 and f64; conv is also a fusion
boundary). Still on host-fallback: sort, scatter,
`Nx.LinAlg.solve`/`qr`/`svd`, complex types, sparse ops — they work
today but are slow. Native shaders for each are 50–100 LOC of vulkano
apiece. Estimated effort to reach feature parity with EXLA: 6–12 months
of focused work, parallelisable.

**Custom `Nx.Defn` compiler.** Done — `Nx.Vulkan.Compiler` (thrust 3).
Eager execution still runs through `Nx.Defn.Evaluator` (one op per
dispatch); passing `compiler: Nx.Vulkan.Compiler` to `Nx.Defn.jit`
instead traces the whole graph and compiles it to a stage schedule:
elementwise chains fuse to one generated shader, an elementwise chain
feeding a reduction fuses to one parallel tree-reduce, and graphs with
`dot`/`conv`/`reduce`/`transpose` boundaries split into on-device stages
(`reshape`/`squeeze` are zero-copy views; tuples multi-output). f32 and
f64. Whole dense/CNN layers, classifier heads, softmax, layernorm and
`x @ Wᵀ` fuse with no interpreter fallback. Remaining perf-heuristic
work (cross-stage CSE) was raced and left default-off. See the README's
[fusion compiler section](README.md#the-nxdefn-fusion-compiler-thrust-3).

**Persistent buffer pool.** Currently per-call buffer allocation
through vulkano's `StandardMemoryAllocator`. Works but costs a
millisecond per dispatch that an explicit pool could reclaim.
Mid-2026 work.

**f64 matmul.** Done — `matmul_f64.spv` ships and rank-2 matmul runs
natively in f64. The backend now dtype-dispatches **native f32** as
well (matmul, conv, elementwise, reduce, transpose), with f64 the
default accumulator policy; f32 is no longer merely cast. General
`Nx.dot` axis configs outside rank-2×rank-2 still host-fall-back.

**Scholar — linalg fast paths.** Linear regression (normal equation
+ SVD) now smoke-tests cleanly via a host-fallback `block/4`
callback that routes `Nx.Block.LinAlg.SVD`/`QR`/`solve`/`cholesky`
through `BinaryBackend`. Coefficients match to 2e-6. Native SVD/QR
shaders would speed things up but aren't blocking correctness.
2–4 weeks to add the most-used ones natively.

## Two-backend history (why both live here)

The spirit backend (`Nx.Vulkan.Backend`) reached production first
— chain-shader synthesis, runtime SPV compilation, content-
addressed disk cache, and a long-lived `Nx.Vulkan.Node` GenServer.
Then a use-after-free in the C++ FFI layer crashed the live trader
three minutes after every restart. The failure surfaced as
`Nx.Vulkan.Native.byte_size` raising `:badarg` on a stale `VkBuf*`
pointer — a classic FFI ownership leak the C++ type system cannot
detect.

The vulkano backend (`Nx.Vulkan.VulkanoBackend`) grew from a spike
that proved the migration was mechanical: same SPV bytes in,
byte-identical chain tensors out, perf within ten percent on the
bench target. It replaced spirit for the production path.

The spirit Elixir backend (`Nx.Vulkan.Backend`) and its `Nx.Vulkan.Fuse`
macro were **dropped** (commit `bb94217`) once vulkano covered the
production path; vulkano is now the only Elixir-facing backend. The
`native/nx_vulkan_native` C++ crate directory is vestigial. Full story:
[*The Backend That Didn't Need to Know*](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html).

## Architecture

```
   ┌─────────────────────────────────────────────────────────┐
   │  Nx layer                                                │
   │  • Nx.Vulkan.VulkanoBackend        (eager backend)       │
   │  • Nx.Vulkan.Compiler              (Nx.Defn fusion JIT)  │
   └──────────────────────────┬──────────────────────────────┘
                              │
   ┌──────────────────────────▼──────────────────────────────┐
   │  Nx.Vulkan.NativeV  (Rustler crate nx_vulkan_vulkano)    │
   │  • Arc<Buffer> resources   • pipeline cache              │
   │  • specialisation          • generic dispatch (JIT SPVs) │
   └──────────────────────────┬──────────────────────────────┘
                              ▼
              ┌─────────────────────────┐
              │  Vulkan driver (loader) │
              └─────────────────────────┘
                              │
              ┌───────────────▼─────────────────────────────┐
              │  priv/shaders/*.spv (f32 + f64 variants)     │
              │  • elementwise binary/unary  • reduce_axis   │
              │  • matmul (tiled)  • conv (im2col + GEMM)     │
              │  • transpose  • select / compare / cast      │
              │  • synthesised leapfrog chain shaders        │
              │  priv/shader_cache/gen_*.spv (JIT-generated) │
              └──────────────────────────────────────────────┘
```

The SPV catalog under `priv/shaders/` backs the eager path; the fusion
compiler generates and caches shaders under `priv/shader_cache/`.
The synthesis pipeline that produces new chain shaders at runtime
(`Nx.Vulkan.Synthesis`, `Nx.Vulkan.ShaderTemplate`,
`Nx.Vulkan.ChainShaderSpecs`) lives in the Elixir layer and is
backend-agnostic.

Old spirit-era infrastructure that survives unchanged:

- **`Nx.Vulkan.Node`** — long-lived named GenServer that owns the
  `vkPipelineCache` blob and serialises dispatch via `with_node/2`.
  Used by the legacy backend; the new backend doesn't require it
  but cooperates with it.
- **`Nx.Vulkan.PipelineCache`** — disk-persistent `vkPipelineCache`
  with UUID validation. Survives BEAM restarts.
- **Runtime chain shader synthesis** — render a `FamilySpec`, hand
  to `Synthesis.compile/1`, get a content-addressed SPV path back.
  ~150 ms cold, 5 ms cache hit. Both backends consume the output.
