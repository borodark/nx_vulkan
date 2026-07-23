# nx_vulkan roadmap

Moved from README on 2026-07-13 so the main page stays focused on
what already works. Milestones and forward-looking items live here.

Plan history: [`PLAN_GPU_NODE.md`](PLAN_GPU_NODE.md) (Phase 1–2 era)
and [`docs/VULKANO_BACKEND_ROADMAP.md`](docs/VULKANO_BACKEND_ROADMAP.md)
(Phase 3+). Per-workstream notes in
[`research/gpu_node/`](research/gpu_node/).

## Status snapshot

**Phase 3 in progress** (July 2026): the vulkano backend covers
stages 1–8 of [the roadmap](docs/VULKANO_BACKEND_ROADMAP.md). Main
branch is stable across Linux + Ampere (RTX 3060 Ti) and FreeBSD +
Kepler (GT 650M, GT 750M). D90 vulkano-only architecture merged
2026-07-13.

| Feature | Status |
|---|---|
| Vulkano buffer lifecycle | ✓ |
| 24 native compute ops via specialised SPVs | ✓ |
| f64 shader paths (binary/unary/reduce) | ✓ |
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
| Custom `Nx.Defn` compiler | 2026 H2 |
| Conv / FFT / sort / scatter | 2026 H2–Q4 |

## Open items

**Op coverage — the long tail.** Convolutions, FFTs, sort, scatter,
`Nx.LinAlg.solve`/`qr`/`svd`, complex types, sparse ops. Most have
host-fallback paths that work today but are slow. Native shaders
for each are 50–100 LOC of vulkano apiece. Estimated effort to
reach feature parity with EXLA: 6–12 months of focused work,
parallelisable.

**Custom `Nx.Defn` compiler.** Today runs through
`Nx.Defn.Evaluator`, which dispatches ops one at a time. EXLA
compiles whole graphs to optimised HLO. A custom Defn compiler that
batches dispatches, fuses elementwise chains, and caches compiled
graphs would close most of the remaining perf gap. Estimated
effort: 3–6 months.

**Persistent buffer pool.** Currently per-call buffer allocation
through vulkano's `StandardMemoryAllocator`. Works but costs a
millisecond per dispatch that an explicit pool could reclaim.
Mid-2026 work.

**f64 matmul.** Done — `matmul_f64.spv` ships and rank-2 matmul runs
natively in f64 (the backend is f64-only compute now; f32 inputs are
cast). General `Nx.dot` axis configs outside rank-2×rank-2 still
host-fall-back.

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

The two coexist while we backfill the long tail of ops. Long-term,
the spirit path retires. Full story:
[*The Backend That Didn't Need to Know*](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html).

## Architecture

```
   ┌─────────────────────────────────────────────────────────┐
   │  Nx layer                                                │
   │  • Nx.Vulkan.VulkanoBackend  (current)                   │
   │  • Nx.Vulkan.Backend         (legacy, C++ path)          │
   └──────────────┬─────────────────────────┬─────────────────┘
                  │                         │
   ┌──────────────▼──────────┐  ┌──────────▼──────────────────┐
   │  Nx.Vulkan.NativeV       │  │  Nx.Vulkan.Native            │
   │  (Rustler crate          │  │  (Rustler crate              │
   │   nx_vulkan_vulkano)     │  │   nx_vulkan_native)          │
   │  • Arc<Buffer> resources │  │  • C++ shim NIFs             │
   │  • pipeline cache        │  │  • opaque VkBuf* pointers    │
   │  • specialisation        │  │                              │
   └──────────┬───────────────┘  └─────────┬────────────────────┘
              │                            │
              │                       ┌────▼─────────┐
              │                       │  C++ shim    │
              │                       │  (legacy)    │
              │                       └────┬─────────┘
              │                            │
              │                       ┌────▼─────────┐
              │                       │   spirit     │
              │                       │   (vendored) │
              │                       └────┬─────────┘
              │                            │
              └──────────┬─────────────────┘
                         ▼
              ┌─────────────────────────┐
              │  Vulkan driver (loader) │
              └─────────────────────────┘
                         │
              ┌──────────▼──────────────┐
              │  priv/shaders/*.spv      │
              │  • elementwise_binary    │
              │  • elementwise_unary     │
              │  • reduce_axis           │
              │  • matmul                │
              │  • transpose             │
              │  • synthesised chain     │
              │    shaders (Mission II)  │
              │  • 9 hand-written leap-  │
              │    frog families         │
              └──────────────────────────┘
```

The SPV catalog under `priv/shaders/` is shared by both backends.
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
