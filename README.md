# Nx.Vulkan

A GPU tensor backend for [Nx](https://github.com/elixir-nx/nx) that runs on **anything with a Vulkan driver** — including FreeBSD, where CUDA and Metal don't exist.

```
✓ Linux + NVIDIA RTX 3060 Ti      (proprietary driver)
✓ FreeBSD + NVIDIA GT 750M        (NVIDIA legacy driver)
✓ FreeBSD + NVIDIA GT 650M        (NVIDIA legacy driver)
```

Two backends live in this repo:

- **`Nx.Vulkan.VulkanoBackend`** — pure-Rust, [vulkano](https://github.com/vulkano-rs/vulkano)-backed, current primary. 24 ops native + host-fallback for the long tail. Validated on Axon training (forward + autograd + SGD) and on the eXMC regime-model log-posterior at f64 precision.
- **`Nx.Vulkan.Backend`** — C++ spirit-backed, predecessor. Chain-shader synthesis pipeline (`Synthesis`, `ShaderTemplate`, `ChainShaderSpecs`) still ships here and produces SPV blobs that either backend can dispatch.

## What works today

| Capability | VulkanoBackend | spirit (legacy) |
|---|---|---|
| Buffer alloc / upload / download | Arc-managed | raw pointer |
| Elementwise binary (add/sub/mul/div/pow/max/min) | ✓ f32 + f64 | ✓ f32 |
| Elementwise unary (exp/log/sqrt/sigmoid/tanh/abs/neg/floor/ceil/sign) | ✓ f32 + f64 | ✓ f32 |
| Reductions (sum, reduce_max, reduce_min, axis + leading + trailing) | ✓ f32 + f64 | ✓ f32 |
| Shape / movement (reshape, squeeze, transpose-2D) | ✓ | ✓ |
| Matmul (rank-2 × rank-2) | ✓ f32 | ✓ f32 |
| Slice / as_type / general dot axes | host-fallback | host-fallback |
| Chain shader synthesis (Mission II) | dispatches generated SPV | dispatches generated SPV |
| `Nx.Defn.grad` (autograd) | works automatically | works automatically |
| Axon training step | **✓ validated** (1e-8 grad match vs BinaryBackend) | partial |
| eXMC NUTS sampler integration | **✓ regime log_p byte-identical** | ✓ chain-shader path |
| Long-running workloads (5000+ dispatches) | **✓ pipeline cache** | ✗ stale-buffer crash class |

The chain-shader dispatch path (`leapfrog_chain_synth`) is shared between both backends — same SPV cache, same content-addressing, same runtime synthesis pipeline. The difference is the backend that handles general Nx tensor ops outside that dispatch.

## Why two backends

The spirit backend reached production first — chain-shader synthesis, runtime SPV compilation, content-addressed disk cache, and a long-lived `Nx.Vulkan.Node` GenServer. Then a use-after-free in the C++ FFI layer crashed the live trader three minutes after every restart. The failure surfaced as `Nx.Vulkan.Native.byte_size` raising `:badarg` on a stale `VkBuf*` pointer — a classic FFI ownership leak the C++ type system cannot detect. The vulkano backend grew from a spike that proved the migration was mechanical: same SPV bytes in, byte-identical chain tensors out, perf within ten percent on the bench target.

The two coexist while we backfill the long tail of ops. Long-term, the spirit path retires.

See [`docs/VULKANO_BACKEND_ROADMAP.md`](docs/VULKANO_BACKEND_ROADMAP.md) for the full stage breakdown. The full story is in [*The Backend That Didn't Need to Know*](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html).

## Benchmarks (May 2026)

Square matmul, milliseconds per dispatch, median of 50–200 iterations:

| size | bin (super-io) | bin (mac-247) | vulkano (super-io) | vulkano (mac-247) | spirit (mac-247) |
|---|---|---|---|---|---|
| 16×16 | 2.76 | 2.51 | 1.18 | **1.06** | 1.16 |
| 64×64 | 130.76 | 158.45 | 7.07 | 7.92 | **7.56** |
| 256×256 | 20,097 | 13,891 | 149.19 | **136.10** | 141.73 |
| 1024×1024 | n/a (hours) | n/a (hours) | 2,323 | 2,843 | 2,845 |

Two observations:

1. **vulkano and spirit agree within 5% on every matmul size where both run** on the same hardware. The C++ path doesn't buy back its maintenance cost.
2. **The Vulkan path beats BinaryBackend by 92–135× at 256×256** on the GT 650M. The GPU is from 2013; what changes is moving the loop off the BEAM scheduler.

The C++ spirit path crashed on Linux super-io mid-bench with a memory-supervisor high-watermark warning — same fragility class that motivated the migration in the first place. The vulkano path completed cleanly on both hosts. Full bench script: [`examples/full_bench.exs`](examples/full_bench.exs).

## Position vs EXLA and EMLX

Three GPU backends exist for Nx today. Each won a different platform first.

| | EXLA | EMLX | Nx.Vulkan.VulkanoBackend |
|---|---|---|---|
| **Backing API** | Google XLA | Apple MLX (Metal) | Khronos Vulkan via vulkano (Rust) |
| **Maturity** | Years; production | Released 2024 | Released 2026 |
| **Linux + NVIDIA CUDA** | ✓ canonical | ✗ | ✓ via Vulkan |
| **macOS + Apple Silicon** | ✗ | ✓ canonical | ✓ via MoltenVK |
| **FreeBSD + NVIDIA** | ✗ | ✗ | **✓ only path** |
| **Windows / WSL2** | partial via TF | ✗ | ✓ (Vulkan ships on Windows) |
| **Op coverage** | full Nx surface (~200) | full Nx surface | 24 native, rest via host fallback |
| **`Nx.Defn.grad` (autograd)** | full | full | **✓ free** (graph transformation) |
| **fp64 compute** | full | none (Metal limit) | ✓ binary/unary/reduce |
| **Production use** | Google scale | Apple devices | eXMC trader on mac-247 |

### The autograd insight

`Nx.Defn.grad` is a graph transformation that runs at compile time on the `Nx.Defn.Expr` AST. For every forward op in the graph, it inserts the corresponding backward op expressed in terms of *more forward ops*. The backend never sees a "backward op" — it just keeps executing forward primitives. Forward op coverage IS gradient coverage when running through `Nx.Defn.Evaluator`.

That means **VulkanoBackend supports gradients for any function expressible in its 24 native ops + host-fallback long tail**. No backward callbacks were written. Validated by running a complete Axon training step (Dense → sigmoid → Dense → MSE → `Nx.Defn.value_and_grad`) on `Nx.Vulkan.VulkanoBackend`, with gradient sum agreeing to 1e-8 against the `BinaryBackend` reference.

### What's missing

**Op coverage — the long tail.** Convolutions, FFTs, sort, scatter, `Nx.LinAlg.solve`/`qr`/`svd`, complex types, sparse ops. Most of these have host-fallback paths that work today but are slow. Native shaders for each are 50–100 LOC of vulkano apiece. Estimated effort to reach feature parity with EXLA: 6–12 months of focused work, parallelisable.

**`Nx.Defn` custom compiler.** Today we run through `Nx.Defn.Evaluator`, which dispatches ops one at a time. EXLA compiles whole graphs to optimised HLO. A custom Defn compiler that batches dispatches, fuses elementwise chains, and caches compiled graphs would close most of the remaining perf gap. Estimated effort: 3–6 months.

**Persistent buffer pool.** Currently per-call buffer allocation through vulkano's `StandardMemoryAllocator`. Works but costs a millisecond per dispatch that an explicit pool could reclaim. **Mid-2026 work.**

**f64 matmul.** `matmul.spv` is f32-only. f64 dot products fall back to host, which is slow for large tensors. **2 weeks** to add the f64 variant.

**Scholar parity.** Not yet smoke-tested. Likely needs a few more callback patterns (general slice, `Nx.LinAlg.solve` host-route already works for normal-equation linear regression). **One day** to validate.

## Quickstart

### As a backend in your project

```elixir
# mix.exs
def deps do
  [
    {:nx, "~> 0.10"},
    {:nx_vulkan, git: "https://github.com/borodark/nx_vulkan"}
  ]
end
```

```elixir
# Build a tensor, transfer to GPU, do work
x_bin = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
x_vk  = Nx.backend_transfer(x_bin, Nx.Vulkan.VulkanoBackend)

y_vk  = Nx.sigmoid(x_vk)
y_bin = Nx.backend_transfer(y_vk, Nx.BinaryBackend)
IO.inspect(Nx.to_list(y_bin))
# [0.7310585975646973, 0.8807970881462097, 0.9525741338729858, 0.9820137619972229]
```

### Try the Axon training example

```sh
git clone https://github.com/borodark/nx_vulkan
cd nx_vulkan
mix deps.get && mix compile
elixir examples/axon_training_loop.exs
```

Runs a 100-step Dense(4→32, tanh)→Dense(1) regression with manual SGD. Compares loss trajectories on `BinaryBackend` vs `VulkanoBackend`. PASS verdict on both Linux + FreeBSD.

### Try the full bench

```sh
mix run examples/full_bench.exs
```

Per-op + end-to-end + robustness across every backend Nx can find. Auto-detects EXLA availability. Runs in ~10 minutes on RTX 3060 Ti, ~15 on GT 650M.

## Why FreeBSD matters

Nx today has three GPU backends. Two of them — EXLA and EMLX — explicitly do not run on FreeBSD. If you have NVIDIA hardware on FreeBSD, Vulkan is the only path. **mac-248** (FreeBSD 15.0 / GT 750M) and **mac-247** (FreeBSD 15.0 / GT 650M Mac Edition) are the canonical bring-up boxes; every commit gets verified there alongside the Linux dev host.

The companion blog series:

- [*The Backend That Didn't Need to Know*](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html) — the C++→vulkano migration; descriptor pool debugging; autograd was free
- [*The GPU That Doesn't Need CUDA*](http://www.dataalienist.com/blog-vulkan-on-freebsd.html) — the FreeBSD Vulkan story (spirit-era)
- [*A Walkable Path Under the Mountain*](http://www.dataalienist.com/blog-walkable-path.html) — eXMC + zed integration

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

The SPV catalog under `priv/shaders/` is shared by both backends. The synthesis pipeline that produces new chain shaders at runtime
(`Nx.Vulkan.Synthesis`, `Nx.Vulkan.ShaderTemplate`,
`Nx.Vulkan.ChainShaderSpecs`) lives in the Elixir layer and is
backend-agnostic.

Old spirit-era infrastructure that survives unchanged:

- **`Nx.Vulkan.Node`** — long-lived named GenServer that owns the `vkPipelineCache` blob and serialises dispatch via `with_node/2`. Used by the legacy backend; new backend doesn't require it but cooperates with it.
- **`Nx.Vulkan.PipelineCache`** — disk-persistent `vkPipelineCache` with UUID validation. Survives BEAM restarts.
- **Runtime chain shader synthesis** — render a `FamilySpec`, hand to `Synthesis.compile/1`, get a content-addressed SPV path back. ~150 ms cold, 5 ms cache hit. Both backends consume the output.

## Building

### Prerequisites

- Erlang/OTP 26+, Elixir 1.17+
- Rust 1.78+
- C++ compiler (only needed for the legacy spirit backend; vulkano is pure Rust)
- Vulkan SDK + `glslangValidator`:
  - Debian/Ubuntu: `apt install libvulkan-dev vulkan-tools glslang-tools`
  - FreeBSD: `pkg install vulkan-loader vulkan-headers vulkan-tools glslang shaderc`

### Build

```sh
mix deps.get
mix compile
```

Vulkano compiles in ~30s on Linux, ~3:18 on FreeBSD 15.0 (mostly dependency compilation). The spirit/C++ path compiles in parallel.

### Rust toolchain pin

`rust-toolchain.toml` pins rustc to 1.85. The reason is in the file's comment; bump when upstream rustler emits a corrected `rustler-sys` signature.

## Status

**Phase 3 in progress** (May 2026): vulkano backend covers stages 1–8 of [the roadmap](docs/VULKANO_BACKEND_ROADMAP.md).

| Feature | Status |
|---|---|
| Vulkano buffer lifecycle (alloc/upload/download/free) | ✓ |
| 24 native compute ops via specialised SPVs | ✓ |
| f64 shader paths (binary/unary/reduce) | ✓ |
| Pipeline cache (correctness + perf) | ✓ |
| Cross-host validation (Linux + 2× FreeBSD) | ✓ |
| Axon training step end-to-end | ✓ |
| eXMC regime log_p (f64) byte-identical | ✓ |
| Autograd via `Nx.Defn.grad` | ✓ |
| Persistent buffer pool | mid-2026 |
| f64 matmul | mid-2026 |
| Scholar smoke test | mid-2026 |
| Custom `Nx.Defn` compiler | 2026 H2 |
| Conv / FFT / sort / scatter | 2026 H2–Q4 |

Plan history is in [`PLAN_GPU_NODE.md`](PLAN_GPU_NODE.md) (Phase 1–2 era) and [`docs/VULKANO_BACKEND_ROADMAP.md`](docs/VULKANO_BACKEND_ROADMAP.md) (Phase 3+). Per-workstream notes in [`research/gpu_node/`](research/gpu_node/).

## Sibling: zed

[`zed`](../zed/) is the declarative ZFS + Elixir deploy tool that orchestrates BEAM nodes. `nx_vulkan` is consumed *inside* deployed BEAM nodes — not as a zed dependency. See `specs/nx-vulkan-execution.md` in the zed repo for the integration story.

## License

Apache 2.0. Same as Spirit and Nx.
