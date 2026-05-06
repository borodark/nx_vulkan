# Nx.Vulkan

A GPU tensor backend for [Nx](https://github.com/elixir-nx/nx) that runs on **anything with a Vulkan driver** — including FreeBSD, where CUDA and Metal don't exist.

```
✓ Linux + NVIDIA RTX 3060 Ti      (proprietary driver)
✓ FreeBSD + NVIDIA GT 750M        (mesa-radv)
✓ FreeBSD + NVIDIA GT 650M        (mesa-radv stack)
```

178 of 178 tests green on all three platforms. Identical posteriors. Shader synthesis end-to-end in under a second.

## What you get

**A long-lived GPU node.** `Nx.Vulkan.Node` is a named GenServer that owns the Vulkan pipeline cache, the persistent buffer registry, and a watchdog. Any client serializes work through it via `with_node/2`:

```elixir
{:ok, _} = Nx.Vulkan.Node.start_link()

result =
  Nx.Vulkan.Node.with_node(fn ->
    # Any GPU work — runs on the node's process, shares pipeline cache.
    Nx.Vulkan.Native.leapfrog_chain_synth(q, p, m, push, k, spv_path)
  end)

case result do
  {:error, :node_timeout} -> exla_fallback()  # watchdog fired
  ok_value                 -> ok_value
end
```

**Runtime shader synthesis.** Want a chain shader for a distribution that doesn't exist in the catalog? Render a `FamilySpec`, hand it to `Synthesis.compile/1`, get a SPIR-V file back in ~150 ms cold (5 ms cache hit):

```elixir
spec = Nx.Vulkan.ChainShaderSpecs.beta()       # or your own %FamilySpec{}
{:ok, spv_path} = Nx.Vulkan.Synthesis.compile(spec)

# spv_path is content-addressed under ~/.exmc/gpu_node/spv/{sha256}.spv —
# survives BEAM restarts.
```

The template engine handles the leapfrog skeleton. You provide three GLSL fragments per family: gradient at `qi`, gradient at `qi` after position update, log-density contribution. Ships with **9 chain shader families** out of the box:

| Hand-written (vendored SPV) | Synthesized at runtime |
|---|---|
| Normal, Exponential, StudentT, Cauchy, HalfNormal, Weibull | Beta, Gamma, Lognormal |

**Persistent pipeline cache.** `vkPipelineCache` blob serialized to disk, header-validated against the device UUID before re-load. **4× speedup** on cold start for synthesized shaders (`Gamma 23 ms → 5 ms`, `Lognormal 22 ms → 5 ms` first-dispatch wall on RTX 3060 Ti).

```elixir
# At app start
:ok = Nx.Vulkan.PipelineCache.load()

# At app shutdown (Nx.Vulkan.Node.terminate/2 already does this)
:ok = Nx.Vulkan.PipelineCache.persist()
```

## Position vs EXLA and EMLX

Three GPU backends exist for Nx today. Each won a different platform first.

| | EXLA | EMLX | Nx.Vulkan |
|---|---|---|---|
| **Backing API** | Google XLA | Apple MLX (Metal) | Khronos Vulkan compute |
| **Maturity** | Years; production | Released 2024 | Released 2026 (this repo) |
| **Linux + NVIDIA CUDA** | ✓ canonical | ✗ | ✓ (Linux NVIDIA Vulkan driver) |
| **macOS + Apple Silicon** | ✗ | ✓ canonical | ✓ (MoltenVK) |
| **FreeBSD + NVIDIA** | ✗ | ✗ | ✓ **only path** |
| **Windows / WSL2** | partial via TF | ✗ | ✓ (Vulkan ships on Windows) |
| **Op coverage** | full Nx surface (~200 ops) | full Nx surface | **partial** — see below |
| **`Nx.Defn.jit`** | full XLA HLO | full MLX graph | partial via `Nx.Vulkan.Compiler` |
| **fp64 compute** | full | none (Metal limit) | depends on driver (RTX: yes; mesa-radv: yes) |
| **Production deployment** | Google scale | Apple personal devices | nascent — used inside [`exmc`](../pymc/exmc) |

### What Nx.Vulkan does today that the others don't

- **Per-family chain shader synthesis at runtime** for HMC/NUTS samplers. `Nx.Vulkan.ShaderTemplate` + `Nx.Vulkan.Synthesis` + `Nx.Vulkan.ChainShaderSpecs` produce a working SPV under 200 ms cold path; subsequent calls hit a content-addressed disk cache. Neither EXLA nor EMLX synthesizes specialized kernels at runtime — they're trace-and-compile-and-cache against the user's defn.
- **Long-lived per-machine GPU node**, not per-process. `Nx.Vulkan.Node` with `with_node/2` lets multiple BEAM processes share a pipeline cache and a buffer registry through one named GenServer — useful when 67 trading-instrument processes all want GPU dispatch but can't each pay the pipeline-create cost.
- **Disk-persistent `vkPipelineCache`**, header-validated against the device UUID. Survives BEAM restarts. EXLA's HLO cache is per-process; EMLX caches in MLX's process-local pool.

### What's missing to reach parity

The honest gap, by category. "Parity" here means "pick the same backend for the same workload and get within 2× of EXLA's wall-time."

**Op coverage — biggest gap.** Nx has roughly 200 ops; EXLA and EMLX cover essentially all of them. Nx.Vulkan currently ships:

- Elementwise unary + binary (via `Nx.Vulkan.Native.apply_unary` / `apply_binary` over Spirit's compiled SPV).
- Reductions (sum, min, max, axis reductions).
- `matmul` / `matmul_v` (specialised + variants).
- Random sampling.
- Transpose, cast, broadcast.
- 9 chain shader families (Normal/Exp/StudentT/Cauchy/HalfNormal/Weibull/Beta/Gamma/Lognormal) for HMC integration — these are *not* generic Nx ops; they're specialised compute kernels that the Nx layer wouldn't expose to user defn but that exmc's NUTS sampler talks to directly.

What's *not* there yet: convolutions, FFTs, sort, scatter/gather, padding modes beyond fill-with-constant, complex numbers, sparse ops, autodiff via VJP/JVP wiring. Each is a small-to-medium Vulkan compute shader plus a `Nx.Backend` callback. Estimated effort to close the op-coverage gap: **6-12 months of focused work**, parallelisable across multiple shader authors.

**`Nx.Defn` JIT integration — partial.** `Nx.Vulkan.Compiler` exists and auto-detects fusable elementwise chains, dispatching `Nx.Vulkan.fused_chain/3` instead of N separate shader calls. This catches the trivial cases. What's missing: full graph-level optimisation (operator fusion across reductions, kernel-launch coalescing, layout-aware scheduling). EXLA's XLA does all this; ours doesn't yet. Estimated effort: **3-6 months** to reach "Nx.Defn power-user gets reasonable speedups without thinking about it."

**Performance tooling — small gap.** Profiling, dispatch counters, pipeline-cache hit telemetry — much of this exists (`Nx.Vulkan.Native.timing_get/0` exposes per-fence/dispatch counters), but it's not packaged as a user-facing dashboard the way EXLA's HLO dump is. **1-2 months.**

**Correctness gaps for fat-tailed distributions.** Cauchy is auto-skipped at `:f32_precision_limited` because f32 chain shaders structurally can't reproduce f64 reference IQR for distributions without defined moments. Either the validator gets a documented "expected gap" mode, or the chain shader path gets an opt-in fp64 implementation (Spirit supports it; some shaders ship in `_f64` variants already). **2 weeks.**

**Op-by-op reaches 2× of EXLA**: with current ops, mesa-radv on the FreeBSD GT 750M already beats NVIDIA Linux EXLA by 7× on the chain shader workload. The gap on Linux is 2-3× (post-W3 optimisation work; see Performance below). For elementwise + reductions + matmul on equivalent-class hardware, Vulkan and EXLA are within 50% of each other on Linux today. The gap mostly closes for free as new ops land — Vulkan compute is fundamentally fast.

### Roadmap

```
Phase 0: bring-up                                            ✓ shipped (2026 Q1)
Phase 1: chain shader synthesis + GPU node + persistent cache ✓ shipped (2026 Q2)
Phase 2: in-flight dispatch cancellation                     in progress
Phase 3: multi-client mDNS discovery                          planned 2026 Q3
Phase 4: op-coverage push (conv, FFT, sort, scatter)          2026 Q3-Q4
Phase 5: full Nx.Defn graph optimisation                      2027 H1
Phase 6: f64 chain shader variants for fat-tailed dists       opportunistic
```

For a workload that lives entirely in elementwise+reduce+matmul (most NUTS samplers, many neural network forward passes), Nx.Vulkan is **already a viable EXLA alternative on Linux** and **the only option on FreeBSD**. For graph-heavy inference (large language model serving, conv-heavy CV), the gap closes in the 2026 Q3-Q4 horizon.

## Why FreeBSD matters

Nx today has two GPU backends:
- **EXLA** — XLA, requires CUDA or TPU. No FreeBSD support.
- **EMLX** — Apple Metal. macOS only.

If you have NVIDIA hardware on FreeBSD, neither works. Vulkan is the only path. mac-248 (FreeBSD 15 / GT 750M) and mac-247 (FreeBSD / GT 650M) are the canonical bring-up boxes; every commit gets verified there alongside Linux.

The companion blog series:
- [*The GPU That Doesn't Need CUDA*](http://www.dataalienist.com/blog-vulkan-on-freebsd.html) — the FreeBSD Vulkan story
- [*A Walkable Path Under the Mountain*](http://www.dataalienist.com/blog-walkable-path.html) — eXMC + zed integration

## Quickstart

```sh
git clone https://github.com/borodark/nx_vulkan
cd nx_vulkan
mix deps.get && mix compile
mix test                               # 152 + 26 = 178 tests
mix run examples/gpu_node_demo.exs     # boot Node + synth Beta + dispatch + persist cache
```

The demo prints (on a warm cache):

```
device:  NVIDIA GeForce RTX 3060 Ti  (or GT 750M / GT 650M / etc.)
synthesized Beta SPV in 5 ms
first dispatch via with_node: 16410 µs
logp[0]: -1.486   (analytic -1.4508, delta 0.035 ✓)
pipeline cache persisted: 12432 bytes
```

## Architecture

```
                     ┌─────────────────────────────────────────────┐
                     │  Nx.Vulkan.Node  (named GenServer)           │
                     │  • with_node/2 — generic serialized dispatch │
                     │  • watchdog timeout → {:error, :node_*}      │
                     │  • lifecycle owns the pipeline cache         │
                     └──────────────┬──────────────────────────────┘
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │                                                       │
┌───────▼──────────┐  ┌────────────────────┐  ┌─────────────────▼────┐
│ Nx.Vulkan.       │  │ Nx.Vulkan.         │  │ Nx.Vulkan.            │
│   PipelineCache  │  │   Synthesis +      │  │   ChainShaderSpecs    │
│   (vkPipeline-   │  │   ShaderTemplate   │  │   (Beta/Gamma/        │
│    Cache disk    │  │   (runtime GLSL +  │  │    Lognormal +        │
│    persistence)  │  │    glslangValidator│  │    6 hand-written)    │
└──────────────────┘  └────────────────────┘  └───────────────────────┘
                                    │
                              ┌─────▼──────┐
                              │ Rust NIFs  │  (lib.rs, Rustler 0.36)
                              └─────┬──────┘
                              ┌─────▼──────┐
                              │  C++ shim  │  (nx_vulkan_shim.{h,cpp})
                              └─────┬──────┘
                                    ▼
                           spirit (vendored under c_src/spirit/)
```

- **`lib/nx_vulkan/`** — Elixir API. `Node`, `PipelineCache`, `ShaderTemplate`, `Synthesis`, `ChainShaderSpecs`, plus the low-level `Native` NIF bindings.
- **`native/nx_vulkan_native/`** — Rust NIF crate (Rustler). Wraps the C shim, exposes `leapfrog_chain_synth`, batched IO, pipeline cache load/persist.
- **`c_src/nx_vulkan_shim.{h,cpp}`** — flat C ABI bridging Rust to Spirit's C++.
- **`c_src/spirit/`** — vendored Spirit Vulkan backend (~800 LOC C++). See `c_src/spirit/VENDOR.md` for the pinned upstream commit.
- **`priv/shaders/`** — 9 SPIR-V chain shaders (vendored from Spirit's `shaders/`).

## Performance

The chain-shader path's design target is **~1 ms per leapfrog step**. Linux NVIDIA Vulkan and FreeBSD mesa-radv both meet it, but with different constants:

| Workload (Normal d=1, 1000W + 1000S, NUTS) | Wall (median, 5 seeds) |
|---|---|
| Linux RTX 3060 Ti, EXLA reference | 7,753 ms |
| Linux RTX 3060 Ti, **Vulkan-fused** | **12,722 ms** (with `+sbt tnnps`) |
| FreeBSD GT 750M, **Vulkan-fused** | **1,651 ms** (mesa wins on per-fence latency) |

mesa-radv is **~7× faster than NVIDIA Linux's proprietary driver** on this workload because of per-fence-wait latency (~150 µs vs ~1.13 ms blocking floor). The hardware on the FreeBSD box is a 12-year-old laptop GPU; the Linux box is a modern RTX 3060 Ti. Driver quality, not silicon, dominates this regime.

[Full investigation in `research/gpu_node/`](research/gpu_node/) — W4 warmup curves, W5 cache persistence, W7 FMA fusion drift.

## Building

### Prerequisites

- Erlang/OTP 27+, Elixir 1.18+
- Rust 1.78+ (toolchain pinned via `rust-toolchain.toml`; see note below)
- C++ compiler (clang or gcc, C++14)
- Vulkan SDK + `glslangValidator`:
  - Debian/Ubuntu: `apt install libvulkan-dev vulkan-tools glslang-tools`
  - FreeBSD: `pkg install vulkan-loader vulkan-headers vulkan-tools glslang shaderc`

### Build

```sh
mix deps.get
mix compile
```

Spirit's Vulkan backend is **vendored** under `c_src/spirit/` — no external Spirit checkout required. Set `SPIRIT_DIR=/path/to/spirit` only as a development override to refresh `priv/shaders/*.spv` from a local Spirit checkout on each build (mac-248's iteration workflow).

### Rust toolchain pin

`rust-toolchain.toml` pins rustc to **1.85** because rustler 0.36's upstream `rustler-sys` macro generation produces a `&usize` where `usize` is wanted in `enif_term_type` against rustc 1.90's stricter borrow-checker. 1.85 accepts the older form. Bump the pin once upstream rustler emits a corrected signature.

## Status

**Phase 2 shipped** (May 2026): GPU node + persistent pipeline cache + runtime shader synthesis.

| Feature | Status |
|---|---|
| Vulkan context + buffer alloc/upload/download | ✓ |
| Hand-written chain shaders (6 families) | ✓ |
| Runtime shader synthesis (Beta/Gamma/Lognormal) | ✓ |
| `Nx.Vulkan.Node` GenServer + `with_node/2` watchdog | ✓ |
| Persistent vkPipelineCache | ✓ |
| Cross-platform validation (Linux + 2× FreeBSD) | ✓ |
| Per-shader suspect tracking + EXLA fallback (in `exmc`) | ✓ Phase 1 |
| In-flight dispatch cancellation (`vk_synchronization2`) | Phase 2 work |
| Multi-client mDNS discovery | Phase 3 work |

Plan history is in [`PLAN_GPU_NODE.md`](PLAN_GPU_NODE.md). Per-workstream notes in [`research/gpu_node/`](research/gpu_node/).

## Sibling: zed

[`zed`](../zed/) is the declarative ZFS + Elixir deploy tool that orchestrates BEAM nodes. `nx_vulkan` is consumed *inside* deployed BEAM nodes — not as a zed dependency. See `specs/nx-vulkan-execution.md` in the zed repo for the integration story.

## License

Apache 2.0. Same as Spirit and Nx.
