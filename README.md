# Nx.Vulkan

A GPU tensor backend for [Nx](https://github.com/elixir-nx/nx) that runs on **anything with a Vulkan driver** — including FreeBSD, where CUDA and Metal don't exist.

```
✓ Linux + NVIDIA RTX 3060 Ti      (proprietary driver)
✓ FreeBSD + NVIDIA GT 750M        (NVIDIA legacy driver)
✓ FreeBSD + NVIDIA GT 650M        (NVIDIA legacy driver)
```

## Goals

- **Cover the FreeBSD gap.** Nx's existing GPU backends (EXLA on
  Linux+CUDA, EMLX on macOS+Metal) don't run on FreeBSD. If you
  have NVIDIA on FreeBSD, this is the only path.
- **Portable Vulkan across vendors.** Same SPV blobs run on NVIDIA,
  AMD RADV, Intel, and via MoltenVK on macOS. The Rust vulkano crate
  keeps the driver surface pure Rust — no C++ FFI ownership traps.
- **Enough coverage that Nx models Just Work.** 24 native ops plus a
  host-fallback long tail let real workloads (Axon training, eXMC
  NUTS sampling, Scholar linear regression) run today without
  op-by-op porting.
- **Deterministic across hardware.** Cross-Kepler runs of the same
  IR produce byte-identical posterior chains. Reproducible science
  across GPU generations is a documented feature.

## What works today

- **24 native compute ops** — elementwise binary/unary, reductions
  (sum / max / min along any axis, leading, or trailing),
  reshape / squeeze / transpose-2D, matmul. Compute is **f64**; f32
  inputs are accepted and cast to f64.
- **Host fallback for the long tail** — slice, `as_type`, general
  `Nx.dot` axes, `Nx.LinAlg.SVD`/`QR`/`solve`/`cholesky` (via
  `Nx.Block.LinAlg`). Slow but correct.
- **`Nx.Defn.grad` autograd**, for free — see [The autograd insight](#the-autograd-insight).
- **Axon training step** end-to-end, gradient sum agrees to 1e-8
  vs `BinaryBackend` reference.
- **eXMC NUTS regime log_p** at f64: byte-identical to CPU
  reference.
- **Scholar linear regression**: coefficients match `BinaryBackend`
  to 2e-6 (SVD via host fallback).
- **Long-running stability** — 7000+ chain-shader dispatches on
  Ampere without crash after the 2026-07 `primary_buffer_count=128`
  fix; unbounded on Kepler.
- **Pipeline cache** on disk, UUID-validated, survives BEAM restarts.

Roadmap and future work: [`ROADMAP.md`](ROADMAP.md).

## Position vs EXLA and EMLX

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
| **fp64 compute** | full | none (Metal limit) | ✓ f64-only compute (binary/unary/reduce/matmul) |

### The autograd insight

`Nx.Defn.grad` is a graph transformation that runs at compile time
on the `Nx.Defn.Expr` AST. For every forward op in the graph, it
inserts the corresponding backward op expressed in terms of *more
forward ops*. The backend never sees a "backward op" — it just keeps
executing forward primitives. Forward op coverage IS gradient
coverage when running through `Nx.Defn.Evaluator`.

That means **VulkanoBackend supports gradients for any function
expressible in its 24 native ops + host-fallback long tail**. No
backward callbacks were written. Validated by running a complete
Axon training step (Dense → sigmoid → Dense → MSE →
`Nx.Defn.value_and_grad`) on `Nx.Vulkan.VulkanoBackend`, with
gradient sum agreeing to 1e-8 against the `BinaryBackend` reference.

## Benchmarks (May 2026)

Square matmul, milliseconds per dispatch, median of 50–200 iterations:

| size | bin (super-io) | bin (mac-247) | vulkano (super-io) | vulkano (mac-247) |
|---|---|---|---|---|
| 16×16 | 2.76 | 2.51 | 1.18 | **1.06** |
| 64×64 | 130.76 | 158.45 | 7.07 | 7.92 |
| 256×256 | 20,097 | 13,891 | 149.19 | **136.10** |
| 1024×1024 | n/a (hours) | n/a (hours) | 2,323 | 2,843 |

The Vulkan path beats `BinaryBackend` by 92–135× at 256×256 on the
GT 650M. The GPU is from 2013; the win is moving the loop off the
BEAM scheduler. Full bench: [`examples/full_bench.exs`](examples/full_bench.exs).

## Quickstart

### As a backend in your project

```elixir
# mix.exs
def deps do
  [
    {:nx, "~> 0.13"},
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

Runs a 100-step Dense(4→32, tanh)→Dense(1) regression with manual
SGD. Compares loss trajectories on `BinaryBackend` vs
`VulkanoBackend`. PASS verdict on both Linux + FreeBSD.

### Try the full bench

```sh
mix run examples/full_bench.exs
```

Per-op + end-to-end + robustness across every backend Nx can find.
Auto-detects EXLA availability. Runs in ~10 minutes on RTX 3060 Ti,
~15 on GT 650M.

## Building

### Prerequisites

- Erlang/OTP 26+, Elixir 1.17+
- Rust 1.78+
- C++ compiler (only needed for the legacy spirit backend; vulkano
  is pure Rust)
- Vulkan SDK + `glslangValidator`:
  - Debian/Ubuntu: `apt install libvulkan-dev vulkan-tools glslang-tools`
  - FreeBSD: `pkg install vulkan-loader vulkan-headers vulkan-tools glslang shaderc`

### Build

```sh
mix deps.get
mix compile
```

Vulkano compiles in ~30s on Linux, ~3:18 on FreeBSD 15.0 (mostly
dependency compilation). The spirit/C++ path compiles in parallel.

### Rust toolchain pin

`rust-toolchain.toml` pins rustc to 1.85. The reason is in the
file's comment; bump when upstream rustler emits a corrected
`rustler-sys` signature.

## Blog series

- [*The Backend That Didn't Need to Know*](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html) — the C++→vulkano migration; descriptor pool debugging; autograd was free
- [*The GPU That Doesn't Need CUDA*](http://www.dataalienist.com/blog-vulkan-on-freebsd.html) — the FreeBSD Vulkan story (spirit-era)
- [*A Walkable Path Under the Mountain*](http://www.dataalienist.com/blog-walkable-path.html) — eXMC + zed integration

## Sibling: zed

[`zed`](../zed/) is the declarative ZFS + Elixir deploy tool that
orchestrates BEAM nodes. `nx_vulkan` is consumed *inside* deployed
BEAM nodes — not as a zed dependency. See `specs/nx-vulkan-execution.md`
in the zed repo for the integration story.

## License

Apache 2.0. Same as Spirit and Nx.
