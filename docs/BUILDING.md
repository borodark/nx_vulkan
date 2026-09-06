# Building and running Nx.Vulkan

**Scope:** dependencies, prerequisites, the examples worth running first, and
the toolchain pin.

## Quickstart

### As a backend in your project

```elixir
# mix.exs
def deps do
  [
    {:nx, "~> 0.13"},
    {:nx_vulkan, "~> 0.4"}
  ]
end
```

Or track `main` directly:

```elixir
{:nx_vulkan, git: "https://github.com/borodark/nx_vulkan"}
```

> On 0.2.0 and training on the GPU? Upgrade. 0.2.0 computed correct results,
> but its backward pass ran largely on the host — a LeNet step took 20.9 s
> there and 84 ms here. See the [CHANGELOG](../CHANGELOG.md).

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
