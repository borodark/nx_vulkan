# Nx.Vulkan

An [Nx](https://github.com/elixir-nx/nx) tensor backend on Vulkan
compute. Wraps [Spirit](https://github.com/borodark/spirit)'s seven
Vulkan shaders (elementwise, unary, reductions, matmul, random,
broadcast) as an Elixir-side `Nx.Backend`.

**Status:** v0.0.1 bootstrap. The plan is in [`PLAN.md`](PLAN.md);
the compute substrate is the Spirit `feature/vulkan-backend` branch
([47 tests + perf characterised on RTX 3060 Ti](https://github.com/borodark/spirit/blob/feature/vulkan-backend/RESULTS_RTX_3060_TI.md)).

## Why this exists

Nx today has two GPU backends:

  - **EXLA** — XLA, requires CUDA or TPU
  - **EMLX** — Apple Metal

On FreeBSD with NVIDIA hardware, neither works. Vulkan is the third
backend, and the only one that runs on FreeBSD. The accompanying
blog post is [*The GPU That Doesn't Need
CUDA*](http://www.dataalienist.com/blog-vulkan-on-freebsd.html).

## Architecture

```
Elixir application                  Rust NIF                  C++ Spirit Vulkan backend
─────────────────                  ────────                  ─────────────────────────
Nx.tensor(...)                      nx_vulkan_alloc    ─►    buf_alloc + upload
defn function                       nx_vulkan_dispatch ─►    dispatch shader
Nx.Defn.jit                         nx_vulkan_download ─►    download

  Nx.Vulkan.Backend  (lib/)         ResourceArc<VkBuf>        VkBuf (lifetime tied
                                    in lib/nx_vulkan/                  to ResourceArc)
                                    native.ex (Rustler)
```

- **`lib/`** — Elixir; `Nx.Backend` impl, top-level API.
- **`native/nx_vulkan_native/`** — Rust NIF crate (Rustler).
- **`c_src/`** — `extern "C"` shim into Spirit's C++ backend
  (`nx_vulkan_shim.{h,cpp}`).

## Layout

```
nx_vulkan/
  PLAN.md                      - milestones to v0.1
  README.md                    - this file
  mix.exs                      - elixir + rustler deps
  lib/
    nx_vulkan.ex               - top-level API
    nx_vulkan/native.ex        - Rustler NIF binding
  native/nx_vulkan_native/
    Cargo.toml
    build.rs                   - compiles Spirit's C++ via cc crate
    src/lib.rs                 - rustler_export_nifs, ResourceArc lifetimes
  c_src/
    nx_vulkan_shim.h           - flat C ABI (extern "C")
    nx_vulkan_shim.cpp         - delegates to spirit::Engine::Backend::vulkan
```

## Building

### Prerequisites

- Erlang/OTP 26+, Elixir 1.17+
- Rust 1.78+ (see toolchain note below)
- C++ compiler (clang or gcc, C++14)
- Vulkan SDK (`libvulkan-dev` on Debian/Ubuntu;
  `pkg install vulkan-headers vulkan-loader` on FreeBSD)
- Spirit checkout at `~/projects/learn_erl/spirit/` with the
  `feature/vulkan-backend` branch — `build.rs` path-deps it.
  Override via `SPIRIT_DIR=/path/to/spirit mix compile`.

### Build

```sh
mix deps.get
mix compile
```

### Toolchain pin (April 2026)

`rust-toolchain.toml` pins rustc to **1.85** because rustler 0.36's
upstream `rustler-sys` macro generation produces a `&usize` where
`usize` is wanted in `enif_term_type` against rustc 1.90's stricter
borrow-checker. 1.85 accepts the older form. Bump the pin once
upstream rustler emits a corrected signature.

## Usage (target — v0.1)

```elixir
iex> Nx.Vulkan.init()
:ok

iex> Nx.Vulkan.device_name()
"NVIDIA GeForce RTX 3060 Ti"

iex> Nx.Vulkan.has_f64?()
true

iex> Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend) |> Nx.exp()
#Nx.Tensor<
  f32[3]
  Nx.Vulkan.Backend
  [2.7182, 7.3890, 20.0855]
>
```

Today (v0.0.1): only `init/0`, `device_name/0`, `has_f64?/0` are
wired. End-to-end smoke verified on RTX 3060 Ti — the Elixir call
reaches Spirit's `vk_init` and reports the device. Tensor
operations land in v0.0.2 once the resource lifetime plumbing is
verified.

## License

Apache 2.0. Same as Spirit and Nx.
