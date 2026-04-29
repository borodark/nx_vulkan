defmodule Nx.Vulkan do
  @moduledoc """
  Nx tensor backend on Vulkan compute.

  Wraps Spirit's Vulkan compute kernels (elementwise, reductions,
  matmul, random) as an `Nx.Backend`. Works on FreeBSD with NVIDIA
  hardware where CUDA does not. Same backend code runs on Linux,
  macOS (via MoltenVK), and any Vulkan-capable GPU.

  ## Status

  v0.0.1 — bootstrap. The plan in `PLAN.md` lays out the 10-milestone
  path to v0.1. This release just initializes Vulkan and reports
  which physical device was selected; tensor types and operators
  land in subsequent commits.

  ## Usage (target)

      iex> Nx.Vulkan.init()
      :ok

      iex> Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
      ...

      iex> Nx.Defn.default_options(default_backend: Nx.Vulkan.Backend)

  ## Files

    * `lib/nx_vulkan.ex`           - this module (top-level API)
    * `lib/nx_vulkan/native.ex`    - Rustler NIF binding (skeleton)
    * `lib/nx_vulkan/backend.ex`   - `Nx.Backend` impl (TBD)
    * `native/nx_vulkan_native/`   - Rust NIF crate
    * `c_src/`                     - extern "C" shim into spirit's
                                     C++ Vulkan backend
  """

  @doc """
  Initialize the Vulkan compute context. Call once at startup.
  Returns `:ok` on success, `{:error, reason}` if no Vulkan-capable
  device is found.
  """
  defdelegate init(), to: Nx.Vulkan.Native

  @doc """
  Returns the name of the selected physical device, or `nil` if
  `init/0` has not been called.
  """
  defdelegate device_name(), to: Nx.Vulkan.Native

  @doc "Returns true if the selected device supports f64 (double precision)."
  defdelegate has_f64?(), to: Nx.Vulkan.Native, as: :has_f64
end
