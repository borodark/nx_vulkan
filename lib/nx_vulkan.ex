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

  # ------------------------------------------------------------------
  # v0.0.2 — tensor lifetime + round-trip
  # ------------------------------------------------------------------

  @doc """
  Upload a list of f32 values to a freshly-allocated GPU buffer.
  Returns `{:ok, tensor_ref}` where `tensor_ref` is an opaque
  `ResourceArc` whose underlying VkBuf is freed when GC'd.

      iex> Nx.Vulkan.init()
      :ok
      iex> {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      iex> {:ok, [1.0, 2.0, 3.0, 4.0]} = Nx.Vulkan.download_f32(t, 4)
  """
  def upload_f32(list) when is_list(list) do
    bin =
      list
      |> Enum.flat_map(fn x -> [<<x::float-32-native>>] end)
      |> IO.iodata_to_binary()

    Nx.Vulkan.Native.upload_binary(bin)
  end

  @doc "Upload a raw binary (already packed f32 little-endian) to GPU memory."
  def upload_binary(bin) when is_binary(bin) do
    Nx.Vulkan.Native.upload_binary(bin)
  end

  @doc """
  Download a GPU buffer back into a list of f32 values. `n_elements`
  must match what was uploaded.
  """
  def download_f32(tensor, n_elements) when is_integer(n_elements) and n_elements >= 0 do
    case Nx.Vulkan.Native.download_binary(tensor, n_elements * 4) do
      {:ok, bin} ->
        floats =
          for <<x::float-32-native <- bin>> do
            x
          end

        {:ok, floats}

      err ->
        err
    end
  end

  @doc "Download as a raw binary (caller does the unpack)."
  def download_binary(tensor, n_bytes) do
    Nx.Vulkan.Native.download_binary(tensor, n_bytes)
  end

  @doc "Byte size of an uploaded tensor."
  defdelegate byte_size(tensor), to: Nx.Vulkan.Native

  # ------------------------------------------------------------------
  # v0.0.3 — elementwise binary ops
  # ------------------------------------------------------------------

  @ops_binary %{
    add: 0,
    multiply: 1,
    subtract: 2,
    divide: 3,
    pow: 4,
    max: 5,
    min: 6
  }

  for {name, op_const} <- @ops_binary do
    @doc """
    Elementwise `#{name}` of two GPU tensors of equal length.
    Returns `{:ok, tensor}` or `{:error, reason}`.
    """
    def unquote(name)(a, b) do
      Nx.Vulkan.Native.apply_binary(a, b, unquote(op_const), shader_path("elementwise_binary.spv"))
    end
  end

  defp shader_path(name) do
    :nx_vulkan
    |> :code.priv_dir()
    |> Path.join("shaders")
    |> Path.join(name)
  end
end
