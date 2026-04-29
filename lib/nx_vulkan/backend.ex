defmodule Nx.Vulkan.Backend do
  @moduledoc """
  `Nx.Backend` implementation on top of the Vulkan compute primitives.

  Tensors are represented by:

      %Nx.Vulkan.Backend{ref: <ResourceArc<VulkanTensor>>, shape: tuple, type: {:f, 32}}

  Storage lives on the GPU; an Elixir reference to the `ResourceArc`
  keeps the GPU buffer alive. When the Elixir reference is GC'd, the
  Rust `Drop` impl frees the buffer.

  ## Status — v0.0.3 baseline

  The backend implements only the operators wired in earlier
  iterations:

    - `from_binary/3`, `to_binary/2`, `backend_copy/3`, `backend_transfer/3`
    - elementwise binary: `add`, `subtract`, `multiply`, `divide`, `pow`,
      `max`, `min`
    - elementwise unary:  `exp`, `log`, `sqrt`, `abs`, `negate`, `sigmoid`,
      `tanh`, `tanh`, `relu`-via-clamp-to-0…
    - reductions:         `sum`, `reduce_max`, `reduce_min` (all-axis only)
    - linear algebra:     `dot/6` (rank-2 × rank-2 matmul)

  Anything else falls back to `Nx.BinaryBackend` automatically via
  `Nx.backend_transfer/2`.

  ## Limitations

    - **f32 only.** f64 is supported by Spirit's shaders via spec
      constant but the type-system wiring takes more work; deferred.
    - **No autograd.** `defn grad` won't work end-to-end. The forward
      path is what this backend proves.
    - **All-axis reductions only.** Per-axis reductions are a
      separate iteration.
    - **No broadcasting in this backend.** Broadcasting happens in
      Nx's frontend before dispatch; we receive already-broadcast
      tensors. (Will swap to the broadcast shader later.)
  """

  @behaviour Nx.Backend

  @enforce_keys [:ref, :shape, :type]
  defstruct [:ref, :shape, :type]

  alias Nx.Tensor, as: T

  @impl true
  def init(opts), do: opts

  # ---------------------------------------------------------------- transfers

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, _opts) do
    ensure_f32!(type)
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(binary)
    put_in(tensor.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  @impl true
  def to_binary(%T{data: %__MODULE__{ref: ref, shape: shape}}, _limit) do
    n_bytes = byte_size_of(shape) * 4

    case Nx.Vulkan.Native.download_binary(ref, n_bytes) do
      {:ok, bin} -> bin
      {:error, reason} -> raise "Nx.Vulkan: download failed: #{inspect(reason)}"
    end
  end

  @impl true
  def backend_copy(%T{} = tensor, backend, opts) do
    bin = to_binary(tensor, byte_size_of(tensor.shape) * 4)
    backend.from_binary(tensor, bin, opts)
  end

  @impl true
  def backend_transfer(%T{} = tensor, backend, opts) do
    backend_copy(tensor, backend, opts)
  end

  @impl true
  def backend_deallocate(%T{}), do: :ok

  # ---------------------------------------------------------------- creation

  @impl true
  def constant(%T{shape: shape, type: type} = tensor, scalar, _opts) do
    ensure_f32!(type)
    n = byte_size_of(shape)
    bin = :binary.copy(<<:erlang.float(scalar)::float-32-native>>, n)
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(tensor.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  # ---------------------------------------------------------------- elementwise binary

  for {op, _} <- %{
        add: 0,
        subtract: 0,
        multiply: 0,
        divide: 0,
        pow: 0,
        max: 0,
        min: 0
      } do
    @impl true
    def unquote(op)(out, a, b) do
      do_binary(out, a, b, unquote(op))
    end
  end

  defp do_binary(%T{shape: shape, type: type} = out, %T{} = a, %T{} = b, op) do
    ensure_f32!(type)
    a_data = to_vulkan!(a)
    b_data = to_vulkan!(b)

    apply_op =
      case op do
        :add -> &Nx.Vulkan.add/2
        :multiply -> &Nx.Vulkan.multiply/2
        :subtract -> &Nx.Vulkan.subtract/2
        :divide -> &Nx.Vulkan.divide/2
        :pow -> &Nx.Vulkan.pow/2
        :max -> &Nx.Vulkan.max/2
        :min -> &Nx.Vulkan.min/2
      end

    {:ok, ref} = apply_op.(a_data.ref, b_data.ref)
    put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  # ---------------------------------------------------------------- elementwise unary

  @unary_ops [
    :exp,
    :log,
    :sqrt,
    :abs,
    :negate,
    :sigmoid,
    :tanh,
    :ceil,
    :floor,
    :sign
  ]

  for op <- @unary_ops do
    @impl true
    def unquote(op)(%T{shape: shape, type: type} = out, a) do
      ensure_f32!(type)
      a_data = to_vulkan!(a)
      apply_op = Function.capture(Nx.Vulkan, unquote(op), 1)
      {:ok, ref} = apply_op.(a_data.ref)
      put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
    end
  end

  # ---------------------------------------------------------------- reductions

  @impl true
  def sum(%T{type: type} = out, %T{} = t, _opts) do
    ensure_f32!(type)
    t_data = to_vulkan!(t)
    {:ok, scalar} = Nx.Vulkan.sum(t_data.ref)
    scalar_to_tensor(out, scalar)
  end

  @impl true
  def reduce_max(%T{type: type} = out, %T{} = t, _opts) do
    ensure_f32!(type)
    t_data = to_vulkan!(t)
    {:ok, scalar} = Nx.Vulkan.reduce_max(t_data.ref)
    scalar_to_tensor(out, scalar)
  end

  @impl true
  def reduce_min(%T{type: type} = out, %T{} = t, _opts) do
    ensure_f32!(type)
    t_data = to_vulkan!(t)
    {:ok, scalar} = Nx.Vulkan.reduce_min(t_data.ref)
    scalar_to_tensor(out, scalar)
  end

  defp scalar_to_tensor(%T{shape: shape, type: type} = out, scalar) do
    bin = :binary.copy(<<scalar::float-32-native>>, byte_size_of(shape))
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  # ---------------------------------------------------------------- matmul

  @impl true
  def dot(%T{shape: out_shape, type: type} = out, %T{shape: a_shape} = a, _ca, _ba,
          %T{shape: b_shape} = b, _cb, _bb) do
    ensure_f32!(type)
    {m, k} = a_shape
    {^k, n} = b_shape
    {^m, ^n} = out_shape

    a_data = to_vulkan!(a)
    b_data = to_vulkan!(b)
    {:ok, ref} = Nx.Vulkan.matmul(a_data.ref, b_data.ref, m, n, k)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  # ---------------------------------------------------------------- reshape

  # Zero-copy on the buffer — the GPU stores byte_size + ResourceArc;
  # shape is metadata in the Elixir tensor. Reshape just rewraps the
  # existing reference under a new shape without touching the GPU.
  @impl true
  def reshape(%T{shape: new_shape, type: type} = out, %T{data: %__MODULE__{ref: ref}}) do
    put_in(out.data, %__MODULE__{ref: ref, shape: new_shape, type: type})
  end

  @impl true
  def squeeze(%T{shape: new_shape, type: type} = out, %T{data: %__MODULE__{ref: ref}}, _axes) do
    # Same shape — squeeze drops trivial axes, but the byte layout is
    # unchanged. Pure metadata update.
    put_in(out.data, %__MODULE__{ref: ref, shape: new_shape, type: type})
  end

  # ---------------------------------------------------------------- transpose

  # 2D only at v0.1.2. Higher-rank transpose materializes via host
  # fallback (defer to v0.1.3 when slicing lands and we can do
  # rank-N permutation natively).
  @impl true
  def transpose(%T{shape: out_shape, type: type} = out, %T{shape: in_shape} = a, axes) do
    ensure_f32!(type)

    case {Tuple.to_list(in_shape), axes} do
      {[m, n], [1, 0]} ->
        a_data = to_vulkan!(a)
        {:ok, ref} = Nx.Vulkan.transpose_2d(a_data.ref, m, n)
        put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})

      _ ->
        # Higher-rank or non-swap permutation — host materialize for now.
        bin = to_binary(a, byte_size_of(in_shape) * 4)
        floats = for <<x::float-32-native <- bin>>, do: x

        in_dims = Tuple.to_list(in_shape)
        out_dims = Tuple.to_list(out_shape)
        n_total = byte_size_of(in_shape)

        permuted =
          for flat_idx <- 0..(n_total - 1) do
            out_coords = unflatten(flat_idx, out_dims)
            in_coords = Enum.map(0..(length(in_dims) - 1)//1,
                                 fn i -> Enum.at(out_coords, Enum.find_index(axes, &(&1 == i))) end)
            Enum.at(floats, flatten(in_coords, in_dims))
          end

        new_bin = permuted |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
        {:ok, ref} = Nx.Vulkan.Native.upload_binary(new_bin)
        put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
    end
  end

  # ---------------------------------------------------------------- broadcast

  # v0.1.2 cut: download → replicate on host → upload. Slow for large
  # tensors but correct for any shape/axes combination. v0.2 wires the
  # broadcast shader (spirit already has it as
  # elementwise_binary_broadcast.spv) for the in-place fast path.
  @impl true
  def broadcast(%T{shape: out_shape, type: type} = out, %T{shape: in_shape} = tensor, _shape, axes) do
    ensure_f32!(type)
    in_bin = to_binary(tensor, byte_size_of(in_shape) * 4)

    # Decode input as a flat list — it'll be re-indexed during expansion.
    input = for <<x::float-32-native <- in_bin>>, do: x

    in_dims = Tuple.to_list(in_shape)
    out_dims = Tuple.to_list(out_shape)
    n_out = byte_size_of(out_shape)

    output =
      for flat_idx <- 0..(n_out - 1) do
        flat_in = flat_in_index(flat_idx, in_dims, out_dims, axes)
        Enum.at(input, flat_in)
      end

    bin = output |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  # Scalar input → every output position maps to flat_in=0.
  defp flat_in_index(_flat_idx, [], _out_dims, _axes), do: 0

  # General N-D: project the output coord onto the input coord using the
  # axes mapping (axes[i] = j means input axis i corresponds to output
  # axis j).
  defp flat_in_index(flat_idx, in_dims, out_dims, axes) do
    out_coords = unflatten(flat_idx, out_dims)
    in_coords = Enum.map(0..(length(in_dims) - 1)//1, fn i -> Enum.at(out_coords, Enum.at(axes, i)) end)
    flatten(in_coords, in_dims)
  end

  defp unflatten(flat, dims) do
    {coords, _} =
      Enum.reduce(Enum.reverse(dims), {[], flat}, fn d, {acc, rem} ->
        {[rem(rem, d) | acc], div(rem, d)}
      end)

    coords
  end

  defp flatten(coords, dims) do
    Enum.zip(coords, dims)
    |> Enum.reduce(0, fn {c, d}, acc -> acc * d + c end)
  end

  # ---------------------------------------------------------------- helpers

  defp ensure_f32!({:f, 32}), do: :ok

  defp ensure_f32!(other) do
    raise ArgumentError,
          "Nx.Vulkan.Backend currently supports only {:f, 32}; got #{inspect(other)}"
  end

  # If the tensor's data isn't already a Vulkan backend, materialise it.
  defp to_vulkan!(%T{data: %__MODULE__{} = data}), do: data
  defp to_vulkan!(%T{} = t) do
    bin = Nx.to_binary(t)
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    %__MODULE__{ref: ref, shape: t.shape, type: t.type}
  end

  defp byte_size_of(shape) do
    shape |> Tuple.to_list() |> Enum.reduce(1, &*/2)
  end

  # ---------------------------------------------------------------- inspect

  @impl true
  def inspect(%T{} = tensor, opts) do
    Nx.Backend.inspect(tensor, to_binary(tensor, byte_size_of(tensor.shape) * 4), opts)
  end
end
