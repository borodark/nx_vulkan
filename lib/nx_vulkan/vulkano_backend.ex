defmodule Nx.Vulkan.VulkanoBackend do
  @moduledoc """
  Pure-Rust (vulkano) `Nx.Backend` implementation. Sibling of
  `Nx.Vulkan.Backend` (C++ spirit-backed); same compute fabric,
  different memory-management story.

  Tensors are represented by:

      %Nx.Vulkan.VulkanoBackend{ref: ResourceArc<VulkanoTensor>,
                                shape: tuple, type: {kind, bits}}

  The `ref` is a Rustler resource owning an `Arc<Subbuffer<u8>>` in
  vulkano. When the BEAM GCs the Elixir reference, vulkano's `Drop`
  runs `vkDestroyBuffer` + `vkFreeMemory`. Stale-handle bugs (where
  a freed `VkBuf*` is read back at the C++ layer) are structurally
  impossible: the `Subbuffer` cannot outlive its `Buffer`.

  ## Status — storage-only baseline

  This module implements **just the storage callbacks** required for
  tensors to round-trip host↔GPU without crashing:

    - `init/1`, `from_binary/3`, `to_binary/2`
    - `backend_copy/3`, `backend_transfer/3`, `backend_deallocate/1`
    - `inspect/2`, `constant/3`, `iota/3`, `eye/2`

  Compute ops (add / multiply / sum / matmul / …) are not yet
  implemented. To use this backend for actual computation,
  configure Nx to fall back via `Nx.BinaryBackend` for ops, or
  call `Nx.backend_transfer(t, Nx.BinaryBackend)` before computing.

  The next port chunk will add per-op compute NIFs to
  `Nx.Vulkan.NativeV` and wire them here.
  """

  @behaviour Nx.Backend

  @enforce_keys [:ref, :shape, :type]
  defstruct [:ref, :shape, :type]

  alias Nx.Tensor, as: T

  # ---------------------------------------------------------------- init

  @impl true
  def init(opts), do: opts

  # ---------------------------------------------------------------- storage

  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, _opts) do
    {:ok, ref} = Nx.Vulkan.NativeV.buf_upload(binary)
    put_in(tensor.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  @impl true
  def to_binary(%T{data: %__MODULE__{ref: ref}, shape: shape, type: type}, _limit) do
    {:ok, bin} = Nx.Vulkan.NativeV.buf_download(ref)
    expected = byte_size_of(shape) * element_bytes(type)

    cond do
      byte_size(bin) == expected -> bin
      byte_size(bin) > expected -> binary_part(bin, 0, expected)
      true -> bin
    end
  end

  @impl true
  def backend_copy(%T{} = tensor, target_backend, opts) do
    expected = byte_size_of(tensor.shape) * element_bytes(tensor.type)
    bin = to_binary(tensor, expected)
    target_backend.from_binary(tensor, bin, opts)
  end

  @impl true
  def backend_transfer(%T{} = tensor, backend, opts) do
    backend_copy(tensor, backend, opts)
  end

  @impl true
  def backend_deallocate(%T{}), do: :ok

  # ---------------------------------------------------------------- inspect

  @impl true
  def inspect(%T{} = tensor, opts) do
    try do
      tensor
      |> backend_copy(Nx.BinaryBackend, [])
      |> Nx.BinaryBackend.inspect(opts)
    catch
      :exit, _ -> Inspect.Algebra.string("#Nx.Vulkan.VulkanoBackend<unreadable>")
      _, _ -> Inspect.Algebra.string("#Nx.Vulkan.VulkanoBackend<unreadable>")
    end
  end

  # ---------------------------------------------------------------- creation

  @impl true
  def constant(%T{shape: shape, type: type} = tensor, scalar, _opts) do
    n = byte_size_of(shape)
    bin = :binary.copy(encode_scalar(scalar, type), n)
    {:ok, ref} = Nx.Vulkan.NativeV.buf_upload(bin)
    put_in(tensor.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  @impl true
  def iota(%T{shape: shape, type: type} = out, axis, _opts) do
    # Materialise on the host via BinaryBackend, then upload.
    iota_t = Nx.iota(shape, type: type, axis: axis, backend: Nx.BinaryBackend)
    from_binary(out, Nx.to_binary(iota_t), [])
  end

  @impl true
  def eye(%T{shape: shape, type: type} = out, _opts) do
    eye_t = Nx.eye(shape, type: type, backend: Nx.BinaryBackend)
    from_binary(out, Nx.to_binary(eye_t), [])
  end

  # ---------------------------------------------------------------- elementwise binary

  # Op codes match `priv/shaders/elementwise_binary.spv` spec constant ID 0:
  #   0=add  1=mul  2=sub  3=div  4=pow  5=max  6=min
  @binary_ops [
    add: 0,
    multiply: 1,
    subtract: 2,
    divide: 3,
    pow: 4,
    max: 5,
    min: 6
  ]

  @elementwise_binary_spv Path.expand(
                            "../../priv/shaders/elementwise_binary.spv",
                            __DIR__
                          )

  for {op, code} <- @binary_ops do
    @impl true
    def unquote(op)(
          %T{shape: shape, type: type} = out,
          %T{data: %__MODULE__{ref: a_ref}},
          %T{data: %__MODULE__{ref: b_ref}}
        ) do
      n = byte_size_of(shape)
      n_bytes = n * element_bytes(type)
      {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n_bytes)

      :ok =
        Nx.Vulkan.NativeV.apply_binary(
          out_ref,
          a_ref,
          b_ref,
          n,
          unquote(code),
          @elementwise_binary_spv
        )

      put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: type})
    end
  end

  # ---------------------------------------------------------------- elementwise unary

  # Op codes match `priv/shaders/elementwise_unary.spv` spec constant ID 0:
  #   0=exp  1=log  2=sqrt  3=abs  4=neg  5=sigmoid  6=tanh  7=relu
  #   8=ceil  9=floor  10=sign  11=reciprocal  12=square
  @unary_ops [
    exp: 0,
    log: 1,
    sqrt: 2,
    abs: 3,
    negate: 4,
    sigmoid: 5,
    tanh: 6,
    floor: 9,
    ceil: 8,
    sign: 10
  ]

  @elementwise_unary_spv Path.expand(
                           "../../priv/shaders/elementwise_unary.spv",
                           __DIR__
                         )

  for {op, code} <- @unary_ops do
    @impl true
    def unquote(op)(
          %T{shape: shape, type: type} = out,
          %T{data: %__MODULE__{ref: a_ref}}
        ) do
      n = byte_size_of(shape)
      n_bytes = n * element_bytes(type)
      {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n_bytes)

      :ok =
        Nx.Vulkan.NativeV.apply_unary(
          out_ref,
          a_ref,
          n,
          unquote(code),
          @elementwise_unary_spv
        )

      put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: type})
    end
  end

  # ---------------------------------------------------------------- reductions

  @reduce_axis_spv Path.expand("../../priv/shaders/reduce_axis.spv", __DIR__)

  @impl true
  def sum(out, t, opts), do: do_reduce(out, t, opts, 0)

  @impl true
  def reduce_max(out, t, opts), do: do_reduce(out, t, opts, 1)

  @impl true
  def reduce_min(out, t, opts), do: do_reduce(out, t, opts, 2)

  # Resolves the (outer, reduce_size, inner) virtual shape from
  # `opts[:axes]`. Supports all-axes (collapse to scalar) and
  # single-axis cases that map cleanly to contiguous slabs. More
  # exotic patterns fall back to BinaryBackend transfer.
  defp do_reduce(
         %T{shape: out_shape, type: type} = out,
         %T{data: %__MODULE__{ref: a_ref}, shape: in_shape} = _a,
         opts,
         op_code
       ) do
    axes = Keyword.get(opts, :axes) || all_axes(in_shape)

    case classify_reduce_axes(in_shape, axes) do
      {:ok, {outer, reduce_size, inner}} ->
        n_out = byte_size_of(out_shape)
        out_bytes = max(n_out, 1) * element_bytes(type)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(out_bytes)

        :ok =
          Nx.Vulkan.NativeV.reduce_axis(
            out_ref,
            a_ref,
            outer,
            reduce_size,
            inner,
            op_code,
            @reduce_axis_spv
          )

        put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})

      :fallback ->
        raise "non-contiguous axes #{inspect(axes)} for shape #{inspect(in_shape)}: not yet supported"
    end
  end

  defp all_axes(shape), do: Enum.to_list(0..(tuple_size(shape) - 1))

  # Classify the reduction shape:
  #   - All axes      → outer=1, reduce=product(shape), inner=1
  #   - Leading axes  → outer=1, reduce=product(reduced), inner=product(remaining)
  #   - Trailing axes → outer=product(remaining), reduce=product(reduced), inner=1
  defp classify_reduce_axes(in_shape, axes) do
    rank = tuple_size(in_shape)
    sorted = Enum.sort(axes)
    dims = Tuple.to_list(in_shape)

    cond do
      sorted == Enum.to_list(0..(rank - 1)) ->
        {:ok, {1, Enum.reduce(dims, 1, &Kernel.*/2), 1}}

      sorted == Enum.to_list(0..(length(sorted) - 1)) ->
        reduced = Enum.take(dims, length(sorted))
        remaining = Enum.drop(dims, length(sorted))
        outer = 1
        reduce_size = Enum.reduce(reduced, 1, &Kernel.*/2)
        inner = Enum.reduce(remaining, 1, &Kernel.*/2)
        {:ok, {outer, reduce_size, inner}}

      sorted == Enum.to_list((rank - length(sorted))..(rank - 1)) ->
        kept = Enum.take(dims, rank - length(sorted))
        reduced = Enum.drop(dims, rank - length(sorted))
        outer = Enum.reduce(kept, 1, &Kernel.*/2)
        reduce_size = Enum.reduce(reduced, 1, &Kernel.*/2)
        inner = 1
        {:ok, {outer, reduce_size, inner}}

      true ->
        :fallback
    end
  end

  # ---------------------------------------------------------------- shape / movement

  @transpose_spv Path.expand("../../priv/shaders/transpose.spv", __DIR__)

  # Reshape + squeeze are zero-copy: same buffer, new shape metadata.
  # The buffer might be physically larger than the new shape implies
  # if it came from an operation that allocated extra slack — that's
  # fine, the metadata determines what bytes get read out.

  @impl true
  def reshape(%T{shape: new_shape, type: type} = out, %T{data: %__MODULE__{ref: ref}}) do
    put_in(out.data, %__MODULE__{ref: ref, shape: new_shape, type: type})
  end

  @impl true
  def squeeze(%T{shape: new_shape, type: type} = out, %T{data: %__MODULE__{ref: ref}}, _axes) do
    put_in(out.data, %__MODULE__{ref: ref, shape: new_shape, type: type})
  end

  # 2D transpose. Higher-rank transposes (axis permutations) fall back
  # to BinaryBackend until we wire a general-rank shader.
  @impl true
  def transpose(
        %T{shape: out_shape, type: type} = out,
        %T{shape: in_shape, data: %__MODULE__{ref: a_ref}},
        axes
      ) do
    rank = tuple_size(in_shape)

    case {rank, axes} do
      {2, [1, 0]} ->
        m = elem(in_shape, 0)
        n = elem(in_shape, 1)
        n_bytes = m * n * element_bytes(type)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n_bytes)

        :ok =
          Nx.Vulkan.NativeV.transpose_2d(
            out_ref,
            a_ref,
            m,
            n,
            @transpose_spv
          )

        put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})

      _ ->
        raise "transpose rank=#{rank} axes=#{inspect(axes)}: only 2D [1,0] supported in stage 4"
    end
  end

  # ---------------------------------------------------------------- helpers

  defp byte_size_of(shape) when is_tuple(shape) do
    shape |> Tuple.to_list() |> Enum.reduce(1, &(&1 * &2))
  end

  defp element_bytes({:f, 32}), do: 4
  defp element_bytes({:f, 64}), do: 8
  defp element_bytes({:s, 8}), do: 1
  defp element_bytes({:s, 16}), do: 2
  defp element_bytes({:s, 32}), do: 4
  defp element_bytes({:s, 64}), do: 8
  defp element_bytes({:u, 8}), do: 1
  defp element_bytes({:u, 16}), do: 2
  defp element_bytes({:u, 32}), do: 4
  defp element_bytes({:u, 64}), do: 8
  defp element_bytes({:bf, 16}), do: 2

  defp encode_scalar(s, {:f, 32}), do: <<s / 1.0::float-32-native>>
  defp encode_scalar(s, {:f, 64}), do: <<s / 1.0::float-64-native>>
  defp encode_scalar(s, {:s, 8}), do: <<trunc(s)::signed-8>>
  defp encode_scalar(s, {:s, 16}), do: <<trunc(s)::signed-16-native>>
  defp encode_scalar(s, {:s, 32}), do: <<trunc(s)::signed-32-native>>
  defp encode_scalar(s, {:s, 64}), do: <<trunc(s)::signed-64-native>>
  defp encode_scalar(s, {:u, 8}), do: <<trunc(s)::unsigned-8>>
  defp encode_scalar(s, {:u, 16}), do: <<trunc(s)::unsigned-16-native>>
  defp encode_scalar(s, {:u, 32}), do: <<trunc(s)::unsigned-32-native>>
  defp encode_scalar(s, {:u, 64}), do: <<trunc(s)::unsigned-64-native>>
  defp encode_scalar(s, {:bf, 16}), do: <<s / 1.0::float-16-native>>
end
