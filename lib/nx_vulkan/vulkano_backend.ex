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
  @elementwise_binary_f64_spv Path.expand(
                                "../../priv/shaders/elementwise_binary_f64.spv",
                                __DIR__
                              )

  defp binary_spv({:f, 32}), do: @elementwise_binary_spv
  defp binary_spv({:f, 64}), do: @elementwise_binary_f64_spv
  defp binary_spv(_), do: nil

  for {op, code} <- @binary_ops do
    @impl true
    def unquote(op)(%T{shape: shape, type: type} = out, a, b) do
      a_v = ensure_on_backend(a)
      b_v = ensure_on_backend(b)
      spv = binary_spv(type)

      shape_match =
        a_v.shape == b_v.shape and a_v.shape == shape and a_v.type == b_v.type and
          a_v.type == type

      if spv != nil and shape_match do
        %T{data: %__MODULE__{ref: a_ref}} = a_v
        %T{data: %__MODULE__{ref: b_ref}} = b_v
        n = byte_size_of(shape)
        n_bytes = n * element_bytes(type)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n_bytes)

        :ok =
          Nx.Vulkan.NativeV.apply_binary(out_ref, a_ref, b_ref, n, unquote(code), spv)

        put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: type})
      else
        binary_op_host_fallback(unquote(op), out, a_v, b_v)
      end
    end
  end

  defp binary_op_host_fallback(op, out, a, b) do
    a_bin = Nx.backend_transfer(a, Nx.BinaryBackend)
    b_bin = Nx.backend_transfer(b, Nx.BinaryBackend)
    result = apply(Nx, op, [a_bin, b_bin])
    host_result(out, result)
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
  @elementwise_unary_f64_spv Path.expand(
                               "../../priv/shaders/elementwise_unary_f64.spv",
                               __DIR__
                             )

  defp unary_spv({:f, 32}), do: @elementwise_unary_spv
  defp unary_spv({:f, 64}), do: @elementwise_unary_f64_spv
  defp unary_spv(_), do: nil

  for {op, code} <- @unary_ops do
    @impl true
    def unquote(op)(%T{shape: shape, type: type} = out, a) do
      a_v = ensure_on_backend(a)
      spv = unary_spv(type)

      if spv != nil and a_v.type == type do
        %T{data: %__MODULE__{ref: a_ref}} = a_v
        n = byte_size_of(shape)
        n_bytes = n * element_bytes(type)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n_bytes)

        :ok = Nx.Vulkan.NativeV.apply_unary(out_ref, a_ref, n, unquote(code), spv)

        put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: type})
      else
        unary_op_host_fallback(unquote(op), out, a_v)
      end
    end
  end

  defp unary_op_host_fallback(op, out, a) do
    a_bin = Nx.backend_transfer(a, Nx.BinaryBackend)
    result = apply(Nx, op, [a_bin])
    host_result(out, result)
  end

  # ---------------------------------------------------------------- reductions

  @reduce_axis_spv Path.expand("../../priv/shaders/reduce_axis.spv", __DIR__)
  @reduce_axis_f64_spv Path.expand("../../priv/shaders/reduce_axis_f64.spv", __DIR__)

  defp reduce_spv({:f, 32}), do: @reduce_axis_spv
  defp reduce_spv({:f, 64}), do: @reduce_axis_f64_spv
  defp reduce_spv(_), do: nil

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
         %T{shape: in_shape} = tensor,
         opts,
         op_code
       ) do
    axes = Keyword.get(opts, :axes) || all_axes(in_shape)

    spv = reduce_spv(type)

    fast_path =
      spv != nil and tensor.type == type and
        match?(%__MODULE__{}, tensor.data) and
        match?({:ok, _}, classify_reduce_axes(in_shape, axes))

    if fast_path do
      %T{data: %__MODULE__{ref: a_ref}} = tensor
      {:ok, {outer, reduce_size, inner}} = classify_reduce_axes(in_shape, axes)
      n_out = max(byte_size_of(out_shape), 1)
      out_bytes = n_out * element_bytes(type)
      {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(out_bytes)

      :ok =
        Nx.Vulkan.NativeV.reduce_axis(out_ref, a_ref, outer, reduce_size, inner, op_code, spv)

      put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
    else
      reduce_op_host_fallback(op_code, out, tensor, opts)
    end
  end

  defp reduce_op_host_fallback(op_code, out, tensor, opts) do
    bin_in = Nx.backend_transfer(tensor, Nx.BinaryBackend)

    op =
      case op_code do
        0 -> :sum
        1 -> :reduce_max
        2 -> :reduce_min
      end

    result = apply(Nx, op, [bin_in, opts])
    host_result(out, result)
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

  # ---------------------------------------------------------------- host-fallback ops

  # as_type — Nx-level cast via BinaryBackend. For f32↔f32 (no-op) we
  # just rewrap. For real casts we round-trip through host.
  @impl true
  def as_type(%T{type: type} = out, %T{type: source_type, data: %__MODULE__{ref: ref}} = tensor) do
    if type == source_type do
      put_in(out.data, %__MODULE__{ref: ref, shape: out.shape, type: type})
    else
      bin_in = Nx.backend_transfer(tensor, Nx.BinaryBackend)
      bin_cast = Nx.as_type(bin_in, type)
      host_result(out, bin_cast)
    end
  end

  # Comparison ops — host-fallback. The elementwise_binary.spv catalog
  # has op codes 7/8/9 (equal/less/greater) but its output is f32, not
  # u8 (the type Nx expects from a comparison). Routing through
  # BinaryBackend keeps the Nx type contract correct. Scholar uses
  # comparison + select heavily; this unblocks the classical-ML target.
  for op <- [:equal, :not_equal, :less, :less_equal, :greater, :greater_equal] do
    @impl true
    def unquote(op)(out, a, b) do
      a_v = ensure_on_backend(a)
      b_v = ensure_on_backend(b)
      a_bin = Nx.backend_transfer(a_v, Nx.BinaryBackend)
      b_bin = Nx.backend_transfer(b_v, Nx.BinaryBackend)
      result = apply(Nx, unquote(op), [a_bin, b_bin])
      host_result(out, result)
    end
  end

  # select(cond, on_true, on_false) — host-fallback.
  @impl true
  def select(out, pred, on_true, on_false) do
    pred_bin = Nx.backend_transfer(ensure_on_backend(pred), Nx.BinaryBackend)
    t_bin = Nx.backend_transfer(ensure_on_backend(on_true), Nx.BinaryBackend)
    f_bin = Nx.backend_transfer(ensure_on_backend(on_false), Nx.BinaryBackend)
    result = Nx.select(pred_bin, t_bin, f_bin)
    host_result(out, result)
  end

  # all/3, any/3 — boolean reductions, host-fallback.
  for op <- [:all, :any] do
    @impl true
    def unquote(op)(out, tensor, opts) do
      bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
      result = apply(Nx, unquote(op), [bin, opts])
      host_result(out, result)
    end
  end

  # block/4 — Nx's "block" callback dispatches Nx.Block-derived structs
  # (SVD, QR, LU, etc.). Host-fallback: transfer every tensor in the
  # input tuple to BinaryBackend, evaluate via its block impl, then
  # transfer outputs back. Scholar's linear regression uses
  # Nx.Block.LinAlg.SVD internally; this unblocks it.
  @impl true
  def block(out, block_def, inputs, opts) do
    transfer_to_bin = fn t ->
      if is_struct(t, Nx.Tensor) and match?(%__MODULE__{}, t.data) do
        Nx.backend_transfer(t, Nx.BinaryBackend)
      else
        t
      end
    end

    inputs_bin =
      cond do
        is_list(inputs) -> Enum.map(inputs, transfer_to_bin)
        is_tuple(inputs) -> inputs |> Tuple.to_list() |> Enum.map(transfer_to_bin) |> List.to_tuple()
        true -> transfer_to_bin.(inputs)
      end

    result = Nx.BinaryBackend.block(out, block_def, inputs_bin, opts)

    # Per Tier 1 of SHAPE_C_PLAN.md: result is already on BinaryBackend,
    # leave it there. Nx supports mixed-backend tensors flowing through
    # the pipeline; the next op auto-transfers if it needs GPU.
    result
  end

  # ---------------------------------------------------------------- slice (host fallback)

  # Slice is host-routed: download the source tensor to BinaryBackend,
  # do the slice there, upload the slab back. A future stage adds a
  # GPU-side slice shader for contiguous prefixes; until then this is
  # correct but copies through host memory.
  @impl true
  def slice(out, tensor, start_indices, lengths, strides) do
    # Delegate to Nx-level slice on BinaryBackend; result stays on host
    # (Tier 1 of SHAPE_C_PLAN.md — avoid the upload-back round trip).
    bin_in = Nx.backend_transfer(tensor, Nx.BinaryBackend)
    bin_result = Nx.slice(bin_in, start_indices, lengths, strides: strides)
    host_result(out, bin_result)
  end

  # ---------------------------------------------------------------- sampler-path host fallbacks

  # Nx.Backend.pad/4 callback. The Nx sampler uses pad to extend tensors
  # along arbitrary axes (NUTS leapfrog scratch buffers, batched chain
  # padding, etc.). No GPU pad shader yet — round-trip through
  # BinaryBackend. Pad value comes in as a tensor; transfer it too.
  @impl true
  def pad(out, tensor, pad_value, padding_config) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    pv_bin = Nx.backend_transfer(ensure_on_backend(pad_value), Nx.BinaryBackend)
    result = Nx.pad(t_bin, pv_bin, padding_config)
    host_result(out, result)
  end

  # put_slice: write `slice` into `tensor` at `start_indices`. Used by
  # NUTS batched leapfrog to accumulate trajectory steps.
  @impl true
  def put_slice(out, tensor, start_indices, slice) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    s_bin = Nx.backend_transfer(ensure_on_backend(slice), Nx.BinaryBackend)
    idx_bin = Enum.map(start_indices, &maybe_transfer_idx/1)
    result = Nx.put_slice(t_bin, idx_bin, s_bin)
    host_result(out, result)
  end

  # indexed_put: scatter updates into tensor at indices. Used to fill
  # NUTS per-step logp slots inside the leapfrog while loop.
  @impl true
  def indexed_put(out, tensor, indices, updates, opts \\ []) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    i_bin = Nx.backend_transfer(ensure_on_backend(indices), Nx.BinaryBackend)
    u_bin = Nx.backend_transfer(ensure_on_backend(updates), Nx.BinaryBackend)
    result = Nx.indexed_put(t_bin, i_bin, u_bin, opts)
    host_result(out, result)
  end

  # indexed_add: scatter-accumulate. Same shape as indexed_put.
  @impl true
  def indexed_add(out, tensor, indices, updates, opts \\ []) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    i_bin = Nx.backend_transfer(ensure_on_backend(indices), Nx.BinaryBackend)
    u_bin = Nx.backend_transfer(ensure_on_backend(updates), Nx.BinaryBackend)
    result = Nx.indexed_add(t_bin, i_bin, u_bin, opts)
    host_result(out, result)
  end

  # broadcast: project a tensor to a new shape along `axes`. Implicit
  # broadcasts during binary ops route through binary_op_host_fallback,
  # but explicit Nx.broadcast/3 needs its own callback. Used at every
  # NUTS init / mass-matrix scaffolding site.
  @impl true
  def broadcast(out, tensor, shape, axes) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.broadcast(t_bin, shape, axes: axes)
    host_result(out, result)
  end

  # concatenate: join tensors along an axis. Used by Sampler.sample_stream
  # and several diagnostic paths.
  #
  # Tier 2 step 1 of SHAPE_C_PLAN.md: GPU-native fast path when all
  # inputs are already VulkanoBackend AND axis == 0. Outer-axis
  # concat is a contiguous byte-level append in row-major layout, so
  # `vkCmdCopyBuffer` per input into a fresh device buffer does it
  # without a shader. Result stays on the device. Other shapes
  # (non-outer axis, mixed backend inputs) fall through to the
  # existing host-fallback.
  @impl true
  def concatenate(out, tensors, axis) do
    cond do
      axis == 0 and all_vulkano?(tensors) ->
        concat_vulkano(out, tensors)

      true ->
        bins = Enum.map(tensors, &Nx.backend_transfer(ensure_on_backend(&1), Nx.BinaryBackend))
        result = Nx.concatenate(bins, axis: axis)
        host_result(out, result)
    end
  end

  defp all_vulkano?(tensors) do
    Enum.all?(tensors, fn
      %T{data: %__MODULE__{}} -> true
      _ -> false
    end)
  end

  defp concat_vulkano(out, tensors) do
    refs = Enum.map(tensors, fn %T{data: %__MODULE__{ref: r}} -> r end)
    {:ok, ref} = Nx.Vulkan.NativeV.concat_buffers(refs)
    put_in(out.data, %__MODULE__{ref: ref, shape: out.shape, type: out.type})
  end

  # stack: combine N tensors along a NEW axis. Nx allocates the
  # output on the default backend (which is VulkanoBackend when the
  # app pins it at boot), then dispatches stack/3 to the OUTPUT's
  # backend — even when the input list is BinaryBackend tensors (the
  # common NUTS trace-building case, where every per-iteration draw is
  # a small BinaryBackend tensor). Tier 1 host fallback: transfer
  # everything to BinaryBackend, stack there, return on BinaryBackend.
  # `tensors` is a LIST despite the singular `tensor` in the @callback
  # signature.
  @impl true
  def stack(out, tensors, axis) do
    bins =
      Enum.map(tensors, fn t ->
        Nx.backend_transfer(ensure_on_backend(t), Nx.BinaryBackend)
      end)

    result = Nx.stack(bins, axis: axis)
    host_result(out, result)
  end

  # gather: pick elements at given index tuples. Underpins take/
  # take_diagonal in Nx 0.10's lowering.
  @impl true
  def gather(out, tensor, indices, opts \\ []) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    i_bin = Nx.backend_transfer(ensure_on_backend(indices), Nx.BinaryBackend)
    result = Nx.gather(t_bin, i_bin, opts)
    host_result(out, result)
  end

  # take: pick along a single axis. Common in trajectory selection.
  # The Nx.Backend.take/4 callback's 4th arg is a keyword (e.g.
  # `[axis: 0]`), NOT a bare integer — forward it through verbatim.
  @impl true
  def take(out, tensor, indices, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    i_bin = Nx.backend_transfer(ensure_on_backend(indices), Nx.BinaryBackend)
    result = Nx.take(t_bin, i_bin, opts)
    host_result(out, result)
  end

  # argmax / argmin: indices of extrema along an axis. Used by
  # credible-interval extraction and PyMC-style posterior summaries.
  # Tier 1 host fallback — no GPU shader yet.
  @impl true
  def argmax(out, tensor, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.argmax(t_bin, opts)
    host_result(out, result)
  end

  @impl true
  def argmin(out, tensor, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.argmin(t_bin, opts)
    host_result(out, result)
  end

  # clip: elementwise clip to [min, max]. min and max arrive as
  # scalar tensors. Could decompose to two elementwise ops on the
  # GPU later; for now, Tier 1 host fallback keeps the contract.
  @impl true
  def clip(out, tensor, min, max) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    min_bin = Nx.backend_transfer(ensure_on_backend(min), Nx.BinaryBackend)
    max_bin = Nx.backend_transfer(ensure_on_backend(max), Nx.BinaryBackend)
    result = Nx.clip(t_bin, min_bin, max_bin)
    host_result(out, result)
  end

  # Indices passed to put_slice can be scalar tensors or integers;
  # normalise to whatever BinaryBackend.put_slice expects.
  defp maybe_transfer_idx(%T{} = t),
    do: Nx.backend_transfer(ensure_on_backend(t), Nx.BinaryBackend)

  defp maybe_transfer_idx(i) when is_integer(i), do: i

  # ---------------------------------------------------------------- linalg

  @matmul_spv Path.expand("../../priv/shaders/matmul.spv", __DIR__)

  # Dot product (matmul) — Nx callback signature:
  #   dot(out, a, contracting_axes_a, batched_axes_a,
  #            b, contracting_axes_b, batched_axes_b)
  #
  # Fast path: rank-2 × rank-2, contracting [1] of a vs [0] of b
  # (standard matmul A·B). f32 only — matmul.spv has no f64 variant
  # in priv/shaders/ yet. Everything else routes through BinaryBackend.
  @impl true
  def dot(%T{shape: out_shape, type: type} = out, a, axes_a, batched_a, b, axes_b, batched_b) do
    a_v = ensure_on_backend(a)
    b_v = ensure_on_backend(b)

    fast_path =
      type == {:f, 32} and a_v.type == {:f, 32} and b_v.type == {:f, 32} and
        tuple_size(a_v.shape) == 2 and tuple_size(b_v.shape) == 2 and
        axes_a == [1] and axes_b == [0] and
        batched_a == [] and batched_b == []

    if fast_path do
      %T{data: %__MODULE__{ref: a_ref}, shape: a_shape} = a_v
      %T{data: %__MODULE__{ref: b_ref}, shape: b_shape} = b_v
      m = elem(a_shape, 0)
      k_a = elem(a_shape, 1)
      n = elem(b_shape, 1)

      out_bytes = m * n * element_bytes(type)
      {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(out_bytes)

      :ok = Nx.Vulkan.NativeV.matmul(out_ref, a_ref, b_ref, m, n, k_a, @matmul_spv)

      put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
    else
      a_bin = Nx.backend_transfer(a_v, Nx.BinaryBackend)
      b_bin = Nx.backend_transfer(b_v, Nx.BinaryBackend)
      result = Nx.dot(a_bin, axes_a, batched_a, b_bin, axes_b, batched_b)
      host_result(out, result)
    end
  end

  # ---------------------------------------------------------------- helpers

  # Tolerate inputs from other backends — Nx.Defn.Evaluator may hand us
  # tensors that haven't been transferred yet (e.g. an Nx.constant
  # produced on BinaryBackend before the op dispatches here).
  defp ensure_on_backend(%T{data: %__MODULE__{}} = t), do: t

  defp ensure_on_backend(%T{} = t) do
    Nx.backend_transfer(t, __MODULE__)
  end

  # Tier 1 of SHAPE_C_PLAN.md — host-fallback result helper. The
  # compute already ran on BinaryBackend; the result tensor's `.data`
  # is `%Nx.BinaryBackend{state: bin}`. Rebuilding it via
  # `from_binary(out, Nx.to_binary(result), [])` would serialise then
  # `buf_upload` to a fresh GPU buffer — pure waste, since the only
  # thing vulkano contributes to a host-fallback op is the round trip.
  # Leave the result on BinaryBackend; Nx handles mixed-backend tensors
  # downstream, and any op that genuinely needs GPU will transfer
  # lazily on first touch.
  defp host_result(%T{} = out, %T{} = result), do: %{out | data: result.data}

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
