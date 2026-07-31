defmodule Nx.Vulkan.VulkanoBackend do
  @moduledoc """
  Pure-Rust (vulkano) `Nx.Backend` implementation. f64-only compute;
  f32 inputs are accepted and cast as needed.

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
  def to_binary(%T{data: %__MODULE__{ref: ref}, type: type}, limit) do
    {:ok, bin} = Nx.Vulkan.NativeV.buf_download(ref)
    # `limit` is the number of ELEMENTS Nx wants (already capped by Nx, and for
    # vectorized tensors it counts the vectorized axes too — so don't clamp to
    # `shape`). The download may carry slack from over-allocation; return exactly
    # the first `limit` elements' bytes. (Previously the limit was ignored, so
    # Nx.to_binary(t, k) returned the whole tensor — found via `doctest Nx`.)
    want = limit * element_bytes(type)
    binary_part(bin, 0, min(want, byte_size(bin)))
  end

  @impl true
  def backend_copy(%T{} = tensor, target_backend, opts) do
    # to_binary/2's limit is an element count — pass the full element count.
    bin = to_binary(tensor, byte_size_of(tensor.shape))
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
  def constant(%T{shape: shape, type: type} = tensor, scalar, opts) do
    case encode_scalar(scalar, type) do
      :error ->
        # dtypes without a native encoder (bf16/f8/complex) build on BinaryBackend
        host_result(tensor, with_binary_backend(fn -> Nx.BinaryBackend.constant(tensor, scalar, opts) end))

      bin when is_binary(bin) ->
        n = byte_size_of(shape)
        {:ok, ref} = Nx.Vulkan.NativeV.buf_upload(:binary.copy(bin, n))
        put_in(tensor.data, %__MODULE__{ref: ref, shape: shape, type: type})
    end
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

  @elementwise_binary_f64_spv Path.expand(
                                "../../priv/shaders/elementwise_binary_f64.spv",
                                __DIR__
                              )
  @elementwise_binary_f32_spv Path.expand(
                                "../../priv/shaders/elementwise_binary_f32.spv",
                                __DIR__
                              )

  defp binary_spv({:f, 64}), do: @elementwise_binary_f64_spv
  defp binary_spv({:f, 32}), do: @elementwise_binary_f32_spv
  defp binary_spv(_), do: nil

  # Broadcasting elementwise binary (rank <= 4) — keeps bias-add / scaling /
  # relu-via-max on the GPU instead of host-falling-back.
  @bcast_binary_f64_spv Path.expand("../../priv/shaders/elementwise_binary_bcast_f64.spv", __DIR__)
  @bcast_binary_f32_spv Path.expand("../../priv/shaders/elementwise_binary_bcast_f32.spv", __DIR__)

  defp bcast_binary_spv({:f, 64}), do: @bcast_binary_f64_spv
  defp bcast_binary_spv({:f, 32}), do: @bcast_binary_f32_spv
  defp bcast_binary_spv(_), do: nil

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
        bspv = bcast_binary_spv(type)

        if bspv != nil and unquote(code) != 4 and bcast_ok?(a_v, b_v, out) do
          gpu_bcast_binary(out, a_v, b_v, unquote(code), bspv)
        else
          binary_op_host_fallback(unquote(op), out, a_v, b_v)
        end
      end
    end
  end

  # Broadcast GPU path is valid when both operands are on this backend, match the
  # output type, and the output rank is 1..4 (Nx guarantees a,b broadcast to
  # out.shape). pow is excluded above (fp64 has no pow).
  defp bcast_ok?(%T{type: t} = a, %T{type: t} = b, %T{shape: os, type: t}) do
    match?(%__MODULE__{}, a.data) and match?(%__MODULE__{}, b.data) and
      tuple_size(os) >= 1 and tuple_size(os) <= 4
  end

  defp bcast_ok?(_a, _b, _out), do: false

  defp gpu_bcast_binary(out, %T{data: %__MODULE__{ref: a_ref}} = a, %T{data: %__MODULE__{ref: b_ref}} = b, code, spv) do
    rank = tuple_size(out.shape)
    outl = Tuple.to_list(out.shape)
    al = pad_left(Tuple.to_list(a.shape), rank)
    bl = pad_left(Tuple.to_list(b.shape), rank)

    params =
      for v <- [rank] ++ pad4(outl) ++ pad4(al) ++ pad4(bl), into: <<>>, do: <<v::signed-32-little>>

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    n = byte_size_of(out.shape)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(out.type))

    :ok = Nx.Vulkan.NativeV.apply_binary_broadcast(out_ref, a_ref, b_ref, params_ref, n, rank, code, spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: out.type})
  end

  defp pad_left(list, rank), do: List.duplicate(1, rank - length(list)) ++ list
  defp pad4(list), do: (list ++ [1, 1, 1, 1]) |> Enum.take(4)

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

  @elementwise_unary_f64_spv Path.expand(
                               "../../priv/shaders/elementwise_unary_f64.spv",
                               __DIR__
                             )
  @elementwise_unary_f32_spv Path.expand(
                               "../../priv/shaders/elementwise_unary_f32.spv",
                               __DIR__
                             )

  defp unary_spv({:f, 64}), do: @elementwise_unary_f64_spv
  defp unary_spv({:f, 32}), do: @elementwise_unary_f32_spv
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

  # Unary ops without GPU shader support — host fallback only.
  # Nx 0.12 requires all Nx.Backend callbacks to be implemented.
  @host_fallback_unary_ops [
    # original batch
    :log1p, :erf, :erfc, :expm1, :cbrt, :rsqrt,
    # trig
    :acos, :acosh, :asin, :asinh, :atan, :atanh,
    :cos, :cosh, :sin, :sinh, :tan,
    # type / check
    :is_infinity, :is_nan, :round,
    # special
    :erf_inv,
    # bitwise unary
    :bitwise_not, :count_leading_zeros, :population_count,
    # complex
    :conjugate, :real, :imag
  ]

  for op <- @host_fallback_unary_ops do
    @impl true
    def unquote(op)(%T{} = out, a) do
      unary_op_host_fallback(unquote(op), out, ensure_on_backend(a))
    end
  end

  # Binary ops without GPU shader support — host fallback only.
  @host_fallback_binary_ops [
    # bitwise
    :bitwise_and, :bitwise_or, :bitwise_xor,
    :left_shift, :right_shift,
    # integer
    :quotient, :remainder,
    # logical
    :logical_and, :logical_or, :logical_xor,
    # trig
    :atan2
  ]

  for op <- @host_fallback_binary_ops do
    @impl true
    def unquote(op)(%T{} = out, a, b) do
      a_bin = Nx.backend_transfer(ensure_on_backend(a), Nx.BinaryBackend)
      b_bin = Nx.backend_transfer(ensure_on_backend(b), Nx.BinaryBackend)
      result = apply(Nx, unquote(op), [a_bin, b_bin])
      host_result(out, result)
    end
  end

  # conv — native im2col + GEMM on the GPU for the common case (f64, or f32 with
  # an f64 accumulator in the GEMM); host fallback otherwise. Nx hands the
  # backend fully-resolved strides, padding ({lo,hi} per spatial dim),
  # input/kernel dilation, group sizes and permutations, plus an output template
  # already in output-permutation layout.
  @conv_im2col_f64_spv Path.expand("../../priv/shaders/conv_im2col_f64.spv", __DIR__)
  @conv_gemm_f64_spv Path.expand("../../priv/shaders/conv_gemm_f64.spv", __DIR__)
  @conv_im2col_f32_spv Path.expand("../../priv/shaders/conv_im2col_f32.spv", __DIR__)
  # conv's GEMM is a matmul, so it honours the same f32 accumulator policy
  # (f32_matmul_accumulator/0): :f64 accumulator by default, :f32 for speed on
  # f64-rate-limited GPUs. im2col is pure f32 movement (no accumulator).
  @conv_gemm_f32_f64acc_spv Path.expand("../../priv/shaders/conv_gemm_f32_f64acc.spv", __DIR__)
  @conv_gemm_f32_f32acc_spv Path.expand("../../priv/shaders/conv_gemm_f32_f32acc.spv", __DIR__)

  defp conv_spvs({:f, 64}), do: {@conv_im2col_f64_spv, @conv_gemm_f64_spv}

  defp conv_spvs({:f, 32}) do
    gemm =
      case f32_matmul_accumulator() do
        :f32 -> @conv_gemm_f32_f32acc_spv
        _ -> @conv_gemm_f32_f64acc_spv
      end

    {@conv_im2col_f32_spv, gemm}
  end

  defp conv_spvs(_), do: nil

  @impl true
  def conv(out, inp, kernel, opts) do
    i = ensure_on_backend(inp)
    k = ensure_on_backend(kernel)

    if conv_gpu_ok?(i, k, out, opts) do
      gpu_conv(out, i, k, opts)
    else
      inp_bin = Nx.backend_transfer(i, Nx.BinaryBackend)
      kernel_bin = Nx.backend_transfer(k, Nx.BinaryBackend)
      host_result(out, Nx.conv(inp_bin, kernel_bin, opts))
    end
  end

  # GPU path covers: spatial rank 1..3, feature/batch groups == 1, identity
  # permutations, f64 or f32 input/kernel/output (all three must match). Any
  # strides, padding and input/kernel dilation are honoured (folded into the
  # im2col index math). Groups > 1, non-identity permutations, mixed/other
  # dtypes and higher rank fall back.
  defp conv_gpu_ok?(%T{shape: ishape} = i, %T{shape: kshape} = k, %T{type: ot}, opts) do
    rank = tuple_size(ishape)
    sr = rank - 2

    match?(%__MODULE__{}, i.data) and match?(%__MODULE__{}, k.data) and
      i.type == ot and k.type == ot and ot in [{:f, 64}, {:f, 32}] and
      sr >= 1 and sr <= 3 and
      Keyword.get(opts, :feature_group_size, 1) == 1 and
      Keyword.get(opts, :batch_group_size, 1) == 1 and
      identity_perm?(opts[:input_permutation], rank) and
      identity_perm?(opts[:kernel_permutation], tuple_size(kshape)) and
      identity_perm?(opts[:output_permutation], rank)
  end

  defp conv_gpu_ok?(_i, _k, _out, _opts), do: false

  defp identity_perm?(nil, _rank), do: true
  defp identity_perm?(perm, rank), do: perm == Enum.to_list(0..(rank - 1)//1)

  defp gpu_conv(
         %T{type: type} = out,
         %T{shape: ishape, data: %__MODULE__{ref: in_ref}},
         %T{shape: kshape, data: %__MODULE__{ref: k_ref}},
         opts
       ) do
    {im2col_spv, gemm_spv} = conv_spvs(type)
    ebytes = element_bytes(type)
    rank = tuple_size(ishape)
    sr = rank - 2
    n = elem(ishape, 0)
    cin = elem(ishape, 1)
    cout = elem(kshape, 0)

    spatial = fn shape -> for ax <- 0..(sr - 1)//1, do: elem(shape, 2 + ax) end
    d = spatial.(ishape)
    kdims = spatial.(kshape)
    odims = spatial.(out.shape)
    pad_lo = Enum.map(opts[:padding], fn {lo, _hi} -> lo end)

    # Pad each per-dim list to length 3 with its identity default so the
    # rank-3 shaders can treat rank 1/2 uniformly (unused dims size 1).
    order = [
      {d, 1},
      {odims, 1},
      {kdims, 1},
      {opts[:strides], 1},
      {pad_lo, 0},
      {opts[:input_dilation], 1},
      {opts[:kernel_dilation], 1}
    ]

    params_bin =
      for {list, default} <- order, v <- pad3(list, default), into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params_bin)

    o_total = Enum.product(odims)
    k_total = Enum.product(kdims)
    k_cols = cin * k_total
    m = n * o_total

    {:ok, col_ref} = Nx.Vulkan.NativeV.buf_alloc(m * k_cols * ebytes)

    :ok =
      Nx.Vulkan.NativeV.conv_im2col(
        col_ref,
        in_ref,
        params_ref,
        n,
        cin,
        o_total,
        k_total,
        k_cols,
        im2col_spv
      )

    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * cout * o_total * ebytes)

    :ok =
      Nx.Vulkan.NativeV.conv_gemm(out_ref, col_ref, k_ref, n, cout, o_total, k_cols, gemm_spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: type})
  end

  defp pad3([a], d), do: [a, d, d]
  defp pad3([a, b], d), do: [a, b, d]
  defp pad3([a, b, c], _d), do: [a, b, c]

  # fft / ifft — native f64 Cooley-Tukey on the GPU for the common case;
  # host fallback otherwise. Nx resolves :length and :axis to concrete ints
  # before dispatch and sets out.type = to_complex(input) (f64 -> c128).
  @fft_bitrev_spv Path.expand("../../priv/shaders/fft_bitrev_load_f64.spv", __DIR__)
  @fft_stage_spv Path.expand("../../priv/shaders/fft_stage_f64.spv", __DIR__)

  @impl true
  def fft(out, tensor, opts), do: do_fft(out, tensor, opts, false)

  @impl true
  def ifft(out, tensor, opts), do: do_fft(out, tensor, opts, true)

  defp do_fft(out, tensor, opts, inverse?) do
    t = ensure_on_backend(tensor)
    length = Keyword.fetch!(opts, :length)
    axis = Keyword.fetch!(opts, :axis)
    rank = tuple_size(t.shape)

    if fft_gpu_ok?(t, out, axis, length, rank) do
      gpu_fft(out, t, length, inverse?)
    else
      op = if inverse?, do: :ifft, else: :fft
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      host_result(out, apply(Nx, op, [t_bin, opts]))
    end
  end

  # GPU path covers: last axis, no pad/slice (length == that axis's size),
  # power-of-two length >= 2, real-f64 or complex-f64 input, c128 output.
  # Everything else (other axes, padded/sliced/non-pow2 lengths, f32/int
  # inputs that map to c64) falls back to BinaryBackend, still correct.
  defp fft_gpu_ok?(%T{shape: shape, type: type} = t, %T{type: {:c, 128}}, axis, length, rank) do
    match?(%__MODULE__{}, t.data) and rank >= 1 and axis == rank - 1 and
      elem(shape, axis) == length and pow2?(length) and length >= 2 and
      type in [{:f, 64}, {:c, 128}]
  end

  defp fft_gpu_ok?(_t, _out, _axis, _length, _rank), do: false

  defp pow2?(n), do: n > 0 and Bitwise.band(n, n - 1) == 0

  defp gpu_fft(out, %T{shape: shape, type: type, data: %__MODULE__{ref: in_ref}}, length, inverse?) do
    n = length
    logn = trunc(:math.log2(n))
    batch = div(byte_size_of(shape), n)
    is_complex = if type == {:c, 128}, do: 1, else: 0
    inv = if inverse?, do: 1, else: 0
    out_bytes = batch * n * element_bytes({:c, 128})
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(out_bytes)

    :ok =
      Nx.Vulkan.NativeV.fft(
        out_ref,
        in_ref,
        n,
        logn,
        batch,
        is_complex,
        inv,
        @fft_bitrev_spv,
        @fft_stage_spv
      )

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: {:c, 128}})
  end

  @impl true
  def from_pointer(out, pointer, backend, offset, opts) do
    Nx.BinaryBackend.from_pointer(out, pointer, backend, offset, opts)
  end

  @impl true
  def to_pointer(%T{} = tensor, opts) do
    t_bin = Nx.backend_transfer(tensor, Nx.BinaryBackend)
    Nx.BinaryBackend.to_pointer(t_bin, opts)
  end

  # ---------------------------------------------------------------- reductions

  @reduce_axis_f64_spv Path.expand("../../priv/shaders/reduce_axis_f64.spv", __DIR__)
  @reduce_axis_f32_spv Path.expand("../../priv/shaders/reduce_axis_f32.spv", __DIR__)

  defp reduce_spv({:f, 64}), do: @reduce_axis_f64_spv
  defp reduce_spv({:f, 32}), do: @reduce_axis_f32_spv
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

  # Explicit `//1` step: under Elixir's post-1.16 / nx-0.13 range semantics a
  # bare `0..(n - 1)` with n == 0 becomes `0..-1` and defaults to a *descending*
  # step (yielding [0, -1]) instead of the empty list. That corrupted the reduce
  # axes for scalar/rank-0 shapes and hung do_reduce. `//1` restores the intended
  # ascending-or-empty behaviour.
  defp all_axes(shape), do: Enum.to_list(0..(tuple_size(shape) - 1)//1)

  # Classify the reduction shape:
  #   - All axes      → outer=1, reduce=product(shape), inner=1
  #   - Leading axes  → outer=1, reduce=product(reduced), inner=product(remaining)
  #   - Trailing axes → outer=product(remaining), reduce=product(reduced), inner=1
  defp classify_reduce_axes(in_shape, axes) do
    rank = tuple_size(in_shape)
    sorted = Enum.sort(axes)
    dims = Tuple.to_list(in_shape)

    cond do
      sorted == Enum.to_list(0..(rank - 1)//1) ->
        {:ok, {1, Enum.reduce(dims, 1, &Kernel.*/2), 1}}

      sorted == Enum.to_list(0..(length(sorted) - 1)//1) ->
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

  # The legacy transpose.spv is an f32 shader — it strides the buffer as
  # 4-byte floats and silently corrupts f64 data. The f64-first backend uses
  # transpose_f64.spv for the (only) GPU-accelerated case (2-D [1,0] f64);
  # every other shape/type host-falls-back.
  @transpose_f64_spv Path.expand("../../priv/shaders/transpose_f64.spv", __DIR__)
  @transpose_f32_spv Path.expand("../../priv/shaders/transpose_f32.spv", __DIR__)

  defp transpose_spv({:f, 64}), do: @transpose_f64_spv
  defp transpose_spv({:f, 32}), do: @transpose_f32_spv
  defp transpose_spv(_), do: nil

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

  # 2-D f64 transpose runs the f64 shader on the GPU. Higher-rank axis
  # permutations, non-[1,0] axes and non-f64 types host-fall-back (correct,
  # avoids the old raise and the f32-shader-on-f64 corruption).
  @impl true
  def transpose(
        %T{shape: out_shape, type: type} = out,
        %T{shape: in_shape, data: %__MODULE__{ref: a_ref}} = tensor,
        axes
      ) do
    spv = transpose_spv(type)

    if spv != nil and tuple_size(in_shape) == 2 and axes == [1, 0] do
      m = elem(in_shape, 0)
      n = elem(in_shape, 1)
      {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(m * n * element_bytes(type))

      :ok = Nx.Vulkan.NativeV.transpose_2d(out_ref, a_ref, m, n, spv)

      put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
    else
      t_bin = Nx.backend_transfer(tensor, Nx.BinaryBackend)
      host_result(out, Nx.transpose(t_bin, axes: axes))
    end
  end

  # ---------------------------------------------------------------- host-fallback ops

  # as_type — same-type is a rewrap; f32<->f64 casts run a GPU shader; other
  # dtype pairs round-trip through BinaryBackend.
  @cast_f32_to_f64_spv Path.expand("../../priv/shaders/cast_f32_to_f64.spv", __DIR__)
  @cast_f64_to_f32_spv Path.expand("../../priv/shaders/cast_f64_to_f32.spv", __DIR__)

  defp cast_spv({:f, 32}, {:f, 64}), do: @cast_f32_to_f64_spv
  defp cast_spv({:f, 64}, {:f, 32}), do: @cast_f64_to_f32_spv
  defp cast_spv(_from, _to), do: nil

  @impl true
  def as_type(%T{type: type} = out, %T{type: source_type, shape: shape, data: %__MODULE__{ref: ref}} = tensor) do
    cond do
      type == source_type ->
        put_in(out.data, %__MODULE__{ref: ref, shape: out.shape, type: type})

      cast_spv(source_type, type) != nil ->
        # f32<->f64 widening/narrowing on the GPU (mixed precision).
        n = byte_size_of(shape)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(type))
        :ok = Nx.Vulkan.NativeV.cast(out_ref, ref, n, cast_spv(source_type, type))
        put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: type})

      true ->
        bin_in = Nx.backend_transfer(tensor, Nx.BinaryBackend)
        host_result(out, Nx.as_type(bin_in, type))
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

  # select(pred, on_true, on_false) — GPU broadcast select (masking / where /
  # relu-grad) when pred is u8 and on_true/on_false/out share an f32/f64 type;
  # host fallback otherwise.
  @select_f32_spv Path.expand("../../priv/shaders/select_f32.spv", __DIR__)
  @select_f64_spv Path.expand("../../priv/shaders/select_f64.spv", __DIR__)

  defp select_spv({:f, 32}), do: @select_f32_spv
  defp select_spv({:f, 64}), do: @select_f64_spv
  defp select_spv(_), do: nil

  @impl true
  def select(%T{type: type, shape: os} = out, pred, on_true, on_false) do
    p = ensure_on_backend(pred)
    t = ensure_on_backend(on_true)
    f = ensure_on_backend(on_false)
    spv = select_spv(type)

    gpu? =
      spv != nil and p.type == {:u, 8} and t.type == type and f.type == type and
        match?(%__MODULE__{}, p.data) and match?(%__MODULE__{}, t.data) and
        match?(%__MODULE__{}, f.data) and tuple_size(os) >= 1 and tuple_size(os) <= 4

    if gpu? do
      gpu_select(out, p, t, f, spv)
    else
      pred_bin = Nx.backend_transfer(p, Nx.BinaryBackend)
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      f_bin = Nx.backend_transfer(f, Nx.BinaryBackend)
      host_result(out, with_binary_backend(fn -> Nx.select(pred_bin, t_bin, f_bin) end))
    end
  end

  defp gpu_select(out, %T{data: %__MODULE__{ref: p_ref}} = p, %T{data: %__MODULE__{ref: t_ref}} = t, %T{data: %__MODULE__{ref: f_ref}} = f, spv) do
    rank = tuple_size(out.shape)

    params =
      for v <-
            [rank] ++
              pad4(Tuple.to_list(out.shape)) ++
              pad4(pad_left(Tuple.to_list(p.shape), rank)) ++
              pad4(pad_left(Tuple.to_list(t.shape), rank)) ++
              pad4(pad_left(Tuple.to_list(f.shape), rank)),
          into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    n = byte_size_of(out.shape)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(out.type))

    :ok = Nx.Vulkan.NativeV.apply_select(out_ref, p_ref, t_ref, f_ref, params_ref, n, rank, spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: out.type})
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
    # start_indices may be dynamic (scalar tensors); they must ride to
    # BinaryBackend too, else Nx.slice calls BinaryBackend.to_binary on a
    # VulkanoBackend index tensor and crashes (found via `doctest Nx`).
    bin_in = Nx.backend_transfer(tensor, Nx.BinaryBackend)
    bin_idx = Enum.map(start_indices, &maybe_transfer_idx/1)
    bin_result = Nx.slice(bin_in, bin_idx, lengths, strides: strides)
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
      # GPU byte-append is only valid when every input already has the output
      # type — a raw concat can't cast. Mixed-type concat (e.g. f32+u8+s64) must
      # host-fall-back so Nx casts to the merged type first (found via doctest Nx).
      axis == 0 and all_vulkano?(tensors) and Enum.all?(tensors, &(&1.type == out.type)) ->
        concat_vulkano(out, tensors)

      true ->
        bins = Enum.map(tensors, &Nx.backend_transfer(ensure_on_backend(&1), Nx.BinaryBackend))
        result = with_binary_backend(fn -> Nx.concatenate(bins, axis: axis) end)
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
    # clip = min(max(t, lo), hi) — composes from our broadcast max/min, so it
    # stays on the GPU (same-type f32/f64) instead of host round-tripping. Mixed
    # types (e.g. f32 tensor, integer bounds) fall back per-op, still correct.
    t = ensure_on_backend(tensor)
    lo = ensure_on_backend(min)
    hi = ensure_on_backend(max)
    result = Nx.min(Nx.max(t, lo), hi)
    host_result(out, result)
  end

  # --- Tier 1 parity batch — host-fallback Nx.Backend callbacks ---
  # Each downloads to BinaryBackend, invokes Nx.<op>, returns on
  # BinaryBackend (host_result contract). No GPU shader yet; promotable
  # later when profiling justifies. (nx 0.13 note: all_close, logical_not,
  # cumulative_*, top_k, take_along_axis and the small-linalg family are no
  # longer Nx.Backend callbacks — Nx routes them through block/4 or composes
  # them from primitives, so they need no explicit clause here.)

  # product: multiplicative reduction. Same shape as sum but with *.
  @impl true
  def product(out, tensor, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.product(t_bin, opts)
    host_result(out, result)
  end

  # reverse: reverse along given axes. Composes from slice in some
  # cases but a direct callback handles general patterns.
  @impl true
  def reverse(out, tensor, axes) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.reverse(t_bin, axes: axes)
    host_result(out, result)
  end

  # sort / argsort — sort family (both still Nx.Backend callbacks)
  @impl true
  def sort(out, tensor, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.sort(t_bin, opts)
    host_result(out, result)
  end

  @impl true
  def argsort(out, tensor, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.argsort(t_bin, opts)
    host_result(out, result)
  end

  # bitcast: reinterpret bytes as different type without conversion
  @impl true
  def bitcast(out, tensor) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.bitcast(t_bin, out.type)
    host_result(out, result)
  end

  # to_batched: split leading axis into chunks. Returns a stream of tensors.
  # nx 0.13 encodes the batch size in the `out` template's leading dim (opts
  # carries only :leftover), so derive it from there — reading opts[:batch_size]
  # yields nil and crashes Nx.to_batched/3.
  @impl true
  def to_batched(%T{shape: out_shape}, tensor, opts) do
    batch_size = elem(out_shape, 0)
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    Nx.to_batched(t_bin, batch_size, opts)
  end

  # --- linalg: triangular_solve is the only remaining Nx.Backend callback ---
  # nx 0.13 dropped cholesky/determinant/solve/qr/lu/svd/eigh from the
  # behaviour and routes each through the block/4 callback (a Nx.Block.LinAlg.*
  # struct); our block/4 transfers to BinaryBackend, so those ops need no
  # explicit clause here. triangular_solve stayed a callback.
  @impl true
  def triangular_solve(out, a, b, opts) do
    a_bin = Nx.backend_transfer(ensure_on_backend(a), Nx.BinaryBackend)
    b_bin = Nx.backend_transfer(ensure_on_backend(b), Nx.BinaryBackend)
    result = Nx.LinAlg.triangular_solve(a_bin, b_bin, opts)
    host_result(out, result)
  end

  # --- Round 2: window family (7 callbacks) ---
  # All delegate to BinaryBackend's window ops via Nx.<op>.

  @impl true
  def window_sum(out, tensor, dimensions, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.window_sum(t_bin, dimensions, opts)
    host_result(out, result)
  end

  @impl true
  def window_product(out, tensor, dimensions, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.window_product(t_bin, dimensions, opts)
    host_result(out, result)
  end

  @impl true
  def window_max(out, tensor, dimensions, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.window_max(t_bin, dimensions, opts)
    host_result(out, result)
  end

  @impl true
  def window_min(out, tensor, dimensions, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.window_min(t_bin, dimensions, opts)
    host_result(out, result)
  end

  @impl true
  def window_reduce(out, tensor, acc, dimensions, opts, fun) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    acc_bin = Nx.backend_transfer(ensure_on_backend(acc), Nx.BinaryBackend)
    result = Nx.window_reduce(t_bin, acc_bin, dimensions, opts, fun)
    host_result(out, result)
  end

  @impl true
  def window_scatter_max(out, tensor, source, init_value, dimensions, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    s_bin = Nx.backend_transfer(ensure_on_backend(source), Nx.BinaryBackend)
    iv_bin = Nx.backend_transfer(ensure_on_backend(init_value), Nx.BinaryBackend)
    result = with_binary_backend(fn -> Nx.window_scatter_max(t_bin, s_bin, iv_bin, dimensions, opts) end)
    host_result(out, result)
  end

  @impl true
  def window_scatter_min(out, tensor, source, init_value, dimensions, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    s_bin = Nx.backend_transfer(ensure_on_backend(source), Nx.BinaryBackend)
    iv_bin = Nx.backend_transfer(ensure_on_backend(init_value), Nx.BinaryBackend)
    result = with_binary_backend(fn -> Nx.window_scatter_min(t_bin, s_bin, iv_bin, dimensions, opts) end)
    host_result(out, result)
  end

  # --- Round 2: generic reduce (1 callback) ---
  # User-supplied function runs on BinaryBackend tensors.
  @impl true
  def reduce(out, tensor, acc, opts, fun) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    acc_bin = Nx.backend_transfer(ensure_on_backend(acc), Nx.BinaryBackend)
    result = Nx.reduce(t_bin, acc_bin, opts, fun)
    host_result(out, result)
  end

  # Indices passed to put_slice can be scalar tensors or integers;
  # normalise to whatever BinaryBackend.put_slice expects.
  defp maybe_transfer_idx(%T{} = t),
    do: Nx.backend_transfer(ensure_on_backend(t), Nx.BinaryBackend)

  defp maybe_transfer_idx(i) when is_integer(i), do: i

  # ---------------------------------------------------------------- linalg

  @matmul_f64_spv Path.expand("../../priv/shaders/matmul_f64.spv", __DIR__)
  # f32 matmul keeps data in f32; the accumulator width is a policy (see
  # F32_PLAN.md). Default :f64 matches a f64-accumulating reference to f32
  # round-off; :f32 is 1.4-1.7x faster on f64-rate-limited GPUs (Kepler/consumer
  # Ampere) at the cost of precision that degrades with K. Opt into the f32 path
  # via the tensor's dtype; f64 storage stays the default.
  @matmul_f32_f64acc_spv Path.expand("../../priv/shaders/matmul_f32_f64acc.spv", __DIR__)
  @matmul_f32_f32acc_spv Path.expand("../../priv/shaders/matmul_f32_f32acc.spv", __DIR__)

  @doc """
  Accumulator width for the f32 GPU GEMM path — governs both `dot`/matmul **and**
  conv's GEMM: `:f64` (default, accuracy-safe) or `:f32` (faster on f64-rate-
  limited GPUs, precision degrades ~√K). Set with `put_f32_matmul_accumulator/1`
  or `config :nx_vulkan, :f32_matmul_accumulator`.
  """
  def f32_matmul_accumulator, do: Application.get_env(:nx_vulkan, :f32_matmul_accumulator, :f64)

  @doc "Set the f32 GEMM accumulator policy (`:f64` | `:f32`). See `f32_matmul_accumulator/0`."
  def put_f32_matmul_accumulator(width) when width in [:f32, :f64] do
    Application.put_env(:nx_vulkan, :f32_matmul_accumulator, width)
  end

  defp matmul_spv({:f, 64}), do: @matmul_f64_spv

  defp matmul_spv({:f, 32}) do
    case f32_matmul_accumulator() do
      :f32 -> @matmul_f32_f32acc_spv
      _ -> @matmul_f32_f64acc_spv
    end
  end

  defp matmul_spv(_), do: nil

  # Dot product (matmul) — Nx callback signature:
  #   dot(out, a, contracting_axes_a, batched_axes_a,
  #            b, contracting_axes_b, batched_axes_b)
  #
  # Fast path: rank-2 × rank-2, contracting [1] of a vs [0] of b
  # (standard matmul A·B). f64 and f32 (f64-accumulator) run native shaders;
  # everything else routes through BinaryBackend.
  @impl true
  def dot(%T{shape: out_shape, type: type} = out, a, axes_a, batched_a, b, axes_b, batched_b) do
    a_v = ensure_on_backend(a)
    b_v = ensure_on_backend(b)
    spv = matmul_spv(type)

    fast_path =
      spv != nil and a_v.type == type and b_v.type == type and
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

      :ok = Nx.Vulkan.NativeV.matmul(out_ref, a_ref, b_ref, m, n, k_a, spv)

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

  # Run a composed Nx fallback with BinaryBackend as the *process default* so any
  # intermediate tensors Nx materialises inside the composition (constants, iota,
  # broadcasts) land on BinaryBackend. Without this, when VulkanoBackend is the
  # default backend (the normal way the backend is used), those intermediates
  # leak onto VulkanoBackend and Nx crashes mixing them with our BinaryBackend
  # inputs. Surfaced by `doctest Nx` (window_scatter_*, reflect, …).
  defp with_binary_backend(fun) do
    prev = Nx.default_backend(Nx.BinaryBackend)

    try do
      fun.()
    after
      Nx.default_backend(prev)
    end
  end

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
  defp element_bytes({:f, 16}), do: 2
  defp element_bytes({:f, 8}), do: 1
  defp element_bytes({:bf, 16}), do: 2
  defp element_bytes({:c, 64}), do: 8
  defp element_bytes({:c, 128}), do: 16

  defp encode_scalar(s, {:f, 16}), do: <<s / 1.0::float-16-native>>
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
  # bf16 / f8 / complex: no correct native bitstring encoder (bf16 ≠ IEEE f16),
  # so signal fallback — constant/3 builds these via BinaryBackend.
  defp encode_scalar(_s, _type), do: :error
end
