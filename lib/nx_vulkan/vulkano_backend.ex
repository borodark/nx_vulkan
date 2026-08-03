defmodule Nx.Vulkan.VulkanoBackend do
  @moduledoc """
  Pure-Rust (vulkano) `Nx.Backend` implementation.

  Tensors are represented by:

      %Nx.Vulkan.VulkanoBackend{ref: ResourceArc<VulkanoTensor>,
                                shape: tuple, type: {kind, bits}}

  The `ref` is a Rustler resource owning an `Arc<Subbuffer<u8>>` in
  vulkano. When the BEAM GCs the Elixir reference, vulkano's `Drop`
  runs `vkDestroyBuffer` + `vkFreeMemory`. Stale-handle bugs (where
  a freed `VkBuf*` is read back at the C++ layer) are structurally
  impossible: the `Subbuffer` cannot outlive its `Buffer`.

  ## Dtypes

  Native **f32 and f64** shaders for the hot ops; the tensor's dtype picks the
  SPIR-V module at dispatch time. f64 is the
  default accumulator policy because correctness came first; f32 wins on
  bandwidth-bound work and is roughly 32× the rate for `dot` on consumer NVIDIA
  cards. Other dtypes (integers, u8, …) take the host fallback below.

  ## Coverage

  Storage callbacks: `init/1`, `from_binary/3`, `to_binary/2`,
  `backend_copy/3`, `backend_transfer/3`, `backend_deallocate/1`,
  `inspect/2`, `constant/3`, `iota/3`, `eye/2`.

  Dispatched to the GPU as native shaders:

    - elementwise binary — `add`, `multiply`, `subtract`, `divide`, `pow`,
      `max`, `min`, plus a rank-≤4 broadcasting variant that keeps bias-add,
      scaling, and relu-via-max resident instead of host-falling-back
    - elementwise unary — `exp`, `log`, `sqrt`, `abs`, `negate`, `sigmoid`,
      `tanh`, `floor`, `ceil`, `sign`
    - `dot`, `conv` (im2col + GEMM), `transpose`
    - reductions — `sum`, `reduce_max`, `reduce_min`

  Everything else Nx asks for is still implemented, via a host fallback that
  reads the tensor back, computes on `Nx.BinaryBackend`, and returns the result
  to the GPU — `argsort`, `fft`, `triangular_solve`, the window ops, the trig
  and bitwise families, and so on. Unsupported here means *slower*, not broken:
  every `Nx` callback returns a correct result on this backend.

  ## Eager vs. fused

  Used directly, this backend is **eager** — one dispatch per op, with an
  intermediate buffer between each. To fuse a chain of ops into a single shader
  and keep intermediates on-device, jit through `Nx.Vulkan.Compiler` instead.
  """

  @behaviour Nx.Backend

  @enforce_keys [:ref, :shape, :type]
  defstruct [:ref, :shape, :type]

  alias Nx.Tensor, as: T

  # Wrap a host-computed result back into `out`, and record the host fallback
  # against the backend callback that performed it.
  #
  # A macro so the callback's {name, arity} is captured at compile time. It
  # cannot be recovered at runtime: every fallback path calls this in tail
  # position, so TCO has already dropped the caller's frame by the time the
  # counter runs. See Nx.Vulkan.Fallback for why counting these matters at all.
  #
  # Must stay above its first use — macros are not available before definition.
  defmacrop host_result(out, result) do
    op = __CALLER__.function

    quote do
      host_result_recorded(unquote(out), unquote(result), unquote(Macro.escape(op)))
    end
  end

  # Explicit-attribution form. The elementwise/reduce helpers are shared by
  # dozens of callbacks, so __CALLER__.function would blame the helper and hide
  # WHICH op fell back — they pass the real op instead.
  defmacrop host_result(out, result, op) do
    quote do
      host_result_recorded(unquote(out), unquote(result), unquote(op))
    end
  end

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

        if bspv != nil and unquote(code) != 4 and bcast_shape_ok?(a_v, b_v, out) do
          # coerce mismatched-dtype operands (e.g. f32 scalar with f64 tensor)
          # to the output type on the GPU, else fall back.
          case {coerce_to(a_v, type), coerce_to(b_v, type)} do
            {%T{} = ca, %T{} = cb} -> gpu_bcast_binary(out, ca, cb, unquote(code), bspv)
            _ -> binary_op_host_fallback(unquote(op), out, a_v, b_v)
          end
        else
          binary_op_host_fallback(unquote(op), out, a_v, b_v)
        end
      end
    end
  end

  # Broadcast GPU path is valid when both operands are on this backend and the
  # output rank is 1..4 (Nx guarantees a,b broadcast to out.shape). Dtype
  # mismatches are handled by coerce_to; pow is excluded above (fp64 has no pow).
  defp bcast_shape_ok?(%T{} = a, %T{} = b, %T{shape: os}) do
    match?(%__MODULE__{}, a.data) and match?(%__MODULE__{}, b.data) and
      tuple_size(os) >= 1 and tuple_size(os) <= 4
  end

  defp bcast_shape_ok?(_a, _b, _out), do: false

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
    host_result(out, result, {op, 3})
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
    host_result(out, result, {op, 2})
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
    i = ensure_on_backend(inp) |> coerce_operand(out.type)
    k = ensure_on_backend(kernel) |> coerce_operand(out.type)

    cond do
      conv_gpu_ok?(i, k, out, opts) ->
        gpu_conv(out, i, k, opts)

      conv_gpu_permuted_ok?(i, k, out, opts) ->
        permuted_gpu_conv(out, i, k, opts)

      true ->
        inp_bin = Nx.backend_transfer(i, Nx.BinaryBackend)
        kernel_bin = Nx.backend_transfer(k, Nx.BinaryBackend)
        # BinaryBackend.conv calls the high-level Nx.pad internally; with the
        # process default backend still VulkanoBackend that would dispatch to our
        # GPU pad and hand conv a Vulkan tensor it then can't to_binary. Pin the
        # default to BinaryBackend for the whole composed op.
        host_result(out, with_binary_backend(fn -> Nx.conv(inp_bin, kernel_bin, opts) end))
    end
  end

  # GPU path covers: spatial rank 1..3, feature/batch groups == 1, f64 or f32
  # input/kernel/output (all three must match). Any strides, padding and
  # input/kernel dilation are honoured (folded into the im2col index math).
  # Non-identity permutations are rotated into the native layout on-device at
  # rank <= 4 (see permuted_gpu_conv/4). Groups > 1, mixed/other dtypes, and
  # rank-5 permuted convs fall back to the host.
  # Everything the native im2col+GEMM path needs EXCEPT a canonical layout.
  defp conv_gpu_core_ok?(%T{shape: ishape} = i, %T{} = k, %T{type: ot}, opts) do
    rank = tuple_size(ishape)
    sr = rank - 2

    match?(%__MODULE__{}, i.data) and match?(%__MODULE__{}, k.data) and
      i.type == ot and k.type == ot and ot in [{:f, 64}, {:f, 32}] and
      sr >= 1 and sr <= 3 and
      Keyword.get(opts, :feature_group_size, 1) == 1 and
      Keyword.get(opts, :batch_group_size, 1) == 1
  end

  defp conv_gpu_core_ok?(_i, _k, _out, _opts), do: false

  # Cast an operand to an op's output dtype on-device.
  #
  # `Nx.Defn.Grad` seeds a gradient at Nx's *default* dtype, so the backward
  # ops of a uniformly-f64 model routinely arrive as f64 x f32 — true of both
  # conv and dot. The native shaders need a single dtype, and rejecting the
  # mismatch dropped the op (the expensive half of a training step) onto the
  # host. The f32<->f64 cast is one shader (coerce_to/2), so paying it beats
  # leaving.
  #
  # Semantics are unchanged: BinaryBackend also converts its inputs to the
  # output type, and `out.type` is the type Nx computed for the op. Anything
  # coerce_to/2 can't cast (integers, tensors not on this backend) is returned
  # untouched and handled by the gate as before.
  defp coerce_operand(%T{type: ot} = tensor, ot), do: tensor

  defp coerce_operand(%T{data: %__MODULE__{}} = tensor, ot) do
    case coerce_to(tensor, ot) do
      nil -> tensor
      coerced -> coerced
    end
  end

  defp coerce_operand(tensor, _ot), do: tensor


  # Already in the native layout — dispatch straight to the shaders.
  defp conv_gpu_ok?(%T{shape: ishape} = i, %T{shape: kshape} = k, out, opts) do
    rank = tuple_size(ishape)

    conv_gpu_core_ok?(i, k, out, opts) and
      identity_perm?(opts[:input_permutation], rank) and
      identity_perm?(opts[:kernel_permutation], tuple_size(kshape)) and
      identity_perm?(opts[:output_permutation], rank)
  end

  defp conv_gpu_ok?(_i, _k, _out, _opts), do: false

  # Non-canonical layout, but one we can rotate into place on-device. This is
  # the conv BACKWARD case: `Nx.Defn.Grad` builds its backward convolutions with
  # `conv_spec_transpose/1`, which swaps the first two axes, so every gradient
  # conv arrives with non-identity permutations. Transposing around the native
  # path costs three permuted copies and keeps the whole thing on the GPU;
  # refusing it used to drop the entire backward pass onto
  # `Nx.BinaryBackend.conv`, which dominated CNN training time.
  #
  # Bounded to rank <= 4 because that is what the transpose shader handles
  # (rank 5 = 3 spatial dims still host-falls-back).
  defp conv_gpu_permuted_ok?(%T{shape: ishape} = i, %T{shape: kshape} = k, out, opts) do
    conv_gpu_core_ok?(i, k, out, opts) and
      tuple_size(ishape) <= 4 and tuple_size(kshape) <= 4
  end

  defp conv_gpu_permuted_ok?(_i, _k, _out, _opts), do: false

  defp identity_perm?(nil, _rank), do: true
  defp identity_perm?(perm, rank), do: perm == Enum.to_list(0..(rank - 1)//1)

  defp identity_axes(rank), do: Enum.to_list(0..(rank - 1)//1)

  defp invert_perm(perm),
    do: perm |> Enum.with_index() |> Enum.sort() |> Enum.map(&elem(&1, 1))

  defp permute_shape(shape, perm),
    do: perm |> Enum.map(&elem(shape, &1)) |> List.to_tuple()

  # Transpose inputs into the native layout, run the native conv, rotate the
  # result back. Mirrors Nx.BinaryBackend.conv/4's own handling: it transposes
  # t/k by the raw input/kernel permutations, computes into a shape given by
  # `out.shape` permuted by the raw output permutation, then transposes that
  # result by the INVERSE of the raw output permutation.
  defp permuted_gpu_conv(%T{shape: out_shape} = out, i, k, opts) do
    rank = tuple_size(i.shape)
    krank = tuple_size(k.shape)

    inp_perm = opts[:input_permutation] || identity_axes(rank)
    ker_perm = opts[:kernel_permutation] || identity_axes(krank)
    out_perm = opts[:output_permutation] || identity_axes(rank)

    i2 = Nx.transpose(i, axes: inp_perm)
    k2 = Nx.transpose(k, axes: ker_perm)

    canon_out = %{
      out
      | shape: permute_shape(out_shape, out_perm),
        names: List.duplicate(nil, rank)
    }

    canon_opts =
      opts
      |> Keyword.put(:input_permutation, identity_axes(rank))
      |> Keyword.put(:kernel_permutation, identity_axes(krank))
      |> Keyword.put(:output_permutation, identity_axes(rank))

    result =
      canon_out
      |> gpu_conv(i2, k2, canon_opts)
      |> Nx.transpose(axes: invert_perm(out_perm))

    %{out | data: result.data}
  end

  defp gpu_conv(
         %T{type: type, shape: oshape} = out,
         %T{shape: ishape, data: %__MODULE__{ref: in_ref}},
         %T{shape: kshape, data: %__MODULE__{ref: k_ref}},
         opts
       ) do
    p = conv_plan(type, ishape, kshape, oshape, opts)
    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(p.params_bin)
    {:ok, col_ref} = Nx.Vulkan.NativeV.buf_alloc(p.m * p.k_cols * p.ebytes)

    :ok =
      Nx.Vulkan.NativeV.conv_im2col(
        col_ref,
        in_ref,
        params_ref,
        p.n,
        p.cin,
        p.o_total,
        p.k_total,
        p.k_cols,
        p.im2col_spv
      )

    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(p.n * p.cout * p.o_total * p.ebytes)

    :ok =
      Nx.Vulkan.NativeV.conv_gemm(
        out_ref,
        col_ref,
        k_ref,
        p.n,
        p.cout,
        p.o_total,
        p.k_cols,
        p.gemm_spv
      )

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: type})
  end

  @doc false
  # Shared conv geometry: resolves shapes + fully-materialised opts into the
  # buffer dims, SPV paths and the packed int params blob consumed by
  # conv_im2col / conv_gemm. Pure — no device calls — so the Nx.Defn multi-stage
  # compiler can compute it once at compile time and reuse it, while the eager
  # backend calls it per dispatch. `oshape` is the output-permutation-layout
  # shape (identity perms only reach here). Reads the f32 accumulator policy at
  # call time via conv_spvs/1.
  def conv_plan(type, ishape, kshape, oshape, opts) do
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
    odims = spatial.(oshape)
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

    o_total = Enum.product(odims)
    k_total = Enum.product(kdims)

    %{
      im2col_spv: im2col_spv,
      gemm_spv: gemm_spv,
      ebytes: ebytes,
      params_bin: params_bin,
      n: n,
      cin: cin,
      cout: cout,
      o_total: o_total,
      k_total: k_total,
      k_cols: cin * k_total,
      m: n * o_total
    }
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

  # Generic rank<=4 permuted transpose. The rank-2/[1,0] shader above is a
  # tiled special case and stays the fast path for matrices; this one handles
  # every other permutation that used to host-fall-back — notably the
  # first-two-axes swap Nx's conv gradient emits.
  @transpose_nd_f64_spv Path.expand("../../priv/shaders/transpose_nd_f64.spv", __DIR__)
  @transpose_nd_f32_spv Path.expand("../../priv/shaders/transpose_nd_f32.spv", __DIR__)

  defp transpose_nd_spv({:f, 64}), do: @transpose_nd_f64_spv
  defp transpose_nd_spv({:f, 32}), do: @transpose_nd_f32_spv
  defp transpose_nd_spv(_), do: nil

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
    nd_spv = transpose_nd_spv(type)
    rank = tuple_size(in_shape)

    cond do
      spv != nil and rank == 2 and axes == [1, 0] ->
        m = elem(in_shape, 0)
        n = elem(in_shape, 1)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(m * n * element_bytes(type))

        :ok = Nx.Vulkan.NativeV.transpose_2d(out_ref, a_ref, m, n, spv)

        put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})

      nd_spv != nil and rank <= 4 ->
        gpu_transpose_nd(out, tensor, axes, nd_spv)

      true ->
        t_bin = Nx.backend_transfer(tensor, Nx.BinaryBackend)
        host_result(out, Nx.transpose(t_bin, axes: axes))
    end
  end

  # params: [rank, in[4], out[4], perm[4]] — shapes left-aligned, padded to 4.
  defp gpu_transpose_nd(
         %T{shape: out_shape, type: type} = out,
         %T{shape: in_shape, data: %__MODULE__{ref: a_ref}},
         axes,
         spv
       ) do
    rank = tuple_size(in_shape)
    n = Nx.size(in_shape)

    params =
      for v <-
            [rank] ++
              pad4(Tuple.to_list(in_shape)) ++
              pad4(Tuple.to_list(out_shape)) ++
              pad4(axes),
          into: <<>>,
          do: <<v::signed-32-little>>

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(type))

    :ok = Nx.Vulkan.NativeV.transpose_nd(out_ref, a_ref, params_ref, n, rank, spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
  end

  # ---------------------------------------------------------------- host-fallback ops

  # as_type — same-type is a rewrap; f32<->f64 casts run a GPU shader; other
  # dtype pairs round-trip through BinaryBackend.
  @cast_f32_to_f64_spv Path.expand("../../priv/shaders/cast_f32_to_f64.spv", __DIR__)
  @cast_f64_to_f32_spv Path.expand("../../priv/shaders/cast_f64_to_f32.spv", __DIR__)

  # Integer -> float. Nx materialises literals as {:s, 32} and then broadcasts
  # them, so a relu gradient's select(x > 0, g, 0) hands us a full s32 TENSOR,
  # not a scalar the rank-0 clause in coerce_to/2 could rebuild. Without this
  # path that select host-fell-back with its whole tensor.
  @cast_s32_to_f32_spv Path.expand("../../priv/shaders/cast_s32_to_f32.spv", __DIR__)
  @cast_s32_to_f64_spv Path.expand("../../priv/shaders/cast_s32_to_f64.spv", __DIR__)

  defp cast_spv({:f, 32}, {:f, 64}), do: @cast_f32_to_f64_spv
  defp cast_spv({:f, 64}, {:f, 32}), do: @cast_f64_to_f32_spv
  defp cast_spv({:s, 32}, {:f, 32}), do: @cast_s32_to_f32_spv
  defp cast_spv({:s, 32}, {:f, 64}), do: @cast_s32_to_f64_spv
  defp cast_spv(_from, _to), do: nil

  # Coerce an on-GPU tensor to `to` type via the f32<->f64 cast shader so
  # mixed-dtype ops (e.g. f64 tensor + f32 scalar literal) stay on the GPU.
  # Returns the coerced %T{} (a no-op when already `to`), or nil when the pair
  # can't be cast on the GPU (non-f32/f64, or not on this backend).
  defp coerce_to(%T{type: to} = t, to), do: t

  # A rank-0 constant of a type the cast shader cannot convert.
  #
  # Nx materialises literals as {:s, 32}: relu is `max(x, 0)`, a mean divides by
  # an integer count. The f32<->f64 cast shader has no integer path, so
  # coerce_to/2 returned nil and the op host-fell-back — a FOUR-BYTE literal
  # dragging a {32, 8, 14, 14} tensor to the CPU with it. In a training step
  # that was max/3 x2, divide/3 x1 and greater/3 x2.
  #
  # Rebuilding the scalar at the target type is one 4-byte round trip, which is
  # nothing against moving the tensor. backend_copy (not transfer) because the
  # source may still be referenced elsewhere in the graph. Non-scalars of an
  # uncastable type still fall back — converting those deserves a real shader.
  defp coerce_to(%T{shape: {}} = t, to) do
    t
    |> Nx.backend_copy(Nx.BinaryBackend)
    |> Nx.as_type(to)
    |> Nx.backend_transfer(__MODULE__)
  end

  defp coerce_to(%T{type: from, shape: shape, data: %__MODULE__{ref: ref}} = t, to) do
    case cast_spv(from, to) do
      nil ->
        nil

      spv ->
        n = byte_size_of(shape)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(to))
        :ok = Nx.Vulkan.NativeV.cast(out_ref, ref, n, spv)
        %{t | type: to, data: %__MODULE__{ref: out_ref, shape: shape, type: to}}
    end
  end

  defp coerce_to(_t, _to), do: nil

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

  # Comparison ops — GPU broadcast -> u8 (packed as u32 words in the shader) when
  # both operands share an f32/f64 type; host fallback otherwise. Same-type
  # f32 comparisons (e.g. x > 0.0) keep the relu-grad mask on the GPU.
  @compare_ops [equal: 0, not_equal: 1, less: 2, less_equal: 3, greater: 4, greater_equal: 5]
  @compare_f32_spv Path.expand("../../priv/shaders/compare_f32.spv", __DIR__)
  @compare_f64_spv Path.expand("../../priv/shaders/compare_f64.spv", __DIR__)

  defp compare_spv({:f, 32}), do: @compare_f32_spv
  defp compare_spv({:f, 64}), do: @compare_f64_spv
  defp compare_spv(_), do: nil

  for {op, code} <- @compare_ops do
    @impl true
    def unquote(op)(out, a, b) do
      a_v = ensure_on_backend(a)
      b_v = ensure_on_backend(b)
      # comparison happens at the merged input type; coerce both operands to it
      # (handles f64 tensor vs f32 scalar) then compare -> u8.
      merged = Nx.Type.merge(a_v.type, b_v.type)
      spv = compare_spv(merged)

      cast =
        if spv != nil and match?(%__MODULE__{}, a_v.data) and match?(%__MODULE__{}, b_v.data) and
             tuple_size(out.shape) >= 1 and tuple_size(out.shape) <= 4 do
          {coerce_to(a_v, merged), coerce_to(b_v, merged)}
        end

      case cast do
        {%T{} = ca, %T{} = cb} ->
          gpu_compare(out, ca, cb, unquote(code), spv)

        _ ->
          a_bin = Nx.backend_transfer(a_v, Nx.BinaryBackend)
          b_bin = Nx.backend_transfer(b_v, Nx.BinaryBackend)
          host_result(out, apply(Nx, unquote(op), [a_bin, b_bin]))
      end
    end
  end

  defp gpu_compare(out, %T{data: %__MODULE__{ref: a_ref}} = a, %T{data: %__MODULE__{ref: b_ref}} = b, code, spv) do
    rank = tuple_size(out.shape)

    params =
      for v <-
            [rank] ++
              pad4(Tuple.to_list(out.shape)) ++
              pad4(pad_left(Tuple.to_list(a.shape), rank)) ++
              pad4(pad_left(Tuple.to_list(b.shape), rank)),
          into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    n = byte_size_of(out.shape)
    # u8 output written as u32 words — pad the buffer to a 4-byte multiple.
    padded = div(n + 3, 4) * 4
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(padded)

    :ok = Nx.Vulkan.NativeV.apply_compare(out_ref, a_ref, b_ref, params_ref, n, rank, code, spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: out.type})
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

    shape_ok? =
      spv != nil and p.type == {:u, 8} and match?(%__MODULE__{}, p.data) and
        match?(%__MODULE__{}, t.data) and match?(%__MODULE__{}, f.data) and
        tuple_size(os) >= 1 and tuple_size(os) <= 4

    # coerce the branches to the output type on the GPU (handles a f32 scalar 0.0
    # against an f64 tensor); pred stays u8.
    branches = if shape_ok?, do: {coerce_to(t, type), coerce_to(f, type)}

    case branches do
      {%T{} = ct, %T{} = cf} ->
        gpu_select(out, p, ct, cf, spv)

      _ ->
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
  @slice_spv Path.expand("../../priv/shaders/slice.spv", __DIR__)

  def slice(out, tensor, start_indices, lengths, strides) do
    t = ensure_on_backend(tensor)
    eb = element_bytes(t.type)
    rank = tuple_size(t.shape)

    # GPU strided copy when starts are static integers, the dtype is 4/8-byte,
    # and rank 1..4. Dynamic (tensor) starts, sub-word dtypes and higher rank
    # host-fall-back. (Dynamic starts must transfer to BinaryBackend too, else
    # Nx.slice calls BinaryBackend.to_binary on a VulkanoBackend index — a bug
    # found via `doctest Nx`.)
    if Enum.all?(start_indices, &is_integer/1) and rem(eb, 4) == 0 and
         match?(%__MODULE__{}, t.data) and rank >= 1 and rank <= 4 do
      gpu_slice(out, t, start_indices, strides, eb)
    else
      bin_in = Nx.backend_transfer(t, Nx.BinaryBackend)
      bin_idx = Enum.map(start_indices, &maybe_transfer_idx/1)
      host_result(out, Nx.slice(bin_in, bin_idx, lengths, strides: strides))
    end
  end

  defp gpu_slice(out, %T{shape: sshape, data: %__MODULE__{ref: in_ref}}, starts, strides, eb) do
    rank = tuple_size(out.shape)
    ews = div(eb, 4)

    params =
      for v <-
            [rank, ews] ++
              pad4(Tuple.to_list(sshape)) ++
              pad4(Tuple.to_list(out.shape)) ++
              pad4(starts) ++
              pad4(strides),
          into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    n = byte_size_of(out.shape)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * eb)

    :ok = Nx.Vulkan.NativeV.apply_slice(out_ref, in_ref, params_ref, n, rank, @slice_spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: out.type})
  end

  # ---------------------------------------------------------------- sampler-path host fallbacks

  @pad_spv Path.expand("../../priv/shaders/pad.spv", __DIR__)

  # Nx.Backend.pad/4 callback. GPU path: a type-generic copy that maps each
  # output element back through the per-axis {low, high, interior} config —
  # elements in an edge pad, an interior gap, or outside the source get the pad
  # value (shader handles negative low/high cropping). Runs for 4/8-byte dtypes,
  # rank 1..4, scalar same-type pad value. Everything else host-falls-back.
  @impl true
  def pad(out, tensor, pad_value, padding_config) do
    t = ensure_on_backend(tensor)
    pv = ensure_on_backend(pad_value)
    eb = element_bytes(t.type)
    rank = tuple_size(t.shape)

    if match?(%__MODULE__{}, t.data) and rem(eb, 4) == 0 and rank >= 1 and rank <= 4 and
         pv.type == t.type and tuple_size(pv.shape) == 0 do
      gpu_pad(out, t, pv, padding_config, eb)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      pv_bin = Nx.backend_transfer(pv, Nx.BinaryBackend)
      host_result(out, Nx.pad(t_bin, pv_bin, padding_config))
    end
  end

  defp gpu_pad(out, %T{shape: sshape, data: %__MODULE__{ref: in_ref}}, pv, padding_config, eb) do
    rank = tuple_size(out.shape)
    ews = div(eb, 4)
    lows = Enum.map(padding_config, fn {lo, _hi, _int} -> lo end)
    interiors = Enum.map(padding_config, fn {_lo, _hi, int} -> int end)

    params =
      for v <-
            [rank, ews] ++
              pad4(Tuple.to_list(sshape)) ++
              pad4(Tuple.to_list(out.shape)) ++
              pad4(lows) ++
              pad4(interiors),
          into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)

    pv_bin = Nx.backend_transfer(pv, Nx.BinaryBackend) |> Nx.to_binary()
    {:ok, padval_ref} = Nx.Vulkan.NativeV.buf_upload(pv_bin)

    n = byte_size_of(out.shape)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * eb)

    :ok = Nx.Vulkan.NativeV.apply_pad(out_ref, in_ref, params_ref, padval_ref, n, rank, @pad_spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: out.type})
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
  @gather_spv Path.expand("../../priv/shaders/gather.spv", __DIR__)

  def gather(out, tensor, indices, opts \\ []) do
    t = ensure_on_backend(tensor)
    idx = ensure_on_backend(indices)
    ishape = t.shape
    rank = tuple_size(ishape)
    idx_rank = tuple_size(idx.shape)
    k = if idx_rank > 0, do: elem(idx.shape, idx_rank - 1), else: 0

    axes =
      case opts[:axes] do
        nil -> if k > 0, do: Enum.to_list(0..(k - 1)), else: []
        given -> Nx.Shape.normalize_axes(ishape, given, t.names)
      end

    eb = element_bytes(t.type)
    ib = element_bytes(idx.type)

    # GPU path: the indexed axes are a leading prefix [0..K-1] (no transpose
    # needed — includes the default all-axes gather), value + index dtypes are
    # 4/8-byte, rank 1..4, both operands GPU-resident.
    if match?(%__MODULE__{}, t.data) and match?(%__MODULE__{}, idx.data) and
         axes == Enum.to_list(0..(k - 1)) and rem(eb, 4) == 0 and rem(ib, 4) == 0 and
         rank >= 1 and rank <= 4 and k >= 1 do
      gpu_gather(out, t, idx, k, eb, ib)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      i_bin = Nx.backend_transfer(idx, Nx.BinaryBackend)
      host_result(out, with_binary_backend(fn -> Nx.gather(t_bin, i_bin, opts) end))
    end
  end

  defp gpu_gather(out, %T{shape: sshape, data: %__MODULE__{ref: in_ref}}, idx, k, eb, ib) do
    dims = Tuple.to_list(sshape)
    ews = div(eb, 4)
    idx_words = div(ib, 4)
    # count = product of the trailing (non-indexed) dims; per-leading-axis
    # stride = product of dims after that axis (row-major).
    count = dims |> Enum.drop(k) |> Enum.reduce(1, &(&1 * &2))

    strides =
      for j <- 0..(k - 1) do
        dims |> Enum.drop(j + 1) |> Enum.reduce(1, &(&1 * &2))
      end

    params =
      for v <- [k, ews, idx_words, count] ++ pad4(strides), into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    n = byte_size_of(out.shape)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * eb)
    %__MODULE__{ref: idx_ref} = idx.data

    :ok = Nx.Vulkan.NativeV.apply_gather(out_ref, in_ref, idx_ref, params_ref, n, k, @gather_spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: out.type})
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
  @reverse_nd_f64_spv Path.expand("../../priv/shaders/reverse_nd_f64.spv", __DIR__)
  @reverse_nd_f32_spv Path.expand("../../priv/shaders/reverse_nd_f32.spv", __DIR__)

  defp reverse_nd_spv({:f, 64}), do: @reverse_nd_f64_spv
  defp reverse_nd_spv({:f, 32}), do: @reverse_nd_f32_spv
  defp reverse_nd_spv(_), do: nil

  @impl true
  def reverse(%T{shape: shape, type: type} = out, tensor, axes) do
    t = ensure_on_backend(tensor)
    spv = reverse_nd_spv(type)
    rank = tuple_size(shape)

    if spv != nil and rank >= 1 and rank <= 4 and match?(%__MODULE__{}, t.data) and
         t.type == type do
      gpu_reverse(out, t, axes, spv)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      host_result(out, Nx.reverse(t_bin, axes: axes))
    end
  end

  # params: [rank, shape[4], rev[4]] — rev[d] is 1 when axis d is reversed.
  defp gpu_reverse(
         %T{shape: shape, type: type} = out,
         %T{data: %__MODULE__{ref: a_ref}},
         axes,
         spv
       ) do
    rank = tuple_size(shape)
    n = Nx.size(shape)
    rev = for d <- 0..(rank - 1)//1, do: if(d in axes, do: 1, else: 0)

    params =
      for v <- [rank] ++ pad4(Tuple.to_list(shape)) ++ pad4(rev), into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(type))

    :ok = Nx.Vulkan.NativeV.reverse_nd(out_ref, a_ref, params_ref, n, rank, spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: type})
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
  #
  # Rank-2 contractions over the OTHER axes are normalised into that form by
  # transposing on-device first (dot_orient/4) rather than dropped to the host.
  # This is the dense layer's backward pass: `y = x·W` contracts [1]/[0] and
  # hits the shader, but Nx expresses its gradients as dot with permuted
  # contraction axes rather than materialising transposes —
  # `∂L/∂x = g·Wᵀ` arrives as [1]/[1] and `∂L/∂W = gᵀ·x` as [0]/[0] — so both
  # used to miss. Every dense layer paid it twice per step, conv or no conv.
  @impl true
  def dot(%T{shape: out_shape, type: type} = out, a, axes_a, batched_a, b, axes_b, batched_b) do
    {a_v, axes_a, b_v, axes_b} =
      dot_orient(
        ensure_on_backend(a) |> coerce_operand(type),
        axes_a,
        ensure_on_backend(b) |> coerce_operand(type),
        axes_b,
        batched_a,
        batched_b
      )

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

  # Rotate a rank-2 contraction into the canonical (M,K)·(K,N) the matmul
  # shader expects: `a` must contract on axis 1, `b` on axis 0. A rank-2
  # transpose is the tiled fast path, so the rotation is a single cheap
  # dispatch — far cheaper than moving a matmul to the host.
  #
  # Only rank-2, single-axis, unbatched contractions are rotated. Anything else
  # (higher rank, batched, multi-axis) is returned untouched for the gate below
  # to reject as before.
  defp dot_orient(%T{shape: as} = a, [aa], %T{shape: bs} = b, [ba], [], [])
       when tuple_size(as) == 2 and tuple_size(bs) == 2 and aa in [0, 1] and ba in [0, 1] do
    a = if aa == 0, do: Nx.transpose(a, axes: [1, 0]), else: a
    b = if ba == 1, do: Nx.transpose(b, axes: [1, 0]), else: b
    {a, [1], b, [0]}
  end

  defp dot_orient(a, axes_a, b, axes_b, _batched_a, _batched_b), do: {a, axes_a, b, axes_b}

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
  # Every host fallback lands here (via the host_result/2 macro above, which
  # supplies `op`). Counting centrally is what makes a silent fallback
  # detectable — see Nx.Vulkan.Fallback for why value-based tests structurally
  # cannot catch one.
  defp host_result_recorded(%T{} = out, %T{} = result, op) do
    Nx.Vulkan.Fallback.note(op)
    %{out | data: result.data}
  end

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
