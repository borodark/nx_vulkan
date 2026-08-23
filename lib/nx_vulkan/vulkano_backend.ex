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
        host_result(
          tensor,
          with_binary_backend(fn -> Nx.BinaryBackend.constant(tensor, scalar, opts) end)
        )

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
    min: 6,
    # Integer-only, and absent from every float shader. Nx types all seven as
    # integer-out by contract (`quotient` "always returns an integer tensor",
    # the bitwise ops and shifts raise on floats), so a float shader can never
    # be handed one of these codes — but `binary_spv/2` refuses the pairing
    # explicitly anyway, because the float shaders' `default:` arm returns 0.0
    # and a silent zero is the worst failure this backend has.
    quotient: 7,
    remainder: 8,
    bitwise_and: 9,
    bitwise_or: 10,
    bitwise_xor: 11,
    left_shift: 12,
    right_shift: 13
  ]

  @elementwise_binary_f64_spv Path.expand(
                                "../../priv/shaders/elementwise_binary_f64.spv",
                                __DIR__
                              )
  @elementwise_binary_f32_spv Path.expand(
                                "../../priv/shaders/elementwise_binary_f32.spv",
                                __DIR__
                              )

  @elementwise_binary_s32_spv Path.expand(
                                "../../priv/shaders/elementwise_binary_s32.spv",
                                __DIR__
                              )

  # Keyed on the (type, op code) PAIR, not the type alone. Codes 0-6 exist in
  # all three shaders; 7-13 only in the integer one. Pairing a float shader with
  # an integer-only code would fall into its `default:` arm and write zeros —
  # correct-looking, silently wrong, and invisible to a value assertion because
  # the host fallback returns the same shape. Refusing the pair sends it to the
  # host instead, which is always right.
  defp binary_spv({:f, 64}, code) when code <= 6, do: @elementwise_binary_f64_spv
  defp binary_spv({:f, 32}, code) when code <= 6, do: @elementwise_binary_f32_spv

  # 3 (divide) and 4 (pow) are the two codes the INTEGER shader lacks, and both
  # have to be named here rather than assumed unreachable. `Nx.divide` on two
  # integers really does return {:f, 32}, so 3 never arrives — but `Nx.pow(2, 4)`
  # is s32 in, s32 out, and it does. The first cut of this clause matched every
  # code, sent pow to a shader with no case 4, and the `default:` arm returned
  # `s32 0` instead of 16. That is the exact silent-zero the float clauses above
  # are guarded against; the integer side needs the same guard.
  defp binary_spv({:s, 32}, code) when code != 3 and code != 4,
    do: @elementwise_binary_s32_spv

  defp binary_spv(_type, _code), do: nil

  # Broadcasting elementwise binary (rank <= 4) — keeps bias-add / scaling /
  # relu-via-max on the GPU instead of host-falling-back.
  @bcast_binary_f64_spv Path.expand(
                          "../../priv/shaders/elementwise_binary_bcast_f64.spv",
                          __DIR__
                        )
  @bcast_binary_f32_spv Path.expand(
                          "../../priv/shaders/elementwise_binary_bcast_f32.spv",
                          __DIR__
                        )

  @bcast_binary_s32_spv Path.expand(
                          "../../priv/shaders/elementwise_binary_bcast_s32.spv",
                          __DIR__
                        )

  # Same (type, code) pairing as binary_spv/2, and the same reason.
  defp bcast_binary_spv({:f, 64}, code) when code <= 6, do: @bcast_binary_f64_spv
  defp bcast_binary_spv({:f, 32}, code) when code <= 6, do: @bcast_binary_f32_spv

  defp bcast_binary_spv({:s, 32}, code) when code != 3 and code != 4,
    do: @bcast_binary_s32_spv

  defp bcast_binary_spv(_type, _code), do: nil

  for {op, code} <- @binary_ops do
    @impl true
    def unquote(op)(%T{shape: shape, type: type} = out, a, b) do
      a_v = ensure_on_backend(a)
      b_v = ensure_on_backend(b)
      spv = binary_spv(type, unquote(code))

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
        bspv = bcast_binary_spv(type, unquote(code))

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
  # output rank is 0..4 (Nx guarantees a,b broadcast to out.shape). Dtype
  # mismatches are handled by coerce_to; pow is excluded above (fp64 has no pow).
  #
  # Rank 0 reaches here only when the two scalars have *different* dtypes — equal
  # shapes and equal types take the flat apply_binary path above — so the old
  # `>= 1` was refusing exactly `Nx.multiply(f64_scalar, f32_scalar)`, the third
  # face of the same gate the T11 rank-0 fix removed from compare/select. It
  # dispatches as rank 1 {1}; see gpu_bcast_binary/5.
  defp bcast_shape_ok?(%T{} = a, %T{} = b, %T{shape: os}) do
    match?(%__MODULE__{}, a.data) and match?(%__MODULE__{}, b.data) and
      tuple_size(os) <= 4
  end

  defp bcast_shape_ok?(_a, _b, _out), do: false

  # Rank 0 dispatches as rank 1 of shape {1}: pad4/1 pads the dim list with 1s
  # and pad_left/2 lifts a scalar operand, so the shader needs only a loop bound
  # of 1 rather than 0.
  defp gpu_bcast_binary(
         out,
         %T{data: %__MODULE__{ref: a_ref}} = a,
         %T{data: %__MODULE__{ref: b_ref}} = b,
         code,
         spv
       ) do
    rank = max(tuple_size(out.shape), 1)
    outl = Tuple.to_list(out.shape)
    al = pad_left(Tuple.to_list(a.shape), rank)
    bl = pad_left(Tuple.to_list(b.shape), rank)

    params =
      for v <- [rank] ++ pad4(outl) ++ pad4(al) ++ pad4(bl),
          into: <<>>,
          do: <<v::signed-32-little>>

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    n = byte_size_of(out.shape)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(out.type))

    :ok =
      Nx.Vulkan.NativeV.apply_binary_broadcast(
        out_ref,
        a_ref,
        b_ref,
        params_ref,
        n,
        rank,
        code,
        spv
      )

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: out.type})
  end

  defp pad_left(list, rank), do: List.duplicate(1, rank - length(list)) ++ list
  defp pad4(list), do: (list ++ [1, 1, 1, 1]) |> Enum.take(4)

  # pad4/1's sibling for arrays whose neutral filler is 0, not 1.
  defp pad0(list), do: (list ++ [0, 0, 0, 0]) |> Enum.take(4)

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
    sign: 10,
    # Integer-only, absent from both float shaders. See unary_spv/2.
    bitwise_not: 13,
    population_count: 14,
    count_leading_zeros: 15
  ]

  @elementwise_unary_f64_spv Path.expand(
                               "../../priv/shaders/elementwise_unary_f64.spv",
                               __DIR__
                             )
  @elementwise_unary_f32_spv Path.expand(
                               "../../priv/shaders/elementwise_unary_f32.spv",
                               __DIR__
                             )

  @elementwise_unary_s32_spv Path.expand(
                               "../../priv/shaders/elementwise_unary_s32.spv",
                               __DIR__
                             )

  # Keyed on (type, op code), as binary_spv/2 is, and additionally narrow on the
  # integer side: the s32 shader implements 3/4/7/8/9/10/12 plus 13-15, and NOT
  # the transcendentals. Those cannot arrive anyway — Nx runs exp/log/sqrt/
  # sigmoid/tanh/reciprocal through `Nx.Type.to_floating/1`, so an s32 input
  # produces a float output template and the `a_v.type == type` guard below
  # sends it to the float shader — but listing the codes it really has keeps the
  # gate honest rather than relying on that argument holding forever.
  @s32_unary_codes [3, 4, 7, 8, 9, 10, 12, 13, 14, 15]

  defp unary_spv({:f, 64}, code) when code <= 12, do: @elementwise_unary_f64_spv
  defp unary_spv({:f, 32}, code) when code <= 12, do: @elementwise_unary_f32_spv

  defp unary_spv({:s, 32}, code) when code in @s32_unary_codes,
    do: @elementwise_unary_s32_spv

  defp unary_spv(_type, _code), do: nil

  for {op, code} <- @unary_ops do
    @impl true
    def unquote(op)(%T{shape: shape, type: type} = out, a) do
      a_v = ensure_on_backend(a)
      spv = unary_spv(type, unquote(code))

      # `a_v.type == type` was the gate, and for the transcendentals it is never
      # true on an integer input: Nx routes exp/log/sqrt/sigmoid/tanh through
      # `Nx.Type.to_floating/1`, so `Nx.exp(s32_tensor)` has an f32 OUTPUT
      # template against an s32 operand and the whole thing went to the host —
      # even though `cast_s32_to_f32.spv` has existed since T11 and the binary
      # path has coerced its operands via `coerce_to/2` all along.
      #
      # Textbook narrow gate (skill §1b): the kernel could always do this, only
      # the `if` said otherwise. Found by W5 T2, which made `sum` resident and
      # let logsumexp's doctests get far enough to fail HERE instead.
      #
      # coerce_to/2 returns nil when no cast shader covers the pair, so an
      # uncastable operand still falls back rather than being forced.
      coerced = if spv != nil, do: coerce_to(a_v, type)

      case coerced do
        %T{data: %__MODULE__{ref: a_ref}} ->
          n = byte_size_of(shape)
          n_bytes = n * element_bytes(type)
          {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n_bytes)

          :ok = Nx.Vulkan.NativeV.apply_unary(out_ref, a_ref, n, unquote(code), spv)

          put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: type})

        _ ->
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
    :log1p,
    :erf,
    :erfc,
    :expm1,
    :cbrt,
    :rsqrt,
    # trig
    :acos,
    :acosh,
    :asin,
    :asinh,
    :atan,
    :atanh,
    :cos,
    :cosh,
    :sin,
    :sinh,
    :tan,
    # type / check — is_nan and is_infinity moved to @predicate_unary_ops (W5)
    :round,
    # special
    :erf_inv,
    # complex
    :conjugate,
    :real,
    :imag
  ]

  for op <- @host_fallback_unary_ops do
    @impl true
    def unquote(op)(%T{} = out, a) do
      unary_op_host_fallback(unquote(op), out, ensure_on_backend(a))
    end
  end

  # Binary ops without GPU shader support — host fallback only.
  # W5 emptied this list of everything but `atan2`. The bitwise, shift, integer
  # and logical families all have shaders now — the first three in @binary_ops
  # and @unary_ops above, the logicals in @compare_ops below, because Nx types
  # THEIR output as a u8 mask rather than as the operand type. `atan2` stays: it
  # is a genuine two-argument transcendental and GLSL.std.450 has no f64 form.
  @host_fallback_binary_ops [:atan2]

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

  defp gpu_fft(
         out,
         %T{shape: shape, type: type, data: %__MODULE__{ref: in_ref}},
         length,
         inverse?
       ) do
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

  # T12. Keyed on the (input, output) PAIR, not one type, because the one case
  # that is not type-preserving is the one that mattered: Nx types the sum of a
  # {:u, 8} tensor as {:u, 32}, and Nx.Defn.Grad's reduce_max rule counts tied
  # maxima by summing exactly such a mask. With no entry here that sum stranded
  # the tensor on the host and took softmax's whole backward pass with it.
  @reduce_axis_u8_to_u32_spv Path.expand("../../priv/shaders/reduce_axis_u8_to_u32.spv", __DIR__)

  @reduce_axis_s32_spv Path.expand("../../priv/shaders/reduce_axis_s32.spv", __DIR__)

  defp reduce_spv({:f, 64}, {:f, 64}), do: @reduce_axis_f64_spv
  defp reduce_spv({:f, 32}, {:f, 32}), do: @reduce_axis_f32_spv
  defp reduce_spv({:u, 8}, {:u, 32}), do: @reduce_axis_u8_to_u32_spv

  # W5 T2. Type-PRESERVING only, which is why it reads {:s, 32} twice rather
  # than matching any integer input. Nx widens `sum` and `product` on narrow
  # integers ({:s, 8} -> {:s, 32}) but leaves `reduce_max`/`reduce_min` alone,
  # so {:s, 8} arrives here as either an (s8, s32) or an (s8, s8) pair depending
  # on the OP, not the dtype. Neither has a shader; both keep falling back. The
  # (in, out) keying is what makes that expressible at all — see the u8 note
  # above, which exists for the identical reason.
  defp reduce_spv({:s, 32}, {:s, 32}), do: @reduce_axis_s32_spv

  defp reduce_spv(_from, _to), do: nil

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

    spv = reduce_spv(tensor.type, type)

    fast_path =
      spv != nil and
        match?(%__MODULE__{}, tensor.data) and
        match?({:ok, _}, classify_reduce_axes(in_shape, axes))

    cond do
      fast_path ->
        %T{data: %__MODULE__{ref: a_ref}} = tensor
        {:ok, {outer, reduce_size, inner}} = classify_reduce_axes(in_shape, axes)
        n_out = max(byte_size_of(out_shape), 1)
        out_bytes = n_out * element_bytes(type)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(out_bytes)

        :ok =
          Nx.Vulkan.NativeV.reduce_axis(out_ref, a_ref, outer, reduce_size, inner, op_code, spv)

        put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})

      # A kept axis in the MIDDLE — the conv bias gradient is
      # sum(axes: [0, 2, 3]) over {N, C, H, W}, keeping C. Those axes are
      # neither a leading prefix nor a trailing suffix, so they do not map to
      # the shader's contiguous (outer, reduce_size, inner) slabs and the whole
      # reduction went to the host. Rotating the kept axes to the front makes
      # it a trailing-suffix reduce, which the existing kernel already does —
      # the same normalise-then-dispatch move conv and dot use, and cheap now
      # that transpose is on the GPU for rank <= 4.
      # Type-preserving only: this branch transposes before reducing. Since W1
      # transpose_nd is a word copy and handles every 4/8-byte dtype, so the
      # transpose is no longer the binding constraint for s32/s64/u32 — what
      # still gates those is `reduce_spv/2` having no integer entry (W5). u8 is
      # the one that genuinely cannot come through here: a word copy cannot
      # address a byte, so routing a mask through would trade one fallback for
      # another. That needs W10's byte-packed writer.
      spv != nil and tensor.type == type and match?(%__MODULE__{}, tensor.data) and
          tuple_size(in_shape) <= 4 ->
        reduce_via_transpose(out, tensor, axes, op_code, spv)

      true ->
        reduce_op_host_fallback(op_code, out, tensor, opts)
    end
  end

  # Transpose kept-axes-first, then reduce the trailing block. `kept` stays in
  # ascending order, which is the axis order Nx gives the output, so the result
  # needs no further rearrangement.
  defp reduce_via_transpose(
         %T{shape: out_shape, type: type} = out,
         %T{shape: in_shape} = tensor,
         axes,
         op_code,
         spv
       ) do
    rank = tuple_size(in_shape)
    dims = Tuple.to_list(in_shape)
    reduced = Enum.sort(axes)
    kept = Enum.to_list(0..(rank - 1)//1) -- reduced

    outer = kept |> Enum.map(&Enum.at(dims, &1)) |> Enum.product()
    reduce_size = reduced |> Enum.map(&Enum.at(dims, &1)) |> Enum.product()

    %T{data: %__MODULE__{ref: a_ref}} = Nx.transpose(tensor, axes: kept ++ reduced)

    n_out = max(byte_size_of(out_shape), 1)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n_out * element_bytes(type))

    :ok = Nx.Vulkan.NativeV.reduce_axis(out_ref, a_ref, outer, reduce_size, 1, op_code, spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
  end

  defp reduce_op_host_fallback(op_code, out, tensor, opts) do
    bin_in = Nx.backend_transfer(tensor, Nx.BinaryBackend)

    op =
      case op_code do
        0 -> :sum
        1 -> :reduce_max
        2 -> :reduce_min
        3 -> :product
      end

    result = apply(Nx, op, [bin_in, opts])
    # Explicit attribution: this helper is shared by sum/reduce_max/reduce_min,
    # so the __CALLER__.function capture would blame `reduce_op_host_fallback/4`
    # and hide which reduction left the GPU. (It did, until strict mode printed
    # the helper's own name in 54 refusals.)
    host_result(out, result, {op, 3})
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
  #
  # W1: one type-generic shader, not an f32/f64 pair. A permuted transpose does
  # no arithmetic, so the element type belongs in the params buffer (`ews`) and
  # not in the GLSL. Any 4/8-byte dtype runs — s32/s64/u32 included; 1- and
  # 2-byte dtypes still host-fall-back, as they do for slice/pad/put_slice/
  # gather, because a word copy cannot address a byte.
  @transpose_nd_spv Path.expand("../../priv/shaders/transpose_nd.spv", __DIR__)

  defp word_copyable?(type), do: rem(element_bytes(type), 4) == 0

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
    rank = tuple_size(in_shape)

    cond do
      spv != nil and rank == 2 and axes == [1, 0] ->
        m = elem(in_shape, 0)
        n = elem(in_shape, 1)
        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(m * n * element_bytes(type))

        :ok = Nx.Vulkan.NativeV.transpose_2d(out_ref, a_ref, m, n, spv)

        put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})

      word_copyable?(type) and rank <= 4 ->
        gpu_transpose_nd(out, tensor, axes, @transpose_nd_spv)

      true ->
        t_bin = Nx.backend_transfer(tensor, Nx.BinaryBackend)
        host_result(out, Nx.transpose(t_bin, axes: axes))
    end
  end

  # params: [rank, ews, in[4], out[4], perm[4]] — shapes left-aligned, padded to
  # 4. `ews` is element_bytes/4, the number of u32 words the shader copies per
  # element; it is what makes one shader serve every 4/8-byte dtype.
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
            [rank, div(element_bytes(type), 4)] ++
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

  # T12. A GPU-produced {:u, 8} mask (compare output, select pred) was
  # consumable by select/4 and nothing else: multiply, sum and as_type on one
  # all host-fell-back for want of these two entries. Nx.Defn.Grad routes
  # reduce_max's gradient through that mask, so softmax's backward pass left
  # the GPU on a tensor-sized payload every time, undetected — the values are
  # bit-identical and only a fallback census could see it.
  @cast_u8_to_f32_spv Path.expand("../../priv/shaders/cast_u8_to_f32.spv", __DIR__)
  @cast_u8_to_f64_spv Path.expand("../../priv/shaders/cast_u8_to_f64.spv", __DIR__)
  @cast_u32_to_f32_spv Path.expand("../../priv/shaders/cast_u32_to_f32.spv", __DIR__)
  @cast_u32_to_f64_spv Path.expand("../../priv/shaders/cast_u32_to_f64.spv", __DIR__)

  defp cast_spv({:f, 32}, {:f, 64}), do: @cast_f32_to_f64_spv
  defp cast_spv({:f, 64}, {:f, 32}), do: @cast_f64_to_f32_spv
  defp cast_spv({:s, 32}, {:f, 32}), do: @cast_s32_to_f32_spv
  defp cast_spv({:s, 32}, {:f, 64}), do: @cast_s32_to_f64_spv
  defp cast_spv({:u, 8}, {:f, 32}), do: @cast_u8_to_f32_spv
  defp cast_spv({:u, 8}, {:f, 64}), do: @cast_u8_to_f64_spv
  defp cast_spv({:u, 32}, {:f, 32}), do: @cast_u32_to_f32_spv
  defp cast_spv({:u, 32}, {:f, 64}), do: @cast_u32_to_f64_spv
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
  def as_type(
        %T{type: type} = out,
        %T{type: source_type, shape: shape, data: %__MODULE__{ref: ref}} = tensor
      ) do
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
  # The logical three sit HERE, with the comparisons, and not with the
  # elementwise binaries where their name suggests they belong. Nx builds their
  # output as `%{left | type: {:u, 8}}` (`Nx.element_wise_pred_op/3`) — a packed
  # mask, which is exactly what the compare shaders already write and what
  # nothing in the elementwise family writes. Filing them by output shape rather
  # than by name also closes their f32/f64 instances, which an integer-only fix
  # would have left on the host.
  @compare_ops [
    equal: 0,
    not_equal: 1,
    less: 2,
    less_equal: 3,
    greater: 4,
    greater_equal: 5,
    logical_and: 6,
    logical_or: 7,
    logical_xor: 8
  ]

  # is_nan/is_infinity are the unary members of the same family: one operand, a
  # u8 mask out. They reuse `gpu_compare/5` with the operand bound to BOTH
  # inputs, so the shader's `y` is simply the same value as `x` and is ignored.
  # That is why they need no dispatch helper and no NIF of their own.
  @predicate_unary_ops [is_nan: 9, is_infinity: 10]

  @compare_f32_spv Path.expand("../../priv/shaders/compare_f32.spv", __DIR__)
  @compare_f64_spv Path.expand("../../priv/shaders/compare_f64.spv", __DIR__)
  @compare_s32_spv Path.expand("../../priv/shaders/compare_s32.spv", __DIR__)

  # No op-code guard here, unlike binary_spv/2 and unary_spv/2: all three
  # shaders implement the full 0-10, so every pairing is real.
  defp compare_spv({:f, 32}), do: @compare_f32_spv
  defp compare_spv({:f, 64}), do: @compare_f64_spv
  defp compare_spv({:s, 32}), do: @compare_s32_spv
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

      # Rank 0 dispatches as rank 1 {1} (see gpu_compare/1). The old `>= 1` here
      # had no shader justification and refused every scalar predicate — see the
      # note on gpu_compare.
      cast =
        if spv != nil and match?(%__MODULE__{}, a_v.data) and match?(%__MODULE__{}, b_v.data) and
             tuple_size(out.shape) <= 4 do
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

  for {op, code} <- @predicate_unary_ops do
    @impl true
    def unquote(op)(out, a) do
      a_v = ensure_on_backend(a)
      spv = compare_spv(a_v.type)

      if spv != nil and match?(%__MODULE__{}, a_v.data) and tuple_size(out.shape) <= 4 do
        # Same tensor on both inputs. Two readonly descriptors aliasing one
        # buffer is legal, and it keeps these on the shared compare path instead
        # of growing a second one-operand kernel.
        gpu_compare(out, a_v, a_v, unquote(code), spv)
      else
        unary_op_host_fallback(unquote(op), out, a_v)
      end
    end
  end

  # A rank-0 comparison dispatches as rank 1 of shape {1}: `pad4/1` already pads
  # the dim list with 1s and `pad_left/2` lifts a scalar operand to {1}, so the
  # only thing the shader needs is a loop bound of 1 instead of 0. Nothing in
  # `compare_f{32,64}.comp` cares about rank beyond that — the guard that used to
  # read `>= 1` was refusing scalars for no reason (skill §1b), and every scalar
  # support check in a probabilistic model went to the host because of it.
  defp gpu_compare(
         out,
         %T{data: %__MODULE__{ref: a_ref}} = a,
         %T{data: %__MODULE__{ref: b_ref}} = b,
         code,
         spv
       ) do
    rank = max(tuple_size(out.shape), 1)

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

  @select_s32_spv Path.expand("../../priv/shaders/select_s32.spv", __DIR__)

  defp select_spv({:f, 32}), do: @select_f32_spv
  defp select_spv({:f, 64}), do: @select_f64_spv
  defp select_spv({:s, 32}), do: @select_s32_spv
  defp select_spv(_), do: nil

  @impl true
  def select(%T{type: type, shape: os} = out, pred, on_true, on_false) do
    p = ensure_on_backend(pred)
    t = ensure_on_backend(on_true)
    f = ensure_on_backend(on_false)
    spv = select_spv(type)

    # The shader reads `pred` as a packed u8 mask, which is what the compare
    # family emits — so the gate demanded `{:u, 8}` and refused everything else.
    # But `Nx.select/3` takes ANY numeric predicate and treats nonzero as true,
    # and its own doctests pass `1`, `0` and `Nx.tensor([0, 1, 0])`: all s32.
    #
    # `Nx.not_equal(pred, 0)` IS that normalisation, and it is itself a GPU op
    # this backend has had since W5 T1 — one compare dispatch against a host
    # round trip for all three operands. Same shape of fix as `gather`'s axis
    # rotation: the kernel could always do the work, the encoding was wrong.
    #
    # A predicate whose dtype has no compare shader falls back through the
    # `match?` below rather than being forced, because `not_equal` will have
    # left it on the host.
    p = if p.type == {:u, 8}, do: p, else: Nx.not_equal(p, 0)

    # Rank 0 dispatches as rank 1 {1} (see gpu_select/5).
    shape_ok? =
      spv != nil and p.type == {:u, 8} and match?(%__MODULE__{}, p.data) and
        match?(%__MODULE__{}, t.data) and match?(%__MODULE__{}, f.data) and
        tuple_size(os) <= 4

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

  # Rank 0 dispatches as rank 1 of shape {1}, exactly as in gpu_compare/5.
  defp gpu_select(
         out,
         %T{data: %__MODULE__{ref: p_ref}} = p,
         %T{data: %__MODULE__{ref: t_ref}} = t,
         %T{data: %__MODULE__{ref: f_ref}} = f,
         spv
       ) do
    rank = max(tuple_size(out.shape), 1)

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

  # all/3, any/3 — boolean reductions. `glsl/allany_*.comp`, which share
  # reduce_axis_*.comp's bindings and push layout and so reuse `reduce_axis/7`
  # with no new NIF, exactly as the argreduce family does.
  @allany_s32_spv Path.expand("../../priv/shaders/allany_s32.spv", __DIR__)
  @allany_f32_spv Path.expand("../../priv/shaders/allany_f32.spv", __DIR__)
  @allany_f64_spv Path.expand("../../priv/shaders/allany_f64.spv", __DIR__)
  @allany_u8_spv Path.expand("../../priv/shaders/allany_u8.spv", __DIR__)

  # Keyed on the INPUT type. The output is always {:u, 8} — Nx types these that
  # way whatever went in — so there is no (in, out) pair to track.
  #
  # The u8 entry is the one that earns its keep: `Nx.all(Nx.greater(a, b))` is
  # the natural idiom, `greater` already emits a u8 mask on the GPU, and without
  # this the mask would be dragged back to the host to be summarised. Same
  # lesson as T12's `{:u, 8} -> {:u, 32}` sum entry — the dtype a gate refuses
  # is usually one the backend itself produced.
  defp allany_spv({:s, 32}), do: @allany_s32_spv
  defp allany_spv({:f, 32}), do: @allany_f32_spv
  defp allany_spv({:f, 64}), do: @allany_f64_spv
  defp allany_spv({:u, 8}), do: @allany_u8_spv
  defp allany_spv(_), do: nil

  @impl true
  def all(out, tensor, opts), do: do_allany(out, tensor, opts, 0, :all, &Nx.all/2)

  @impl true
  def any(out, tensor, opts), do: do_allany(out, tensor, opts, 1, :any, &Nx.any/2)

  defp do_allany(%T{shape: out_shape, type: out_type} = out, tensor, opts, op_code, op, host_fun) do
    t = ensure_on_backend(tensor)
    spv = allany_spv(t.type)
    axes = Keyword.get(opts, :axes) || all_axes(t.shape)

    fast_path? =
      spv != nil and match?(%__MODULE__{}, t.data) and out_type == {:u, 8} and
        match?({:ok, _}, classify_reduce_axes(t.shape, axes))

    if fast_path? do
      %T{data: %__MODULE__{ref: a_ref}} = t
      {:ok, {outer, reduce_size, inner}} = classify_reduce_axes(t.shape, axes)
      n_out = max(byte_size_of(out_shape), 1)
      # u8 out, written as u32 words — pad the buffer to a 4-byte multiple, the
      # same as gpu_compare/5.
      {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(div(n_out + 3, 4) * 4)

      :ok =
        Nx.Vulkan.NativeV.reduce_axis(out_ref, a_ref, outer, reduce_size, inner, op_code, spv)

      put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: out_type})
    else
      bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      # Explicit attribution — shared helper, see do_argreduce/6.
      host_result(out, host_fun.(bin, opts), {op, 3})
    end
  end

  # block/4 — Nx's "block" callback dispatches Nx.Block-derived structs
  # (SVD, QR, LU, etc.). Host-fallback: transfer every tensor in the
  # input tuple to BinaryBackend, evaluate via its block impl, then
  # transfer outputs back. Scholar's linear regression uses
  # Nx.Block.LinAlg.SVD internally; this unblocks it.
  #
  # The parameter names follow `Nx.Backend`'s contract, which is
  # `block(struct, output, args, fun)` — NOT `(out, block_def, ...)`, which is
  # what they were called here. The order was right, so the callback worked;
  # only the names lied, and the first thing to read them (T13's attribution,
  # below) keyed the census off the output template and reported every block
  # in the API as `Nx.Tensor`.
  # W4 — blocks evaluated ON THIS BACKEND rather than transferred wholesale.
  #
  # The nine `{:block, _}` allowlist entries in `Nx.Vulkan.Fallback` are blocks
  # whose bodies genuinely belong on the host: `Nx.LinAlg.SVD`'s composes into
  # ~350 separate ops, so transferring once and noting once is both faster and a
  # far more legible census than 350 lines of shrapnel.
  #
  # These are the opposite case. Their bodies compose ops this backend already
  # has shaders for, so evaluating the body here does real GPU work — and where
  # a constituent op has no GPU path, IT reports, naming the actual gap instead
  # of hiding it behind "a block fell back". `Nx.logical_not/1` is a compare
  # against zero; `Nx.take/3` and `Nx.take_along_axis/3` are `gather/4`, which
  # has had a shader all along.
  #
  # Deliberately NOT noted as a fallback: nothing has left the device at this
  # point. The constituent ops are individually gated and individually
  # attributed, which is the whole reason routing beats an allowlist line.
  # Measured on super-io (RTX 3060 Ti), every result checked element-wise
  # against `Nx.BinaryBackend`. What each decomposes into once routed:
  #
  #   Nx.logical_not/1 f32      0 fallbacks — resident
  #   Nx.logical_not/1 s32      equal/3        (W5's integer bucket)
  #   Nx.take/3 axis 0          0 fallbacks — resident
  #   Nx.take/3 axis 1          gather/4       (GPU path wants leading-prefix axes)
  #   Nx.take_along_axis/3      concatenate/3
  #   Nx.top_k/2                argsort/3      (already an allowlisted decision)
  #   Nx.cumulative_*/2 axis 0  0 fallbacks — resident
  #   Nx.cumulative_*/2 axis 1  concatenate/3 ×2
  #
  # Twelve opaque `{:block, _}` fallbacks become three named gaps —
  # `concatenate/3`, `gather/4` off-prefix, and integer `equal`/`add`. That is
  # the argument for routing over an allowlist line: an allowlist entry would
  # have recorded a decision about `cumulative_sum` when the thing actually
  # missing is a concatenate shader, which five of these ops share.
  @device_blocks [
    Nx.Block.LogicalNot,
    Nx.Block.Take,
    Nx.Block.TakeAlongAxis,
    Nx.Block.TopK,
    Nx.Block.CumulativeSum,
    Nx.Block.CumulativeProduct,
    Nx.Block.CumulativeMin,
    Nx.Block.CumulativeMax
  ]

  @doc false
  @spec device_blocks() :: [module()]
  def device_blocks, do: @device_blocks

  @impl true
  def block(block_struct, output, args, fun) when is_list(args) do
    if block_kind(block_struct) in @device_blocks do
      # Every tensor arg must be on THIS backend before the body runs — the
      # exact mirror of what `host_block/4` does downward, and for the same
      # reason. nx dispatches a multi-arg op to ONE backend, so a body called
      # with `tensor` here and `indices` on BinaryBackend resolves to
      # `Nx.BinaryBackend.gather/3` and hands it a Vulkano tensor, which dies in
      # `to_binary/1` with no clause. `Nx.take/3` reaches that state through
      # `Nx.padding_with_index/2`, whose index tensor is built on the default
      # backend while the operand arrived from elsewhere.
      apply(fun, [block_struct | Enum.map(args, &to_device/1)])
    else
      host_block(block_struct, output, args, fun)
    end
  end

  def block(block_struct, output, args, fun),
    do: host_block(block_struct, output, args, fun)

  defp to_device(%T{} = t), do: ensure_on_backend(t)
  defp to_device(other), do: other

  defp host_block(block_struct, output, args, fun) do
    transfer_to_bin = fn t ->
      if is_struct(t, Nx.Tensor) and match?(%__MODULE__{}, t.data) do
        Nx.backend_transfer(t, Nx.BinaryBackend)
      else
        t
      end
    end

    args_bin =
      cond do
        is_list(args) -> Enum.map(args, transfer_to_bin)
        is_tuple(args) -> args |> Tuple.to_list() |> Enum.map(transfer_to_bin) |> List.to_tuple()
        true -> transfer_to_bin.(args)
      end

    # T13. Every block/4 call is a host fallback by construction — there is no
    # GPU path for any `Nx.Block.*` and the result comes back on BinaryBackend
    # — so this is noted unconditionally rather than by inspecting the result.
    #
    # Attribution is per `Nx.Block.*` struct, NOT a single `{:block, 4}`. One
    # entry would have to be allowlisted wholesale, which is the op-family
    # wildcard `@allowlist` forbids, and it would make `Nx.all_close` (an
    # assertion helper used throughout this suite) raise under `:raise`
    # alongside a genuinely missing `cumulative_sum` shader. Keyed by struct,
    # the two are separable: each carries its own reason, and when someone
    # writes a scan shader they delete one line.
    #
    # Until this landed the whole family — Nx.LinAlg (svd/qr/lu/cholesky/solve/
    # eigh/determinant), top_k, cumulative_*, take, all_close, fft2 — was
    # invisible to count/1 AND to strict mode, so "zero fallbacks" meant "zero
    # recorded" and a green strict run said nothing about any of it.
    Nx.Vulkan.Fallback.note({:block, block_kind(block_struct)}, block_meta(output))

    # W3: `with_binary_backend/1` is load-bearing, not tidiness. Transferring
    # the ARGS is not enough — every `fun` here is a defn, and
    # `Nx.Defn.Evaluator` materialises its constants and intermediates on the
    # process DEFAULT backend, which is this one. So `Nx.LinAlg.LU.lu/1` ran its
    # pivot search through VulkanoBackend on tensors nobody transferred, and
    # `Nx.LinAlg.lu(Nx.eye(2))` returned P = [[0,1],[1,0]], L = [[1,0],[1,1]]
    # and U = all zeros — for the identity matrix. `Nx.LinAlg.solve/2` then
    # raised `can't solve for singular matrix` on a non-singular input.
    #
    # This was invisible for as long as it existed because the same call raised
    # `ArithmeticError` in `encode_scalar/2` first (nx composes solve's pivot
    # search through `Nx.Constants.neg_infinity`, an ATOM). Fixing the encoder
    # is what made the wrong answer reachable. A raise is a much better failure
    # than a plausible wrong matrix, which is the only reason this was ever
    # found.
    result =
      with_binary_backend(fn -> Nx.BinaryBackend.block(block_struct, output, args_bin, fun) end)

    # Per Tier 1 of SHAPE_C_PLAN.md: result is already on BinaryBackend,
    # leave it there. Nx supports mixed-backend tensors flowing through
    # the pipeline; the next op auto-transfers if it needs GPU.
    result
  end

  # `Nx.Block.LinAlg.SVD` etc. Anything that is not a struct (no such case in
  # nx 0.13, but the callback's contract does not forbid it) reports as-is
  # rather than crashing the op it is trying to instrument.
  defp block_kind(%mod{}), do: mod
  defp block_kind(other), do: other

  # The allowlist's rank conditions want a tensor. `out` is a tensor for most
  # blocks and a tuple for the multi-output ones (SVD -> {u, s, vt}); hand over
  # the first tensor found, or nil.
  defp block_meta(%T{} = out), do: out
  defp block_meta(out) when is_tuple(out), do: out |> Tuple.to_list() |> block_meta()
  defp block_meta([%T{} = t | _]), do: t
  defp block_meta([_ | rest]), do: block_meta(rest)
  defp block_meta(_), do: nil

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
  # rank 0..4. Everything else host-falls-back.
  #
  # The gate used to require `pv.type == t.type`, which is the {:s, 32} literal
  # trap of skill §1b a second time over: `Nx.pad(f64_tensor, 0.0, cfg)` hands
  # this callback an f32 (or s32) scalar, and a four-byte constant dragged the
  # whole tensor to the host. Nx's own output type is the merge of the two, so
  # coerce the pad value to `out.type` instead of comparing types — coerce_to/2
  # rebuilds a rank-0 constant at the target type for one 4-byte round trip, and
  # returns nil for anything it cannot convert (integer storage), which still
  # host-falls-back.
  #
  # The source tensor is deliberately NOT coerced: `t.type == type` fails only
  # when the pad value is *wider* than the tensor (f64 value, f32 tensor), and
  # in exactly that case `Nx.BinaryBackend.pad/4` — the reference this backend
  # is required to match bit-for-bit — casts the pad value to `out.type` but not
  # the tensor, and splices an 8-byte value into a 4-byte-element binary. Its
  # answer is wrong, and reproducing it is still the contract, so that case goes
  # to the host rather than being silently "fixed" here.
  @impl true
  def pad(%T{type: type} = out, tensor, pad_value, padding_config) do
    t = ensure_on_backend(tensor)
    pv = ensure_on_backend(pad_value)
    eb = element_bytes(type)
    rank = tuple_size(t.shape)

    cpv = if match?(%__MODULE__{}, pv.data) and tuple_size(pv.shape) == 0, do: coerce_to(pv, type)

    # An empty source or an empty result means a zero-byte buffer binding, which
    # is not a thing Vulkan will accept — those stay on the host.
    if match?(%__MODULE__{}, t.data) and t.type == type and cpv != nil and rem(eb, 4) == 0 and
         rank <= 4 and Nx.size(t.shape) > 0 and Nx.size(out.shape) > 0 do
      gpu_pad(out, t, cpv, padding_config, eb)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      # as_type/2 to `out.type` before re-entering Nx.pad/3: Nx computed the
      # callback's out type from a *number* pad value (`binary_type/2` keeps
      # {:u, 8} against a literal 0), while re-running it here passes a tensor
      # and merges strictly to {:s, 32}. Without the cast the host path returns
      # a differently-typed binary under a u8 header and the values come back
      # wrong — pre-existing, found by the T11 parity sweep
      # (`Nx.pad(Nx.iota({4}, type: :u8), 0, [{1, 1, 0}])`).
      pv_bin = pv |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.as_type(type)
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

    # backend_copy, not backend_transfer: the pad value may still be referenced
    # elsewhere in the graph, and transfer deallocates the device buffer.
    pv_bin = Nx.backend_copy(pv, Nx.BinaryBackend) |> Nx.to_binary()
    {:ok, padval_ref} = Nx.Vulkan.NativeV.buf_upload(pv_bin)

    n = byte_size_of(out.shape)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * eb)

    :ok = Nx.Vulkan.NativeV.apply_pad(out_ref, in_ref, params_ref, padval_ref, n, rank, @pad_spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out.shape, type: out.type})
  end

  # put_slice: write `slice` into `tensor` at `start_indices`. Used by NUTS
  # batched leapfrog to accumulate trajectory steps, and by every `PointMap`
  # unpack of a flat parameter vector — the op that decides whether a
  # probabilistic model stays resident (bench_results/EXMC_PEROP_RACE.md).
  #
  # GPU path: an index-remap overlay (`glsl/put_slice.comp`) — one thread per
  # output element, reading the slice inside the window and the tensor outside
  # it. 4/8-byte dtypes, rank 1..4, start indices that resolve to numbers.
  # Rank 0 is excluded on purpose and with evidence, unlike the compare/select
  # gate this task removed: `Nx.BinaryBackend.put_slice/5` raises on a scalar
  # (`make_anchors` maps over `:init`), so answering it here would diverge from
  # the reference by succeeding.
  # Nx clamps the starts to [0, dim - slice_dim] (BinaryBackend.clamp_indices),
  # so that happens here, on the host, and the shader assumes an in-bounds
  # window.
  @put_slice_spv Path.expand("../../priv/shaders/put_slice.spv", __DIR__)

  @impl true
  def put_slice(%T{type: type, shape: shape} = out, tensor, start_indices, slice) do
    t = ensure_on_backend(tensor)
    s = ensure_on_backend(slice)
    eb = element_bytes(type)
    rank = tuple_size(shape)

    ct = if match?(%__MODULE__{}, t.data), do: coerce_to(t, type)
    cs = if match?(%__MODULE__{}, s.data), do: coerce_to(s, type)
    starts = static_starts(start_indices)

    if ct != nil and cs != nil and starts != nil and rem(eb, 4) == 0 and rank >= 1 and rank <= 4 and
         t.shape == shape and tuple_size(s.shape) == rank and
         Nx.size(shape) > 0 and Nx.size(s.shape) > 0 do
      clamped =
        [Tuple.to_list(shape), starts, Tuple.to_list(s.shape)]
        |> Enum.zip_with(fn [dim, start, len] -> min(max(start, 0), dim - len) end)

      gpu_put_slice(out, ct, cs, clamped, eb)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      s_bin = Nx.backend_transfer(s, Nx.BinaryBackend)
      idx_bin = Enum.map(start_indices, &maybe_transfer_idx/1)
      result = Nx.put_slice(t_bin, idx_bin, s_bin)
      host_result(out, result)
    end
  end

  # Nx.put_slice/3 hands the backend either a list of plain integers or, if any
  # index is a tensor, a list of rank-0 tensors (`Nx.to_indices/1`). A scalar
  # tensor is 4-8 bytes: reading it back is cheaper by orders of magnitude than
  # sending the tensor it indexes to the host, so resolve it rather than bail.
  # Returns nil for anything that is not a number or a rank-0 numeric tensor.
  defp static_starts(start_indices) do
    Enum.reduce_while(start_indices, [], fn idx, acc ->
      case idx do
        i when is_integer(i) ->
          {:cont, acc ++ [i]}

        %T{shape: {}} = t ->
          {:cont, acc ++ [t |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_number() |> trunc()]}

        _ ->
          {:halt, nil}
      end
    end)
  end

  # params: [rank, ews, T[4], S[4], start[4]] — T = tensor (== output) dims,
  # S = slice dims, start clamped. Rank 0 dispatches with rank 0: the shader's
  # decompose loop does not run and every one of the (single) output elements is
  # "inside", which is the correct answer for a scalar put_slice.
  defp gpu_put_slice(
         %T{shape: shape} = out,
         %T{data: %__MODULE__{ref: in_ref}},
         %T{shape: sshape, data: %__MODULE__{ref: slice_ref}},
         starts,
         eb
       ) do
    rank = tuple_size(shape)
    ews = div(eb, 4)

    params =
      for v <-
            [rank, ews] ++
              pad4(Tuple.to_list(shape)) ++
              pad4(Tuple.to_list(sshape)) ++
              pad4(starts),
          into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    n = byte_size_of(shape)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * eb)

    :ok =
      Nx.Vulkan.NativeV.apply_put_slice(
        out_ref,
        in_ref,
        slice_ref,
        params_ref,
        n,
        rank,
        @put_slice_spv
      )

    put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: out.type})
  end

  # indexed_put: scatter updates into tensor at indices. Used to fill
  # NUTS per-step logp slots inside the leapfrog while loop.
  @impl true
  def indexed_put(out, tensor, indices, updates, opts \\ []) do
    scatter_op(out, tensor, indices, updates, opts, 0, :indexed_put, &Nx.indexed_put/4)
  end

  # indexed_add: scatter-accumulate. Same shape as indexed_put, but NOT the same
  # concurrency story — see scatter_op/7.
  @impl true
  def indexed_add(out, tensor, indices, updates, opts \\ []) do
    scatter_op(out, tensor, indices, updates, opts, 1, :indexed_add, &Nx.indexed_add/4)
  end

  @scatter_spv Path.expand("../../priv/shaders/scatter.spv", __DIR__)

  # `glsl/scatter.comp` — the inverse of `gather.comp`, same index arithmetic and
  # the same params layout with source and destination swapped. Both were
  # unconditional host fallbacks at EVERY dtype until now, which is also where
  # `Nx.LinAlg.invert/1` died (MISSION §3.3).
  #
  # The two ops differ in exactly one way, and it decides their dtype gates:
  #
  #   * `indexed_put` DOCUMENTS the race. "In case of repeating indices, the
  #     result is non-deterministic, since the operation happens in parallel when
  #     running on devices such as the GPU." So a plain word write is not a
  #     tolerated approximation, it is the specified behaviour, and every
  #     4/8-byte dtype can use it.
  #
  #   * `indexed_add` must accumulate duplicates deterministically, which needs
  #     an atomic. Integer `atomicAdd` is core GLSL 4.30 and works on the
  #     two's-complement bit pattern, so s32/u32 are exact through a `uint`
  #     view. FLOAT indexed_add stays on the host for the same reason
  #     overlapping pooling backward does — `GL_EXT_shader_atomic_float` is not
  #     guaranteed on the Kepler fleet.
  #
  # Shared with gather: the indexed axes must be the leading prefix [0..K-1].
  # Anything else needs a transpose first and host-falls-back for now.
  defp scatter_op(%T{type: type} = out, tensor, indices, updates, opts, op_code, op, host_fun) do
    t = ensure_on_backend(tensor)
    idx = ensure_on_backend(indices)
    upd = ensure_on_backend(updates)

    rank = tuple_size(t.shape)
    idx_rank = tuple_size(idx.shape)
    k = if idx_rank > 0, do: elem(idx.shape, idx_rank - 1), else: 0
    eb = element_bytes(type)
    ib = element_bytes(idx.type)

    axes =
      case opts[:axes] do
        nil -> if k > 0, do: Enum.to_list(0..(k - 1)), else: []
        given -> Nx.Shape.normalize_axes(t.shape, given, t.names)
      end

    shape_ok? =
      match?(%__MODULE__{}, t.data) and match?(%__MODULE__{}, idx.data) and
        match?(%__MODULE__{}, upd.data) and
        idx_rank == 2 and k >= 1 and rank >= 1 and rank <= 4 and
        axes == Enum.to_list(0..(k - 1)) and
        rem(eb, 4) == 0 and rem(ib, 4) == 0 and
        (op_code == 0 or (integer_type?(type) and eb == 4))

    # Nx PROMOTES here — `Nx.indexed_add(Nx.tensor([1]), idx, Nx.tensor([1.0]))`
    # is an s32 target and f32 updates producing an f32 result, and both of its
    # own doctests exercise it. Requiring `t.type == type and upd.type == type`
    # therefore refused a coercible operand, which is the same narrow gate the
    # unary path had: `Nx.LinAlg.invert/1` hit it with s32 updates into an f32
    # target and fell back at the last step of a chain that had otherwise made
    # it onto the device. coerce_to/2 returns nil when no cast shader covers the
    # pair, so a genuinely uncastable operand still falls back.
    coerced = if shape_ok?, do: {coerce_to(t, type), coerce_to(upd, type)}

    case coerced do
      {%T{} = ct, %T{} = cu} ->
        gpu_scatter(out, ct, idx, cu, k, eb, ib, op_code)

      _ ->
        t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
        i_bin = Nx.backend_transfer(idx, Nx.BinaryBackend)
        u_bin = Nx.backend_transfer(upd, Nx.BinaryBackend)
        # Explicit attribution: this helper is shared by indexed_put and
        # indexed_add, so the __CALLER__.function capture would blame
        # `scatter_op/8` and hide which of the two left the GPU. It did exactly
        # that for one run — `Nx.LinAlg.invert/1`'s census reported
        # `{:scatter_op, 7}`, which names nothing a reader can act on.
        host_result(out, host_fun.(t_bin, i_bin, u_bin, opts), {op, 5})
    end
  end

  defp integer_type?({:s, _}), do: true
  defp integer_type?({:u, _}), do: true
  defp integer_type?(_), do: false

  # params: [K, ews, idx_words, count, stride[4]] — byte-identical to gather's,
  # because it is the same walk in the other direction.
  defp gpu_scatter(
         %T{shape: out_shape, type: type} = out,
         %T{shape: t_shape, data: %__MODULE__{ref: t_ref}},
         %T{data: %__MODULE__{ref: idx_ref}},
         %T{shape: u_shape, data: %__MODULE__{ref: u_ref}},
         k,
         eb,
         ib,
         op_code
       ) do
    dims = Tuple.to_list(t_shape)
    count = dims |> Enum.drop(k) |> Enum.reduce(1, &(&1 * &2))
    strides = for j <- 0..(k - 1), do: dims |> Enum.drop(j + 1) |> Enum.reduce(1, &(&1 * &2))

    params =
      for v <- [k, div(eb, 4), div(ib, 4), count] ++ pad4(strides), into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)

    # Seed the output with the target. A scatter writes only the elements the
    # indices name; everything else has to survive, so `buf_alloc` (zeroed) is
    # wrong here. `concat_buffers/1` on a single buffer is a device-to-device
    # copy that waits before returning, which also orders it ahead of the
    # dispatch below.
    {:ok, out_ref} = Nx.Vulkan.NativeV.concat_buffers([t_ref])

    n = byte_size_of(u_shape)

    :ok =
      Nx.Vulkan.NativeV.apply_scatter(
        out_ref,
        u_ref,
        idx_ref,
        params_ref,
        n,
        k,
        op_code,
        @scatter_spv
      )

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
  end

  # broadcast: project a tensor to a new shape along `axes`. Implicit
  # broadcasts during binary ops route through binary_op_host_fallback,
  # but explicit Nx.broadcast/3 needs its own callback. Used at every
  # NUTS init / mass-matrix scaffolding site.
  # W1: type-generic, see the note on @transpose_nd_spv.
  @broadcast_nd_spv Path.expand("../../priv/shaders/broadcast_nd.spv", __DIR__)

  @impl true
  def broadcast(%T{type: type} = out, tensor, shape, axes) do
    t = ensure_on_backend(tensor)
    out_rank = tuple_size(shape)
    in_rank = tuple_size(t.shape)

    if word_copyable?(type) and out_rank >= 1 and out_rank <= 4 and in_rank <= 4 and
         match?(%__MODULE__{}, t.data) and t.type == type do
      gpu_broadcast(out, t, shape, axes, @broadcast_nd_spv)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      host_result(out, Nx.broadcast(t_bin, shape, axes: axes))
    end
  end

  # params: [out_rank, in_rank, ews, out[4], in[4], axes[4]] — input axis i lands
  # on output axis axes[i]; an input axis of size 1 repeats. `ews` sits after
  # BOTH ranks here, unlike its two siblings which have only one.
  defp gpu_broadcast(
         %T{type: type} = out,
         %T{shape: in_shape, data: %__MODULE__{ref: a_ref}},
         shape,
         axes,
         spv
       ) do
    out_rank = tuple_size(shape)
    in_rank = tuple_size(in_shape)
    n = Nx.size(shape)

    params =
      for v <-
            [out_rank, in_rank, div(element_bytes(type), 4)] ++
              pad4(Tuple.to_list(shape)) ++
              pad4(Tuple.to_list(in_shape)) ++
              pad4(axes),
          into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(type))

    :ok = Nx.Vulkan.NativeV.broadcast_nd(out_ref, a_ref, params_ref, n, spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: type})
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

      # W4's census made this the highest-leverage gap in the backend: an
      # axis > 0 concatenate was the ONLY thing keeping `Nx.take_along_axis/3`
      # and all four `Nx.cumulative_*/2` off the GPU, and `associative_scan`
      # calls it log2(n) times per reduction — each one previously round-tripping
      # the whole tensor through the host.
      #
      # Same type gate as axis 0 and for the same reason (a raw copy cannot
      # cast). 4/8-byte dtypes only, because the shader copies u32 words;
      # 1/2-byte types fall back as they do for slice/pad/put_slice. rank <= 4
      # is pad4's ceiling.
      concat_nd_eligible?(out, tensors, axis) ->
        concat_nd_vulkano(out, tensors, axis)

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

  @concat_nd_spv Path.expand("../../priv/shaders/concat_nd.spv", __DIR__)

  # `all_vulkano?`, deliberately, and the deliberation is worth recording because
  # the looser gate was tried and reverted.
  #
  # Requiring only ONE resident operand — upload the rest, keep the result here —
  # looks like the SKILL §1b move: the shader can plainly compute it, so why
  # refuse. It made four `Nx.mode/2` doctests crash. Promoting the operands makes
  # the RESULT resident, and `Nx.take_along_axis/3` then hands that resident
  # index tensor to `Nx.gather/3` alongside a host operand; nx resolves a
  # multi-arg op to ONE backend, picks `Nx.BinaryBackend.gather/3`, and it dies
  # in `to_binary/1` with no clause.
  #
  # So the looser gate did not remove a mixed-backend pair, it moved one
  # downstream where this backend cannot fix it. §1b says gate on what the
  # kernel cannot do — the kernel can, but the *caller* cannot, and that is
  # still a real constraint. Uniformity at the boundary is what W4's `to_device`
  # buys and what this preserves.
  #
  # This costs nothing where residency is actually measured: under a
  # VulkanoBackend default — production, and `nx_doctest_test.exs`'s `setup` —
  # `take_along_axis`'s `Nx.iota/2` and `reshape` are resident too, so every
  # operand is, and the fast path runs.
  defp concat_nd_eligible?(%T{shape: out_shape, type: type}, tensors, axis) do
    rank = tuple_size(out_shape)

    axis > 0 and axis < rank and rank <= 4 and
      rem(element_bytes(type), 4) == 0 and all_vulkano?(tensors) and
      Enum.all?(tensors, &(&1.type == type)) and
      Enum.all?(tensors, &(tuple_size(&1.shape) == rank))
  end

  # One dispatch per input, each writing its own disjoint slab of the output.
  # Offsets accumulate along the concat axis in input order, which is exactly
  # what Nx.concatenate/2 means.
  defp concat_nd_vulkano(%T{shape: out_shape, type: type} = out, tensors, axis) do
    rank = tuple_size(out_shape)
    eb = element_bytes(type)
    out_dims = pad4(Tuple.to_list(out_shape))

    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(byte_size_of(out_shape) * eb)

    Enum.reduce(tensors, 0, fn %T{shape: in_shape, data: %__MODULE__{ref: a_ref}}, offset ->
      n = byte_size_of(in_shape)

      params =
        for v <-
              [rank, div(eb, 4), axis, offset] ++
                pad4(Tuple.to_list(in_shape)) ++ out_dims,
            into: <<>>,
            do: <<v::signed-32-little>>

      {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)

      :ok = Nx.Vulkan.NativeV.concat_nd(out_ref, a_ref, params_ref, n, rank, @concat_nd_spv)

      offset + elem(in_shape, axis)
    end)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
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
  # stack/3 is `concatenate/3` with a size-1 axis inserted first, and that is not
  # an approximation — it is exactly what Nx.BinaryBackend does
  # (`Tuple.insert_at(shape, axis, 1)` then `bin_concatenate`). It arrives here
  # with the ORIGINAL tensors, so the insert is this backend's job.
  #
  # Which makes this pure routing rather than a kernel: `reshape/2` here is
  # metadata only (same buffer, new shape), and `concatenate/3` has had a shader
  # since `concat_nd` — a byte append at axis 0, the index-remap kernel above it.
  # The op was transferring wholesale to the host for want of two lines.
  #
  # Types are merged by Nx before dispatch, so an operand can arrive narrower
  # than `out.type`; `coerce_to/2` casts it on the device, and returns nil when
  # no cast shader covers the pair, which sends the whole thing to the host.
  # That mirrors what BinaryBackend does with its own `as_type` call.
  @impl true
  def stack(%T{type: type} = out, tensors, axis) do
    lifted =
      Enum.reduce_while(tensors, [], fn t, acc ->
        t = ensure_on_backend(t)

        case coerce_to(t, type) do
          %T{shape: shape} = ct ->
            {:cont, [Nx.reshape(ct, Tuple.insert_at(shape, axis, 1)) | acc]}

          _ ->
            {:halt, nil}
        end
      end)

    case lifted do
      nil ->
        stack_host_fallback(out, tensors, axis)

      list ->
        concatenate(out, Enum.reverse(list), axis)
    end
  end

  defp stack_host_fallback(out, tensors, axis) do
    bins =
      Enum.map(tensors, fn t ->
        Nx.backend_transfer(ensure_on_backend(t), Nx.BinaryBackend)
      end)

    host_result(out, with_binary_backend(fn -> Nx.stack(bins, axis: axis) end), {:stack, 3})
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

    # The shader wants the indexed axes as a leading prefix [0..K-1]. When they
    # are not, ROTATE rather than refuse — the same normalise-then-dispatch move
    # `dot_orient/6` makes for matmul, and the reason SKILL §1b gives for
    # preferring it: the kernel can do the work, only the layout is wrong.
    #
    # `perm = axes ++ everything else, in order` puts them there, and the OUTPUT
    # NEEDS NO ROTATION BACK. Nx defines a gather's result as the index batch
    # dims followed by the non-indexed source dims IN THEIR ORIGINAL RELATIVE
    # ORDER, and a transpose that only moves the indexed axes to the front
    # leaves that order untouched. That is what makes this two lines instead of
    # a second permutation.
    #
    # The transpose is available exactly when the gather is: `transpose_nd` is a
    # word copy for rank <= 4 and 4-byte-divisible dtypes, which this gate
    # already requires. One extra dispatch against a host round trip for the
    # whole tensor is the same trade dot makes.
    prefix? = axes == Enum.to_list(0..(k - 1))

    rotatable? =
      match?(%__MODULE__{}, t.data) and match?(%__MODULE__{}, idx.data) and
        rem(eb, 4) == 0 and rem(ib, 4) == 0 and rank >= 1 and rank <= 4 and k >= 1

    cond do
      rotatable? and prefix? ->
        gpu_gather(out, t, idx, k, eb, ib)

      rotatable? ->
        perm = axes ++ Enum.reject(0..(rank - 1)//1, &(&1 in axes))
        gpu_gather(out, Nx.transpose(t, axes: perm), idx, k, eb, ib)

      true ->
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
  @argreduce_f32_spv Path.expand("../../priv/shaders/argreduce_f32.spv", __DIR__)
  @argreduce_f64_spv Path.expand("../../priv/shaders/argreduce_f64.spv", __DIR__)
  @argreduce_s32_spv Path.expand("../../priv/shaders/argreduce_s32.spv", __DIR__)

  # Keyed on the INPUT type only. Unlike reduce_spv/2 there is no (in, out) pair
  # to track, because the output is an index rather than a value — always a
  # 4-byte integer, whatever the input dtype.
  defp argreduce_spv({:f, 32}), do: @argreduce_f32_spv
  defp argreduce_spv({:f, 64}), do: @argreduce_f64_spv
  defp argreduce_spv({:s, 32}), do: @argreduce_s32_spv
  defp argreduce_spv(_), do: nil

  @impl true
  def argmax(out, tensor, opts), do: do_argreduce(out, tensor, opts, 0, :argmax, &Nx.argmax/2)

  @impl true
  def argmin(out, tensor, opts), do: do_argreduce(out, tensor, opts, 2, :argmin, &Nx.argmin/2)

  # `glsl/argreduce_*.comp`. These reuse `reduce_axis/7` verbatim — same
  # bindings, same (outer, reduce_size, inner, op) push layout — so no new NIF
  # was needed even though the output dtype differs from the input's.
  #
  # Nx hands this callback `[tie_break:, axis:, keep_axis:]` with `out.shape`
  # already contracted, so `keep_axis` needs no handling here: it changes the
  # shape Nx computed and not the number of output slots, which stays
  # outer * inner either way.
  #
  # `axis: nil` means reduce EVERYTHING to a flat index. That is the same
  # (1, n, 1) slab `classify_reduce_axes/2` already returns for an all-axes
  # reduction, and the shader's loop variable is then the flat index — so the
  # two cases share one path rather than needing a separate flatten.
  defp do_argreduce(
         %T{shape: out_shape, type: out_type} = out,
         tensor,
         opts,
         base_code,
         op,
         host_fun
       ) do
    t = ensure_on_backend(tensor)
    spv = argreduce_spv(t.type)
    axes = if opts[:axis] == nil, do: all_axes(t.shape), else: [opts[:axis]]
    tie_high = opts[:tie_break] == :high

    fast_path? =
      spv != nil and match?(%__MODULE__{}, t.data) and
        element_bytes(out_type) == 4 and integer_type?(out_type) and
        match?({:ok, _}, classify_reduce_axes(t.shape, axes))

    if fast_path? do
      %T{data: %__MODULE__{ref: a_ref}} = t
      {:ok, {outer, reduce_size, inner}} = classify_reduce_axes(t.shape, axes)
      n_out = max(byte_size_of(out_shape), 1)
      {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n_out * element_bytes(out_type))
      op_code = base_code + if tie_high, do: 1, else: 0

      :ok =
        Nx.Vulkan.NativeV.reduce_axis(out_ref, a_ref, outer, reduce_size, inner, op_code, spv)

      put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: out_type})
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      # Explicit attribution — this helper is shared by argmax and argmin, so the
      # __CALLER__.function capture would name `do_argreduce/6` instead.
      host_result(out, host_fun.(t_bin, opts), {op, 3})
    end
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

  # product: multiplicative reduction. Same shape as sum but with *, and now the
  # same code path — op code 3, added to all three reduce shaders at W5 T2. It
  # was an unconditional host fallback at EVERY dtype before that, f32 included,
  # which is why it did not appear in W5's dtype-gated census bucket.
  @impl true
  def product(out, t, opts), do: do_reduce(out, t, opts, 3)

  # reverse: reverse along given axes. Composes from slice in some
  # cases but a direct callback handles general patterns.
  # W1: type-generic, see the note on @transpose_nd_spv.
  @reverse_nd_spv Path.expand("../../priv/shaders/reverse_nd.spv", __DIR__)

  @impl true
  def reverse(%T{shape: shape, type: type} = out, tensor, axes) do
    t = ensure_on_backend(tensor)
    rank = tuple_size(shape)

    if word_copyable?(type) and rank >= 1 and rank <= 4 and match?(%__MODULE__{}, t.data) and
         t.type == type do
      gpu_reverse(out, t, axes, @reverse_nd_spv)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      host_result(out, Nx.reverse(t_bin, axes: axes))
    end
  end

  # params: [rank, ews, shape[4], rev[4]] — rev[d] is 1 when axis d is reversed.
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
      for v <-
            [rank, div(element_bytes(type), 4)] ++ pad4(Tuple.to_list(shape)) ++ pad4(rev),
          into: <<>> do
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
  # bitcast is a REINTERPRETATION of the same bytes, and Nx raises on mismatched
  # bit widths before dispatch ("cannot bitcast types with different bit sizes"),
  # so this backend only ever sees a same-width relabel. That is metadata, not
  # work — exactly like `reshape/2` above, and for the same reason: the buffer
  # is unchanged and only the type attached to it differs.
  #
  # It was transferring the whole tensor to the host to do nothing to it.
  #
  # The clause is narrow on purpose. An operand that is not resident falls
  # through to the host path rather than being uploaded, because a bitcast is
  # the one op where a round trip buys literally nothing.
  def bitcast(%T{type: type, shape: shape} = out, %T{data: %__MODULE__{ref: ref}}) do
    put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  def bitcast(out, tensor) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    result = Nx.bitcast(t_bin, out.type)
    host_result(out, result, {:bitcast, 2})
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

  # Both were unconditional host fallbacks at every dtype until W5 T2 — not
  # integer gaps at all, despite sitting in the register's @integer_dtype bucket
  # because Nx's doctests for them happen to be s32. Op codes 2 and 3, added to
  # all three window shaders, and they now share window_reduce_op/6's gate with
  # window_max/window_min: rank <= 4, no padding, no dilation.
  @impl true
  def window_sum(out, tensor, dimensions, opts) do
    window_reduce_op(out, tensor, dimensions, opts, 2, &Nx.window_sum/3)
  end

  @impl true
  def window_product(out, tensor, dimensions, opts) do
    window_reduce_op(out, tensor, dimensions, opts, 3, &Nx.window_product/3)
  end

  @window_reduce_f64_spv Path.expand("../../priv/shaders/window_reduce_f64.spv", __DIR__)
  @window_reduce_f32_spv Path.expand("../../priv/shaders/window_reduce_f32.spv", __DIR__)

  @window_reduce_s32_spv Path.expand("../../priv/shaders/window_reduce_s32.spv", __DIR__)

  defp window_reduce_spv({:f, 64}), do: @window_reduce_f64_spv
  defp window_reduce_spv({:f, 32}), do: @window_reduce_f32_spv
  defp window_reduce_spv({:s, 32}), do: @window_reduce_s32_spv
  defp window_reduce_spv(_), do: nil

  @impl true
  def window_max(out, tensor, dimensions, opts) do
    window_reduce_op(out, tensor, dimensions, opts, 0, &Nx.window_max/3)
  end

  @impl true
  def window_min(out, tensor, dimensions, opts) do
    window_reduce_op(out, tensor, dimensions, opts, 1, &Nx.window_min/3)
  end

  # Pooling's forward pass. The GPU path covers the standard case — rank <= 4,
  # no padding, no window dilation — which is what Axon's max_pool/avg_pool
  # emit. Padded or dilated windows still host-fall-back; the shader indexes
  # straight into the source and has no notion of an out-of-bounds element.
  defp window_reduce_op(%T{type: type} = out, tensor, dimensions, opts, op_code, host_fun) do
    t = ensure_on_backend(tensor)
    spv = window_reduce_spv(type)
    rank = tuple_size(t.shape)
    strides = Keyword.get(opts, :strides) || List.duplicate(1, rank)
    padding = Keyword.get(opts, :padding) || :valid
    dilations = Keyword.get(opts, :window_dilations) || List.duplicate(1, rank)
    pad_lo = pad_lo(padding, rank)

    # `no_padding?` and an all-ones dilation check used to be part of this gate,
    # and they were the largest single residual after W5 T2 at 23 doctests —
    # none of them a dtype problem, since the f32 cases were refused identically.
    # The shader handles both now; what is left to refuse is NEGATIVE padding,
    # which Nx allows as a form of cropping and which the skip-out-of-bounds
    # trick cannot express (a negative pad removes real elements rather than
    # adding implicit ones).
    if spv != nil and rank >= 1 and rank <= 4 and match?(%__MODULE__{}, t.data) and
         t.type == type and pad_lo != nil and Enum.all?(dilations, &(&1 >= 1)) do
      gpu_window_reduce(out, t, dimensions, strides, pad_lo, dilations, op_code, spv)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      host_result(out, host_fun.(t_bin, dimensions, opts))
    end
  end

  # The low pad per axis, or nil if this padding config is one the shader cannot
  # express. Nx.Shape.pool resolves :valid/:same into a {lo, hi} list before the
  # backend ever sees it, but :valid is still accepted here because the callback
  # is public and other callers exist.
  #
  # Only `lo` is needed: `hi` affects the OUTPUT SHAPE, which Nx has already
  # computed and handed us, and the shader derives everything else from that.
  defp pad_lo(:valid, rank), do: List.duplicate(0, rank)

  defp pad_lo(list, rank) when is_list(list) and length(list) == rank do
    if Enum.all?(list, fn {lo, hi} -> lo >= 0 and hi >= 0 end) do
      Enum.map(list, fn {lo, _hi} -> lo end)
    end
  end

  defp pad_lo(_, _rank), do: nil

  # Still used by window_scatter_max/6, whose shader is a different design — one
  # thread per INPUT element, which is what lets it avoid float atomics — and
  # genuinely cannot take padding. Kept for that caller alone; window_reduce_op
  # stopped needing it when the reduce shaders learned to skip out-of-bounds.
  defp no_padding?(:valid), do: true
  defp no_padding?(list) when is_list(list), do: Enum.all?(list, &(&1 == {0, 0}))
  defp no_padding?(_), do: false

  # params: [rank, in[4], out[4], win[4], strides[4]]
  # params: [rank, in[4], out[4], win[4], strides[4], pad_lo[4], dil[4]]
  #
  # pad_lo is padded with ZEROS rather than pad4/1's ones — a 1 there would shift
  # every unused axis by one element. It is the one array in this file whose
  # filler is not 1, which is why it does not use pad4/1.
  defp gpu_window_reduce(
         %T{shape: out_shape, type: type} = out,
         %T{shape: in_shape, data: %__MODULE__{ref: a_ref}},
         dimensions,
         strides,
         pad_lo,
         dilations,
         op_code,
         spv
       ) do
    rank = tuple_size(in_shape)
    n = Nx.size(out_shape)
    win = if is_tuple(dimensions), do: Tuple.to_list(dimensions), else: dimensions

    params =
      for v <-
            [rank] ++
              pad4(Tuple.to_list(in_shape)) ++
              pad4(Tuple.to_list(out_shape)) ++
              pad4(win) ++
              pad4(strides) ++
              pad0(pad_lo) ++
              pad4(dilations),
          into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(type))

    :ok = Nx.Vulkan.NativeV.window_reduce(out_ref, a_ref, params_ref, n, rank, op_code, spv)

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
  end

  @impl true
  # `with_binary_backend/1` is load-bearing here, and only became so at W5.
  #
  # `fun` is USER code: Nx.BinaryBackend calls it with scalar tensors built on
  # the DEFAULT backend, which in this suite is this one. Before integer
  # elementwise had shaders, `Nx.max/2` on two s32 scalars fell back and handed
  # back a BinaryBackend tensor, so the reduction worked by accident. Now it
  # stays resident, and BinaryBackend's next `to_binary/1` gets a Vulkano
  # tensor and dies with no clause.
  #
  # Same shape as the `Nx.mode/2` breakage in NEXT.md §1.1: making an op
  # resident does not remove a mixed-backend pair, it moves one somewhere this
  # backend cannot fix it. Pinning the default for the duration of the callback
  # is the fix, and every other fallback that evaluates composed ops already
  # does it.
  def window_reduce(out, tensor, acc, dimensions, opts, fun) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    acc_bin = Nx.backend_transfer(ensure_on_backend(acc), Nx.BinaryBackend)

    result =
      with_binary_backend(fn -> Nx.window_reduce(t_bin, acc_bin, dimensions, opts, fun) end)

    host_result(out, result)
  end

  @scatter_max_f64_spv Path.expand("../../priv/shaders/window_scatter_max_f64.spv", __DIR__)
  @scatter_max_f32_spv Path.expand("../../priv/shaders/window_scatter_max_f32.spv", __DIR__)

  defp scatter_max_spv({:f, 64}), do: @scatter_max_f64_spv
  defp scatter_max_spv({:f, 32}), do: @scatter_max_f32_spv
  defp scatter_max_spv(_), do: nil

  @impl true
  def window_scatter_max(%T{type: type} = out, tensor, source, init_value, dimensions, opts) do
    t = ensure_on_backend(tensor) |> coerce_operand(type)
    # Nx hands init_value in as an {:s, 32} literal — the sixth op in this
    # backend to be blocked by an integer constant, so coerce rather than
    # demand an exact type match.
    src = ensure_on_backend(source) |> coerce_operand(type)
    iv = ensure_on_backend(init_value) |> coerce_operand(type)
    spv = scatter_max_spv(type)
    rank = tuple_size(t.shape)
    win = if is_tuple(dimensions), do: Tuple.to_list(dimensions), else: dimensions
    strides = Keyword.get(opts, :strides) || List.duplicate(1, rank)
    padding = Keyword.get(opts, :padding) || :valid

    # Non-overlapping only: the shader runs one thread per INPUT element, which
    # is what lets it write each output slot exactly once without float atomics
    # (GL_EXT_shader_atomic_float is not guaranteed on the Kepler fleet).
    # Overlapping windows, padding, and non-f32/f64 stay on the host.
    if spv != nil and rank >= 1 and rank <= 4 and
         match?(%__MODULE__{}, t.data) and match?(%__MODULE__{}, src.data) and
         match?(%__MODULE__{}, iv.data) and
         t.type == type and src.type == type and iv.type == type and
         no_padding?(padding) and
         Enum.zip(strides, win) |> Enum.all?(fn {st, w} -> st >= w end) do
      gpu_window_scatter_max(out, t, src, iv, win, strides, spv)
    else
      t_bin = Nx.backend_transfer(t, Nx.BinaryBackend)
      s_bin = Nx.backend_transfer(src, Nx.BinaryBackend)
      iv_bin = Nx.backend_transfer(iv, Nx.BinaryBackend)

      host_result(
        out,
        with_binary_backend(fn ->
          Nx.window_scatter_max(t_bin, s_bin, iv_bin, dimensions, opts)
        end)
      )
    end
  end

  # params: [rank, in[4], win_grid[4], win[4], strides[4]]
  defp gpu_window_scatter_max(
         %T{shape: out_shape, type: type} = out,
         %T{shape: in_shape, data: %__MODULE__{ref: a_ref}},
         %T{shape: src_shape, data: %__MODULE__{ref: src_ref}},
         %T{data: %__MODULE__{ref: iv_ref}},
         win,
         strides,
         spv
       ) do
    rank = tuple_size(in_shape)
    n = Nx.size(in_shape)

    params =
      for v <-
            [rank] ++
              pad4(Tuple.to_list(in_shape)) ++
              pad4(Tuple.to_list(src_shape)) ++
              pad4(win) ++
              pad4(strides),
          into: <<>> do
        <<v::signed-32-little>>
      end

    {:ok, params_ref} = Nx.Vulkan.NativeV.buf_upload(params)
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(type))

    :ok =
      Nx.Vulkan.NativeV.window_scatter_max(
        out_ref,
        a_ref,
        src_ref,
        iv_ref,
        params_ref,
        n,
        rank,
        spv
      )

    put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})
  end

  @impl true
  def window_scatter_min(out, tensor, source, init_value, dimensions, opts) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    s_bin = Nx.backend_transfer(ensure_on_backend(source), Nx.BinaryBackend)
    iv_bin = Nx.backend_transfer(ensure_on_backend(init_value), Nx.BinaryBackend)

    result =
      with_binary_backend(fn -> Nx.window_scatter_min(t_bin, s_bin, iv_bin, dimensions, opts) end)

    host_result(out, result)
  end

  # --- Round 2: generic reduce (1 callback) ---
  # User-supplied function runs on BinaryBackend tensors.
  @impl true
  # Same user-`fun` hazard as window_reduce/6 above — see the note there. This
  # one has not fired yet only because `reduce/5`'s own doctests fall back
  # before the fun runs; the pin closes it rather than waiting for T2 to
  # surface it.
  def reduce(out, tensor, acc, opts, fun) do
    t_bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
    acc_bin = Nx.backend_transfer(ensure_on_backend(acc), Nx.BinaryBackend)
    result = with_binary_backend(fn -> Nx.reduce(t_bin, acc_bin, opts, fun) end)
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

  @matmul_s32_spv Path.expand("../../priv/shaders/matmul_s32.spv", __DIR__)

  # No accumulator policy for integers, unlike the f32 pair above. :f32 and
  # :f64 are both defensible approximations of an exact float sum; on integers
  # only one answer matches BinaryBackend, which wraps mod 2^32.
  defp matmul_spv({:s, 32}), do: @matmul_s32_spv

  defp matmul_spv(_), do: nil

  @matmul_batched_f32_spv Path.expand("../../priv/shaders/matmul_batched_f32.spv", __DIR__)
  @matmul_batched_f64_spv Path.expand("../../priv/shaders/matmul_batched_f64.spv", __DIR__)
  @matmul_batched_s32_spv Path.expand("../../priv/shaders/matmul_batched_s32.spv", __DIR__)

  # The batched family has no f32 accumulator POLICY. `matmul_spv/1` offers the
  # :f32 variant as an opt-in for f64-rate-limited GPUs; the batched shader is
  # f64-accumulating only, because there is no benchmark justifying a second
  # variant and F32_PLAN.md's numbers were measured on the unbatched pair.
  defp matmul_batched_spv({:f, 32}), do: @matmul_batched_f32_spv
  defp matmul_batched_spv({:f, 64}), do: @matmul_batched_f64_spv
  defp matmul_batched_spv({:s, 32}), do: @matmul_batched_s32_spv
  defp matmul_batched_spv(_), do: nil

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

    # ANY unbatched contraction is a matmul once both operands are laid out for
    # it. `dot_orient/6` above rotates the rank-2 cases; this generalises the
    # same idea to every rank and axis count, which is the last structural gap
    # in the dot path and needs no new kernel:
    #
    #   a -> transpose to [free_a..., contracted_a...] -> reshape {M, K}
    #   b -> transpose to [contracted_b..., free_b...] -> reshape {K, N}
    #   matmul -> {M, N} -> reshape to out.shape
    #
    # M, K and N are the PRODUCTS of those dim groups, so a rank-4 contraction
    # over two axes is the same dispatch as a rank-2 one over a single axis.
    #
    # Three things make it correct rather than merely plausible:
    #
    #   * `axes_a[i]` contracts with `axes_b[i]` POSITIONALLY. Putting each list
    #     in its given order on the inside of both operands keeps those pairings
    #     aligned once the group is flattened into K.
    #   * Nx defines the output as a's free dims followed by b's free dims, each
    #     in their original relative order — which is exactly what {M, N} unrolls
    #     to. No output permutation is needed, for the same reason the `gather`
    #     rotation needs none.
    #   * An empty `axes_a` is an outer product, and it falls out rather than
    #     being special-cased: the contracted group is empty, so K is the empty
    #     product 1 and the shapes are {M, 1} and {1, N}.
    #
    # Reshape here is metadata only and a transpose is skipped when the
    # permutation is already the identity, so the common rank-2 case still costs
    # exactly one dispatch.
    # `0..-1//1` is the EMPTY range, which is what a rank-0 operand needs: no
    # axes, so no free axes either. Clamping the rank up to 1 here instead
    # produced `[0]` and then `elem({}, 0)` — Nx.dot(scalar, [], [], vec, [], [])
    # is a real doctest and it crashed rather than falling back.
    free_a = Enum.reject(0..(tuple_size(a_v.shape) - 1)//1, &(&1 in axes_a))
    free_b = Enum.reject(0..(tuple_size(b_v.shape) - 1)//1, &(&1 in axes_b))

    resident? =
      a_v.type == type and b_v.type == type and
        tuple_size(a_v.shape) <= 4 and tuple_size(b_v.shape) <= 4 and
        match?(%__MODULE__{}, a_v.data) and match?(%__MODULE__{}, b_v.data)

    general? = spv != nil and resident? and batched_a == [] and batched_b == []

    # BATCHED contractions are the same reduction with a batch dimension in
    # front. Nx guarantees the batch axes are "successive dimensions starting
    # from 0" and that both sides carry the same count, so the batch is always a
    # leading prefix on both operands and needs no rotation:
    #
    #   a -> [batch..., free_a..., contracted_a...] -> reshape {B, M, K}
    #   b -> [batch..., contracted_b..., free_b...] -> reshape {B, K, N}
    #
    # Half the doctests reaching here are batched only because an operand is
    # `Nx.vectorize`d — Nx turns a vectorised axis into a leading batch axis, so
    # this closes vectorised `dot` as a side effect rather than as a separate
    # feature.
    #
    # The batch rides the third dispatch dimension, capped at 65535 by
    # maxComputeWorkGroupCount[2]. Beyond that it falls back rather than looping
    # per matrix: the per-dispatch cost is what made the vectorised `reduce/5`
    # fold lose to the host, and a loop here would reintroduce it.
    bspv = matmul_batched_spv(type)
    nbatch = batched_a |> Enum.map(&elem(a_v.shape, &1)) |> Enum.product()

    batched? =
      bspv != nil and resident? and batched_a != [] and
        batched_a == Enum.to_list(0..(length(batched_a) - 1)//1) and
        batched_b == Enum.to_list(0..(length(batched_b) - 1)//1) and
        length(batched_a) == length(batched_b) and nbatch <= 65_535

    cond do
      batched? ->
        free_a = Enum.reject(free_a, &(&1 in batched_a))
        free_b = Enum.reject(free_b, &(&1 in batched_b))

        k = axes_a |> Enum.map(&elem(a_v.shape, &1)) |> Enum.product()
        m = free_a |> Enum.map(&elem(a_v.shape, &1)) |> Enum.product()
        n = free_b |> Enum.map(&elem(b_v.shape, &1)) |> Enum.product()

        a2 = dot_flatten(a_v, batched_a ++ free_a ++ axes_a, {nbatch, m, k})
        b2 = dot_flatten(b_v, batched_b ++ axes_b ++ free_b, {nbatch, k, n})

        %T{data: %__MODULE__{ref: a_ref}} = a2
        %T{data: %__MODULE__{ref: b_ref}} = b2

        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(nbatch * m * n * element_bytes(type))

        :ok =
          Nx.Vulkan.NativeV.matmul_batched(out_ref, a_ref, b_ref, nbatch, m, n, k, bspv)

        put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})

      general? ->
        k = axes_a |> Enum.map(&elem(a_v.shape, &1)) |> Enum.product()
        m = free_a |> Enum.map(&elem(a_v.shape, &1)) |> Enum.product()
        n = free_b |> Enum.map(&elem(b_v.shape, &1)) |> Enum.product()

        a2 = dot_flatten(a_v, free_a ++ axes_a, {m, k})
        b2 = dot_flatten(b_v, axes_b ++ free_b, {k, n})

        %T{data: %__MODULE__{ref: a_ref}} = a2
        %T{data: %__MODULE__{ref: b_ref}} = b2

        {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(m * n * element_bytes(type))
        :ok = Nx.Vulkan.NativeV.matmul(out_ref, a_ref, b_ref, m, n, k, spv)

        put_in(out.data, %__MODULE__{ref: out_ref, shape: out_shape, type: type})

      true ->
        a_bin = Nx.backend_transfer(a_v, Nx.BinaryBackend)
        b_bin = Nx.backend_transfer(b_v, Nx.BinaryBackend)
        result = Nx.dot(a_bin, axes_a, batched_a, b_bin, axes_b, batched_b)
        host_result(out, result, {:dot, 7})
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

  # Rank-1 operands are promoted to rank 2 with a length-1 axis, which turns
  # three more contraction shapes into the one the shader already does:
  #
  #   vec·vec  {K}   ·{K}    -> {1,K}·{K,1} -> {1,1}, reshaped to {}
  #   mat·vec  {M,K} ·{K}    -> {M,K}·{K,1} -> {M,1}, reshaped to {M}
  #   vec·mat  {K}   ·{K,N}  -> {1,K}·{K,N} -> {1,N}, reshaped to {N}
  #
  # No new shader and no new dispatch — `out_shape` is whatever Nx computed, and
  # the result buffer is byte-identical either way because a length-1 axis costs
  # nothing in a row-major layout. This is a pure gate widening (skill §1b), and
  # it closes these shapes for FLOATS as well, which is why it is worth more
  # than the integer matmul it shipped alongside: `Nx.dot/2` on two f32 vectors
  # was going to the host with a shader sitting right there.
  #
  # `Nx.reshape/2` on this backend is metadata only, so the promotion is free.
  defp dot_orient(%T{shape: as} = a, [0], %T{shape: bs} = b, [0], [], [])
       when tuple_size(as) == 1 and tuple_size(bs) == 1 do
    {Nx.reshape(a, {1, elem(as, 0)}), [1], Nx.reshape(b, {elem(bs, 0), 1}), [0]}
  end

  defp dot_orient(%T{shape: as} = a, [aa], %T{shape: bs} = b, [0], [], [])
       when tuple_size(as) == 2 and tuple_size(bs) == 1 and aa in [0, 1] do
    a = if aa == 0, do: Nx.transpose(a, axes: [1, 0]), else: a
    {a, [1], Nx.reshape(b, {elem(bs, 0), 1}), [0]}
  end

  defp dot_orient(%T{shape: as} = a, [0], %T{shape: bs} = b, [ba], [], [])
       when tuple_size(as) == 1 and tuple_size(bs) == 2 and ba in [0, 1] do
    b = if ba == 1, do: Nx.transpose(b, axes: [1, 0]), else: b
    {Nx.reshape(a, {1, elem(as, 0)}), [1], b, [0]}
  end

  defp dot_orient(a, axes_a, b, axes_b, _batched_a, _batched_b), do: {a, axes_a, b, axes_b}

  # Permute a tensor's axes into `perm` and flatten to `shape`. The transpose is
  # skipped when `perm` is already the identity — reshape is metadata here, so
  # the identity case costs nothing at all.
  defp dot_flatten(t, perm, shape) do
    rank = tuple_size(t.shape)
    t = if perm == Enum.to_list(0..(rank - 1)//1), do: t, else: Nx.transpose(t, axes: perm)
    Nx.reshape(t, shape)
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
  # Every host fallback lands here (via the host_result/2 macro above, which
  # supplies `op`). Counting centrally is what makes a silent fallback
  # detectable — see Nx.Vulkan.Fallback for why value-based tests structurally
  # cannot catch one. `out` is passed as the strict-mode metadata: under
  # `host_fallback: :raise` it supplies the shape and dtype in the error, and
  # gates the rank-5+ allowlist entries.
  #
  # A fallback is "the result left the device", so only a result whose data is
  # NOT on this backend is recorded. Not every caller of this helper actually
  # host-falls-back: clip/4 composes GPU min/max and stays resident, and was
  # being counted (and, under :raise, refused) for a round trip it never made.
  # Strict mode found that — a censor that cries wolf gets switched off.
  defp host_result_recorded(%T{} = out, %T{} = result, op) do
    unless match?(%__MODULE__{}, result.data), do: Nx.Vulkan.Fallback.note(op, out)
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

  # Non-finite floats arrive as ATOMS. `Nx.Constants.infinity/1`,
  # `neg_infinity/1` and `nan/1` return `:infinity | :neg_infinity | :nan`, and
  # every numeric clause below raises `ArithmeticError` on one — `s / 1.0` and
  # `trunc(s)` alike. That is W3: `Nx.LinAlg.solve(Nx.eye(2), [1.0, 2.0])`
  # raised on a shipped backend, for an op the allowlist documents as supported
  # via the host, because nx composes solve's pivot search through
  # `Nx.Constants.neg_infinity` and the resulting constant landed here.
  #
  # The bit patterns are IEEE-754 and match `Nx.BinaryBackend` byte for byte
  # (checked, not assumed). Writing them as integers with `-native` keeps the
  # byte order correct on either endianness, which `<<0x7F800000::32>>` would
  # not; there is no float literal for these, which is the whole reason nx uses
  # atoms in the first place.
  defp encode_scalar(:infinity, {:f, 16}), do: <<0x7C00::unsigned-16-native>>
  defp encode_scalar(:infinity, {:f, 32}), do: <<0x7F800000::unsigned-32-native>>
  defp encode_scalar(:infinity, {:f, 64}), do: <<0x7FF0000000000000::unsigned-64-native>>
  defp encode_scalar(:neg_infinity, {:f, 16}), do: <<0xFC00::unsigned-16-native>>
  defp encode_scalar(:neg_infinity, {:f, 32}), do: <<0xFF800000::unsigned-32-native>>
  defp encode_scalar(:neg_infinity, {:f, 64}), do: <<0xFFF0000000000000::unsigned-64-native>>
  defp encode_scalar(:nan, {:f, 16}), do: <<0x7E00::unsigned-16-native>>
  defp encode_scalar(:nan, {:f, 32}), do: <<0x7FC00000::unsigned-32-native>>
  defp encode_scalar(:nan, {:f, 64}), do: <<0x7FF8000000000000::unsigned-64-native>>

  # A non-finite atom at any OTHER dtype (integer storage, bf16/f8/complex) has
  # no encoding here. Signal fallback rather than guessing: constant/3 then
  # rebuilds it on BinaryBackend, which is the reference this backend is
  # required to match — including when the reference itself refuses.
  defp encode_scalar(s, _type) when s in [:infinity, :neg_infinity, :nan], do: :error

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
