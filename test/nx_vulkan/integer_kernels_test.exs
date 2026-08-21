defmodule Nx.Vulkan.IntegerKernelsTest do
  @moduledoc """
  W5 T1 — the s32 elementwise, compare and select shaders, and the six places
  where the obvious GLSL gives a different answer from `Nx.BinaryBackend`.

  `glsl/elementwise_binary_s32.comp`, `elementwise_binary_bcast_s32.comp`,
  `elementwise_unary_s32.comp`, `compare_s32.comp`, `select_s32.comp`, plus op
  codes 6-10 added to the two float compare shaders.

  ## Why bit-equality, with no tolerance at all

  Every other kernel in this repo is checked "within dtype eps". On integers
  there is no eps: `Nx.BinaryBackend` and the GPU either agree exactly or one of
  them is wrong. That makes this the strictest correctness surface in the
  backend, and it is why the traps below are worth pinning individually rather
  than trusting a few round-number cases.

  ## The six traps

  Each of these was measured against `Nx.BinaryBackend` before the shader was
  written, not recalled, and each would be a plausible thing for someone to
  "fix" in the wrong direction later:

    1. **s32 arithmetic wraps.** `2e9 + 2e9` is `-294967296`, not saturation and
       not a widened result. This is the OPPOSITE of the f32 reduce shader,
       which accumulates in `double` precisely because BinaryBackend sums floats
       in f64. Same principle — match the reference — opposite conclusion.
    2. **Element-width wrapping.** `{:s, 8}` multiply wraps at 8 bits
       (`100 * 100 = 16`), not at 32. There is no s8 shader, so this one is
       pinned as a *fallback*: the test exists to stop someone widening the gate
       to `rem(element_bytes, 4) == 0` on the theory that a word copy is a word
       copy. Arithmetic is not a copy.
    3. **`remainder` takes the sign of the dividend.** `-7 rem 3 = -1` and
       `7 rem -3 = 1`. GLSL's `%` is *undefined* for negative operands, so the
       shader cannot use it and computes `x - trunc_div(x, y) * y` instead.
    4. **`quotient` truncates toward zero.** `-7 / 3 = -2`, not `-3`. Same GLSL
       caveat, same workaround.
    5. **`count_leading_zeros(0)` is 32.** `findMSB` returns `-1` for zero *and*
       for -1 when given a signed int, and reports the most significant ZERO bit
       for negatives — so the shader counts over the unsigned bit pattern, where
       `31 - findMSB` is exact at all three boundaries.
    6. **Right shift on a negative is arithmetic.** `-8 >> 1 = -4`, sign
       preserved.

  ## Residency is asserted separately, and by refusal

  A host fallback returns a bit-identical result — it *is* `BinaryBackend`, the
  reference these tests compare against — so no value assertion here can see
  whether the kernel ran. `Fallback.strict/1` is used rather than
  `count_total/1 == 0` because the count is a lower bound: `:raise` fires on the
  first refused op and names the *cause*, where a census names only the visible
  edge of a cascade.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.Fallback
  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp gpu(list, type), do: Nx.tensor(list, type: type, backend: VulkanoBackend)
  defp host(list, type), do: Nx.tensor(list, type: type, backend: Nx.BinaryBackend)

  # Run `fun` on both backends and assert the results are byte-identical and the
  # dtypes agree. `fun` takes the tensor constructor so the same expression is
  # built twice, once per backend.
  defp assert_parity(fun) do
    got = fun.(&gpu/2) |> Nx.backend_transfer(Nx.BinaryBackend)
    expected = fun.(&host/2)

    assert Nx.type(got) == Nx.type(expected)
    assert Nx.shape(got) == Nx.shape(expected)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(expected)
    got
  end

  # Parity AND the kernel actually ran. Strict mode raises on the first refused
  # fallback, so a green assertion here means every op in `fun` stayed resident.
  defp assert_parity_and_residency(fun) do
    got = assert_parity(fun)
    Fallback.strict(fn -> fun.(&gpu/2) end)
    got
  end

  describe "the six semantics traps" do
    test "1. s32 arithmetic wraps rather than widening or saturating" do
      # If a future reduce/accumulate shader widens to double "for safety", this
      # is the test that catches it.
      assert_parity_and_residency(fn t ->
        Nx.add(t.([2_000_000_000, -2_000_000_000], {:s, 32}), t.([2_000_000_000, -2_000_000_000], {:s, 32}))
      end)

      assert_parity_and_residency(fn t ->
        Nx.multiply(t.([2_000_000_000], {:s, 32}), t.([3], {:s, 32}))
      end)

      # The literal value, so the intent survives a refactor of the helper.
      got = Nx.add(gpu([2_000_000_000], {:s, 32}), gpu([2_000_000_000], {:s, 32}))
      assert Nx.to_flat_list(got) == [-294_967_296]
    end

    test "2. s8 wraps at EIGHT bits, which is why there is no s8 shader" do
      # Value parity only — s8 has no kernel and must keep falling back. Pinned
      # so that widening the gate to any 4-byte-divisible dtype fails here
      # rather than silently computing 100 * 100 = 10000 in 32-bit registers.
      assert_parity(fn t -> Nx.multiply(t.([100], {:s, 8}), t.([100], {:s, 8})) end)
      assert Nx.to_flat_list(Nx.multiply(gpu([100], {:s, 8}), gpu([100], {:s, 8}))) == [16]
    end

    test "3. remainder takes the sign of the DIVIDEND, in all four combinations" do
      assert_parity_and_residency(fn t ->
        Nx.remainder(t.([7, -7, 7, -7], {:s, 32}), t.([3, 3, -3, -3], {:s, 32}))
      end)

      assert Nx.to_flat_list(
               Nx.remainder(gpu([7, -7, 7, -7], {:s, 32}), gpu([3, 3, -3, -3], {:s, 32}))
             ) == [1, -1, 1, -1]
    end

    test "4. quotient truncates toward zero, in all four combinations" do
      assert_parity_and_residency(fn t ->
        Nx.quotient(t.([7, -7, 7, -7], {:s, 32}), t.([3, 3, -3, -3], {:s, 32}))
      end)

      assert Nx.to_flat_list(
               Nx.quotient(gpu([7, -7, 7, -7], {:s, 32}), gpu([3, 3, -3, -3], {:s, 32}))
             ) == [2, -2, -2, 2]
    end

    test "5. count_leading_zeros at 0, 1 and -1" do
      assert_parity_and_residency(fn t ->
        Nx.count_leading_zeros(t.([0, 1, -1, 255, 256], {:s, 32}))
      end)

      assert Nx.to_flat_list(Nx.count_leading_zeros(gpu([0, 1, -1, 255], {:s, 32}))) ==
               [32, 31, 0, 24]
    end

    test "6. right shift on a negative is arithmetic, not logical" do
      assert_parity_and_residency(fn t ->
        Nx.right_shift(t.([-8, 8, -1], {:s, 32}), t.([1, 1, 1], {:s, 32}))
      end)

      assert Nx.to_flat_list(Nx.right_shift(gpu([-8], {:s, 32}), gpu([1], {:s, 32}))) == [-4]
    end
  end

  describe "elementwise binary, s32" do
    test "the arithmetic four" do
      a = [5, -3, 0, 17]
      b = [2, 4, -1, -6]

      for op <- [:add, :subtract, :multiply, :max, :min] do
        assert_parity_and_residency(fn t -> apply(Nx, op, [t.(a, {:s, 32}), t.(b, {:s, 32})]) end)
      end
    end

    test "bitwise and shifts" do
      a = [12, -1, 0, 255]
      b = [10, 7, 3, 1]

      for op <- [:bitwise_and, :bitwise_or, :bitwise_xor, :left_shift, :right_shift] do
        assert_parity_and_residency(fn t -> apply(Nx, op, [t.(a, {:s, 32}), t.(b, {:s, 32})]) end)
      end
    end

    test "a scalar literal broadcasts without dragging the tensor to the host" do
      # Nx materialises literals as {:s, 32}. Before the bcast s32 shader this
      # was a four-byte constant pulling a whole tensor off the device.
      assert_parity_and_residency(fn t -> Nx.add(t.([1, 2, 3], {:s, 32}), 5) end)
      assert_parity_and_residency(fn t -> Nx.multiply(t.([1, 2, 3], {:s, 32}), -2) end)
    end

    test "rank-2 broadcasting against a row" do
      assert_parity_and_residency(fn t ->
        Nx.add(
          Nx.reshape(t.([1, 2, 3, 4, 5, 6], {:s, 32}), {2, 3}),
          Nx.reshape(t.([10, 20, 30], {:s, 32}), {1, 3})
        )
      end)
    end

    test "integer divide is NOT an integer op — Nx returns f32" do
      # The s32 shader has no divide case on purpose. If this ever starts
      # returning an integer type, the shader needs a code 3 and this test is
      # where that shows up.
      assert Nx.type(Nx.divide(gpu([7], {:s, 32}), gpu([2], {:s, 32}))) == {:f, 32}
    end
  end

  describe "elementwise unary, s32" do
    test "sign, abs, negate" do
      for op <- [:sign, :abs, :negate] do
        assert_parity_and_residency(fn t -> apply(Nx, op, [t.([-9, 0, 9, -1], {:s, 32})]) end)
      end
    end

    test "bitwise_not and population_count" do
      for op <- [:bitwise_not, :population_count] do
        assert_parity_and_residency(fn t -> apply(Nx, op, [t.([0, -1, 5, 255], {:s, 32})]) end)
      end
    end

    test "transcendentals on an integer input go to the FLOAT shader" do
      # Nx runs these through Nx.Type.to_floating/1, so the output template is
      # f32 and the integer shader never sees code 0/1/2. Asserting the type
      # documents why elementwise_unary_s32.comp has no exp case.
      assert Nx.type(Nx.exp(gpu([1, 2], {:s, 32}))) == {:f, 32}
      assert_parity(fn t -> Nx.exp(t.([1, 2], {:s, 32})) end)
    end
  end

  describe "compare and the logical family" do
    test "the six comparisons on s32 produce a u8 mask" do
      for op <- [:equal, :not_equal, :greater, :less, :greater_equal, :less_equal] do
        got =
          assert_parity_and_residency(fn t ->
            apply(Nx, op, [t.([1, 5, 3], {:s, 32}), t.([3, 3, 3], {:s, 32})])
          end)

        assert Nx.type(got) == {:u, 8}
      end
    end

    test "logical_and/or/xor treat any nonzero as true" do
      for op <- [:logical_and, :logical_or, :logical_xor] do
        got =
          assert_parity_and_residency(fn t ->
            apply(Nx, op, [t.([-1, 0, 1], {:s, 32}), t.([1, 1, 0], {:s, 32})])
          end)

        assert Nx.type(got) == {:u, 8}
      end
    end

    test "logical ops across MIXED dtypes, which is a real Nx doctest" do
      # element_wise_pred_op does not merge operand types, so f32-vs-s32 reaches
      # the backend as-is and relies on Nx.Type.merge + coerce_to.
      assert_parity_and_residency(fn t ->
        Nx.logical_and(t.([-1.0, 0.0, 1.0], {:f, 32}), t.([1, 1, 0], {:s, 32}))
      end)
    end

    test "is_nan and is_infinity are constant false on integers" do
      # Not a stub: neither value is representable in s32, so false is the
      # answer BinaryBackend gives too.
      for op <- [:is_nan, :is_infinity] do
        got = assert_parity_and_residency(fn t -> apply(Nx, op, [t.([1, 0, -3], {:s, 32})]) end)
        assert Nx.to_flat_list(got) == [0, 0, 0]
      end
    end

    test "is_nan and is_infinity still detect the real thing on f32" do
      assert_parity_and_residency(fn t ->
        Nx.is_nan(t.([:nan, 1.0, :infinity], {:f, 32}))
      end)

      assert_parity_and_residency(fn t ->
        Nx.is_infinity(t.([:nan, 1.0, :infinity, :neg_infinity], {:f, 32}))
      end)
    end
  end

  describe "select with integer branches" do
    test "a u8 mask selects between s32 branches" do
      assert_parity_and_residency(fn t ->
        Nx.select(
          Nx.greater(t.([1, 5, 3], {:s, 32}), t.([3, 3, 3], {:s, 32})),
          t.([10, 20, 30], {:s, 32}),
          t.([90, 80, 70], {:s, 32})
        )
      end)
    end

    test "select broadcasts its branches" do
      assert_parity_and_residency(fn t ->
        Nx.select(Nx.greater(t.([1, 5], {:s, 32}), 3), t.([10, 20], {:s, 32}), 0)
      end)
    end
  end

  describe "dtypes that must keep falling back" do
    # Each of these is a decision, not an oversight: T1 is a 32-bit job, and a
    # kernel for these dtypes would need Int64 or an 8/16-bit storage extension
    # that the Kepler fleet does not guarantee. Value parity is still asserted —
    # the host path is correct, which is the whole point of having one.
    test "s64, u32, s16 and s8 arithmetic is correct on the host" do
      for type <- [{:s, 64}, {:u, 32}, {:s, 16}, {:s, 8}] do
        assert_parity(fn t -> Nx.add(t.([1, 2, 3], type), t.([10, 20, 30], type)) end)
        assert_parity(fn t -> Nx.max(t.([1, 9, 3], type), t.([5, 5, 5], type)) end)
      end
    end
  end
end
