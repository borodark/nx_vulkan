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
  # `:allow` is deliberate and load-bearing. This helper asserts VALUE parity and
  # makes no residency claim, so it must work for the cases that are supposed to
  # fall back — s8/s64/u32 arithmetic, negative padding, batched dot. Under
  # `sh scripts/strict_test.sh` the whole suite runs with fallbacks refused, and
  # without this scope those tests would raise on precisely the behaviour they
  # exist to pin. Residency is asserted separately, and strictly, below.
  #
  # INTEGER results are compared EXACTLY and float results within eps, and the
  # split is not fussiness — it is the difference between the two bars this file
  # tests against. On integers there is no eps: the GPU and BinaryBackend either
  # agree bit-for-bit or one is wrong, which is what makes every wrap and
  # sign-convention trap here checkable at all. On floats the GPU is allowed to
  # differ, and Vulkan says by how much.
  #
  # This distinction was found the hard way, on mac-247: `Nx.sqrt` of an s32 9
  # is exactly 3.0 on Ampere and 3.000000238418579 on the Kepler GT 650M. Vulkan
  # permits `sqrt` up to 3 ULP of error and Kepler spends that budget where
  # Ampere does not. An exact float comparison here passed on one box and failed
  # on another, which is the whole argument for running this suite across the
  # fleet rather than trusting the box it was written on.
  defp assert_parity(fun) do
    got =
      Fallback.strict(:allow, fn -> fun.(&gpu/2) end)
      |> Nx.backend_transfer(Nx.BinaryBackend)

    expected = fun.(&host/2)

    assert Nx.type(got) == Nx.type(expected)
    assert Nx.shape(got) == Nx.shape(expected)

    case Nx.type(got) do
      {f, _} when f in [:f, :bf] ->
        assert Nx.to_flat_list(got) == Nx.to_flat_list(expected) or
                 Nx.to_number(Nx.all_close(got, expected, rtol: 1.0e-5, atol: 1.0e-8)) == 1,
               "float parity: got #{inspect(Nx.to_flat_list(got))}, " <>
                 "expected #{inspect(Nx.to_flat_list(expected))}"

      _ ->
        assert Nx.to_flat_list(got) == Nx.to_flat_list(expected)
    end

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
        Nx.add(
          t.([2_000_000_000, -2_000_000_000], {:s, 32}),
          t.([2_000_000_000, -2_000_000_000], {:s, 32})
        )
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

      got =
        Fallback.strict(:allow, fn -> Nx.multiply(gpu([100], {:s, 8}), gpu([100], {:s, 8})) end)

      assert Nx.to_flat_list(got) == [16]
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

    test "transcendentals on an integer input are COERCED, not host-fallen-back" do
      # Nx runs these through Nx.Type.to_floating/1, so the output template is
      # f32 and the integer shader never sees code 0/1/2. Asserting the type
      # documents why elementwise_unary_s32.comp has no exp case.
      assert Nx.type(Nx.exp(gpu([1, 2], {:s, 32}))) == {:f, 32}

      # It used to fall back here, pinned at exactly 1, and closing that gate is
      # what this assertion was for — it fired the moment coerce_to/2 was wired
      # into the unary path. Now the operand is cast s32 -> f32 on the device
      # and the whole expression stays resident.
      assert_parity_and_residency(fn t -> Nx.exp(t.([1, 2], {:s, 32})) end)
      assert_parity_and_residency(fn t -> Nx.sqrt(t.([4, 9], {:s, 32})) end)
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

  describe "T2 — axis reductions" do
    test "sum, product, reduce_max and reduce_min on s32" do
      for op <- [:sum, :product, :reduce_max, :reduce_min] do
        assert_parity_and_residency(fn t -> apply(Nx, op, [t.([3, -9, 7, 2], {:s, 32})]) end)
      end
    end

    test "trap 1 again, in the accumulator: sum and product WRAP" do
      # The single most important assertion in T2. reduce_axis_f32.comp
      # accumulates in `double` because BinaryBackend sums floats in f64; the
      # s32 shader must NOT, because BinaryBackend computes integers mod 2^32.
      # A `double acc` here would return 4000000000 and 6000000000 — nicer, and
      # not what the reference says.
      assert_parity_and_residency(fn t ->
        Nx.sum(t.([2_000_000_000, 2_000_000_000], {:s, 32}))
      end)

      assert_parity_and_residency(fn t -> Nx.product(t.([2_000_000_000, 3], {:s, 32})) end)

      assert Nx.to_number(Nx.sum(gpu([2_000_000_000, 2_000_000_000], {:s, 32}))) ==
               -294_967_296

      assert Nx.to_number(Nx.product(gpu([2_000_000_000, 3], {:s, 32}))) == 1_705_032_704
    end

    test "reducing one axis of a rank-2 tensor" do
      for axes <- [[0], [1]] do
        assert_parity_and_residency(fn t ->
          Nx.sum(Nx.reshape(t.([1, 2, 3, 4, 5, 6], {:s, 32}), {2, 3}), axes: axes)
        end)
      end
    end

    test "f32 product needs a WIDE accumulator, unlike the integer one" do
      # 1e20 * 1e20 overflows f32 to `inf` on the first multiply, but
      # BinaryBackend returns 1.00000002e20 — so the intermediate is f64 even
      # though inputs and result are f32. The mirror image of the trap above:
      # same rule (match the reference), opposite implementation.
      got =
        assert_parity_and_residency(fn t ->
          Nx.product(t.([1.0e20, 1.0e20, 1.0e-20], {:f, 32}))
        end)

      refute Nx.to_number(got) == :infinity
    end

    test "product on a u8 mask widens to u32" do
      # This arm used to be `min`'s by fallthrough, so Nx.product on a u8 tensor
      # answered the minimum. Nx's own doctest caught it.
      assert_parity_and_residency(fn t -> Nx.product(t.([[10, 20], [30, 40]], {:u, 8})) end)
      assert Nx.to_number(Nx.product(gpu([[10, 20], [30, 40]], {:u, 8}))) == 240_000
    end
  end

  describe "T2 — window reductions" do
    test "window_sum, window_product, window_max and window_min on s32" do
      for op <- [:window_sum, :window_product, :window_max, :window_min] do
        assert_parity_and_residency(fn t -> apply(Nx, op, [t.([1, 5, 3, 4], {:s, 32}), {2}]) end)
      end
    end

    test "window_sum wraps too, and does NOT widen narrow integers" do
      assert_parity_and_residency(fn t ->
        Nx.window_sum(t.([2_000_000_000, 2_000_000_000], {:s, 32}), {2})
      end)

      # Nx widens `sum` on {:s, 8} to {:s, 32} but leaves `window_sum` at
      # {:s, 8}. A per-OP rule, not a per-dtype one — which is why reduce_spv/2
      # is keyed on the (in, out) pair rather than on the input type.
      assert Nx.type(Fallback.strict(:allow, fn -> Nx.sum(gpu([1, 2, 3], {:s, 8})) end)) ==
               {:s, 32}

      assert Nx.type(
               Fallback.strict(:allow, fn -> Nx.window_sum(gpu([1, 2, 3], {:s, 8}), {2}) end)
             ) ==
               {:s, 8}
    end

    test "window_sum and window_product on f32 now run on the GPU too" do
      # Neither had a shader at ANY dtype before T2, so these were never
      # integer gaps despite living in the register's @integer_dtype bucket.
      for op <- [:window_sum, :window_product] do
        assert_parity_and_residency(fn t ->
          apply(Nx, op, [t.([1.0, 5.0, 3.0, 4.0], {:f, 32}), {2}])
        end)
      end
    end

    test "a rank-2 window" do
      assert_parity_and_residency(fn t ->
        Nx.window_sum(Nx.reshape(t.([1, 2, 3, 4, 5, 6, 7, 8, 9], {:s, 32}), {3, 3}), {2, 2})
      end)
    end

    test "padded windows run on the GPU, for every op and every dtype" do
      # This gate used to refuse them — 23 doctests, and never a dtype problem
      # since the f32 cases were refused identically. Nx pads with the OP'S
      # IDENTITY, and for all four ops skipping an out-of-bounds element is the
      # same as combining with that identity, so the shader needs no literals.
      for op <- [:window_sum, :window_product, :window_max, :window_min] do
        assert_parity_and_residency(fn t ->
          apply(Nx, op, [t.([1, 2, 3], {:s, 32}), {2}, [padding: [{1, 1}]]])
        end)

        assert_parity_and_residency(fn t ->
          apply(Nx, op, [t.([1.0, 2.0, 3.0], {:f, 32}), {2}, [padding: :same]])
        end)
      end
    end

    test "dilated windows run on the GPU" do
      for op <- [:window_sum, :window_max] do
        assert_parity_and_residency(fn t ->
          apply(Nx, op, [t.([1, 2, 3, 4], {:s, 32}), {2}, [window_dilations: [2]]])
        end)
      end

      assert_parity_and_residency(fn t ->
        Nx.window_max(Nx.reshape(t.([1, 2, 3, 4, 5, 6, 7, 8, 9], {:s, 32}), {3, 3}), {2, 2},
          window_dilations: [2, 2]
        )
      end)
    end

    test "padding combines with strides, and with rank 2" do
      assert_parity_and_residency(fn t ->
        Nx.window_sum(t.([1, 2, 3, 4], {:s, 32}), {2}, padding: [{1, 1}], strides: [2])
      end)

      assert_parity_and_residency(fn t ->
        Nx.window_sum(Nx.reshape(t.([1, 2, 3, 4, 5, 6, 7, 8, 9], {:s, 32}), {3, 3}), {2, 2},
          padding: [{1, 0}, {0, 1}]
        )
      end)
    end

    test "a window that is ENTIRELY padding returns the identity" do
      # The edge the skip-out-of-bounds design misses: with nothing to seed
      # from, max/min have to name -inf/+inf (INT_MIN/INT_MAX on s32)
      # explicitly. Reachable whenever a pad is at least as wide as the window.
      # Every value assertion above passed WITHOUT this handling, because every
      # window they used touched at least one real element — a differential
      # test found it, reading did not.
      assert_parity_and_residency(fn t ->
        Nx.window_max(t.([1, 2, 3], {:s, 32}), {2}, padding: [{2, 2}])
      end)

      assert_parity_and_residency(fn t ->
        Nx.window_min(t.([1, 2, 3], {:s, 32}), {2}, padding: [{2, 2}])
      end)

      assert_parity_and_residency(fn t ->
        Nx.window_max(t.([1.0, 2.0], {:f, 32}), {2}, padding: [{2, 2}])
      end)

      got = Nx.window_max(gpu([1, 2, 3], {:s, 32}), {2}, padding: [{2, 2}])
      assert Nx.to_flat_list(got) == [-2_147_483_648, 1, 2, 3, 3, -2_147_483_648]

      f = Nx.window_max(gpu([1.0, 2.0], {:f, 32}), {2}, padding: [{2, 2}])
      assert Nx.to_flat_list(f) == [:neg_infinity, 1.0, 2.0, 2.0, :neg_infinity]
    end

    test "NEGATIVE padding still falls back" do
      # Nx allows a negative pad as a form of cropping, which removes real
      # elements rather than adding implicit ones — the skip-out-of-bounds
      # trick cannot express it. `pad_lo/2` returns nil and the op goes to the
      # host, which is correct.
      assert_parity(fn t ->
        Nx.window_sum(t.([1, 2, 3, 4], {:s, 32}), {2}, padding: [{-1, 0}])
      end)
    end
  end

  describe "scatter — indexed_put and indexed_add" do
    test "indexed_put writes only what the indices name" do
      # The elements NOT named have to survive, which is why the output is
      # seeded with a copy of the target rather than a zeroed buffer.
      assert_parity_and_residency(fn t ->
        Nx.indexed_put(t.([9, 8, 7, 6], {:s, 32}), t.([[1]], {:s, 32}), t.([99], {:s, 32}))
      end)
    end

    test "indexed_put at f32, f64 and s32" do
      for type <- [{:s, 32}, {:f, 32}, {:f, 64}] do
        assert_parity_and_residency(fn t ->
          Nx.indexed_put(t.([0, 0, 0], type), t.([[1], [2]], {:s, 32}), t.([2, 4], type))
        end)
      end
    end

    test "indexed_put on a rank-3 tensor, and writing whole blocks" do
      assert_parity_and_residency(fn t ->
        Nx.indexed_put(
          Nx.reshape(t.([0, 1, 2, 3, 4, 5], {:s, 32}), {1, 2, 3}),
          t.([[0, 0, 0], [0, 1, 1], [0, 0, 2]], {:s, 32}),
          t.([1, 3, -2], {:s, 32})
        )
      end)

      # K < rank: each index row names a leading coord and a contiguous block of
      # `count` elements is written.
      assert_parity_and_residency(fn t ->
        Nx.indexed_put(
          Nx.reshape(t.([0, 1, 2, 3, 4, 5], {:s, 32}), {2, 3}),
          t.([[1]], {:s, 32}),
          Nx.reshape(t.([7, 8, 9], {:s, 32}), {1, 3})
        )
      end)
    end

    test "indexed_add ACCUMULATES duplicate indices — the atomic" do
      # The one behavioural difference between the two ops. indexed_put
      # documents its race; indexed_add must be deterministic, and that is what
      # the integer atomicAdd buys.
      got =
        assert_parity_and_residency(fn t ->
          Nx.indexed_add(
            t.([0, 0, 0], {:s, 32}),
            t.([[0], [0], [0], [1]], {:s, 32}),
            t.([1, 2, 3, 4], {:s, 32})
          )
        end)

      assert Nx.to_flat_list(got) == [6, 4, 0]
    end

    test "indexed_add wraps at s32, like every other integer op here" do
      assert_parity_and_residency(fn t ->
        Nx.indexed_add(
          t.([2_000_000_000], {:s, 32}),
          t.([[0], [0]], {:s, 32}),
          t.([2_000_000_000, 1], {:s, 32})
        )
      end)
    end

    test "FLOAT indexed_add stays on the host — a decision, not a gap" do
      # An f32 atomicAdd needs GL_EXT_shader_atomic_float, which the Kepler
      # fleet does not guarantee. Same constraint that keeps overlapping pooling
      # backward on the host. Value parity still holds; only residency differs.
      assert_parity(fn t ->
        Nx.indexed_add(t.([1.0], {:f, 32}), t.([[0], [0]], {:s, 32}), t.([1.0, 1.0], {:f, 32}))
      end)

      assert Fallback.strict(:allow, fn ->
               Fallback.count_total(fn ->
                 Nx.indexed_add(gpu([1.0], {:f, 32}), gpu([[0]], {:s, 32}), gpu([1.0], {:f, 32}))
               end)
             end) == 1
    end

    test "Nx PROMOTES both target and updates, and both are coerced" do
      # Nx's own doctests for indexed_add cover both directions. Requiring exact
      # type equality refused them, and that is where Nx.LinAlg.invert/1 fell
      # back at the last step of an otherwise-resident chain.
      assert_parity_and_residency(fn t ->
        Nx.indexed_put(t.([1.0, 2.0], {:f, 32}), t.([[0]], {:s, 32}), t.([9], {:s, 32}))
      end)

      # For indexed_ADD the same promotion goes the other way: an s32 target
      # with f32 updates promotes to an f32 RESULT, and a float indexed_add is
      # the atomic case that stays on the host. So this one is parity-only —
      # the coercion is not what refuses it, the output dtype is.
      assert_parity(fn t ->
        Nx.indexed_add(t.([1], {:s, 32}), t.([[0], [0]], {:s, 32}), t.([1.0, 1.0], {:f, 32}))
      end)

      # An integer-in, integer-out promotion DOES stay resident.
      assert_parity_and_residency(fn t ->
        Nx.indexed_add(t.([1, 1], {:s, 32}), t.([[0], [0]], {:s, 32}), t.([2, 3], {:s, 32}))
      end)
    end

    test "Nx.LinAlg.invert/1 no longer falls back at indexed_put" do
      # The motivation MISSION §3.3 records: invert composes at the Nx level, so
      # with_binary_backend/1 never sees it and it died at indexed_put/5. What
      # is left is the two allowlisted LinAlg blocks and nothing else.
      a = Nx.tensor([[2.0, 0.0], [0.0, 4.0]], backend: VulkanoBackend)

      {_res, counts} =
        Fallback.strict(:allow, fn -> Fallback.count(fn -> Nx.LinAlg.invert(a) end) end)

      refute Map.has_key?(counts, {:indexed_put, 5})
      refute Map.has_key?(counts, {:scatter_op, 8})
    end

    test "non-prefix axes still fall back" do
      # Shared with gather: the shader's strides assume the indexed axes are the
      # leading prefix [0..K-1]. Anything else needs a transpose first.
      assert_parity(fn t ->
        Nx.indexed_put(
          Nx.reshape(t.([0, 1, 2, 3, 4, 5], {:s, 32}), {2, 3}),
          t.([[1]], {:s, 32}),
          Nx.reshape(t.([7, 8], {:s, 32}), {1, 2}),
          axes: [1]
        )
      end)
    end
  end

  describe "argmax / argmin" do
    defp r3(t, type),
      do: Nx.reshape(t.([4, 2, 3, 1, -5, 3, 6, 2, 3, 4, 8, 3], type), {2, 2, 3})

    test "flat (no :axis) returns a FLAT index" do
      for type <- [{:s, 32}, {:f, 32}, {:f, 64}] do
        assert_parity_and_residency(fn t -> Nx.argmax(r3(t, type)) end)
        assert_parity_and_residency(fn t -> Nx.argmin(r3(t, type)) end)
      end
    end

    test "along an axis, with and without :keep_axis" do
      for axis <- [0, 2], keep <- [false, true] do
        assert_parity_and_residency(fn t ->
          Nx.argmax(r3(t, {:s, 32}), axis: axis, keep_axis: keep)
        end)

        assert_parity_and_residency(fn t ->
          Nx.argmin(r3(t, {:f, 32}), axis: axis, keep_axis: keep)
        end)
      end
    end

    test ":tie_break — :low keeps the FIRST extreme, :high the LAST" do
      # Invisible on any input without duplicates, which most test data is.
      assert Nx.to_number(Nx.argmax(gpu([1, 3, 3, 2], {:s, 32}))) == 1
      assert Nx.to_number(Nx.argmax(gpu([1, 3, 3, 2], {:s, 32}), tie_break: :high)) == 2
      assert Nx.to_number(Nx.argmin(gpu([2, 1, 1, 3], {:s, 32}))) == 1
      assert Nx.to_number(Nx.argmin(gpu([2, 1, 1, 3], {:s, 32}), tie_break: :high)) == 2

      for tb <- [:low, :high] do
        assert_parity_and_residency(fn t ->
          Nx.argmax(t.([1, 3, 3, 2], {:s, 32}), tie_break: tb)
        end)

        assert_parity_and_residency(fn t ->
          Nx.argmin(t.([2, 1, 1, 3], {:s, 32}), tie_break: tb)
        end)

        assert_parity_and_residency(fn t -> Nx.argmax(t.([5, 5, 5], {:s, 32}), tie_break: tb) end)
      end
    end

    test "NaN is absorbing, and it is LAST-NaN-wins" do
      # BinaryBackend's rule is one line — `x == :nan or comparator.(...)` — and
      # IEEE comparison gets both halves wrong on its own, because `v < best`
      # and `v > best` are FALSE for any NaN operand. Without the special case
      # the shader reports index 0 for all of these.
      for f <- [&Nx.argmax/2, &Nx.argmin/2] do
        # A NaN CANDIDATE always replaces the incumbent...
        assert_parity_and_residency(fn t -> f.(t.([2.0, :nan, 4.0], {:f, 32}), []) end)
        # ...including another NaN, so this is 2 even at the default :low.
        assert_parity_and_residency(fn t -> f.(t.([:nan, 5.0, :nan], {:f, 32}), []) end)
        # ...and a NaN INCUMBENT is unbeatable by any number.
        assert_parity_and_residency(fn t -> f.(t.([:nan, 5.0, 1.0], {:f, 32}), []) end)
      end

      assert Nx.to_number(Nx.argmax(gpu([:nan, 5.0, :nan], {:f, 32}))) == 2
      assert Nx.to_number(Nx.argmin(gpu([:nan, 5.0, 1.0], {:f, 32}))) == 0
    end

    test "infinities need no special case — IEEE ordering is already right" do
      assert_parity_and_residency(fn t -> Nx.argmax(t.([1.0, :infinity, 2.0], {:f, 32})) end)
      assert_parity_and_residency(fn t -> Nx.argmin(t.([1.0, :neg_infinity, 2.0], {:f, 32})) end)
      assert_parity_and_residency(fn t -> Nx.argmax(t.([1.0, :nan, :infinity], {:f, 32})) end)
    end

    test "a non-default :type still works" do
      assert_parity_and_residency(fn t -> Nx.argmax(t.([1, 9, 3], {:s, 32}), type: {:u, 32}) end)
    end

    test "an s64 input falls back — no shader, and no Int64 capability" do
      assert_parity(fn t -> Nx.argmax(t.([1, 9, 3], {:s, 64})) end)
    end
  end

  describe "all / any" do
    test "flat, over every supported input dtype" do
      for type <- [{:s, 32}, {:f, 32}, {:f, 64}] do
        assert_parity_and_residency(fn t -> Nx.all(t.([0, 1, 2], type)) end)
        assert_parity_and_residency(fn t -> Nx.all(t.([1, 2, 3], type)) end)
        assert_parity_and_residency(fn t -> Nx.any(t.([0, 0, 2], type)) end)
        assert_parity_and_residency(fn t -> Nx.any(t.([0, 0, 0], type)) end)
      end
    end

    test "over an axis, with and without :keep_axes" do
      m = fn t -> Nx.reshape(t.([-1, 0, 1, 2, 3, 4], {:s, 32}), {2, 3}) end

      for axes <- [[0], [1]], keep <- [false, true] do
        assert_parity_and_residency(fn t -> Nx.all(m.(t), axes: axes, keep_axes: keep) end)
        assert_parity_and_residency(fn t -> Nx.any(m.(t), axes: axes, keep_axes: keep) end)
      end
    end

    test "on a u8 MASK, which is the whole point of the u8 entry" do
      # `Nx.all(Nx.greater(a, b))` is the natural idiom and `greater` already
      # emits a u8 mask on the GPU. Without an allany_u8 shader the mask would
      # be dragged back to the host purely to be summarised — the same lesson
      # T12's {:u, 8} -> {:u, 32} sum entry records.
      assert_parity_and_residency(fn t ->
        Nx.all(Nx.greater(t.([1, 5, 3], {:s, 32}), t.([0, 0, 0], {:s, 32})))
      end)

      assert_parity_and_residency(fn t ->
        Nx.any(Nx.greater(t.([1, 5, 3], {:s, 32}), t.([9, 9, 9], {:s, 32})))
      end)
    end

    test "NaN is TRUTHY, and needs no special case" do
      # `NaN != 0.0` is true in IEEE and BinaryBackend agrees, so unlike
      # argreduce_*.comp — where NaN had to be handled explicitly — the plain
      # comparison is already right here.
      assert_parity_and_residency(fn t -> Nx.all(t.([:nan, 1.0], {:f, 32})) end)
      assert Nx.to_number(Nx.all(gpu([:nan, 1.0], {:f, 32}))) == 1
    end

    test "an output wider than one packed word" do
      # The output is written 4 results per u32 word, so anything past 4 slots
      # exercises the packing rather than a single-word special case.
      assert_parity_and_residency(fn t ->
        Nx.all(Nx.reshape(t.(Enum.to_list(1..20), {:s, 32}), {2, 10}), axes: [0])
      end)

      assert_parity_and_residency(fn t ->
        Nx.any(Nx.reshape(t.(List.duplicate(0, 20), {:s, 32}), {2, 10}), axes: [0])
      end)
    end
  end

  describe "dot — the s32 matmul and rank-1 promotion" do
    test "rank-2 x rank-2 on s32" do
      m22 = fn t, ty -> Nx.reshape(t.([1, 2, 3, 4], ty), {2, 2}) end

      for type <- [{:s, 32}, {:f, 32}, {:f, 64}] do
        assert_parity_and_residency(fn t -> Nx.dot(m22.(t, type), m22.(t, type)) end)
      end
    end

    test "the s32 accumulator WRAPS, like every other integer kernel here" do
      assert_parity_and_residency(fn t ->
        Nx.dot(t.([2_000_000_000, 2_000_000_000], {:s, 32}), t.([2, 2], {:s, 32}))
      end)

      assert Nx.to_number(
               Nx.dot(gpu([2_000_000_000, 2_000_000_000], {:s, 32}), gpu([2, 2], {:s, 32}))
             ) == -589_934_592
    end

    test "rank-1 operands are promoted rather than refused — and it helps FLOATS" do
      # vec·vec, mat·vec and vec·mat all become the (M,K)·(K,N) the shader
      # already does, by adding a length-1 axis that costs nothing in a
      # row-major layout. No new shader, no new dispatch — a pure gate widening.
      # `Nx.dot/2` on two f32 vectors was going to the host with the matmul
      # shader sitting right there.
      m22 = fn t, ty -> Nx.reshape(t.([1, 2, 3, 4], ty), {2, 2}) end

      for type <- [{:s, 32}, {:f, 32}] do
        assert_parity_and_residency(fn t -> Nx.dot(t.([1, 2, 3], type), t.([4, 5, 6], type)) end)
        assert_parity_and_residency(fn t -> Nx.dot(m22.(t, type), t.([5, 6], type)) end)
        assert_parity_and_residency(fn t -> Nx.dot(t.([5, 6], type), m22.(t, type)) end)
      end
    end

    test "a non-square matmul, so the tiling is not exercised only at K = 16" do
      assert_parity_and_residency(fn t ->
        Nx.dot(
          Nx.reshape(t.(Enum.to_list(1..20), {:s, 32}), {4, 5}),
          Nx.reshape(t.(Enum.to_list(1..15), {:s, 32}), {5, 3})
        )
      end)
    end

    test "batched and higher-rank contractions still fall back" do
      # Not a dtype gap: these need a real tensordot/batched-matmul
      # generalisation, and the f32 cases are refused identically. 11 doctests.
      assert_parity(fn t ->
        Nx.dot(
          Nx.reshape(t.(Enum.to_list(1..8), {:s, 32}), {2, 2, 2}),
          Nx.reshape(t.(Enum.to_list(1..8), {:s, 32}), {2, 2, 2})
        )
      end)
    end
  end

  describe "stack — routed to concatenate, not a kernel" do
    test "every axis, including the new trailing one" do
      m = fn t -> Nx.reshape(t.([1, 2, 3, 4], {:s, 32}), {2, 2}) end
      n = fn t -> Nx.reshape(t.([5, 6, 7, 8], {:s, 32}), {2, 2}) end

      for axis <- [0, 1, 2] do
        assert_parity_and_residency(fn t -> Nx.stack([m.(t), n.(t)], axis: axis) end)
      end
    end

    test "scalars, vectors and rank 3" do
      assert_parity_and_residency(fn t ->
        Nx.stack([t.(1, {:s, 32}), t.(2, {:s, 32}), t.(3, {:s, 32})])
      end)

      assert_parity_and_residency(fn t ->
        Nx.stack([t.([1, 2, 3], {:s, 32}), t.([4, 5, 6], {:s, 32})], axis: 1)
      end)

      assert_parity_and_residency(fn t ->
        Nx.stack(
          [
            Nx.reshape(t.(Enum.to_list(1..8), {:s, 32}), {2, 2, 2}),
            Nx.reshape(t.(Enum.to_list(9..16), {:s, 32}), {2, 2, 2})
          ],
          axis: 1
        )
      end)
    end

    test "more than two operands" do
      assert_parity_and_residency(fn t ->
        Nx.stack([t.([1, 2], {:s, 32}), t.([3, 4], {:s, 32}), t.([5, 6], {:s, 32})], axis: 1)
      end)
    end

    test "MIXED operand types are coerced on the device" do
      # Nx merges the types before dispatch, so an operand can arrive narrower
      # than out.type. coerce_to/2 casts it here rather than sending the whole
      # stack to the host — the same thing BinaryBackend's own as_type call does.
      assert_parity_and_residency(fn t ->
        Nx.stack([t.([1, 2], {:s, 32}), t.([3.0, 4.0], {:f, 32})])
      end)
    end
  end

  describe "gather — off-prefix axes are rotated, not refused" do
    defp m23(t, type), do: Nx.reshape(t.([1, 2, 3, 4, 5, 6], type), {2, 3})
    defp r234(t, type), do: Nx.reshape(t.(Enum.to_list(1..24), type), {2, 3, 4})

    test "take along every axis of a rank-2 and a rank-3 source" do
      assert_parity_and_residency(fn t ->
        Nx.take(m23(t, {:s, 32}), t.([2, 0], {:s, 32}), axis: 1)
      end)

      for axis <- [0, 1, 2] do
        assert_parity_and_residency(fn t ->
          Nx.take(r234(t, {:s, 32}), t.([1, 0], {:s, 32}), axis: axis)
        end)
      end
    end

    test "every axes combination a rank-3 gather can name" do
      # The rotation has to preserve the ORDER of the non-indexed dims, which is
      # what makes a back-transpose unnecessary. A wrong permutation shows up
      # here and nowhere else.
      cases = [
        {[1], [[0], [2]]},
        {[2], [[3], [0]]},
        {[0, 2], [[0, 3], [1, 1]]},
        {[1, 2], [[0, 3], [2, 1]]},
        {[0, 1], [[0, 2], [1, 1]]}
      ]

      for {axes, idx} <- cases do
        flat = List.flatten(idx)
        shape = {length(idx), length(hd(idx))}

        assert_parity_and_residency(fn t ->
          Nx.gather(r234(t, {:s, 32}), Nx.reshape(t.(flat, {:s, 32}), shape), axes: axes)
        end)
      end
    end

    test "rotation works for floats as well as integers" do
      assert_parity_and_residency(fn t ->
        Nx.gather(m23(t, {:f, 32}), Nx.reshape(t.([2, 0], {:s, 32}), {2, 1}), axes: [1])
      end)
    end

    test "a leading-prefix gather still takes the direct path" do
      # No transpose should be inserted when the axes already lead.
      assert_parity_and_residency(fn t ->
        Nx.gather(m23(t, {:s, 32}), Nx.reshape(t.([0, 2, 1, 1], {:s, 32}), {2, 2}), axes: [0, 1])
      end)
    end
  end

  describe "bitcast — a relabel, not a conversion" do
    test "same-width reinterpretation across every pair Nx allows" do
      # Nx raises on mismatched bit widths before dispatch, so the backend only
      # ever sees a same-width relabel of the same bytes.
      assert_parity_and_residency(fn t -> Nx.bitcast(t.([0, 0, 0], {:s, 32}), :f32) end)
      assert_parity_and_residency(fn t -> Nx.bitcast(t.([-1, 1], {:s, 32}), :u32) end)
      assert_parity_and_residency(fn t -> Nx.bitcast(t.([200, 1], {:u, 8}), :s8) end)
      assert_parity_and_residency(fn t -> Nx.bitcast(t.([-1, 1], {:s, 8}), :u8) end)
      assert_parity_and_residency(fn t -> Nx.bitcast(t.([1.5], {:f, 64}), :s64) end)
    end

    test "the bytes really are untouched — a round trip is the identity" do
      assert_parity_and_residency(fn t ->
        Nx.bitcast(Nx.bitcast(t.([1.5, -2.5], {:f, 32}), :s32), :f32)
      end)

      assert Nx.to_flat_list(Nx.bitcast(gpu([1_065_353_216], {:s, 32}), :f32)) == [1.0]
    end

    test "it does NOT upload a host operand" do
      # A round trip buys literally nothing for a relabel, so a non-resident
      # operand stays on the host rather than being promoted to be renamed.
      host_t = Nx.tensor([1, 2], type: {:s, 32}, backend: Nx.BinaryBackend)
      r = Fallback.strict(:allow, fn -> Nx.bitcast(host_t, :f32) end)
      assert r.data.__struct__ == Nx.BinaryBackend
    end
  end

  describe "dot — any unbatched contraction is a matmul" do
    defp r234t(t, type), do: Nx.reshape(t.(Enum.to_list(1..24), type), {2, 3, 4})
    defp m23t(t, type), do: Nx.reshape(t.(Enum.to_list(1..6), type), {2, 3})

    test "contracting any single axis of a rank-3 pair" do
      for ax <- [0, 1, 2] do
        assert_parity_and_residency(fn t ->
          Nx.dot(r234t(t, {:s, 32}), [ax], [], r234t(t, {:s, 32}), [ax], [])
        end)
      end
    end

    test "contracting TWO axes, including a reversed pairing" do
      # axes_a[i] contracts with axes_b[i] positionally, so [2,0] against [2,0]
      # is a different contraction from [0,2] against [0,2]. Flattening the
      # group in the given order is what keeps those aligned.
      for pair <- [{[0, 1], [0, 1]}, {[2, 0], [2, 0]}, {[1, 2], [1, 2]}] do
        {aa, bb} = pair

        assert_parity_and_residency(fn t ->
          Nx.dot(r234t(t, {:s, 32}), aa, [], r234t(t, {:s, 32}), bb, [])
        end)
      end
    end

    test "an EMPTY contraction is an outer product, and needs no special case" do
      # The contracted group is empty, so K is the empty product 1 and the
      # operands reshape to {M, 1} and {1, N}.
      assert_parity_and_residency(fn t ->
        Nx.dot(t.([1, 2, 3], {:s, 32}), [], [], t.([4, 5], {:s, 32}), [], [])
      end)
    end

    test "rank-0 operands, in either position" do
      # `0..-1//1` is the empty range. Clamping the rank up to 1 instead gave
      # `[0]` and then elem({}, 0) — a crash, not a fallback.
      assert_parity_and_residency(fn t ->
        Nx.dot(t.(3, {:s, 32}), [], [], t.([1, 2], {:s, 32}), [], [])
      end)

      assert_parity_and_residency(fn t ->
        Nx.dot(t.([1, 2], {:s, 32}), [], [], t.(3, {:s, 32}), [], [])
      end)

      assert_parity_and_residency(fn t ->
        Nx.dot(t.(3, {:s, 32}), [], [], t.(4, {:s, 32}), [], [])
      end)
    end

    test "floats take the same path" do
      for type <- [{:f, 32}, {:f, 64}] do
        assert_parity_and_residency(fn t ->
          Nx.dot(r234t(t, type), [2], [], r234t(t, type), [2], [])
        end)
      end
    end

    test "the rank-2 case still works and still wraps" do
      assert_parity_and_residency(fn t ->
        Nx.dot(m23t(t, {:s, 32}), [1], [], m23t(t, {:s, 32}), [1], [])
      end)

      assert Nx.to_number(
               Nx.dot(gpu([2_000_000_000, 2_000_000_000], {:s, 32}), gpu([2, 2], {:s, 32}))
             ) == -589_934_592
    end

    test "BATCHED contractions run on the batched kernel" do
      # Nx requires batch axes to be successive dimensions starting from 0, so
      # the batch is always a leading prefix on both operands and needs no
      # rotation — only the flatten to {B, M, K} and {B, K, N}.
      assert_parity_and_residency(fn t ->
        u = Nx.reshape(t.([1, 1, 2, 2], {:s, 32}), {2, 1, 2})
        v = Nx.reshape(t.([3, 3, 4, 4], {:s, 32}), {2, 2, 1})
        Nx.dot(u, [2], [0], v, [1], [0])
      end)

      assert_parity_and_residency(fn t ->
        u = Nx.reshape(t.(Enum.to_list(1..12), {:s, 32}), {2, 3, 2})
        v = Nx.reshape(t.(Enum.to_list(1..8), {:s, 32}), {2, 2, 2})
        Nx.dot(u, [2], [0], v, [1], [0])
      end)

      # A batch larger than 2, so the third dispatch dimension is exercised
      # beyond the degenerate case.
      assert_parity_and_residency(fn t ->
        u = Nx.reshape(t.(Enum.to_list(1..16), {:s, 32}), {4, 2, 2})
        Nx.dot(u, [2], [0], u, [1], [0])
      end)
    end

    test "batched works at every dtype with a matmul shader" do
      for type <- [{:f, 32}, {:f, 64}] do
        assert_parity_and_residency(fn t ->
          u = Nx.reshape(t.(Enum.to_list(1..12), type), {2, 3, 2})
          v = Nx.reshape(t.(Enum.to_list(1..8), type), {2, 2, 2})
          Nx.dot(u, [2], [0], v, [1], [0])
        end)
      end
    end

    test "a VECTORIZED dot is a batched dot, and closes for free" do
      # Nx turns a vectorized axis into a leading batch axis, so half the
      # doctests this closed were never about batching as the user wrote it.
      assert_parity_and_residency(fn t ->
        u = Nx.vectorize(Nx.reshape(t.([1, 1, 2, 2], {:s, 32}), {2, 1, 2}), :x)
        v = Nx.vectorize(Nx.reshape(t.([3, 3, 4, 4], {:s, 32}), {2, 2, 1}), :x)
        Nx.dot(u, [1], [], v, [0], [])
      end)
    end
  end

  describe "select — any numeric predicate, not just a u8 mask" do
    test "an s32 predicate is normalised rather than refused" do
      # Nx.select treats nonzero as true and its own doctests pass 1, 0 and
      # Nx.tensor([0, 1, 0]) — all s32. The shader wants a packed u8 mask, and
      # `Nx.not_equal(pred, 0)` produces one on the device.
      assert_parity_and_residency(fn t ->
        Nx.select(t.([0, 1, 0], {:s, 32}), t.([1, 2, 3], {:s, 32}), t.([4, 5, 6], {:s, 32}))
      end)

      assert_parity_and_residency(fn t ->
        Nx.select(t.(1, {:s, 32}), t.([1, 2, 3], {:s, 32}), t.([4, 5, 6], {:s, 32}))
      end)

      assert_parity_and_residency(fn t ->
        Nx.select(t.(0, {:s, 32}), t.([1, 2, 3], {:s, 32}), t.([4, 5, 6], {:s, 32}))
      end)
    end

    test "NEGATIVE and fractional predicates are true, not just 1" do
      # `!= 0`, not `== 1`. A predicate normalisation that clamped or compared
      # against 1 would pass every test above and fail here.
      got =
        assert_parity_and_residency(fn t ->
          Nx.select(t.([-1, 0, 2], {:s, 32}), t.([1, 2, 3], {:s, 32}), t.([4, 5, 6], {:s, 32}))
        end)

      assert Nx.to_flat_list(got) == [1, 5, 3]

      assert_parity_and_residency(fn t ->
        Nx.select(t.([0.0, 1.5, 0.0], {:f, 32}), t.([1, 2, 3], {:s, 32}), t.([4, 5, 6], {:s, 32}))
      end)
    end

    test "a u8 mask still takes the direct path" do
      assert_parity_and_residency(fn t ->
        Nx.select(
          Nx.greater(t.([2, 4, 6], {:s, 32}), t.([1, 5, 5], {:s, 32})),
          t.([2, 4, 6], {:s, 32}),
          t.([1, 3, 5], {:s, 32})
        )
      end)
    end

    test "broadcasting branches, and rank 3" do
      assert_parity_and_residency(fn t ->
        Nx.select(
          t.(0, {:s, 32}),
          Nx.reshape(t.([1, 2], {:s, 32}), {1, 2}),
          Nx.reshape(t.([3, 4], {:s, 32}), {2, 1})
        )
      end)

      assert_parity_and_residency(fn t ->
        Nx.select(
          Nx.reshape(t.([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], {:s, 32}), {2, 2, 3}),
          Nx.reshape(t.(Enum.to_list(1..12), {:s, 32}), {2, 2, 3}),
          Nx.reshape(t.(Enum.to_list(13..24), {:s, 32}), {2, 2, 3})
        )
      end)
    end
  end

  describe "as_type float -> integer — three rules, not one" do
    test "NaN is 0 and the infinities SATURATE to the destination's limits" do
      for {type, expected} <- [
            {{:u, 8}, [255, 0, 0]},
            {{:s, 32}, [2_147_483_647, 0, -2_147_483_648]},
            {{:u, 32}, [4_294_967_295, 0, 0]}
          ] do
        got =
          assert_parity_and_residency(fn t ->
            Nx.as_type(t.([:infinity, :nan, :neg_infinity], {:f, 32}), type)
          end)

        assert Nx.to_flat_list(got) == expected
      end
    end

    test "a FINITE out-of-range value WRAPS — it does not saturate" do
      # The trap: the same conversion saturates for infinity and wraps for 300.0.
      # An implementation that clamped everything would pass the test above and
      # fail here, and vice versa.
      got =
        assert_parity_and_residency(fn t ->
          Nx.as_type(t.([0.0, 1.9, -1.9, 255.0, 256.0, 300.0, -1.0], {:f, 32}), {:u, 8})
        end)

      assert Nx.to_flat_list(got) == [0, 1, 255, 255, 0, 44, 255]

      got32 =
        assert_parity_and_residency(fn t ->
          Nx.as_type(t.([1.0e10, -1.0e10], {:f, 32}), {:s, 32})
        end)

      assert Nx.to_flat_list(got32) == [1_410_065_408, -1_410_065_408]
    end

    test "truncation is toward ZERO, not floor" do
      got =
        assert_parity_and_residency(fn t ->
          Nx.as_type(t.([0.0, 1.9, -1.9, 2.5, -2.5], {:f, 32}), {:s, 32})
        end)

      assert Nx.to_flat_list(got) == [0, 1, -1, 2, -2]
    end

    test "the extremes, where the exactness argument does the work" do
      # `int(1.0e10)` is UNDEFINED in GLSL, so the modulo happens in floating
      # point first and has to be exact. Above 2^55 an f32 is already a multiple
      # of 2^32 and the answer is 0; below it, the double arithmetic is exact.
      # 1e15 is the one that would break a naive implementation — it is large
      # enough to need the wrap and small enough that the answer is not 0.
      got =
        assert_parity_and_residency(fn t ->
          Nx.as_type(t.([1.0e15, 1.0e20, 1.0e30, 3.0e38, -1.0e20], {:f, 32}), {:s, 32})
        end)

      assert Nx.to_flat_list(got) == [-1_543_503_872, 0, 0, 0, 0]
    end

    test "a u8 output that is not a multiple of four" do
      # Packed four results per word, so a tail that does not fill a word is the
      # case to get wrong.
      for len <- [1, 2, 3, 5, 7] do
        assert_parity_and_residency(fn t ->
          Nx.as_type(t.(Enum.map(1..len, &(&1 * 1.0)), {:f, 32}), {:u, 8})
        end)
      end
    end

    test "casts TO a float still work, and casts we do not have still fall back" do
      assert_parity_and_residency(fn t -> Nx.as_type(t.([1.5, 2.5], {:f, 32}), {:f, 64}) end)
      # s32 -> u8 is an integer-to-integer cast; no shader, still the host.
      assert_parity(fn t -> Nx.as_type(t.([300, -5], {:s, 32}), {:u, 8}) end)
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
