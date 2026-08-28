defmodule Nx.Vulkan.U32Test do
  @moduledoc """
  `{:u, 32}` arithmetic — the last dtype where a whole family left the device.

  Thirty of thirty-four u32 operations host-fell-back before this. Only one
  doctest saw it (`Nx.quotient/2`), which is why the gap sat in the register
  looking like a one-line job for as long as it did.

  **u32 cannot use the widen/truncate route that s8/u8/s16/u16 use.** Those work
  because every narrow value has an s32 image; `3_000_000_000` does not. So u32
  needs its own `uint` shaders — and the temptation to reuse the s32 ones for
  "the codes that are the same" is the trap this file exists to pin.

  In two's complement these are BIT-IDENTICAL between s32 and u32, and reusing
  the signed shader for them would be correct:

      add, subtract, multiply, pow, bitwise_and/or/xor, left_shift,
      negate, bitwise_not, population_count, count_leading_zeros,
      sum, product, dot, all/any, select

  And these are NOT, where a signed kernel returns a plausible wrong number:

      max, min, quotient, remainder, right_shift (logical, not arithmetic),
      sign (0 or 1, never -1), and all six comparisons — plus argmax/argmin,
      which compare.

  Every case below uses at least one operand above 2^31 (`3_000_000_000`,
  `4_294_967_295`, `2_147_483_648`), because that is the ONLY region where the
  two readings differ. A test written on small values passes either way and
  proves nothing.

  Values are checked against `Nx.BinaryBackend` and residency is asserted
  separately — a fallback returns a bit-identical answer, so the value
  assertions alone cannot see one.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp check(build) do
    got = build.(VulkanoBackend)
    assert match?(%VulkanoBackend{}, got.data), "expected the result to stay on the GPU"
    ref = build.(Nx.BinaryBackend)
    assert Nx.type(got) == Nx.type(ref)
    assert Nx.shape(got) == Nx.shape(ref)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  # Deliberately straddling 2^31, and a length that is not a power of two.
  @big [4_294_967_295, 3_000_000_000, 2_147_483_648, 7, 0, 1, 2_147_483_647]
  @div [7, 2, 3, 2, 5, 1, 6]

  defp t(list, b), do: Nx.tensor(list, type: {:u, 32}, backend: b)

  describe "elementwise binary, same shape" do
    for op <- [
          :add,
          :subtract,
          :multiply,
          :max,
          :min,
          :quotient,
          :remainder,
          :bitwise_and,
          :bitwise_or,
          :bitwise_xor
        ] do
      test "#{op}" do
        op = unquote(op)
        check(fn b -> apply(Nx, op, [t(@big, b), t(@div, b)]) end)
      end
    end

    test "the shifts — right_shift must be LOGICAL" do
      # 4294967295 >>> 1 is 2147483647. An ARITHMETIC shift would give
      # 4294967295 back (all ones stays all ones), which is in range, is a
      # perfectly plausible u32, and is wrong.
      shifts = [1, 1, 1, 2, 3, 31, 0]
      check(fn b -> Nx.right_shift(t(@big, b), t(shifts, b)) end)
      check(fn b -> Nx.left_shift(t(@big, b), t(shifts, b)) end)

      assert Nx.to_flat_list(
               Nx.right_shift(
                 t([4_294_967_295, 2_147_483_648], VulkanoBackend),
                 t([1, 1], VulkanoBackend)
               )
             ) == [2_147_483_647, 1_073_741_824]
    end

    test "pow wraps mod 2^32, exactly as the signed kernel does" do
      # pow(3, 20) is 3486784401 unsigned, which is the same bit pattern as
      # -808182895 signed. The `uint` accumulator wraps identically.
      check(fn b -> Nx.pow(t([2, 3, 2], b), t([32, 20, 5], b)) end)
    end

    test "divide goes to f32 and always did" do
      check(fn b -> Nx.divide(t(@big, b), t(@div, b)) end)
    end
  end

  describe "the unsigned comparisons — where a signed kernel is plausibly wrong" do
    for op <- [:equal, :not_equal, :less, :less_equal, :greater, :greater_equal] do
      test "#{op}" do
        op = unquote(op)
        check(fn b -> apply(Nx, op, [t(@big, b), t(@div, b)]) end)
      end
    end

    test "3_000_000_000 is GREATER than 2, not less" do
      # Read as s32, 3_000_000_000 is -1294967296 and every one of these flips.
      big = t([3_000_000_000], VulkanoBackend)
      two = t([2], VulkanoBackend)
      assert Nx.to_flat_list(Nx.greater(big, two)) == [1]
      assert Nx.to_flat_list(Nx.less(big, two)) == [0]
      assert Nx.to_flat_list(Nx.max(big, two)) == [3_000_000_000]
      assert Nx.to_flat_list(Nx.min(big, two)) == [2]
    end
  end

  describe "elementwise unary" do
    for op <- [:abs, :negate, :sign, :bitwise_not, :population_count, :count_leading_zeros] do
      test "#{op}" do
        op = unquote(op)
        check(fn b -> apply(Nx, op, [t(@big, b)]) end)
      end
    end

    test "sign is 0 or 1 and NEVER -1" do
      # A signed `sign()` would answer -1 for anything above 2^31, and -1 is not
      # representable in u32 — it would come back as 4294967295.
      assert Nx.to_flat_list(Nx.sign(t([0, 1, 3_000_000_000, 4_294_967_295], VulkanoBackend))) ==
               [0, 1, 1, 1]
    end

    test "clz and population_count read the BITS, so they are unchanged" do
      assert Nx.to_flat_list(
               Nx.count_leading_zeros(t([0, 1, 4_294_967_295, 2_147_483_648], VulkanoBackend))
             ) == [32, 31, 0, 0]

      assert Nx.to_flat_list(
               Nx.population_count(t([0, 1, 4_294_967_295, 2_147_483_648], VulkanoBackend))
             ) == [0, 1, 32, 1]
    end
  end

  describe "broadcasting" do
    test "a scalar operand rides the bcast kernel" do
      for op <- [:add, :multiply, :max, :min, :quotient] do
        check(fn b -> apply(Nx, op, [t(@big, b), Nx.tensor(3, type: {:u, 32}, backend: b)]) end)
      end
    end

    test "rank-2 against rank-1" do
      check(fn b ->
        Nx.add(
          Nx.reshape(t([1, 2, 3, 4_294_967_295, 5, 6], b), {2, 3}),
          t([10, 20, 3_000_000_000], b)
        )
      end)
    end
  end

  describe "reductions" do
    test "sum and product WRAP, and stay {:u, 32}" do
      check(fn b -> Nx.sum(t([3_000_000_000, 3_000_000_000], b)) end)
      check(fn b -> Nx.product(t([3_000_000_000, 3], b)) end)
      assert Nx.type(Nx.sum(t(@big, VulkanoBackend))) == {:u, 32}
    end

    test "reduce_max / reduce_min compare UNSIGNED" do
      check(fn b -> Nx.reduce_max(t(@big, b)) end)
      check(fn b -> Nx.reduce_min(t(@big, b)) end)

      assert Nx.to_flat_list(Nx.reduce_max(t([3_000_000_000, 1], VulkanoBackend))) ==
               [3_000_000_000]
    end

    test "over an axis of a rank-3 tensor, including the middle one" do
      build = fn b -> Nx.reshape(t(Enum.map(1..24, &(&1 * 200_000_000)), b), {2, 3, 4}) end

      for axis <- 0..2 do
        check(fn b -> Nx.sum(build.(b), axes: [axis]) end)
        check(fn b -> Nx.reduce_max(build.(b), axes: [axis]) end)
      end
    end

    test "argmax / argmin compare UNSIGNED too" do
      check(fn b -> Nx.argmax(t(@big, b)) end)
      check(fn b -> Nx.argmin(t(@big, b)) end)
      check(fn b -> Nx.argmax(t(@big, b), tie_break: :high) end)
      assert Nx.to_flat_list(Nx.argmax(t([1, 3_000_000_000, 2], VulkanoBackend))) == [1]
    end

    test "all / any test against zero, which is signedness-free" do
      check(fn b -> Nx.all(t([1, 3_000_000_000, 0], b)) end)
      check(fn b -> Nx.any(t([0, 0, 3_000_000_000], b)) end)
    end
  end

  describe "the ops that can reuse the signed kernel, and why" do
    test "select is a word copy — no arithmetic, so signedness cannot matter" do
      check(fn b ->
        Nx.select(
          Nx.tensor([1, 0, 1, 0, 1, 0, 1], type: {:u, 8}, backend: b),
          t(@big, b),
          t(@div, b)
        )
      end)
    end

    test "dot accumulates with add and multiply, both of which wrap identically" do
      check(fn b -> Nx.dot(t([3_000_000_000, 2], b), t([3, 1], b)) end)

      check(fn b ->
        Nx.dot(
          Nx.reshape(t([1, 2, 3, 4_294_967_295], b), {2, 2}),
          Nx.reshape(t([3_000_000_000, 1, 2, 3], b), {2, 2})
        )
      end)
    end
  end

  describe "as_type" do
    test "to and from the float types" do
      check(fn b -> Nx.as_type(t([0, 1, 3_000_000_000], b), :f32) end)
      check(fn b -> Nx.as_type(t([0, 1, 3_000_000_000], b), :f64) end)
    end

    test "to and from s32 is a reinterpretation of the same bits" do
      check(fn b -> Nx.as_type(t([3_000_000_000], b), :s32) end)
      check(fn b -> Nx.as_type(Nx.tensor([-1], type: {:s, 32}, backend: b), :u32) end)
    end
  end
end
