defmodule Nx.Vulkan.NarrowIntTest do
  @moduledoc """
  s8 / u8 / s16 / u16 elementwise arithmetic, without a narrow-integer ALU.

  `Nx.BinaryBackend` computes every narrow integer op in full precision and
  truncates the result to the destination width, so

      widen -> the existing s32 kernel -> truncate

  reproduces it exactly. `cast_narrow_to_s32.comp` and `cast_s32_to_narrow.comp`
  are that pair; nothing else was written.

  The register filed this family under "needs 8/16-bit storage", which was wrong
  twice over. Storage already worked — a `{:s, 8}` tensor has always been
  device-resident, packed the same way a u8 mask is. And the arithmetic never
  needed the storage extension: it needed two casts and a routing decision.

  **The values are the easy half.** A host fallback returns a bit-identical
  answer, so every value assertion below would pass with the whole feature
  reverted. `assert_resident/1` is the half that can see anything.

  The cases that actually discriminate:

    * **Overflow.** `127 + 2` at `{:s, 8}` is `-127`, not `127`. Truncation, not
      saturation — the opposite of the rule `as_type` applies to a non-finite
      float, in the same backend.
    * **Sign extension.** `200` at `{:u, 8}` must widen to `200`, not `-56`.
    * **Width-dependent unaries.** `count_leading_zeros(1)` is 7 at `{:s, 8}`
      and 31 at `{:s, 32}`; `population_count(-1)` is 8 at `{:s, 8}`. Both are
      defined on the declared width's BITS, so both must zero-extend where
      arithmetic sign-extends.
    * **The packed tail.** A length that is not a multiple of 4 (or 2) leaves a
      partial final word, and the shader must define all of it.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp assert_resident(t) do
    assert match?(%VulkanoBackend{}, t.data),
           "expected the result to stay on the GPU, got #{inspect(t.data.__struct__)}"

    t
  end

  defp check(build) do
    got = build.(VulkanoBackend) |> assert_resident()
    ref = build.(Nx.BinaryBackend)
    assert Nx.type(got) == Nx.type(ref)
    assert Nx.shape(got) == Nx.shape(ref)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  # Deliberately awkward: the extremes of each width, values that overflow when
  # combined, negatives, and a LENGTH THAT IS NOT A MULTIPLE OF 4 so the packed
  # tail word is exercised.
  defp operands({:s, 8}),
    do: {[127, -128, 100, -100, 3, -3, 0, 1, 42], [2, 3, 100, 7, -2, 5, 4, -1, 7]}

  defp operands({:u, 8}),
    do: {[0, 1, 200, 255, 128, 7, 99, 254, 3], [2, 3, 100, 7, 2, 5, 4, 1, 7]}

  defp operands({:s, 16}),
    do: {[32767, -32768, 300, -300, 3, -3, 0, 1, 999], [2, 3, 100, 7, -2, 5, 4, -1, 7]}

  defp operands({:u, 16}),
    do: {[0, 1, 60000, 65535, 32768, 7, 99, 254, 3], [2, 3, 100, 7, 2, 5, 4, 1, 7]}

  @types [{:s, 8}, {:u, 8}, {:s, 16}, {:u, 16}]

  # `divide` and `pow` are absent on purpose: Nx types `divide` on integers as a
  # float, and the s32 kernel has no `pow` — so neither can reach a narrow
  # integer output, and narrow_binary/5 refuses both codes explicitly.
  @binary [
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
  ]

  describe "elementwise binary, same shape" do
    for type <- @types, op <- @binary do
      type = Macro.escape(type)

      test "#{inspect(type)} #{op}" do
        {as, bs} = operands(unquote(type))
        op = unquote(op)

        check(fn b ->
          apply(Nx, op, [
            Nx.tensor(as, type: unquote(type), backend: b),
            Nx.tensor(bs, type: unquote(type), backend: b)
          ])
        end)
      end
    end
  end

  describe "the shifts — the count must be in range for every element" do
    for type <- @types do
      type = Macro.escape(type)

      test "#{inspect(type)} left_shift / right_shift" do
        {as, _} = operands(unquote(type))
        shifts = Enum.map(1..length(as), &rem(&1, 4))

        for op <- [:left_shift, :right_shift] do
          check(fn b ->
            apply(Nx, op, [
              Nx.tensor(as, type: unquote(type), backend: b),
              Nx.tensor(shifts, type: unquote(type), backend: b)
            ])
          end)
        end
      end
    end
  end

  describe "broadcasting — a scalar operand rides the s32 bcast kernel" do
    for type <- @types do
      type = Macro.escape(type)

      test "#{inspect(type)} tensor + scalar" do
        {as, _} = operands(unquote(type))

        check(fn b ->
          Nx.add(
            Nx.tensor(as, type: unquote(type), backend: b),
            Nx.tensor(3, type: unquote(type), backend: b)
          )
        end)
      end
    end
  end

  describe "elementwise unary" do
    for type <- @types, op <- [:negate, :abs, :sign, :bitwise_not] do
      type = Macro.escape(type)

      test "#{inspect(type)} #{op}" do
        {as, _} = operands(unquote(type))
        op = unquote(op)
        check(fn b -> apply(Nx, op, [Nx.tensor(as, type: unquote(type), backend: b)]) end)
      end
    end
  end

  describe "the width-dependent unaries" do
    for type <- @types do
      type = Macro.escape(type)

      test "#{inspect(type)} count_leading_zeros / population_count" do
        {as, _} = operands(unquote(type))

        for op <- [:count_leading_zeros, :population_count] do
          check(fn b -> apply(Nx, op, [Nx.tensor(as, type: unquote(type), backend: b)]) end)
        end
      end
    end

    test "clz is counted at the DECLARED width, not at 32" do
      # The assertion that would have caught a missing `- (32 - bits)`: the same
      # value, four widths, four different answers.
      one = fn t, b -> Nx.count_leading_zeros(Nx.tensor([1], type: t, backend: b)) end

      for {type, expected} <- [{{:s, 8}, 7}, {{:u, 8}, 7}, {{:s, 16}, 15}, {{:u, 16}, 15}] do
        assert Nx.to_flat_list(one.(type, VulkanoBackend)) == [expected]
      end

      # And zero, where the 32-bit answer is 32 and the narrow one is the width.
      zero = fn t, b -> Nx.count_leading_zeros(Nx.tensor([0], type: t, backend: b)) end

      for {type, expected} <- [{{:s, 8}, 8}, {{:u, 8}, 8}, {{:s, 16}, 16}, {{:u, 16}, 16}] do
        assert Nx.to_flat_list(zero.(type, VulkanoBackend)) == [expected]
      end
    end

    test "population_count reads the BITS, so a negative is not sign-extended" do
      # -1 at {:s, 8} has eight set bits, not thirty-two. Sign-extending on the
      # way up would answer 32 and look entirely plausible.
      assert Nx.to_flat_list(
               Nx.population_count(
                 Nx.tensor([-1, -128, 127], type: {:s, 8}, backend: VulkanoBackend)
               )
             ) == [8, 1, 7]
    end
  end

  describe "overflow truncates — it does NOT saturate" do
    test "{:s, 8} addition wraps at 127" do
      a = Nx.tensor([127, 127, -128], type: {:s, 8}, backend: VulkanoBackend)
      b = Nx.tensor([1, 2, -1], type: {:s, 8}, backend: VulkanoBackend)
      assert Nx.to_flat_list(Nx.add(a, b)) == [-128, -127, 127]
    end

    test "{:u, 8} multiplication wraps at 255" do
      a = Nx.tensor([200, 16, 255], type: {:u, 8}, backend: VulkanoBackend)
      b = Nx.tensor([2, 16, 255], type: {:u, 8}, backend: VulkanoBackend)
      assert Nx.to_flat_list(Nx.multiply(a, b)) == [144, 0, 1]
    end

    test "an unsigned narrow value widens by ZERO extension" do
      # 200 at {:u, 8} must widen to 200. Sign-extending would make it -56, and
      # then `min(200, 100)` would answer 200 instead of 100 — a wrong answer
      # that is still in range and still looks like a plausible minimum.
      a = Nx.tensor([200, 255, 128], type: {:u, 8}, backend: VulkanoBackend)
      b = Nx.tensor([100, 100, 100], type: {:u, 8}, backend: VulkanoBackend)
      assert Nx.to_flat_list(Nx.min(a, b)) == [100, 100, 100]
      assert Nx.to_flat_list(Nx.max(a, b)) == [200, 255, 128]
      assert Nx.to_flat_list(Nx.quotient(a, b)) == [2, 2, 1]
    end
  end

  describe "the packed tail" do
    for {type, per_word} <- [{{:s, 8}, 4}, {{:u, 8}, 4}, {{:s, 16}, 2}, {{:u, 16}, 2}] do
      type = Macro.escape(type)

      test "#{inspect(type)} at every length around a word boundary" do
        for n <- 1..(unquote(per_word) * 2 + 1) do
          check(fn b ->
            t = Nx.tensor(Enum.map(1..n, &rem(&1 * 37, 100)), type: unquote(type), backend: b)
            Nx.add(t, t)
          end)
        end
      end
    end
  end

  describe "as_type across the narrow widths" do
    test "narrow -> 32-bit widens by the SOURCE's signedness" do
      check(fn b -> Nx.as_type(Nx.tensor([200, 255, 1], type: {:u, 8}, backend: b), :s32) end)
      check(fn b -> Nx.as_type(Nx.tensor([-1, 127, -128], type: {:s, 8}, backend: b), :s32) end)
      # s8 -1 sign-extends to s32 -1, whose BITS are u32 4294967295 — which is
      # what Nx answers. Zero-extending would give 255.
      check(fn b -> Nx.as_type(Nx.tensor([-1], type: {:s, 8}, backend: b), :u32) end)
      check(fn b -> Nx.as_type(Nx.tensor([-1, 200], type: {:s, 16}, backend: b), :f32) end)
      check(fn b -> Nx.as_type(Nx.tensor([-1, 200], type: {:s, 16}, backend: b), :f64) end)
    end

    test "32-bit -> narrow TRUNCATES" do
      check(fn b -> Nx.as_type(Nx.tensor([300, -5, 1], backend: b), :u8) end)
      check(fn b -> Nx.as_type(Nx.tensor([300, -5, 1], backend: b), :s8) end)
      check(fn b -> Nx.as_type(Nx.tensor([70000, -5, 1], backend: b), :s16) end)
    end

    test "narrow -> narrow round-trips through s32" do
      check(fn b -> Nx.as_type(Nx.tensor([-1, 127], type: {:s, 8}, backend: b), :u8) end)
      check(fn b -> Nx.as_type(Nx.tensor([200, 255], type: {:u, 8}, backend: b), :s8) end)
      check(fn b -> Nx.as_type(Nx.tensor([-1, 127], type: {:s, 8}, backend: b), :s16) end)
      check(fn b -> Nx.as_type(Nx.tensor([70000, -5], type: {:s, 32}, backend: b), :u16) end)
    end

    # TAGGED for the same reason as the non-contiguous case in
    # reduce_axes_test.exs: the REFUSAL is what this test asserts, so under
    # NXV_HOST_FALLBACK=raise it must opt out or it turns strict_test.sh red.
    @tag :host_fallback_expected
    test "a FLOAT source is refused, and the reason is not squeamishness" do
      # Nx saturates a non-finite float to the DESTINATION's range: :infinity is
      # 127 at {:s, 8}. Composing float -> s32 -> truncate saturates to
      # 2147483647 and then truncates to -1. u8 is the one width where the
      # composition happens to agree, which is an accident and not a rule — so
      # narrow_as_type/2 refuses float sources outright and leaves them to
      # cast_to_int_spv/2's direct shaders.
      nf = fn b -> Nx.tensor([:infinity, :nan, :neg_infinity], backend: b) end

      for {type, expected} <- [
            {:u8, [255, 0, 0]},
            {:s8, [127, 0, -128]},
            {:s16, [32767, 0, -32768]},
            {:u16, [65535, 0, 0]}
          ] do
        assert Nx.to_flat_list(Nx.as_type(nf.(VulkanoBackend), type)) == expected
      end
    end
  end

  describe "reductions over a narrow int" do
    for type <- @types do
      type = Macro.escape(type)

      test "#{inspect(type)} sum / product widen, reduce_max / reduce_min do NOT" do
        {as, _} = operands(unquote(type))
        t = fn b -> Nx.tensor(as, type: unquote(type), backend: b) end

        # Nx widens sum and product to a 32-bit accumulator but keeps the narrow
        # type for max/min. Both destinations have to work: one retypes the
        # widened buffer, the other truncates it.
        check(fn b -> Nx.sum(t.(b)) end)
        check(fn b -> Nx.product(t.(b)) end)
        check(fn b -> Nx.reduce_max(t.(b)) end)
        check(fn b -> Nx.reduce_min(t.(b)) end)

        assert Nx.type(Nx.reduce_max(t.(VulkanoBackend))) == unquote(type)
      end
    end

    test "over an axis of a rank-3 narrow tensor" do
      t = fn b ->
        Nx.reshape(Nx.tensor(Enum.map(1..24, &(&1 - 12)), type: {:s, 8}, backend: b), {2, 3, 4})
      end

      for axis <- 0..2 do
        check(fn b -> Nx.sum(t.(b), axes: [axis]) end)
        check(fn b -> Nx.reduce_max(t.(b), axes: [axis]) end)
      end
    end
  end

  describe "MIXED narrow operands — each side widens by its OWN signedness" do
    test "s8 % u8 -> s16, which is Nx's own doctest" do
      # Nx promotes a mixed narrow pair to a THIRD narrow type. Requiring all
      # three types to be equal refused exactly this, and it is not a corner
      # case Nx invented for the docs — any two narrow tensors of different
      # width or signedness land here.
      check(fn b ->
        Nx.remainder(
          Nx.tensor(-11, type: {:s, 8}, backend: b),
          Nx.tensor(10, type: {:u, 8}, backend: b)
        )
      end)

      assert Nx.type(
               Nx.remainder(
                 Nx.tensor(-11, type: {:s, 8}, backend: VulkanoBackend),
                 Nx.tensor(10, type: {:u, 8}, backend: VulkanoBackend)
               )
             ) == {:s, 16}
    end

    test "every mixed pair of the four widths" do
      for ta <- @types, tb <- @types, ta != tb do
        check(fn b ->
          Nx.add(
            Nx.tensor([1, 2, 100], type: ta, backend: b),
            Nx.tensor([3, 4, 100], type: tb, backend: b)
          )
        end)
      end
    end
  end

  describe "coerce_to — a narrow operand against a FLOAT output" do
    test "s8 / s8 is f32, which is Nx's divide doctest" do
      # Nx types integer division as a float, so the OUTPUT is f32 while both
      # operands are s8. cast_spv/2 had no entry for that pair and the whole op
      # went to the host — to reach a cast that already existed one widening
      # away.
      check(fn b ->
        Nx.divide(
          Nx.tensor([[1], [2]], type: {:s, 8}, backend: b),
          Nx.tensor([[10, 20]], type: {:s, 8}, backend: b)
        )
      end)
    end

    test "a narrow tensor against a float tensor" do
      for type <- @types do
        check(fn b ->
          Nx.multiply(
            Nx.tensor([1, 2, 3], type: type, backend: b),
            Nx.tensor([1.5, 2.5, 0.5], type: {:f, 32}, backend: b)
          )
        end)

        check(fn b ->
          Nx.subtract(
            Nx.tensor([10, 20, 30], type: type, backend: b),
            Nx.tensor([1.5, 2.5, 0.5], type: {:f, 64}, backend: b)
          )
        end)
      end
    end
  end

  describe "broadcast — a WORD copy cannot address a byte" do
    test "a packed narrow tensor broadcasts by widening first" do
      for type <- @types do
        check(fn b -> Nx.broadcast(Nx.tensor([1, 0, 3], type: type, backend: b), {2, 3}) end)
        check(fn b -> Nx.broadcast(Nx.tensor([[1], [2]], type: type, backend: b), {2, 4}) end)
        check(fn b -> Nx.broadcast(Nx.tensor(7, type: type, backend: b), {2, 3, 4}) end)
      end
    end

    test "tril / triu, which is what needed it" do
      # Nx.tri/3 builds a u8 mask and broadcasts it to the tensor's shape. That
      # broadcast was the whole fallback — the multiply that consumes the mask
      # has been resident since T12, so the op was leaving the device to move
      # bytes it never computed on.
      check(fn b -> Nx.triu(Nx.iota({2, 3, 4}, backend: b)) end)
      check(fn b -> Nx.tril(Nx.iota({2, 3, 4}, backend: b)) end)
      check(fn b -> Nx.triu(Nx.iota({3, 3}, backend: b), k: 1) end)
      check(fn b -> Nx.tril(Nx.iota({3, 3}, backend: b), k: -1) end)
    end
  end

  describe "multi-dimensional" do
    test "{:s, 8} rank 3, and the result is still resident" do
      check(fn b ->
        t =
          Nx.reshape(Nx.tensor(Enum.map(1..24, &(&1 - 12)), type: {:s, 8}, backend: b), {2, 3, 4})

        Nx.multiply(t, t)
      end)
    end
  end
end
