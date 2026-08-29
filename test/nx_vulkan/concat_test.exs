defmodule Nx.Vulkan.ConcatTest do
  @moduledoc """
  `glsl/concat_nd.comp` — concatenation along any axis, and the five ops that
  were blocked on it.

  Axis 0 was always on the GPU: a row-major axis-0 concat is a byte append, and
  `concat_buffers/1` does it with no index arithmetic. Axis > 0 needed a kernel,
  and W4's block census is what identified it as worth writing — an axis > 0
  concatenate was the ONLY host fallback left in `Nx.take_along_axis/3` and all
  four `Nx.cumulative_*/2`, and `associative_scan` calls it log2(n) times per
  reduction.

  Two things every case here asserts, because either alone is insufficient:

    * **bit-equality with `Nx.BinaryBackend`.** Concat is a pure copy, so
      "within eps" is not the bar. A misplaced slab is a wrong answer, not a
      rounding difference.
    * **zero recorded fallbacks.** The host fallback returns a bit-identical
      result, so a value assertion passes whether or not the kernel ran.

  Values are all distinct across every input, so a slab written at the wrong
  offset cannot coincide with the right answer.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.Fallback
  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp host(shape, type, seed) do
    n = shape |> Tuple.to_list() |> Enum.reduce(1, &(&1 * &2))

    data =
      case type do
        {:f, _} -> for i <- 1..n, do: seed * 1000 + i * 1.0
        _ -> for i <- 1..n, do: seed * 1000 + i
      end

    Nx.tensor(data, type: type, backend: Nx.BinaryBackend) |> Nx.reshape(shape)
  end

  # Concatenate the same inputs on both backends; assert identical bytes AND
  # that the GPU path actually ran.
  defp assert_parity_and_residency(shapes, type, axis) do
    hosts = shapes |> Enum.with_index() |> Enum.map(fn {s, i} -> host(s, type, i + 1) end)
    gpus = Enum.map(hosts, &Nx.backend_transfer(&1, VulkanoBackend))

    expected = Nx.concatenate(hosts, axis: axis)
    {got, counts} = Fallback.count(fn -> Nx.concatenate(gpus, axis: axis) end)

    assert Nx.shape(got) == Nx.shape(expected)
    assert Nx.type(got) == Nx.type(expected)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(expected)

    assert got.data.__struct__ == VulkanoBackend,
           "result left the device: #{inspect(got.data.__struct__)}"

    assert counts == %{},
           "expected a resident concatenate, got fallbacks: #{inspect(counts)}"
  end

  describe "concat_nd — parity and residency" do
    # rank 2..4, inner and trailing axes, uneven splits, 1..3 inputs.
    @cases [
      {[{2, 3}, {2, 4}], 1},
      {[{2, 3}, {2, 4}, {2, 1}], 1},
      {[{5, 2}, {5, 2}], 1},
      {[{2, 3, 4}, {2, 1, 4}], 1},
      {[{2, 3, 4}, {2, 3, 2}], 2},
      {[{2, 3, 4}, {2, 3, 2}, {2, 3, 5}], 2},
      {[{2, 2, 3, 2}, {2, 2, 1, 2}], 2},
      {[{2, 2, 3, 2}, {2, 2, 3, 3}], 3},
      {[{1, 2, 2, 2}, {1, 3, 2, 2}], 1},
      {[{3, 4}], 1}
    ]

    # 4- and 8-byte dtypes both, because the shader copies u32 WORDS and `ews`
    # is what makes one kernel serve both. An 8-byte type with ews wrong reads
    # half of each element — silent, and only visible on f64/s64.
    for type <- [{:f, 32}, {:f, 64}, {:s, 32}, {:s, 64}] do
      for {shapes, axis} <- @cases do
        test "#{inspect(type)} #{inspect(shapes)} axis #{axis}" do
          assert_parity_and_residency(unquote(Macro.escape(shapes)), unquote(type), unquote(axis))
        end
      end
    end
  end

  # Every test in here provokes a host fallback on purpose, which is exactly what
  # `scripts/strict_test.sh` refuses. Tagged rather than allowlisted: an
  # allowlist entry would excuse `concatenate/3` for the whole suite and hide a
  # real regression, whereas the tag excuses only these three assertions.
  describe "the gates that must still fall back" do
    @describetag :host_fallback_expected

    test "mixed input types fall back so Nx can cast to the merged type first" do
      a = Nx.tensor([[1.0, 2.0]], type: {:f, 32}, backend: VulkanoBackend)
      b = Nx.tensor([[3, 4]], type: {:s, 32}, backend: VulkanoBackend)

      r = Nx.concatenate([a, b], axis: 1)
      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0, 4.0]
    end

    test "a 1-byte dtype falls back, as it does for slice/pad/put_slice" do
      a = Nx.tensor([[1, 2]], type: {:u, 8}, backend: VulkanoBackend)
      b = Nx.tensor([[3]], type: {:u, 8}, backend: VulkanoBackend)

      {r, counts} = Fallback.count(fn -> Nx.concatenate([a, b], axis: 1) end)

      assert Nx.to_flat_list(r) == [1, 2, 3]
      assert counts == %{{:concatenate, 3} => 1}
    end

    # A MIXED set of operands falls back on purpose, and this test exists to stop
    # someone "fixing" it. Uploading the host operands would make the RESULT
    # resident, and `Nx.take_along_axis/3` then hands that resident index tensor
    # to `Nx.gather/3` next to a host operand; nx resolves a multi-arg op to ONE
    # backend, picks `Nx.BinaryBackend.gather/3`, and it dies in `to_binary/1`
    # with no clause. Four `Nx.mode/2` doctests caught exactly that. The looser
    # gate does not remove a mixed-backend pair, it moves one downstream where
    # this backend cannot fix it.
    #
    # RE-RUN 2026-08-23 and it still fails, which is worth recording because the
    # conditions had changed enough to make it worth asking: `gather/4` now
    # rotates off-prefix axes instead of refusing them, and `select/4` now
    # normalises any numeric predicate. Neither helps. Promoting the operands
    # still produces `FunctionClauseError ... Nx.BinaryBackend.to_binary/1` on
    # exactly four `Nx.mode/2` doctests, with a resident `s32[1][5][2]` index
    # tensor as the argument.
    #
    # The reason those six `Nx.mode/2` doctests fall back is NOT this gate. It is
    # `sort/3`, which is allowlisted with no GPU sort and no plan for one
    # (MISSION §3.2) — everything downstream of it is on the host as a
    # consequence, and `concatenate/3` is merely where the census first notices.
    # Closing them needs a GPU sort, not a looser concat.
    test "mixed residency falls back — promoting operands breaks Nx.gather downstream" do
      a = Nx.tensor([[1.0, 2.0]], backend: VulkanoBackend)
      b = Nx.tensor([[3.0, 4.0]], backend: Nx.BinaryBackend)

      {r, counts} = Fallback.count(fn -> Nx.concatenate([a, b], axis: 1) end)

      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0, 4.0]
      assert counts == %{{:concatenate, 3} => 1}
    end

    test "axis 0 still takes the cheaper byte-append path, not this shader" do
      a = Nx.tensor([[1.0, 2.0]], backend: VulkanoBackend)
      b = Nx.tensor([[3.0, 4.0]], backend: VulkanoBackend)

      {r, counts} = Fallback.count(fn -> Nx.concatenate([a, b], axis: 0) end)

      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0, 4.0]
      assert Nx.shape(r) == {2, 2}
      assert counts == %{}
    end
  end

  describe "the ops W4's census said were blocked on this" do
    setup do
      %{
        t: Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: VulkanoBackend),
        h: Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Nx.BinaryBackend)
      }
    end

    for op <- [:cumulative_sum, :cumulative_product, :cumulative_min, :cumulative_max] do
      test "Nx.#{op}/2 on axis 1 is fully resident", %{t: t, h: h} do
        op = unquote(op)

        {got, counts} = Fallback.count(fn -> apply(Nx, op, [t, [axis: 1]]) end)

        assert Nx.to_flat_list(got) == Nx.to_flat_list(apply(Nx, op, [h, [axis: 1]]))
        assert counts == %{}
      end
    end

    test "cumulative_sum honours :reverse and stays resident", %{t: t, h: h} do
      {got, counts} = Fallback.count(fn -> Nx.cumulative_sum(t, axis: 1, reverse: true) end)

      assert Nx.to_flat_list(got) == Nx.to_flat_list(Nx.cumulative_sum(h, axis: 1, reverse: true))
      assert counts == %{}
    end

    # Under a VulkanoBackend DEFAULT, which is what production and
    # nx_doctest_test.exs's setup use. The default matters here and nowhere else
    # in this file: take_along_axis's body builds its index tensor with
    # `Nx.iota/2`, which materialises on the process default, so the concat's
    # operands are all resident only when that default is this backend.
    test "Nx.take_along_axis/3 is fully resident under a Vulkano default", %{t: t, h: h} do
      hi = Nx.tensor([[0, 0, 1], [1, 1, 0]], backend: Nx.BinaryBackend)
      expected = Nx.to_flat_list(Nx.take_along_axis(h, hi, axis: 0))

      Nx.default_backend(VulkanoBackend)
      gi = Nx.tensor([[0, 0, 1], [1, 1, 0]], backend: VulkanoBackend)

      {got, counts} = Fallback.count(fn -> Nx.take_along_axis(t, gi, axis: 0) end)

      assert Nx.to_flat_list(got) == expected
      assert counts == %{}
    end
  end

  describe "sub-word dtypes on axis 0 — the padded-splice bug" do
    # `concat_buffers` splices by each buffer's `n_bytes`, which for a sub-word
    # dtype is the PADDED size: a {:u, 8} tensor of 1 element occupies 4 bytes,
    # an {:s, 16} of 5 occupies 12. Byte-appending those put padding into the
    # INTERIOR of the result and dropped as many real elements off the tail.
    #
    # A WRONG ANSWER, not a crash. It predates the uninitialised-allocator
    # change — the spliced bytes read 0 under deliberate memory poisoning, so
    # the offsets were already wrong while padding was still zeroed. That change
    # only made the wrongness non-deterministic.
    #
    # This file parametrised {:f,32}, {:f,64}, {:s,32}, {:s,64} — all
    # word-copyable — and its one 1-byte case asserts a FALLBACK on axis 1. The
    # axis-0 byte-append path had never been exercised with a sub-word dtype,
    # which is why a documented rule ("1/2-byte types fall back") could be
    # missing from one of the two gates that needed it.
    # TAGGED because the fallback IS the fix. Sub-word axis-0 concat now
    # host-falls-back by design, so under NXV_HOST_FALLBACK=raise these raise.
    #
    # This is the FOURTH time this session that a test whose subject is a
    # fallback went in without the tag, and each time only strict_test.sh could
    # see it — `mix test` passes, because a fallback returns a bit-identical
    # answer. The rule keeps having to be relearned because the green `mix test`
    # is the one you look at. Run all three scripts.
    @tag :host_fallback_expected
    test "the case that was wrong: concatenating two 1-slot u8 reductions" do
      build = fn b ->
        t = Nx.tensor([[1, 0]], backend: b)
        Nx.concatenate([Nx.all(t, axes: [1]), Nx.any(t, axes: [1])])
      end

      assert Nx.to_flat_list(build.(VulkanoBackend)) == [0, 1]
      assert Nx.to_flat_list(build.(VulkanoBackend)) == Nx.to_flat_list(build.(Nx.BinaryBackend))
    end

    for type <- [{:u, 8}, {:s, 8}, {:s, 16}, {:u, 16}] do
      type = Macro.escape(type)

      @tag :host_fallback_expected
      test "#{inspect(type)} at lengths that pad" do
        type = unquote(type)

        # Lengths chosen so the buffer is padded: at 1 byte per element,
        # anything not a multiple of 4; at 2 bytes, anything odd.
        for n <- [1, 2, 3, 5, 7, 9] do
          build = fn b ->
            # COMPUTED, not uploaded. An `Nx.tensor(...)` buffer is exact-sized;
            # padding exists only on KERNEL-ALLOCATED buffers, so operands built
            # by upload cannot exercise the splice at all. The first version of
            # this test used them and passed with the guard reverted — four of
            # six cases were decorative. The fleet caught it by checking the
            # tests bite rather than assuming they did.
            x = Nx.tensor(Enum.map(1..n, &rem(&1 * 7, 50)), type: type, backend: b)
            y = Nx.tensor(Enum.map(1..n, &rem(&1 * 13, 50)), type: type, backend: b)
            a = Nx.add(x, x)
            c = Nx.add(y, y)
            # THREE operands, and the third carries nonzero data past the second
            # splice. A two-way concat with an all-zero tail passes even when the
            # offsets are wrong — that false negative hid this during the fleet
            # run until a three-way case was tried.
            Nx.concatenate([a, c, a])
          end

          assert Nx.to_flat_list(build.(VulkanoBackend)) ==
                   Nx.to_flat_list(build.(Nx.BinaryBackend)),
                 "#{inspect(type)} concat of three #{n}-element tensors disagreed"
        end
      end
    end

    test "word-copyable dtypes still take the GPU path" do
      # The fix must not cost residency for the dtypes that were always correct.
      for type <- [{:f, 32}, {:f, 64}, {:s, 32}, {:u, 32}] do
        a = Nx.iota({5}, type: type, backend: VulkanoBackend)
        got = Nx.concatenate([a, a])

        assert match?(%VulkanoBackend{}, got.data),
               "#{inspect(type)} axis-0 concat should still be GPU-resident"
      end
    end
  end
end
