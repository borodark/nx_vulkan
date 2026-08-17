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
end
