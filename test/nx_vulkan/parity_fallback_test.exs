defmodule Nx.Vulkan.ParityFallbackTest do
  @moduledoc """
  Verifies that every op nx 0.13 removed from the `Nx.Backend` behaviour —
  and which therefore has NO explicit clause in VulkanoBackend anymore —
  still produces correct results on a VulkanoBackend tensor, matching a pure
  `BinaryBackend` reference in f64.

  Two dispatch paths are exercised:

    * **via `block/4`**: cholesky, determinant, solve, qr, lu, svd, eigh,
      top_k, cumulative_*, all_close (Nx builds a `Nx.Block.*` struct and
      calls the backend's block/4, which transfers to BinaryBackend).
    * **via primitive composition**: take, take_along_axis, logical_not
      (Nx lowers these to gather/slice/elementwise, each host-fallback).

  Also covers the fallback callbacks that stayed (sort, argsort,
  triangular_solve, product, reverse, window_sum, gather, argmax) as a guard
  against regressions from the Phase-1 cleanup.
  """

  use ExUnit.Case, async: false

  # The subject of this module IS the host-fallback path — every test here
  # asserts a fallback computes the right answer. Excluded from the strict
  # run (scripts/strict_test.sh), which asserts fallbacks do not happen.
  @moduletag :host_fallback_expected

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  # Run `build.(backend)` on both backends and assert the results match.
  # `build` receives a backend module and must construct its inputs on that
  # backend, then run the op. Handles tuple returns (qr/lu/svd/eigh).
  defp assert_parity(build) do
    ref = build.(Nx.BinaryBackend)
    got = build.(VulkanoBackend)
    assert_close(ref, got)
  end

  defp assert_close(a, b) when is_tuple(a) and is_tuple(b) do
    assert tuple_size(a) == tuple_size(b)

    Enum.zip(Tuple.to_list(a), Tuple.to_list(b))
    |> Enum.each(fn {x, y} -> assert_close(x, y) end)
  end

  defp assert_close(%Nx.Tensor{} = a, %Nx.Tensor{} = b) do
    ab = Nx.backend_copy(a, Nx.BinaryBackend)
    bb = Nx.backend_copy(b, Nx.BinaryBackend)

    assert Nx.shape(ab) == Nx.shape(bb),
           "shape mismatch: #{inspect(Nx.shape(ab))} vs #{inspect(Nx.shape(bb))}"

    assert Nx.type(ab) == Nx.type(bb),
           "type mismatch: #{inspect(Nx.type(ab))} vs #{inspect(Nx.type(bb))}"

    close? = Nx.all_close(ab, bb, atol: 1.0e-10, rtol: 1.0e-10) |> Nx.to_number()

    assert close? == 1,
           "value mismatch:\n  ref=#{inspect(Nx.to_flat_list(ab))}\n  got=#{inspect(Nx.to_flat_list(bb))}"
  end

  # --- fixtures (built per-backend) ---

  defp sym_pd(b), do: Nx.tensor([[6.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 4.0]], type: {:f, 64}, backend: b)
  defp gen3(b), do: Nx.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]], type: {:f, 64}, backend: b)
  defp lower3(b), do: Nx.tensor([[2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [1.0, 4.0, 5.0]], type: {:f, 64}, backend: b)
  defp b3(b), do: Nx.tensor([1.0, 2.0, 3.0], type: {:f, 64}, backend: b)
  defp vec(b), do: Nx.tensor([3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0], type: {:f, 64}, backend: b)
  defp mat(b), do: Nx.tensor([[3.0, 1.0, 4.0], [1.5, 5.0, 9.0], [2.0, 6.0, 5.0]], type: {:f, 64}, backend: b)

  describe "linalg — routed through block/4 (removed as callbacks in nx 0.13)" do
    test "cholesky", do: assert_parity(fn b -> Nx.LinAlg.cholesky(sym_pd(b)) end)
    test "determinant", do: assert_parity(fn b -> Nx.LinAlg.determinant(gen3(b)) end)
    test "solve", do: assert_parity(fn b -> Nx.LinAlg.solve(gen3(b), b3(b)) end)
    test "qr", do: assert_parity(fn b -> Nx.LinAlg.qr(gen3(b)) end)
    test "lu", do: assert_parity(fn b -> Nx.LinAlg.lu(gen3(b)) end)
    test "svd", do: assert_parity(fn b -> Nx.LinAlg.svd(gen3(b)) end)
    test "eigh", do: assert_parity(fn b -> Nx.LinAlg.eigh(sym_pd(b)) end)
  end

  describe "linalg — triangular_solve stayed a callback" do
    test "triangular_solve", do: assert_parity(fn b -> Nx.LinAlg.triangular_solve(lower3(b), b3(b)) end)
  end

  describe "reductions/scans removed as callbacks — via block/4" do
    test "top_k", do: assert_parity(fn b -> Nx.top_k(vec(b), k: 3) end)
    test "cumulative_sum", do: assert_parity(fn b -> Nx.cumulative_sum(vec(b)) end)
    test "cumulative_max", do: assert_parity(fn b -> Nx.cumulative_max(vec(b)) end)
    test "cumulative_min", do: assert_parity(fn b -> Nx.cumulative_min(vec(b)) end)
    test "cumulative_product", do: assert_parity(fn b -> Nx.cumulative_product(vec(b)) end)
    test "all_close (equal)", do: assert_parity(fn b -> Nx.all_close(vec(b), vec(b)) end)

    test "all_close (unequal)",
      do: assert_parity(fn b -> Nx.all_close(vec(b), Nx.add(vec(b), 1.0)) end)
  end

  describe "ops removed — via primitive composition" do
    test "take", do: assert_parity(fn b -> Nx.take(mat(b), Nx.tensor([0, 2], backend: b)) end)

    test "take_along_axis",
      do:
        assert_parity(fn b ->
          Nx.take_along_axis(mat(b), Nx.argsort(mat(b), axis: 1), axis: 1)
        end)

    test "logical_not",
      do: assert_parity(fn b -> Nx.logical_not(Nx.tensor([1, 0, 1, 0], type: {:u, 8}, backend: b)) end)
  end

  describe "fallback callbacks that stayed — regression guard" do
    test "sort", do: assert_parity(fn b -> Nx.sort(vec(b)) end)
    test "argsort", do: assert_parity(fn b -> Nx.argsort(vec(b)) end)
    test "product", do: assert_parity(fn b -> Nx.product(vec(b)) end)
    test "reverse", do: assert_parity(fn b -> Nx.reverse(vec(b)) end)
    test "window_sum", do: assert_parity(fn b -> Nx.window_sum(vec(b), {3}) end)
    test "argmax", do: assert_parity(fn b -> Nx.argmax(vec(b)) end)
    test "gather", do: assert_parity(fn b -> Nx.gather(mat(b), Nx.tensor([[0, 0], [1, 2]], backend: b)) end)
    test "all", do: assert_parity(fn b -> Nx.all(Nx.tensor([1, 1, 0], type: {:u, 8}, backend: b)) end)
    test "any", do: assert_parity(fn b -> Nx.any(Nx.tensor([0, 0, 1], type: {:u, 8}, backend: b)) end)
    test "to_batched", do: assert_parity(fn b -> Nx.to_batched(mat(b), 1) |> Enum.at(1) end)
  end
end
