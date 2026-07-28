defmodule Nx.Vulkan.MatmulF32Test do
  @moduledoc """
  f32 matmul path (f32 storage, f64 accumulator) — verifies it dispatches on
  the GPU, keeps the f32 dtype, and matches a BinaryBackend reference (which
  itself accumulates in f64) to f32 round-off. Guards the f64 path too.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp maxdiff(a, b) do
    Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend), Nx.backend_copy(b, Nx.BinaryBackend))
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  defp pair({m, k, n}, type) do
    al = for i <- 1..(m * k), do: :math.sin(i * 0.7) * 0.5
    bl = for i <- 1..(k * n), do: :math.cos(i * 0.9) * 0.5
    av = Nx.tensor(al, type: type, backend: VulkanoBackend) |> Nx.reshape({m, k})
    bv = Nx.tensor(bl, type: type, backend: VulkanoBackend) |> Nx.reshape({k, n})
    ab = Nx.tensor(al, type: type, backend: Nx.BinaryBackend) |> Nx.reshape({m, k})
    bb = Nx.tensor(bl, type: type, backend: Nx.BinaryBackend) |> Nx.reshape({k, n})
    {Nx.dot(av, bv), Nx.dot(ab, bb)}
  end

  describe "f32 matmul on GPU" do
    for shape <- [{8, 8, 8}, {16, 32, 8}, {32, 64, 16}, {1, 128, 1}, {5, 30, 7}] do
      test "#{inspect(shape)}" do
        {got, ref} = pair(unquote(Macro.escape(shape)), {:f, 32})
        assert match?(%VulkanoBackend{}, got.data), "expected on-GPU dispatch"
        assert Nx.type(got) == {:f, 32}
        assert maxdiff(got, ref) < 1.0e-4
      end
    end
  end

  test "f64 matmul path still works (regression)" do
    {got, ref} = pair({16, 32, 8}, {:f, 64})
    assert match?(%VulkanoBackend{}, got.data)
    assert Nx.type(got) == {:f, 64}
    assert maxdiff(got, ref) < 1.0e-12
  end
end
