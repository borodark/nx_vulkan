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

  describe "f32 matmul accumulator policy" do
    setup do
      prev = VulkanoBackend.f32_matmul_accumulator()
      on_exit(fn -> VulkanoBackend.put_f32_matmul_accumulator(prev) end)
      :ok
    end

    test "defaults to :f64" do
      assert VulkanoBackend.f32_matmul_accumulator() == :f64
    end

    test "both policies dispatch on GPU and stay correct (well-conditioned)" do
      for policy <- [:f64, :f32] do
        VulkanoBackend.put_f32_matmul_accumulator(policy)
        {got, ref} = pair({32, 128, 16}, {:f, 32})
        assert match?(%VulkanoBackend{}, got.data), "#{policy} should stay on GPU"
        assert Nx.type(got) == {:f, 32}
        assert maxdiff(got, ref) < 1.0e-4
      end
    end

    test ":f32 accumulator is less accurate than :f64 on an ill-conditioned dot" do
      # [1e9, 1, ..., 1, -1e9]·[1,..,1]: the f32 accumulator drops the 1s once
      # 1e9 is resident (ulp ~64 > 1); the f64 accumulator keeps them.
      k = 512
      a = for(i <- 0..(k - 1), do: cond(do: (i == 0 -> 1.0e9; i == k - 1 -> -1.0e9; true -> 1.0)))
      b = List.duplicate(1.0, k)
      truth = k - 2

      run = fn ->
        av = Nx.tensor(a, type: {:f, 32}, backend: VulkanoBackend) |> Nx.reshape({1, k})
        bv = Nx.tensor(b, type: {:f, 32}, backend: VulkanoBackend) |> Nx.reshape({k, 1})
        Nx.dot(av, bv) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list() |> hd()
      end

      VulkanoBackend.put_f32_matmul_accumulator(:f64)
      assert_in_delta run.(), truth, 1.0

      VulkanoBackend.put_f32_matmul_accumulator(:f32)
      # f32 accumulator collapses (drops the 1.0 terms) -> far from truth
      assert abs(run.() - truth) > 100.0
    end
  end
end
