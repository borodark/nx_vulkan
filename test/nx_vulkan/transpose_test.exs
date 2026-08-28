defmodule Nx.Vulkan.TransposeTest do
  @moduledoc """
  Regression guard for transpose. The legacy transpose.spv is an f32 shader;
  applying it to f64 tensors strided the buffer as 4-byte floats and corrupted
  the data (undetected because no test transposed an f64 tensor and checked
  values). The f64 path uses transpose_f64.spv.

  Rank-2 `[1, 0]` takes the tiled shader; every other permutation up to rank 4
  goes through the generic `transpose_nd` shader. Rank 5+ and non-f32/f64 types
  still host-fall-back. All verified against BinaryBackend.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp maxdiff(a, b) do
    assert Nx.shape(a) == Nx.shape(b)

    Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend), b)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  defp both(shape, type) do
    {m, n} = shape
    data = Enum.map(1..(m * n), &(&1 * 1.0))

    v =
      Nx.tensor(data, type: type, backend: VulkanoBackend) |> Nx.reshape(shape) |> Nx.transpose()

    b =
      Nx.tensor(data, type: type, backend: Nx.BinaryBackend)
      |> Nx.reshape(shape)
      |> Nx.transpose()

    {v, b}
  end

  describe "f64 2-D transpose runs on GPU and matches BinaryBackend" do
    for shape <- [{2, 2}, {3, 3}, {4, 4}, {8, 8}, {30, 5}, {5, 30}, {2, 4}, {1, 8}, {8, 1}] do
      test "#{inspect(shape)}" do
        {v, b} = both(unquote(Macro.escape(shape)), {:f, 64})
        assert match?(%VulkanoBackend{}, v.data), "expected on-GPU dispatch"
        assert maxdiff(v, b) == 0.0
      end
    end
  end

  describe "f32 2-D transpose runs on GPU and matches BinaryBackend" do
    test "f32 non-square on GPU" do
      {v, b} = both({5, 30}, {:f, 32})
      assert match?(%VulkanoBackend{}, v.data)
      assert Nx.type(v) == {:f, 32}
      assert maxdiff(v, b) == 0.0
    end
  end

  describe "general permutations run on the GPU (transpose_nd)" do
    # The tiled rank-2/[1,0] shader is the fast path for matrices; every other
    # permutation up to rank 4 goes through the generic transpose_nd shader
    # rather than host-falling-back. A permutation moves bits without doing
    # arithmetic, so parity here is exact, not approximate.
    for {shape, axes} <- [
          {{2, 3, 4}, [2, 0, 1]},
          {{2, 3, 4}, [1, 0, 2]},
          {{2, 3, 4}, [0, 2, 1]},
          {{2, 3, 4, 5}, [1, 0, 2, 3]},
          {{2, 3, 4, 5}, [3, 2, 1, 0]},
          {{2, 3, 4, 5}, [0, 2, 3, 1]}
        ] do
      test "#{inspect(shape)} axes #{inspect(axes)} stays on the GPU and is exact" do
        shape = unquote(Macro.escape(shape))
        axes = unquote(axes)
        d = Enum.map(1..Tuple.product(shape), &(&1 * 1.0))

        v =
          Nx.tensor(d, type: {:f, 64}, backend: VulkanoBackend)
          |> Nx.reshape(shape)
          |> Nx.transpose(axes: axes)

        b =
          Nx.tensor(d, type: {:f, 64}, backend: Nx.BinaryBackend)
          |> Nx.reshape(shape)
          |> Nx.transpose(axes: axes)

        assert match?(%VulkanoBackend{}, v.data)
        assert maxdiff(v, b) == 0.0
      end
    end

    test "rank 5 still host-falls-back but stays correct" do
      d = Enum.map(1..32, &(&1 * 1.0))
      axes = [4, 3, 2, 1, 0]

      v =
        Nx.tensor(d, type: {:f, 64}, backend: VulkanoBackend)
        |> Nx.reshape({2, 2, 2, 2, 2})
        |> Nx.transpose(axes: axes)

      b =
        Nx.tensor(d, type: {:f, 64}, backend: Nx.BinaryBackend)
        |> Nx.reshape({2, 2, 2, 2, 2})
        |> Nx.transpose(axes: axes)

      refute match?(%VulkanoBackend{}, v.data)
      assert maxdiff(v, b) == 0.0
    end
  end

  test "transpose feeds dot correctly (A^T · b), the logistic-regression path" do
    a =
      Nx.tensor(Enum.map(1..15, &(&1 * 0.3)), type: {:f, 64}, backend: VulkanoBackend)
      |> Nx.reshape({3, 5})

    x = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 64}, backend: VulkanoBackend) |> Nx.reshape({3, 1})
    ab = Nx.backend_copy(a, Nx.BinaryBackend)
    xb = Nx.backend_copy(x, Nx.BinaryBackend)
    got = Nx.dot(Nx.transpose(a), x)
    assert maxdiff(got, Nx.dot(Nx.transpose(ab), xb)) < 1.0e-12
  end
end
