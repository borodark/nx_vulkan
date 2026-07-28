defmodule Nx.Vulkan.TransposeTest do
  @moduledoc """
  Regression guard for 2-D transpose. The legacy transpose.spv is an f32
  shader; applying it to f64 tensors strided the buffer as 4-byte floats and
  corrupted the data (undetected because no test transposed an f64 tensor and
  checked values). The f64 path now uses transpose_f64.spv; other shapes/types
  host-fall-back. Verified against BinaryBackend.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp maxdiff(a, b) do
    assert Nx.shape(a) == Nx.shape(b)
    Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend), b) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  defp both(shape, type) do
    {m, n} = shape
    data = Enum.map(1..(m * n), &(&1 * 1.0))
    v = Nx.tensor(data, type: type, backend: VulkanoBackend) |> Nx.reshape(shape) |> Nx.transpose()
    b = Nx.tensor(data, type: type, backend: Nx.BinaryBackend) |> Nx.reshape(shape) |> Nx.transpose()
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

  describe "unsupported transpose host-falls-back but stays correct" do
    test "rank-3 permutation falls back" do
      d = Enum.map(1..24, &(&1 * 1.0))
      v = Nx.tensor(d, type: {:f, 64}, backend: VulkanoBackend) |> Nx.reshape({2, 3, 4}) |> Nx.transpose(axes: [2, 0, 1])
      b = Nx.tensor(d, type: {:f, 64}, backend: Nx.BinaryBackend) |> Nx.reshape({2, 3, 4}) |> Nx.transpose(axes: [2, 0, 1])
      refute match?(%VulkanoBackend{}, v.data)
      assert maxdiff(v, b) == 0.0
    end
  end

  test "transpose feeds dot correctly (A^T · b), the logistic-regression path" do
    a = Nx.tensor(Enum.map(1..15, &(&1 * 0.3)), type: {:f, 64}, backend: VulkanoBackend) |> Nx.reshape({3, 5})
    x = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 64}, backend: VulkanoBackend) |> Nx.reshape({3, 1})
    ab = Nx.backend_copy(a, Nx.BinaryBackend)
    xb = Nx.backend_copy(x, Nx.BinaryBackend)
    got = Nx.dot(Nx.transpose(a), x)
    assert maxdiff(got, Nx.dot(Nx.transpose(ab), xb)) < 1.0e-12
  end
end
