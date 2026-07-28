defmodule Nx.Vulkan.F32OpsTest do
  @moduledoc """
  f32 GPU compute path across the op families extended in F32_PLAN.md:
  elementwise binary/unary, axis reductions (f64 accumulator), 2-D transpose
  and conv. Each runs on a VulkanoBackend f32 tensor and is compared to a
  BinaryBackend reference. Algebraic/movement/reduction ops are exact; f32
  transcendentals (exp/log/pow/tanh/sigmoid) agree to ~f32 ulp, so those use a
  1e-5 tolerance.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  @tol 1.0e-5

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

  defp f32(list, shape, backend), do: Nx.tensor(list, type: {:f, 32}, backend: backend) |> Nx.reshape(shape)

  # unary: run f on both backends, assert on-GPU + f32 + within tol
  defp unary(shape, f, tol) do
    data = for i <- 1..Tuple.product(shape), do: :math.sin(i * 0.5) * 2.0
    got = f.(f32(data, shape, VulkanoBackend))
    ref = f.(f32(data, shape, Nx.BinaryBackend))
    assert match?(%VulkanoBackend{}, got.data), "expected on-GPU dispatch"
    assert Nx.type(got) == {:f, 32}
    assert maxdiff(got, ref) <= tol
  end

  defp binary(shape, f, tol) do
    a = for i <- 1..Tuple.product(shape), do: :math.sin(i * 0.5) * 2.0
    b = for i <- 1..Tuple.product(shape), do: :math.cos(i * 0.3) + 1.5
    got = f.(f32(a, shape, VulkanoBackend), f32(b, shape, VulkanoBackend))
    ref = f.(f32(a, shape, Nx.BinaryBackend), f32(b, shape, Nx.BinaryBackend))
    assert match?(%VulkanoBackend{}, got.data)
    assert Nx.type(got) == {:f, 32}
    assert maxdiff(got, ref) <= tol
  end

  describe "elementwise binary f32" do
    test "add", do: binary({64}, &Nx.add/2, 0.0)
    test "multiply", do: binary({64}, &Nx.multiply/2, 0.0)
    test "subtract", do: binary({64}, &Nx.subtract/2, 0.0)
    test "divide", do: binary({64}, &Nx.divide/2, @tol)
    test "max", do: binary({64}, &Nx.max/2, 0.0)
    test "min", do: binary({64}, &Nx.min/2, 0.0)
    test "pow", do: binary({64}, fn a, b -> Nx.pow(Nx.abs(a), b) end, @tol)
  end

  describe "elementwise unary f32" do
    test "exp", do: unary({64}, &Nx.exp/1, @tol)
    test "sqrt", do: unary({64}, fn x -> Nx.sqrt(Nx.abs(x)) end, @tol)
    test "sigmoid", do: unary({64}, &Nx.sigmoid/1, @tol)
    test "tanh", do: unary({64}, &Nx.tanh/1, @tol)
    test "negate", do: unary({64}, &Nx.negate/1, 0.0)
    test "abs", do: unary({64}, &Nx.abs/1, 0.0)
    test "floor", do: unary({64}, &Nx.floor/1, 0.0)
    test "ceil", do: unary({64}, &Nx.ceil/1, 0.0)
  end

  describe "reductions f32 (f64 accumulator)" do
    test "sum all", do: unary({128}, &Nx.sum/1, @tol)
    test "sum axis 0", do: unary({3, 4}, fn x -> Nx.sum(x, axes: [0]) end, @tol)
    test "reduce_max axis 1", do: unary({3, 4}, fn x -> Nx.reduce_max(x, axes: [1]) end, 0.0)
    test "reduce_min all", do: unary({50}, &Nx.reduce_min/1, 0.0)
  end

  describe "movement f32" do
    test "transpose non-square", do: unary({5, 30}, &Nx.transpose/1, 0.0)
    test "transpose square", do: unary({16, 16}, &Nx.transpose/1, 0.0)
  end

  test "conv f32 (multichannel) on GPU matches BinaryBackend" do
    ci = for i <- 1..(1 * 3 * 6 * 6), do: :math.sin(i * 0.2)
    ck = for i <- 1..(4 * 3 * 3 * 3), do: :math.cos(i * 0.3)
    got = Nx.conv(f32(ci, {1, 3, 6, 6}, VulkanoBackend), f32(ck, {4, 3, 3, 3}, VulkanoBackend))
    ref = Nx.conv(f32(ci, {1, 3, 6, 6}, Nx.BinaryBackend), f32(ck, {4, 3, 3, 3}, Nx.BinaryBackend))
    assert match?(%VulkanoBackend{}, got.data)
    assert Nx.type(got) == {:f, 32}
    assert maxdiff(got, ref) <= @tol
  end
end
