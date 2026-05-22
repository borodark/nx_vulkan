defmodule Nx.Vulkan.VulkanoBackendTest do
  @moduledoc """
  Coverage for `Nx.Vulkan.VulkanoBackend` — the pure-Rust (vulkano) Nx
  backend that landed in Mission II. The sibling `Nx.Vulkan.Backend`
  (C++ spirit) already has ~144 test references in `nx_vulkan_test.exs`;
  this file is the equivalent regression sweep for the Rust path.

  Host-fallback specific tests (pad, put_slice, indexed_put, broadcast,
  concatenate, gather, take, select, as_type, slice) live in
  `vulkano_backend_host_fallback_test.exs`. This file covers the
  GPU-resident fast paths and the storage / movement callbacks.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  @moduletag :vulkan_live

  defp v(t), do: Nx.backend_transfer(t, VulkanoBackend)

  defp f32(list) do
    Nx.tensor(list, type: :f32, backend: Nx.BinaryBackend)
  end

  defp close?(actual, expected, tol \\ 1.0e-5) do
    actual_list = Nx.to_flat_list(actual)
    expected_list = Nx.to_flat_list(expected)

    length(actual_list) == length(expected_list) and
      Enum.zip(actual_list, expected_list)
      |> Enum.all?(fn {a, e} -> abs(a - e) <= tol end)
  end

  describe "storage round-trip" do
    test "from_binary + to_binary preserves bytes (f32)" do
      data = <<1.0::float-32-native, 2.0::float-32-native, 3.0::float-32-native>>
      t = Nx.from_binary(data, :f32, backend: VulkanoBackend)
      assert t.data.__struct__ == VulkanoBackend
      assert Nx.to_binary(t) == data
      assert Nx.to_flat_list(t) == [1.0, 2.0, 3.0]
    end

    test "from_binary + to_binary preserves bytes (f64)" do
      data = <<1.0::float-64-native, 2.0::float-64-native>>
      t = Nx.from_binary(data, :f64, backend: VulkanoBackend)
      assert Nx.to_binary(t) == data
    end

    test "backend_transfer BinaryBackend -> VulkanoBackend round-trip" do
      orig = f32([1.0, 2.0, 3.0, 4.0])
      vk = Nx.backend_transfer(orig, VulkanoBackend)
      assert vk.data.__struct__ == VulkanoBackend
      back = Nx.backend_transfer(vk, Nx.BinaryBackend)
      assert back.data.__struct__ == Nx.BinaryBackend
      assert Nx.to_flat_list(back) == [1.0, 2.0, 3.0, 4.0]
    end

    test "constant fills tensor with scalar" do
      t = Nx.tensor(7.5, type: :f32, backend: VulkanoBackend) |> Nx.broadcast({4})
      # Note: broadcast is a host-fallback so the result is BinaryBackend
      # after Tier 1 — what matters is the values are correct.
      assert Nx.to_flat_list(t) == [7.5, 7.5, 7.5, 7.5]
    end

    test "iota produces 0..n-1" do
      t = Nx.iota({5}, type: :f32, backend: VulkanoBackend)
      assert t.data.__struct__ == VulkanoBackend
      assert Nx.to_flat_list(t) == [0.0, 1.0, 2.0, 3.0, 4.0]
    end

    test "eye produces identity matrix" do
      t = Nx.eye(3, type: :f32, backend: VulkanoBackend)
      assert t.data.__struct__ == VulkanoBackend
      assert Nx.to_flat_list(t) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    end
  end

  describe "binary SPV ops (GPU fast path)" do
    test "add" do
      a = v(f32([1.0, 2.0, 3.0, 4.0]))
      b = v(f32([10.0, 20.0, 30.0, 40.0]))
      assert Nx.to_flat_list(Nx.add(a, b)) == [11.0, 22.0, 33.0, 44.0]
    end

    test "multiply" do
      a = v(f32([1.0, 2.0, 3.0]))
      b = v(f32([2.0, 3.0, 4.0]))
      assert Nx.to_flat_list(Nx.multiply(a, b)) == [2.0, 6.0, 12.0]
    end

    test "subtract" do
      a = v(f32([10.0, 20.0, 30.0]))
      b = v(f32([1.0, 2.0, 3.0]))
      assert Nx.to_flat_list(Nx.subtract(a, b)) == [9.0, 18.0, 27.0]
    end

    test "divide" do
      a = v(f32([10.0, 20.0, 30.0]))
      b = v(f32([2.0, 4.0, 5.0]))
      assert Nx.to_flat_list(Nx.divide(a, b)) == [5.0, 5.0, 6.0]
    end

    test "pow" do
      a = v(f32([2.0, 3.0, 4.0]))
      b = v(f32([2.0, 2.0, 2.0]))
      assert close?(Nx.pow(a, b), f32([4.0, 9.0, 16.0]))
    end

    test "max elementwise" do
      a = v(f32([1.0, 5.0, 3.0]))
      b = v(f32([2.0, 4.0, 6.0]))
      assert Nx.to_flat_list(Nx.max(a, b)) == [2.0, 5.0, 6.0]
    end

    test "min elementwise" do
      a = v(f32([1.0, 5.0, 3.0]))
      b = v(f32([2.0, 4.0, 6.0]))
      assert Nx.to_flat_list(Nx.min(a, b)) == [1.0, 4.0, 3.0]
    end
  end

  describe "unary SPV ops (GPU fast path)" do
    test "exp" do
      a = v(f32([0.0, 1.0, 2.0]))
      assert close?(Nx.exp(a), f32([1.0, :math.exp(1.0), :math.exp(2.0)]), 1.0e-4)
    end

    test "log" do
      a = v(f32([1.0, :math.exp(1.0), :math.exp(2.0)]))
      assert close?(Nx.log(a), f32([0.0, 1.0, 2.0]), 1.0e-4)
    end

    test "sqrt" do
      a = v(f32([1.0, 4.0, 9.0, 16.0]))
      assert close?(Nx.sqrt(a), f32([1.0, 2.0, 3.0, 4.0]))
    end

    test "abs" do
      a = v(f32([-3.0, -1.5, 0.0, 1.5, 3.0]))
      assert Nx.to_flat_list(Nx.abs(a)) == [3.0, 1.5, 0.0, 1.5, 3.0]
    end

    test "negate" do
      a = v(f32([1.0, -2.0, 3.0]))
      assert Nx.to_flat_list(Nx.negate(a)) == [-1.0, 2.0, -3.0]
    end

    test "sigmoid" do
      a = v(f32([0.0]))
      [val] = Nx.to_flat_list(Nx.sigmoid(a))
      assert_in_delta val, 0.5, 1.0e-5
    end

    test "tanh" do
      a = v(f32([0.0, 1.0]))
      result = Nx.to_flat_list(Nx.tanh(a))
      assert_in_delta Enum.at(result, 0), 0.0, 1.0e-5
      assert_in_delta Enum.at(result, 1), :math.tanh(1.0), 1.0e-4
    end

    test "floor / ceil / sign" do
      a = v(f32([-1.7, -0.3, 0.0, 0.3, 1.7]))
      assert Nx.to_flat_list(Nx.floor(a)) == [-2.0, -1.0, 0.0, 0.0, 1.0]
      assert Nx.to_flat_list(Nx.ceil(a)) == [-1.0, 0.0, 0.0, 1.0, 2.0]
      assert Nx.to_flat_list(Nx.sign(a)) == [-1.0, -1.0, 0.0, 1.0, 1.0]
    end
  end

  describe "reductions (SPV fast path)" do
    test "sum all-axes" do
      a = v(f32([1.0, 2.0, 3.0, 4.0]))
      assert Nx.to_number(Nx.sum(a)) == 10.0
    end

    test "reduce_max all-axes" do
      a = v(f32([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]))
      assert Nx.to_number(Nx.reduce_max(a)) == 9.0
    end

    test "reduce_min all-axes" do
      a = v(f32([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0]))
      assert Nx.to_number(Nx.reduce_min(a)) == 1.0
    end

    test "sum along trailing axis" do
      a = v(f32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
      result = Nx.sum(a, axes: [1])
      assert Nx.to_flat_list(result) == [6.0, 15.0]
    end
  end

  describe "shape / movement (zero-copy or near)" do
    test "reshape preserves data" do
      a = v(f32([1.0, 2.0, 3.0, 4.0]))
      r = Nx.reshape(a, {2, 2})
      assert r.data.__struct__ == VulkanoBackend
      assert Nx.shape(r) == {2, 2}
      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0, 4.0]
    end

    test "squeeze rank-1" do
      a = v(Nx.tensor([[1.0, 2.0, 3.0]], type: :f32, backend: Nx.BinaryBackend))
      r = Nx.squeeze(a, axes: [0])
      assert Nx.shape(r) == {3}
      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0]
    end

    test "transpose 2D [1,0] fast path" do
      a = v(Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], type: :f32, backend: Nx.BinaryBackend))
      r = Nx.transpose(a)
      assert Nx.shape(r) == {2, 3}
      assert Nx.to_flat_list(r) == [1.0, 3.0, 5.0, 2.0, 4.0, 6.0]
    end
  end

  describe "matmul (dot rank-2 f32 fast path)" do
    test "{2,3} @ {3,2}" do
      a = v(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32, backend: Nx.BinaryBackend))
      b = v(Nx.tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], type: :f32, backend: Nx.BinaryBackend))
      r = Nx.dot(a, b)
      # Expected: [[58, 64], [139, 154]]
      assert Nx.to_flat_list(r) == [58.0, 64.0, 139.0, 154.0]
      assert Nx.shape(r) == {2, 2}
    end

    test "identity matmul" do
      a = v(Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], type: :f32, backend: Nx.BinaryBackend))
      i = v(Nx.eye(3, type: :f32, backend: Nx.BinaryBackend))
      assert close?(Nx.dot(a, i), a)
    end
  end

  describe "mixed-backend composition (Tier 1 makes this realistic)" do
    test "vulkano + binary = follow-up op succeeds" do
      # After Tier 1, host-fallback ops return BinaryBackend tensors.
      # A follow-up op with a VulkanoBackend operand must work.
      vk = v(f32([1.0, 2.0, 3.0]))
      bin_result = Nx.broadcast(vk, {6, 3}) # broadcast = host-fallback -> BinaryBackend
      # Confirm Tier 1 contract on the intermediate
      assert bin_result.data.__struct__ == Nx.BinaryBackend
      # Now add a VulkanoBackend tensor to it
      other_vk = v(Nx.iota({6, 3}, type: :f32, backend: Nx.BinaryBackend))
      combined = Nx.add(bin_result, other_vk)
      assert length(Nx.to_flat_list(combined)) == 18
    end
  end
end
