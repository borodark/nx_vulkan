defmodule Nx.VulkanTest do
  use ExUnit.Case, async: false
  # No doctest — the moduledoc shows usage shape, not an executable
  # example (device name varies per host; tensor ops aren't wired
  # yet in v0.0.1).

  describe "v0.0.1 — bootstrap" do
    test "init/0 returns :ok on a Vulkan-capable host" do
      assert :ok = Nx.Vulkan.init()
    end

    test "init/0 is idempotent" do
      assert :ok = Nx.Vulkan.init()
      assert :ok = Nx.Vulkan.init()
    end

    test "device_name/0 returns a non-empty string after init" do
      :ok = Nx.Vulkan.init()
      name = Nx.Vulkan.device_name()
      assert is_binary(name)
      assert String.length(name) > 0
    end

    test "has_f64?/0 returns a boolean" do
      :ok = Nx.Vulkan.init()
      assert is_boolean(Nx.Vulkan.has_f64?())
    end
  end

  describe "v0.0.2 — tensor round-trip" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "upload_f32 + download_f32 is the identity (small)" do
      {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      assert {:ok, [1.0, 2.0, 3.0, 4.0]} = Nx.Vulkan.download_f32(t, 4)
    end

    test "byte_size matches the uploaded length" do
      {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      assert Nx.Vulkan.byte_size(t) == 12
    end

    test "round-trip preserves single value" do
      {:ok, t} = Nx.Vulkan.upload_f32([42.0])
      assert {:ok, [42.0]} = Nx.Vulkan.download_f32(t, 1)
    end

    test "round-trip preserves negative + sub-integer values" do
      input = [-1.5, 0.25, 3.14159, -0.0001, 1.0e6]
      {:ok, t} = Nx.Vulkan.upload_f32(input)
      assert {:ok, output} = Nx.Vulkan.download_f32(t, 5)

      Enum.zip(input, output)
      |> Enum.each(fn {a, b} ->
        assert_in_delta a, b, 1.0e-3
      end)
    end

    test "round-trip works at 1024 elements (multi-workgroup-class buffer)" do
      input = Enum.map(1..1024, fn i -> i / 1.0 end)
      {:ok, t} = Nx.Vulkan.upload_f32(input)
      assert {:ok, output} = Nx.Vulkan.download_f32(t, 1024)
      assert length(output) == 1024
      assert hd(output) == 1.0
      assert List.last(output) == 1024.0
    end

    test "size_mismatch error when download size != upload size" do
      {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      assert {:error, :size_mismatch} = Nx.Vulkan.download_binary(t, 99)
    end

    test "upload_binary takes raw bytes" do
      bin = <<1.0::float-32-native, 2.0::float-32-native>>
      {:ok, t} = Nx.Vulkan.upload_binary(bin)
      {:ok, ^bin} = Nx.Vulkan.download_binary(t, 8)
    end

    test "many tensors live concurrently and are GC'd cleanly" do
      tensors =
        for i <- 1..50 do
          {:ok, t} = Nx.Vulkan.upload_f32([i * 1.0])
          t
        end

      # Verify all 50 are independent
      results =
        Enum.map(tensors, fn t ->
          {:ok, [v]} = Nx.Vulkan.download_f32(t, 1)
          v
        end)

      assert results == Enum.map(1..50, &(&1 * 1.0))
    end
  end

  describe "v0.0.3 — elementwise binary" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "add" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      {:ok, b} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0, 40.0])
      {:ok, c} = Nx.Vulkan.add(a, b)
      assert {:ok, [11.0, 22.0, 33.0, 44.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "multiply" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      {:ok, b} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0, 40.0])
      {:ok, c} = Nx.Vulkan.multiply(a, b)
      assert {:ok, [10.0, 40.0, 90.0, 160.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "subtract" do
      {:ok, a} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      {:ok, c} = Nx.Vulkan.subtract(a, b)
      assert {:ok, [9.0, 18.0, 27.0]} = Nx.Vulkan.download_f32(c, 3)
    end

    test "divide" do
      {:ok, a} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0, 40.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 4.0, 5.0, 8.0])
      {:ok, c} = Nx.Vulkan.divide(a, b)
      assert {:ok, [5.0, 5.0, 6.0, 5.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "max + min" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 5.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 4.0, 3.0])

      {:ok, mx} = Nx.Vulkan.max(a, b)
      assert {:ok, [2.0, 5.0, 3.0]} = Nx.Vulkan.download_f32(mx, 3)

      {:ok, mn} = Nx.Vulkan.min(a, b)
      assert {:ok, [1.0, 4.0, 3.0]} = Nx.Vulkan.download_f32(mn, 3)
    end

    test "pow" do
      {:ok, a} = Nx.Vulkan.upload_f32([2.0, 3.0, 4.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 2.0, 0.5])
      {:ok, c} = Nx.Vulkan.pow(a, b)
      {:ok, [r1, r2, r3]} = Nx.Vulkan.download_f32(c, 3)
      assert_in_delta r1, 4.0, 1.0e-3
      assert_in_delta r2, 9.0, 1.0e-3
      assert_in_delta r3, 2.0, 1.0e-3
    end

    test "size mismatch returns :size_mismatch" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      assert {:error, :size_mismatch} = Nx.Vulkan.add(a, b)
    end

    test "chained ops keep producing fresh tensors" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0])

      # (a + b) * a = [11, 22, 33] * [1, 2, 3] = [11, 44, 99]
      {:ok, c} = Nx.Vulkan.add(a, b)
      {:ok, d} = Nx.Vulkan.multiply(c, a)
      assert {:ok, [11.0, 44.0, 99.0]} = Nx.Vulkan.download_f32(d, 3)
    end

    test "1024 elements (multi-workgroup)" do
      input = Enum.map(1..1024, fn i -> i * 1.0 end)
      {:ok, a} = Nx.Vulkan.upload_f32(input)
      {:ok, b} = Nx.Vulkan.upload_f32(input)
      {:ok, c} = Nx.Vulkan.add(a, b)
      {:ok, result} = Nx.Vulkan.download_f32(c, 1024)
      assert hd(result) == 2.0
      assert List.last(result) == 2048.0
    end
  end
end
