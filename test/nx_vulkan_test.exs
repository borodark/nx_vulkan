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
end
