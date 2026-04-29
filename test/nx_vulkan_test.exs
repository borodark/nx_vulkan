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
end
