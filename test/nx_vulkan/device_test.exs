defmodule Nx.Vulkan.DeviceTest do
  @moduledoc """
  Device classification used by hardware-dependent perf heuristics (the
  many-slot fused reduce auto-enables only on `:weak` GPUs).
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.Device

  describe "classify/2 (pure name+type heuristic)" do
    test "software / integrated / virtual devices are weak" do
      assert Device.classify("llvmpipe (LLVM 15)", "Cpu") == :weak
      assert Device.classify("Intel UHD Graphics", "IntegratedGpu") == :weak
      assert Device.classify("Virtio-GPU", "VirtualGpu") == :weak
      assert Device.classify("whatever", "Other") == :weak
    end

    test "older low-end discrete NVIDIA GeForce GT line is weak" do
      assert Device.classify("NVIDIA GeForce GT 650M", "DiscreteGpu") == :weak
      assert Device.classify("NVIDIA GeForce GT 750M", "DiscreteGpu") == :weak
    end

    test "software rasterisers by name are weak even if reported as a GPU" do
      assert Device.classify("llvmpipe", "DiscreteGpu") == :weak
      assert Device.classify("SwiftShader Device", "DiscreteGpu") == :weak
    end

    test "high-end / modern discrete GPUs are strong" do
      assert Device.classify("NVIDIA GeForce RTX 3060 Ti", "DiscreteGpu") == :strong
      assert Device.classify("NVIDIA GeForce GTX 1080", "DiscreteGpu") == :strong
      assert Device.classify("AMD Radeon RX 6800", "DiscreteGpu") == :strong
    end
  end

  describe "class/0 env override (uncached, wins over the device)" do
    test "NXV_GPU_CLASS forces the class regardless of the real device" do
      System.put_env("NXV_GPU_CLASS", "weak")
      assert Device.class() == :weak
      assert Device.weak?()

      System.put_env("NXV_GPU_CLASS", "strong")
      assert Device.class() == :strong
      refute Device.weak?()
    after
      System.delete_env("NXV_GPU_CLASS")
    end
  end

  describe "class/0 on the live device" do
    setup do
      {:ok, _} = Application.ensure_all_started(:nx_vulkan)
      System.delete_env("NXV_GPU_CLASS")
      :ok
    end

    test "returns a known class for the actual GPU" do
      assert Device.class() in [:weak, :strong]
    end
  end
end
