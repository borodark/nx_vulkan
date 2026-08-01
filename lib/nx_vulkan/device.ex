defmodule Nx.Vulkan.Device do
  @moduledoc """
  GPU device classification for hardware-dependent perf heuristics.

  Some fused-kernel strategies only win on GPUs whose eager path is weak (few
  cores / low throughput) and lose on strong ones — e.g. the many-slot fused
  reduce wins ~4.4x on a GT 650M (Kepler) but regresses ~0.44x on an RTX 3060 Ti
  (Ampere), because the strong GPU's one-thread-per-slot eager reduce is already
  well-fed by thousands of slots. `class/0` labels the active Vulkan device
  `:weak` or `:strong` so the compiler can auto-enable those paths on weak
  hardware only.

  Classification is a heuristic over the Vulkan device name + type (there is no
  portable core-count in core Vulkan). It is intentionally conservative:
  anything not recognised as weak is `:strong`, so a maybe-regressing path is
  never auto-enabled on an unknown GPU. Override with `NXV_GPU_CLASS=weak|strong`
  (the override is not cached, so it is honoured immediately — useful for tests
  and for tuning a GPU the heuristic misjudges).
  """

  @doc """
  Device class — `:weak | :strong`. The env override `NXV_GPU_CLASS` wins and is
  read every call; the device-derived class is queried once and cached.
  """
  def class do
    case System.get_env("NXV_GPU_CLASS") do
      "weak" -> :weak
      "strong" -> :strong
      _ -> cached_device_class()
    end
  end

  @doc "True when the active GPU is classified `:weak`."
  def weak?, do: class() == :weak

  defp cached_device_class do
    case :persistent_term.get({__MODULE__, :class}, nil) do
      nil ->
        c =
          case safe_device_name() do
            {:ok, name, type} -> classify(name, type)
            # Unknown device: assume capable so we never auto-enable a path that
            # regresses on strong GPUs.
            _ -> :strong
          end

        :persistent_term.put({__MODULE__, :class}, c)
        c

      c ->
        c
    end
  end

  @doc """
  Classify a Vulkan device from its name + type string (e.g. `"NVIDIA GeForce
  GT 650M"`, `"DiscreteGpu"`). Pure — exposed for testing.

    * Software / integrated / virtual devices are `:weak` (low compute throughput).
    * The entry-level / older discrete NVIDIA GeForce **GT** line (Kepler/Fermi
      GT 6xx/7xx) is `:weak`; the high-end GTX/RTX line and everything else
      discrete is `:strong`.
  """
  def classify(_name, type) when type in ["Cpu", "IntegratedGpu", "VirtualGpu", "Other"],
    do: :weak

  def classify(name, _type) do
    n = String.downcase(name)

    cond do
      String.contains?(n, "geforce gt ") -> :weak
      String.contains?(n, "llvmpipe") -> :weak
      String.contains?(n, "software") -> :weak
      String.contains?(n, "swiftshader") -> :weak
      true -> :strong
    end
  end

  defp safe_device_name do
    Nx.Vulkan.NativeV.device_name()
  rescue
    _ -> :error
  catch
    _, _ -> :error
  end
end
