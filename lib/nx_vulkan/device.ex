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

  @doc """
  Whether the active device supports 64-bit floats in shaders (`shaderFloat64`).

  Every `_f64.spv` kernel and every generated f64 fused kernel needs it; without
  it, pipeline creation fails at dispatch, so callers must gate on this and take
  the host fallback. Queried once and cached. Conservatively `false` if the
  device cannot be reached, so an f64 GPU path is never attempted blind.

  Note this is not a niche capability for this backend: the eager path is
  f64-first, and even the *f32* fused reduce accumulates in `double`. A device
  without `shaderFloat64` is severely limited here regardless of this flag.

  Override with `NXV_F64=0|1` (read every call, like `NXV_GPU_CLASS`) to force
  the f64 GPU paths off or on — `0` is how the host-fallback path is exercised
  on a machine whose GPU does support f64.
  """
  def f64? do
    case System.get_env("NXV_F64") do
      "0" -> false
      "1" -> true
      _ -> cached_f64()
    end
  end

  defp cached_f64 do
    case :persistent_term.get({__MODULE__, :f64}, nil) do
      nil ->
        v =
          case safe_supports_f64() do
            {:ok, bool} when is_boolean(bool) -> bool
            _ -> false
          end

        :persistent_term.put({__MODULE__, :f64}, v)
        v

      v ->
        v
    end
  end

  defp safe_supports_f64 do
    Nx.Vulkan.NativeV.device_supports_f64()
  rescue
    _ -> :error
  catch
    _, _ -> :error
  end

  defp safe_device_name do
    Nx.Vulkan.NativeV.device_name()
  rescue
    _ -> :error
  catch
    _, _ -> :error
  end
end
