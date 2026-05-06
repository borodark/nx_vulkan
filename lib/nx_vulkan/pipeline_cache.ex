defmodule Nx.Vulkan.PipelineCache do
  @moduledoc """
  Phase 2 W5 — disk persistence for spirit's `VkPipelineCache`.

  The vkPipelineCache holds compiled SPIR-V → device ISA. Without
  persistence, every BEAM restart pays the full backend-compile cost
  on first dispatch (~30-200 ms per shader on NVIDIA Linux). With
  persistence, the second-and-later restarts of a process that has
  the cache file get cache hits — `vkCreateComputePipelines` skips
  the compile entirely.

  ## File layout

      ~/.exmc/gpu_node/pipeline_cache/{device_uuid_hex}.bin

  The `device_uuid_hex` is the lowercased hex of `VkPhysicalDevice
  pipelineCacheUUID` (16 bytes → 32 hex chars). Different GPUs and
  different drivers produce different UUIDs, so the file is
  inherently device-specific. The driver itself silently rejects
  blobs whose embedded header doesn't match the running device, so
  even if the wrong file lands in the directory it just produces
  a fresh empty cache (with a stderr warning).

  ## Usage

      # At application start, before any pipelines are built:
      Nx.Vulkan.PipelineCache.load()

      # At application shutdown, or periodically:
      Nx.Vulkan.PipelineCache.persist()
  """

  @cache_dir Path.expand("~/.exmc/gpu_node/pipeline_cache")

  @doc """
  Load the disk cache (if present) into spirit's pipeline cache.
  Returns `:ok` on success, including the "no file yet" case.
  """
  def load do
    File.mkdir_p!(@cache_dir)

    case Nx.Vulkan.Native.pipeline_cache_load(default_path()) do
      :ok -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @doc """
  Persist spirit's current pipeline cache to disk via atomic
  write-temp-rename. Returns `:ok`.
  """
  def persist do
    File.mkdir_p!(@cache_dir)

    case Nx.Vulkan.Native.pipeline_cache_persist(default_path()) do
      :ok -> :ok
      {:error, reason} -> {:error, reason}
    end
  end

  @doc "Path the cache file would land at for the current device."
  def default_path do
    Path.join(@cache_dir, "#{device_uuid_hex()}.bin")
  end

  @doc "Lowercased hex of the device's 16-byte pipelineCacheUUID."
  def device_uuid_hex do
    case Nx.Vulkan.Native.device_uuid() do
      {:ok, bin} when byte_size(bin) == 16 ->
        Base.encode16(bin, case: :lower)

      {:error, :not_initialized} ->
        raise "Vulkan device not initialized — call Nx.Vulkan.Native.init() first"
    end
  end

  @doc """
  Convenience for benchmarks / tests — clear the cache directory.
  """
  def clear do
    File.rm_rf(@cache_dir)
    :ok
  end
end
