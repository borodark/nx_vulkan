defmodule Nx.Vulkan.PipelineCache do
  @moduledoc """
  Pipeline cache stub. The spirit C++ backend (which owned the
  VkPipelineCache) has been removed. Vulkano manages its own
  pipeline caching internally. These functions are no-ops retained
  for API compatibility with callers that haven't been updated.
  """

  @cache_dir Path.expand("~/.exmc/gpu_node/pipeline_cache")

  def load, do: :ok
  def persist, do: :ok
  def default_path, do: Path.join(@cache_dir, "vulkano.bin")
  def device_uuid_hex, do: "0000000000000000" <> "0000000000000000"

  def clear do
    # `_ =` on purpose: clearing a cache dir that may not exist is not a failure.
    _ = File.rm_rf(@cache_dir)
    :ok
  end
end
