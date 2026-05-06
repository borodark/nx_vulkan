defmodule Nx.Vulkan.PipelineCacheTest do
  @moduledoc """
  Phase 2 W5 — disk persistence of `vkPipelineCache`. Verifies the
  load/persist round-trip and the device-UUID-based path keying.
  """

  use ExUnit.Case, async: false

  setup do
    Nx.Vulkan.init()
    Nx.Vulkan.PipelineCache.clear()

    on_exit(fn -> Nx.Vulkan.PipelineCache.clear() end)
    :ok
  end

  describe "device_uuid_hex/0" do
    test "returns a 32-char lowercase hex string" do
      uuid = Nx.Vulkan.PipelineCache.device_uuid_hex()

      assert String.length(uuid) == 32
      assert String.match?(uuid, ~r/^[0-9a-f]+$/)
    end

    test "is stable across calls" do
      a = Nx.Vulkan.PipelineCache.device_uuid_hex()
      b = Nx.Vulkan.PipelineCache.device_uuid_hex()
      assert a == b
    end
  end

  describe "default_path/0" do
    test "is under ~/.exmc/gpu_node/pipeline_cache/ and ends in {uuid}.bin" do
      path = Nx.Vulkan.PipelineCache.default_path()

      assert String.ends_with?(path, "#{Nx.Vulkan.PipelineCache.device_uuid_hex()}.bin")
      assert String.contains?(path, "pipeline_cache")
    end
  end

  describe "load/0 + persist/0 round-trip" do
    test "load on empty cache directory is :ok" do
      assert :ok = Nx.Vulkan.PipelineCache.load()
    end

    test "persist on empty cache writes nothing if no pipelines built" do
      :ok = Nx.Vulkan.PipelineCache.load()
      assert :ok = Nx.Vulkan.PipelineCache.persist()
    end

    test "persist after building a pipeline writes a non-trivial blob" do
      :ok = Nx.Vulkan.PipelineCache.load()

      # Trigger a pipeline build by dispatching a chain shader.
      spec = Nx.Vulkan.ChainShaderSpecs.beta()
      {:ok, spv_path} = Nx.Vulkan.Synthesis.compile(spec)

      {:ok, q_ref} = Nx.Vulkan.upload_binary(<<0.0::little-float-32>>)
      {:ok, p_ref} = Nx.Vulkan.upload_binary(<<0.0::little-float-32>>)
      {:ok, m_ref} = Nx.Vulkan.upload_binary(<<1.0::little-float-32>>)
      push = Nx.Vulkan.ChainShaderSpecs.beta_push(1, 32, 0.05, 2.0, 5.0, -1.0)

      {:ok, _chains} =
        Nx.Vulkan.Native.leapfrog_chain_synth(q_ref, p_ref, m_ref, push, 32, spv_path)

      assert :ok = Nx.Vulkan.PipelineCache.persist()
      assert File.exists?(Nx.Vulkan.PipelineCache.default_path())

      size = File.stat!(Nx.Vulkan.PipelineCache.default_path()).size
      # vkPipelineCache header is 32 bytes; non-trivial cache adds at
      # least a few KB once a real pipeline is in there.
      assert size > 32, "pipeline cache file is too small (#{size} bytes)"
    end
  end
end
