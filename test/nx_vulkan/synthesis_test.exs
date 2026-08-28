defmodule Nx.Vulkan.SynthesisTest do
  @moduledoc """
  Phase 1 — `Nx.Vulkan.Synthesis.compile/1` end-to-end: render GLSL
  → invoke `glslangValidator` → cache by sha256 → return SPV path.

  These tests require `glslangValidator` on PATH. If it's missing,
  the synthesis path is broken and `compile/1` should return an
  error tuple — the tests cover both the happy path and that
  failure shape.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.Synthesis

  setup do
    Synthesis.clear_cache()
    on_exit(fn -> Synthesis.clear_cache() end)
    :ok
  end

  describe "compile/1 cold path" do
    test "Beta spec compiles to a non-empty SPV file under the cache dir" do
      spec = Nx.Vulkan.ChainShaderSpecs.beta()
      {:ok, spv_path} = Synthesis.compile(spec)

      assert File.exists?(spv_path)
      assert String.ends_with?(spv_path, ".spv")
      assert File.stat!(spv_path).size > 0

      # SPIR-V binaries start with the magic word 0x07230203.
      <<magic::little-32, _rest::binary>> = File.read!(spv_path)

      assert magic == 0x07230203,
             "not a SPIR-V file (got magic 0x#{Integer.to_string(magic, 16)})"
    end

    test "Gamma + Lognormal also compile cleanly" do
      for spec_fn <- [
            &Nx.Vulkan.ChainShaderSpecs.gamma/0,
            &Nx.Vulkan.ChainShaderSpecs.lognormal/0
          ] do
        {:ok, path} = Synthesis.compile(spec_fn.())
        assert File.exists?(path)
      end
    end
  end

  describe "compile/1 cache hit" do
    test "second call returns the same path and is fast" do
      spec = Nx.Vulkan.ChainShaderSpecs.beta()

      # Warm the cache.
      {:ok, path1} = Synthesis.compile(spec)
      stat1 = File.stat!(path1)

      # Cache hit: should not re-compile.
      {us, {:ok, path2}} = :timer.tc(fn -> Synthesis.compile(spec) end)
      stat2 = File.stat!(path2)

      assert path1 == path2
      assert stat1.mtime == stat2.mtime, "cache hit should not rewrite the file"

      # Cache hit must be much faster than a glslangValidator invocation.
      # Cold compile takes 50-200 ms; cache hit should be < 50 ms.
      assert us < 50_000, "cache hit took #{div(us, 1000)} ms"
    end

    test "different specs map to different cache files" do
      {:ok, beta_path} = Synthesis.compile(Nx.Vulkan.ChainShaderSpecs.beta())
      {:ok, gamma_path} = Synthesis.compile(Nx.Vulkan.ChainShaderSpecs.gamma())

      assert beta_path != gamma_path
      assert File.exists?(beta_path)
      assert File.exists?(gamma_path)
    end
  end

  describe "compile_with_source/1" do
    test "returns both the SPV path and the rendered GLSL" do
      spec = Nx.Vulkan.ChainShaderSpecs.beta()
      {:ok, spv_path, glsl} = Synthesis.compile_with_source(spec)

      assert File.exists?(spv_path)
      assert is_binary(glsl)
      assert glsl =~ "#version 450"
      assert glsl =~ "for family: beta"
    end
  end

  describe "compile/1 failure path" do
    test "returns {:error, _} on a deliberately broken spec" do
      bad_spec = %Nx.Vulkan.ShaderTemplate.FamilySpec{
        name: "broken",
        push_fields: "    float alpha;",
        # Reference to undefined identifier — glslangValidator should reject.
        grad_block: "float grad_q = nonexistent_function(qi);",
        grad_block_n: "float grad_qn = nonexistent_function(qi);",
        logp_block: "float lp_i = 0.0;",
        logp_final: "partial[0]"
      }

      assert {:error, reason} = Synthesis.compile(bad_spec)
      assert is_map(reason)
      assert reason.exit != 0
      assert is_binary(reason.stderr)
      assert reason.stderr != ""
    end
  end
end
