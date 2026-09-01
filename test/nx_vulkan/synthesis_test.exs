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
      spec = Nx.Vulkan.ChainShaderSpecs.beta(2.0, 5.0, 1.7)
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
            fn -> Nx.Vulkan.ChainShaderSpecs.gamma(3.0, 2.0, 0.9) end,
            fn -> Nx.Vulkan.ChainShaderSpecs.lognormal(0.0, 1.0) end
          ] do
        {:ok, path} = Synthesis.compile(spec_fn.())
        assert File.exists?(path)
      end
    end
  end

  describe "compile/1 cache hit" do
    test "second call returns the same path and is fast" do
      spec = Nx.Vulkan.ChainShaderSpecs.beta(2.0, 5.0, 1.7)

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
      {:ok, beta_path} = Synthesis.compile(Nx.Vulkan.ChainShaderSpecs.beta(2.0, 5.0, 1.7))
      {:ok, gamma_path} = Synthesis.compile(Nx.Vulkan.ChainShaderSpecs.gamma(3.0, 2.0, 0.9))

      assert beta_path != gamma_path
      assert File.exists?(beta_path)
      assert File.exists?(gamma_path)
    end
  end

  describe "compile_with_source/1" do
    test "returns both the SPV path and the rendered GLSL" do
      spec = Nx.Vulkan.ChainShaderSpecs.beta(2.0, 5.0, 1.7)
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
        params: %{"alpha" => 1.0},
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
  describe "end-to-end dispatch" do
    @moduletag :gpu

    # The first caller this path has ever had inside nx_vulkan. Until the push
    # block was reduced to the NIF's fixed header and family params were baked
    # as literals, a templated shader could not be dispatched at all: the NIF
    # sends sizeof(PushBlock) = 20 bytes, so the family fields the template used
    # to declare were never written, and the header disagreed besides — the
    # template's `n` sat where the NIF reads `k_steps`.
    #
    # Nothing called these NIFs, so nothing contradicted it. That is why
    # 8cce91c's batched readback had to be measured by a downstream consumer:
    # a repo cannot benchmark a path it cannot drive.
    test "a baked Beta spec dispatches and returns correctly-sized chains" do
      spec = Nx.Vulkan.ChainShaderSpecs.beta(2.0, 5.0, 1.7047480922384253)
      {:ok, spv} = Synthesis.compile(spec)

      d = 3
      k = 8
      f32 = fn l -> for v <- l, into: <<>>, do: <<v::float-32-little>> end

      q = f32.([0.1, -0.2, 0.3])
      p = f32.([0.0, 0.05, -0.05])
      inv_mass = f32.([1.0, 1.0, 1.0])
      push = Nx.Vulkan.ChainShaderSpecs.push(k, 0, d, 0.01)

      assert byte_size(push) == 20

      assert {:ok, {q_chain, p_chain, grad_chain, logp_chain}} =
               Nx.Vulkan.NativeV.leapfrog_chain_synth(q, p, inv_mass, push, k, spv)

      assert byte_size(q_chain) == k * d * 4
      assert byte_size(p_chain) == k * d * 4
      assert byte_size(grad_chain) == k * d * 4

      # logp is one scalar PER STEP, not per dimension. Invisible at d == 1,
      # which is why d == 3 here.
      assert byte_size(logp_chain) == k * 4

      # Values are finite, and a small eps barely moves q on the first step.
      <<q0::float-32-little, _::binary>> = q_chain
      <<lp0::float-32-little, _::binary>> = logp_chain
      assert q0 == q0, "q_chain[0] is NaN"
      assert lp0 == lp0, "logp_chain[0] is NaN"
      assert_in_delta q0, 0.1, 0.01
    end

    test "the rendered source carries no un-baked family parameter" do
      glsl = Nx.Vulkan.ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.gamma(3.0, 2.0, 0.9))
      refs = Regex.scan(~r/pc\.\w+/, glsl) |> List.flatten() |> Enum.uniq() |> Enum.sort()

      # Only the NIF's header fields may survive. Anything else would be read
      # from push constants that are never written.
      assert refs == ["pc.K", "pc.d", "pc.eps"]
    end
  end

end
