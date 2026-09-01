defmodule Nx.Vulkan.ShaderTemplateTest do
  @moduledoc """
  Phase 1 — `Nx.Vulkan.ShaderTemplate.render/1` produces valid GLSL
  for chain shader family specs. Compilation and dispatch are
  covered by SynthesisTest; this file only validates the textual
  rendering.
  """

  use ExUnit.Case, async: true

  alias Nx.Vulkan.ShaderTemplate
  alias Nx.Vulkan.ShaderTemplate.FamilySpec

  describe "render/1" do
    test "renders the canonical skeleton with all hole substitutions" do
      spec = %FamilySpec{
        name: "demo",
        params: %{"p1" => 1.5, "p2" => 2.25},
        grad_block: "float grad_q = in_bounds ? -pc.p1 * qi : 0.0;",
        grad_block_n: "float grad_qn = in_bounds ? -pc.p1 * qi : 0.0;",
        logp_block: "float lp_i = in_bounds ? -0.5 * pc.p1 * qi * qi : 0.0;",
        logp_final: "partial[0]"
      }

      glsl = ShaderTemplate.render(spec)

      # Headers + skeleton invariants.
      assert glsl =~ "#version 450"
      assert glsl =~ "for family: demo"
      assert glsl =~ "layout (local_size_x = 256) in;"
      assert glsl =~ "shared float partial[256];"

      # The push block is EXACTLY the NIF's fixed header — no family fields.
      # Anything declared past `eps` would be dropped by push_constants, which
      # sends sizeof(PushBlock) = 20 bytes and no more.
      assert glsl =~ "uint  K;"
      assert glsl =~ "uint  n_obs;"
      assert glsl =~ "uint  d;"
      assert glsl =~ "float eps;"
      refute glsl =~ "float p1;"
      refute glsl =~ "float p2;"

      # Family params are baked as literals, so no pc.<param> survives.
      refute glsl =~ "pc.p1"
      refute glsl =~ "pc.p2"

      # All three GLSL holes were filled.
      assert glsl =~ "float grad_q = in_bounds ? -1.5 * qi : 0.0;"
      assert glsl =~ "float grad_qn = in_bounds ? -1.5 * qi : 0.0;"
      assert glsl =~ "float lp_i = in_bounds ? -0.5 * 1.5 * qi * qi : 0.0;"
      assert glsl =~ "logp_chain[k] = partial[0];"
    end

    test "logp_final supports n*logp_const idiom" do
      spec = %FamilySpec{
        name: "with_const",
        params: %{"logp_const" => 0.75},
        grad_block: "float grad_q = 0.0;",
        grad_block_n: "float grad_qn = 0.0;",
        logp_block: "float lp_i = 0.0;",
        logp_final: "partial[0] + float(pc.d) * pc.logp_const"
      }

      glsl = ShaderTemplate.render(spec)
      # pc.d survives (header field); pc.logp_const is baked to its literal.
      assert glsl =~ "logp_chain[k] = partial[0] + float(pc.d) * 0.75;"
    end

    test "produces the same output for the catalog's Beta spec across calls" do
      a = ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.beta(2.0, 5.0, 1.7))
      b = ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.beta(2.0, 5.0, 1.7))
      assert a == b
    end

    test "different specs render different GLSL" do
      beta = ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.beta(2.0, 5.0, 1.7))
      gamma = ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.gamma(3.0, 2.0, 0.9))

      refute beta == gamma
      assert beta =~ "for family: beta"
      assert gamma =~ "for family: gamma"
    end
  end
end
