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
        push_fields: "    float p1;\n    float p2;",
        grad_block: "float grad_q = in_bounds ? -p1 * qi : 0.0;",
        grad_block_n: "float grad_qn = in_bounds ? -p1 * qi : 0.0;",
        logp_block: "float lp_i = in_bounds ? -0.5 * p1 * qi * qi : 0.0;",
        logp_final: "partial[0]"
      }

      glsl = ShaderTemplate.render(spec)

      # Headers + skeleton invariants.
      assert glsl =~ "#version 450"
      assert glsl =~ "for family: demo"
      assert glsl =~ "layout (local_size_x = 256) in;"
      assert glsl =~ "shared float partial[256];"

      # Push-constant fields landed inside the Push block.
      assert glsl =~ "float p1;"
      assert glsl =~ "float p2;"

      # All three GLSL holes were filled.
      assert glsl =~ "float grad_q = in_bounds ? -p1 * qi : 0.0;"
      assert glsl =~ "float grad_qn = in_bounds ? -p1 * qi : 0.0;"
      assert glsl =~ "float lp_i = in_bounds ? -0.5 * p1 * qi * qi : 0.0;"
      assert glsl =~ "logp_chain[k] = partial[0];"
    end

    test "logp_final supports n*logp_const idiom" do
      spec = %FamilySpec{
        name: "with_const",
        push_fields: "    float logp_const;",
        grad_block: "float grad_q = 0.0;",
        grad_block_n: "float grad_qn = 0.0;",
        logp_block: "float lp_i = 0.0;",
        logp_final: "partial[0] + float(pc.n) * pc.logp_const"
      }

      glsl = ShaderTemplate.render(spec)
      assert glsl =~ "logp_chain[k] = partial[0] + float(pc.n) * pc.logp_const;"
    end

    test "produces the same output for the catalog's Beta spec across calls" do
      a = ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.beta())
      b = ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.beta())
      assert a == b
    end

    test "different specs render different GLSL" do
      beta = ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.beta())
      gamma = ShaderTemplate.render(Nx.Vulkan.ChainShaderSpecs.gamma())

      refute beta == gamma
      assert beta =~ "for family: beta"
      assert gamma =~ "for family: gamma"
    end
  end
end
