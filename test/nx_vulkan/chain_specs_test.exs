defmodule Nx.Vulkan.ChainSpecsTest do
  @moduledoc """
  `Nx.Vulkan.ChainShaderSpecs` push-constant packing, and the
  `Nx.Vulkan.PipelineCache` stub.

  The push builders were at 0% coverage, and they are the wrong thing to leave
  untested: each packs a C struct that a shader reads by OFFSET. A field in the
  wrong order or the wrong width is not a crash — it is a shader reading `alpha`
  out of `eps`'s bytes and producing plausible numbers. Nothing downstream
  would look wrong.

  So these assert the LAYOUT, byte by byte, against the documented struct —
  not merely that the function returns a binary.
  """
  use ExUnit.Case, async: true

  alias Nx.Vulkan.ChainShaderSpecs, as: Specs

  # The documented C struct, shared by all three families:
  #   uint n; uint K; float eps; float a; float b; float logp_const;

  defp unpack(
         <<n::little-32, k::little-32, eps::little-float-32, a::little-float-32,
           b::little-float-32, c::little-float-32>>
       ) do
    {n, k, eps, a, b, c}
  end

  describe "push/4" do
    test "packs the NIF's fixed 20-byte header" do
      bin = Specs.push(32, 7, 3, 0.05)
      assert byte_size(bin) == 20

      <<k::little-32, n_obs::little-32, d::little-32, pad::little-32,
        eps::little-float-32>> = bin

      assert k == 32
      assert n_obs == 7
      assert d == 3
      assert pad == 0
      assert_in_delta eps, 0.05, 1.0e-7
    end

    test "carries no family parameters" do
      # They are baked into the shader source. The NIF pushes
      # sizeof(PushBlock) = 20 bytes, so anything appended here would be
      # dropped before it reached the GPU — which is what the old
      # beta_push/6, gamma_push/6 and lognormal_push/5 were doing.
      assert byte_size(Specs.push(1, 1, 1, 0.1)) == 20
    end

    test "field order matches the NIF, not the old shader convention" do
      # The old builders put the DIMENSION first, where the NIF reads k_steps.
      <<first::little-32, _::binary>> = Specs.push(99, 0, 3, 0.1)
      assert first == 99, "offset 0 is k_steps, not d"
    end
  end

  describe "the three family specs render" do
    test "beta, gamma and lognormal each produce distinct GLSL" do
      alias Nx.Vulkan.ShaderTemplate

      rendered =
        for spec <- [
            Specs.beta(2.0, 5.0, 1.7),
            Specs.gamma(3.0, 2.0, 0.9),
            Specs.lognormal(0.0, 1.0)
          ],
          do: ShaderTemplate.render(spec)

      assert length(Enum.uniq(rendered)) == 3, "two families rendered identical source"
      for src <- rendered, do: assert(String.contains?(src, "#version"))
    end
  end

  describe "ShaderTemplate's auto-derived grad block" do
    # `render/1` derives the second grad block by renaming locals when a spec
    # leaves `grad_block_n: nil`. ALL THREE shipped specs supply their own, so
    # this path never runs in production — which is why ShaderTemplate sat at
    # 75%. It is still documented behaviour ("we do this automatically if you
    # set `grad_block_n: nil`") and its renaming is a pair of regexes with
    # lookarounds, which is exactly the kind of code that is quietly wrong.
    alias Nx.Vulkan.ShaderTemplate
    alias Nx.Vulkan.ShaderTemplate.FamilySpec

    defp minimal_spec(grad_block_n) do
      %FamilySpec{
        name: "probe",
        params: %{"alpha" => 1.0},
        grad_block: "float q = exp(q_uc);\n grad_q = pc.alpha - q;",
        grad_block_n: grad_block_n,
        logp_block: "float q = exp(q_uc);",
        logp_final: "logp"
      }
    end

    test "renames the intermediate local so it cannot shadow the outer one" do
      src = ShaderTemplate.render(minimal_spec(nil))

      # `float q` becomes `float qn`, and bare `q` references follow it —
      # otherwise the derived block would read the OUTER q and silently compute
      # the gradient at the wrong point.
      assert src =~ "float qn"
      assert src =~ "grad_qn"

      # The original block must survive unrenamed alongside it.
      assert src =~ "float q ="
      assert src =~ "grad_q ="
    end

    test "an explicit grad_block_n is used verbatim, not derived" do
      explicit = "float custom = 1.0;\n grad_qn = custom;"
      src = ShaderTemplate.render(minimal_spec(explicit))

      assert src =~ "float custom"
      refute src =~ "float qn", "an explicit grad_block_n must not be rewritten"
    end

    test "the derived and explicit paths differ, so the rename is doing work" do
      derived = ShaderTemplate.render(minimal_spec(nil))
      explicit = ShaderTemplate.render(minimal_spec("grad_qn = 0.0;"))
      refute derived == explicit
    end
  end

  describe "Nx.Vulkan.Device — the NXV_F64 override, both directions" do
    # `NXV_F64=0` is exercised by compiler_test (it is how the f64 host-fallback
    # path gets tested on a machine whose GPU supports f64). `=1` was not, and
    # it is the more dangerous direction: it forces f64 GPU paths ON, which on a
    # device without shaderFloat64 fails at pipeline creation rather than
    # falling back.
    alias Nx.Vulkan.Device

    test "NXV_F64 wins over the device, in both directions" do
      previous = System.get_env("NXV_F64")

      try do
        System.put_env("NXV_F64", "0")
        refute Device.f64?()

        System.put_env("NXV_F64", "1")
        assert Device.f64?()
      after
        if previous, do: System.put_env("NXV_F64", previous), else: System.delete_env("NXV_F64")
      end
    end
  end

  describe "PipelineCache is a STUB — pinned so nobody trusts it" do
    # Its moduledoc says so: the C++ backend that owned the VkPipelineCache was
    # removed and vulkano manages its own caching. The functions are retained
    # for API compatibility and do nothing.
    #
    # `device_uuid_hex/0` is the one worth pinning: it is named like a real
    # device identifier and returns sixteen zero bytes on every machine. Code
    # keying a cache on it would collide across every GPU in the fleet.
    alias Nx.Vulkan.PipelineCache

    test "load/persist are no-ops returning :ok" do
      assert PipelineCache.load() == :ok
      assert PipelineCache.persist() == :ok
      assert PipelineCache.clear() == :ok
    end

    test "device_uuid_hex/0 is a CONSTANT, not a device identity" do
      assert PipelineCache.device_uuid_hex() == String.duplicate("0", 32)
    end

    test "default_path/0 names a file under the cache dir" do
      assert String.ends_with?(PipelineCache.default_path(), "/pipeline_cache/vulkano.bin")
    end
  end
end
