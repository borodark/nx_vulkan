defmodule Nx.Vulkan.ChainF64Test do
  @moduledoc """
  The six f64 chain families, ported from the hand-written
  `glsl/leapfrog_chain_*_f64.comp` shaders onto the templated path.

  Those shaders were undriveable by this repo's own NIFs — 32-byte push blocks
  with family params inline against a NIF that pushes a fixed 24-byte header —
  and nothing called the chain NIFs, so nothing ever said so. These tests are
  the contradiction that was missing.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.{ChainShaderSpecsF64, NativeV, ShaderTemplate, Synthesis}

  @moduletag :gpu

  defp f64(list), do: for(v <- list, into: <<>>, do: <<v::float-64-little>>)

  defp push(k, d, eps),
    do: <<k::little-32, 0::little-32, d::little-32, 0::little-32, eps::little-float-64>>

  defp doubles(bin), do: for(<<v::float-64-little <- bin>>, do: v)

  describe "every family" do
    test "renders with no un-baked parameter and compiles to SPIR-V" do
      for spec <- ChainShaderSpecsF64.all() do
        glsl = ShaderTemplate.render(spec)

        # Only the NIF's header fields may survive. Anything else would be read
        # from push constants that are never written.
        left = Regex.scan(~r/pc\.\w+/, glsl) |> List.flatten() |> Enum.uniq() |> Enum.sort()
        assert left == ["pc.K", "pc.d", "pc.eps"], "#{spec.name} left #{inspect(left)}"

        assert glsl =~ "GL_ARB_gpu_shader_fp64"
        assert glsl =~ "double qi"

        assert {:ok, path} = Synthesis.compile(spec), "#{spec.name} failed to compile"
        assert File.stat!(path).size > 0
      end
    end

    test "dispatches and returns correctly-sized chains" do
      d = 3
      k = 5

      for spec <- ChainShaderSpecsF64.all() do
        {:ok, spv} = Synthesis.compile(spec)

        assert {:ok, {q_chain, p_chain, grad_chain, logp_chain}} =
                 NativeV.leapfrog_chain_synth_f64(
                   f64([0.2, -0.1, 0.4]),
                   f64([0.0, 0.1, -0.1]),
                   f64([1.0, 1.0, 1.0]),
                   push(k, d, 0.01),
                   k,
                   spv
                 ),
               "#{spec.name} failed to dispatch"

        assert byte_size(q_chain) == k * d * 8, spec.name
        assert byte_size(p_chain) == k * d * 8, spec.name
        assert byte_size(grad_chain) == k * d * 8, spec.name

        # One scalar per STEP, not per dimension. Invisible at d == 1.
        assert byte_size(logp_chain) == k * 8, spec.name

        for v <- doubles(q_chain) ++ doubles(logp_chain) do
          assert v == v, "#{spec.name} produced NaN"
        end
      end
    end
  end

  describe "Normal, against a host reference" do
    # The repo's bar is bit-for-bit agreement with a host computation, not a
    # tolerance. Baking makes that reachable: the old shader derived log(sigma)
    # on the GPU via double(log(float(...))), an f32 round trip. Here it is a
    # full f64 :math.log/1 on the host, so both sides run the same arithmetic.
    test "reproduces a host leapfrog exactly" do
      mu = 0.0
      sigma = 1.0
      d = 3
      k = 6
      eps = 0.05
      q0 = [0.5, -0.3, 0.1]
      p0 = [0.2, 0.0, -0.1]
      mass = [1.0, 1.0, 1.0]

      {:ok, spv} = Synthesis.compile(ChainShaderSpecsF64.normal(mu, sigma))

      {:ok, {q_chain, _p, _g, logp_chain}} =
        NativeV.leapfrog_chain_synth_f64(
          f64(q0),
          f64(p0),
          f64(mass),
          push(k, d, eps),
          k,
          spv
        )

      inv_var = 1.0 / (sigma * sigma)
      const = -(:math.log(sigma) + 0.9189385332046727)

      {host_q, host_lp} =
        Enum.reduce(1..k, {{q0, p0}, []}, fn _, {{q, p}, acc} ->
          g = Enum.map(q, fn qi -> -(qi - mu) * inv_var end)
          p_half = Enum.zip_with(p, g, fn pi, gi -> pi + 0.5 * eps * gi end)
          step = Enum.zip_with(p_half, mass, &(&1 * &2))
          q2 = Enum.zip_with(q, step, fn qi, s -> qi + eps * s end)
          g2 = Enum.map(q2, fn qi -> -(qi - mu) * inv_var end)
          p2 = Enum.zip_with(p_half, g2, fn pi, gi -> pi + 0.5 * eps * gi end)

          lp =
            Enum.reduce(q2, 0.0, fn qi, a ->
              z = (qi - mu) / sigma
              a + -0.5 * z * z
            end) + d * const

          {{q2, p2}, acc ++ [{q2, lp}]}
        end)
        |> then(fn {_, acc} ->
          {acc |> Enum.map(&elem(&1, 0)) |> List.flatten(), Enum.map(acc, &elem(&1, 1))}
        end)

      assert doubles(q_chain) == host_q
      assert doubles(logp_chain) == host_lp
    end
  end
end
