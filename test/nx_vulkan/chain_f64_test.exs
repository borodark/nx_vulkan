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
  describe "the d <= 256 workgroup bound" do
    # The shaders dispatch ONE workgroup at local_size_x = 256. Past that the
    # chains get an undefined tail AND the logp tree reduce sums only the first
    # 256 elements — a silently wrong log-probability, not a truncated one.
    #
    # This went unenforced for a long time and was harmless only by accident:
    # d sat near 13 because of a push-block budget that bounded bytes the GPU
    # never receives. That accident is gone.
    test "d = 256 runs and d = 257 is refused" do
      {:ok, spv} = Synthesis.compile(ChainShaderSpecsF64.normal(0.0, 1.0))
      k = 2

      ok_d = 256
      q = f64(for i <- 1..ok_d, do: i / 1000.0)
      p = f64(for _ <- 1..ok_d, do: 0.0)
      m = f64(for _ <- 1..ok_d, do: 1.0)

      assert {:ok, {qc, _, _, lc}} =
               NativeV.leapfrog_chain_synth_f64(q, p, m, push(k, ok_d, 0.01), k, spv)

      assert byte_size(qc) == k * ok_d * 8
      assert byte_size(lc) == k * 8
      for v <- doubles(qc) ++ doubles(lc), do: assert(v == v, "NaN at d = 256")

      over = 257
      q2 = f64(for i <- 1..over, do: i / 1000.0)
      p2 = f64(for _ <- 1..over, do: 0.0)
      m2 = f64(for _ <- 1..over, do: 1.0)

      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth_f64(q2, p2, m2, push(k, over, 0.01), k, spv),
             "d = 257 must be refused, not silently miscomputed"
    end
  end

  describe "batched f64 chains" do
    # N chains in one submission. The correctness bar is not "plausible" — each
    # instance must be BIT-IDENTICAL to the same chain dispatched alone, because
    # the batched shader is the same skeleton with indices offset by `inst` and
    # any difference means the offsetting is wrong.
    defp chain_inputs(d, c) do
      {f64(for i <- 1..d, do: (c * 10 + i) / 100.0), f64(for i <- 1..d, do: (c - i) / 50.0),
       f64(for _ <- 1..d, do: 1.0)}
    end

    test "EVERY family batches identically to its single dispatch" do
      # Only Normal was covered before. Weibull is the one that matters most:
      # it is the sole family with `helpers` — GLSL functions emitted before
      # main() — and nothing had checked that they behave under `inst`-offset
      # indexing. They take q_uc as a parameter so they should be index-blind,
      # but "should be" is what this test replaces.
      d = 3
      k = 5
      ni = 3
      eps = 0.01

      for spec <- ChainShaderSpecsF64.all() do
        {:ok, single_spv} = Synthesis.compile(spec)
        {:ok, batch_spv} = Synthesis.compile(ChainShaderSpecsF64.batched(spec))

        chains = for c <- 1..ni, do: chain_inputs(d, c)

        singles =
          for {q, p, m} <- chains do
            {:ok, r} = NativeV.leapfrog_chain_synth_f64(q, p, m, push(k, d, eps), k, single_spv)
            r
          end

        join = fn idx -> chains |> Enum.map(&elem(&1, idx)) |> Enum.join() end

        assert {:ok, {bq, bp, bg, bl}} =
                 NativeV.leapfrog_chain_synth_batch_f64(
                   join.(0),
                   join.(1),
                   join.(2),
                   ChainShaderSpecsF64.batch_push(k, 0, d, ni, eps),
                   k,
                   batch_spv
                 ),
               "#{spec.name} failed to dispatch batched"

        cs = k * d * 8
        ls = k * 8

        for {{sq, sp, sg, sl}, i} <- Enum.with_index(singles) do
          assert binary_part(bq, i * cs, cs) == sq, "#{spec.name} q_chain, instance #{i}"
          assert binary_part(bp, i * cs, cs) == sp, "#{spec.name} p_chain, instance #{i}"
          assert binary_part(bg, i * cs, cs) == sg, "#{spec.name} grad_chain, instance #{i}"
          assert binary_part(bl, i * ls, ls) == sl, "#{spec.name} logp_chain, instance #{i}"
        end

        # and nothing is silently NaN, which bit-identity alone would not catch
        # if BOTH paths produced NaN
        for v <- doubles(bq) ++ doubles(bl) do
          assert v == v, "#{spec.name} produced NaN under batching"
        end
      end
    end

    test "each instance is bit-identical to its own single dispatch" do
      d = 4
      k = 6
      ni = 4
      eps = 0.02
      spec = ChainShaderSpecsF64.normal(0.0, 1.0)
      {:ok, single_spv} = Synthesis.compile(spec)
      {:ok, batch_spv} = Synthesis.compile(ChainShaderSpecsF64.batched(spec))

      chains = for c <- 1..ni, do: chain_inputs(d, c)

      singles =
        for {q, p, m} <- chains do
          {:ok, r} = NativeV.leapfrog_chain_synth_f64(q, p, m, push(k, d, eps), k, single_spv)
          r
        end

      join = fn idx -> chains |> Enum.map(&elem(&1, idx)) |> Enum.join() end

      assert {:ok, {bq, bp, bg, bl}} =
               NativeV.leapfrog_chain_synth_batch_f64(
                 join.(0),
                 join.(1),
                 join.(2),
                 ChainShaderSpecsF64.batch_push(k, 0, d, ni, eps),
                 k,
                 batch_spv
               )

      assert byte_size(bq) == ni * k * d * 8
      assert byte_size(bl) == ni * k * 8

      chain_stride = k * d * 8
      logp_stride = k * 8

      for {{sq, sp, sg, sl}, idx} <- Enum.with_index(singles) do
        assert binary_part(bq, idx * chain_stride, chain_stride) == sq, "q_chain, instance #{idx}"
        assert binary_part(bp, idx * chain_stride, chain_stride) == sp, "p_chain, instance #{idx}"
        assert binary_part(bg, idx * chain_stride, chain_stride) == sg, "grad, instance #{idx}"
        assert binary_part(bl, idx * logp_stride, logp_stride) == sl, "logp, instance #{idx}"
      end
    end

    test "instances do not bleed into each other" do
      # Instance 0 gets inputs that would produce a very different trajectory
      # from instance 1. If the `inst` offset were dropped anywhere, they would
      # come back equal — which a size check cannot see.
      d = 3
      k = 4
      spec = ChainShaderSpecsF64.batched(ChainShaderSpecsF64.normal(0.0, 1.0))
      {:ok, spv} = Synthesis.compile(spec)

      a = f64([2.0, 2.0, 2.0])
      b = f64([-5.0, -5.0, -5.0])
      zero = f64([0.0, 0.0, 0.0])
      ones = f64([1.0, 1.0, 1.0])

      assert {:ok, {q_chain, _, _, logp}} =
               NativeV.leapfrog_chain_synth_batch_f64(
                 a <> b,
                 zero <> zero,
                 ones <> ones,
                 ChainShaderSpecsF64.batch_push(k, 0, d, 2, 0.05),
                 k,
                 spv
               )

      stride = k * d * 8
      refute binary_part(q_chain, 0, stride) == binary_part(q_chain, stride, stride)
      refute binary_part(logp, 0, k * 8) == binary_part(logp, k * 8, k * 8)
    end

    test "n_instances = 1 matches the single-instance path exactly" do
      d = 5
      k = 3
      spec = ChainShaderSpecsF64.normal(0.0, 1.0)
      {:ok, single_spv} = Synthesis.compile(spec)
      {:ok, batch_spv} = Synthesis.compile(ChainShaderSpecsF64.batched(spec))
      {q, p, m} = chain_inputs(d, 1)

      {:ok, {sq, _, _, sl}} =
        NativeV.leapfrog_chain_synth_f64(q, p, m, push(k, d, 0.01), k, single_spv)

      {:ok, {bq, _, _, bl}} =
        NativeV.leapfrog_chain_synth_batch_f64(
          q,
          p,
          m,
          ChainShaderSpecsF64.batch_push(k, 0, d, 1, 0.01),
          k,
          batch_spv
        )

      assert bq == sq
      assert bl == sl
    end
  end

  describe "the prefix property that padding depends on" do
    # Batching chains of unequal depth means dispatching all of them at the
    # DEEPEST K and handing each caller back only the steps it asked for. That
    # is only sound if a longer dispatch's prefix is exactly what a shorter
    # dispatch would have produced.
    #
    # It is not obviously true — a shader could accumulate, reduce differently
    # at the tail, or carry state across steps in a way that made step 3 of a
    # K=7 run differ from step 3 of a K=3 run. Here each step writes
    # q_chain[k*d+i] from state that only depends on earlier steps, so it holds
    # — but "holds by inspection" is what this test replaces, and the failure it
    # guards against is a plausible wrong posterior rather than an error.
    #
    # exmc's BatchCoordinator flush relies on this to slice padded instances
    # back to their requested depth.
    test "a K=7 dispatch's first n steps equal a K=n dispatch, bit for bit" do
      d = 4
      eps = 0.02
      {:ok, spv} = Synthesis.compile(ChainShaderSpecsF64.normal(0.0, 1.0))

      q = f64([0.5, -0.3, 0.1, 0.7])
      p = f64([0.2, 0.0, -0.1, 0.05])
      m = f64([1.0, 1.0, 1.0, 1.0])

      run = fn k ->
        {:ok, r} = NativeV.leapfrog_chain_synth_f64(q, p, m, push(k, d, eps), k, spv)
        r
      end

      {q7, p7, g7, l7} = run.(7)

      for short <- [1, 3, 5] do
        {qs, ps, gs, ls} = run.(short)
        chain = short * d * 8
        logp = short * 8

        assert binary_part(q7, 0, chain) == qs, "q_chain prefix at K=#{short}"
        assert binary_part(p7, 0, chain) == ps, "p_chain prefix at K=#{short}"
        assert binary_part(g7, 0, chain) == gs, "grad_chain prefix at K=#{short}"
        assert binary_part(l7, 0, logp) == ls, "logp_chain prefix at K=#{short}"
      end
    end

    test "the same holds through the batched path" do
      # Two instances, both dispatched at the padded depth, each sliced back.
      d = 3
      k_deep = 6
      eps = 0.01
      spec = ChainShaderSpecsF64.normal(0.0, 1.0)
      {:ok, single} = Synthesis.compile(spec)
      {:ok, batch} = Synthesis.compile(ChainShaderSpecsF64.batched(spec))

      {qa, pa, ma} = {f64([0.4, -0.2, 0.6]), f64([0.1, 0.0, -0.1]), f64([1.0, 1.0, 1.0])}
      {qb, pb, mb} = {f64([-0.5, 0.3, 0.2]), f64([0.0, 0.2, 0.1]), f64([1.0, 1.0, 1.0])}

      # what B would have got had it asked for 2 steps on its own
      {:ok, {b_short, _, _, l_short}} =
        NativeV.leapfrog_chain_synth_f64(qb, pb, mb, push(2, d, eps), 2, single)

      {:ok, {bq, _, _, bl}} =
        NativeV.leapfrog_chain_synth_batch_f64(
          qa <> qb,
          pa <> pb,
          ma <> mb,
          ChainShaderSpecsF64.batch_push(k_deep, 0, d, 2, eps),
          k_deep,
          batch
        )

      # instance 1 is B; slice its first 2 steps out of the padded 6
      inst_stride = k_deep * d * 8
      b_padded = binary_part(bq, inst_stride, inst_stride)
      assert binary_part(b_padded, 0, 2 * d * 8) == b_short

      logp_stride = k_deep * 8
      b_logp = binary_part(bl, logp_stride, logp_stride)
      assert binary_part(b_logp, 0, 2 * 8) == l_short
    end
  end

end
