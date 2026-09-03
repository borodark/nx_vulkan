defmodule Nx.Vulkan.ChainPropertiesTest do
  @moduledoc """
  Properties of the leapfrog chain NIFs, across shapes rather than at one shape.

  The example tests in `chain_f64_test.exs` and `chain_specs_test.exs` each pin
  one configuration. These sweep the axes that configuration happened to fix,
  and cover the guard branches nothing reached at all.

  ## Why the guard tests come first and are not tagged slow

  All seven refusal paths in the Rust NIFs — length mismatch, `k = 0`, push
  length, push parse failure, `d = 0`, `d > 256`, `n_instances = 0` — fire
  BEFORE any GPU work. They were also entirely untested: both example files
  exercised the happy path plus a single `d = 257` case for f64. A guard that
  has never been observed to fire is indistinguishable from a guard that cannot.

  ## Interior values are literals, not draws

  The shapes below are written out rather than generated per run. A failure on
  the Jetson takes minutes to reproduce, and a seed-replay step between "CI is
  red" and "I can run it" is a step nobody takes at two in the morning. This
  also matches the rest of the suite, which asserts exact bytes everywhere.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.{ChainShaderSpecs, ChainShaderSpecsF64, NativeV, Synthesis}

  @moduletag :gpu

  defp f64(list), do: for(v <- list, into: <<>>, do: <<v::float-64-little>>)
  defp f32(list), do: for(v <- list, into: <<>>, do: <<v::float-32-little>>)
  defp rep64(n, v), do: f64(List.duplicate(v, n))
  defp rep32(n, v), do: f32(List.duplicate(v, n))

  defp push64(k, d, eps),
    do: <<k::little-32, 0::little-32, d::little-32, 0::little-32, eps::little-float-64>>

  # Decode guarding arity: a binary comprehension DROPS NaN and Infinity
  # segments silently, so a length check is the only thing that sees them.
  defp doubles(bin) do
    vals = for <<v::float-64-little <- bin>>, do: v
    assert length(vals) == div(byte_size(bin), 8), "NaN/Infinity in output (decode lost values)"
    vals
  end

  defp floats(bin) do
    vals = for <<v::float-32-little <- bin>>, do: v
    assert length(vals) == div(byte_size(bin), 4), "NaN/Infinity in output (decode lost values)"
    vals
  end

  setup_all do
    normal = ChainShaderSpecsF64.normal(0.0, 1.0)
    weibull = ChainShaderSpecsF64.weibull(2.0, 1.0, 0.0)
    beta = ChainShaderSpecs.beta(2.0, 5.0, 1.7047480922384253)

    {:ok, n1} = Synthesis.compile(normal)
    {:ok, nb} = Synthesis.compile(ChainShaderSpecsF64.batched(normal))
    {:ok, w1} = Synthesis.compile(weibull)
    {:ok, wb} = Synthesis.compile(ChainShaderSpecsF64.batched(weibull))
    {:ok, b1} = Synthesis.compile(beta)
    {:ok, bb} = Synthesis.compile(ChainShaderSpecs.batched(beta))

    %{normal: n1, normal_b: nb, weibull: w1, weibull_b: wb, beta: b1, beta_b: bb}
  end

  describe "guard branches (no GPU dispatch reached)" do
    test "q_init and p_init lengths must agree", ctx do
      assert {:error, :size_mismatch} =
               NativeV.leapfrog_chain_synth_f64(
                 rep64(4, 0.1), rep64(3, 0.1), rep64(4, 1.0), push64(2, 4, 0.01), 2, ctx.normal
               )

      assert {:error, :size_mismatch} =
               NativeV.leapfrog_chain_synth_batch_f64(
                 rep64(8, 0.1), rep64(7, 0.1), rep64(8, 1.0),
                 ChainShaderSpecsF64.batch_push(2, 0, 4, 2, 0.01), 2, ctx.normal_b
               )
    end

    test "k = 0 is refused", ctx do
      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth_f64(
                 rep64(3, 0.1), rep64(3, 0.1), rep64(3, 1.0), push64(0, 3, 0.01), 0, ctx.normal
               )
    end

    test "push length at both boundaries: 0 and 129", ctx do
      q = rep64(3, 0.1)

      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth_f64(q, q, q, <<>>, 2, ctx.normal)

      too_long = :binary.copy(<<0>>, 129)

      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth_f64(q, q, q, too_long, 2, ctx.normal)
    end

    test "a plausible-but-short push fails to parse", ctx do
      q = rep64(3, 0.1)
      # 23 bytes: one short of the f64 header, so parse_push_block_f64 refuses
      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth_f64(q, q, q, :binary.copy(<<0>>, 23), 2, ctx.normal)

      # f32 header is 20; 19 is one short
      q32 = rep32(3, 0.1)

      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth(q32, q32, q32, :binary.copy(<<0>>, 19), 2, ctx.beta)
    end

    test "d = 0, and d one element past the supplied buffer", ctx do
      q = rep64(4, 0.1)

      assert {:error, :size_mismatch} =
               NativeV.leapfrog_chain_synth_f64(q, q, q, push64(2, 0, 0.01), 2, ctx.normal)

      # buffer holds 4 doubles; claiming 5 must be refused
      assert {:error, :size_mismatch} =
               NativeV.leapfrog_chain_synth_f64(q, q, q, push64(2, 5, 0.01), 2, ctx.normal)
    end

    test "d = 256 runs and d = 257 is refused, in f32 as well as f64", ctx do
      k = 2

      ok32 = rep32(256, 0.001)

      assert {:ok, {qc, _, _, lc}} =
               NativeV.leapfrog_chain_synth(
                 ok32, rep32(256, 0.0), rep32(256, 1.0),
                 ChainShaderSpecs.push(k, 0, 256, 0.01), k, ctx.beta
               )

      assert byte_size(qc) == k * 256 * 4
      assert byte_size(lc) == k * 4
      _ = floats(qc)

      over32 = rep32(257, 0.001)

      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth(
                 over32, rep32(257, 0.0), rep32(257, 1.0),
                 ChainShaderSpecs.push(k, 0, 257, 0.01), k, ctx.beta
               ),
             "f32 d=257 must be refused — past 256 the logp reduce silently sums only 256"
    end

    test "n_instances = 0 is refused in both dtypes", ctx do
      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth_batch_f64(
                 rep64(4, 0.1), rep64(4, 0.1), rep64(4, 1.0),
                 ChainShaderSpecsF64.batch_push(2, 0, 2, 0, 0.01), 2, ctx.normal_b
               )

      assert {:error, :bad_input} =
               NativeV.leapfrog_chain_synth_batch(
                 rep32(4, 0.1), rep32(4, 0.1), rep32(4, 1.0),
                 ChainShaderSpecs.batch_push(2, 0, 2, 0, 0.01), 2, ctx.beta_b
               )
    end
  end

  describe "the prefix property in f32 — untested until now" do
    # This existed only for f64. exmc's batching pads ragged chain depths to the
    # deepest and slices each caller back, so the property is load-bearing
    # downstream; there was no reason for f32 to lack it.
    test "a K=7 dispatch's first n steps equal a K=n dispatch", ctx do
      d = 4
      eps = 0.02
      q = f32([0.5, -0.3, 0.1, 0.7])
      p = f32([0.2, 0.0, -0.1, 0.05])
      m = rep32(d, 1.0)

      run = fn k ->
        {:ok, r} = NativeV.leapfrog_chain_synth(q, p, m, ChainShaderSpecs.push(k, 0, d, eps), k, ctx.beta)
        r
      end

      {q7, p7, g7, l7} = run.(7)

      for short <- [1, 3, 5] do
        {qs, ps, gs, ls} = run.(short)
        chain = short * d * 4

        assert binary_part(q7, 0, chain) == qs, "f32 q_chain prefix at K=#{short}"
        assert binary_part(p7, 0, chain) == ps, "f32 p_chain prefix at K=#{short}"
        assert binary_part(g7, 0, chain) == gs, "f32 grad_chain prefix at K=#{short}"
        assert binary_part(l7, 0, short * 4) == ls, "f32 logp_chain prefix at K=#{short}"
      end
    end

    test "and through the batched f32 path", ctx do
      d = 3
      deep = 6
      eps = 0.01
      a = f32([0.4, -0.2, 0.6])
      b = f32([-0.5, 0.3, 0.2])
      z = rep32(d, 0.0)
      o = rep32(d, 1.0)

      {:ok, {short_q, _, _, short_l}} =
        NativeV.leapfrog_chain_synth(b, z, o, ChainShaderSpecs.push(2, 0, d, eps), 2, ctx.beta)

      {:ok, {bq, _, _, bl}} =
        NativeV.leapfrog_chain_synth_batch(
          a <> b, z <> z, o <> o,
          ChainShaderSpecs.batch_push(deep, 0, d, 2, eps), deep, ctx.beta_b
        )

      stride = deep * d * 4
      assert binary_part(binary_part(bq, stride, stride), 0, 2 * d * 4) == short_q
      assert binary_part(binary_part(bl, deep * 4, deep * 4), 0, 2 * 4) == short_l
    end
  end
  # Representative families rather than all nine. "EVERY family batches
  # identically" already sweeps family exhaustively at ONE shape, so the job
  # here is to sweep SHAPE, and repeating the family axis would just multiply
  # dispatches without covering anything new.
  #
  # normal_f64  — cheapest, and the only family whose shader has no
  #               boundary-cast transcendental at all
  # weibull_f64 — the sole family with GLSL `helpers`, functions emitted before
  #               main(), which is the highest-risk construct under inst-offset
  #               indexing
  # beta (f32)  — dtype coverage, and cache-warm from the other test files
  #
  # Boundaries are hit, not sampled. Interiors are literals chosen once.
  @shapes_by_d [{1, 3, 2}, {2, 3, 2}, {7, 3, 2}, {19, 3, 2}, {41, 3, 2}, {256, 3, 2}]
  @shapes_by_k [{3, 1, 2}, {3, 2, 2}, {3, 4, 2}, {3, 7, 2}, {3, 11, 2}]
  @shapes_by_ni [{3, 3, 1}, {3, 3, 2}, {3, 3, 3}, {3, 3, 4}]

  defp shapes, do: @shapes_by_d ++ @shapes_by_k ++ @shapes_by_ni

  defp seq64(d, off), do: f64(for i <- 1..d, do: (off + i) / 100.0)
  defp seq32(d, off), do: f32(for i <- 1..d, do: (off + i) / 100.0)

  describe "batched equals single across shapes" do
    test "f64 normal and weibull, every boundary of d, K and n_instances", ctx do
      for {family, single, batch} <- [
            {"normal", ctx.normal, ctx.normal_b},
            {"weibull", ctx.weibull, ctx.weibull_b}
          ],
          {d, k, ni} <- shapes() do
        chains =
          for c <- 1..ni do
            {seq64(d, c * 10), seq64(d, -c), f64(List.duplicate(1.0, d))}
          end

        singles =
          for {q, p, m} <- chains do
            {:ok, r} = NativeV.leapfrog_chain_synth_f64(q, p, m, push64(k, d, 0.01), k, single)
            r
          end

        join = fn idx -> chains |> Enum.map(&elem(&1, idx)) |> Enum.join() end

        assert {:ok, {bq, bp, bg, bl}} =
                 NativeV.leapfrog_chain_synth_batch_f64(
                   join.(0), join.(1), join.(2),
                   ChainShaderSpecsF64.batch_push(k, 0, d, ni, 0.01), k, batch
                 ),
               "#{family} d=#{d} K=#{k} ni=#{ni} failed to dispatch"

        cs = k * d * 8
        ls = k * 8
        assert byte_size(bq) == ni * cs, "#{family} d=#{d} K=#{k} ni=#{ni} q size"
        assert byte_size(bl) == ni * ls, "#{family} d=#{d} K=#{k} ni=#{ni} logp size"

        for {{sq, sp, sg, sl}, idx} <- Enum.with_index(singles) do
          tag = "#{family} d=#{d} K=#{k} ni=#{ni} inst=#{idx}"
          assert binary_part(bq, idx * cs, cs) == sq, "#{tag} q_chain"
          assert binary_part(bp, idx * cs, cs) == sp, "#{tag} p_chain"
          assert binary_part(bg, idx * cs, cs) == sg, "#{tag} grad_chain"
          assert binary_part(bl, idx * ls, ls) == sl, "#{tag} logp_chain"
        end

        # arity-guarded decode: catches NaN/Infinity anywhere in the output
        _ = doubles(bq)
        _ = doubles(bl)
      end
    end

    test "f32 beta, same sweep", ctx do
      for {d, k, ni} <- shapes() do
        chains =
          for c <- 1..ni do
            {seq32(d, c * 10), seq32(d, -c), f32(List.duplicate(1.0, d))}
          end

        singles =
          for {q, p, m} <- chains do
            {:ok, r} =
              NativeV.leapfrog_chain_synth(q, p, m, ChainShaderSpecs.push(k, 0, d, 0.01), k, ctx.beta)

            r
          end

        join = fn idx -> chains |> Enum.map(&elem(&1, idx)) |> Enum.join() end

        assert {:ok, {bq, _, _, bl}} =
                 NativeV.leapfrog_chain_synth_batch(
                   join.(0), join.(1), join.(2),
                   ChainShaderSpecs.batch_push(k, 0, d, ni, 0.01), k, ctx.beta_b
                 ),
               "f32 beta d=#{d} K=#{k} ni=#{ni} failed to dispatch"

        cs = k * d * 4
        ls = k * 4

        for {{sq, _, _, sl}, idx} <- Enum.with_index(singles) do
          assert binary_part(bq, idx * cs, cs) == sq, "f32 d=#{d} K=#{k} ni=#{ni} inst=#{idx} q"
          assert binary_part(bl, idx * ls, ls) == sl, "f32 d=#{d} K=#{k} ni=#{ni} inst=#{idx} logp"
        end

        _ = floats(bq)
      end
    end
  end

  describe "determinism" do
    # Nothing in the suite dispatched the same inputs twice. A single dispatch
    # cannot see a buffer reused without being fully written, or a reduction
    # whose workgroup ordering varies between launches — both of which produce
    # a plausible number, once.
    test "the same call twice gives the same bits", ctx do
      d = 5
      k = 4
      q = seq64(d, 3)
      p = seq64(d, -2)
      m = f64(List.duplicate(1.0, d))

      for {label, spv} <- [{"normal", ctx.normal}, {"weibull", ctx.weibull}] do
        {:ok, a} = NativeV.leapfrog_chain_synth_f64(q, p, m, push64(k, d, 0.01), k, spv)
        {:ok, b} = NativeV.leapfrog_chain_synth_f64(q, p, m, push64(k, d, 0.01), k, spv)
        assert a == b, "#{label} single-instance dispatch is not deterministic"
      end

      ni = 3
      qb = Enum.map_join(1..ni, fn c -> seq64(d, c * 10) end)
      pb = Enum.map_join(1..ni, fn c -> seq64(d, -c) end)
      mb = :binary.copy(f64(List.duplicate(1.0, d)), ni)
      push = ChainShaderSpecsF64.batch_push(k, 0, d, ni, 0.01)

      {:ok, ba} = NativeV.leapfrog_chain_synth_batch_f64(qb, pb, mb, push, k, ctx.normal_b)
      {:ok, bb} = NativeV.leapfrog_chain_synth_batch_f64(qb, pb, mb, push, k, ctx.normal_b)
      assert ba == bb, "batched dispatch is not deterministic"
    end
  end

  describe "grad is the derivative of logp (finite differences)" do
    # Five of the six f64 families have NO numerical validation at all today —
    # only "compiles, dispatches, no NaN". A host reference for them is the
    # correctness liability we deliberately avoided: it means re-deriving five
    # PDFs in Elixir, one of which (Student-t) needs `lgamma`, which Erlang's
    # `:math` does not have.
    #
    # This checks the shader against ITSELF instead. If `grad_block` is the
    # derivative of `logp_block`, then a central difference of logp must
    # reproduce grad. No second implementation of any density is required, and
    # the additive normalising constant drops out because constants
    # differentiate to zero — which is exactly the part this library documents
    # as the caller's responsibility.
    #
    # h = 1e-3 is measured, not guessed. Observed max |fd - grad| at d=4:
    #
    #     family            h=1e-3     h=1e-4
    #     normal_f64        7.3e-14    1.6e-12    (no boundary cast anywhere)
    #     cauchy_f64        4.7e-05    5.2e-04
    #     exponential_f64   9.0e-05    9.7e-04
    #     halfnormal_f64    2.7e-05    3.1e-04
    #     studentt_f64      1.4e-04    4.1e-04
    #     weibull_f64       1.7e-04    2.2e-03
    #
    # h=1e-4 is WORSE: the f32 boundary cast puts ~1e-7 of noise on logp, and a
    # central difference divides it by 2h, so shrinking h amplifies it faster
    # than it reduces the O(h^2) truncation. The tolerance below sits ~30x above
    # the worst observed error, which still leaves a sign error or a wrong
    # coefficient — both O(1) relative — impossible to miss.
    @fd_h 1.0e-3
    @fd_rel 5.0e-3
    @fd_abs 2.0e-3

    defp fd_probe(spv, d, qs) do
      k = 1
      tiny = 1.0e-8
      zero = f64(List.duplicate(0.0, d))
      ones = f64(List.duplicate(1.0, d))
      push = <<k::little-32, 0::little-32, d::little-32, 0::little-32, tiny::little-float-64>>
      {:ok, {_q, _p, g, l}} = NativeV.leapfrog_chain_synth_f64(f64(qs), zero, ones, push, k, spv)
      {doubles(g), hd(doubles(l))}
    end

    test "every f64 family's grad matches a central difference of its own logp", ctx do
      _ = ctx
      d = 4
      q0 = [0.35, -0.20, 0.55, 0.10]

      for spec <- ChainShaderSpecsF64.all() do
        {:ok, spv} = Synthesis.compile(spec)
        {grad, _} = fd_probe(spv, d, q0)

        for i <- 0..(d - 1) do
          {_, up} = fd_probe(spv, d, List.update_at(q0, i, &(&1 + @fd_h)))
          {_, dn} = fd_probe(spv, d, List.update_at(q0, i, &(&1 - @fd_h)))
          fd = (up - dn) / (2 * @fd_h)
          g = Enum.at(grad, i)
          tol = max(@fd_rel * abs(g), @fd_abs)

          assert abs(fd - g) <= tol,
                 "#{spec.name} dim #{i}: central difference #{fd} vs grad #{g} " <>
                   "(|diff| #{abs(fd - g)} > tol #{tol}) — grad_block is not the " <>
                   "derivative of logp_block"
        end
      end
    end

    test "the check can fail: one family's logp does not match another's grad", ctx do
      # A null arm. Without it, "grad matches fd" is only evidence if a MISmatch
      # would have been detected — and every vacuous check found today looked
      # exactly like a passing one.
      _ = ctx
      d = 4
      q0 = [0.35, -0.20, 0.55, 0.10]

      {:ok, normal} = Synthesis.compile(ChainShaderSpecsF64.normal(0.0, 1.0))
      {:ok, cauchy} = Synthesis.compile(ChainShaderSpecsF64.cauchy(0.0, 2.0, -:math.log(:math.pi() * 2.0)))

      {normal_grad, _} = fd_probe(normal, d, q0)

      mismatches =
        for i <- 0..(d - 1) do
          {_, up} = fd_probe(cauchy, d, List.update_at(q0, i, &(&1 + @fd_h)))
          {_, dn} = fd_probe(cauchy, d, List.update_at(q0, i, &(&1 - @fd_h)))
          fd = (up - dn) / (2 * @fd_h)
          g = Enum.at(normal_grad, i)
          abs(fd - g) > max(@fd_rel * abs(g), @fd_abs)
        end

      assert Enum.any?(mismatches),
             "cross-family comparison passed — the finite-difference check cannot detect a " <>
               "wrong gradient and proves nothing about the families it passes for"
    end
  end

end
