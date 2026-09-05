defmodule Nx.Vulkan.FallbackTest do
  @moduledoc """
  Asserts *where the work ran*, which no assertion on values can do.

  A host fallback returns a bit-identical result to the GPU path — it computes
  on `Nx.BinaryBackend`, which is the reference the rest of the suite compares
  against. So a correctness test cannot tell "ran on the GPU" from "silently
  didn't," and the conv backward pass ran on the CPU for the entire life of the
  conv shaders without a single test noticing.

  These tests close that hole from the other side: they count fallbacks and
  assert the count.

  Two kinds of test live here, and the second kind is meant to fail eventually:

    * **native** — ops that must stay on the GPU. A regression that quietly
      reroutes one to the host fails here instead of just getting slower.
    * **known fallbacks** — ops that legitimately fall back today, pinned with
      the exact count. When one moves on-device its test fails, which is the
      reminder to promote it. Do not relax these to `>= 0`; the number is the
      point.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.Fallback
  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp t(shape, seed) do
    size = Tuple.product(shape)
    data = for i <- 1..size, do: :math.sin(seed * 0.7 + i * 0.41) + 1.5
    Nx.tensor(data, type: {:f, 64}, backend: VulkanoBackend) |> Nx.reshape(shape)
  end

  describe "the counter itself" do
    test "counts nothing when everything is native" do
      a = t({4, 4}, 1)
      b = t({4, 4}, 2)
      assert Fallback.count_total(fn -> Nx.add(a, b) end) == 0
    end

    test "counts a known fallback and attributes it to the callback" do
      # sort has no shader and is not planned to get one — a stable specimen.
      # This test previously used reverse/3, which then got a shader; that is
      # the intended lifecycle, but a meta-test should not ride on it.
      a = t({4, 4}, 1)
      {_result, counts} = Fallback.count(fn -> Nx.sort(a, axis: 1) end)

      assert counts == %{{:sort, 3} => 1}
    end

    test "is off by default — note/1 outside count/1 is a no-op" do
      refute Fallback.recording?()
      # :allow explicitly: note/1 with a synthetic op is the point of the test,
      # and under a strict default it would (correctly) be refused.
      assert Fallback.strict(:allow, fn -> Fallback.note({:whatever, 1}) end) == :ok
      refute Fallback.recording?()
    end

    test "nests without losing the outer tally" do
      a = t({4, 4}, 1)

      {_r, outer} =
        Fallback.count(fn ->
          _ = Nx.sort(a, axis: 1)
          {_r2, inner} = Fallback.count(fn -> Nx.sort(a, axis: 1) end)
          assert inner == %{{:sort, 3} => 1}
          Nx.sort(a, axis: 1)
        end)

      assert outer == %{{:sort, 3} => 2}
    end
  end

  describe "native — these must not leave the GPU" do
    test "elementwise binary and unary" do
      a = t({8, 8}, 1)
      b = t({8, 8}, 2)

      assert Fallback.count_total(fn ->
               a |> Nx.add(b) |> Nx.multiply(b) |> Nx.sqrt() |> Nx.negate()
             end) == 0
    end

    test "broadcasting bias-add" do
      x = t({4, 6}, 1)
      bias = t({6}, 2)
      assert Fallback.count_total(fn -> Nx.add(x, bias) end) == 0
    end

    test "matmul and reductions" do
      a = t({8, 6}, 1)
      b = t({6, 5}, 2)

      assert Fallback.count_total(fn -> Nx.dot(a, b) end) == 0
      assert Fallback.count_total(fn -> Nx.sum(a) end) == 0
      assert Fallback.count_total(fn -> Nx.reduce_max(a, axes: [1]) end) == 0
    end

    test "transpose — rank 2 and the general permuted path" do
      assert Fallback.count_total(fn -> Nx.transpose(t({6, 4}, 1)) end) == 0
      assert Fallback.count_total(fn -> Nx.transpose(t({2, 3, 4}, 1), axes: [2, 0, 1]) end) == 0

      assert Fallback.count_total(fn ->
               Nx.transpose(t({2, 3, 4, 5}, 1), axes: [1, 0, 2, 3])
             end) == 0
    end

    test "an integer scalar operand does not drag its tensor to the host" do
      # Nx materialises literals as rank-0 {:s, 32}: relu is max(x, 0), a mean
      # divides by an integer count. The cast shader has no integer path, so
      # these used to host-fall-back — a four-byte literal moving a whole
      # tensor to the CPU. Worth five fallbacks per training step.
      x = t({8, 4}, 1)
      zero = Nx.tensor(0, backend: VulkanoBackend)
      two = Nx.tensor(2, backend: VulkanoBackend)

      assert Fallback.count_total(fn -> Nx.max(x, zero) end) == 0
      assert Fallback.count_total(fn -> Nx.divide(x, two) end) == 0
      assert Fallback.count_total(fn -> Nx.greater(x, zero) end) == 0
    end

    test "integer-scalar results still match BinaryBackend" do
      x = t({6, 3}, 2)
      xh = Nx.backend_copy(x, Nx.BinaryBackend)
      zero = Nx.tensor(0, backend: VulkanoBackend)
      zh = Nx.tensor(0, backend: Nx.BinaryBackend)

      for {got, want} <- [
            {Nx.max(x, zero), Nx.max(xh, zh)},
            {Nx.divide(x, Nx.tensor(2, backend: VulkanoBackend)),
             Nx.divide(xh, Nx.tensor(2, backend: Nx.BinaryBackend))},
            {Nx.greater(x, zero), Nx.greater(xh, zh)}
          ] do
        d =
          Nx.subtract(
            Nx.backend_copy(got, Nx.BinaryBackend) |> Nx.as_type({:f, 64}),
            Nx.as_type(want, {:f, 64})
          )
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert d < 1.0e-12
      end
    end

    test "reverse — the conv input-gradient's kernel reversal" do
      # Was a pure host fallback with no shader at all. Worse than its single
      # count suggested: it stranded its output on BinaryBackend, so Nx ran
      # everything downstream there too, invisible to this counter.
      assert Fallback.count_total(fn -> Nx.reverse(t({4, 5}, 1), axes: [0]) end) == 0
      assert Fallback.count_total(fn -> Nx.reverse(t({2, 3, 4}, 1), axes: [0, 2]) end) == 0
      assert Fallback.count_total(fn -> Nx.reverse(t({4, 3, 3, 3}, 1), axes: [2, 3]) end) == 0
    end

    test "reverse matches BinaryBackend exactly" do
      for {shape, axes} <- [{{4, 5}, [1]}, {{2, 3, 4}, [0, 1, 2]}, {{2, 3, 4, 5}, [1, 2]}] do
        v = t(shape, 2)
        h = Nx.backend_copy(v, Nx.BinaryBackend)

        d =
          Nx.subtract(
            Nx.reverse(v, axes: axes) |> Nx.backend_copy(Nx.BinaryBackend),
            Nx.reverse(h, axes: axes)
          )
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert d == 0.0, "reverse #{inspect(shape)} axes #{inspect(axes)} diverged by #{d}"
      end
    end

    test "broadcast — including the relu-gradient zero fill" do
      # Had no shader at all and always went to the host. Its own cost was the
      # smaller half: it stranded the result on BinaryBackend, which is what
      # made select/4 fall back on the s32 zeros it produced.
      assert Fallback.count_total(fn -> Nx.broadcast(t({3}, 1), {2, 3}, axes: [1]) end) == 0
      assert Fallback.count_total(fn -> Nx.broadcast(t({3, 1}, 1), {3, 4}, axes: [0, 1]) end) == 0
      assert Fallback.count_total(fn -> Nx.broadcast(t({4}, 1), {2, 3, 4}, axes: [2]) end) == 0

      scalar = Nx.tensor(1.5, type: {:f, 64}, backend: VulkanoBackend)
      assert Fallback.count_total(fn -> Nx.broadcast(scalar, {8, 8}, axes: []) end) == 0
    end

    test "broadcast matches BinaryBackend exactly" do
      for {ishape, oshape, axes} <- [
            {{3}, {2, 3}, [1]},
            {{2, 1, 4}, {2, 3, 4}, [0, 1, 2]},
            {{2, 1, 1, 5}, {2, 3, 4, 5}, [0, 1, 2, 3]}
          ] do
        v = t(ishape, 3)
        h = Nx.backend_copy(v, Nx.BinaryBackend)

        d =
          Nx.subtract(
            Nx.broadcast(v, oshape, axes: axes) |> Nx.backend_copy(Nx.BinaryBackend),
            Nx.broadcast(h, oshape, axes: axes)
          )
          |> Nx.abs()
          |> Nx.reduce_max()
          |> Nx.to_number()

        assert d == 0.0, "broadcast #{inspect(ishape)}->#{inspect(oshape)} diverged by #{d}"
      end
    end

    test "reduce with a kept axis in the middle — the conv bias gradient" do
      # sum(axes: [0, 2, 3]) over {N, C, H, W} keeps C, which is neither a
      # leading prefix nor a trailing suffix, so it did not map to the reduce
      # shader's contiguous (outer, reduce_size, inner) slabs. Rotating the
      # kept axes to the front makes it a trailing-suffix reduce.
      x = t({8, 4, 6, 6}, 1)

      assert Fallback.count_total(fn -> Nx.sum(x, axes: [0, 2, 3]) end) == 0
      assert Fallback.count_total(fn -> Nx.reduce_max(x, axes: [0, 2, 3]) end) == 0
      assert Fallback.count_total(fn -> Nx.sum(t({4, 5, 6}, 1), axes: [0, 2]) end) == 0
      assert Fallback.count_total(fn -> Nx.sum(t({4, 5, 6, 7}, 1), axes: [1, 3]) end) == 0
    end

    test "middle-axis reduce matches BinaryBackend" do
      for {shape, axes} <- [
            {{8, 4, 6, 6}, [0, 2, 3]},
            {{4, 5, 6}, [0, 2]},
            {{4, 5, 6, 7}, [1, 3]}
          ] do
        v = t(shape, 2)
        h = Nx.backend_copy(v, Nx.BinaryBackend)

        for {name, f} <- [
              sum: &Nx.sum/2,
              reduce_max: &Nx.reduce_max/2,
              reduce_min: &Nx.reduce_min/2
            ] do
          d =
            Nx.subtract(
              f.(v, axes: axes) |> Nx.backend_copy(Nx.BinaryBackend),
              f.(h, axes: axes)
            )
            |> Nx.abs()
            |> Nx.reduce_max()
            |> Nx.to_number()

          assert d < 1.0e-9, "#{name} #{inspect(shape)} axes #{inspect(axes)} diverged by #{d}"
        end
      end
    end

    test "pooling — forward and backward stay on the GPU" do
      x = t({2, 3, 6, 6}, 1)
      w = {1, 1, 2, 2}
      st = [1, 1, 2, 2]

      assert Fallback.count_total(fn -> Nx.window_max(x, w, strides: st) end) == 0

      src = t({2, 3, 3, 3}, 2)

      # INTEGER init_value, exactly as Nx's gradient passes it. My first
      # parity test used Nx.tensor(0.0) — a float — and passed while the real
      # path still fell back on {:s, 32}. Six ops in this backend have now
      # been blocked by an integer literal, so the test must use the untidy
      # value the real caller does.
      iv = Nx.tensor(0, backend: VulkanoBackend)

      assert Fallback.count_total(fn -> Nx.window_scatter_max(x, src, iv, w, strides: st) end) ==
               0

      # Correctness is checked against the SOURCE, not against BinaryBackend.
      #
      # Nx.BinaryBackend.window_scatter_max/5 round-trips f64 values through an
      # f32 intermediate — for src[0] = 2.4715269558223154 it returns
      # 2.471526861190796. The shader copies the f64 value exactly, so a
      # bit-equality assertion against the host reference fails on the
      # reference's own precision loss. (window_scatter_max is already listed
      # as a real open bug in nx_doctest_test.exs's @backlog.)
      #
      # Every scattered value must therefore be either init (0.0) or an EXACT
      # element of src — which is a stronger check than agreeing with a
      # degraded reference.
      got = Nx.window_scatter_max(x, src, iv, w, strides: st) |> Nx.backend_copy(Nx.BinaryBackend)
      src_vals = src |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_flat_list() |> MapSet.new()

      for v <- Nx.to_flat_list(got) do
        assert v == 0.0 or MapSet.member?(src_vals, v),
               "scattered value #{v} is neither init nor an exact src element"
      end

      # ...and exactly one element per window receives a source value.
      nonzero = Nx.to_flat_list(got) |> Enum.count(&(&1 != 0.0))
      assert nonzero == Nx.size({2, 3, 3, 3}), "expected one scatter per window, got #{nonzero}"
    end

    test "pooling backward is exact on TIES — the case random data misses" do
      # remainder(3.0) makes nearly every window contain duplicates, and some
      # entirely uniform. Nx gives the gradient to the LAST maximum in
      # row-major order (verified against BinaryBackend directly), so the
      # shader scans with `>=`. With `>` this passes on random data and is
      # wrong wherever values repeat — and a relu's output is full of exact
      # ties at zero.
      for {shape, win, st} <- [
            {{2, 3, 6, 6}, {1, 1, 2, 2}, [1, 1, 2, 2]},
            {{4, 6}, {2, 3}, [2, 3]}
          ] do
        h = Nx.iota(shape, type: {:f, 64}) |> Nx.remainder(3.0)
        wshape = Nx.shape(Nx.window_max(h, win, strides: st))
        src_t = Nx.iota(wshape, type: {:f, 64}) |> Nx.add(1.0)
        iv = Nx.tensor(0.0, type: {:f, 64})

        want = Nx.window_scatter_max(h, src_t, iv, win, strides: st)

        got =
          Nx.window_scatter_max(
            Nx.backend_transfer(h, VulkanoBackend),
            Nx.backend_transfer(src_t, VulkanoBackend),
            Nx.backend_transfer(iv, VulkanoBackend),
            win,
            strides: st
          )
          |> Nx.backend_transfer(Nx.BinaryBackend)

        d = Nx.subtract(want, got) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
        assert d == 0.0, "tie-break diverged on #{inspect(shape)} by #{d}"
      end
    end

    test "conv forward" do
      x = t({2, 3, 7, 7}, 1)
      k = t({4, 3, 3, 3}, 2)
      assert Fallback.count_total(fn -> Nx.conv(x, k, padding: :same) end) == 0
    end

    test "conv with a non-identity permutation — the regression that started this" do
      # Supplied in the permuted layout so the declared permutation recovers a
      # valid NCHW conv. Before permuted_gpu_conv/4 this was a host fallback,
      # and being a host fallback is precisely what nothing detected.
      x = t({3, 2, 5, 5}, 1)
      k = t({3, 4, 3, 3}, 2)

      opts = [
        input_permutation: [1, 0, 2, 3],
        kernel_permutation: [1, 0, 2, 3],
        output_permutation: [1, 0, 2, 3]
      ]

      assert Fallback.count_total(fn -> Nx.conv(x, k, opts) end) == 0
    end

    test "dot in every rank-2 contraction orientation" do
      # y = x·W contracts [1]/[0] and always hit the shader. Its gradients do
      # not: Nx emits ∂L/∂x = g·Wᵀ as [1]/[1] and ∂L/∂W = gᵀ·x as [0]/[0],
      # permuting the contraction axes instead of materialising a transpose.
      # Both used to fall back, so every dense layer paid it twice per step.
      a = t({4, 6}, 1)
      b_kn = t({6, 5}, 2)
      b_nk = t({5, 6}, 2)

      assert Fallback.count_total(fn -> Nx.dot(a, [1], [], b_kn, [0], []) end) == 0
      assert Fallback.count_total(fn -> Nx.dot(a, [1], [], b_nk, [1], []) end) == 0
      assert Fallback.count_total(fn -> Nx.dot(a, [0], [], t({4, 5}, 3), [0], []) end) == 0
      assert Fallback.count_total(fn -> Nx.dot(a, [0], [], t({5, 4}, 3), [1], []) end) == 0
    end

    test "a dense-layer gradient performs no host dot" do
      x = t({8, 6}, 1)
      w = t({6, 4}, 2)

      grad = fn ww, xx -> Nx.Defn.grad(ww, fn w2 -> Nx.sum(Nx.dot(xx, w2)) end) end

      {_r, counts} =
        Fallback.count(fn -> Nx.Defn.jit_apply(grad, [w, x], compiler: Nx.Defn.Evaluator) end)

      assert counts[{:dot, 7}] == nil,
             "the dense gradient went back to the host: #{inspect(counts)}"
    end

    test "rank-0 compare and select — the scalar support check" do
      # The gate read `tuple_size(out.shape) >= 1`, so every scalar predicate
      # went to the host: 108 of the 137 fallbacks in one eXMC value_and_grad
      # (bench_results/EXMC_PEROP_RACE.md). Nothing in compare_f*/select_f*
      # needed rank >= 1 — elementwise arithmetic handled rank 0 the whole time.
      # Rank 0 now dispatches as rank 1 {1}.
      x = Nx.tensor(0.7, type: {:f, 64}, backend: VulkanoBackend)
      y = Nx.tensor(0.2, type: {:f, 64}, backend: VulkanoBackend)

      assert Fallback.count_total(fn -> Nx.equal(x, y) end) == 0
      assert Fallback.count_total(fn -> Nx.not_equal(x, y) end) == 0
      assert Fallback.count_total(fn -> Nx.greater(x, y) end) == 0
      assert Fallback.count_total(fn -> Nx.less(x, y) end) == 0
      assert Fallback.count_total(fn -> Nx.greater_equal(x, y) end) == 0
      assert Fallback.count_total(fn -> Nx.less_equal(x, y) end) == 0
      assert Fallback.count_total(fn -> Nx.select(Nx.greater(x, y), x, y) end) == 0

      # the shape a distribution's log_p is actually made of: `x > 0` guarding
      # a computation, with an out-of-support constant on the other branch
      assert Fallback.count_total(fn ->
               Nx.select(
                 Nx.greater(x, 0),
                 Nx.multiply(x, 2.0),
                 Nx.Constants.neg_infinity({:f, 64})
               )
             end) == 0

      # f32 too, and a scalar predicate over vector branches
      x32 = Nx.tensor(0.7, type: {:f, 32}, backend: VulkanoBackend)
      v = t({4}, 1)

      assert Fallback.count_total(fn -> Nx.greater(x32, 0) end) == 0
      assert Fallback.count_total(fn -> Nx.select(Nx.greater(x, 0), v, v) end) == 0
    end

    test "rank-0 compare and select match BinaryBackend exactly" do
      for type <- [{:f, 32}, {:f, 64}], {a, b} <- [{1.5, 0.5}, {0.5, 0.5}, {-1.0, 0.0}] do
        g = fn v -> Nx.tensor(v, type: type, backend: VulkanoBackend) end
        h = fn v -> Nx.tensor(v, type: type, backend: Nx.BinaryBackend) end

        for op <- [:equal, :not_equal, :less, :less_equal, :greater, :greater_equal] do
          got = apply(Nx, op, [g.(a), g.(b)]) |> Nx.backend_copy(Nx.BinaryBackend)
          want = apply(Nx, op, [h.(a), h.(b)])

          assert Nx.to_binary(got) == Nx.to_binary(want),
                 "#{op}(#{a}, #{b}) #{inspect(type)} diverged"
        end

        got =
          Nx.select(Nx.greater(g.(a), g.(b)), g.(a), g.(b)) |> Nx.backend_copy(Nx.BinaryBackend)

        want = Nx.select(Nx.greater(h.(a), h.(b)), h.(a), h.(b))

        assert Nx.to_binary(got) == Nx.to_binary(want),
               "select(#{a}, #{b}) #{inspect(type)} diverged"
      end
    end

    test "a mixed-dtype scalar pair stays on the GPU" do
      # Same gate, third face: two scalars of *different* dtypes miss the flat
      # apply_binary path (types differ) and land in the broadcast path, whose
      # rank check also read `>= 1`. Same-dtype scalars never exercised it.
      a = Nx.tensor(1.5, type: {:f, 64}, backend: VulkanoBackend)
      b = Nx.tensor(0.5, type: {:f, 32}, backend: VulkanoBackend)

      for op <- [:multiply, :add, :subtract, :divide, :max, :min] do
        assert Fallback.count_total(fn -> apply(Nx, op, [a, b]) end) == 0,
               "#{op} on a mixed-dtype scalar pair left the GPU"
      end
    end

    test "pad — every padding_config the shader claims" do
      # pad has had a shader since thrust 2, but the gate required the pad value
      # to already carry the tensor's dtype, and `Nx.pad(t, 0.0, cfg)` hands it
      # an f32 (or s32) literal: 8 fallbacks per eXMC gradient.
      x = t({6}, 1)

      assert Fallback.count_total(fn -> Nx.pad(x, 0.0, [{1, 2, 0}]) end) == 0
      assert Fallback.count_total(fn -> Nx.pad(x, 0, [{1, 2, 0}]) end) == 0
      assert Fallback.count_total(fn -> Nx.pad(x, 0.0, [{0, 0, 2}]) end) == 0
      assert Fallback.count_total(fn -> Nx.pad(x, 0.0, [{-1, -2, 0}]) end) == 0
      assert Fallback.count_total(fn -> Nx.pad(x, 0.0, [{2, 3, 1}]) end) == 0

      assert Fallback.count_total(fn -> Nx.pad(t({2, 3}, 1), 0.0, [{1, 1, 0}, {0, 2, 1}]) end) ==
               0

      assert Fallback.count_total(fn ->
               Nx.pad(t({2, 2, 3, 2}, 1), 0.0, [{0, 1, 0}, {1, 0, 1}, {0, 0, 2}, {1, 1, 0}])
             end) == 0

      assert Fallback.count_total(fn ->
               Nx.pad(Nx.tensor(3.25, type: {:f, 64}, backend: VulkanoBackend), 0.0, [])
             end) == 0
    end

    test "pad matches BinaryBackend exactly" do
      cases = [
        {{6}, [{1, 2, 0}]},
        {{6}, [{0, 0, 2}]},
        {{6}, [{2, 3, 1}]},
        {{6}, [{-2, 0, 0}]},
        {{6}, [{-1, -2, 0}]},
        {{6}, [{-1, 1, 1}]},
        {{1}, [{2, 2, 3}]},
        {{2, 3}, [{1, 1, 0}, {0, 2, 1}]},
        {{2, 3}, [{-1, 0, 0}, {0, -1, 0}]},
        {{2, 3, 2}, [{1, 0, 1}, {0, 1, 0}, {2, 2, 0}]},
        {{2, 2, 3, 2}, [{0, 1, 0}, {1, 0, 1}, {0, 0, 2}, {1, 1, 0}]}
      ]

      for {shape, cfg} <- cases, pv <- [0.0, 0, -7.5] do
        v = t(shape, 4)
        h = Nx.backend_copy(v, Nx.BinaryBackend)

        got = Nx.pad(v, pv, cfg) |> Nx.backend_copy(Nx.BinaryBackend)
        want = Nx.pad(h, pv, cfg)

        assert Nx.to_binary(got) == Nx.to_binary(want),
               "pad #{inspect(shape)} #{inspect(cfg)} value #{pv} diverged"
      end
    end

    test "put_slice — the overlay that decides residency" do
      # No shader at any rank until T11. Worse than its count: `PointMap` unpacks
      # the flat parameter vector with it, so once the position vector landed on
      # BinaryBackend everything downstream computed there unrecorded.
      x = t({6}, 1)
      s = t({2}, 2)

      assert Fallback.count_total(fn -> Nx.put_slice(x, [2], s) end) == 0
      assert Fallback.count_total(fn -> Nx.put_slice(x, [0], s) end) == 0
      # starts are clamped to [0, dim - slice_dim], here and in BinaryBackend
      assert Fallback.count_total(fn -> Nx.put_slice(x, [99], s) end) == 0
      assert Fallback.count_total(fn -> Nx.put_slice(x, [-3], s) end) == 0
      # a tensor start index resolves to a number rather than dropping the op
      assert Fallback.count_total(fn ->
               Nx.put_slice(x, [Nx.tensor(2, backend: VulkanoBackend)], s)
             end) == 0

      assert Fallback.count_total(fn -> Nx.put_slice(t({3, 4}, 1), [1, 1], t({2, 2}, 2)) end) == 0

      assert Fallback.count_total(fn ->
               Nx.put_slice(t({2, 2, 3, 4}, 1), [1, 0, 1, 2], t({1, 1, 2, 2}, 2))
             end) == 0

      # and the chain it exists for: unpack a flat vector, then compute
      assert Fallback.count_total(fn ->
               t({8}, 3)
               |> Nx.put_slice([0], t({2}, 4))
               |> Nx.put_slice([4], t({2}, 5))
               |> Nx.multiply(2.0)
               |> Nx.sum()
             end) == 0
    end

    test "put_slice matches BinaryBackend exactly" do
      cases = [
        {{6}, {2}, [2]},
        {{6}, {2}, [0]},
        {{6}, {2}, [4]},
        {{6}, {2}, [5]},
        {{6}, {2}, [99]},
        {{6}, {2}, [-3]},
        {{6}, {6}, [0]},
        {{6}, {1}, [3]},
        {{1}, {1}, [0]},
        {{3, 4}, {2, 2}, [1, 1]},
        {{3, 4}, {2, 2}, [2, 3]},
        {{3, 4}, {1, 4}, [2, 0]},
        {{2, 3, 4}, {1, 2, 2}, [1, 1, 2]},
        {{2, 2, 3, 4}, {1, 1, 2, 2}, [1, 0, 1, 2]}
      ]

      for {tshape, sshape, starts} <- cases do
        v = t(tshape, 6)
        s = t(sshape, 7)
        vh = Nx.backend_copy(v, Nx.BinaryBackend)
        sh = Nx.backend_copy(s, Nx.BinaryBackend)

        got = Nx.put_slice(v, starts, s) |> Nx.backend_copy(Nx.BinaryBackend)
        want = Nx.put_slice(vh, starts, sh)

        assert Nx.to_binary(got) == Nx.to_binary(want),
               "put_slice #{inspect(tshape)} <- #{inspect(sshape)} at #{inspect(starts)} diverged"

        # the same call with tensor start indices takes the same path
        idx = Enum.map(starts, &Nx.tensor(&1, backend: VulkanoBackend))
        got_idx = Nx.put_slice(v, idx, s) |> Nx.backend_copy(Nx.BinaryBackend)

        assert Nx.to_binary(got_idx) == Nx.to_binary(want),
               "put_slice with tensor starts #{inspect(starts)} diverged"
      end
    end

    test "put_slice and pad are exact in f32 as well as f64" do
      f32 = fn shape, seed ->
        t(shape, seed) |> Nx.as_type({:f, 32})
      end

      v = f32.({3, 4}, 8)
      s = f32.({2, 2}, 9)
      vh = Nx.backend_copy(v, Nx.BinaryBackend)
      sh = Nx.backend_copy(s, Nx.BinaryBackend)

      assert Nx.to_binary(Nx.put_slice(v, [1, 1], s) |> Nx.backend_copy(Nx.BinaryBackend)) ==
               Nx.to_binary(Nx.put_slice(vh, [1, 1], sh))

      assert Nx.to_binary(
               Nx.pad(v, 0.0, [{1, 1, 0}, {0, 2, 1}])
               |> Nx.backend_copy(Nx.BinaryBackend)
             ) ==
               Nx.to_binary(Nx.pad(vh, 0.0, [{1, 1, 0}, {0, 2, 1}]))

      assert Fallback.count_total(fn -> Nx.put_slice(v, [1, 1], s) end) == 0
      assert Fallback.count_total(fn -> Nx.pad(v, 0.0, [{1, 1, 0}, {0, 2, 1}]) end) == 0
    end

    test "conv gradient performs no host fallback at all" do
      x = t({2, 3, 7, 7}, 1)
      k = t({4, 3, 3, 3}, 2)

      grad = fn kk, xx ->
        Nx.Defn.grad(kk, fn k2 -> Nx.sum(Nx.conv(xx, k2, padding: :same)) end)
      end

      {_result, counts} =
        Fallback.count(fn ->
          Nx.Defn.jit_apply(grad, [k, x], compiler: Nx.Defn.Evaluator)
        end)

      # Zero. The counter is what proved this was not already true: one conv
      # used to fall back here, and not for the reason anyone guessed. The
      # gradient seed for Nx.sum is built at Nx's default f32 while the input
      # is f64, so the kernel-gradient conv arrived as f64 x f32 and failed
      # `i.type == ot and k.type == ot`. conv_coerce/2 now casts the odd
      # operand on-device instead of dropping the conv to the host.
      assert counts == %{},
             "the conv gradient went back to the host: #{inspect(counts)}"
    end
  end

  describe "u8 comparison masks — T12" do
    test "a GPU-produced u8 mask is consumable — not just by select/4" do
      # T12. The compare shaders produce a {:u, 8} mask on-device, which was the
      # point, but select/4 was the only op that could take one back: multiply,
      # sum and as_type on a mask all host-fell-back for want of a cast.
      x = Nx.tensor([[1.0, 5.0, 2.0], [9.0, 3.0, 4.0]], type: {:f, 32}, backend: VulkanoBackend)
      mask = fn -> Nx.equal(x, Nx.reduce_max(x, axes: [1], keep_axes: true)) end

      assert Nx.type(mask.()) == {:u, 8}

      assert Fallback.count_total(fn -> Nx.select(mask.(), x, x) end) == 0
      assert Fallback.count_total(fn -> Nx.multiply(mask.(), x) end) == 0
      assert Fallback.count_total(fn -> Nx.as_type(mask.(), {:f, 32}) end) == 0
      # sum of a u8 is typed {:u, 32} by Nx, so this one needed a reduce whose
      # input and output types differ — not a cast.
      assert Nx.type(Nx.sum(mask.())) == {:u, 32}
      assert Fallback.count_total(fn -> Nx.sum(mask.()) end) == 0
    end

    test "u8 mask consumption matches BinaryBackend exactly" do
      l = [[1.0, 5.0, 2.0], [9.0, 3.0, 4.0]]

      for {label, f} <- [
            {"multiply",
             fn b ->
               m = b |> mask_of()
               Nx.multiply(m, tf32(l, b))
             end},
            {"sum", fn b -> Nx.sum(mask_of(b)) end},
            {"sum axis", fn b -> Nx.sum(mask_of(b), axes: [1]) end},
            {"as_type", fn b -> Nx.as_type(mask_of(b), {:f, 32}) end}
          ] do
        gpu = f.(VulkanoBackend) |> Nx.backend_transfer(Nx.BinaryBackend)
        host = f.(Nx.BinaryBackend)
        assert Nx.to_binary(gpu) == Nx.to_binary(host), "#{label} diverged from the host"
      end
    end

    test "softmax's backward pass no longer leaves the GPU" do
      # This is what T12 was for. Nx.Defn.Grad's reduce_max rule builds a u8
      # tie mask, sums it to count ties, and divides by that count — three ops
      # on a mask, none of which had a GPU path. The values were bit-identical
      # throughout, so only a census could ever have seen it.
      softmax = fn t ->
        e = Nx.exp(Nx.subtract(t, Nx.reduce_max(t, axes: [1], keep_axes: true)))
        Nx.divide(e, Nx.sum(e, axes: [1], keep_axes: true))
      end

      grad_fn = fn x -> Nx.Defn.grad(x, &Nx.sum(softmax.(&1))) end

      for l <- [[[1.0, 5.0, 2.0], [9.0, 3.0, 4.0]], [[5.0, 5.0, 2.0], [3.0, 3.0, 3.0]]] do
        assert Fallback.count_total(fn ->
                 Nx.Defn.jit_apply(grad_fn, [tf32(l, VulkanoBackend)],
                   compiler: Nx.Defn.Evaluator
                 )
               end) == 0

        gpu =
          Nx.Defn.jit_apply(grad_fn, [tf32(l, VulkanoBackend)], compiler: Nx.Defn.Evaluator)
          |> Nx.backend_transfer(Nx.BinaryBackend)

        host =
          Nx.Defn.jit_apply(grad_fn, [tf32(l, Nx.BinaryBackend)], compiler: Nx.Defn.Evaluator)

        assert Nx.to_binary(gpu) == Nx.to_binary(host)
      end
    end

    test "reduce_max's gradient splits across TIES on the GPU, exactly" do
      # Ties are the case random data never produces and the whole reason the
      # mask exists — with one maximum the gradient is a one-hot and any
      # half-correct implementation passes.
      f = fn x -> Nx.Defn.grad(x, &Nx.sum(Nx.reduce_max(&1, axes: [1]))) end

      for {l, expected} <- [
            {[[1.0, 5.0, 2.0]], [0.0, 1.0, 0.0]},
            {[[5.0, 5.0, 2.0]], [0.5, 0.5, 0.0]},
            {[[7.0, 7.0, 7.0]], [1 / 3, 1 / 3, 1 / 3]}
          ] do
        assert Fallback.count_total(fn ->
                 Nx.Defn.jit_apply(f, [tf32(l, VulkanoBackend)], compiler: Nx.Defn.Evaluator)
               end) == 0

        gpu =
          Nx.Defn.jit_apply(f, [tf32(l, VulkanoBackend)], compiler: Nx.Defn.Evaluator)
          |> Nx.backend_transfer(Nx.BinaryBackend)

        host = Nx.Defn.jit_apply(f, [tf32(l, Nx.BinaryBackend)], compiler: Nx.Defn.Evaluator)

        assert Nx.to_binary(gpu) == Nx.to_binary(host)
        assert Nx.to_flat_list(gpu) == Nx.to_flat_list(Nx.tensor(expected, type: {:f, 32}))
      end
    end
  end

  describe "known fallbacks — pinned so promoting one is noticed" do
    test "u8 reduce_max/reduce_min are RESIDENT — the packed writer arrived" do
      # This asserted the opposite. Its reasoning was sound and its conclusion
      # expired: sum of a u8 is {:u, 32} and reduce_axis_u8_to_u32 handles it,
      # while max/min keep the {:u, 8} output type, "which would need a
      # byte-PACKED writer rather than a word one".
      #
      # It does, and cast_s32_to_narrow.comp is that writer. The narrow-integer
      # pair widens the mask to s32, reduces with the existing kernel and packs
      # the result back down, so no {:u, 8} reduce kernel was needed either.
      #
      # Unlike the middle-axis pin below, this one was not defending a mistake —
      # it named a real missing capability. It is kept because a pin whose
      # premise has been satisfied should say so out loud.
      m = mask_of(VulkanoBackend)
      assert Nx.type(Nx.reduce_max(m)) == {:u, 8}
      assert Fallback.count_total(fn -> Nx.reduce_max(m) end) == 0
      assert Fallback.count_total(fn -> Nx.reduce_min(m) end) == 0

      host = Nx.backend_transfer(m, Nx.BinaryBackend)
      assert Nx.to_number(Nx.reduce_max(m)) == Nx.to_number(Nx.reduce_max(host))
      assert Nx.to_number(Nx.reduce_min(m)) == Nx.to_number(Nx.reduce_min(host))
    end

    test "a MIDDLE-axis u8 sum is RESIDENT — this pin was wrong, and how" do
      # This test used to assert the opposite, on this reasoning:
      #
      #   "The middle-axis case rotates kept axes to the front and reduces the
      #    trailing block — but that rotation is a transpose, and transpose_nd
      #    has no u8 path, so routing a mask through it would trade one
      #    fallback for another."
      #
      # The premise is false. A middle-axis reduction needs NO rotation: the
      # shaders push (outer, reduce_size, inner) and stride by `inner`, which
      # already expresses a run of axes sitting anywhere in the shape. Nothing
      # had to be transposed; `classify_reduce_axes/2` simply refused to emit
      # the slab. See its comment and test/nx_vulkan/reduce_axes_test.exs.
      #
      # Worth keeping as a test rather than deleting, because the failure mode
      # is instructive: a pin that records a BELIEF about the implementation
      # rather than a measured limit will defend the belief. This one held a
      # narrow gate shut across four op families for as long as it was trusted.
      i = Nx.iota({2, 3, 4}, type: {:f, 32}, backend: VulkanoBackend)
      m = Nx.greater(i, Nx.tensor(2.0, type: {:f, 32}, backend: VulkanoBackend))

      assert Fallback.count_total(fn -> Nx.sum(m, axes: [1]) end) == 0

      assert Nx.to_flat_list(Nx.sum(m, axes: [1])) ==
               Nx.to_flat_list(Nx.sum(Nx.backend_transfer(m, Nx.BinaryBackend), axes: [1]))
    end

    test "OVERLAPPING pooling backward still falls back" do
      # One thread per input element is what avoids float atomics, and that
      # only holds when windows do not overlap. stride < window would need
      # GL_EXT_shader_atomic_float, which the Kepler fleet does not guarantee.
      h = t({1, 1, 5, 5}, 1)
      src_t = t({1, 1, 4, 4}, 2)
      iv = Nx.tensor(0.0, type: {:f, 64}, backend: VulkanoBackend)

      {_r, counts} =
        Fallback.count(fn ->
          Nx.window_scatter_max(h, src_t, iv, {1, 1, 2, 2}, strides: [1, 1, 1, 1])
        end)

      assert counts == %{{:window_scatter_max, 6} => 1}
    end

    @tag :host_fallback_expected
    test "block/4 is attributed per Nx.Block struct, not as one {:block, 4}" do
      # T13. Until this landed, block/4 transferred to BinaryBackend without
      # passing host_result/2, so Nx.LinAlg, top_k, cumulative_*, take and
      # all_close were invisible to count/1 AND to strict mode: "zero
      # fallbacks" meant "zero recorded".
      x = t({4}, 1)

      # W4 decided all twelve remaining blocks, and decided most of them by
      # ROUTING rather than allowlisting: their bodies now run on this backend,
      # so no `{:block, _}` is recorded at all and the constituent op reports
      # instead. That census is what paid off — it said `cumulative_sum` was not
      # missing a "scan shader" but `concatenate/3`, shared with three sibling
      # cumulative ops and `take_along_axis/3`. `glsl/concat_nd.comp` closed it,
      # and this is now fully resident: an allowlist entry for the block would
      # have recorded a decision and left five ops on the host.
      {_r, counts} = Fallback.count(fn -> Nx.cumulative_sum(x) end)
      assert counts[{:block, Nx.Block.CumulativeSum}] == nil
      assert counts == %{}

      # top_k's only host component is the sort, which IS a standing decision.
      # Routing makes it inherit that entry honestly instead of restating it,
      # and the values/indices come back GPU-resident.
      {r, counts} = Fallback.count(fn -> Nx.top_k(x, k: 2) end)
      assert counts[{:block, Nx.Block.TopK}] == nil
      assert counts == %{{:argsort, 3} => 1}
      assert elem(r, 0).data.__struct__ == VulkanoBackend

      # Blocks that ARE still transferred stay attributed per struct — the
      # property that lets a genuine gap be refused while all_close, this
      # suite's own assertion helper, stays permitted.
      {_r, counts} = Fallback.count(fn -> Nx.all_close(x, x) end)
      assert counts[{:block, Nx.Block.AllClose}] == 1
      assert counts[{:block, Nx.Block.CumulativeSum}] == nil
    end

    @tag :host_fallback_expected
    test "what an Nx.LinAlg call costs no longer depends on the DEFAULT backend" do
      # This test used to assert the OPPOSITE, and was right to: nx composes SVD
      # from ordinary ops, and where their intermediates landed decided whether
      # this backend ever saw them.
      #
      #   default BinaryBackend  -> composition ran on the host, census = 1
      #   default VulkanoBackend -> intermediates came back here one at a time,
      #                             census = several hundred
      #
      # Same call, same input tensor, two orders of magnitude apart. W3 closed
      # that gap: block/4 now wraps the body in with_binary_backend/1, because
      # transferring the ARGS never governed where the defn body computed — the
      # evaluator materialises constants and intermediates on the process
      # default. So the composition stays on the host either way, and the census
      # is the same number twice.
      #
      # This is not only tidier. It is why Nx.LinAlg.lu/1 stopped returning a
      # wrong matrix for the identity: those hundreds of round trips were the
      # bug's mechanism, not just its cost. See Nx.Vulkan.LinAlgTest.
      m = Nx.tensor([[4.0, 1.0], [1.0, 3.0]], type: {:f, 64}, backend: VulkanoBackend)
      expected = %{{:block, Nx.Block.LinAlg.SVD} => 1}

      {_r, host_default} = Fallback.count(fn -> Nx.LinAlg.svd(m) end)
      assert host_default == expected

      previous = Nx.default_backend(VulkanoBackend)

      try do
        {_r, gpu_default} = Fallback.count(fn -> Nx.LinAlg.svd(m) end)

        # Pinned at equality, not at "small". If this grows again, a block body
        # has started leaking onto the GPU one intermediate at a time and the
        # correctness bug W3 fixed is reachable once more.
        assert gpu_default == expected,
               "an Nx.LinAlg call recorded #{inspect(gpu_default)} with the GPU as " <>
                 "default backend, but #{inspect(expected)} with the host as default. " <>
                 "block/4 is leaking the default backend into the defn body again."
      after
        Nx.default_backend(previous)
      end
    end

    test "sort/argsort" do
      x = t({16}, 1)
      assert Fallback.count_total(fn -> Nx.sort(x) end) > 0
    end
  end

  defp tf32(l, backend), do: Nx.tensor(l, type: {:f, 32}, backend: backend)

  defp mask_of(backend) do
    x = tf32([[1.0, 5.0, 2.0], [9.0, 3.0, 4.0]], backend)
    Nx.equal(x, Nx.reduce_max(x, axes: [1], keep_axes: true))
  end
  describe "pow through the broadcast path" do
    # `Nx.pow(t, 2.0)` is the ordinary way to write a square, and a scalar
    # exponent takes the BROADCAST path. That path had no `pow` arm, so float
    # pow left the GPU for anyone who wrote it naturally, while
    # `Nx.pow(t, tensor_of_twos)` stayed resident. Found by censusing a per-op
    # Weibull leapfrog; confirmed independently on Tegra.
    test "a scalar exponent stays on the GPU and matches the same-shape path" do
      vals = [3.0, 1.5, 7.0]

      # f32: both forms resident, and they must agree exactly.
      t32 = Nx.tensor(vals, type: {:f, 32}, backend: Nx.Vulkan.VulkanoBackend)
      s32 = Nx.tensor(0.5, type: {:f, 32}, backend: Nx.Vulkan.VulkanoBackend)
      m32 = Nx.tensor([0.5, 0.5, 0.5], type: {:f, 32}, backend: Nx.Vulkan.VulkanoBackend)

      {bcast32, c1} = Nx.Vulkan.Fallback.count(fn -> Nx.pow(t32, s32) end)
      {direct32, c2} = Nx.Vulkan.Fallback.count(fn -> Nx.pow(t32, m32) end)

      assert Map.values(c1) |> Enum.sum() == 0, "f32 bcast pow fell back"
      assert Map.values(c2) |> Enum.sum() == 0, "f32 same-shape pow fell back"
      assert Nx.to_flat_list(bcast32) == Nx.to_flat_list(direct32)

      # f64 broadcasting pow stays on the HOST on purpose — MISSION.md §3.2
      # declines the f32 boundary cast. Pinned so that "optimising" it onto the
      # GPU has to come here and change a decision, not just widen a gate.
      t64 = Nx.tensor(vals, type: {:f, 64}, backend: Nx.Vulkan.VulkanoBackend)
      s64 = Nx.tensor(0.5, type: {:f, 64}, backend: Nx.Vulkan.VulkanoBackend)

      {bcast64, c3} = Nx.Vulkan.Fallback.count(fn -> Nx.pow(t64, s64) end)

      assert Map.values(c3) |> Enum.sum() == 1, "f64 bcast pow should still fall back"
      assert hd(Nx.to_flat_list(bcast64)) == :math.pow(3.0, 0.5),
             "f64 pow must keep full precision on the host path"
    end
  end

  describe "an unrecognised host_fallback mode" do
    # Measured before the fix: `config :nx_vulkan, host_fallback: :raies` made
    # an ALLOWLISTED op return :ok as usual and a non-allowlisted one raise
    # CaseClauseError — "no case clause matching: :raies" — from inside
    # enforce/3, naming neither the config key nor the misspelling. Meanwhile
    # `strict/2` rejected the identical typo with FunctionClauseError, so the
    # two entry points disagreed.
    #
    # The dangerous row was the first one, not the second: a suite with no
    # refused op is exactly what strict mode exists to certify, so on a clean
    # codebase the typo was never detected and the run certified nothing.

    setup do
      on_exit(fn -> Application.delete_env(:nx_vulkan, :host_fallback) end)
      :ok
    end

    test "raises naming the config key, on the FIRST fallback of any kind" do
      Application.put_env(:nx_vulkan, :host_fallback, :raies)
      meta = Nx.tensor([[1.0, 2.0]], type: {:f, 64}, backend: Nx.BinaryBackend)

      # An allowlisted op. This is the case that used to return :ok in silence.
      err = assert_raise ArgumentError, fn -> Fallback.note({:sort, 3}, meta) end

      assert err.message =~ "host_fallback: :raies"
      assert err.message =~ ":allow"
      assert err.message =~ ":raise"
    end

    test "the null arm — every valid mode still passes through untouched" do
      meta = Nx.tensor([[1.0, 2.0]], type: {:f, 64}, backend: Nx.BinaryBackend)

      for mode <- [:allow, :warn, :raise] do
        Application.put_env(:nx_vulkan, :host_fallback, mode)
        assert Fallback.mode() == mode
        assert Fallback.note({:sort, 3}, meta) == :ok, "allowlisted op broke under #{mode}"
      end

      # And the refusal still refuses with the RIGHT error, not ArgumentError.
      Application.put_env(:nx_vulkan, :host_fallback, :raise)

      assert_raise Nx.Vulkan.HostFallbackError, fn ->
        Fallback.note({:add, 3}, Nx.tensor([1.0], backend: Nx.BinaryBackend))
      end
    end

    test "an in-process strict/2 mode is not re-validated against config" do
      # strict/2 is already guarded by `mode in @modes`, so the process path
      # needs no second check — and must keep working while a bad value sits in
      # the application env, or the fix would have made the escape hatch
      # unusable exactly when someone needs it to debug the bad value.
      Application.put_env(:nx_vulkan, :host_fallback, :nonsense)

      assert Fallback.strict(:allow, fn -> Fallback.mode() end) == :allow
      assert_raise FunctionClauseError, fn -> Fallback.strict(:nonsense, fn -> :ok end) end
    end
  end

end
