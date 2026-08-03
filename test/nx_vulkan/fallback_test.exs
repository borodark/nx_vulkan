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
      assert Fallback.note({:whatever, 1}) == :ok
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
          Nx.subtract(Nx.backend_copy(got, Nx.BinaryBackend) |> Nx.as_type({:f, 64}),
            Nx.as_type(want, {:f, 64}))
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
      for {shape, axes} <- [{{8, 4, 6, 6}, [0, 2, 3]}, {{4, 5, 6}, [0, 2]}, {{4, 5, 6, 7}, [1, 3]}] do
        v = t(shape, 2)
        h = Nx.backend_copy(v, Nx.BinaryBackend)

        for {name, f} <- [sum: &Nx.sum/2, reduce_max: &Nx.reduce_max/2, reduce_min: &Nx.reduce_min/2] do
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

      assert counts[{:dot, 7}] == nil, "the dense gradient went back to the host: #{inspect(counts)}"
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

  describe "known fallbacks — pinned so promoting one is noticed" do
    test "window_max/4 and window_scatter_max — max-pooling, forward and backward" do
      x = t({1, 2, 4, 4}, 1)

      {_r, counts} =
        Fallback.count(fn -> Nx.window_max(x, {1, 1, 2, 2}, strides: [1, 1, 2, 2]) end)

      assert counts == %{{:window_max, 4} => 1},
             "window_max moved on-device — promote it out of this test"
    end

    test "sort/argsort" do
      x = t({16}, 1)
      assert Fallback.count_total(fn -> Nx.sort(x) end) > 0
    end
  end
end
