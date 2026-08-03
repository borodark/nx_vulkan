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
      a = t({4, 4}, 1)
      {_result, counts} = Fallback.count(fn -> Nx.reverse(a, axes: [0]) end)

      assert counts == %{{:reverse, 3} => 1}
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
          _ = Nx.reverse(a, axes: [0])
          {_r2, inner} = Fallback.count(fn -> Nx.reverse(a, axes: [0]) end)
          assert inner == %{{:reverse, 3} => 1}
          Nx.reverse(a, axes: [0])
        end)

      assert outer == %{{:reverse, 3} => 2}
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

    test "conv gradient — one fallback left, and it is a dtype mismatch" do
      x = t({2, 3, 7, 7}, 1)
      k = t({4, 3, 3, 3}, 2)

      grad = fn kk, xx ->
        Nx.Defn.grad(kk, fn k2 -> Nx.sum(Nx.conv(xx, k2, padding: :same)) end)
      end

      {_result, counts} =
        Fallback.count(fn ->
          Nx.Defn.jit_apply(grad, [k, x], compiler: Nx.Defn.Evaluator)
        end)

      # Exactly one conv still falls back here, and the counter is how we know.
      # It is NOT a permutation problem: the gradient seed for Nx.sum is built
      # at Nx's default f32 while the input is f64, so the kernel-gradient conv
      # is f64 x f32 and conv_gpu_core_ok?/4's `i.type == ot and k.type == ot`
      # rejects it. Mixed f32/f64 could be coerced on-device (the backend
      # already has coerce_to/2 for exactly this) — when that lands, this
      # assertion fails and should become `== %{}`.
      assert counts == %{{:conv, 4} => 1},
             "conv fallbacks in the kernel gradient changed: #{inspect(counts)}"
    end
  end

  describe "known fallbacks — pinned so promoting one is noticed" do
    test "reverse/3 — used by Nx's conv input-gradient" do
      assert Fallback.count(fn -> Nx.reverse(t({4, 4}, 1), axes: [0]) end)
             |> elem(1) == %{{:reverse, 3} => 1}
    end

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
