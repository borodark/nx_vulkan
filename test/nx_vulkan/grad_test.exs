defmodule Nx.Vulkan.GradTest do
  @moduledoc """
  Backward-pass parity: `Nx.Defn.grad` on `Nx.Vulkan.VulkanoBackend` must match
  `Nx.BinaryBackend` for every op the backend accelerates.

  This suite exists because its absence hid a real regression. `Nx.Defn.Grad`
  *generates* ops nobody writes by hand, and the backend's GPU fast paths were
  all gated on shapes a forward pass produces. Conv was the worst case: its
  gradient emits convolutions with the first two axes swapped
  (`conv_spec_transpose/1`), which failed the identity-permutation check, so the
  entire backward pass silently ran on `Nx.BinaryBackend` — correct results,
  ~30 s per CNN training step, and no signal anywhere that it was happening.

  So the rule these tests encode: **a fast path is not covered until its
  gradient is covered.** Every case below asserts numerical agreement with the
  host reference.

  Gradient inputs are passed as arguments to `Nx.Defn.jit_apply/3`, never
  captured in the closure — `Nx.Defn.grad` rejects a captured tensor that is on
  a non-default backend.

  ## Tolerances

  Pure-arithmetic chains (add/mul/dot/conv/sum) agree with the host to ~1e-10.
  Anything containing a **transcendental** (exp/log/tanh/sigmoid, hence softmax
  and any tanh-activated net) is compared at ~1e-6: SPIR-V's GLSL.std.450 has
  no f64 transcendentals, so those are boundary-cast through f32 and carry f32
  precision even in an f64 graph. That is a documented property of the backend,
  not slop — see `docs/EXMC_VULKAN_DOS_AND_DONTS.md`.

  ## Residency

  These tests assert *numbers*, not which device produced them. Residency of the
  thing that regressed — a permuted conv staying on the GPU — is asserted in
  `Nx.Vulkan.ConvTest`. A gradient's final tensor is often host-resident anyway
  because `reverse/3` (used by the conv input-gradient) is still a host
  fallback, which is the next residency leak in the backward chain.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  # Deterministic, non-degenerate data (no zeros, no ties) so gradients of
  # max/abs/sign-like ops are well defined and comparable.
  defp gen(shape, backend, seed, type) do
    size = Tuple.product(shape)
    data = for i <- 1..size, do: :math.sin(seed * 0.7 + i * 0.41) + 1.5
    Nx.tensor(data, type: type, backend: backend) |> Nx.reshape(shape)
  end

  defp max_abs_diff(a, b) do
    Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend), Nx.backend_copy(b, Nx.BinaryBackend))
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  # Differentiate `fun` wrt its first argument on both backends and compare.
  defp grad_parity(fun, shapes, opts \\ []) do
    type = Keyword.get(opts, :type, {:f, 64})
    tol = Keyword.get(opts, :tol, 1.0e-10)

    vk_args = for {s, i} <- Enum.with_index(shapes), do: gen(s, VulkanoBackend, i + 1, type)
    host_args = for {s, i} <- Enum.with_index(shapes), do: gen(s, Nx.BinaryBackend, i + 1, type)

    # Differentiate wrt the first argument. Each tensor is a real jit argument
    # (never a closure capture), which is what Nx.Defn.grad requires of a
    # tensor living on a non-default backend.
    grad_fun =
      case length(shapes) do
        1 -> fn a -> Nx.Defn.grad(a, fn x -> fun.(x) end) end
        2 -> fn a, b -> Nx.Defn.grad(a, fn x -> fun.(x, b) end) end
        3 -> fn a, b, c -> Nx.Defn.grad(a, fn x -> fun.(x, b, c) end) end
      end

    got = Nx.Defn.jit_apply(grad_fun, vk_args, compiler: Nx.Defn.Evaluator)
    ref = Nx.Defn.jit_apply(grad_fun, host_args, compiler: Nx.Defn.Evaluator)

    assert Nx.shape(got) == Nx.shape(ref)
    assert max_abs_diff(got, ref) < tol
    got
  end

  describe "elementwise chains" do
    test "add/multiply/subtract chain" do
      grad_parity(&Nx.sum(Nx.multiply(Nx.add(&1, &2), Nx.subtract(&1, &2))), [{4, 5}, {4, 5}])
    end

    test "divide and pow" do
      grad_parity(&Nx.sum(Nx.divide(Nx.pow(&1, 2), &2)), [{3, 4}, {3, 4}])
    end

    test "exp / log / sqrt" do
      # exp/log boundary-cast through f32 (no f64 transcendentals in SPIR-V).
      grad_parity(&Nx.sum(Nx.exp(Nx.divide(&1, 10.0))), [{4, 4}], tol: 1.0e-6)
      grad_parity(&Nx.sum(Nx.log(&1)), [{4, 4}], tol: 1.0e-6)
      # sqrt IS native in f64 — hold it to the tight tolerance.
      grad_parity(&Nx.sum(Nx.sqrt(&1)), [{4, 4}])
    end

    test "tanh / sigmoid — the activation gradients an MLP actually uses" do
      grad_parity(&Nx.sum(Nx.tanh(&1)), [{5, 3}], tol: 1.0e-6)
      grad_parity(&Nx.sum(Nx.sigmoid(&1)), [{5, 3}], tol: 1.0e-6)
    end

    test "max / min against a second tensor (relu-shaped)" do
      grad_parity(&Nx.sum(Nx.max(&1, &2)), [{4, 4}, {4, 4}])
      grad_parity(&Nx.sum(Nx.min(&1, &2)), [{4, 4}, {4, 4}])
    end

    test "broadcasting binary — bias-add shape" do
      grad_parity(&Nx.sum(Nx.add(&1, &2)), [{4, 6}, {6}])
    end
  end

  describe "reductions" do
    test "sum over all axes and over one axis" do
      grad_parity(&Nx.sum(Nx.multiply(&1, &1)), [{4, 5}])
      grad_parity(&Nx.sum(Nx.sum(&1, axes: [0])), [{4, 5}])
    end

    test "mean" do
      grad_parity(&Nx.mean(Nx.multiply(&1, &1)), [{3, 6}])
    end

    # OPEN, not waived: reduce_max's gradient builds a {:u, 8} comparison mask
    # on the GPU, then `as_type`s it to float and `sum`s it — and neither has a
    # u8 path, so both host-fall-back. Found by strict mode; see T12 in
    # PLAN_AFTER_BACKWARD_PASS.md. Values are correct, residency is not.
    @tag :host_fallback_open
    test "reduce_max — gradient routes to the argmax slot" do
      grad_parity(&Nx.sum(Nx.reduce_max(&1, axes: [1])), [{4, 5}])
    end
  end

  describe "matmul and transpose" do
    test "dot — gradient wrt the left operand" do
      grad_parity(&Nx.sum(Nx.dot(&1, &2)), [{4, 3}, {3, 5}])
    end

    test "transpose round-trip" do
      grad_parity(&Nx.sum(Nx.multiply(Nx.transpose(&1), Nx.transpose(&1))), [{4, 6}])
    end

    test "x @ Wᵀ — the dense-layer shape" do
      grad_parity(&Nx.sum(Nx.dot(&1, Nx.transpose(&2))), [{4, 3}, {5, 3}])
    end
  end

  describe "softmax / layernorm composites" do
    # OPEN, not waived: same u8-mask root cause as reduce_max above — softmax
    # contains a reduce_max, whose gradient mask is multiplied and summed on the
    # host. See T12 in PLAN_AFTER_BACKWARD_PASS.md.
    @tag :host_fallback_open
    test "softmax" do
      softmax = fn x ->
        m = Nx.reduce_max(x, axes: [1], keep_axes: true)
        e = Nx.exp(Nx.subtract(x, m))
        Nx.divide(e, Nx.sum(e, axes: [1], keep_axes: true))
      end

      grad_parity(&Nx.sum(Nx.multiply(softmax.(&1), softmax.(&1))), [{4, 5}], tol: 1.0e-6)
    end

    test "layernorm-shaped mean/variance normalisation" do
      norm = fn x ->
        mu = Nx.mean(x, axes: [1], keep_axes: true)
        c = Nx.subtract(x, mu)
        var = Nx.mean(Nx.multiply(c, c), axes: [1], keep_axes: true)
        Nx.divide(c, Nx.sqrt(Nx.add(var, 1.0e-6)))
      end

      grad_parity(&Nx.sum(norm.(&1)), [{4, 6}])
    end
  end

  describe "conv — the regression this suite was written for" do
    # Nx's conv gradient emits convolutions whose input/kernel/output
    # permutations swap the first two axes. Before permuted_gpu_conv/4 the whole
    # backward pass host-fell-back; these pin the numbers, and ConvTest pins
    # that a permuted conv still dispatches on the GPU.

    test "grad wrt kernel matches the host reference" do
      grad_parity(
        fn k, x -> Nx.sum(Nx.conv(x, k, padding: :same)) end,
        [{4, 3, 3, 3}, {2, 3, 7, 7}]
      )
    end

    test "grad wrt input matches the host reference" do
      grad_parity(
        fn x, k -> Nx.sum(Nx.conv(x, k, padding: :same)) end,
        [{2, 3, 7, 7}, {4, 3, 3, 3}]
      )
    end

    test "strided + padded conv gradient" do
      grad_parity(
        fn k, x -> Nx.sum(Nx.conv(x, k, strides: [2, 2], padding: [{1, 1}, {1, 1}])) end,
        [{6, 4, 3, 3}, {2, 4, 8, 8}]
      )
    end

    test "conv gradient in f32" do
      grad_parity(
        fn k, x -> Nx.sum(Nx.conv(x, k, padding: :same)) end,
        [{4, 3, 3, 3}, {2, 3, 7, 7}],
        type: {:f, 32},
        tol: 1.0e-3
      )
    end

    test "conv feeding a nonlinearity — a real layer's backward pass" do
      grad_parity(
        fn k, x -> Nx.sum(Nx.tanh(Nx.conv(x, k, padding: :same))) end,
        [{4, 3, 3, 3}, {2, 3, 6, 6}],
        tol: 1.0e-6
      )
    end
  end

  describe "pooling" do
    # window_max / window_scatter_max are still host fallbacks. They must stay
    # numerically correct; when they move on-device these tests are the guard.
    test "window_max gradient" do
      grad_parity(
        &Nx.sum(Nx.window_max(&1, {1, 1, 2, 2}, strides: [1, 1, 2, 2])),
        [{1, 2, 4, 4}]
      )
    end
  end

  describe "end-to-end" do
    test "a two-layer MLP's full gradient matches the host" do
      mlp = fn w1, x, w2 ->
        h = Nx.tanh(Nx.dot(x, w1))
        Nx.sum(Nx.multiply(Nx.dot(h, w2), 2.0))
      end

      grad_parity(mlp, [{6, 8}, {4, 6}, {8, 3}], tol: 1.0e-5)
    end

    test "a conv → activation → dense head, differentiated wrt the kernel" do
      net = fn k, x, w ->
        c = Nx.tanh(Nx.conv(x, k, padding: :same))
        flat = Nx.reshape(c, {2, 4 * 5 * 5})
        Nx.sum(Nx.dot(flat, w))
      end

      grad_parity(net, [{4, 3, 3, 3}, {2, 3, 5, 5}, {100, 2}])
    end
  end
end
