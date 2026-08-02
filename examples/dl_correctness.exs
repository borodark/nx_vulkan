# Classic deep-learning correctness checks on Nx.Vulkan.VulkanoBackend.
#
# Each example runs the identical computation on VulkanoBackend (real Vulkan
# f64 compute — conv/fft on the GPU) and on Nx.BinaryBackend (the reference),
# and reports the max absolute difference. Bit-identical / machine-epsilon
# agreement is the pass bar.
#
#   mix run examples/dl_correctness.exs

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend

# ---- helpers -------------------------------------------------------------

defmodule H do
  # deterministic pseudo-random list of length n in [-1, 1)
  def rand(n, seed) do
    for i <- 1..n, do: :math.sin(seed * 12.9898 + i * 78.233) * 0.7
  end

  def t(list, shape, backend), do: Nx.tensor(list, type: {:f, 64}, backend: backend) |> Nx.reshape(shape)

  def maxdiff(a, b) do
    ab = Nx.backend_copy(a, Nx.BinaryBackend)
    bb = Nx.backend_copy(b, Nx.BinaryBackend)

    if Nx.shape(ab) != Nx.shape(bb) do
      :shape_mismatch
    else
      Nx.subtract(ab, bb) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
    end
  end

  def report(label, diff, on_gpu \\ nil) do
    tag = if diff == :shape_mismatch or diff > 1.0e-8, do: "FAIL", else: "PASS"
    gpu = if on_gpu == nil, do: "", else: "  on_gpu=#{on_gpu}"
    IO.puts("  [#{tag}] #{String.pad_trailing(label, 40)} maxdiff=#{inspect(diff)}#{gpu}")
  end

  def softmax(x) do
    m = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    e = Nx.exp(Nx.subtract(x, m))
    Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
  end

  def relu(x), do: Nx.max(x, 0.0)
end

# =========================================================================
IO.puts("\n== 1. LeNet-style CNN forward pass (exercises the conv shader) ==")
# input 1x1x12x12 -> conv(4,1,3,3) -> relu -> maxpool 2x2 -> conv(8,4,3,3)
# -> relu -> flatten -> dense(72x10) -> softmax
cnn = fn backend ->
  x = H.t(H.rand(1 * 1 * 12 * 12, 1), {1, 1, 12, 12}, backend)
  k1 = H.t(H.rand(4 * 1 * 3 * 3, 2), {4, 1, 3, 3}, backend)
  k2 = H.t(H.rand(8 * 4 * 3 * 3, 3), {8, 4, 3, 3}, backend)
  w = H.t(H.rand(72 * 10, 4), {72, 10}, backend)

  conv1 = Nx.conv(x, k1)
  a1 = H.relu(conv1)
  p1 = Nx.window_max(a1, {1, 1, 2, 2}, strides: [1, 1, 2, 2])
  a2 = Nx.conv(p1, k2) |> H.relu()
  flat = Nx.reshape(a2, {1, 72})
  logits = Nx.dot(flat, w)
  {H.softmax(logits), conv1}
end

{probs_v, conv_out_v} = cnn.(VulkanoBackend)
{probs_b, _} = cnn.(Nx.BinaryBackend)
H.report("CNN class probabilities", H.maxdiff(probs_v, probs_b), match?(%VulkanoBackend{}, conv_out_v.data))
IO.puts("     probs = #{inspect(Nx.to_flat_list(probs_v) |> Enum.map(&Float.round(&1, 4)))}")

# =========================================================================
IO.puts("\n== 2. MLP training step with autodiff (matmul/relu/softmax + grad) ==")
# x(8,20) -> W1(20,16)+b1 -> relu -> W2(16,3)+b2 -> softmax; cross-entropy;
# one SGD step. Compare loss, gradient norm, and updated W1.
mlp_step = fn backend ->
  x = H.t(H.rand(8 * 20, 10), {8, 20}, backend)
  y = H.t(H.rand(8 * 3, 11), {8, 3}, backend) |> H.softmax()
  w1 = H.t(H.rand(20 * 16, 12), {20, 16}, backend)
  b1 = H.t(H.rand(16, 13), {16}, backend)
  w2 = H.t(H.rand(16 * 3, 14), {16, 3}, backend)
  b2 = H.t(H.rand(3, 15), {3}, backend)

  # All tensors are lifted as jit params so grad's closure captures no
  # non-default-backend tensors (Nx.Defn.grad requirement).
  step =
    Nx.Defn.jit(
      fn x, y, w1, b1, w2, b2 ->
        Nx.Defn.value_and_grad({w1, b1, w2, b2}, fn {w1, b1, w2, b2} ->
          h = Nx.add(Nx.dot(x, w1), b1) |> H.relu()
          logits = Nx.add(Nx.dot(h, w2), b2)
          probs = H.softmax(logits)
          Nx.mean(Nx.negate(Nx.sum(Nx.multiply(y, Nx.log(Nx.add(probs, 1.0e-9))), axes: [-1])))
        end)
      end,
      compiler: Nx.Defn.Evaluator
    )

  {loss, {gw1, _gb1, _gw2, _gb2} = grads} = step.(x, y, w1, b1, w2, b2)

  gnorm =
    grads
    |> Tuple.to_list()
    |> Enum.map(&(Nx.sum(Nx.pow(&1, 2)) |> Nx.to_number()))
    |> Enum.sum()

  w1_updated = Nx.subtract(w1, Nx.multiply(gw1, 0.1))
  {loss, gnorm, w1_updated}
end

{loss_v, gnorm_v, w1_v} = mlp_step.(VulkanoBackend)
{loss_b, gnorm_b, w1_b} = mlp_step.(Nx.BinaryBackend)
H.report("loss", abs(Nx.to_number(loss_v) - Nx.to_number(loss_b)))
H.report("grad L2 norm", abs(gnorm_v - gnorm_b))
H.report("W1 after one SGD step", H.maxdiff(w1_v, w1_b))
IO.puts("     loss(vulkan)=#{Float.round(Nx.to_number(loss_v), 6)}  loss(binary)=#{Float.round(Nx.to_number(loss_b), 6)}")

# =========================================================================
IO.puts("\n== 3. Logistic regression — 20 gradient-descent steps ==")
logreg = fn backend ->
  x = H.t(H.rand(30 * 5, 20), {30, 5}, backend)
  y = H.t(for(i <- 1..30, do: if(rem(i, 2) == 0, do: 1.0, else: 0.0)), {30, 1}, backend)
  w0 = H.t(List.duplicate(0.0, 5), {5, 1}, backend)

  Enum.reduce(1..20, w0, fn _, w ->
    p = Nx.sigmoid(Nx.dot(x, w))
    grad = Nx.dot(Nx.transpose(x), Nx.subtract(p, y)) |> Nx.divide(30.0)
    Nx.subtract(w, Nx.multiply(grad, 0.5))
  end)
end

H.report("weights after 20 GD steps", H.maxdiff(logreg.(VulkanoBackend), logreg.(Nx.BinaryBackend)))

# =========================================================================
IO.puts("\n== 4. FFT convolution theorem: ifft(fft(a) .* fft(b)) == circular conv ==")
# exercises the fft/ifft GPU path (real input -> c128, then complex-input ifft)
n = 8
a = H.rand(n, 30)
b = H.rand(n, 31)

# direct circular convolution reference (plain Elixir)
circ = for k <- 0..(n - 1) do
  Enum.reduce(0..(n - 1), 0.0, fn j, acc -> acc + Enum.at(a, j) * Enum.at(b, rem(k - j + n, n)) end)
end

fftconv = fn backend ->
  av = H.t(a, {n}, backend)
  bv = H.t(b, {n}, backend)
  Nx.ifft(Nx.multiply(Nx.fft(av), Nx.fft(bv))) |> Nx.real()
end

conv_v = fftconv.(VulkanoBackend)
conv_b = fftconv.(Nx.BinaryBackend)
circ_t = Nx.tensor(circ, type: {:f, 64}, backend: Nx.BinaryBackend)
H.report("fft-conv (vulkan vs binary)", H.maxdiff(conv_v, conv_b), match?(%VulkanoBackend{}, Nx.fft(H.t(a, {n}, VulkanoBackend)).data))
H.report("fft-conv vs direct circular", H.maxdiff(conv_v, circ_t))

IO.puts("")
