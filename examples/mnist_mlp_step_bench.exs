# T1 — batched command submission, measured on the graph that motivated it.
#
# `bench_results/MNIST_EXLA_RACE.md` timed one `value_and_grad` training step of
# the Axon MNIST MLP at 14.140 ms eager on super-io against EXLA's 0.715 ms, and
# showed that whole-graph fusion makes it WORSE (0.76x). An optimisation that
# removes work from the shaders cannot explain a gap that fusing the shaders
# widens — so the deficit is per-dispatch cost, which is what batching attacks.
#
# The model is the Axon one written out in plain Nx (Axon is not a dependency
# of this repo, and the race harness that used it was a scratch project):
#
#     input {batch, 1, 28, 28} -> flatten -> dense 128 + relu
#                              -> dense 10 + softmax -> categorical cross-entropy
#
# A/B against the control, which is the pre-batching submit-per-dispatch path:
#
#     mix run examples/mnist_mlp_step_bench.exs                  # batched
#     NXV_BATCH_MAX=0 mix run examples/mnist_mlp_step_bench.exs  # control
#     NXV_BATCH_MAX=16 mix run examples/mnist_mlp_step_bench.exs # sweep the cap
#
# `NXV_BATCH_MAX` is read once per OS process, so the arms must be separate
# `mix run` invocations — you cannot sweep it from inside one script.

alias Nx.Vulkan.VulkanoBackend

# Without this, every tensor `defn` materialises internally — grad constants,
# the scalar in `Nx.max(h, 0.0)` — lands on Nx.BinaryBackend and drags the
# graph to the CPU. The first version of this benchmark omitted it and
# measured 6.8 s per step, which is the BinaryBackend row of the race table.
Nx.global_default_backend(VulkanoBackend)

{:ok, name, kind} = Nx.Vulkan.NativeV.device_name()
batch_max = System.get_env("NXV_BATCH_MAX", "64 (default)")
IO.puts("device: #{name} (#{kind})")
IO.puts("NXV_BATCH_MAX: #{batch_max}\n")

batch = 32

# A tensor that is *on* the GPU is not the same as an op that *ran* on it, and
# neither is visible in a value assertion. Worse, residency is sticky: one
# host fallback moves its result to BinaryBackend and everything downstream
# computes there WITHOUT being recorded, because the fallback counter only
# sees ops that reach this backend. The second version of this benchmark built
# its inputs with `Nx.sin` — a single unsupported unary, one recorded fallback
# — and the 32x784x128 matmul underneath it then took 1039 ms on the CPU while
# the counter reported `%{}`. Assert residency of every input explicitly.
resident! = fn t, label ->
  case t.data do
    %VulkanoBackend{} ->
      t

    other ->
      raise "#{label} is on #{inspect(other.__struct__)}, not the GPU — " <>
              "a host fallback upstream has silently moved this graph to the CPU"
  end
end

# Deterministic inputs. `tanh` is one of the supported unary op codes, so it
# stays on-device; `sin`/`cos` are not, and would host-fall-back.
#
# Two iotas along different axes rather than one flat ramp, so the weights are
# not rank-degenerate, and `amp` scales AFTER the tanh — feeding a large ramp
# straight into tanh saturates every element to 1.0, which makes the softmax
# underflow and the loss NaN. (The guard below caught exactly that.)
gen = fn shape, amp ->
  rank = tuple_size(shape)
  i0 = Nx.iota(shape, axis: 0, type: :f32, backend: VulkanoBackend)

  base =
    if rank > 1 do
      i1 = Nx.iota(shape, axis: rank - 1, type: :f32, backend: VulkanoBackend)
      Nx.subtract(Nx.multiply(i0, 0.017), Nx.multiply(i1, 0.023))
    else
      Nx.multiply(i0, 0.017)
    end

  base |> Nx.tanh() |> Nx.multiply(amp)
end

x = resident!.(gen.({batch, 1, 28, 28}, 1.0), "x")

# A valid probability target: strictly positive, rows summing to 1.
y = gen.({batch, 10}, 1.0) |> Nx.abs() |> Nx.add(0.1)
y = resident!.(Nx.divide(y, Nx.sum(y, axes: [1], keep_axes: true)), "y")

# Amplitudes ~ 1/sqrt(fan_in) so activations and logits stay O(1).
params = {
  resident!.(gen.({784, 128}, 0.05), "w1"),
  resident!.(gen.({128}, 0.01), "b1"),
  resident!.(gen.({128, 10}, 0.1), "w2"),
  resident!.(gen.({10}, 0.01), "b2")
}

loss_fn = fn {w1, b1, w2, b2}, xx, yy ->
  h =
    xx
    |> Nx.reshape({batch, 784})
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.max(0.0)

  logits = h |> Nx.dot(w2) |> Nx.add(b2)

  # softmax, shifted by the row max for stability
  shifted = Nx.subtract(logits, Nx.reduce_max(logits, axes: [1], keep_axes: true))
  exps = Nx.exp(shifted)
  probs = Nx.divide(exps, Nx.sum(exps, axes: [1], keep_axes: true))

  yy
  |> Nx.multiply(Nx.log(probs))
  |> Nx.sum(axes: [1])
  |> Nx.negate()
  |> Nx.mean()
end

step = fn p, xx, yy -> Nx.Defn.value_and_grad(p, &loss_fn.(&1, xx, yy)) end

run = fn ->
  {loss, grads} = Nx.Defn.jit_apply(step, [params, x, y], compiler: Nx.Defn.Evaluator)
  # Reading the loss back is what forces the batch to the GPU. Without a
  # readback, deferred dispatch would time the *recording* of the step.
  {Nx.to_number(Nx.backend_transfer(loss, Nx.BinaryBackend)), grads}
end

# The gradients are the bulk of the step; if they were computed on the host the
# timing below would be measuring BinaryBackend, not dispatch cost.
{_l0, g0} = Nx.Defn.jit_apply(step, [params, x, y], compiler: Nx.Defn.Evaluator)
g0 |> Tuple.to_list() |> Enum.with_index(1) |> Enum.each(fn {g, i} -> resident!.(g, "grad #{i}") end)

{loss, grads} = run.()

# A diverged model runs fast — a 635x figure in this repo's history came from a
# model producing NaN. Refuse to report a timing without checking.
if not is_number(loss) or loss != loss do
  raise "loss is not a finite number: #{inspect(loss)}"
end

# Force every gradient to the host too, so nothing in the step is quietly
# unmeasured because its result was never read.
{g1, _, _, _} = grads
_ = Nx.to_number(Nx.sum(Nx.backend_transfer(g1, Nx.BinaryBackend)))

IO.puts("loss: #{loss}")

{fallbacks, counts} =
  case Code.ensure_loaded(Nx.Vulkan.Fallback) do
    {:module, _} ->
      {_out, c} = Nx.Vulkan.Fallback.count(fn -> run.() end)
      {Enum.sum(Map.values(c)), c}

    _ ->
      {:unknown, %{}}
  end

IO.puts("host fallbacks in one step: #{inspect(fallbacks)} #{inspect(counts)}\n")

reps = 20

times =
  for _ <- 1..5 do
    {us, _} = :timer.tc(fn -> for _ <- 1..reps, do: run.() end)
    us / reps / 1000
  end

IO.puts("per-step ms (best of 5 x #{reps}): #{Float.round(Enum.min(times), 3)}")
IO.puts("            median of the 5 runs: #{Float.round(Enum.sort(times) |> Enum.at(2), 3)}")
IO.puts("            all: #{inspect(Enum.map(times, &Float.round(&1, 3)))}")
