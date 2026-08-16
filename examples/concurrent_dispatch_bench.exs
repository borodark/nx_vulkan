# T1 under concurrency — the regime every batching number so far has skipped.
#
# `bench_results/BATCHED_DISPATCH.md` measures one training step in ONE BEAM
# process, on three hosts, and finds 1.45-1.71x. That number is real and this
# benchmark does not dispute it. But the pending queue that produces it is a
# single `OnceLock<Mutex<Vec<RecordFn>>>` static in the NIF — one queue per
# BEAM VM, shared by every process — and `submit_and_wait` ends in a
# device-wide `queue.wait_idle()`. Neither of those costs anything when there
# is one dispatcher. The workloads this backend is actually deployed into have
# many: exmc runs a GenServer per instrument (LIMITATIONS.md §7 puts the queue
# 67 jobs deep), and any Phoenix inference endpoint is N-concurrent by default.
#
# Three effects are predicted and none is measured:
#
#   1. The batch is a shared bucket. Two processes' graphs interleave into one
#      command buffer, so at NXV_BATCH_MAX=64 with 10 dispatchers no graph ever
#      gets a batch of its own. Batching's win should decay in N.
#   2. Convoy. Any process's readback flushes EVERYONE's pending work and waits
#      on the whole device. A scalar download should be able to block behind an
#      unrelated matmul, which shows up as a p95/p50 blowout, not a mean one.
#   3. Blast radius. A batch holds more descriptor sets alive at once; pool
#      pressure was the going-in concern for Kepler and it scales with N.
#
# This measures throughput and the latency tail against N, so the three can be
# told apart. It is deliberately agnostic about the answer: if throughput is
# flat in N, the shared bucket costs nothing and the GPU-node work in T1's
# follow-ups is unnecessary.
#
#     mix run examples/concurrent_dispatch_bench.exs
#     NXV_BATCH_MAX=0 mix run examples/concurrent_dispatch_bench.exs   # control
#     PROCS=1,2,4,8,16 REPS=20 mix run examples/concurrent_dispatch_bench.exs
#
# `NXV_BATCH_MAX` is read once per OS process (a `OnceLock` in the NIF), so the
# cap sweep MUST be separate `mix run` invocations — see scripts/concurrency_race.sh.
#
# Every guard in examples/mnist_mlp_step_bench.exs is kept here, and for the
# reasons recorded there: without the global default backend the graph silently
# runs on BinaryBackend; without residency assertions a single host fallback
# moves a 1039 ms matmul to the CPU while the counter reports `%{}`; without a
# NaN check a diverged model reports a flattering time for computing nothing.

alias Nx.Vulkan.VulkanoBackend

Nx.global_default_backend(VulkanoBackend)

procs_list =
  (System.get_env("PROCS") || "1,2,4,8,16")
  |> String.split(",", trim: true)
  |> Enum.map(&String.to_integer(String.trim(&1)))

reps = String.to_integer(System.get_env("REPS") || "20")
batch = 32

{:ok, dev_name, dev_kind} = Nx.Vulkan.NativeV.device_name()
batch_max = System.get_env("NXV_BATCH_MAX", "64 (default)")

IO.puts("""
=== T1 under concurrency ===
device        : #{dev_name} (#{dev_kind})
NXV_BATCH_MAX : #{batch_max}
schedulers    : #{System.schedulers_online()}
procs         : #{Enum.join(procs_list, ", ")}
reps/process  : #{reps}
""")

# --- the model, verbatim from mnist_mlp_step_bench.exs --------------------

resident! = fn t, label ->
  case t.data do
    %VulkanoBackend{} ->
      t

    other ->
      raise "#{label} is on #{inspect(other.__struct__)}, not the GPU — " <>
              "a host fallback upstream has silently moved this graph to the CPU"
  end
end

gen = fn shape, amp, jitter ->
  rank = tuple_size(shape)
  i0 = Nx.iota(shape, axis: 0, type: :f32, backend: VulkanoBackend)

  base =
    if rank > 1 do
      i1 = Nx.iota(shape, axis: rank - 1, type: :f32, backend: VulkanoBackend)
      Nx.subtract(Nx.multiply(i0, 0.017 + jitter), Nx.multiply(i1, 0.023))
    else
      Nx.multiply(i0, 0.017 + jitter)
    end

  base |> Nx.tanh() |> Nx.multiply(amp)
end

loss_fn = fn {w1, b1, w2, b2}, xx, yy ->
  h =
    xx
    |> Nx.reshape({batch, 784})
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.max(0.0)

  logits = h |> Nx.dot(w2) |> Nx.add(b2)

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

# Each worker builds its OWN parameters. Sharing one tensor across N processes
# would be a kinder workload than the deployment it stands in for — 67
# instruments are 67 distinct parameter sets, not one read 67 times — and it
# would also let the driver reuse residency in a way the real thing cannot.
# `jitter` keeps the tensors numerically distinct per worker.
build_workload = fn jitter ->
  x = resident!.(gen.({batch, 1, 28, 28}, 1.0, jitter), "x")

  y = gen.({batch, 10}, 1.0, jitter) |> Nx.abs() |> Nx.add(0.1)
  y = resident!.(Nx.divide(y, Nx.sum(y, axes: [1], keep_axes: true)), "y")

  params = {
    resident!.(gen.({784, 128}, 0.05, jitter), "w1"),
    resident!.(gen.({128}, 0.01, jitter), "b1"),
    resident!.(gen.({128, 10}, 0.1, jitter), "w2"),
    resident!.(gen.({10}, 0.01, jitter), "b2")
  }

  {params, x, y}
end

run_step = fn {params, x, y} ->
  {loss, _grads} = Nx.Defn.jit_apply(step, [params, x, y], compiler: Nx.Defn.Evaluator)
  # Reading the SCALAR loss back is what forces the pending batch to the GPU.
  # Without a readback this would time the *recording* of a step, and under
  # batching that charges a whole loop to whichever iteration trips the cap.
  #
  # Read back the loss and NOTHING ELSE. A `buf_download` calls `flush_pending`,
  # which submits every queued dispatch — the gradients included — and
  # `submit_and_wait` blocks on the whole command buffer, so the scalar is
  # sufficient to account for all the work. The first version of this benchmark
  # also transferred a {784,128} gradient and summed it on the host every step,
  # which added ~400 KB over PCIe plus a 100k-element BinaryBackend reduction to
  # each timed iteration. That is a large constant added equally to every arm,
  # and on a GT 750M it flattened NXV_BATCH_MAX=0/4/64 to within noise of each
  # other — i.e. it hid the very effect being measured, and disagreed with this
  # repo's own published 1.45x on the same box. If a future change makes the
  # timed path transfer anything but a scalar, that is a bug.
  Nx.to_number(Nx.backend_transfer(loss, Nx.BinaryBackend))
end

# --- residency + sanity, once, before any timing -------------------------

warm = build_workload.(0.0)
l0 = run_step.(warm)

if not is_number(l0) or l0 != l0 do
  raise "loss is not a finite number: #{inspect(l0)} — a diverged model runs fast and means nothing"
end

{_l, g0} = Nx.Defn.jit_apply(step, [elem(warm, 0), elem(warm, 1), elem(warm, 2)], compiler: Nx.Defn.Evaluator)
g0 |> Tuple.to_list() |> Enum.with_index(1) |> Enum.each(fn {g, i} -> resident!.(g, "grad #{i}") end)

{_out, counts} = Nx.Vulkan.Fallback.count(fn -> run_step.(warm) end)
IO.puts("loss: #{l0}")
IO.puts("host fallbacks in one step: #{Enum.sum(Map.values(counts))} #{inspect(counts)}\n")

# --- percentiles ---------------------------------------------------------

pct = fn sorted, p ->
  n = length(sorted)
  idx = min(n - 1, max(0, round(p * (n - 1))))
  Enum.at(sorted, idx)
end

fmt = fn f -> :erlang.float_to_binary(f * 1.0, decimals: 2) end

# --- the cohort ----------------------------------------------------------
#
# All N workers build and warm up FIRST, then block on a barrier, then start
# together. Without the barrier the early starters finish while the late ones
# are still compiling pipelines, and the result is staggered work reported as
# concurrent work — which would understate exactly the contention being
# measured.

run_cohort = fn n ->
  parent = self()

  tasks =
    for i <- 1..n do
      Task.async(fn ->
        w = build_workload.(i * 0.0003)
        # warm this worker's own pipelines/shader cache before the barrier
        _ = run_step.(w)

        send(parent, {:ready, self()})

        receive do
          :go -> :ok
        end

        times =
          for _ <- 1..reps do
            {us, _} = :timer.tc(fn -> run_step.(w) end)
            us / 1000
          end

        {times, :erlang.monotonic_time(:microsecond)}
      end)
    end

  for _ <- 1..n do
    receive do
      {:ready, _pid} -> :ok
    after
      120_000 -> raise "a worker never reported ready"
    end
  end

  t_start = :erlang.monotonic_time(:microsecond)
  Enum.each(tasks, fn t -> send(t.pid, :go) end)

  results = Task.await_many(tasks, :infinity)

  t_end = results |> Enum.map(&elem(&1, 1)) |> Enum.max()
  wall_ms = (t_end - t_start) / 1000

  all = results |> Enum.flat_map(&elem(&1, 0)) |> Enum.sort()
  total_steps = n * reps

  %{
    n: n,
    wall_ms: wall_ms,
    throughput: total_steps / (wall_ms / 1000),
    p50: pct.(all, 0.50),
    p95: pct.(all, 0.95),
    max: List.last(all),
    mean: Enum.sum(all) / length(all)
  }
end

IO.puts("--- cohort sweep ---\n")

IO.puts(
  "  #{String.pad_trailing("N", 4)} #{String.pad_leading("wall ms", 10)} " <>
    "#{String.pad_leading("steps/s", 9)} #{String.pad_leading("scaling", 8)} " <>
    "#{String.pad_leading("mean ms", 9)} #{String.pad_leading("p50", 8)} " <>
    "#{String.pad_leading("p95", 8)} #{String.pad_leading("p95/p50", 8)} #{String.pad_leading("max", 9)}"
)

base_tp = nil

{rows, _} =
  Enum.map_reduce(procs_list, base_tp, fn n, base ->
    r = run_cohort.(n)
    base = base || r.throughput
    scaling = r.throughput / base

    IO.puts(
      "  #{String.pad_trailing(Integer.to_string(n), 4)} " <>
        "#{String.pad_leading(fmt.(r.wall_ms), 10)} " <>
        "#{String.pad_leading(fmt.(r.throughput), 9)} " <>
        "#{String.pad_leading(fmt.(scaling) <> "x", 8)} " <>
        "#{String.pad_leading(fmt.(r.mean), 9)} " <>
        "#{String.pad_leading(fmt.(r.p50), 8)} " <>
        "#{String.pad_leading(fmt.(r.p95), 8)} " <>
        "#{String.pad_leading(fmt.(r.p95 / r.p50), 8)} " <>
        "#{String.pad_leading(fmt.(r.max), 9)}"
    )

    {Map.put(r, :scaling, scaling), base}
  end)

IO.puts("""

  scaling = throughput relative to N=#{hd(procs_list)}. Perfect serialisation is
  1.00x at every N (the device does the same total work, one graph at a time).
  Above 1.00x means concurrent dispatchers keep the GPU better fed than one
  does. Below 1.00x means contention is costing more than the overlap wins.

  p95/p50 is the convoy indicator: a shared flush makes a step's latency depend
  on unrelated work, which widens the tail without necessarily moving the mean.
""")

# --- machine-readable, for the fleet collector ---------------------------

json =
  Jason.encode!(%{
    device: dev_name,
    device_type: dev_kind,
    batch_max: to_string(batch_max),
    schedulers: System.schedulers_online(),
    reps: reps,
    batch: batch,
    loss: l0,
    fallbacks: Enum.sum(Map.values(counts)),
    rows: rows
  })

out =
  System.get_env("OUT") ||
    Path.join(
      "bench_results",
      "concurrency_#{String.replace(dev_name, ~r/[^A-Za-z0-9]+/, "_")}_cap#{String.replace(to_string(batch_max), ~r/[^0-9]/, "")}.json"
    )

File.mkdir_p!(Path.dirname(out))
File.write!(out, json)
IO.puts("wrote #{out}")
