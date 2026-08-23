# W5 kernels vs the host fallbacks they replaced.
#
#   mix run examples/w5_kernels_race.exs
#
# W5 moved 265 doctests onto the GPU, measured by residency. Residency is not
# speed, and this repo has a retraction on the record for exactly that gap —
# 0.2.0 shipped a backward pass that was resident and ~250x SLOWER than the host
# (docs/BACKWARD_PASS_AUDIT.md). So: every family W5 added, raced against the
# path it replaced, on whatever box this runs on.
#
# BOTH ARMS END WITH THE ANSWER ON THE HOST, which is the only fair framing:
#
#   gpu  — resident inputs, native kernel, then backend_transfer to force a
#          flush and a full readback. Without that last step this measures
#          ENQUEUE cost and nothing else: dispatches batch up to NXV_BATCH_MAX
#          (64 by default) before the queue is submitted.
#   host — resident inputs, transferred down, computed on Nx.BinaryBackend,
#          result left there. This is precisely what the old fallback did.
#
# The host arm is if anything FLATTERED: a real fallback leaves its result on
# BinaryBackend, so everything downstream of it also runs there, uncounted here.
# The audit's headline finding was that fallback cost is dominated by the
# largest tensor rather than the count — 3 fallbacks to 1 took a LeNet step from
# 20 929 ms to 84 ms.
#
# SIZES ARE CHOSEN FOR THE HOST ARM, not the GPU one. Nx.BinaryBackend is an
# Elixir loop; a 65k-index scatter or a 1M-element argmax there costs seconds,
# and the first cut of this file was unrunnable because it multiplied that by
# iters and again by replicates. Where the host arm is pathologically slow the
# iteration count drops to 1-3 — which is exactly the regime where the GPU win
# is largest and least in doubt, so precision is cheapest to give up there.
#
# MEDIAN OF 5 REPLICATES, not a single run. mac-248 (GT 750M) runs ±11-13% and
# has already produced one retracted "hardware crossover" that was noise;
# mac-247 (GT 650M) is the quiet box at ±2-4%. Do not believe a 15% effect on
# 248 from one sample.

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend, as: V
alias Nx.BinaryBackend, as: B

{:ok, dev, kind} = Nx.Vulkan.NativeV.device_name()
IO.puts("\ndevice: #{dev} (#{kind})")

# Load average, sampled around every race. A box that is busy with someone
# else's work produces numbers that look like hardware findings and are not:
# a first run of this file on mac-247 reported `sum` at 251 ms while `argmax`
# and `all` — SAME NIF, same shape, same dispatch — came in at 0.9 and 2.4. No
# hardware story explains that, and the box had an eXMC build on it. Sampling
# the load makes contamination visible instead of leaving it to be argued about.
loadavg = fn ->
  case System.cmd("uptime", []) do
    {out, 0} ->
      case Regex.run(~r/average[s]?:\s*([0-9.]+)/, out) do
        [_, v] -> String.to_float(v)
        _ -> 0.0
      end

    _ ->
      0.0
  end
end

host = System.cmd("hostname", ["-s"]) |> elem(0) |> String.trim()
commit = System.cmd("git", ["rev-parse", "--short", "HEAD"]) |> elem(0) |> String.trim()

seq = fn n, s -> for i <- 1..n, do: rem(i * 7 + s, 97) - 48 end
fseq = fn n, s -> for i <- 1..n, do: :math.sin(s * 0.7 + i * 0.013) end

gt = fn list, shape, type -> Nx.tensor(list, type: type, backend: V) |> Nx.reshape(shape) end

# One timed sample: `iters` runs of `thunk`, ms/op.
sample = fn thunk, iters ->
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: thunk.() end)
  us / iters / 1000.0
end

median = fn xs ->
  s = Enum.sort(xs)
  n = length(s)
  if rem(n, 2) == 1, do: Enum.at(s, div(n, 2)), else: (Enum.at(s, div(n, 2) - 1) + Enum.at(s, div(n, 2))) / 2
end

# 5 replicates, median reported, spread kept so noise is visible rather than
# averaged away.
bench = fn thunk, iters ->
  thunk.()
  xs = for _ <- 1..5, do: sample.(thunk, iters)
  {median.(xs), Enum.min(xs), Enum.max(xs)}
end

race = fn label, iters, gpu_thunk, host_thunk ->
  IO.write("  #{label} ... ")
  load_before = loadavg.()
  {g, gmin, gmax} = bench.(gpu_thunk, iters)
  {h, hmin, hmax} = bench.(host_thunk, iters)
  load_after = loadavg.()
  spread = if g > 0, do: (gmax - gmin) / g * 100, else: 0.0
  IO.puts("gpu #{Float.round(g, 2)}ms  host #{Float.round(h, 2)}ms  #{Float.round(h / g, 2)}x  load #{load_before}->#{load_after}")

  %{
    op: label,
    gpu_ms: Float.round(g, 3),
    host_ms: Float.round(h, 3),
    speedup: Float.round(h / g, 2),
    gpu_spread_pct: Float.round(spread, 1),
    host_spread_pct: Float.round((hmax - hmin) / h * 100, 1),
    load_before: load_before,
    load_after: load_after
  }
end

# Every GPU arm ends in a readback; every host arm starts with a download.
gpu_of = fn f -> fn -> f.() |> Nx.backend_transfer(B) end end
host_of = fn tensors, f -> fn -> apply(f, Enum.map(tensors, &Nx.backend_copy(&1, B))) end end

results = []

# ---- integer elementwise binary (T1) ------------------------------------
results = results ++
  for {n, iters} <- [{4_096, 30}, {262_144, 10}, {1_048_576, 5}] do
    a = gt.(seq.(n, 1), {n}, {:s, 32})
    b = gt.(seq.(n, 2), {n}, {:s, 32})
    race.("add s32 n=#{n}", iters, gpu_of.(fn -> Nx.add(a, b) end), host_of.([a, b], &Nx.add/2))
  end

# ---- integer compare + select (T1) --------------------------------------
results = results ++
  for {n, iters} <- [{262_144, 10}] do
    a = gt.(seq.(n, 1), {n}, {:s, 32})
    b = gt.(seq.(n, 2), {n}, {:s, 32})
    results_cmp = race.("greater s32 n=#{n}", iters,
      gpu_of.(fn -> Nx.greater(a, b) end), host_of.([a, b], &Nx.greater/2))
    results_cmp
  end

results = results ++
  for {n, iters} <- [{262_144, 10}] do
    p = gt.(seq.(n, 3), {n}, {:s, 32}) |> Nx.greater(0)
    a = gt.(seq.(n, 1), {n}, {:s, 32})
    b = gt.(seq.(n, 2), {n}, {:s, 32})
    race.("select s32 n=#{n}", iters,
      gpu_of.(fn -> Nx.select(p, a, b) end),
      host_of.([p, a, b], &Nx.select/3))
  end

# ---- integer axis reduce (T2) -------------------------------------------
results = results ++
  for {rows, cols, iters} <- [{512, 512, 10}, {1024, 1024, 5}] do
    a = gt.(seq.(rows * cols, 1), {rows, cols}, {:s, 32})
    race.("sum s32 #{rows}x#{cols} axis 1", iters,
      gpu_of.(fn -> Nx.sum(a, axes: [1]) end),
      host_of.([a], fn t -> Nx.sum(t, axes: [1]) end))
  end

# ---- window reduce, including the padded gate (T2 + the gate fix) -------
results = results ++
  for {n, iters} <- [{262_144, 3}] do
    a = gt.(fseq.(n, 1), {512, 512}, {:f, 32})
    r1 = race.("window_sum f32 512x512 {3,3}", iters,
      gpu_of.(fn -> Nx.window_sum(a, {3, 3}) end),
      host_of.([a], fn t -> Nx.window_sum(t, {3, 3}) end))
    r1
  end

results = results ++
  for {iters} <- [{3}] do
    a = gt.(fseq.(262_144, 1), {512, 512}, {:f, 32})
    race.("window_sum f32 PADDED :same", iters,
      gpu_of.(fn -> Nx.window_sum(a, {3, 3}, padding: :same) end),
      host_of.([a], fn t -> Nx.window_sum(t, {3, 3}, padding: :same) end))
  end

# ---- scatter (indexed_put / indexed_add) --------------------------------
results = results ++
  for {n, k, iters} <- [{262_144, 8_192, 2}] do
    t = gt.(seq.(n, 1), {n}, {:s, 32})
    idx = gt.(for(i <- 1..k, do: rem(i * 37, n)), {k, 1}, {:s, 32})
    upd = gt.(seq.(k, 5), {k}, {:s, 32})
    r1 = race.("indexed_put s32 n=#{n} k=#{k}", iters,
      gpu_of.(fn -> Nx.indexed_put(t, idx, upd) end),
      host_of.([t, idx, upd], &Nx.indexed_put/3))
    r1
  end

results = results ++
  for {n, k, iters} <- [{262_144, 8_192, 2}] do
    t = gt.(seq.(n, 1), {n}, {:s, 32})
    idx = gt.(for(i <- 1..k, do: rem(i * 37, n)), {k, 1}, {:s, 32})
    upd = gt.(seq.(k, 5), {k}, {:s, 32})
    race.("indexed_add s32 n=#{n} k=#{k}", iters,
      gpu_of.(fn -> Nx.indexed_add(t, idx, upd) end),
      host_of.([t, idx, upd], &Nx.indexed_add/3))
  end

# ---- argmax / all -------------------------------------------------------
results = results ++
  for {rows, cols, iters} <- [{512, 512, 3}] do
    a = gt.(seq.(rows * cols, 1), {rows, cols}, {:s, 32})
    r1 = race.("argmax s32 #{rows}x#{cols} axis 1", iters,
      gpu_of.(fn -> Nx.argmax(a, axis: 1) end),
      host_of.([a], fn t -> Nx.argmax(t, axis: 1) end))
    r1
  end

results = results ++
  for {rows, cols, iters} <- [{512, 512, 3}] do
    a = gt.(seq.(rows * cols, 1), {rows, cols}, {:s, 32})
    race.("all s32 #{rows}x#{cols} axis 1", iters,
      gpu_of.(fn -> Nx.all(a, axes: [1]) end),
      host_of.([a], fn t -> Nx.all(t, axes: [1]) end))
  end

# ---- integer matmul (T3) ------------------------------------------------
results = results ++
  for {m, iters} <- [{128, 5}, {256, 3}, {512, 1}] do
    a = gt.(seq.(m * m, 1), {m, m}, {:s, 32})
    b = gt.(seq.(m * m, 2), {m, m}, {:s, 32})
    race.("dot s32 #{m}x#{m}", iters, gpu_of.(fn -> Nx.dot(a, b) end), host_of.([a, b], &Nx.dot/2))
  end

IO.puts("")
IO.puts(String.pad_trailing("op", 34) <> String.pad_leading("gpu ms", 10) <>
        String.pad_leading("host ms", 11) <> String.pad_leading("speedup", 10) <>
        String.pad_leading("gpu±%", 8) <> String.pad_leading("host±%", 8))
IO.puts(String.duplicate("-", 81))

busy = Enum.any?(results, &(&1.load_after > 1.5))

for r <- results do
  # A "regression" measured on a loaded box is not a finding. Say so on the row
  # rather than letting the word REGRESSION stand unqualified.
  flag = cond do
    r.speedup < 1.0 and r.load_after > 1.5 -> "  <-- slower, BUT load #{r.load_after} — RE-RUN IDLE"
    r.speedup < 1.0 -> "  <-- REGRESSION"
    r.gpu_spread_pct > 50.0 -> "  (noisy: ±#{r.gpu_spread_pct}%)"
    r.speedup < 1.5 -> "  (marginal)"
    true -> ""
  end

  IO.puts(String.pad_trailing(r.op, 34) <>
          String.pad_leading(Float.to_string(r.gpu_ms), 10) <>
          String.pad_leading(Float.to_string(r.host_ms), 11) <>
          String.pad_leading(Float.to_string(r.speedup) <> "x", 10) <>
          String.pad_leading(Float.to_string(r.gpu_spread_pct), 8) <>
          String.pad_leading(Float.to_string(r.host_spread_pct), 8) <> flag)
end

File.mkdir_p!("bench_results")
path = "bench_results/w5_race_#{host}_#{commit}.json"

File.write!(path, Jason.encode_to_iodata!(%{
  host: host, commit: commit, device: dev, device_kind: kind,
  replicates: 5, note: "median of 5; both arms end with the answer on the host",
  box_was_busy: busy,
  results: results
}, pretty: true))

if busy do
  IO.puts("\n*** THIS BOX WAS NOT IDLE. Load exceeded 1.5 during the run, so every")
  IO.puts("*** number above is suspect and any regression is unproven. Re-run when")
  IO.puts("*** `uptime` is quiet before drawing a conclusion from it.")
end

IO.puts("\nwrote #{path}")
