# f32 vs f64 race on VulkanoBackend, across the GPU compute families.
#
# Triggerable per host: run it on any Vulkan box (GPU or llvmpipe) to produce a
# labelled, comparable report. See scripts/race.sh for a one-command wrapper.
#
#   mix run examples/f32_vs_f64_race.exs
#
# Writes bench_results/f32_race_<host>_<commit>.json and prints a table + the
# device it ran on. On llvmpipe/CPU this mostly measures the memory-bandwidth /
# SIMD edge of f32 (~2x on bandwidth-bound ops); on real GPU hardware f64 is
# rate-limited to ~1/24 (GT 650M) .. 1/32 (consumer RTX) of f32, so the
# compute-bound ops (matmul/conv) gain far more there. Read CPU numbers as a
# lower bound.

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend

rnd = fn n, s -> for i <- 1..n, do: :math.sin(s * 0.7 + i * 0.013) end
tf = fn list, shape, type -> Nx.tensor(list, type: type, backend: VulkanoBackend) |> Nx.reshape(shape) end

# time one op (0-arity thunk) over `iters` runs, ms/op; also report on_gpu.
run = fn thunk, iters ->
  r = thunk.()
  on_gpu = match?(%VulkanoBackend{}, r.data)
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: thunk.() end)
  {us / iters / 1000.0, on_gpu}
end

# race one op-builder across dtypes; build.(type) -> 0-arity thunk. Returns a
# result map.
race = fn label, iters, build ->
  {f64ms, g64} = run.(build.({:f, 64}), iters)
  {f32ms, g32} = run.(build.({:f, 32}), iters)
  %{op: label, f64_ms: Float.round(f64ms, 3), f32_ms: Float.round(f32ms, 3),
    speedup: Float.round(f64ms / f32ms, 2), on_gpu: g64 and g32}
end

results = []

results = results ++
  for {m, k, n, iters} <- [{128, 128, 128, 20}, {256, 256, 256, 10}, {512, 512, 512, 4}] do
    race.("matmul #{m}x#{k}x#{n}", iters, fn type ->
      a = tf.(rnd.(m * k, 1), {m, k}, type)
      b = tf.(rnd.(k * n, 2), {k, n}, type)
      fn -> Nx.dot(a, b) end
    end)
  end

results = results ++
  for {ishape, kshape, iters} <- [
        {{1, 8, 24, 24}, {16, 8, 3, 3}, 10},
        {{1, 16, 32, 32}, {32, 16, 3, 3}, 4}
      ] do
    race.("conv #{elem(ishape, 1)}->#{elem(kshape, 0)}ch #{elem(ishape, 2)}sq", iters, fn type ->
      iv = tf.(rnd.(Tuple.product(ishape), 1), ishape, type)
      kv = tf.(rnd.(Tuple.product(kshape), 2), kshape, type)
      fn -> Nx.conv(iv, kv) end
    end)
  end

n_el = 1_000_000

results = results ++ [
  race.("elementwise add 1M", 20, fn type ->
    a = tf.(rnd.(n_el, 1), {n_el}, type)
    b = tf.(rnd.(n_el, 2), {n_el}, type)
    fn -> Nx.add(a, b) end
  end),
  race.("elementwise tanh 1M", 20, fn type ->
    a = tf.(rnd.(n_el, 3), {n_el}, type)
    fn -> Nx.tanh(a) end
  end),
  race.("sum 1M (full)", 20, fn type ->
    a = tf.(rnd.(n_el, 4), {n_el}, type)
    fn -> Nx.sum(a) end
  end),
  race.("sum 1024x1024 axis0", 20, fn type ->
    a = tf.(rnd.(1_048_576, 5), {1024, 1024}, type)
    fn -> Nx.sum(a, axes: [0]) end
  end)
]

# ---- metadata ----
{:ok, hostname} = :inet.gethostname()
hostname = to_string(hostname)

{device, dtype} =
  case Nx.Vulkan.NativeV.device_name() do
    {:ok, name, t} -> {name, t}
    _ -> {"unknown", "unknown"}
  end

commit =
  case System.cmd("git", ["rev-parse", "--short", "HEAD"], stderr_to_stdout: true) do
    {sha, 0} -> String.trim(sha)
    _ -> "unknown"
  end

report = %{
  hostname: hostname,
  device: device,
  device_type: dtype,
  commit: commit,
  timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
  otp: :erlang.system_info(:otp_release) |> to_string(),
  results: results
}

# ---- console table ----
IO.puts("\n  Device: #{device} (#{dtype})   host=#{hostname}   commit=#{commit}")
IO.puts("  op                        f64 ms     f32 ms   speedup   on_gpu")
IO.puts("  " <> String.duplicate("-", 66))

Enum.each(results, fn r ->
  IO.puts(
    "  #{String.pad_trailing(r.op, 24)}" <>
      "#{String.pad_leading(to_string(r.f64_ms), 9)}  " <>
      "#{String.pad_leading(to_string(r.f32_ms), 9)}   " <>
      "#{String.pad_leading(to_string(r.speedup), 5)}x   #{r.on_gpu}"
  )
end)

# ---- report artifact ----
File.mkdir_p!("bench_results")
path = "bench_results/f32_race_#{hostname}_#{commit}.json"
File.write!(path, Jason.encode!(report, pretty: true))
IO.puts("\n  wrote #{path}\n")
