# bench/stress.exs — push the wrapper until it cracks.
#
# Run with: mix run bench/stress.exs

defmodule Bench do
  def section(name) do
    IO.puts("\n#{IO.ANSI.bright()}=== #{name} ===#{IO.ANSI.reset()}")
  end

  def time(label, fun) do
    {us, result} = :timer.tc(fun)
    ms = us / 1000.0
    IO.puts("  #{String.pad_trailing(label, 50)} #{:io_lib.format("~9.3f", [ms])} ms")
    result
  end
end

:ok = Nx.Vulkan.init()
IO.puts("device: #{Nx.Vulkan.device_name()}")

# ---------- 1. RACE: concurrency under the SUBMIT_LOCK ----------
Bench.section("RACE: 20 procs × 10 ops (with mutex protection)")

Bench.time("20 procs × 10 ops × 1024-elem add", fn ->
  inputs = Enum.map(1..1024, fn i -> i / 1.0 end)

  tasks =
    for p <- 1..20 do
      Task.async(fn ->
        for _ <- 1..10 do
          {:ok, a} = Nx.Vulkan.upload_f32(inputs)
          {:ok, b} = Nx.Vulkan.upload_f32(inputs)
          {:ok, c} = Nx.Vulkan.add(a, b)
          {:ok, [head | _]} = Nx.Vulkan.download_f32(c, 1024)
          if head != 2.0, do: raise("data race! got #{head} in proc #{p}")
        end
        :ok
      end)
    end

  results = Enum.map(tasks, &Task.await(&1, 300_000))
  IO.puts("    completed: #{length(results)} procs, all :ok — no DEVICE_LOST")
end)

# ---------- 2. STRESS: alloc/free cycles ----------
Bench.section("STRESS: 5k alloc/free cycles (single tensor lifetime churn)")

Bench.time("5k allocations + GC", fn ->
  for _ <- 1..5_000 do
    {:ok, _t} = Nx.Vulkan.upload_f32([1.0])
  end
  :erlang.garbage_collect()
  :ok
end)

# ---------- 3. STRESS: progressively larger tensors ----------
Bench.section("STRESS: scale up tensor size to find OOM (8 GB VRAM ceiling)")

scales_mb = [1, 16, 64, 256, 1024, 2048]
for mb <- scales_mb do
  n = div(mb * 1024 * 1024, 4)
  data = :binary.copy(<<1.0::float-32-native>>, n)

  upload_us =
    case :timer.tc(fn -> Nx.Vulkan.Native.upload_binary(data) end) do
      {us, {:ok, t}} ->
        # add to itself + free
        {add_us, _} = :timer.tc(fn -> Nx.Vulkan.add(t, t) end)
        :erlang.garbage_collect()
        {us, add_us}

      {us, err} ->
        IO.puts("  #{mb} MB: upload returned #{inspect(err)} after #{us / 1000.0} ms")
        {us, 0}
    end

  case upload_us do
    {us, 0} -> :ok
    {upload_ms_us, add_us} ->
      IO.puts("  #{String.pad_leading(Integer.to_string(mb), 5)} MB: upload=#{:io_lib.format("~7.1f", [upload_ms_us / 1000.0])} ms  add=#{:io_lib.format("~7.1f", [add_us / 1000.0])} ms")
  end
end

# ---------- 4. SPEED: Nx.Vulkan vs Nx.BinaryBackend ----------
Bench.section("SPEED: Vulkan vs CPU (BinaryBackend) — Nx.add at varied sizes")

sizes = [1_024, 16_384, 262_144, 1_048_576, 4_194_304]

for n <- sizes do
  data = Enum.map(1..n, fn i -> rem(i, 100) * 1.0 end)

  t_cpu = Nx.tensor(data, backend: Nx.BinaryBackend)
  t_gpu = Nx.tensor(data, backend: Nx.Vulkan.Backend)

  cpu_us = :timer.tc(fn -> Enum.each(1..10, fn _ -> Nx.add(t_cpu, t_cpu) end) end) |> elem(0)
  gpu_us = :timer.tc(fn -> Enum.each(1..10, fn _ -> Nx.add(t_gpu, t_gpu) end) end) |> elem(0)

  cpu_ms = cpu_us / 10_000.0
  gpu_ms = gpu_us / 10_000.0
  speedup = if gpu_ms > 0, do: Float.round(cpu_ms / gpu_ms, 2), else: 0.0
  IO.puts("  N=#{String.pad_leading(Integer.to_string(n), 8)}  CPU=#{:io_lib.format("~9.3f", [cpu_ms])} ms  GPU=#{:io_lib.format("~9.3f", [gpu_ms])} ms  speedup=#{speedup}x")
end

# ---------- 5. BREAKAGE: invalid math ----------
Bench.section("BREAKAGE: GPU semantics for divide-by-zero, log(neg), sqrt(neg)")

{:ok, a}     = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
{:ok, zeros} = Nx.Vulkan.upload_f32([0.0, 0.0, 0.0])
{:ok, neg}   = Nx.Vulkan.upload_f32([-1.0, -4.0, -16.0])

{:ok, div_z}   = Nx.Vulkan.divide(a, zeros)
{:ok, log_neg} = Nx.Vulkan.log(neg)
{:ok, sqrt_neg} = Nx.Vulkan.sqrt(neg)

{:ok, dr}  = Nx.Vulkan.download_f32(div_z, 3)
{:ok, lnr} = Nx.Vulkan.download_f32(log_neg, 3)
{:ok, snr} = Nx.Vulkan.download_f32(sqrt_neg, 3)

IO.inspect(dr,  label: "  [1,2,3] / [0,0,0]")
IO.inspect(lnr, label: "  log([-1, -4, -16])")
IO.inspect(snr, label: "  sqrt([-1, -4, -16])")

# ---------- 6. STRESS: progressively larger matmul ----------
Bench.section("STRESS: matmul scale — find the GFLOPS ceiling")

for d <- [128, 256, 512, 1024, 2048] do
  n = d * d
  data = :binary.copy(<<1.0::float-32-native>>, n)

  with {:ok, a} <- Nx.Vulkan.Native.upload_binary(data),
       {:ok, b} <- Nx.Vulkan.Native.upload_binary(data) do
    {us, result} = :timer.tc(fn -> Nx.Vulkan.matmul(a, b, d, d, d) end)

    case result do
      {:ok, _c} ->
        flops = 2.0 * d * d * d
        gflops = flops / (us / 1_000_000.0) / 1.0e9
        IO.puts("  #{String.pad_leading(Integer.to_string(d), 5)}×#{d}: #{:io_lib.format("~9.3f", [us / 1000.0])} ms  #{Float.round(gflops, 1)} GFLOPS")

      err ->
        IO.puts("  #{d}×#{d}: FAILED → #{inspect(err)}")
    end
  end

  :erlang.garbage_collect()
end

IO.puts("\n#{IO.ANSI.bright()}=== bench complete — no device-lost ===#{IO.ANSI.reset()}")
