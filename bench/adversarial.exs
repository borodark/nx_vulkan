# bench/adversarial.exs — second-round breakage hunt.
#
# Round 1 fixed the queue race + NaN/Inf decoder.
# Round 2 goes after: VRAM exhaustion, leak detection, edge shapes,
# and process-death cleanup.
#
# Run with: mix run bench/adversarial.exs

defmodule Adv do
  def section(name) do
    IO.puts("\n#{IO.ANSI.bright()}=== #{name} ===#{IO.ANSI.reset()}")
  end

  def vram_used_mb do
    {out, 0} =
      System.cmd("nvidia-smi", ~w(--query-gpu=memory.used --format=csv,noheader,nounits))
    out |> String.trim() |> String.to_integer()
  rescue
    _ -> -1
  end
end

:ok = Nx.Vulkan.init()
IO.puts("device: #{Nx.Vulkan.device_name()}")
IO.puts("vram baseline: #{Adv.vram_used_mb()} MB")

# ---------- 1. EDGE SHAPES ----------
Adv.section("EDGE: empty tensor (N=0)")
result =
  try do
    Nx.Vulkan.upload_f32([])
  rescue
    e -> {:exception, Exception.message(e)}
  catch
    kind, val -> {:caught, kind, val}
  end

IO.inspect(result, label: "  upload_f32([])")

Adv.section("EDGE: single-element tensor (N=1)")
{:ok, t1} = Nx.Vulkan.upload_f32([42.0])
{:ok, [v]} = Nx.Vulkan.download_f32(t1, 1)
IO.puts("  N=1 round-trip: #{v}  #{if v == 42.0, do: "OK", else: "FAIL"}")

Adv.section("EDGE: off-by-one workgroup boundaries")
for n <- [255, 256, 257, 511, 512, 513, 1023, 1024, 1025] do
  data = Enum.map(1..n, fn i -> i / 1.0 end)
  {:ok, t} = Nx.Vulkan.upload_f32(data)
  {:ok, back} = Nx.Vulkan.download_f32(t, n)

  if back == data do
    IO.puts("  N=#{String.pad_leading(Integer.to_string(n), 5)}: OK")
  else
    IO.puts("  N=#{String.pad_leading(Integer.to_string(n), 5)}: FAIL — first mismatch at index #{Enum.find_index(0..(n-1), fn i -> Enum.at(back, i) != Enum.at(data, i) end)}")
  end
end

# ---------- 2. VRAM EXHAUSTION ----------
Adv.section("VRAM: push toward exhaustion (3060 Ti has 8 GB)")

# Hold each successive allocation so VRAM accumulates
held = []
held =
  Enum.reduce_while([512, 1024, 2048, 3072, 4096, 5120, 6144], held, fn mb, acc ->
    n = div(mb * 1024 * 1024, 4)
    data = :binary.copy(<<1.0::float-32-native>>, n)

    case Nx.Vulkan.Native.upload_binary(data) do
      {:ok, t} ->
        used = Adv.vram_used_mb()
        IO.puts("  +#{String.pad_leading(Integer.to_string(mb), 5)} MB → vram used: #{used} MB  (held: #{length(acc) + 1} buffers)")
        {:cont, [t | acc]}

      {:error, reason} ->
        IO.puts("  +#{mb} MB → FAIL #{inspect(reason)}  (cleanly returned, didn't crash)")
        {:halt, acc}
    end
  end)

held = nil   # let GC reclaim
:erlang.garbage_collect()
Process.sleep(500)
IO.puts("  after GC: vram used: #{Adv.vram_used_mb()} MB")

# ---------- 3. LEAK DETECTION: 30s sustained alloc/free ----------
Adv.section("LEAK: 30s of 1-MB alloc/free cycles, watch VRAM")

baseline = Adv.vram_used_mb()
IO.puts("  baseline before churn: #{baseline} MB")

deadline = System.monotonic_time(:millisecond) + 30_000
data = :binary.copy(<<1.0::float-32-native>>, div(1024 * 1024, 4))
count = :counters.new(1, [])

loop = fn loop ->
  if System.monotonic_time(:millisecond) < deadline do
    {:ok, _t} = Nx.Vulkan.Native.upload_binary(data)
    :counters.add(count, 1, 1)
    if rem(:counters.get(count, 1), 1_000) == 0 do
      :erlang.garbage_collect()
    end
    loop.(loop)
  end
end

loop.(loop)
:erlang.garbage_collect()
Process.sleep(500)

n = :counters.get(count, 1)
final = Adv.vram_used_mb()
delta = final - baseline
IO.puts("  performed #{n} alloc/free cycles in 30s")
IO.puts("  vram delta: #{delta} MB  (#{if abs(delta) < 100, do: "no leak", else: "POSSIBLE LEAK"})")

# ---------- 4. PROCESS DEATH MID-TENSOR ----------
Adv.section("PROC DEATH: kill a process holding a tensor; verify no leak")

baseline2 = Adv.vram_used_mb()
IO.puts("  baseline: #{baseline2} MB")

for cycle <- 1..200 do
  pid =
    spawn(fn ->
      # Hold ~10 MB
      data = :binary.copy(<<1.0::float-32-native>>, div(10 * 1024 * 1024, 4))
      {:ok, _t} = Nx.Vulkan.Native.upload_binary(data)
      receive do
        :die -> :ok
      end
    end)

  # Brief wait for the process to upload
  Process.sleep(2)
  Process.exit(pid, :kill)
end

:erlang.garbage_collect()
Process.sleep(1000)
final2 = Adv.vram_used_mb()
delta2 = final2 - baseline2
IO.puts("  200 procs created+killed mid-tensor")
IO.puts("  vram delta: #{delta2} MB  (#{if abs(delta2) < 200, do: "no leak", else: "POSSIBLE LEAK"})")

# ---------- 5. NIF UNDER LOAD: 50 procs × 100 quick ops ----------
Adv.section("CONCURRENCY: 50 procs × 100 ops (mutex serialization)")

t0 = System.monotonic_time(:millisecond)
inputs = Enum.map(1..1024, fn i -> i / 1.0 end)

tasks =
  for _p <- 1..50 do
    Task.async(fn ->
      for _ <- 1..100 do
        {:ok, a} = Nx.Vulkan.upload_f32(inputs)
        {:ok, b} = Nx.Vulkan.upload_f32(inputs)
        {:ok, _c} = Nx.Vulkan.add(a, b)
        :ok
      end
    end)
  end

Enum.each(tasks, &Task.await(&1, 600_000))
elapsed = System.monotonic_time(:millisecond) - t0
ops_per_sec = round(50 * 100 / (elapsed / 1000))
IO.puts("  50 × 100 = 5000 ops in #{elapsed} ms = #{ops_per_sec} ops/sec  (mutex-serialized)")

# ---------- 6. WRONG-SHAPE MATMUL ----------
Adv.section("BREAKAGE: matmul shape lies")

{:ok, a4} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
{:ok, b4} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])

# Promise 6-element shape from 4-element inputs
result = Nx.Vulkan.matmul(a4, b4, 2, 2, 3)
IO.inspect(result, label: "  matmul(A=2x3, B=3x2, but inputs only 4 elements each)")

# Wildly oversize shape
result2 = Nx.Vulkan.matmul(a4, b4, 1000, 1000, 1000)
IO.inspect(result2, label: "  matmul(A=1000x1000) with 4-elem inputs")

IO.puts("\n#{IO.ANSI.bright()}=== adversarial round complete ===#{IO.ANSI.reset()}")
IO.puts("device still alive: #{Nx.Vulkan.device_name()}")
