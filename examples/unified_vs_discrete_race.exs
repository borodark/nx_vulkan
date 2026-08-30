# unified_vs_discrete_race.exs — Races 1 and 4 of PLAN_UNIFIED_VS_DISCRETE_RACE.md
#
#   mix run examples/unified_vs_discrete_race.exs
#
# ONE tracked harness, run identically on every box. That is the point: the last
# time this project let each box write its own probe script in /tmp, two of the
# three schemes could not detect the thing they were testing and nobody could
# compare them because they were never in the same place.
#
# WHAT IS BEING ASKED. Not "which box is faster" — an RTX 3060 Ti beats a Tegra
# X1 at 5W and no benchmark is needed to know it. The question is whether unified
# memory changes the SHAPE of the cost curve.
#
# So every number is reported as a RATIO TO THIS BOX'S OWN BASELINE, never as a
# cross-box absolute. That is what makes a Jetson measurement defensible at all:
# its OTP is built --disable-jit and it runs 2 of 4 cores at 5W, so any absolute
# time is partly measuring a crippled host. A ratio between two measurements on
# the SAME box cancels that.
#
# TIMER DISCIPLINE (plan rules 1-6, and they are not optional):
#   * inputs built on the device, OUTSIDE the timer
#   * NativeV.flush() INSIDE the timer — dispatch is batched, so without this
#     the timer measures enqueue, not work
#   * garbage_collect OUTSIDE the timer
#   * medians of >= 9, plus min and max, because a mean hides throttling
#   * uptime before AND after; this project has withdrawn one contended table
#   * a thermal control: the first measurement is repeated last, and if the two
#     disagree by more than 10% the run is declared void rather than reported

alias Nx.Vulkan.VulkanoBackend, as: VB
alias Nx.Vulkan.NativeV

reps = String.to_integer(System.get_env("NXV_RACE_REPS") || "9")

box =
  case :inet.gethostname() do
    {:ok, h} -> to_string(h)
    _ -> "unknown"
  end

device =
  case NativeV.device_name() do
    {:ok, name, type} -> "#{name} (#{type})"
    other -> inspect(other)
  end

load = fn ->
  case System.cmd("uptime", [], stderr_to_stdout: true) do
    {out, 0} -> out |> String.trim() |> String.split("load average") |> List.last()
    _ -> "unavailable"
  end
end

median = fn xs ->
  s = Enum.sort(xs)
  n = length(s)

  if rem(n, 2) == 1,
    do: Enum.at(s, div(n, 2)),
    else: (Enum.at(s, div(n, 2) - 1) + Enum.at(s, div(n, 2))) / 2
end

# Time `fun` `reps` times. The flush is INSIDE; the gc is OUTSIDE.
# A SINGLE untimed warmup is not enough, and assuming it is would have wrecked
# this whole race. A modern discrete GPU idles its clocks down hard: super-io's
# RTX 3060 Ti sits at 210 MHz against a 2100 MHz maximum — 10% of speed — and
# boosts only under sustained load. So an op measured cold runs at a tenth of
# the clock an op measured warm gets, and the difference has nothing to do with
# memory architecture.
#
# That produced a 288% "thermal drift" on an idle box whose load had not moved:
# Race 1 left the GPU boosted, allocation-bound Race 4 let it fall back to idle,
# and the repeat measurement caught it cold. Read naively that is throttling.
# It is the opposite — the box is FASTER when busy, not slower.
#
# The Jetson runs nvpmodel 5W with far less headroom to boost, so an
# uncontrolled comparison would show it winning short bursts purely because the
# Ampere was idling. That is precisely the false "unified memory changes the
# cost curve" finding this plan exists to avoid.
#
# So: warm until the clock has had time to ramp, not just once.
measure = fn fun ->
  warm_until = System.monotonic_time(:millisecond) + 250

  Stream.repeatedly(fn ->
    _ = fun.()
    :ok = NativeV.flush()
  end)
  |> Enum.take_while(fn _ -> System.monotonic_time(:millisecond) < warm_until end)

  samples =
    for _ <- 1..reps do
      :erlang.garbage_collect()
      t0 = System.monotonic_time(:microsecond)
      _ = fun.()
      :ok = NativeV.flush()
      t1 = System.monotonic_time(:microsecond)
      (t1 - t0) / 1000.0
    end

  %{median: median.(samples), min: Enum.min(samples), max: Enum.max(samples)}
end

IO.puts("\n=== unified vs discrete: Races 1 and 4 ===")
IO.puts("box:    #{box}")
IO.puts("device: #{device}")
IO.puts("reps:   #{reps}")

gpu_clock = fn ->
  case System.cmd(
         "nvidia-smi",
         ["--query-gpu=clocks.sm,clocks.max.sm,utilization.gpu", "--format=csv,noheader"],
         stderr_to_stdout: true
       ) do
    {out, 0} -> String.trim(out)
    _ -> "n/a (no nvidia-smi)"
  end
end

IO.puts("load before: #{load.()}")
IO.puts("gpu clock before: #{gpu_clock.()}")

# ---------------------------------------------------------------------------
# RACE 1 — arithmetic intensity sweep.
#
# {n,k} x {k,n} does 2*n^2*k FLOPs and moves ~(2nk + n^2) elements. Sweeping k
# at fixed n walks arithmetic intensity across an order of magnitude WITHOUT
# changing the kernel — so any shape change in the curve is about data
# movement, not about a different code path being selected.
# ---------------------------------------------------------------------------

n = 512
ks = [4, 16, 64, 256, 1024]

IO.puts("\n--- Race 1: arithmetic intensity (n = #{n}, f32 matmul) ---")
IO.puts("     k      ms    GFLOP/s     GB/s")

race1 =
  for k <- ks do
    # built on the device, outside the timer
    a = Nx.iota({n, k}, type: {:f, 32}, backend: VB)
    b = Nx.iota({k, n}, type: {:f, 32}, backend: VB)
    :ok = NativeV.flush()

    m = measure.(fn -> Nx.dot(a, b) end)

    flops = 2 * n * n * k
    bytes = (2 * n * k + n * n) * 4
    gflops = flops / (m.median / 1000.0) / 1.0e9
    gbs = bytes / (m.median / 1000.0) / 1.0e9

    IO.puts(
      "  #{String.pad_leading("#{k}", 4)}  #{:erlang.float_to_binary(m.median, decimals: 3)}" <>
        "  #{:erlang.float_to_binary(gflops, decimals: 2)}" <>
        "  #{:erlang.float_to_binary(gbs, decimals: 3)}"
    )

    %{k: k, ms: m.median, ms_min: m.min, ms_max: m.max, gflops: gflops, gbs: gbs}
  end

# Normalised to this box's own k = 1024 point. THIS is the comparable number.
base1 = Enum.find(race1, &(&1.k == 1024)).gflops

IO.puts("\n  normalised to this box's own k=1024 GFLOP/s (the cross-box comparable):")

race1_norm =
  for r <- race1 do
    rel = r.gflops / base1

    IO.puts(
      "    k=#{String.pad_leading("#{r.k}", 4)}  #{:erlang.float_to_binary(rel, decimals: 4)}"
    )

    Map.put(r, :rel_to_k1024, rel)
  end

# ---------------------------------------------------------------------------
# RACE 4 — the allocation cliff's SLOPE.
#
# Four boxes have already settled that the cliff exists at 32 MiB and is
# vulkano's dedicated-allocation threshold, not a memory-architecture artifact
# (poison_control, and NEXT.md 1.4a). What is NOT settled is whether the
# POST-cliff slope differs — which is what would decide whether large outputs
# should be chunked below the cliff on one box and not another.
# ---------------------------------------------------------------------------

IO.puts("\n--- Race 4: allocation cliff slope (24-40 MiB, 2 MiB steps) ---")
IO.puts("     MiB   alloc ms  zeroed ms")

race4 =
  for mib <- 24..40//2 do
    bytes = mib * 1024 * 1024
    a = measure.(fn -> {:ok, _} = NativeV.buf_alloc(bytes) end)
    :erlang.garbage_collect()
    z = measure.(fn -> {:ok, _} = NativeV.buf_alloc_zeroed(bytes) end)
    :erlang.garbage_collect()

    IO.puts(
      "  #{String.pad_leading("#{mib}", 4)}   #{String.pad_leading(:erlang.float_to_binary(a.median, decimals: 3), 8)}" <>
        "   #{String.pad_leading(:erlang.float_to_binary(z.median, decimals: 3), 8)}"
    )

    %{mib: mib, alloc_ms: a.median, zeroed_ms: z.median}
  end

below = Enum.filter(race4, &(&1.mib < 32))
above = Enum.filter(race4, &(&1.mib >= 32))

slope = fn rows, key ->
  case rows do
    [] ->
      0.0

    _ ->
      {lo, hi} = {List.first(rows), List.last(rows)}
      d = hi.mib - lo.mib
      if d == 0, do: 0.0, else: (Map.get(hi, key) - Map.get(lo, key)) / d
  end
end

slopes = %{
  alloc_below: slope.(below, :alloc_ms),
  alloc_above: slope.(above, :alloc_ms),
  zeroed_below: slope.(below, :zeroed_ms),
  zeroed_above: slope.(above, :zeroed_ms)
}

IO.puts("\n  ms per MiB:")

IO.puts(
  "    buf_alloc         below cliff #{:erlang.float_to_binary(slopes.alloc_below, decimals: 4)}  above #{:erlang.float_to_binary(slopes.alloc_above, decimals: 4)}"
)

IO.puts(
  "    buf_alloc_zeroed  below cliff #{:erlang.float_to_binary(slopes.zeroed_below, decimals: 4)}  above #{:erlang.float_to_binary(slopes.zeroed_above, decimals: 4)}"
)

# ---------------------------------------------------------------------------
# THERMAL CONTROL — repeat Race 1's first point. If the box throttled under the
# sustained load above, this disagrees with the original and the run is VOID.
# Reporting a throttled curve as a memory-architecture finding is exactly the
# failure this project has already published once.
# ---------------------------------------------------------------------------

# Anchor the control on the LARGEST k, not the first one. k=4 is ~0.8 ms of
# which most is the ~170 us submission floor plus pipeline overhead — it is the
# noisiest point in the sweep, and Race 4 has just churned the allocator with
# 40 MiB requests right before it. Anchoring there produced a 95.7% "drift" on
# an idle 88-core box whose load had not moved (2.63 -> 2.66), i.e. a false
# VOID. The largest k is compute-dominated, which is where sustained-load
# throttling would actually show, and it is the least sensitive to dispatch
# jitter.
# Race 4 just allocated ~180 buffers of 24-40 MiB. They are unreachable, but a
# Rustler resource is freed on BEAM GC, so without forcing one the driver is
# still holding several GB of them when the control runs. That looks EXACTLY
# like thermal throttling and is not: it is this harness's own garbage. Reclaim
# it before measuring, or the control reports the script's litter as the box's
# condition.
Enum.each(:erlang.processes(), &:erlang.garbage_collect/1)
:ok = NativeV.flush()

k0 = List.last(ks)
a0 = Nx.iota({n, k0}, type: {:f, 32}, backend: VB)
b0 = Nx.iota({k0, n}, type: {:f, 32}, backend: VB)
:ok = NativeV.flush()
control = measure.(fn -> Nx.dot(a0, b0) end)

first = Enum.find(race1, &(&1.k == k0)).ms
drift = abs(control.median - first) / first

IO.puts("\n--- Thermal control (k = #{k0}, the compute-dominated point, repeated last) ---")

IO.puts(
  "  first: #{:erlang.float_to_binary(first, decimals: 3)} ms   last: #{:erlang.float_to_binary(control.median, decimals: 3)} ms"
)

IO.puts("  drift: #{:erlang.float_to_binary(drift * 100, decimals: 1)}%")

void? = drift > 0.10

if void?,
  do: IO.puts("  *** DRIFT > 10% — THIS RUN IS VOID, the box throttled or was contended ***")

load_after = load.()
clock_after = gpu_clock.()
IO.puts("\nload after: #{load_after}")
IO.puts("gpu clock after: #{clock_after}")

path = "bench_results/unified_vs_discrete_#{box}.json"
File.mkdir_p!("bench_results")

File.write!(
  path,
  Jason.encode_to_iodata!(
    %{
      box: box,
      device: device,
      reps: reps,
      race1: race1_norm,
      race1_normalised_to: "own k=1024 gflops",
      race4: race4,
      race4_slopes_ms_per_mib: slopes,
      thermal_control: %{first_ms: first, last_ms: control.median, drift: drift, void: void?},
      load_after: String.trim(load_after),
      gpu_clock_after: clock_after
    },
    pretty: true
  )
)

IO.puts("wrote #{path}")
IO.puts(if void?, do: "\nRACE: VOID (thermal drift)", else: "\nRACE: OK")
