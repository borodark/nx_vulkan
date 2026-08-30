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
    {out, 0} ->
      case Regex.run(~r/load averages?:\s*(.*)$/, String.trim(out)) do
        [_, nums] -> String.trim(nums)
        _ -> String.trim(out)
      end

    _ ->
      "unavailable"
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
# THE WARM LOOP MUST DROP WHAT IT ALLOCATES. `buf_alloc` returns a Rustler
# resource freed only when the owning process is GC'd. Binding it to `_` makes
# it garbage instantly, but garbage is not freed until a collection runs — and a
# magic ref is a few words on the BEAM heap, so the VM feels no pressure while
# the driver holds gigabytes.
#
# Without the collect below, this loop killed Race 4 outright on both Keplers
# with "alloc buffer: a non-validation error occurred".
#
# The iteration counts are worth stating carefully, because the obvious reading
# is wrong. The BROKEN loop managed only ~40 iterations in 250 ms — but not
# because each one was slow in itself. Retained allocations degrade the next
# allocation, so the loop grinds to a halt and then fails. Remove the leak and
# the SAME loop, still flushing, does ~22,000 iterations in the same 250 ms.
# `flush` is not the cost: `buf_alloc` enqueues no GPU work for it to drain, and
# it measures 7.5 us on the Ampere and 3 us on a Kepler.
#
# It is NOT simple VRAM exhaustion, and it is worth saying so because that was
# the first diagnosis and it was wrong. Both boxes failed at the identical
# retained allocation (#49) despite having 981 MiB and 1999 MiB of VRAM, and
# 48 x 26 MiB = 1.22 GiB already exceeds the smaller card entirely. So the
# ceiling does not scale with capacity: it is an allocator/driver bookkeeping
# limit, plausibly because uninitialised `buf_alloc` pages never commit. The GC
# fixes it either way, but do not reason about it as a memory-pressure problem.
#
# The subtle part, and why this was never merely a small-card bug: the retained
# footprint grows with the buffer size being measured, so the self-inflicted
# pressure is COLLINEAR with Race 4's independent variable. On an 8 GB card it
# is absorbed silently and the run still reports RACE: OK — a contaminated
# slope that looks clean. Every Race 4 number taken before this fix is void,
# super-io's included.
measure = fn fun ->
  # 600 ms and at least 3 iterations. The Tegra needs ~500 ms to reach its boost
  # state, and a single k=1024 dot there takes 365 ms — so the old 250 ms budget
  # ran exactly ONE warm iteration on the box that most needed warming, and
  # crossed the boost threshold only because that one cold dot happened to take
  # 698 ms. Marginal by luck, not by design.
  warm_until = System.monotonic_time(:millisecond) + 600

  warm = fn self, iters ->
    if System.monotonic_time(:millisecond) < warm_until or iters < 3 do
      _ = fun.()
      :ok = NativeV.flush()
      :erlang.garbage_collect()
      self.(self, iters + 1)
    end
  end

  warm.(warm, 0)

  # TWO timers per rep, and the split is the point.
  #
  # `enqueue` is everything before the queue is submitted: the Nx frontend,
  # shape and type checks, buffer binding, command recording — all HOST work.
  # `total` adds the flush, i.e. the GPU actually doing it.
  #
  # This exists because the plan's rule 6 was wrong, and it was my rule. It said
  # normalising to the box's own baseline cancels the crippled host. That holds
  # only for a MULTIPLICATIVE host factor. Measured on the Jetson, host cost is
  # roughly CONSTANT and ADDITIVE (~3-6 ms regardless of k), so it does not
  # cancel in a ratio at all — it depresses the small-k points far more than
  # k=1024, in proportion to how slow the host is: 28-30% of measured time at
  # k=4 on a --disable-jit 5W box against 1.6% at k=1024, and a fraction of a
  # percent on a fast host. The two leftmost points were not cross-box
  # comparable even after normalising.
  #
  # Direction matters: unified memory should make the Jetson look relatively
  # BETTER at small k, while this confound pushes it WORSE there. So it biases
  # against the hypothesis — a positive effect would be a lower bound, but a
  # null at small k would have been uninterpretable.
  samples =
    for _ <- 1..reps do
      :erlang.garbage_collect()
      t0 = System.monotonic_time(:microsecond)
      _ = fun.()
      t1 = System.monotonic_time(:microsecond)
      :ok = NativeV.flush()
      t2 = System.monotonic_time(:microsecond)
      # Per-rep GPU time is t2-t1, kept as its own sample. Do NOT reconstruct it
      # as median(total) - median(enqueue): that is a difference of medians, and
      # host time here is BIMODAL — mac-247 measured clean modes at ~0.61 ms and
      # ~1.23 ms, almost exactly 2x apart, with the median landing on whichever
      # dominates a given sample and flipping between runs and between k. A
      # difference of medians can then be off by the whole 0.6 ms mode gap,
      # which at k=4 is 50% of the GPU time — precisely where it matters most.
      # Taking the median of the per-rep differences removes the exposure for
      # free, and is what the two quantities were always meant to be.
      {(t1 - t0) / 1000.0, (t2 - t0) / 1000.0, (t2 - t1) / 1000.0}
    end

  enq = Enum.map(samples, &elem(&1, 0))
  tot = Enum.map(samples, &elem(&1, 1))
  gpu = Enum.map(samples, &elem(&1, 2))

  %{
    median: median.(tot),
    min: Enum.min(tot),
    max: Enum.max(tot),
    enqueue: median.(enq),
    # Reported alongside the median because the host distribution is bimodal:
    # a single host% figure is not "value +/- noise", it is one of two modes.
    # mac-247's k=4 host share is properly 33-50% depending on which one lands.
    enqueue_min: Enum.min(enq),
    gpu: median.(gpu)
  }
end

IO.puts("\n=== unified vs discrete: Races 1 and 4 ===")
IO.puts("box:    #{box}")
IO.puts("device: #{device}")
IO.puts("reps:   #{reps}")

gpu_clock = fn ->
  case System.cmd(
         "nvidia-smi",
         # Kepler on FreeBSD HAS nvidia-smi but reports [N/A] for clocks.sm, so
         # the no-nvidia-smi branch never fires and the output is a row of
         # [N/A]. Ask for temperature and pstate too: those boxes do report
         # them, and they are a usable throttle proxy when clocks are absent.
         [
           "--query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu,pstate",
           "--format=csv,noheader"
         ],
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
# Extra points below 64. The quantity that matters turned out to be the
# INTERCEPT of ms = a + b*k (see the fit below), and a 5-point sweep with only
# two points under 64 estimates it from almost nothing.
ks = [1, 2, 4, 8, 16, 32, 64, 256, 1024]

IO.puts("\n--- Race 1: arithmetic intensity (n = #{n}, f32 matmul) ---")
IO.puts("     k   total_ms  host_ms   gpu_ms  host%   GFLOP/s(gpu)")

race1 =
  for k <- ks do
    # built on the device, outside the timer
    a = Nx.iota({n, k}, type: {:f, 32}, backend: VB)
    b = Nx.iota({k, n}, type: {:f, 32}, backend: VB)
    :ok = NativeV.flush()

    m = measure.(fn -> Nx.dot(a, b) end)

    flops = 2 * n * n * k
    bytes = (2 * n * k + n * n) * 4
    gpu_ms = max(m.gpu, 0.001)
    gflops = flops / (gpu_ms / 1000.0) / 1.0e9
    gbs = bytes / (gpu_ms / 1000.0) / 1.0e9
    host_pct = m.enqueue / m.median * 100

    IO.puts(
      "  #{String.pad_leading("#{k}", 4)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(m.median, decimals: 3), 8)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(m.enqueue, decimals: 3), 7)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(gpu_ms, decimals: 3), 7)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(host_pct, decimals: 1), 5)}%" <>
        "  #{:erlang.float_to_binary(gflops, decimals: 2)}"
    )

    %{
      k: k,
      ms: m.median,
      ms_min: m.min,
      ms_max: m.max,
      host_ms: m.enqueue,
      host_ms_min: m.enqueue_min,
      gpu_ms: gpu_ms,
      host_pct: host_pct,
      gflops: gflops,
      gbs: gbs
    }
  end

# Normalised to this box's own k = 1024 point. THIS is the comparable number.
# THE DISPATCH FLOOR — and this is the number the race is really about.
#
# mac-248 fitted ms = a + b*k on both discrete boxes and found the fixed term
# nearly IDENTICAL (GT 750M a ~ 1.16 ms, RTX 3060 Ti a ~ 1.48 ms) while the
# per-k rate differed 3.8x. So the whole normalised-curve shape falls out of the
# single ratio a/b: the weaker card escapes its own floor at lower k purely
# because its per-unit work is slower. That is a submission-overhead-to-
# throughput ratio, NOT a memory-architecture signature.
#
# Two consequences, both bad for the original design. Race 1's shape comparison
# has low power — two same-category discrete boxes already differ 4x at k=4 from
# this ratio alone, so a Jetson difference has to clear that before it means
# anything. And the sharper quantity is `a` itself: the fixed per-dispatch cost
# is where host-to-device submission lives, and submission is precisely what
# unified memory should move. The harness was not reporting it at all.
#
# Fitted on GPU-only time, so the host frontend is already excluded.
fit = fn rows ->
  n = length(rows)
  xs = Enum.map(rows, &(&1.k * 1.0))
  ys = Enum.map(rows, & &1.gpu_ms)
  mx = Enum.sum(xs) / n
  my = Enum.sum(ys) / n
  num = Enum.zip(xs, ys) |> Enum.map(fn {x, y} -> (x - mx) * (y - my) end) |> Enum.sum()
  den = xs |> Enum.map(fn x -> (x - mx) * (x - mx) end) |> Enum.sum()
  b = if den == 0.0, do: 0.0, else: num / den
  {my - b * mx, b}
end

{intercept, per_k} = fit.(race1)

IO.puts("\n  dispatch floor, fitted gpu_ms = a + b*k over all #{length(ks)} points:")
IO.puts("    a (fixed per-dispatch cost) = #{:erlang.float_to_binary(intercept, decimals: 4)} ms")
IO.puts("    b (per unit of k)           = #{:erlang.float_to_binary(per_k, decimals: 5)} ms")

IO.puts(
  "    a/b                         = #{:erlang.float_to_binary(intercept / max(per_k, 1.0e-9), decimals: 1)}"
)

IO.puts("  `a` is the cross-box number to compare: it is the submission cost")
IO.puts("  unified memory should reduce. The normalised curve below is a/b in")
IO.puts("  disguise and mostly reflects GPU strength, not memory architecture.")

# TILE QUANTISATION. If two adjacent k values take the same GPU time despite a
# 4x difference in FLOPs, the kernel is padding small k up to a fixed workgroup
# tile and the nominal arithmetic intensity at that point is FICTIONAL — it is
# measuring padding, not intensity. Both Keplers show k=4 and k=16 within a
# fraction of a percent of each other (1.830 vs 1.838 ms; 1.457 vs 1.454) while
# doing 4x the work, so this is not hypothetical and it is not box-specific.
#
# It matters because the low-k end is exactly where a transfer-dominated regime
# would live, i.e. exactly where unified memory should show up. A sweep whose
# left edge is quantised cannot answer the question there.
quantised =
  race1
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.filter(fn [a, b] ->
    # Two signatures, and the second is the stronger one. NEAR-EQUAL: same GPU
    # time for 4x the FLOPs. INVERTED: the larger k is actually FASTER, which no
    # amount of arithmetic intensity can explain and only padding can. super-io
    # shows the inversion (k=4 0.818 ms vs k=16 0.599 ms); both Keplers show the
    # near-equality. The first detector caught only the Keplers' shape.
    near_equal = abs(b.gpu_ms - a.gpu_ms) / max(a.gpu_ms, 0.001) < 0.10
    inverted = b.gpu_ms < a.gpu_ms
    near_equal or inverted
  end)
  |> Enum.map(fn [a, b] -> {a.k, b.k} end)

if quantised != [] do
  IO.puts("\n  !! TILE QUANTISATION suspected between k pairs: #{inspect(quantised)}")
  IO.puts("     Those points take the same GPU time for 4x the FLOPs, so their")
  IO.puts("     nominal arithmetic intensity is fictional. Do not read the left")
  IO.puts("     edge of this sweep as a transfer-dominated regime.")
end

# Normalised on the GPU-ONLY figure, not the total — see the note in `measure`
# about why the total does not normalise away a slow host.
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

# Least-squares fit, not endpoint-to-endpoint. The endpoint form is a two-point
# estimate in which a single ragged interior sample sets the whole answer, and
# Race 4 has ragged samples: mac-247 measured 38 MiB at 121.8 ms against 36's
# 64.1 and 40's 73.5. A fit uses every point and degrades gracefully; the
# endpoint version would have reported that box's slope off two numbers, one of
# which happens to sit next to an outlier.
slope = fn rows, key ->
  n = length(rows)

  if n < 2 do
    0.0
  else
    xs = Enum.map(rows, & &1.mib)
    ys = Enum.map(rows, &Map.get(&1, key))
    mx = Enum.sum(xs) / n
    my = Enum.sum(ys) / n

    num =
      Enum.zip(xs, ys)
      |> Enum.map(fn {x, y} -> (x - mx) * (y - my) end)
      |> Enum.sum()

    den = xs |> Enum.map(fn x -> (x - mx) * (x - mx) end) |> Enum.sum()
    if den == 0.0, do: 0.0, else: num / den
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
      quantised_k_pairs: Enum.map(quantised, &Tuple.to_list/1),
      dispatch_floor_ms: intercept,
      per_k_ms: per_k,
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
