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
# CLOCK PINNING. The Jetson traced devfreq during an actual race and found full
# clock takes 2.25 s, not the ~500 ms its earlier probe suggested — because that
# probe used k=1024, which pins the GPU at 99.7% load, while the race's low-k
# points only reach 50-65% duty (the 3-6 ms host gap between dispatches idles
# the GPU, and nvhost_podgov ramps on UTILISATION). So k=1 was measured at
# ~384-460 MHz and k=8 onward at 614.4 MHz, and the non-monotonic gpu_ms column
# it saw (12.7 -> 9.2 -> 9.1 -> 6.9 -> 6.8) was the clock climbing, not the
# workload. Every low-k point — the ones with leverage on the intercept — was
# fitted through a contaminated left edge.
#
# Fix: before each measurement, run a SATURATING workload to pull the clock up,
# rather than relying on the op under test to do it. A small op cannot warm the
# clock it needs, because being small is what keeps the clock down.
#
# This does not fully rescue the smallest points. The Jetson pinned to 614 MHz
# and still watched the clock fall to 230 MHz DURING a k=1 measurement: low
# utilisation is what the governor reacts to, so on an integrated DVFS part the
# dispatch floor and the clock governor are coupled. Kepler never leaves P0 and
# has no such coupling — a methodological asymmetry that would masquerade as a
# memory-architecture difference if left unsaid.
pin_a = Nx.iota({512, 512}, type: {:f, 32}, backend: VB)
pin_b = Nx.iota({512, 512}, type: {:f, 32}, backend: VB)
:ok = NativeV.flush()

pin_clock = fn ->
  until = System.monotonic_time(:millisecond) + 800

  loop = fn self ->
    if System.monotonic_time(:millisecond) < until do
      _ = Nx.dot(pin_a, pin_b)
      :ok = NativeV.flush()
      :erlang.garbage_collect()
      self.(self)
    end
  end

  loop.(loop)
end

measure = fn fun ->
  pin_clock.()

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

IO.puts(
  "reps:   #{reps} (race 4: #{String.to_integer(System.get_env("NXV_RACE_ALLOC_REPS") || "25")})"
)

gpu_clock_smi = fn ->
  case System.cmd(
         "nvidia-smi",
         # Kepler on FreeBSD HAS nvidia-smi but reports [N/A] for clocks.sm, so
         # a "does nvidia-smi exist" test is not enough to know whether clock
         # telemetry is available. Ask for temperature and pstate too: those
         # boxes do report them, and they are a usable throttle proxy.
         [
           "--query-gpu=clocks.sm,clocks.max.sm,utilization.gpu,temperature.gpu,pstate",
           "--format=csv,noheader"
         ],
         stderr_to_stdout: true
       ) do
    {out, 0} -> String.trim(out)
    _ -> "n/a (nvidia-smi failed)"
  end
end

gpu_clock = fn ->
  # `System.cmd/3` RAISES :enoent when the binary is missing — it does NOT
  # return an error tuple — so the "n/a (no nvidia-smi)" fallback was
  # unreachable and this function CRASHED the entire run on any box without it.
  # Tegra ships tegrastats, not nvidia-smi, so the harness could not run on the
  # TREATMENT box at all: the Jetson had to shim a fake nvidia-smi into PATH to
  # produce any result. Guard with find_executable, and read Tegra's real clock
  # from sysfs, which is where it actually lives.
  tegra = "/sys/class/devfreq/57000000.gpu/cur_freq"

  cond do
    File.exists?(tegra) ->
      case File.read(tegra) do
        {:ok, hz} ->
          mhz = (hz |> String.trim() |> String.to_integer()) / 1_000_000
          "#{:erlang.float_to_binary(mhz, decimals: 1)} MHz (tegra devfreq)"

        _ ->
          "n/a (tegra devfreq unreadable)"
      end

    System.find_executable("nvidia-smi") == nil ->
      "n/a (no nvidia-smi)"

    true ->
      gpu_clock_smi.()
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
# RACE 1b — SEPARATING SUBMISSION FROM FIXED GPU WORK (the Jetson's design).
#
# `a` from the fit above is NOT a clean cross-box number, and the Jetson showed
# why. Against super-io its throughput term b is 23.3x, its fitted a is 11.2x
# and its empirical floor 14x. If the floor were purely GPU-side padded-tile
# work it would scale like b (23x); if purely host-to-device submission it would
# scale ~1x. It sits between, so `a` is a MIXTURE of the two, and comparing raw
# `a` across boxes differing 23x in throughput conflates them. Normalising by b
# instead (a/b: 15.6 vs 32) just picks the other convention and flips the
# apparent direction. Neither is evidence.
#
# The separation is available WITHIN a box, with no cross-box assumption at all:
# hold k small and fixed (inside the quantised region, where GPU time does not
# depend on k) and sweep n. Padded-tile GPU work scales with the n^2 output;
# host submission does not scale with n at all. So
#
#     floor(n) = s + c*n^2
#
# and `s` is the submission cost alone — which is the term unified memory should
# move, and the only one comparable across boxes on its own terms.
# ---------------------------------------------------------------------------

k_fixed = 8
# n=2048 added because the fast boxes need leverage. The Jetson pointed out that
# super-io's n=64/128/256 all flattened to ~0.255 ms — THREE points sitting on
# the tile-quantisation floor — leaving only n=512 and n=1024 carrying real n^2
# signal. That is an exactly-determined two-point fit with no redundancy, and it
# means super-io's s was largely reading its quantisation floor rather than its
# submission cost. Output at n=2048 is 16 MiB, comfortably under the 32 MiB
# allocator cliff and within the Jetson's memory.
ns = [64, 128, 256, 512, 1024, 2048]

IO.puts("\n--- Race 1b: submission vs fixed GPU work (k = #{k_fixed}, sweeping n) ---")
IO.puts("      n    gpu_ms   host_ms")

race1b =
  for nn <- ns do
    a = Nx.iota({nn, k_fixed}, type: {:f, 32}, backend: VB)
    b = Nx.iota({k_fixed, nn}, type: {:f, 32}, backend: VB)
    :ok = NativeV.flush()
    m = measure.(fn -> Nx.dot(a, b) end)

    IO.puts(
      "  #{String.pad_leading("#{nn}", 5)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(m.gpu, decimals: 3), 8)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(m.enqueue, decimals: 3), 8)}"
    )

    %{n: nn, gpu_ms: m.gpu, host_ms: m.enqueue}
  end

# Fit gpu_ms = s + c*n^2. The intercept is submission; the slope is per-output-
# element GPU work.
fit_n2 = fn rows ->
  cnt = length(rows)
  xs = Enum.map(rows, &(&1.n * &1.n * 1.0))
  ys = Enum.map(rows, & &1.gpu_ms)
  mx = Enum.sum(xs) / cnt
  my = Enum.sum(ys) / cnt
  num = Enum.zip(xs, ys) |> Enum.map(fn {x, y} -> (x - mx) * (y - my) end) |> Enum.sum()
  den = xs |> Enum.map(fn x -> (x - mx) * (x - mx) end) |> Enum.sum()
  c = if den == 0.0, do: 0.0, else: num / den
  {my - c * mx, c}
end

# EXCLUDE FLOORED POINTS FROM THE FIT. A point whose gpu_ms sits on the
# quantisation plateau carries no n^2 information — 4x the output for 2% more
# time is padding, not work — and because those are the SMALL-n points they have
# the most leverage on the intercept. Fitting through them makes `s` read the
# floor. Keep points at least 2x above the observed floor, and fall back to the
# full set (loudly) if that leaves too few to fit.
# Walk DOWN from the largest n and keep points while they actually scale with
# n^2. Doubling n quadruples the output, so a real point should be ~4x the one
# below it; allow >= 2x for noise. The first pair that fails is where the
# quantisation plateau starts, and everything below it is floor.
#
# A "2x the minimum" test is not good enough: super-io measured n=64 at 0.557 ms
# ABOVE n=128's 0.265 ms — a noisy point that such a test admits precisely
# because the noise made it large, and it has the most leverage on the intercept
# of any point in the sweep.
sorted_desc = Enum.sort_by(race1b, & &1.n, :desc)

above_floor =
  sorted_desc
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.reduce_while([hd(sorted_desc)], fn [bigger, smaller], acc ->
    # 3.5, matching the annotation below. These were 2.0 and 3.5 — two
    # thresholds for one concept — so the table marked a step floored while the
    # fit kept the point. 247's n=256 sits at 2.89 against 248's 3.21: both
    # admitted by the >= 2.0 rule while differently contaminated, which is
    # exactly how two same-architecture controls disagree on `s`.
    if bigger.gpu_ms / max(smaller.gpu_ms, 1.0e-9) >= 3.5 do
      {:cont, [smaller | acc]}
    else
      {:halt, acc}
    end
  end)
  |> Enum.sort_by(& &1.n)

kept_ns = MapSet.new(above_floor, & &1.n)
floored = Enum.reject(race1b, &MapSet.member?(kept_ns, &1.n))

{fit_rows, fit_note} =
  if length(above_floor) >= 3 do
    {above_floor,
     "excluding #{length(floored)} floored point(s): n=#{Enum.map_join(floored, ",", &"#{&1.n}")}"}
  else
    # Fall back to the LARGEST THREE points, not to all of them. mac-247 saw
    # that under the 3.5 threshold its run A retained 2 points and run B only 1,
    # so an "all points" fallback would have made two replicates of the same box
    # take DIFFERENT paths — one fitting a stricter window, one fitting
    # everything — and any movement in `s` between them would then be the
    # fallback rule rather than the box. The largest three are the least
    # floored available whatever the ratios say, and every run takes the same
    # path.
    {Enum.take(Enum.sort_by(race1b, & &1.n, :desc), 3) |> Enum.sort_by(& &1.n),
     "only #{length(above_floor)} point(s) cleared the floor test — falling back to the largest 3, so s is contaminated"}
  end

# FIXED window for the headline, detector as diagnostic. mac-247 replicated on
# an idle box minutes apart and the DETECTOR ITSELF flipped — run 1 excluded
# n=64 and 128, run 2 only n=64. That discrete choice moves `s`, and a fixed
# window is actually tighter (5.7% vs 6.9%). It also gives every box identical
# treatment, which is the argument for one tracked harness.
fixed_rows = Enum.filter(race1b, &(&1.n >= 256))
{submission_ms, per_elem_ms} = fit_n2.(fixed_rows)
{submission_adaptive, _} = fit_n2.(fit_rows)
floored_in_window = Enum.filter(floored, &(&1.n >= 256))

# LEVERAGE SHARE. The Jetson showed adding n=2048 for "leverage" partly
# backfired: with x = n^2 spanning 16384..4194304, the top point alone carries
# ~94% of the OLS leverage, so the line is set by the largest one or two points
# and `s` is a long extrapolation back to x=0. And `c` is NOT constant — its
# residuals put small-n systematically below the line because the large-n regime
# is bandwidth-bound and steeper. A steeper high-n slope extrapolated to zero
# inflates the intercept. That is part of why s moved 0.454 -> 0.680 there and
# 0.226 -> 0.303 on super-io: some of that is the floored-point fix working, and
# some is this artifact, and they are currently entangled.
#
# Not solved here. Reported, so nobody reads `s` as a direct measurement.
lev_xs = Enum.map(fixed_rows, &(&1.n * &1.n * 1.0))
lev_mx = Enum.sum(lev_xs) / length(lev_xs)
lev_tot = lev_xs |> Enum.map(fn x -> (x - lev_mx) * (x - lev_mx) end) |> Enum.sum()
lev_max = lev_xs |> Enum.map(fn x -> (x - lev_mx) * (x - lev_mx) end) |> Enum.max()
leverage_share = if lev_tot > 0.0, do: lev_max / lev_tot, else: 1.0

IO.puts("\n  fitted gpu_ms = s + c*n^2  (#{fit_note}):")

IO.puts(
  "    s (SUBMISSION cost, n-independent) = #{:erlang.float_to_binary(submission_ms, decimals: 4)} ms"
)

IO.puts(
  "    c (per output element)             = #{:erlang.float_to_binary(per_elem_ms * 1.0e6, decimals: 4)} ns"
)

# `s` is HOST work, so it carries a host-speed confound in place of the GPU-speed
# one it removes: a 2x slower host yields a 2x higher s with identical memory
# architecture. The Jetson is --disable-jit on 2 cores at 5W, so this is not
# hypothetical — its s/super-io ratio (2.0x) is about what its host alone would
# predict. Dividing by the box's own measured host time gives a figure that does
# not move with host speed.
# h0 — the n-INDEPENDENT host floor, not a median over the whole sweep.
#
# Two separate defects in the old median. 247: host time is drawn from a sticky
# two-level distribution, so a median of six draws is a lottery — one different
# draw would have stepped s/host_ms by 33% with no change in physics. And the
# Jetson: `host_ms` is not n-independent at all, rising 5.4x across the sweep
# (1.20 -> 6.59 ms) as Nx frontend work grows with tensor size, so the median
# lands mid-slope. Dividing an n-independent numerator by an n-dependent
# denominator is only comparable across boxes if host_ms rises by the SAME
# factor on each — and a fast JIT host is flatter, which would bias the ratio by
# about the size of the effect being measured.
#
# h0 = the median host_ms over the flat low-n points, which the floor filter has
# already identified. Both terms n-independent. On the Jetson this took
# reproducibility from 12.8% to 3.9%.
h0_rows = if floored != [], do: floored, else: Enum.take(Enum.sort_by(race1b, & &1.n), 2)

# MIN, not median. Taking the median over the floored rows did not fix the
# lottery: 247 drew h0 = 0.641 and 1.223 on consecutive runs — the two host
# levels exactly, one each — and s/h0 then read 0.2150 and 0.1160, a 60% spread
# against raw s's 2.9%. The minimum tracks the low mode and replicates to 3.4%
# where the median gives 8.3%.
host_ref = h0_rows |> Enum.map(& &1.host_ms) |> Enum.min()

IO.puts(
  "    s (adaptive window, diagnostic)    = #{:erlang.float_to_binary(submission_adaptive, decimals: 4)} ms"
)

IO.puts(
  "    h0 (n-independent host floor)      = #{:erlang.float_to_binary(host_ref, decimals: 4)} ms"
)

IO.puts(
  "    s / h0                             = #{:erlang.float_to_binary(submission_ms / max(host_ref, 1.0e-9), decimals: 4)}"
)

IO.puts(
  "    top-point leverage share           = #{:erlang.float_to_binary(leverage_share * 100, decimals: 1)}%"
)

# PER-BOX SCALING RATIOS. Doubling n quadruples the output, so a point in the
# true n^2 regime is 4.00x the one below it. Anything well under 4 is still on
# the quantisation plateau.
#
# This is printed because the EXTENT of the floor is a property of each box's
# tile geometry, and that is the residual structural risk in `s`. mac-248
# measured 1.41 / 2.17 / 3.21 / 3.67 / 3.98 — its true n^2 regime only begins
# near n=1024, so a fixed window at n>=256 still carries floored points and
# leaves s biased upward 10-20% there. If the floor reaches to a DIFFERENT n on
# another box, the same rule biases each box's s by a different amount, which is
# a reduced version of the structural problem that sank `a`.
#
# Excluding further is not free: through n=256 leaves only three points and 248's
# replicates then diverged 14%, worse than the bias removed. The window is a
# bias-variance tradeoff, so print the ratios and let the bias be visible rather
# than assumed equal across boxes.
IO.puts("\n  n^2 scaling ratios (4.00 = true n^2 regime, well under = still floored):")

race1b
|> Enum.sort_by(& &1.n)
|> Enum.chunk_every(2, 1, :discard)
|> Enum.each(fn [a, b] ->
  r = b.gpu_ms / max(a.gpu_ms, 1.0e-9)

  IO.puts(
    "    n=#{String.pad_leading("#{a.n}", 4)} -> #{String.pad_leading("#{b.n}", 4)}   " <>
      "#{:erlang.float_to_binary(r, decimals: 2)}" <>
      if(r < 3.5, do: "   <- floored", else: "")
  )
end)

# `s` is a component of every dispatch, so it cannot exceed the smallest
# dispatch measured. The Jetson fitted s = 0.7767 ms on a run whose own n=64
# dispatch took 0.568 ms — the fitted submission cost exceeding a whole dispatch
# by 37%. The floored points are poor data but an excellent BOUND.
min_dispatch = race1b |> Enum.map(& &1.gpu_ms) |> Enum.min()

if submission_ms > min_dispatch do
  IO.puts(
    "  !! s = #{:erlang.float_to_binary(submission_ms, decimals: 4)} ms EXCEEDS the smallest measured dispatch " <>
      "(#{:erlang.float_to_binary(min_dispatch, decimals: 4)} ms)."
  )

  IO.puts("     Submission cannot cost more than a whole dispatch, so this fit is")
  IO.puts("     over-extrapolated and s is an upper bound, not a measurement.")
end

IO.puts("  `s` is the part of the dispatch floor that does NOT scale with GPU")
IO.puts("  work. But s is HOST work, so raw s carries a host-speed confound in")
IO.puts("  place of the GPU-speed one it removes — compare s/host_ms across boxes,")
IO.puts("  not raw s.")

# ---------------------------------------------------------------------------
# RACE 1c — SUBMISSION AS A SLOPE, not an intercept.
#
# This exists because the control pair failed. The two Keplers agree on `c` to
# 1.7% and on their total dispatch floor to 12%, but differ on `s` by 1.9x —
# and with no agreement between two same-architecture controls there is no scale
# on which to judge the Jetson.
#
# The cause is conditioning, not physics. `s` is an INTERCEPT extrapolated back
# to n=0 from points that are all far from it; mac-248 measured `c` (a slope)
# to 0.44% and `s` (an intercept) to 8.8% ON THE SAME DATA. Intercepts are
# ill-conditioned and every defect found so far — leverage concentration, a
# non-constant c, residual quantisation floor — lands on the intercept.
#
# So measure it as a slope instead. `flush_locked` records every queued dispatch
# into ONE command buffer and does ONE submit_and_wait, so flushes are
# countable: run a FIXED number of dispatches split across a VARYING number of
# flushes, and
#
#     total(F) = base + F * s_flush
#
# where base is all the per-dispatch record and GPU work (constant, since the
# dispatch count is fixed) and the slope is the per-submission cost. That is the
# ~170 us floor DTrace already attributed 75% of to vkQueueWaitIdle, and it is
# the term a bus crossing lives in — i.e. the one unified memory should move.
#
# No extrapolation to zero, no dependence on the quantisation floor, and the
# quantity is read off a slope.
# ---------------------------------------------------------------------------

n_1c = 256
dispatch_count = 32
flush_counts = [1, 2, 4, 8, 16, 32]

IO.puts(
  "\n--- Race 1c: submission as a slope (#{dispatch_count} dispatches, varying flushes) ---"
)

IO.puts("  flushes    total_ms   ms/flush-step")

c1 = Nx.iota({n_1c, 8}, type: {:f, 32}, backend: VB)
c2 = Nx.iota({8, n_1c}, type: {:f, 32}, backend: VB)
:ok = NativeV.flush()

race1c =
  for f <- flush_counts do
    per_flush = div(dispatch_count, f)

    run = fn ->
      Enum.each(1..f, fn _ ->
        Enum.each(1..per_flush, fn _ -> Nx.dot(c1, c2) end)
        :ok = NativeV.flush()
      end)
    end

    pin_clock.()
    # warm
    run.()
    run.()

    samples =
      for _ <- 1..reps do
        :erlang.garbage_collect()
        t0 = System.monotonic_time(:microsecond)
        run.()
        t1 = System.monotonic_time(:microsecond)
        (t1 - t0) / 1000.0
      end

    med = median.(samples)

    IO.puts(
      "  #{String.pad_leading("#{f}", 7)}   #{String.pad_leading(:erlang.float_to_binary(med, decimals: 3), 9)}"
    )

    %{flushes: f, total_ms: med}
  end

fit_lin = fn rows, xf ->
  cnt = length(rows)
  xs = Enum.map(rows, xf)
  ys = Enum.map(rows, & &1.total_ms)
  mx = Enum.sum(xs) / cnt
  my = Enum.sum(ys) / cnt
  num = Enum.zip(xs, ys) |> Enum.map(fn {x, y} -> (x - mx) * (y - my) end) |> Enum.sum()
  den = xs |> Enum.map(fn x -> (x - mx) * (x - mx) end) |> Enum.sum()
  b = if den == 0.0, do: 0.0, else: num / den
  {my - b * mx, b}
end

{base_ms, s_flush_ols} = fit_lin.(race1c, &(&1.flushes * 1.0))

# MEDIAN OF ADJACENT MARGINALS, not OLS. 247's recommendation, and its data
# makes the case. Within a run the estimator is excellent — residuals +/-0.125 ms
# on values spanning 19-36 ms, and s_flush moving 0.5% across fit windows where
# `s` moved 5x. But ONE contaminated point wrecks the OLS: its run B read F=2 at
# 28.4 ms where the trend says 19.6 (+45%), and since each point is a median of
# 9 reps that is a sustained block anomaly, not a spike — the same sticky-block
# pathology behind the bimodal host levels and Race 4's cold first size.
#
#   OLS over all six      532.8 vs 412.5   25.4% between replicates
#   OLS dropping F=2      534.4 vs 515.6    3.6%
#   median of marginals   532.9 vs 522.1    2.0%
#
# The median needs no window choice and no outlier detector: a corrupted point
# poisons two adjacent pairs and the median walks past both. 247's run B
# marginals were 9399, -3414, 281, 540, 522 us — the median lands on 522.
# F=1 IS A SEPARATE REGIME AND IS EXCLUDED. mac-248 measured the F=1->2 marginal
# at 839 and 933 us against a settled ~350 — reproducible across both runs, so
# structural rather than a spike. With a single flush the one submit_and_wait
# overlaps all 32 dispatches; at F=2 a mid-sequence sync is forced that cannot
# overlap. That step is real physics but it is not the per-submission cost, and
# including it inflates the estimate.
#
# Both Keplers show the same shape — 247's first marginals are 751 and 727
# against its settled ~530 — so this is not one box's quirk. Dropping F=1
# tightens 248's between-run spread from 6.4% to 2.4%, matching 247's quality.
# super-io curves the OTHER way (its low-F points sit below its slope), so the
# exclusion is justified on the control boxes and simply removes a point
# elsewhere.
marginals =
  race1c
  |> Enum.sort_by(& &1.flushes)
  |> Enum.filter(&(&1.flushes >= 2))
  |> Enum.chunk_every(2, 1, :discard)
  |> Enum.map(fn [a, b] -> (b.total_ms - a.total_ms) / (b.flushes - a.flushes) end)

s_flush = median.(marginals)

IO.puts("\n  total_ms = base + F * s_flush:")

IO.puts(
  "    s_flush (PER-SUBMISSION cost) = #{:erlang.float_to_binary(s_flush * 1000, decimals: 1)} us   [median of adjacent marginals]"
)

IO.puts(
  "    s_flush (OLS, diagnostic)     = #{:erlang.float_to_binary(s_flush_ols * 1000, decimals: 1)} us" <>
    "   [read TOGETHER with the median: agreement = clean, divergence = contaminated]"
)

IO.puts(
  "    adjacent marginals (us)       = " <>
    Enum.map_join(marginals, ", ", &:erlang.float_to_binary(&1 * 1000, decimals: 0))
)

# A NEGATIVE MARGINAL IS PHYSICALLY IMPOSSIBLE. Only the submission count varies
# between adjacent points, so more submissions cannot take less total time. If
# one appears, something else moved during Race 1c and the run cannot be used.
#
# The Jetson hit this: marginals 3.779, 2.155, 2.709, -0.572, 0.924 ms with the
# table non-monotonic (F=16 at 95.9 ms BELOW F=8 at 100.5). Its hypothesis fits:
# 32 dispatches at n=256 is a low-duty workload, low duty is what makes podgov
# drop the clock, and MORE flushes means more sustained activity means a HIGHER
# clock — which pushes totals down as F rises and bends the marginals this way.
#
# That makes Race 1c itself DVFS-confounded on an integrated part — the same
# coupling that defeated `a` there, where the quantity measured and the clock
# governor are not independent. On that box the two estimators disagree by 164%
# (2155 vs 815 us) against mac-247's 2.0%, because a median of marginals is well
# conditioned only when the marginals cluster, and these span -0.572 to +3.779.
marginal_anomaly = Enum.filter(marginals, &(&1 < 0.0))

if marginal_anomaly != [] do
  IO.puts("\n  !! NEGATIVE MARGINAL — physically impossible for a submission cost.")
  IO.puts("     More flushes cannot take less total time when only the flush count")
  IO.puts("     varies, so something else moved during Race 1c. On an integrated")
  IO.puts("     DVFS part the likely cause is the clock rising with flush count,")
  IO.puts("     since more flushes means more sustained activity. s_flush from")
  IO.puts("     this run is not usable.")
end

# `base` is a FREE contamination detector, also 247's: the dispatch count is
# fixed, so base must be identical across replicates of the same box. Its runs
# read 18.677 vs 21.340 (14% apart) with the bad point and 18.640 vs 18.838
# (0.9%) without — so a base disagreement between replicates flags a bad Race 1c
# run without needing to know which point went wrong.
IO.puts(
  "    base (#{dispatch_count} dispatches of work)     = #{:erlang.float_to_binary(base_ms, decimals: 4)} ms   [across replicates: should be identical]"
)

IO.puts("  s_flush is a SLOPE, so it does not depend on extrapolating to zero,")
IO.puts("  on the quantisation floor, or on which points enter the fit. This is")
IO.puts("  the number the control pair should agree on.")

# ---------------------------------------------------------------------------
# RACE 2 — THE PRICE OF CROSSING THE BOUNDARY, measured WITHIN one box.
#
# This is the design the cross-box approach should have been from the start.
# Races 1, 1b and 1c all ended up comparing a number from one machine against a
# number from another, and that failed for a reason no estimator could fix: the
# two Keplers disagree 1.39x on submission cost, they are different SKUs on
# different Mac hosts, one idles into P8 and the other never leaves P0. "Same
# architecture" was never "same hardware", and every quantity we tried to
# compare was host- and driver-dominated.
#
# So compare nothing across boxes. Run BOTH arms on the SAME machine:
#
#   (a) resident:   upload once -> N ops on device -> download once
#   (b) round-trip: N x (upload -> 1 op -> download)
#
# Both do exactly N ops. (b) additionally pays N-1 extra uploads and downloads,
# so
#
#   boundary cost per crossing = (b - a) / (N - 1)
#   compute cost per op        = a / N          (approximately)
#   PRICE = boundary / compute
#
# PRICE is dimensionless: how many ops-worth of time one host<->device round trip
# costs on this box. Host speed, GPU speed, driver overhead and SKU all appear in
# BOTH terms and cancel to first order — which is exactly what defeated the
# cross-box comparisons. It is the ratio the original plan called for and the
# only quantity in this experiment that does not need two machines to mean
# something.
#
# What unified memory predicts: on a discrete card a crossing pays PCIe while
# compute does not, so PRICE should be large and should GROW with transfer size.
# On unified memory there is no bus to pay, so PRICE should be smaller and
# flatter in size. The SHAPE of PRICE against size is the answer, measured
# independently on each box and never subtracted across them.
# ---------------------------------------------------------------------------

n_ops = 32
sizes_kib = [64, 256, 1024, 4096, 16384]

IO.puts("\n--- Race 2: price of a host<->device round trip, in ops (within-box) ---")
IO.puts("     KiB   resident_ms  roundtrip_ms   boundary_ms   compute_ms    PRICE")

race2 =
  for kib <- sizes_kib do
    bytes = kib * 1024
    bin = :binary.copy(<<0, 0, 128, 63>>, div(bytes, 4))

    resident = fn ->
      t = Nx.from_binary(bin, {:f, 32}, backend: VB)
      out = Enum.reduce(1..n_ops, t, fn _, acc -> Nx.multiply(acc, 1.0) end)
      _ = Nx.backend_transfer(out, Nx.BinaryBackend)
      :ok
    end

    roundtrip = fn ->
      Enum.each(1..n_ops, fn _ ->
        t = Nx.from_binary(bin, {:f, 32}, backend: VB)
        o = Nx.multiply(t, 1.0)
        _ = Nx.backend_transfer(o, Nx.BinaryBackend)
      end)
    end

    a = measure.(resident)
    :erlang.garbage_collect()
    b = measure.(roundtrip)
    :erlang.garbage_collect()

    boundary = (b.median - a.median) / (n_ops - 1)
    compute = a.median / n_ops
    price = boundary / max(compute, 1.0e-9)

    IO.puts(
      "  #{String.pad_leading("#{kib}", 6)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(a.median, decimals: 3), 11)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(b.median, decimals: 3), 12)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(boundary, decimals: 4), 12)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(compute, decimals: 4), 11)}" <>
        "  #{String.pad_leading(:erlang.float_to_binary(price, decimals: 2), 8)}"
    )

    %{
      kib: kib,
      resident_ms: a.median,
      roundtrip_ms: b.median,
      boundary_ms: boundary,
      compute_ms: compute,
      price: price
    }
  end

# A negative boundary cost is impossible: (b) does strictly more work than (a).
race2_anomaly = Enum.filter(race2, &(&1.boundary_ms < 0.0))

if race2_anomaly != [] do
  IO.puts(
    "\n  !! NEGATIVE BOUNDARY COST at KiB=" <>
      Enum.map_join(race2_anomaly, ",", &"#{&1.kib}") <>
      " — impossible, (b) does strictly more work than (a). Run not usable."
  )
end

price_lo = hd(race2).price
price_hi = List.last(race2).price

IO.puts(
  "\n  PRICE at #{hd(race2).kib} KiB = #{:erlang.float_to_binary(price_lo, decimals: 2)} ops"
)

IO.puts(
  "  PRICE at #{List.last(race2).kib} KiB = #{:erlang.float_to_binary(price_hi, decimals: 2)} ops"
)

IO.puts(
  "  growth over the sweep       = #{:erlang.float_to_binary(price_hi / max(price_lo, 1.0e-9), decimals: 2)}x"
)

IO.puts("  A crossing costs this many ops-worth of time ON THIS BOX. Both terms")
IO.puts("  share the same host, driver, SKU and GPU, so those cancel — which is")
IO.puts("  what no cross-box quantity in this experiment managed.")

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

# Warm the allocator before the FIRST timed size. mac-247 measured 24 MiB
# zeroed at 2.656 ms against 1.208/1.211/1.224 in three prior runs — a 2.2x cold
# spike on whichever size is measured first. With only four points below the
# cliff, one cold first point is enough to flip the fitted slope's SIGN, and it
# did: that run reported zeroed_below = -0.1633 ms/MiB. Zeroing more memory
# cannot take less time.
#
# `measure` already warms per-call, but the first call of a new NIF in a fresh
# process pays driver first-touch that per-call warming does not cover, and
# Race 1b now runs between Race 1 and Race 4, so the allocator arrives in a
# different state than it used to.
for _ <- 1..3 do
  {:ok, _} = NativeV.buf_alloc(24 * 1024 * 1024)
  {:ok, _} = NativeV.buf_alloc_zeroed(24 * 1024 * 1024)
  :erlang.garbage_collect()
end

# Race 4 gets its OWN rep count and no clock pin.
#
# No pin because allocation is driver and host work, not shader work — pinning
# the GPU clock cannot help it, and mac-248 showed it actively perturbs: its
# above-cliff zeroed times shifted 25-30% systematically once Race 1b began
# running between Race 1 and Race 4, changing the allocator state Race 4
# inherits.
#
# More reps because at 9 the zeroed slope is not measurable. mac-248 ran four
# times on a quiet box and got zeroed_above of 1.91 / 2.46 / 1.10 / 1.56 — a
# 2.24x spread, which makes any cross-box comparison of that number meaningless.
# alloc_below is the solid one: 0.0000-0.0001 every run, with the 32 MiB cliff
# reproducing 4/4.
alloc_reps = String.to_integer(System.get_env("NXV_RACE_ALLOC_REPS") || "25")

measure_alloc = fn fun ->
  for _ <- 1..3 do
    _ = fun.()
    :erlang.garbage_collect()
  end

  samples =
    for _ <- 1..alloc_reps do
      :erlang.garbage_collect()
      t0 = System.monotonic_time(:microsecond)
      _ = fun.()
      :ok = NativeV.flush()
      t1 = System.monotonic_time(:microsecond)
      (t1 - t0) / 1000.0
    end

  %{median: median.(samples), min: Enum.min(samples), max: Enum.max(samples)}
end

race4 =
  for mib <- 24..40//2 do
    bytes = mib * 1024 * 1024
    a = measure_alloc.(fn -> {:ok, _} = NativeV.buf_alloc(bytes) end)
    :erlang.garbage_collect()
    z = measure_alloc.(fn -> {:ok, _} = NativeV.buf_alloc_zeroed(bytes) end)
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

# A negative slope is physically impossible — zeroing more memory cannot take
# less time — so it is a measurement fault, not a finding. The run that produced
# one still printed RACE: OK, because the thermal control re-measures a Race 1
# MATMUL and so certified 1.6% drift while the headline Race 4 slope was
# negative. The control was watching the wrong race. A sign check costs nothing.
# Only the ZEROED series. Zeroing more memory must take longer, so a negative
# slope there is physically impossible and indicates a measurement fault — which
# is what mac-247 hit with zeroed_below = -0.1633 after a cold first sample.
#
# `buf_alloc` is different in kind: below the cliff it is genuinely O(1), flat at
# 0.006-0.02 ms with a slope of 0.0000 on every box. Its fitted slope is noise
# about zero and goes negative roughly half the time, so checking its sign voids
# healthy runs for a quantity that has no sign to check. super-io produced
# -0.0005 and -0.0007 on consecutive clean runs.
slope_anomaly =
  slopes
  |> Map.to_list()
  |> Enum.filter(fn {k, v} -> v < 0.0 and String.starts_with?(Atom.to_string(k), "zeroed") end)

if slope_anomaly != [] do
  IO.puts("\n  !! NEGATIVE SLOPE — physically impossible, this is a measurement fault:")

  Enum.each(slope_anomaly, fn {k, v} ->
    IO.puts("     #{k} = #{:erlang.float_to_binary(v, decimals: 4)} ms/MiB")
  end)

  IO.puts("     Race 4 for this run is not usable. The thermal control cannot")
  IO.puts("     see this — it re-measures a Race 1 matmul, not an allocation.")
end

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
drift_compute = abs(control.median - first) / first

# AN ALLOCATION CONTROL TOO. The compute control certified mac-247's run Q at
# 1.6% drift while that run's allocation was 2.2x slow across every below-cliff
# point and its submission cost was inflated 60%. Race 1's matmuls were simply
# fine while the two things Races 1c and 4 measure were both bad.
#
# That is the same blind spot as before in a new place: a control has to
# re-measure the KIND of work whose result it is certifying. Three independent
# checks caught run Q — the negative-slope check, the `base` mismatch, and the
# uniform inflation of the Race 4 series — but the one the harness calls "the
# thermal control" was not among them, and it is the one that decides
# RACE: OK.
alloc_first = Enum.find(race4, &(&1.mib == 24)).zeroed_ms
alloc_control = measure_alloc.(fn -> {:ok, _} = NativeV.buf_alloc_zeroed(24 * 1024 * 1024) end)
drift_alloc = abs(alloc_control.median - alloc_first) / alloc_first

drift = max(drift_compute, drift_alloc)

IO.puts("\n--- Thermal control (k = #{k0}, the compute-dominated point, repeated last) ---")

IO.puts(
  "  first: #{:erlang.float_to_binary(first, decimals: 3)} ms   last: #{:erlang.float_to_binary(control.median, decimals: 3)} ms"
)

IO.puts("  compute drift:    #{:erlang.float_to_binary(drift_compute * 100, decimals: 1)}%")

IO.puts(
  "  allocation drift: #{:erlang.float_to_binary(drift_alloc * 100, decimals: 1)}%" <>
    "   (24 MiB zeroed: #{:erlang.float_to_binary(alloc_first, decimals: 3)} -> #{:erlang.float_to_binary(alloc_control.median, decimals: 3)} ms)"
)

void? =
  drift > 0.10 or slope_anomaly != [] or marginal_anomaly != [] or race2_anomaly != []

cond do
  drift > 0.10 ->
    IO.puts("  *** DRIFT > 10% — THIS RUN IS VOID, the box throttled or was contended ***")

  slope_anomaly != [] ->
    IO.puts("  *** VOID — Race 4 produced a negative slope (see above) ***")

  marginal_anomaly != [] ->
    IO.puts("  *** VOID — Race 1c produced a negative marginal (see above) ***")

  true ->
    :ok
end

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
      race1b: race1b,
      submission_ms: submission_ms,
      submission_over_host: submission_ms / max(host_ref, 1.0e-9),
      submission_adaptive_ms: submission_adaptive,
      h0_host_floor_ms: host_ref,
      leverage_share: leverage_share,
      race1c: race1c,
      race2: race2,
      race2_price_low_kib: price_lo,
      race2_price_high_kib: price_hi,
      race2_price_growth: price_hi / max(price_lo, 1.0e-9),
      race2_negative_boundary: length(race2_anomaly),
      s_flush_us: s_flush * 1000,
      s_flush_ols_us: s_flush_ols * 1000,
      race1c_marginals_us: Enum.map(marginals, &(&1 * 1000)),
      race1c_negative_marginals: length(marginal_anomaly),
      race1c_base_ms: base_ms,
      min_dispatch_ms: min_dispatch,
      s_exceeds_min_dispatch: submission_ms > min_dispatch,
      floored_in_fixed_window: Enum.map(floored_in_window, & &1.n),
      n2_scaling_ratios:
        race1b
        |> Enum.sort_by(& &1.n)
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [a, b] ->
          %{from: a.n, to: b.n, ratio: b.gpu_ms / max(a.gpu_ms, 1.0e-9)}
        end),
      race1b_fit_note: fit_note,
      race1b_floored_ns: Enum.map(floored, & &1.n),
      per_output_elem_ns: per_elem_ms * 1.0e6,
      per_k_ms: per_k,
      race1_normalised_to: "own k=1024 gflops",
      race4: race4,
      race4_slopes_ms_per_mib: slopes,
      thermal_control: %{
        first_ms: first,
        last_ms: control.median,
        drift: drift,
        drift_compute: drift_compute,
        drift_alloc: drift_alloc,
        alloc_first_ms: alloc_first,
        alloc_last_ms: alloc_control.median,
        void: void?
      },
      slope_anomaly: Enum.map(slope_anomaly, fn {k, v} -> %{series: k, ms_per_mib: v} end),
      load_after: String.trim(load_after),
      gpu_clock_after: clock_after
    },
    pretty: true
  )
)

IO.puts("wrote #{path}")

IO.puts(
  cond do
    drift > 0.10 -> "\nRACE: VOID (thermal drift)"
    slope_anomaly != [] -> "\nRACE: VOID (negative Race 4 slope)"
    marginal_anomaly != [] -> "\nRACE: VOID (negative Race 1c marginal)"
    true -> "\nRACE: OK"
  end
)
