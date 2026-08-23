# Should `window_reduce/6` be implemented, or allowlisted like `reduce/5`?
#
#   mix run bench/window_reduce_fold_vs_host.exs
#
# `Nx.window_reduce/5` takes an arbitrary user fun, so no shader expresses it —
# the same starting position as `reduce/5`, which lost to the host at every size
# and by 12x at reduce_size 4096 (bench/reduce_fold_vs_host.exs) and was
# allowlisted as a result.
#
# BUT THE FOLD LENGTH IS DIFFERENT, and that is the whole question.
#
#   reduce/5      folds over the REDUCED AXIS -> one dispatch per element of it.
#                 A 4096-long axis is 4096 dispatches. That is why it lost.
#   window_reduce folds over the WINDOW -> one dispatch per window position,
#                 typically 4 or 9, REGARDLESS of how large the tensor is.
#
# So the two may land on opposite sides of the same argument, and the trend
# across window sizes matters more than any single number: if the fold's cost
# tracks the WINDOW while the host's tracks the DATA, they cross over and the
# gate is worth widening.
#
# The vectorised fold: for each of the prod(window_dims) offsets inside the
# window, take a STRIDED slice of the whole input at that offset — that is the
# w-th element of every window at once — and fold it into an accumulator with
# the user's fun. Overlapping windows simply give overlapping slices.
#
# Valid padding, unit dilation, rank 2 — the case a gate would cover.
#
# Run it on MORE THAN ONE BOX. NEXT.md §5: win/loss crossovers here are
# hardware-specific, and one of them has already been retracted as noise.

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend, as: V
alias Nx.BinaryBackend, as: B

{:ok, dev, _} = Nx.Vulkan.NativeV.device_name()
host = System.cmd("hostname", ["-s"]) |> elem(0) |> String.trim()
load = System.cmd("uptime", []) |> elem(0)
IO.puts("\ndevice: #{dev}   host: #{host}")
IO.puts("load before run:#{String.slice(load, -24, 24)}")

# One dispatch per window OFFSET, each a strided slice covering every window at
# once. `len = (out - 1) * stride + 1` is the extent the strided slice must span
# to yield exactly `out` elements.
fold = fn t, acc0, {w0, w1}, {s0, s1}, {o0, o1}, fun ->
  acc = Nx.broadcast(Nx.tensor(acc0, type: Nx.type(t), backend: V), {o0, o1})

  for i <- 0..(w0 - 1), j <- 0..(w1 - 1), reduce: acc do
    a ->
      plane = Nx.slice(t, [i, j], [(o0 - 1) * s0 + 1, (o1 - 1) * s1 + 1], strides: [s0, s1])
      fun.(plane, a)
  end
end

# Median of 3. Both arms end with the answer on the host, as in
# examples/w5_kernels_race.exs — without the readback the GPU arm measures
# ENQUEUE cost, since dispatches batch up to NXV_BATCH_MAX before submission.
#
# THE COLLECT IS NOT TIDINESS. The first version of this file interleaved the
# two arms per case, and the host arm's `Nx.window_reduce` on BinaryBackend
# allocates so heavily that its GC pressure landed on the NEXT fold measurement:
# 512x512 with a 3x3 window reported 41.05 ms interleaved against 9.95 ms
# measured alone, verified at 3, 20 and 400 iterations so it was not warm-up.
# That is a 4x error, and in the direction that HIDES the trend the benchmark
# exists to show — it made the fold look flat in the window size when it is
# close to linear in it. Forcing a collect before each timed run removes it.
bench = fn f ->
  f.()
  :erlang.garbage_collect()

  xs =
    for _ <- 1..3 do
      :erlang.garbage_collect()
      {us, r} = :timer.tc(f)
      {us / 1000.0, r}
    end

  sorted = xs |> Enum.map(&elem(&1, 0)) |> Enum.sort()
  {Enum.at(sorted, 1), xs |> List.last() |> elem(1)}
end

IO.puts("")
IO.puts(String.pad_trailing("case", 26) <> String.pad_leading("dispatches", 12) <>
        String.pad_leading("fold ms", 11) <> String.pad_leading("host ms", 11) <>
        String.pad_leading("verdict", 18))
IO.puts(String.duplicate("-", 78))

# 512x512 with an 8x8 window is deliberately absent: its HOST arm is ~120 s per
# iteration, which is six minutes for one row that only restates the trend. The
# GT 650M number for it is in the DTrace write-up if it is ever wanted.
for {rows, cols, w} <- [{64, 64, 2}, {256, 256, 2}, {256, 256, 3}, {512, 512, 3}, {512, 512, 5}] do
  o0 = rows - w + 1
  o1 = cols - w + 1
  data = for i <- 1..(rows * cols), do: rem(i, 17)

  gt = Nx.tensor(data, type: {:s, 32}, backend: V) |> Nx.reshape({rows, cols})
  ht = Nx.tensor(data, type: {:s, 32}, backend: B) |> Nx.reshape({rows, cols})

  {fold_ms, fold_res} =
    bench.(fn ->
      fold.(gt, 0, {w, w}, {1, 1}, {o0, o1}, fn x, y -> Nx.add(x, y) end)
      |> Nx.backend_transfer(B)
    end)

  {host_ms, host_res} =
    bench.(fn -> Nx.window_reduce(ht, 0, {w, w}, [], fn x, y -> Nx.add(x, y) end) end)

  same = Nx.to_flat_list(fold_res) == Nx.to_flat_list(host_res)

  verdict =
    cond do
      not same -> "MISMATCH — STOP"
      fold_ms < host_ms -> "fold wins #{Float.round(host_ms / fold_ms, 1)}x"
      true -> "fold LOSES #{Float.round(host_ms / fold_ms, 2)}x"
    end

  IO.puts(String.pad_trailing("#{rows}x#{cols} window #{w}x#{w}", 26) <>
          String.pad_leading(Integer.to_string(w * w), 12) <>
          String.pad_leading(Float.to_string(Float.round(fold_ms, 2)), 11) <>
          String.pad_leading(Float.to_string(Float.round(host_ms, 2)), 11) <>
          String.pad_leading(verdict, 18))
end

IO.puts("""

READ THE TREND, not the absolute numbers. The fold does prod(window_dims)
dispatches whatever the tensor size, where reduce/5's did one per element of the
reduced axis. If the fold holds roughly flat as the DATA grows while the host
arm scales with it, the gate is worth widening; if the fold scales with the
WINDOW faster than the host does, it is the reduce/5 answer again.

A MISMATCH line means the fold is wrong, not slow — stop and fix that first.
""")
