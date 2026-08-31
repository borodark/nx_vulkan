# Item 1 with GPU CLOCK AS AN EXPLICIT DIMENSION.
#
# Every previous number in this investigation was taken at an unrecorded clock,
# which is how a 210 MHz reading became a hardware finding in NEXT_SESSION.md.
# Here each measurement window is bracketed with epoch-ms marks and joined
# against an external nvidia-smi logger afterwards, so the clock that ACTUALLY
# held during the samples is reported next to them. nvidia-smi is never called
# in-process: the ~50-100 ms it takes is itself enough GPU idle to drop boost.
#
# Three states are induced, not requested (no root, so no `nvidia-smi -lgc`):
#   idle    - sleep before each sample; GPU settles to its 210 MHz floor
#   partial - a short burst before each sample; catches the ramp
#   boost   - sustained dispatch; the state a real workload runs in
alias Nx.Vulkan.VulkanoBackend, as: VB
alias Nx.Vulkan.NativeV

clock_log = System.get_env("CLOCK_LOG") || "/tmp/clock.log"
sh = fn f -> Path.expand("priv/shaders/#{f}", File.cwd!()) end
flat_spv = sh.("elementwise_binary_f32.spv"); bcast_spv = sh.("elementwise_binary_bcast_f32.spv")

median = fn xs -> s = Enum.sort(xs); n = length(s)
  if n == 0, do: 0.0,
  else: (if rem(n,2)==1, do: Enum.at(s,div(n,2)), else: (Enum.at(s,div(n,2)-1)+Enum.at(s,div(n,2)))/2) end
f3 = fn v -> :erlang.float_to_binary(v/1.0, decimals: 3) end
now = fn -> :os.system_time(:millisecond) end

bn = div(64*1024*1024, 4)
{:ok, ga} = NativeV.buf_alloc(bn*4); {:ok, go} = NativeV.buf_alloc(bn*4); :ok = NativeV.flush()
burst = fn budget ->
  stop = System.monotonic_time(:millisecond) + budget
  loop = fn self -> if System.monotonic_time(:millisecond) < stop do
    Enum.each(1..20, fn _ -> :ok = NativeV.apply_binary(go, ga, ga, bn, 1, flat_spv) end)
    :ok = NativeV.flush(); self.(self) end end
  loop.(loop)
end

# {label, preamble before the window, per-sample setup}
states = [
  {"idle",    fn -> Process.sleep(1200) end, fn -> Process.sleep(400) end},
  {"partial", fn -> Process.sleep(800) end,  fn -> burst.(120) end},
  {"boost",   fn -> burst.(700) end,         fn -> burst.(40) end}
]

windows = :ets.new(:win, [:bag, :public])
sample = fn key, pre, per, f, reps ->
  pre.()
  t_begin = now.()
  xs = for _ <- 1..reps do
    per.()
    t0 = System.monotonic_time(:microsecond); f.(); :ok = NativeV.flush()
    (System.monotonic_time(:microsecond)-t0)/1000.0
  end
  :ets.insert(windows, {key, t_begin, now.()})
  median.(xs)
end

params_for = fn n -> for v <- [1,n,1,1,1,n,1,1,1,1,1,1,1], into: <<>>, do: <<v::signed-32-little>> end
sizes = [48, 64, 128]
results = :ets.new(:res, [:set, :public])

for rep <- 1..2, mib <- sizes do
  n = div(mib*1024*1024, 4); bytes = n*4
  {:ok, a} = NativeV.buf_alloc(bytes); {:ok, b} = NativeV.buf_alloc(bytes)
  {:ok, o} = NativeV.buf_alloc(bytes); {:ok, one} = NativeV.buf_alloc(4)
  {:ok, prm} = NativeV.buf_upload(params_for.(n))
  # cheap resident tensor: host memcpy + one upload, NOT Nx.iota (minutes of
  # host work at 33M elements, with the GPU idling the whole time)
  x = Nx.from_binary(:binary.copy(<<0,0,128,63>>, n), {:f,32}, backend: VB)
  :ok = NativeV.flush()

  quantities = [
    {"flat_re",  3, fn -> :ok = NativeV.apply_binary(o,a,b,n,1,flat_spv) end},
    {"bcast_re", 2, fn -> :ok = NativeV.apply_binary_broadcast(o,a,one,prm,n,1,1,bcast_spv) end},
    {"alloc",    1, fn -> {:ok,_} = NativeV.buf_alloc(bytes) end},
    {"nx_mul",   2, fn -> _ = Nx.multiply(x, 1.0) end}
  ]
  for {sname, pre, per} <- states, {qname, mult, f} <- quantities do
    key = {rep, mib, sname, qname}
    ms = sample.(key, pre, per, f, 9)
    :ets.insert(results, {key, ms, mult, bytes})
  end
  :erlang.garbage_collect()
end

# join each window against the external clock log
Process.sleep(400)
clocks =
  case File.read(clock_log) do
    {:ok, body} ->
      for line <- String.split(body, "\n", trim: true),
          [ts, c] <- [String.split(line, " ", trim: true)],
          {t, ""} <- [Integer.parse(ts)], {cv, ""} <- [Integer.parse(c)], do: {t, cv}
    _ -> []
  end
IO.puts("\nclock log: #{length(clocks)} samples")

clock_for = fn key ->
  case :ets.lookup(windows, key) do
    [{_, t0, t1} | _] ->
      inw = for {t, c} <- clocks, t >= t0 - 50, t <= t1 + 50, do: c
      if inw == [], do: {0, 0}, else: {round(median.(Enum.map(inw, &(&1/1.0)))), Enum.min(inw)}
    _ -> {0, 0}
  end
end

IO.puts("\n64 MiB f32 on super-io, median of 9, clock measured out-of-process")
IO.puts("  rep  MiB  state    quantity   ms        GB/s   clk_med  clk_min")
for rep <- 1..2, mib <- sizes, {sname,_,_} <- states,
    qname <- ["flat_re","bcast_re","alloc","nx_mul"] do
  key = {rep, mib, sname, qname}
  [{_, ms, mult, bytes}] = :ets.lookup(results, key)
  {cmed, cmin} = clock_for.(key)
  IO.puts([
    String.pad_leading("#{rep}",5), String.pad_leading("#{mib}",5), "  ",
    String.pad_trailing(sname,8), " ", String.pad_trailing(qname,9), " ",
    String.pad_leading(f3.(ms),7), " ", String.pad_leading(f3.(mult*bytes/(ms/1000.0)/1.0e9),10), " ",
    String.pad_leading("#{cmed}",8), " ", String.pad_leading("#{cmin}",8)])
end

# clock sensitivity: how much does each quantity change from idle -> boost?
IO.puts("\nclock sensitivity at each size (idle ms / boost ms):")
IO.puts("  MiB  quantity    rep1    rep2   <- >1 means the quantity is GPU-clock-bound")
for mib <- sizes, qname <- ["flat_re","bcast_re","alloc","nx_mul"] do
  r = for rep <- 1..2 do
    [{_, i, _, _}] = :ets.lookup(results, {rep, mib, "idle", qname})
    [{_, b, _, _}] = :ets.lookup(results, {rep, mib, "boost", qname})
    i / b
  end
  IO.puts(["#{String.pad_leading("#{mib}",5)}  ", String.pad_trailing(qname,9),
           Enum.map(r, fn v -> String.pad_leading(f3.(v), 8) end)])
end

# slope per state, fixed cost separated from per-byte cost
IO.puts("\nslope per clock state (least squares over #{inspect sizes} MiB):")
IO.puts("  rep  state    quantity     GB/s   fixed cost ms")
for rep <- 1..2, {sname,_,_} <- states, qname <- ["flat_re","bcast_re","alloc","nx_mul"] do
  pts = for mib <- sizes do
    [{_, ms, mult, bytes}] = :ets.lookup(results, {rep, mib, sname, qname})
    {bytes/1.0, ms, mult}
  end
  mult = elem(hd(pts), 2)
  nn = length(pts)
  sx = Enum.sum(Enum.map(pts, &elem(&1,0))); sy = Enum.sum(Enum.map(pts, &elem(&1,1)))
  sxx = Enum.sum(Enum.map(pts, fn {x,_,_} -> x*x end))
  sxy = Enum.sum(Enum.map(pts, fn {x,y,_} -> x*y end))
  m = (nn*sxy - sx*sy)/(nn*sxx - sx*sx)
  ic = (sy - m*sx)/nn
  gbs = if m > 0, do: mult/(m/1000.0)/1.0e9, else: 0.0
  IO.puts([String.pad_leading("#{rep}",5), "  ", String.pad_trailing(sname,8), " ",
           String.pad_trailing(qname,9), String.pad_leading(f3.(gbs),9), "   ", f3.(ic)])
end
