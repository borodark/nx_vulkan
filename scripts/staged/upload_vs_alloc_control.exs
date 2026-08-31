# CONTROL: is the 430 GB/s real, or is it reading uncommitted pages?
# buf_alloc does not initialise. If the driver backs an untouched buffer
# lazily, a shader reading it may never touch DRAM, and the bandwidth number
# is fiction. Compare against buffers filled with real uploaded data.
alias Nx.Vulkan.NativeV
sh = fn f -> Path.expand("priv/shaders/#{f}", File.cwd!()) end
flat_spv = sh.("elementwise_binary_f32.spv")
median = fn xs -> s = Enum.sort(xs); n = length(s)
  if rem(n,2)==1, do: Enum.at(s,div(n,2)), else: (Enum.at(s,div(n,2)-1)+Enum.at(s,div(n,2)))/2 end
f3 = fn v -> :erlang.float_to_binary(v/1.0, decimals: 3) end

bn = div(64*1024*1024, 4)
{:ok, ga} = NativeV.buf_alloc(bn*4); {:ok, go} = NativeV.buf_alloc(bn*4); :ok = NativeV.flush()
burst = fn budget ->
  stop = System.monotonic_time(:millisecond) + budget
  loop = fn self -> if System.monotonic_time(:millisecond) < stop do
    Enum.each(1..20, fn _ -> :ok = NativeV.apply_binary(go, ga, ga, bn, 1, flat_spv) end)
    :ok = NativeV.flush(); self.(self) end end
  loop.(loop)
end

IO.puts("\nflat_re at boost, uninitialised vs uploaded operands (median of 9, 2 reps)")
IO.puts("   MiB  operands       ms      3n GB/s")
for rep <- 1..2, mib <- [48, 64, 128] do
  n = div(mib*1024*1024, 4); bytes = n*4
  # varied non-zero payload so no page can be a shared zero page
  payload = :binary.copy(<<0x3F, 0x80, 0x00, 0x00, 0x40, 0x49, 0x0F, 0xDB>>, div(n, 2))
  {:ok, a_raw} = NativeV.buf_alloc(bytes); {:ok, b_raw} = NativeV.buf_alloc(bytes)
  {:ok, a_up} = NativeV.buf_upload(payload); {:ok, b_up} = NativeV.buf_upload(payload)
  {:ok, o} = NativeV.buf_alloc(bytes)
  :ok = NativeV.flush()

  for {tag, a, b} <- [{"buf_alloc", a_raw, b_raw}, {"uploaded ", a_up, b_up}] do
    burst.(700)
    xs = for _ <- 1..9 do
      burst.(40)
      t0 = System.monotonic_time(:microsecond)
      :ok = NativeV.apply_binary(o, a, b, n, 1, flat_spv); :ok = NativeV.flush()
      (System.monotonic_time(:microsecond)-t0)/1000.0
    end
    ms = median.(xs)
    IO.puts(["  rep#{rep} ", String.pad_leading("#{mib}",4), "  ", tag, " ",
      String.pad_leading(f3.(ms),8), " ", String.pad_leading(f3.(3*bytes/(ms/1000.0)/1.0e9),12)])
  end
  :erlang.garbage_collect()
end
