# Is the buf_upload cliff cumulative BAR1 pressure rather than buffer size?
# Hold N x 32 MiB uploaded buffers live and read each newest one from a shader.
# Prediction: full speed while cumulative <= BAR1 free (219 MiB here), then a
# fall to PCIe speed, with the SAME 32 MiB buffer size throughout.
alias Nx.Vulkan.NativeV
sh = fn f -> Path.expand("priv/shaders/#{f}", File.cwd!()) end
flat_spv = sh.("elementwise_binary_f32.spv")
median = fn xs -> s = Enum.sort(xs); n = length(s)
  if rem(n,2)==1, do: Enum.at(s,div(n,2)), else: (Enum.at(s,div(n,2)-1)+Enum.at(s,div(n,2)))/2 end
f3 = fn v -> :erlang.float_to_binary(v/1.0, decimals: 3) end
bar1 = fn -> {s,0} = System.cmd("nvidia-smi", ~w(-q)); 
  Regex.run(~r/BAR1 Memory Usage.*?Used\s+:\s+(\d+) MiB/s, s) |> then(fn [_, v] -> v end) end

bn = div(64*1024*1024, 4)
{:ok, ga} = NativeV.buf_alloc(bn*4); {:ok, go} = NativeV.buf_alloc(bn*4); :ok = NativeV.flush()
burst = fn budget ->
  stop = System.monotonic_time(:millisecond) + budget
  loop = fn self -> if System.monotonic_time(:millisecond) < stop do
    Enum.each(1..20, fn _ -> :ok = NativeV.apply_binary(go, ga, ga, bn, 1, flat_spv) end)
    :ok = NativeV.flush(); self.(self) end end
  loop.(loop)
end

mib = 32
n = div(mib*1024*1024, 4); bytes = n*4
payload = :binary.copy(<<0x3F,0x80,0x00,0x00,0x40,0x49,0x0F,0xDB>>, div(n,2))
{:ok, other} = NativeV.buf_alloc(bytes)
{:ok, o} = NativeV.buf_alloc(bytes)
:ok = NativeV.flush()

IO.puts("\n#{mib} MiB uploaded buffers, held live and accumulating.")
IO.puts("BAR1 total 256 MiB. Each row reads the NEWEST uploaded buffer via the shader.")
IO.puts("   N  cumulative   ms      3n GB/s   BAR1 used")
_held =
  Enum.reduce(1..10, [], fn i, held ->
    {:ok, up} = NativeV.buf_upload(payload)
    :ok = NativeV.flush()
    burst.(500)
    xs = for _ <- 1..9 do
      burst.(40)
      t0 = System.monotonic_time(:microsecond)
      :ok = NativeV.apply_binary(o, up, other, n, 1, flat_spv); :ok = NativeV.flush()
      (System.monotonic_time(:microsecond)-t0)/1000.0
    end
    ms = median.(xs)
    IO.puts([String.pad_leading("#{i}",4), String.pad_leading("#{i*mib} MiB",12), "  ",
      String.pad_leading(f3.(ms),7), " ", String.pad_leading(f3.(3*bytes/(ms/1000.0)/1.0e9),9),
      "   ", String.pad_leading(bar1.(), 6), " MiB"])
    [up | held]
  end)
IO.puts("\n(buffers still held: the list is returned so nothing is GC'd mid-sweep)")
