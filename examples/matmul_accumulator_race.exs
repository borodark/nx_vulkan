# Accumulator width vs speed for f32 matmul, on the actual device.
#
# Times three matmul shaders head-to-head:
#   * matmul_f64            — f64 storage + f64 accumulator (the default)
#   * matmul_f32_f64acc     — f32 storage + f64 accumulator (accuracy-safe f32)
#   * matmul_f32_naive      — f32 storage + f32 accumulator (fast f32)
#
#   mix run examples/matmul_accumulator_race.exs
#
# On a GPU whose f64 units are rate-limited (Kepler ~1/24, consumer Ampere
# ~1/32 of f32), the f64 ACCUMULATOR — not the storage — is the bottleneck for
# compute-bound matmul: matmul_f32_f64acc does the same rate-limited f64 MACs as
# matmul_f64 (plus f32->f64 conversions), so it is *slower* than f64, while the
# pure-f32 accumulator is faster. This is the accuracy/speed knob the f32 path
# needs for compute-bound ops (see F32_PLAN.md).

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend
alias Nx.Vulkan.NativeV

f64_spv = Path.expand("priv/shaders/matmul_f64.spv")
f32acc_spv = Path.expand("priv/shaders/matmul_f32_f64acc.spv")
f32naive_spv = Path.expand("priv/shaders/matmul_f32_naive.spv")

rnd = fn n -> for i <- 1..n, do: :math.sin(i * 0.01) end

time = fn m, k, n, type, spv, iters ->
  a = Nx.tensor(rnd.(m * k), type: type, backend: VulkanoBackend) |> Nx.reshape({m, k})
  b = Nx.tensor(rnd.(k * n), type: type, backend: VulkanoBackend) |> Nx.reshape({k, n})
  %{data: %{ref: ar}} = a
  %{data: %{ref: br}} = b
  eb = if type == {:f, 64}, do: 8, else: 4

  run = fn ->
    {:ok, o} = NativeV.buf_alloc(m * n * eb)
    :ok = NativeV.matmul(o, ar, br, m, n, k, spv)
    o
  end

  run.()
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: run.() end)
  us / iters / 1000.0
end

{device, dtype} =
  case NativeV.device_name() do
    {:ok, name, t} -> {name, t}
    _ -> {"unknown", "unknown"}
  end

IO.puts("\n  Device: #{device} (#{dtype})")
IO.puts("  matmul       f64 ms   f32/f64acc (x)     f32/f32acc (x)")
IO.puts("  " <> String.duplicate("-", 56))

for {m, k, n, it} <- [{256, 256, 256, 10}, {512, 512, 512, 4}] do
  f64 = time.(m, k, n, {:f, 64}, f64_spv, it)
  facc = time.(m, k, n, {:f, 32}, f32acc_spv, it)
  fnv = time.(m, k, n, {:f, 32}, f32naive_spv, it)

  IO.puts(
    "  #{String.pad_trailing("#{m}x#{k}x#{n}", 12)}" <>
      "#{String.pad_leading(Float.round(f64, 2) |> to_string(), 6)}   " <>
      "#{String.pad_leading(Float.round(facc, 2) |> to_string(), 7)} (#{Float.round(f64 / facc, 2)}x)     " <>
      "#{String.pad_leading(Float.round(fnv, 2) |> to_string(), 7)} (#{Float.round(f64 / fnv, 2)}x)"
  )
end

IO.puts("")
