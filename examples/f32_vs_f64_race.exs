# f32 vs f64 matmul race on VulkanoBackend.
#
# IMPORTANT: this host runs Vulkan via llvmpipe (software, CPU). x86 does f64
# natively, so this measures mostly the memory-bandwidth / SIMD-width advantage
# of f32 (~2x ceiling), NOT the real-GPU story. On actual GPU hardware f64 is
# rate-limited to ~1/24 (GT 650M) .. 1/32 (consumer RTX) of f32, so the f32
# speedup there is far larger. Read this number as a lower bound.
#
#   mix run examples/f32_vs_f64_race.exs

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend

time_matmul = fn m, k, n, type, iters ->
  al = for i <- 1..(m * k), do: :math.sin(i * 0.01)
  bl = for i <- 1..(k * n), do: :math.cos(i * 0.01)
  av = Nx.tensor(al, type: type, backend: VulkanoBackend) |> Nx.reshape({m, k})
  bv = Nx.tensor(bl, type: type, backend: VulkanoBackend) |> Nx.reshape({k, n})

  # warmup (pipeline build + cache)
  _ = Nx.dot(av, bv)

  {us, _} =
    :timer.tc(fn ->
      for _ <- 1..iters, do: Nx.dot(av, bv)
    end)

  us / iters / 1000.0
end

IO.puts("\n  size (m×k×n)      f64 ms   f32 ms   speedup   (llvmpipe/CPU — lower bound)")
IO.puts("  " <> String.duplicate("-", 68))

for {m, k, n, iters} <- [{128, 128, 128, 20}, {256, 256, 256, 10}, {512, 512, 512, 4}] do
  f64 = time_matmul.(m, k, n, {:f, 64}, iters)
  f32 = time_matmul.(m, k, n, {:f, 32}, iters)
  speedup = f64 / f32

  IO.puts(
    "  #{String.pad_trailing("#{m}×#{k}×#{n}", 16)} " <>
      "#{String.pad_leading(Float.round(f64, 2) |> to_string(), 7)}  " <>
      "#{String.pad_leading(Float.round(f32, 2) |> to_string(), 7)}   " <>
      "#{String.pad_leading(Float.round(speedup, 2) |> to_string(), 5)}x"
  )
end

IO.puts("")
