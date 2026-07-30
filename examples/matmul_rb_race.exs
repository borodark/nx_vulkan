# Register-blocked (32×32, 2×2/thread) vs plain-tiled (16×16) matmul, per device.
#
#   mix run examples/matmul_rb_race.exs
#
# The tiled shaders are the shipped default (matmul_*.spv, 16×16, dispatched by
# NativeV.matmul). The register-blocked shaders (matmul_*_rb32.spv, dispatched by
# NativeV.matmul32) are NOT wired as default — they REGRESSED on Kepler (GT 650M):
# f32/f32acc 512³ 2.68×→1.78×, f64 15.8→22ms. Register blocking raises arithmetic
# intensity and typically helps modern GPUs (more registers/occupancy), so this
# benchmark exists to check whether RB wins on a given card before adopting it.
#
# All three variants are validated for correctness elsewhere; this only times.

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend
alias Nx.Vulkan.NativeV

spv = fn name -> Path.expand("priv/shaders/#{name}.spv") end
rnd = fn n -> for i <- 1..n, do: :math.sin(i * 0.01) end

# time `iters` dispatches of `nif` (:matmul tiled | :matmul32 rb) on the given spv
time = fn sz, type, which, spv_path, iters ->
  a = Nx.tensor(rnd.(sz * sz), type: type, backend: VulkanoBackend) |> Nx.reshape({sz, sz})
  b = Nx.tensor(rnd.(sz * sz), type: type, backend: VulkanoBackend) |> Nx.reshape({sz, sz})
  %{data: %{ref: ar}} = a
  %{data: %{ref: br}} = b
  eb = if type == {:f, 64}, do: 8, else: 4

  run = fn ->
    {:ok, o} = NativeV.buf_alloc(sz * sz * eb)
    :ok = apply(NativeV, which, [o, ar, br, sz, sz, sz, spv_path])
    o
  end

  for _ <- 1..5, do: run.()
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: run.() end)
  us / iters / 1000.0
end

{device, dtype} =
  case NativeV.device_name() do
    {:ok, name, t} -> {name, t}
    _ -> {"unknown", "unknown"}
  end

variants = [
  {"f64", {:f, 64}, "matmul_f64", "matmul_f64_rb32"},
  {"f32/f64acc", {:f, 32}, "matmul_f32_f64acc", "matmul_f32_f64acc_rb32"},
  {"f32/f32acc", {:f, 32}, "matmul_f32_f32acc", "matmul_f32_f32acc_rb32"}
]

IO.puts("\n  Device: #{device} (#{dtype})")
IO.puts("  variant       size    tiled ms   rb32 ms   rb/tiled")
IO.puts("  " <> String.duplicate("-", 54))

for {label, type, tiled, rb} <- variants do
  for {sz, it} <- [{512, 8}, {1024, 4}] do
    t = time.(sz, type, :matmul, spv.(tiled), it)
    r = time.(sz, type, :matmul32, spv.(rb), it)
    IO.puts(
      "  #{String.pad_trailing(label, 12)}#{String.pad_leading("#{sz}³", 6)}  " <>
        "#{String.pad_leading(Float.round(t, 2) |> to_string(), 8)}  " <>
        "#{String.pad_leading(Float.round(r, 2) |> to_string(), 8)}   " <>
        "#{String.pad_leading(Float.round(t / r, 2) |> to_string(), 5)}x"
    )
  end
end

IO.puts("\n  rb/tiled > 1.0 means register blocking is faster on this card.\n")
