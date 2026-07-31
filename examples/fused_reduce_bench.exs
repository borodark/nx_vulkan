# Thrust 3 — parallel fused reduce vs eager.
#
# The eager backend's `reduce_axis` is one-thread-per-output-slot, so a FULL
# reduction (one slot) runs single-threaded — this is exactly where EXLA out-ran
# the eager backend (sum 5.6x in bench_results/EXLA_BASELINE.md). The fusion
# compiler generates a parallel workgroup-per-slot shared-memory tree reduce and
# fuses the elementwise chain into it, so the whole reduce axis is reduced by a
# 256-thread workgroup in one dispatch.
#
#   mix run examples/fused_reduce_bench.exs

alias Nx.Vulkan.VulkanoBackend

{:ok, name, _} = Nx.Vulkan.NativeV.device_name()
IO.puts("device: #{name}\n")

best = fn f ->
  _ = f.() |> Nx.backend_transfer(Nx.BinaryBackend)

  1..5
  |> Enum.map(fn _ ->
    {us, _} = :timer.tc(fn -> for _ <- 1..20, do: f.() |> Nx.backend_transfer(Nx.BinaryBackend) end)
    us / 20 / 1000
  end)
  |> Enum.min()
  |> Float.round(3)
end

bench = fn label, shape, chain ->
  a = Nx.iota(shape, type: :f32, backend: VulkanoBackend) |> Nx.multiply(1.0e-6)
  b = Nx.add(a, 0.5)
  fused = Nx.Defn.jit(chain, compiler: Nx.Vulkan.Compiler)

  ce = chain.(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
  cf = fused.(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
  err = Nx.subtract(ce, cf) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

  e = best.(fn -> chain.(a, b) end)
  f = best.(fn -> fused.(a, b) end)
  IO.puts("#{label}: eager #{e} ms | fused #{f} ms | #{Float.round(e / f, 2)}x  (err #{err})")
end

IO.puts("Full reductions (few slots — fused by default):")
bench.("  sum(a*b) 256x256      ", {256, 256}, fn x, y -> Nx.sum(Nx.multiply(x, y)) end)
bench.("  sum(a*b) 1024x1024    ", {1024, 1024}, fn x, y -> Nx.sum(Nx.multiply(x, y)) end)
bench.("  sum(tanh(a*b+a)) 512² ", {512, 512}, fn x, y -> Nx.sum(Nx.tanh(Nx.add(Nx.multiply(x, y), x))) end)

IO.puts("\nMany-slot per-axis reduction (falls back to eager by default — no regression):")
bench.("  sum axes:[1] 2048x256 ", {2048, 256}, fn x, y -> Nx.sum(Nx.multiply(x, y), axes: [1]) end)
