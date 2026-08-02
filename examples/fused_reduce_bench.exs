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

IO.puts("\nOutside the few-slot regime — falls back to eager by default (no regression):")
# narrow reduce (< 256) and non-contiguous (axes:[0], inner>1) fall back
bench.("  sum axes:[1] 4096x8   ", {4096, 8}, fn x, y -> Nx.sum(Nx.multiply(x, y), axes: [1]) end)
bench.("  sum axes:[0] 256x2048 ", {256, 2048}, fn x, y -> Nx.sum(Nx.multiply(x, y), axes: [0]) end)

# The many-slot wide-reduce regime is grid-stride-capable and wins on weak GPUs
# (~4.4x on GT 650M) but REGRESSES on strong ones (~0.44x on RTX 3060 Ti), so it
# is opt-in only. Force it here to show its (hardware-dependent) numbers.
System.put_env("NXV_FUSE_REDUCE", "1")
IO.puts("\nMany-slot WIDE reduce, forced via NXV_FUSE_REDUCE=1 (grid-stride, HW-dependent):")
bench.("  sum axes:[1] 4096x256 ", {4096, 256}, fn x, y -> Nx.sum(Nx.multiply(x, y), axes: [1]) end)
bench.("  sum axes:[1] 16384x256", {16384, 256}, fn x, y -> Nx.sum(Nx.multiply(x, y), axes: [1]) end)
System.delete_env("NXV_FUSE_REDUCE")
