# f32 vs f64 race on VulkanoBackend, across the GPU compute families.
#
# IMPORTANT: this host runs Vulkan via llvmpipe (software, CPU). x86 does f64
# natively and the f32 kernels still accumulate in f64, so this mostly measures
# the memory-bandwidth / SIMD-width edge of f32 (~2x ceiling), NOT the real-GPU
# story. On actual GPU hardware f64 is rate-limited to ~1/24 (GT 650M) .. 1/32
# (consumer RTX) of f32, so the f32 speedup there is far larger. Read these
# numbers as a lower bound.
#
#   mix run examples/f32_vs_f64_race.exs

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend

rnd = fn n, s -> for i <- 1..n, do: :math.sin(s * 0.7 + i * 0.013) end
tf = fn list, shape, type -> Nx.tensor(list, type: type, backend: VulkanoBackend) |> Nx.reshape(shape) end

# time one op (a 0-arity thunk) over `iters` runs, in ms/op; also report whether
# the result stayed on the GPU (i.e. did not host-fall-back).
run = fn thunk, iters ->
  r = thunk.()  # warmup + on_gpu probe
  on_gpu = match?(%VulkanoBackend{}, r.data)
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: thunk.() end)
  {us / iters / 1000.0, on_gpu}
end

row = fn label, f64ms, f32ms, gpu ->
  IO.puts(
    "  #{String.pad_trailing(label, 24)}" <>
      "#{String.pad_leading(Float.round(f64ms, 3) |> to_string(), 9)}  " <>
      "#{String.pad_leading(Float.round(f32ms, 3) |> to_string(), 9)}   " <>
      "#{String.pad_leading(Float.round(f64ms / f32ms, 2) |> to_string(), 5)}x   " <>
      "#{gpu}"
  )
end

# race one op-builder across dtypes. build.(type) returns a 0-arity thunk.
race = fn label, iters, build ->
  {f64ms, g64} = run.(build.({:f, 64}), iters)
  {f32ms, g32} = run.(build.({:f, 32}), iters)
  row.(label, f64ms, f32ms, g64 and g32)
end

IO.puts("\n  op                         f64 ms     f32 ms   speedup   on_gpu   (llvmpipe/CPU — lower bound)")
IO.puts("  " <> String.duplicate("-", 82))

# ---- matmul ----
for {m, k, n, iters} <- [{128, 128, 128, 20}, {256, 256, 256, 10}, {512, 512, 512, 4}] do
  race.("matmul #{m}×#{k}×#{n}", iters, fn type ->
    a = tf.(rnd.(m * k, 1), {m, k}, type)
    b = tf.(rnd.(k * n, 2), {k, n}, type)
    fn -> Nx.dot(a, b) end
  end)
end

# ---- conv ----
for {ishape, kshape, iters} <- [
      {{1, 8, 24, 24}, {16, 8, 3, 3}, 10},
      {{1, 16, 32, 32}, {32, 16, 3, 3}, 4}
    ] do
  race.("conv #{elem(ishape, 1)}→#{elem(kshape, 0)}ch #{elem(ishape, 2)}²", iters, fn type ->
    iv = tf.(rnd.(Tuple.product(ishape), 1), ishape, type)
    kv = tf.(rnd.(Tuple.product(kshape), 2), kshape, type)
    fn -> Nx.conv(iv, kv) end
  end)
end

# ---- elementwise ----
n_el = 1_000_000

race.("elementwise add 1M", 20, fn type ->
  a = tf.(rnd.(n_el, 1), {n_el}, type)
  b = tf.(rnd.(n_el, 2), {n_el}, type)
  fn -> Nx.add(a, b) end
end)

race.("elementwise tanh 1M", 20, fn type ->
  a = tf.(rnd.(n_el, 3), {n_el}, type)
  fn -> Nx.tanh(a) end
end)

# ---- reductions ----
race.("sum 1M (full)", 20, fn type ->
  a = tf.(rnd.(n_el, 4), {n_el}, type)
  fn -> Nx.sum(a) end
end)

race.("sum 1024×1024 axis0", 20, fn type ->
  a = tf.(rnd.(1_048_576, 5), {1024, 1024}, type)
  fn -> Nx.sum(a, axes: [0]) end
end)

IO.puts("")
