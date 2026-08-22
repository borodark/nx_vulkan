# Why `{:reduce, 5}` is on Nx.Vulkan.Fallback's allowlist rather than
# implemented. Run with: mix run bench/reduce_fold_vs_host.exs
#
# `Nx.reduce/4` takes an arbitrary user fun, so no shader can express it. The
# obvious workaround is to vectorise the fold: view the tensor as
# (outer, reduce_size, inner), slice one plane per step, and evaluate the fun on
# resident tensors so it composes GPU ops — the same move W4 made for
# Nx.Block.*. This measures that against the host path it would replace.
#
# It loses at every size, and the gap widens with the reduced axis, because the
# cost is per-dispatch launch overhead rather than compute. A log2-step tree
# reduce would fix it but needs the fun to be ASSOCIATIVE, which Nx.reduce does
# not guarantee — it is a left fold.
#
# Measured on super-io (RTX 3060 Ti), 2026-08-22:
#
#   reduce_size      fold      host
#             8   0.97 ms   0.19 ms
#            64   6.12 ms   3.02 ms
#           512  39.81 ms  22.01 ms
#          4096 440.62 ms  37.40 ms
alias Nx.Vulkan.VulkanoBackend, as: V
alias Nx.BinaryBackend, as: B

fold = fn t, acc0, reduce_size, outer, inner, fun ->
  t3 = Nx.reshape(t, {outer, reduce_size, inner})
  acc = Nx.broadcast(Nx.tensor(acc0, type: Nx.type(t), backend: V), {outer, inner})

  Enum.reduce(0..(reduce_size - 1), acc, fn r, a ->
    plane = Nx.slice(t3, [0, r, 0], [outer, 1, inner]) |> Nx.reshape({outer, inner})
    fun.(plane, a)
  end)
end

bench = fn label, f ->
  f.()
  {us, res} = :timer.tc(fn -> for _ <- 1..5, do: f.() end)
  IO.puts("  #{label}: #{Float.round(us / 5 / 1000, 2)} ms  (#{inspect(res |> List.last() |> Nx.to_flat_list() |> Enum.take(2))})")
end

for {n_red, inner} <- [{8, 4}, {64, 16}, {512, 16}, {4096, 4}] do
  IO.puts("reduce_size=#{n_red}, outer=1, inner=#{inner}  (#{n_red * inner} elements)")
  data = for i <- 1..(n_red * inner), do: rem(i, 7)
  gt = Nx.tensor(data, type: {:s, 32}, backend: V) |> Nx.reshape({n_red, inner})

  bench.("on-device fold ", fn ->
    fold.(gt, 0, n_red, 1, inner, fn x, y -> Nx.add(x, y) end) |> Nx.backend_transfer(B)
  end)

  bench.("host fallback  ", fn ->
    Nx.reduce(Nx.backend_transfer(Nx.backend_copy(gt, B), B), 0, [axes: [0]], fn x, y -> Nx.add(x, y) end)
  end)
end
