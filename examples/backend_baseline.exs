# Backend baseline: VulkanoBackend vs EXLA vs BinaryBackend.
#
# Times representative Nx / DL ops on every backend available in the running
# project, and checks Vulkano + EXLA against BinaryBackend for correctness. EXLA
# is optional — the script races whatever is loaded, so it runs as Vulkano-vs-
# Binary inside nx_vulkan, and as the full three-way in a project that also
# depends on {:exla, "~> 0.13"} (e.g. on a CUDA host). This is the thrust-1
# "how far from best" measurement (ROADMAP_NEXT_BEST_NX.md).
#
#   mix run examples/backend_baseline.exs

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend

exla? = Code.ensure_loaded?(EXLA.Backend)

backends =
  [{"binary", Nx.BinaryBackend}, {"vulkano", VulkanoBackend}] ++
    if exla?, do: [{"exla", EXLA.Backend}], else: []

rnd = fn n, s -> for i <- 1..n, do: :math.sin(s * 0.7 + i * 0.01) end
tf = fn list, shape, backend -> Nx.tensor(list, type: {:f, 32}, backend: backend) |> Nx.reshape(shape) end

# force a result to materialise (backend_copy 1 elem) so timing includes compute
force = fn t -> Nx.backend_copy(t, Nx.BinaryBackend) |> Nx.to_flat_list() |> hd() end

time = fn build, iters ->
  thunk = build.()
  _ = force.(thunk.())
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: force.(thunk.()) end)
  us / iters / 1000.0
end

maxdiff = fn a, b ->
  Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend), Nx.backend_copy(b, Nx.BinaryBackend))
  |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
end

# each workload: {label, iters, fn backend -> {thunk, result_for_correctness} end}
# Sizes kept moderate so pure-Elixir BinaryBackend finishes; the point is the
# relative EXLA-vs-Vulkano comparison. Bump these on a GPU host to see the gap
# widen (both GPUs pull away from Binary).
workloads = [
  {"matmul 256x256", 8, fn b ->
     a = tf.(rnd.(256 * 256, 1), {256, 256}, b)
     w = tf.(rnd.(256 * 256, 2), {256, 256}, b)
     fn -> Nx.dot(a, w) end
   end},
  {"conv 2x8x16x16 k16", 5, fn b ->
     i = tf.(rnd.(2 * 8 * 16 * 16, 1), {2, 8, 16, 16}, b)
     k = tf.(rnd.(16 * 8 * 3 * 3, 2), {16, 8, 3, 3}, b)
     fn -> Nx.conv(i, k) end
   end},
  {"tanh 100k", 20, fn b ->
     a = tf.(rnd.(100_000, 3), {100_000}, b)
     fn -> Nx.tanh(a) end
   end},
  {"sum 256x256", 20, fn b ->
     a = tf.(rnd.(65_536, 4), {256, 256}, b)
     fn -> Nx.sum(a) end
   end},
  {"mlp fwd 64x128->128->10", 8, fn b ->
     x = tf.(rnd.(64 * 128, 1), {64, 128}, b)
     w1 = tf.(rnd.(128 * 128, 2), {128, 128}, b)
     w2 = tf.(rnd.(128 * 10, 3), {128, 10}, b)
     fn ->
       h = Nx.dot(x, w1) |> Nx.max(0.0)
       Nx.dot(h, w2)
     end
   end}
]

{:ok, host} = :inet.gethostname()

device =
  case Nx.Vulkan.NativeV.device_name() do
    {:ok, name, _} -> name
    _ -> "?"
  end

IO.puts("\n  host=#{host}  vulkan=#{device}  exla=#{exla?}")
IO.puts("  workload                       " <> Enum.map_join(backends, "", fn {n, _} -> String.pad_leading(n <> " ms", 12) end) <> "   correctness")
IO.puts("  " <> String.duplicate("-", 30 + 12 * length(backends) + 16))

for {label, iters, build} <- workloads do
  results =
    for {name, backend} <- backends do
      {name, backend, time.(fn -> build.(backend) end, iters)}
    end

  # correctness: vulkano/exla vs binary
  {_, _, _} = hd(results)
  ref = (build.(Nx.BinaryBackend)).()
  errs =
    for {name, backend, _} <- results, name != "binary" do
      got = (build.(backend)).()
      "#{name}<#{Float.round(maxdiff.(got, ref), 6)}"
    end

  row =
    Enum.map_join(results, "", fn {_, _, ms} -> String.pad_leading(Float.round(ms, 3) |> to_string(), 12) end)

  IO.puts("  #{String.pad_trailing(label, 30)}#{row}   #{Enum.join(errs, " ")}")
end

IO.puts("")
