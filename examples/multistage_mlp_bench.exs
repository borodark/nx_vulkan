# Thrust 3 — multi-stage split (dot boundaries): whole NN layers fuse.
#
# A graph with a `dot` (matmul) isn't a single fusable region, so before this it
# fell back to Nx.Defn.Evaluator (per-op dispatch). The compiler now splits it
# into a stage schedule: each matmul is a stage, and each maximal elementwise
# region (e.g. `relu(dot + bias)`) is ONE generated shader whose inputs may be
# earlier stages' GPU buffers. Intermediates stay on-device.
#
#   mix run examples/multistage_mlp_bench.exs
#
# Note: on matmul-dominated graphs the speedup over the (already on-GPU) eager
# path is modest — the matmul is the bottleneck and multi-stage saves the
# elementwise dispatch overhead + intermediates around it, not the matmul. The
# structural win is whole-graph compilation with no Evaluator fallback.

alias Nx.Vulkan.VulkanoBackend

{:ok, name, _} = Nx.Vulkan.NativeV.device_name()
IO.puts("device: #{name}\n")

key = fn shape, s -> Nx.iota(shape, type: :f32, backend: VulkanoBackend) |> Nx.multiply(s) end

best = fn f ->
  _ = f.() |> Nx.backend_transfer(Nx.BinaryBackend)

  1..5
  |> Enum.map(fn _ ->
    {us, _} = :timer.tc(fn -> for _ <- 1..30, do: f.() |> Nx.backend_transfer(Nx.BinaryBackend) end)
    us / 30 / 1000
  end)
  |> Enum.min()
  |> Float.round(3)
end

bench = fn label, fun, args ->
  fused = Nx.Defn.jit(fun, compiler: Nx.Vulkan.Compiler)
  ce = apply(fun, args) |> Nx.backend_transfer(Nx.BinaryBackend)
  cf = apply(fused, args) |> Nx.backend_transfer(Nx.BinaryBackend)
  err = Nx.subtract(ce, cf) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  e = best.(fn -> apply(fun, args) end)
  f = best.(fn -> apply(fused, args) end)
  IO.puts("#{label}: eager #{e} ms | multi-stage #{f} ms | #{Float.round(e / f, 2)}x  (err #{err})")
end

x = key.({256, 256}, 1.0e-4)
w1 = key.({256, 256}, 1.0e-4)
b1 = key.({256}, 1.0e-3)
w2 = key.({256, 64}, 1.0e-4)
b2 = key.({64}, 1.0e-3)

# single layer: relu(x @ W + b) — matmul stage + one fused epilogue stage
bench.(
  "relu(x@W1 + b1)          (2 stages)",
  fn a, ww, bb -> Nx.max(Nx.add(Nx.dot(a, ww), bb), 0.0) end,
  [x, w1, b1]
)

# 2-layer MLP forward — 4 stages: dot, fused tanh+b, dot, fused +b
bench.(
  "MLP: (relu(x@W1+b1))@W2+b2 (4 stages)",
  fn a, ww1, bb1, ww2, bb2 ->
    Nx.add(Nx.dot(Nx.max(Nx.add(Nx.dot(a, ww1), bb1), 0.0), ww2), bb2)
  end,
  [x, w1, b1, w2, b2]
)
