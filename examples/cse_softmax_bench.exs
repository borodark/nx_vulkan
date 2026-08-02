# Thrust 3 — cross-stage CSE race: softmax / layernorm.
#
# A subexpression shared across a stage boundary (e.g. the softmax numerator
# `n = exp(x - max(x))`, used by both the final divide and the sum(n) reduce) is
# now MATERIALISED once and read as a buffer, instead of being re-inlined
# (recomputed) into every consumer. This trades an extra dispatch + buffer for
# the recompute — a genuine tradeoff whose winner depends on the shared node's
# cost vs dispatch overhead, so it MUST be raced across the fleet.
#
#   mix run examples/cse_softmax_bench.exs
#
# Compares, per shape: eager | fused CSE-on | fused CSE-off (NXV_CSE=0 forces the
# re-inline path in-process). Reports x-over-eager for each and the CSE on/off
# ratio (>1 => CSE helps here). Correctness is checked against BinaryBackend.

alias Nx.Vulkan.VulkanoBackend

{:ok, name, type} = Nx.Vulkan.NativeV.device_name()
IO.puts("device: #{name} (#{type})\n")

key = fn shape -> Nx.iota(shape, type: :f32, backend: VulkanoBackend) |> Nx.multiply(1.0e-3) end

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

# full softmax over the last axis: n = exp(x - max(x)); n / sum(n)
softmax = fn a ->
  n = Nx.exp(Nx.subtract(a, Nx.reduce_max(a, axes: [1], keep_axes: true)))
  Nx.divide(n, Nx.sum(n, axes: [1], keep_axes: true))
end

# Cross-stage CSE is default-OFF; NXV_CSE=1 opts the hoisting IN. jit compiles
# lazily on first call, so set the env, build a fresh jit, warm it, then measure.
run = fn label, fun, args ->
  ref = apply(fun, args) |> Nx.backend_transfer(Nx.BinaryBackend)

  System.put_env("NXV_CSE", "1")
  on = Nx.Defn.jit(fun, compiler: Nx.Vulkan.Compiler)
  con = apply(on, args) |> Nx.backend_transfer(Nx.BinaryBackend)
  System.delete_env("NXV_CSE")

  off = Nx.Defn.jit(fun, compiler: Nx.Vulkan.Compiler)
  coff = apply(off, args) |> Nx.backend_transfer(Nx.BinaryBackend)

  err_on = Nx.subtract(ref, con) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  err_off = Nx.subtract(ref, coff) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

  e = best.(fn -> apply(fun, args) end)

  System.put_env("NXV_CSE", "1")
  t_on = best.(fn -> apply(on, args) end)
  System.delete_env("NXV_CSE")

  t_off = best.(fn -> apply(off, args) end)

  IO.puts(
    "#{label}: eager #{e}ms | CSE-on #{t_on}ms (#{Float.round(e / t_on, 2)}x) | " <>
      "CSE-off #{t_off}ms (#{Float.round(e / t_off, 2)}x) | on/off #{Float.round(t_off / t_on, 2)}x " <>
      "(err on #{Float.round(err_on, 8)} / off #{Float.round(err_off, 8)})"
  )
end

for rows <- [64, 256, 1024], cols <- [64, 256, 1024] do
  run.("softmax {#{rows},#{cols}}", softmax, [key.({rows, cols})])
end
