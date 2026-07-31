# Thrust 3 — fused JIT vs eager per-op dispatch.
#
# A long elementwise chain over a large f32 tensor. The eager VulkanoBackend
# issues one GPU dispatch (and one intermediate buffer) PER op; the fusion
# compiler generates a single shader for the whole chain and dispatches it once.
# This measures the launch-overhead + memory-traffic win that is EXLA's edge.
#
#   mix run examples/fused_jit_bench.exs

alias Nx.Vulkan.VulkanoBackend

{:ok, name, _type} = Nx.Vulkan.NativeV.device_name()
IO.puts("device: #{name}\n")

n = 1_000_000
a = Nx.iota({n}, type: :f32, backend: VulkanoBackend) |> Nx.multiply(1.0e-6)
b = Nx.add(a, 0.5)

# A 10-op elementwise chain: tanh(sigmoid(a*b + a) * b - a) + sqrt(a*a + b*b)
chain = fn x, y ->
  left = Nx.tanh(Nx.subtract(Nx.multiply(Nx.sigmoid(Nx.add(Nx.multiply(x, y), x)), y), x))
  right = Nx.sqrt(Nx.add(Nx.multiply(x, x), Nx.multiply(y, y)))
  Nx.add(left, right)
end

fused = Nx.Defn.jit(chain, compiler: Nx.Vulkan.Compiler)

# warmup (also triggers shader generation + compile for the fused path)
_ = chain.(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
_ = fused.(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)

time = fn f ->
  iters = 50
  {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: f.() |> Nx.backend_transfer(Nx.BinaryBackend) end)
  Float.round(us / iters / 1000, 3)
end

eager_ms = time.(fn -> chain.(a, b) end)
fused_ms = time.(fn -> fused.(a, b) end)

# correctness: fused vs eager
ce = chain.(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
cf = fused.(a, b) |> Nx.backend_transfer(Nx.BinaryBackend)
max_err = Nx.subtract(ce, cf) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()

IO.puts("n = #{n}, 10-op elementwise chain")
IO.puts("eager (per-op dispatch): #{eager_ms} ms")
IO.puts("fused (1 dispatch):      #{fused_ms} ms")
IO.puts("speedup:                 #{Float.round(eager_ms / fused_ms, 2)}x")
IO.puts("max abs error:           #{max_err}")
