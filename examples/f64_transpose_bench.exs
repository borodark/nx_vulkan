# Thrust 3 — f64 fusion + transpose-as-a-boundary vs eager per-op dispatch.
#
# Two increments to measure:
#   * transpose is now a stage, so `relu(x @ Wᵀ + b)` — the standard dense-layer
#     form — schedules as transpose + matmul + one fused epilogue instead of
#     dropping the whole graph to the Evaluator.
#   * f64 graphs fuse at all now (the codegen used to be f32-only).
#
# The eager column is the same function over VulkanoBackend tensors: every op is
# its own dispatch with its own intermediate buffer.
#
#   mix run examples/f64_transpose_bench.exs
#
# NOTE on method: consumer NVIDIA cards idle their SM clock (210 MHz vs 2100 on
# a 3060 Ti) and the f64 path in particular jitters heavily — a single timed
# burst measures clock ramp, not the kernel. Everything below is a MEDIAN of
# several rounds after a soak. Quote ratios from one process, never absolute ms
# across runs.

alias Nx.Vulkan.VulkanoBackend

{:ok, dev_name, dev_type} = Nx.Vulkan.NativeV.device_name()
{:ok, f64_ok} = Nx.Vulkan.NativeV.device_supports_f64()
host = :inet.gethostname() |> elem(1) |> to_string()
commit = System.cmd("git", ["rev-parse", "--short", "HEAD"]) |> elem(0) |> String.trim()

IO.puts("\n  Device: #{dev_name} (#{dev_type})   host=#{host}   commit=#{commit}")
IO.puts("  shaderFloat64: #{f64_ok}\n")

rounds = 7
iters = 20
med = fn l -> Enum.sort(l) |> Enum.at(div(length(l), 2)) end

# Two measures. "resident" leaves the result on the GPU — valid because every
# dispatch NIF blocks on wait_idle, so the work really has completed when the
# call returns. "+xfer" adds the download to host, which both columns pay
# equally but which dominates at 512x512 f64 (2 MB per iteration) and compresses
# every ratio toward 1.0. The resident column is the one that shows what the
# compiler actually changes.
time = fn f, transfer? ->
  run = if transfer?, do: fn -> f.() |> Nx.backend_transfer(Nx.BinaryBackend) end, else: f
  for _ <- 1..5, do: run.()

  for _ <- 1..rounds do
    {us, _} = :timer.tc(fn -> for _ <- 1..iters, do: run.() end)
    us / iters / 1000
  end
  |> med.()
end

mk = fn shape, type ->
  Nx.iota(shape, type: type, backend: VulkanoBackend) |> Nx.multiply(1.0e-3)
end

# relu(x @ Wᵀ + b) — dense layer with a transposed weight
dense_t = fn x, w, b -> Nx.max(Nx.add(Nx.dot(x, Nx.transpose(w)), b), 0.0) end
# relu(x @ W + b) — same layer, weight already in the dot's layout
dense = fn x, w, b -> Nx.max(Nx.add(Nx.dot(x, w), b), 0.0) end
matmul = fn x, w -> Nx.dot(x, w) end

cases =
  for n <- [128, 256, 512], type <- [{:f, 32}, {:f, 64}] do
    x = mk.({n, n}, type)
    w = mk.({n, n}, type)
    b = mk.({n}, type)
    label = fn nm -> "#{nm} #{n}x#{n} #{if type == {:f, 32}, do: "f32", else: "f64"}" end

    [
      {label.("relu(x@Wᵀ+b)"), fn -> dense_t.(x, w, b) end,
       Nx.Defn.jit(dense_t, compiler: Nx.Vulkan.Compiler) |> then(&fn -> &1.(x, w, b) end)},
      {label.("relu(x@W+b)"), fn -> dense.(x, w, b) end,
       Nx.Defn.jit(dense, compiler: Nx.Vulkan.Compiler) |> then(&fn -> &1.(x, w, b) end)},
      {label.("x@W"), fn -> matmul.(x, w) end,
       Nx.Defn.jit(matmul, compiler: Nx.Vulkan.Compiler) |> then(&fn -> &1.(x, w) end)}
    ]
  end
  |> List.flatten()

IO.puts(
  "  workload                     eager ms   fused ms  speedup | +xfer eager  fused  speedup | max_err"
)

IO.puts("  " <> String.duplicate("-", 104))

results =
  for {label, eager_f, fused_f} <- cases do
    e = time.(eager_f, false)
    f = time.(fused_f, false)
    ex = time.(eager_f, true)
    fx = time.(fused_f, true)

    got = fused_f.()
    on_gpu = match?(%VulkanoBackend{}, got.data)

    err =
      Nx.subtract(
        Nx.backend_transfer(got, Nx.BinaryBackend),
        Nx.backend_transfer(eager_f.(), Nx.BinaryBackend)
      )
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    r = fn v, d -> Float.round(v, d) end

    pad = fn v, w -> String.pad_leading(to_string(r.(v, 3)), w) end

    IO.puts(
      "  #{String.pad_trailing(label, 27)} #{pad.(e, 8)}   #{pad.(f, 8)}  " <>
        "#{String.pad_leading(to_string(r.(e / f, 2)) <> "x", 6)} | " <>
        "#{pad.(ex, 10)} #{pad.(fx, 6)}  #{String.pad_leading(to_string(r.(ex / fx, 2)) <> "x", 6)} | " <>
        "#{:erlang.float_to_binary(err * 1.0, [{:decimals, 10}])}"
    )

    %{
      workload: label,
      eager_ms: r.(e, 3),
      fused_ms: r.(f, 3),
      speedup: r.(e / f, 2),
      eager_ms_with_transfer: r.(ex, 3),
      fused_ms_with_transfer: r.(fx, 3),
      speedup_with_transfer: r.(ex / fx, 2),
      max_err: err,
      fused_on_gpu: on_gpu
    }
  end

report = %{
  device: dev_name,
  device_type: dev_type,
  hostname: host,
  commit: commit,
  shader_float64: f64_ok,
  method: "median of #{rounds} rounds x #{iters} iters after a 5-iter soak; resident columns leave the result on-device (dispatches block on wait_idle), +xfer columns add the host download",
  results: results
}

File.mkdir_p!("bench_results")
path = "bench_results/f64_transpose_#{host}_#{commit}.json"
File.write!(path, Jason.encode!(report, pretty: true))
IO.puts("\n  wrote #{path}")
