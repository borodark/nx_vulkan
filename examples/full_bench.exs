# Multi-backend bench: per-op + end-to-end + robustness. Iteration
# counts and sizes are PER-BACKEND-PER-WORKLOAD so BinaryBackend
# doesn't eat hours of wall clock on a 1024×1024 matmul.

defmodule FullBench do
  @hosts_with_exla ["super-io"]

  # Per-backend matmul scaling: (size, reps).
  # BinaryBackend caps at 256 because larger is hours of CPU.
  # GPU backends go up to 1024 with low rep counts at the top.
  @matmul_sched %{
    "BinaryBackend" => [{16, 100}, {64, 50}, {128, 20}, {256, 8}],
    "VulkanoBackend" => [{16, 200}, {64, 200}, {256, 100}, {1024, 30}],
    "spirit" => [{16, 200}, {64, 200}, {256, 100}, {1024, 30}],
    "EXLA" => [{16, 200}, {64, 200}, {256, 100}, {1024, 50}]
  }

  def main do
    {hostname, 0} = System.cmd("hostname", ["-s"])
    host = String.trim(hostname)
    IO.puts("\n========================================")
    IO.puts("HOST: #{host}")
    IO.puts("DATE: #{DateTime.utc_now() |> DateTime.to_iso8601()}")
    IO.puts("========================================\n")

    backends = available_backends(host)
    IO.puts("backends: #{Enum.map(backends, &elem(&1, 0)) |> Enum.join(", ")}\n")

    bench_a(backends)
    bench_b(backends)
    bench_c(backends)
  end

  defp available_backends(host) do
    base = [
      {"BinaryBackend", Nx.BinaryBackend},
      {"VulkanoBackend", Nx.Vulkan.VulkanoBackend}
    ]

    base =
      if Code.ensure_loaded?(Nx.Vulkan.Backend) do
        base ++ [{"spirit", Nx.Vulkan.Backend}]
      else
        base
      end

    if host in @hosts_with_exla and Code.ensure_loaded?(EXLA) do
      base ++ [{"EXLA", EXLA.Backend}]
    else
      base
    end
  end

  # ---- Bench A: per-op latency curves ----

  defp bench_a(backends) do
    IO.puts("=== BENCH A: per-op latency ===")

    for {name, mod} <- backends do
      IO.puts("\n[#{name}]")

      sched = Map.get(@matmul_sched, name, [{16, 100}])

      for {m, reps} <- sched do
        time_op("matmul #{m}", reps, fn ->
          a = make_tensor({m, m}, mod)
          b = make_tensor({m, m}, mod)
          Nx.dot(a, b)
        end)
      end

      add_size = if name == "BinaryBackend", do: 4096, else: 16384
      time_op("add #{add_size}", 100, fn ->
        a = make_tensor({add_size}, mod)
        b = make_tensor({add_size}, mod)
        Nx.add(a, b)
      end)

      sig_size = if name == "BinaryBackend", do: 4096, else: 16384
      time_op("sigmoid #{sig_size}", 100, fn ->
        a = make_tensor({sig_size}, mod)
        Nx.sigmoid(a)
      end)

      sum_size = if name == "BinaryBackend", do: 256, else: 1024
      time_op("sum #{sum_size}×#{sum_size}", 50, fn ->
        a = make_tensor({sum_size, sum_size}, mod)
        Nx.sum(a)
      end)
    end
  end

  # ---- Bench B: end-to-end workloads ----

  defp bench_b(backends) do
    IO.puts("\n\n=== BENCH B: end-to-end ===")

    for {name, mod} <- backends do
      IO.puts("\n[#{name}]")
      bench_axon_training_step(mod)
      bench_regime_log_p(mod, name)
    end
  end

  defp bench_axon_training_step(backend_mod) do
    if Code.ensure_loaded?(Axon) do
      model =
        Axon.input("x", shape: {nil, 8})
        |> Axon.dense(16, activation: :sigmoid)
        |> Axon.dense(2)

      {init_fn, predict_fn} = Axon.build(model, mode: :train)
      params = init_fn.(%{"x" => Nx.template({32, 8}, :f32)}, Axon.ModelState.empty())
      params = transfer_state(params, backend_mod)

      x = make_tensor({32, 8}, backend_mod)
      y = make_tensor({32, 2}, backend_mod)

      grad_fn = fn p, x_in, y_in ->
        Nx.Defn.value_and_grad(p, fn pp ->
          out = predict_fn.(pp, %{"x" => x_in}).prediction
          d = Nx.subtract(out, y_in)
          Nx.divide(Nx.sum(Nx.multiply(d, d)), Nx.tensor(32.0))
        end)
      end

      time_op("Axon training step", 30, fn ->
        Nx.Defn.jit_apply(grad_fn, [params, x, y], compiler: Nx.Defn.Evaluator)
      end)
    end
  end

  defp bench_regime_log_p(backend_mod, _name) do
    if Code.ensure_loaded?(Exmc.Trading.RegimeModel) do
      returns = for _ <- 1..200, do: :rand.uniform() * 0.02 - 0.01
      {ir, _} = Exmc.Trading.RegimeModel.build(returns, num_samples: 1, num_warmup: 1, ncp: false)
      {:ok, comps} = Exmc.NUTS.CustomSynth.extract_components(ir)
      fun = Exmc.NUTS.CustomSynth.MultiRvCustomSpec.compose_logp_defn(comps)

      q = Nx.tensor([0.01, 0.05, 0.02, 0.05, 0.02, 0.05, 0.05, 0.05], type: :f32, backend: backend_mod)
      obs = Nx.tensor(returns, type: :f32, backend: backend_mod)

      time_op("exmc regime log_p", 20, fn ->
        Nx.Defn.jit_apply(fun, [q, obs], compiler: Nx.Defn.Evaluator)
      end)
    end
  end

  # ---- Bench C: robustness ----

  defp bench_c(backends) do
    IO.puts("\n\n=== BENCH C: robustness (5000 mixed dispatches) ===")

    for {name, mod} <- backends do
      IO.puts("\n[#{name}]")

      # BinaryBackend would take forever for 128×128 matmul; size down.
      size = if name == "BinaryBackend", do: 32, else: 128
      n = if name == "BinaryBackend", do: 500, else: 5000

      a = make_tensor({size, size}, mod)

      try do
        {micros, _} =
          :timer.tc(fn ->
            Enum.reduce(1..n, a, fn _, acc ->
              Nx.dot(acc, a) |> Nx.sigmoid() |> Nx.divide(Nx.tensor(2.0))
            end)
          end)

        per = micros / n / 1000.0
        IO.puts("#{n} iter, size #{size}×#{size}: #{Float.round(micros / 1_000_000, 1)}s total, #{Float.round(per, 3)} ms/iter — OK")
      rescue
        e -> IO.puts("CRASHED: #{Exception.message(e)}")
      catch
        k, r -> IO.puts("CAUGHT #{k}: #{inspect(r)}")
      end
    end
  end

  # ---- helpers ----

  defp make_tensor(shape, backend) do
    n = shape |> Tuple.to_list() |> Enum.reduce(1, &*/2)

    Nx.iota({n}, type: :f32, backend: Nx.BinaryBackend)
    |> Nx.divide(Nx.tensor(n * 1.0))
    |> Nx.reshape(shape)
    |> Nx.backend_transfer(backend)
  end

  defp transfer_state(model_state, backend) do
    %{
      model_state
      | data:
          Map.new(model_state.data, fn {layer, params} ->
            {layer, Map.new(params, fn {k, v} -> {k, Nx.backend_transfer(v, backend)} end)}
          end)
    }
  end

  defp time_op(label, n_iter, fun) do
    fun.()
    fun.()
    {micros, _} = :timer.tc(fn -> for _ <- 1..n_iter, do: fun.() end)
    per = micros / n_iter / 1000.0
    IO.puts("  #{label}: #{Float.round(per, 3)} ms/iter")
  end
end

FullBench.main()
