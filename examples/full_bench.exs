# Comprehensive bench: per-op + end-to-end + robustness across all
# available Nx backends. Designed to run from any project that has
# the dependencies it needs (Axon, EXLA where applicable, nx_vulkan).
#
# Invoke from exmc/ on super-io (has EXLA available) or from
# nx_vulkan/ on mac-247 (no EXLA on FreeBSD).
#
# Usage:
#   cd ~/projects/learn_erl/pymc/exmc && mix run /tmp/full_bench.exs
#   cd ~/exmc-r22/exmc          && mix run /tmp/full_bench.exs

defmodule FullBench do
  @hosts_with_exla ["super-io"]

  def main do
    {hostname, 0} = System.cmd("hostname", ["-s"])
    host = String.trim(hostname)

    IO.puts("\n========================================")
    IO.puts("HOST: #{host}")
    IO.puts("DATE: #{DateTime.utc_now() |> DateTime.to_iso8601()}")
    IO.puts("========================================\n")

    backends = available_backends(host)
    IO.puts("backends available: #{inspect(backends)}\n")

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
      if Code.ensure_loaded?(Nx.Vulkan.Backend) and host in ["super-io", "mac"] do
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
    IO.puts("=== BENCH A: per-op latency curves ===\n")

    matmul_sizes = [16, 64, 256, 1024]
    matmul_reps = [200, 200, 100, 30]

    for backend <- backends do
      IO.puts("backend: #{elem(backend, 0)}")

      for {m, reps} <- Enum.zip(matmul_sizes, matmul_reps) do
        time_op("  matmul #{m}×#{m}", reps, fn ->
          a = make_tensor({m, m}, elem(backend, 1))
          b = make_tensor({m, m}, elem(backend, 1))
          Nx.dot(a, b)
        end)
      end

      time_op("  add 16k", 500, fn ->
        a = make_tensor({16384}, elem(backend, 1))
        b = make_tensor({16384}, elem(backend, 1))
        Nx.add(a, b)
      end)

      time_op("  sigmoid 16k", 500, fn ->
        a = make_tensor({16384}, elem(backend, 1))
        Nx.sigmoid(a)
      end)

      time_op("  sum 1024×1024", 200, fn ->
        a = make_tensor({1024, 1024}, elem(backend, 1))
        Nx.sum(a)
      end)

      IO.puts("")
    end
  end

  # ---- Bench B: end-to-end workloads ----

  defp bench_b(backends) do
    IO.puts("=== BENCH B: end-to-end workloads ===\n")

    for backend <- backends do
      {name, mod} = backend
      IO.puts("backend: #{name}")

      bench_axon_training_step(mod)
      bench_regime_log_p(mod)

      IO.puts("")
    end
  end

  defp bench_axon_training_step(backend_mod) do
    unless Code.ensure_loaded?(Axon) do
      IO.puts("  (Axon not loaded — skip training step)")
      :ok
    else
      model =
        Axon.input("x", shape: {nil, 8})
        |> Axon.dense(16, activation: :sigmoid)
        |> Axon.dense(2)

      {init_fn, predict_fn} = Axon.build(model, mode: :train)
      params = init_fn.(%{"x" => Nx.template({32, 8}, :f32)}, Axon.ModelState.empty())

      x = make_tensor({32, 8}, backend_mod)
      y = make_tensor({32, 2}, backend_mod)
      params = transfer_state(params, backend_mod)

      grad_fn = fn p, x_in, y_in ->
        Nx.Defn.value_and_grad(p, fn pp ->
          out = predict_fn.(pp, %{"x" => x_in}).prediction
          d = Nx.subtract(out, y_in)
          Nx.divide(Nx.sum(Nx.multiply(d, d)), Nx.tensor(32.0))
        end)
      end

      time_op("  Axon training step", 100, fn ->
        Nx.Defn.jit_apply(grad_fn, [params, x, y], compiler: Nx.Defn.Evaluator)
      end)
    end
  end

  defp bench_regime_log_p(backend_mod) do
    unless Code.ensure_loaded?(Exmc.Trading.RegimeModel) do
      IO.puts("  (Exmc not loaded — skip regime)")
      :ok
    else
      returns = for _ <- 1..200, do: :rand.uniform() * 0.02 - 0.01
      {ir, _} = Exmc.Trading.RegimeModel.build(returns, num_samples: 1, num_warmup: 1, ncp: false)
      {:ok, comps} = Exmc.NUTS.CustomSynth.extract_components(ir)
      fun = Exmc.NUTS.CustomSynth.MultiRvCustomSpec.compose_logp_defn(comps)

      q_list = [0.01, 0.05, 0.02, 0.05, 0.02, 0.05, 0.05, 0.05]
      q = Nx.tensor(q_list, type: :f32, backend: backend_mod)
      obs = Nx.tensor(returns, type: :f32, backend: backend_mod)

      time_op("  exmc regime log_p", 50, fn ->
        Nx.Defn.jit_apply(fun, [q, obs], compiler: Nx.Defn.Evaluator)
      end)
    end
  end

  # ---- Bench C: 10k-iteration robustness ----

  defp bench_c(backends) do
    IO.puts("=== BENCH C: robustness (5000 mixed dispatches) ===\n")

    for {name, mod} <- backends do
      IO.puts("backend: #{name}")
      run_robustness(mod)
      IO.puts("")
    end
  end

  defp run_robustness(backend_mod) do
    n = 5000
    a = make_tensor({128, 128}, backend_mod)

    try do
      {micros, _} =
        :timer.tc(fn ->
          Enum.reduce(1..n, a, fn _, acc ->
            Nx.dot(acc, a) |> Nx.sigmoid() |> Nx.divide(Nx.tensor(2.0))
          end)
        end)

      per_iter = micros / n / 1000.0
      IO.puts("  #{n} iter: #{Float.round(micros / 1_000_000, 1)}s total, #{Float.round(per_iter, 3)} ms/iter")
    rescue
      e ->
        IO.puts("  CRASHED at some iteration: #{Exception.message(e)}")
    catch
      kind, reason ->
        IO.puts("  CAUGHT #{kind}: #{inspect(reason)}")
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
    per_iter = micros / n_iter / 1000.0
    IO.puts("#{label}: #{Float.round(per_iter, 3)} ms/iter (n=#{n_iter})")
  end
end

FullBench.main()
