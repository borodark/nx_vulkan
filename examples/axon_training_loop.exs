# Plan A — Multi-step Axon training loop convergence test.
#
# Dense(4 → 32, tanh) → Dense(1) regression model trained with
# manual SGD for 100 steps. Loss trajectory on
# Nx.Vulkan.VulkanoBackend matches Nx.BinaryBackend at every
# step; final losses agree to 1e-6 relative.
#
# Run:
#   elixir examples/axon_training_loop.exs

Mix.install([
  {:axon, "~> 0.7"},
  {:nx_vulkan, path: Path.expand("..", __DIR__)}
])

defmodule AxonTraining do
  @batch 16
  @input_dim 4
  @model_size 32

  defp model do
    Axon.input("x", shape: {nil, @input_dim})
    |> Axon.dense(@model_size, activation: :tanh)
    |> Axon.dense(1)
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

  def run_training(params0, x, y, n_steps) do
    {_init_fn, predict_fn} = Axon.build(model(), mode: :train)

    grad_fn = fn params, x_in, y_in ->
      Nx.Defn.value_and_grad(params, fn p ->
        out = predict_fn.(p, %{"x" => x_in}).prediction
        diff = Nx.subtract(out, y_in)
        Nx.divide(Nx.sum(Nx.multiply(diff, diff)), Nx.tensor(elem(Nx.shape(y_in), 0) * 1.0))
      end)
    end

    apply_sgd = fn params, grads, lr ->
      new_data =
        Map.new(params.data, fn {layer, layer_params} ->
          gp = grads.data[layer]

          {layer,
           Map.new(layer_params, fn {pname, w} ->
             g = gp[pname]
             {pname, Nx.subtract(w, Nx.multiply(g, Nx.tensor(lr)))}
           end)}
        end)

      %{params | data: new_data}
    end

    {losses, _} =
      Enum.reduce(1..n_steps, {[], params0}, fn _, {losses, p} ->
        {loss, grads} = Nx.Defn.jit_apply(grad_fn, [p, x, y], compiler: Nx.Defn.Evaluator)
        p_new = apply_sgd.(p, grads, 0.01)
        ln = loss |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_number()
        {[ln | losses], p_new}
      end)

    Enum.reverse(losses)
  end

  def main do
    IO.puts("=== Plan A: 100-step SGD training on VulkanoBackend ===")

    {init_fn, _} = Axon.build(model(), mode: :train)

    params0 =
      init_fn.(%{"x" => Nx.template({@batch, @input_dim}, :f32)}, Axon.ModelState.empty())

    key = Nx.Random.key(42)
    {x_bin, key} = Nx.Random.normal(key, shape: {@batch, @input_dim})
    true_w = Nx.tensor([2.0, -1.0, 0.5, 3.0], type: :f32)
    {noise, _} = Nx.Random.normal(key, shape: {@batch, 1})

    y_bin =
      Nx.dot(x_bin, Nx.reshape(true_w, {@input_dim, 1}))
      |> Nx.add(Nx.multiply(noise, Nx.tensor(0.1)))

    IO.puts("\nrunning BinaryBackend (reference)...")
    losses_bin = run_training(params0, x_bin, y_bin, 100)
    IO.puts("  initial loss: #{Float.round(List.first(losses_bin), 4)}")
    IO.puts("  step 50 loss: #{Float.round(Enum.at(losses_bin, 50), 4)}")
    IO.puts("  final loss:   #{Float.round(List.last(losses_bin), 4)}")

    IO.puts("\nrunning VulkanoBackend...")
    params0_vk = transfer_state(params0, Nx.Vulkan.VulkanoBackend)
    x_vk = Nx.backend_transfer(x_bin, Nx.Vulkan.VulkanoBackend)
    y_vk = Nx.backend_transfer(y_bin, Nx.Vulkan.VulkanoBackend)
    losses_vk = run_training(params0_vk, x_vk, y_vk, 100)
    IO.puts("  initial loss: #{Float.round(List.first(losses_vk), 4)}")
    IO.puts("  step 50 loss: #{Float.round(Enum.at(losses_vk, 50), 4)}")
    IO.puts("  final loss:   #{Float.round(List.last(losses_vk), 4)}")

    initial = List.first(losses_bin)
    final_bin = List.last(losses_bin)
    final_vk = List.last(losses_vk)

    convergence_factor_bin = initial / final_bin
    convergence_factor_vk = initial / final_vk
    rel = abs(final_bin - final_vk) / max(abs(final_bin), 1.0e-10)

    pairs = Enum.zip(losses_bin, losses_vk)

    max_step_diff =
      pairs
      |> Enum.map(fn {b, v} -> abs(b - v) / max(abs(b), 1.0e-10) end)
      |> Enum.max()

    IO.puts("\n=== verdict ===")
    IO.puts("convergence factor (bin):       #{Float.round(convergence_factor_bin, 1)}x")
    IO.puts("convergence factor (vk):        #{Float.round(convergence_factor_vk, 1)}x")
    IO.puts("final loss relative difference: #{rel}")
    IO.puts("max per-step relative diff:     #{max_step_diff}")

    ok? =
      convergence_factor_bin > 100 and convergence_factor_vk > 100 and
        rel < 1.0e-4 and max_step_diff < 0.05

    IO.puts("\nresult: #{if ok?, do: "PASS", else: "FAIL"}")
  end
end

AxonTraining.main()
