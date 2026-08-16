# How much of the "GPU wins on width" result is really "BinaryBackend is slow"?
#
# Same arithmetic as the MODEL_SCALING sweep's likelihood (a d-component
# Gaussian scale mixture over n_obs observations, value_and_grad w.r.t. the
# log-sigmas), timed under BinaryBackend vs EXLA. No exmc, no nx_vulkan:
# this exists only to calibrate the CPU reference's constant factor.

defmodule Ref do
  import Nx.Defn

  defn logp(q, o) do
    sg = Nx.exp(q)
    o2 = Nx.new_axis(o, 0)
    sg2 = Nx.new_axis(sg, 1)
    z = o2 / sg2
    ll = -0.9189385332046727 - Nx.log(sg2) - 0.5 * z * z
    a = ll - Nx.log(Nx.size(q) * 1.0)
    mx = Nx.reduce_max(a, axes: [0])
    Nx.sum(mx + Nx.log(Nx.sum(Nx.exp(a - Nx.new_axis(mx, 0)), axes: [0])))
  end

  defn vag(q, o), do: value_and_grad(q, &logp(&1, o))
end

ms = fn f ->
  t0 = :erlang.monotonic_time(:microsecond)
  r = f.()
  {(:erlang.monotonic_time(:microsecond) - t0) / 1000, r}
end

cells =
  (System.get_env("CELLS") || "8:60,8:600,8:6000,8:60000,64:60,256:60,64:600,256:600")
  |> String.split(",")
  |> Enum.map(fn s -> s |> String.split(":") |> Enum.map(&String.to_integer/1) |> List.to_tuple() end)

reps = String.to_integer(System.get_env("REPS") || "5")

all_arms = [
  {"binary", Nx.BinaryBackend, Nx.Defn.Evaluator, []},
  {"exla_host", {EXLA.Backend, client: :host}, EXLA, [client: :host]},
  {"exla_cuda", {EXLA.Backend, client: :cuda}, EXLA, [client: :cuda]}
]
arms = (case System.get_env("ARMS") do
  nil -> all_arms
  s -> Enum.filter(all_arms, fn {l, _, _, _} -> l in String.split(s, ",") end)
end)

IO.puts("#X nx=#{Application.spec(:nx, :vsn)} exla=#{Application.spec(:exla, :vsn)} reps=#{reps}")

for {d, n} <- cells, {label, backend, compiler, copts} <- arms do
  out =
    try do
      prev = Nx.default_backend(backend)
      :rand.seed(:exsss, {1, 2, 3})
      o = Nx.tensor(for(_ <- 1..n, do: :rand.normal() * 0.012), type: :f64)
      q = Nx.broadcast(Nx.tensor(-4.0, type: :f64), {d})

      f = fn -> Nx.Defn.jit_apply(&Ref.vag/2, [q, o], compiler: compiler) end
      settle = fn {lp, _g} -> Nx.to_number(Nx.backend_transfer(lp, Nx.BinaryBackend)) end

      lp0 = settle.(f.())
      settle.(f.())
      {tp, _} = ms.(fn -> settle.(f.()) end)
      iters = max(1, min(200, round(400 / max(tp, 0.05))))

      per =
        for(_ <- 1..reps, do: elem(ms.(fn -> for(_ <- 1..iters, do: settle.(f.())) end), 0) / iters)
        |> Enum.sort()
        |> Enum.at(div(reps, 2))

      Nx.default_backend(prev)
      "#X d=#{d} n_obs=#{n} arm=#{label} ms=#{:erlang.float_to_binary(per, decimals: 3)} iters=#{iters} logp=#{:erlang.float_to_binary(lp0 * 1.0, decimals: 4)}"
    rescue
      e -> "#X d=#{d} n_obs=#{n} arm=#{label} ERROR=#{Exception.message(e) |> String.slice(0, 100) |> String.replace("\n", " ")}"
    end

  IO.puts(out)
end
