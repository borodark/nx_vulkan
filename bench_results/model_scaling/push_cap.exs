# Push-constants ceiling only: extract_components + Push.pack, no GLSL
# emission and no glslang, so it costs milliseconds where SWEEP_MODE=synth
# costs minutes. Confirms where {:unsupported, :push_too_large} starts.
alias Exmc.Builder
alias Exmc.Dist.{HalfCauchy, Normal, Custom}

t = fn v -> Nx.tensor(v, type: :f64, backend: Nx.BinaryBackend) end
obs = Nx.tensor(for(_ <- 1..60, do: 0.01), type: :f64, backend: Nx.BinaryBackend)

build = fn d, dist ->
  names = for k <- 1..d, do: "s#{k}"
  params = if dist == HalfCauchy, do: %{scale: t.(0.02)}, else: %{mu: t.(0.0), sigma: t.(1.0)}

  ir =
    Enum.reduce(names, Builder.data(Builder.new_ir(), obs), fn nm, ir ->
      Builder.rv(ir, nm, dist, params)
    end)

  f = fn _x, p -> Nx.sum(Nx.multiply(p.__obs_data, Map.fetch!(p, :s1))) end
  pm = names |> Map.new(&{String.to_atom(&1), &1}) |> Map.put(:__obs_data, "__obs_data")
  ir = Custom.rv(ir, "lik", Custom.new(f), pm)
  Builder.obs(ir, "lik_obs", "lik", t.(0.0))
end

for {label, dist} <- [{"HalfCauchy (1 float/RV)", HalfCauchy}, {"Normal (2 floats/RV)", Normal}] do
  IO.puts("#P #{label}")

  for d <- 1..16 do
    {:ok, c} = Exmc.NUTS.CustomSynth.extract_components(build.(d, dist))
    spec = Exmc.NUTS.CustomSynth.Push.build(c, K: 32, eps: 0.05, n_obs: 60)

    res =
      case Exmc.NUTS.CustomSynth.Push.pack(spec) do
        {:ok, _b, n} -> "#{n}B ok"
        {:error, r} -> "#{r}"
      end

    IO.puts("#P d=#{d} #{res}")
  end
end
