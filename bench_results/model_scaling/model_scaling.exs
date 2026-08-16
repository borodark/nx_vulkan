# Where (if anywhere) does a GPU arm overtake the CPU as the model gets wider?
#
# EXMC_PEROP_RACE.md established that at d = 8 / n_obs = 60 no GPU path beats
# BinaryBackend on super-io, and closed with the claim that the GPU case for
# eXMC "has to be made on WIDTH". This script sweeps width and tests it.
#
# Two model families, same posterior, different graph shape:
#
#   S ("scalar")  d scalar HalfCauchy RVs, mixture unrolled in Elixir.
#                 This is the eXMC-idiomatic shape (RegimeModel writes its
#                 3-component mixture exactly this way) and the only shape
#                 CustomSynth models correctly. Unpack graph grows O(d).
#
#   V ("vector")  ONE HalfCauchy RV of shape {d}, mixture vectorised over a
#                 {d, n_obs} tensor. Dispatch count is CONSTANT in d and
#                 n_obs. This is the GPU's best case; it is not chain-shader
#                 eligible (CustomSynth scalarises vector RVs).
#
# Both compute  sum_i logsumexp_k [ log(1/d) + Normal(y_i | 0, sigma_k) ]
# and agree to the last digit on the CPU, which is the cross-check that the
# two graphs are the same model.
#
# Usage:
#   SWEEP_MODE=grad MODELS=S,V DIMS=8,32,128 NOBS=60 REPS=5 mix run bench/model_scaling.exs
#   SWEEP_MODE=nuts DIMS=8 NOBS=60,6000 WARMUP=25 SAMPLES=25 mix run bench/model_scaling.exs
#   SWEEP_MODE=synth DIMS=2,4,8,12,13,14 mix run bench/model_scaling.exs
#   TYPE=f32 ...

alias Exmc.Builder
alias Exmc.Dist.{HalfCauchy, Custom}

defmodule Sweep do
  @ln2pi_half 0.9189385332046727

  def t(v, ty), do: Nx.tensor(v, type: ty, backend: Nx.BinaryBackend)

  def obs(n, ty) do
    :rand.seed(:exsss, {1, 2, 3})
    Nx.tensor(for(_ <- 1..n, do: :rand.normal() * 0.012), type: ty, backend: Nx.BinaryBackend)
  end

  # Scalar-vs-{n_obs} broadcasts only, exactly one Nx.sum, no rank-2 tensors:
  # the subset CustomSynth.Glsl can emit.
  def mix_unrolled(o, sigs, d, ty) do
    eps = t(1.0e-8, ty)
    lw = t(-:math.log(d * 1.0), ty)
    c = t(-@ln2pi_half, ty)
    half = t(0.5, ty)

    lls =
      Enum.map(sigs, fn s ->
        sg = Nx.max(s, eps)
        z = Nx.divide(o, sg)
        Nx.add(Nx.subtract(Nx.subtract(c, Nx.log(sg)), Nx.multiply(half, Nx.multiply(z, z))), lw)
      end)

    m = Enum.reduce(tl(lls), hd(lls), &Nx.max/2)
    s = Enum.reduce(lls, nil, fn ll, acc ->
      e = Nx.exp(Nx.subtract(ll, m))
      if acc, do: Nx.add(acc, e), else: e
    end)

    Nx.sum(Nx.add(m, Nx.log(s)))
  end

  # Vectorised: one {d, n_obs} tensor, dispatch count independent of both.
  def mix_vec(o, sig, d, ty) do
    sg = Nx.max(sig, t(1.0e-8, ty))
    o2 = Nx.new_axis(o, 0)
    sg2 = Nx.new_axis(sg, 1)
    z = Nx.divide(o2, sg2)

    ll =
      Nx.subtract(
        Nx.subtract(t(-@ln2pi_half, ty), Nx.log(sg2)),
        Nx.multiply(t(0.5, ty), Nx.multiply(z, z))
      )

    a = Nx.add(ll, t(-:math.log(d * 1.0), ty))
    mx = Nx.reduce_max(a, axes: [0])
    Nx.sum(Nx.add(mx, Nx.log(Nx.sum(Nx.exp(Nx.subtract(a, Nx.new_axis(mx, 0))), axes: [0]))))
  end

  def build("S", d, n, ty) do
    names = for k <- 1..d, do: "s#{k}"
    atoms = Enum.map(names, &String.to_atom/1)

    ir =
      Enum.reduce(names, Builder.data(Builder.new_ir(), obs(n, ty)), fn nm, ir ->
        Builder.rv(ir, nm, HalfCauchy, %{scale: t(0.02, ty)})
      end)

    f = fn _x, p -> mix_unrolled(p.__obs_data, Enum.map(atoms, &Map.fetch!(p, &1)), d, ty) end
    params = names |> Map.new(&{String.to_atom(&1), &1}) |> Map.put(:__obs_data, "__obs_data")
    ir = Custom.rv(ir, "lik", Custom.new(f), params)
    {Builder.obs(ir, "lik_obs", "lik", t(0.0, ty)), Map.new(names, &{&1, t(0.02, ty)})}
  end

  # "W" is "V" with the mixture axis last: {n_obs, d} reduced over axis 1
  # instead of {d, n_obs} reduced over axis 0. Same posterior, same op count.
  # It exists to check that the CPU arm's cost is genuine width and not
  # BinaryBackend being bad at a strided reduction — if V and W disagree on
  # the CPU, the crossover is a backend artefact, not a scaling law.
  def mix_vec_t(o, sig, d, ty) do
    sg = Nx.max(sig, t(1.0e-8, ty))
    o2 = Nx.new_axis(o, 1)
    sg2 = Nx.new_axis(sg, 0)
    z = Nx.divide(o2, sg2)

    ll =
      Nx.subtract(
        Nx.subtract(t(-@ln2pi_half, ty), Nx.log(sg2)),
        Nx.multiply(t(0.5, ty), Nx.multiply(z, z))
      )

    a = Nx.add(ll, t(-:math.log(d * 1.0), ty))
    mx = Nx.reduce_max(a, axes: [1])
    Nx.sum(Nx.add(mx, Nx.log(Nx.sum(Nx.exp(Nx.subtract(a, Nx.new_axis(mx, 1))), axes: [1]))))
  end

  def build(m, d, n, ty) when m in ["V", "W"] do
    ir =
      Builder.new_ir()
      |> Builder.data(obs(n, ty))
      |> Builder.rv("sig", HalfCauchy, %{scale: t(0.02, ty)}, shape: {d})

    f =
      if m == "W",
        do: fn _x, p -> mix_vec_t(p.__obs_data, p.sig, d, ty) end,
        else: fn _x, p -> mix_vec(p.__obs_data, p.sig, d, ty) end
    ir = Custom.rv(ir, "lik", Custom.new(f), %{sig: "sig", __obs_data: "__obs_data"})
    {Builder.obs(ir, "lik_obs", "lik", t(0.0, ty)), %{"sig" => Nx.broadcast(t(0.02, ty), {d})}}
  end

  def ms(f) do
    t0 = :erlang.monotonic_time(:microsecond)
    r = f.()
    {(:erlang.monotonic_time(:microsecond) - t0) / 1000, r}
  end

  # Read back the SCALAR logp and nothing else. On VulkanoBackend a
  # buf_download flushes the pending batch and blocks on the whole command
  # buffer, so the scalar accounts for the gradient too; transferring the
  # {d} gradient as well would add a constant to every arm and, at large d,
  # a host-side reduction that is not the thing being measured.
  # (examples/concurrent_dispatch_bench.exs in nx_vulkan, and the note there.)
  def settle({logp, _grad}) do
    Nx.to_number(Nx.backend_transfer(Nx.reshape(logp, {}), Nx.BinaryBackend))
  end

  def stats(xs) do
    s = Enum.sort(xs)
    n = length(s)
    med = if rem(n, 2) == 1, do: Enum.at(s, div(n, 2)), else: (Enum.at(s, div(n, 2) - 1) + Enum.at(s, div(n, 2))) / 2
    {med, hd(s), List.last(s)}
  end

  def fmt(x) when is_float(x), do: :erlang.float_to_binary(x, decimals: 3)
  def fmt(x), do: to_string(x)
end

env = fn n, d -> System.get_env(n) || d end
env_i = fn n, d -> String.to_integer(System.get_env(n) || Integer.to_string(d)) end
env_l = fn n, d -> (System.get_env(n) || d) |> String.split(",", trim: true) |> Enum.map(&String.trim/1) end

# NB: the app's runtime.exs overwrites MODE, so this knob is SWEEP_MODE.
mode = env.("SWEEP_MODE", "grad")
models = env_l.("MODELS", "S,V")
dims = env_l.("DIMS", "8") |> Enum.map(&String.to_integer/1)
nobs = env_l.("NOBS", "60") |> Enum.map(&String.to_integer/1)
arms = env_l.("ARMS", "cpu,perop")
reps = env_i.("REPS", 5)
ty = if env.("TYPE", "f64") == "f32", do: :f32, else: :f64
budget_ms = env_i.("BUDGET_MS", 400)
max_iters = env_i.("MAX_ITERS", 200)
warmup = env_i.("WARMUP", 25)
samples = env_i.("SAMPLES", 25)
tag = env.("TAG", "")

if ty == :f32, do: Application.put_env(:exmc, :force_precision, :f32)
Application.put_env(:exmc, :allow_vulkan_perop_sampling, true)

IO.puts("#H mode=#{mode} models=#{Enum.join(models, "/")} dims=#{Enum.join(dims, ",")} nobs=#{Enum.join(nobs, ",")} arms=#{Enum.join(arms, "/")} type=#{ty} reps=#{reps} tag=#{tag}")

case Nx.Vulkan.NativeV.device_name() do
  {:ok, name, kind} -> IO.puts("#H device=#{name} (#{kind})")
  o -> IO.puts("#H device=#{inspect(o)}")
end

IO.puts("#H uptime=#{String.trim(:os.cmd(~c"uptime") |> to_string())}")

# ---------------------------------------------------------------- synth mode
if mode == "synth" do
  for m <- models, d <- dims, n <- nobs do
    {ir, _} = Sweep.build(m, d, n, ty)

    push =
      case Exmc.NUTS.CustomSynth.extract_components(ir) do
        {:ok, c} ->
          spec = Exmc.NUTS.CustomSynth.Push.build(c, K: 32, eps: 0.05, n_obs: n)
          case Exmc.NUTS.CustomSynth.Push.pack(spec) do
            {:ok, _b, nb} -> "#{nb}B"
            {:error, r} -> to_string(r)
          end
        o -> inspect(o) |> String.slice(0, 40)
      end

    {tsyn, res} =
      Sweep.ms(fn ->
        try do
          case Exmc.NUTS.CustomSynth.synthesise(ir) do
            {:ok, _} -> "ok"
            other -> inspect(other) |> String.slice(0, 40)
          end
        rescue
          e -> "raised:" <> (Exception.message(e) |> String.slice(0, 60))
        catch
          k, r -> "#{k}:#{inspect(r) |> String.slice(0, 40)}"
        end
      end)

    IO.puts("#S model=#{m} d=#{d} n_obs=#{n} push=#{push} synth=#{res} synth_ms=#{Sweep.fmt(tsyn)}")
  end
end

# ---------------------------------------------------------------- grad mode
grad_arm = fn m, d, n, arm ->
  {ir, _init} = Sweep.build(m, d, n, ty)
  # "exla" is outside the three arms EXMC_PEROP_RACE.md raced. It is here
  # because "is the GPU the right place for the effort" is not answerable
  # against BinaryBackend alone on a box that has EXLA (super-io does; the
  # FreeBSD Kepler fleet does not).
  {cenv, backend} =
    case arm do
      "cpu" -> {:none, Nx.BinaryBackend}
      "exla" -> {:exla, EXLA.Backend}
      _ -> {:vulkan, Nx.Vulkan.VulkanoBackend}
    end

  Application.put_env(:exmc, :compiler, cenv)
  Process.delete(:exmc_chain_meta)
  prev = Nx.default_backend(backend)

  out =
    try do
      # Compiler.value_and_grad/1 uses the SAME build_vag_fn as
      # compile_for_sampling/2 but skips ChainShaderCodegen.detect_meta/1.
      # The chain shader plays no part in a bare value_and_grad, and its
      # synthesis cost is minutes at d >= 12 — see the #S table.
      #
      # "fused" swaps the defn compiler for Nx.Vulkan.Compiler (whole-graph
      # fusion). Exmc.JIT.jit/2 hardcodes Nx.Defn.Evaluator on the vulkan
      # path, so that arm cannot go through Compiler.value_and_grad/1 and is
      # built by hand — same closure, different compiler.
      {tc, {v, pm}} =
        Sweep.ms(fn ->
          if arm == "fused" do
            {logp_fn, pm} = Exmc.Compiler.compile(ir)
            {Nx.Defn.jit(fn flat -> Nx.Defn.value_and_grad(flat, logp_fn) end,
               compiler: Nx.Vulkan.Compiler), pm}
          else
            Exmc.Compiler.value_and_grad(ir)
          end
        end)

      q0 = Nx.broadcast(Nx.tensor(-4.0, type: ty, backend: backend), {pm.size})

      lp0 = Sweep.settle(v.(q0))
      if not is_number(lp0) or lp0 != lp0, do: raise("logp not finite: #{inspect(lp0)}")
      Sweep.settle(v.(q0))

      # pilot: pick an iteration count that fills the budget
      {tp, _} = Sweep.ms(fn -> Sweep.settle(v.(q0)) end)
      iters = max(1, min(max_iters, round(budget_ms / max(tp, 0.05))))

      # A cell where one gradient costs seconds cannot afford five replicates
      # and does not need them — the arms are orders of magnitude apart there.
      reps = if tp > 2000.0, do: min(reps, 3), else: reps

      pers =
        for _ <- 1..reps do
          {t, _} = Sweep.ms(fn -> for _ <- 1..iters, do: Sweep.settle(v.(q0)) end)
          t / iters
        end

      {med, lo, hi} = Sweep.stats(pers)

      fb =
        if backend == Nx.Vulkan.VulkanoBackend do
          {r, c} = Nx.Vulkan.Fallback.count(fn -> v.(q0) end)
          Sweep.settle(r)
          Enum.sum(Map.values(c))
        else
          0
        end

      # count/1 is a LOWER bound: the first fallback strands the tensor on
      # BinaryBackend and everything after it is invisible. strict(:raise)
      # fires on the FIRST refused fallback, before the tensor leaves the
      # device, so it is the residency assertion that actually holds.
      strict =
        if backend == Nx.Vulkan.VulkanoBackend do
          try do
            Sweep.settle(Nx.Vulkan.Fallback.strict(:raise, fn -> v.(q0) end))
            "clean"
          rescue
            e -> "RAISED:" <> (Exception.message(e) |> String.slice(0, 70) |> String.replace("\n", " "))
          end
        else
          "-"
        end

      fbmap =
        if backend == Nx.Vulkan.VulkanoBackend and fb > 0 do
          {r, c} = Nx.Vulkan.Fallback.count(fn -> v.(q0) end)
          Sweep.settle(r)
          inspect(c) |> String.replace(" ", "")
        else
          "-"
        end

      "#G model=#{m} d=#{d} n_obs=#{n} type=#{ty} arm=#{arm} ms=#{Sweep.fmt(med)} lo=#{Sweep.fmt(lo)} hi=#{Sweep.fmt(hi)} iters=#{iters} reps=#{reps} fallbacks=#{fb} logp=#{:erlang.float_to_binary(lp0 * 1.0, decimals: 6)} compile_ms=#{Sweep.fmt(tc)} strict=#{strict} fbmap=#{fbmap}"
    rescue
      e -> "#G model=#{m} d=#{d} n_obs=#{n} type=#{ty} arm=#{arm} ERROR=#{Exception.message(e) |> String.slice(0, 140) |> String.replace("\n", " ")}"
    catch
      k, r -> "#G model=#{m} d=#{d} n_obs=#{n} type=#{ty} arm=#{arm} ERROR=#{k}:#{inspect(r) |> String.slice(0, 120)}"
    end

  Nx.default_backend(prev)
  Process.delete(:exmc_chain_meta)
  IO.puts(out)
end

if mode == "grad" do
  for m <- models, n <- nobs, d <- dims, arm <- arms, do: grad_arm.(m, d, n, arm)
end

# ---------------------------------------------------------------- nuts mode
nuts_arm = fn m, d, n, arm ->
  {ir, init} = Sweep.build(m, d, n, ty)

  {cenv, backend, strip} =
    case arm do
      "cpu" -> {:none, Nx.BinaryBackend, false}
      "perop" -> {:vulkan, Nx.Vulkan.VulkanoBackend, true}
      "chain" -> {:vulkan, Nx.Vulkan.VulkanoBackend, false}
    end

  Application.put_env(:exmc, :compiler, cenv)
  Process.delete(:exmc_chain_meta)
  prev = Nx.default_backend(backend)

  out =
    try do
      compiled = Exmc.Compiler.compile_for_sampling(ir)
      {vg, sf, pm, nc, mu, cm} = compiled
      compiled = if strip, do: {vg, sf, pm, nc, mu, nil}, else: compiled

      {t, {_trace, st}} =
        Sweep.ms(fn ->
          Exmc.NUTS.Sampler.sample_compiled(compiled, init,
            num_warmup: warmup, num_samples: samples, seed: 42, ncp: false)
        end)

      "#N model=#{m} d=#{d} n_obs=#{n} type=#{ty} arm=#{arm} wall_ms=#{Sweep.fmt(t)} ms_per_iter=#{Sweep.fmt(t / (warmup + samples))} eps=#{:erlang.float_to_binary(st.step_size * 1.0, decimals: 5)} div=#{st.divergences} chain=#{if strip, do: "stripped", else: if(cm, do: "yes", else: "no")}"
    rescue
      e -> "#N model=#{m} d=#{d} n_obs=#{n} type=#{ty} arm=#{arm} ERROR=#{Exception.message(e) |> String.slice(0, 140) |> String.replace("\n", " ")}"
    catch
      k, r -> "#N model=#{m} d=#{d} n_obs=#{n} type=#{ty} arm=#{arm} ERROR=#{k}:#{inspect(r) |> String.slice(0, 120)}"
    end

  Nx.default_backend(prev)
  Process.delete(:exmc_chain_meta)
  IO.puts(out)
end

if mode == "nuts" do
  for m <- models, n <- nobs, d <- dims, arm <- arms, do: nuts_arm.(m, d, n, arm)
end

IO.puts("#H done uptime=#{String.trim(:os.cmd(~c"uptime") |> to_string())}")
