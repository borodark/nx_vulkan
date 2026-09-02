defmodule Nx.Vulkan.ChainShaderSpecsF64 do
  @moduledoc """
  f64 chain families, ported from the six hand-written
  `glsl/leapfrog_chain_*_f64.comp` shaders onto `Nx.Vulkan.ShaderTemplate`.

  ## Why these were ported

  The hand-written shaders were **undriveable by this repo's own NIFs** and had
  been since they were written. Each declared a 32-byte push block with family
  parameters inline — `{uint n; uint K; double eps; double mu; double sigma}` —
  while `leapfrog_chain_synth_f64` pushes a fixed `sizeof(PushBlockF64)` = 24
  bytes laid out `{k_steps, n_obs, d, _pad, eps}`. So `mu` and `sigma` were
  never forwarded, and the header disagreed at every field besides: the
  shader's `n` (the dimension) sat where the NIF writes `k_steps`.

  Nothing called those NIFs from this repo, so nothing ever contradicted it.
  The `.spv` shipped in the hex package for months in that state.

  ## What changed

  Family parameters are **baked into the generated source as literals**, which
  is the design the only working caller (eXMC) arrived at independently — its
  synthesised shaders declare exactly the 24-byte header and carry priors as
  `OpConstant`s. Baking is free here: priors are fixed for a run, only `q`/`p`
  vary, and `Nx.Vulkan.Synthesis.compile/1` is content-addressed on the GLSL.

  ## A precision improvement that fell out

  GLSL.std.450 has no f64 transcendentals, so the hand-written shaders computed
  parameter-derived constants by boundary-casting through f32 —
  `double(log(float(pc.sigma)))` in `normal_f64`. Baking moves that to the host,
  where it is a full f64 `:math.log/1`. **The runtime boundary casts remain**
  wherever the argument depends on `qi` — `log(denom_n)` in Student-t,
  `exp(qi)` in Exponential, Half-Normal and Weibull — because those cannot be
  precomputed. Their precision cost is unchanged from the originals.

  ## Constants stay the caller's job

  Where the original `*_push` builder took a precomputed normalisation
  (`log_pi_scale`, `log_const`, `logp_const`), so does the spec here. That
  keeps `lgamma` and friends out of this library's dependency surface, and it
  means this port re-derives none of the original math — only Normal's
  `log(sigma)`, which is a log of a positive number.
  """

  alias Nx.Vulkan.ShaderTemplate.FamilySpec

  @log_2pi_half 0.9189385332046727

  @doc """
  Normal(μ, σ) on the unconstrained line.

  `logp_const` is derived here rather than taken: it is
  `-(log σ + ½ log 2π)`, and computing it on the host in f64 is strictly better
  than the shader's old `double(log(float(pc.sigma)))`.
  """
  def normal(mu, sigma) when is_number(mu) and is_number(sigma) and sigma > 0 do
    %FamilySpec{
      name: "normal_f64",
      dtype: :f64,
      params: %{
        "mu" => mu,
        "inv_var" => 1.0 / (sigma * sigma),
        "inv_sigma" => 1.0 / sigma,
        "logp_const" => -(:math.log(sigma) + @log_2pi_half)
      },
      grad_block: "double grad_q = in_bounds ? -(qi - pc.mu) * pc.inv_var : 0.0LF;",
      grad_block_n: "double grad_qn = in_bounds ? -(qi - pc.mu) * pc.inv_var : 0.0LF;",
      logp_block: """
      double zi = (qi - pc.mu) * pc.inv_sigma;
      double lp_i = in_bounds ? -0.5LF * zi * zi : 0.0LF;\
      """,
      logp_final: "partial[0] + double(pc.d) * pc.logp_const"
    }
  end

  @doc """
  Cauchy(loc, scale). `log_pi_scale` is `-log(π · scale)`, precomputed by the
  caller as the original `cauchy_push` required.
  """
  def cauchy(loc, scale, log_pi_scale)
      when is_number(loc) and is_number(scale) and is_number(log_pi_scale) do
    %FamilySpec{
      name: "cauchy_f64",
      dtype: :f64,
      params: %{
        "loc" => loc,
        "inv_scale" => 1.0 / scale,
        "scale2" => scale * scale,
        "log_pi_scale" => log_pi_scale
      },
      grad_block: """
      double diff = qi - pc.loc;
      double grad_q = in_bounds ? -2.0LF * diff / (pc.scale2 + diff * diff) : 0.0LF;\
      """,
      grad_block_n: """
      double diff_n = qi - pc.loc;
      double grad_qn = in_bounds ? -2.0LF * diff_n / (pc.scale2 + diff_n * diff_n) : 0.0LF;\
      """,
      logp_block: """
      double z_lp = (qi - pc.loc) * pc.inv_scale;
      double lp_i = in_bounds ? -double(log(float(1.0LF + z_lp * z_lp))) : 0.0LF;\
      """,
      logp_final: "partial[0] + double(pc.d) * pc.log_pi_scale"
    }
  end

  @doc "Exponential(λ) on log-unconstrained space."
  def exponential(lambda) when is_number(lambda) do
    %FamilySpec{
      name: "exponential_f64",
      dtype: :f64,
      params: %{"lambda" => lambda},
      grad_block:
        "double grad_q = in_bounds ? 1.0LF - pc.lambda * double(exp(float(qi))) : 0.0LF;",
      grad_block_n:
        "double grad_qn = in_bounds ? 1.0LF - pc.lambda * double(exp(float(qi))) : 0.0LF;",
      logp_block: "double lp_i = in_bounds ? qi - pc.lambda * double(exp(float(qi))) : 0.0LF;",
      logp_final: "partial[0]"
    }
  end

  @doc """
  Half-Normal(σ) on log-unconstrained space. `log_const` is the per-element
  normalisation the original `halfnormal_push` took.
  """
  def halfnormal(sigma, log_const) when is_number(sigma) and is_number(log_const) do
    %FamilySpec{
      name: "halfnormal_f64",
      dtype: :f64,
      params: %{"inv_sigma2" => 1.0 / (sigma * sigma), "log_const" => log_const},
      grad_block: """
      double exp_2q = double(exp(float(2.0LF * qi)));
      double grad_q = in_bounds ? 1.0LF - exp_2q * pc.inv_sigma2 : 0.0LF;\
      """,
      grad_block_n: """
      double exp_2qn = double(exp(float(2.0LF * qi)));
      double grad_qn = in_bounds ? 1.0LF - exp_2qn * pc.inv_sigma2 : 0.0LF;\
      """,
      logp_block: """
      double exp_2q_lp = double(exp(float(2.0LF * qi)));
      double lp_i = in_bounds ? qi - 0.5LF * exp_2q_lp * pc.inv_sigma2 : 0.0LF;\
      """,
      logp_final: "partial[0] + double(pc.d) * pc.log_const"
    }
  end

  @doc """
  Student-t(μ, σ, ν). `logp_const` is
  `lgamma((ν+1)/2) - lgamma(ν/2) - ½log(πν) - log σ`, precomputed by the caller
  exactly as the original `studentt_push` required — `lgamma` stays out of this
  library.
  """
  def studentt(mu, sigma, nu, logp_const)
      when is_number(mu) and is_number(sigma) and is_number(nu) and is_number(logp_const) do
    inv_sigma2 = 1.0 / (sigma * sigma)
    inv_nu = 1.0 / nu

    %FamilySpec{
      name: "studentt_f64",
      dtype: :f64,
      params: %{
        "mu" => mu,
        "inv_sigma2" => inv_sigma2,
        "inv_nu" => inv_nu,
        "grad_coeff" => -(nu + 1.0) * inv_nu * inv_sigma2,
        "logp_exp" => -0.5 * (nu + 1.0),
        "logp_const" => logp_const
      },
      grad_block: """
      double diff = qi - pc.mu;
      double denom = 1.0LF + diff * diff * pc.inv_sigma2 * pc.inv_nu;
      double grad_q = in_bounds ? pc.grad_coeff * diff / denom : 0.0LF;\
      """,
      grad_block_n: """
      double diff_n = qi - pc.mu;
      double denom_n = 1.0LF + diff_n * diff_n * pc.inv_sigma2 * pc.inv_nu;
      double grad_qn = in_bounds ? pc.grad_coeff * diff_n / denom_n : 0.0LF;\
      """,
      logp_block: """
      double diff_lp = qi - pc.mu;
      double denom_lp = 1.0LF + diff_lp * diff_lp * pc.inv_sigma2 * pc.inv_nu;
      double lp_i = in_bounds ? pc.logp_exp * double(log(float(denom_lp))) : 0.0LF;\
      """,
      logp_final: "partial[0] + double(pc.d) * pc.logp_const"
    }
  end

  @doc """
  Weibull(k, λ) on log-unconstrained space. `logp_const` is
  `n · (log k - k · log λ)` — note it is NOT multiplied by `d` in `logp_final`,
  matching the original shader, which folded the count into the constant.

  The shape parameter is named `shape` in the spec, not `k`: the template's
  step loop uses `k`, and a parameter of the same name would be a live trap for
  anyone editing the blocks. Baking removes the collision entirely, but the
  naming keeps it obvious.
  """
  def weibull(shape, lambda, logp_const)
      when is_number(shape) and is_number(lambda) and is_number(logp_const) do
    %FamilySpec{
      name: "weibull_f64",
      dtype: :f64,
      params: %{"shape" => shape, "inv_lambda" => 1.0 / lambda, "logp_const" => logp_const},
      helpers: """
      double weibull_grad(double q_uc) {
          double ratio = double(exp(float(q_uc))) * pc.inv_lambda;
          return pc.shape * (1.0LF - double(pow(float(ratio), float(pc.shape))));
      }

      double weibull_logp_contrib(double q_uc) {
          double ratio = double(exp(float(q_uc))) * pc.inv_lambda;
          return pc.shape * q_uc - double(pow(float(ratio), float(pc.shape)));
      }\
      """,
      grad_block: "double grad_q = in_bounds ? weibull_grad(qi) : 0.0LF;",
      grad_block_n: "double grad_qn = in_bounds ? weibull_grad(qi) : 0.0LF;",
      logp_block: "double lp_i = in_bounds ? weibull_logp_contrib(qi) : 0.0LF;",
      logp_final: "partial[0] + pc.logp_const"
    }
  end

  @doc "Every f64 family, with representative parameters. For tests and sweeps."
  def all do
    [
      normal(0.0, 1.0),
      cauchy(0.0, 2.0, -:math.log(:math.pi() * 2.0)),
      exponential(1.5),
      halfnormal(1.0, -0.5 * :math.log(:math.pi() / 2.0)),
      studentt(0.0, 1.0, 5.0, -1.0),
      weibull(2.0, 1.0, 0.0)
    ]
  end
  @doc """
  Turn any f64 family spec into its BATCHED form: one workgroup per chain,
  indices offset by instance. Same skeleton, so the two cannot drift.
  """
  def batched(%FamilySpec{} = spec), do: %{spec | name: spec.name <> "_batch", batched: true}

  @doc """
  The batched push header: `{k_steps, n_obs, d, n_instances, eps}`, 24 bytes.

  `n_instances` sits where the single-instance block keeps `_pad`, which is why
  the two layouts agree on their first twelve bytes.
  """
  def batch_push(k, n_obs, d, n_instances, eps)
      when is_integer(k) and is_integer(n_obs) and is_integer(d) and is_integer(n_instances) do
    <<k::little-32, n_obs::little-32, d::little-32, n_instances::little-32,
      eps::little-float-64>>
  end

end
