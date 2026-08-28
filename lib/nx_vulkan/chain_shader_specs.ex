defmodule Nx.Vulkan.ChainShaderSpecs do
  @moduledoc """
  Catalog of family specs for templated chain-shader synthesis.

  Each spec is a `%ShaderTemplate.FamilySpec{}` describing how to fill
  the holes in `Nx.Vulkan.ShaderTemplate`'s GLSL skeleton.

  Phase 1 ships:
    * `:beta`      — Beta(α, β) on logit-unconstrained space (q ∈ ℝ → q* = sigmoid(q))
    * `:gamma`     — Gamma(α, β) on log-unconstrained space (q ∈ ℝ → q* = exp(q))
    * `:lognormal` — Lognormal(μ, σ) on log-unconstrained space.
                     **Mathematically reduces to Normal(μ, σ) on q_uc**
                     because the Jacobian `log|dq/dq_uc| = q_uc` exactly
                     cancels the `-log(q)` = `-q_uc` term in log p(q).
                     Shipped to prove the template handles same-skeleton-
                     different-name cases. Use the hand-written Normal
                     shader in production unless you specifically need
                     :lognormal as a label.

  Existing 6 hand-written shaders (Normal, Exponential, StudentT,
  Cauchy, HalfNormal, Weibull) are NOT replicated here; they live as
  literal SPV files in `nx_vulkan/priv/shaders/`. Goal of Phase 1 is
  to demonstrate a NEW shader synthesized from scratch.
  """

  alias Nx.Vulkan.ShaderTemplate.FamilySpec

  @doc """
  Beta(α, β) on logit-unconstrained space.

  Unconstrained: q_uc = logit(q), q = sigmoid(q_uc) ∈ (0, 1).

  log p(q | α, β) = (α-1) log q + (β-1) log(1-q) - log B(α, β)
  log p(q_uc) = log p(q) + log|dq/dq_uc| = log p(q) + log(q (1-q))
              = α log q + β log(1-q) - log B(α, β)

  d/dq_uc log p(q_uc) = α (1-q) - β q = α - (α+β) q
  """
  def beta do
    %FamilySpec{
      name: "beta",
      push_fields: """
      float alpha;
      float beta_param;
      float logp_const;\
      """,
      grad_block: """
      float q_actual = 1.0 / (1.0 + exp(-qi));
      float grad_q = in_bounds ? pc.alpha - (pc.alpha + pc.beta_param) * q_actual : 0.0;\
      """,
      grad_block_n: """
      float q_actual_n = 1.0 / (1.0 + exp(-qi));
      float grad_qn = in_bounds ? pc.alpha - (pc.alpha + pc.beta_param) * q_actual_n : 0.0;\
      """,
      logp_block: """
      float q_lp = 1.0 / (1.0 + exp(-qi));
      float lp_i = in_bounds ? pc.alpha * log(q_lp) + pc.beta_param * log(1.0 - q_lp) : 0.0;\
      """,
      logp_final: "partial[0] + float(pc.n) * pc.logp_const"
    }
  end

  @doc """
  Push-constant byte layout for Beta. Packed as the C struct:
      uint  n; uint K; float eps; float alpha; float beta_param; float logp_const;
  Total: 24 bytes.

  Caller computes `logp_const = lgamma(α+β) - lgamma(α) - lgamma(β)`
  and passes it explicitly — keeps `lgamma` out of `nx_vulkan`'s
  dependency surface.

  Returns binary suitable for the shim's push-constants pointer.
  """
  def beta_push(n, k, eps, alpha, beta, logp_const)
      when is_number(alpha) and is_number(beta) and is_number(logp_const) do
    <<n::little-32, k::little-32, eps::little-float-32, alpha::little-float-32,
      beta::little-float-32, logp_const::little-float-32>>
  end

  @doc """
  Gamma(α, β) on log-unconstrained space.

  Unconstrained: q_uc = log(q), q = exp(q_uc) ∈ (0, ∞).

  log p(q | α, β) = α·log β + (α-1) log q - β q - lgamma(α)
  log p(q_uc) = log p(q) + log|dq/dq_uc| = log p(q) + q_uc
              = α·log β + α·q_uc - β·exp(q_uc) - lgamma(α)

  d/dq_uc log p(q_uc) = α - β·exp(q_uc)

  Push fields share Beta's layout (alpha, beta_param, logp_const). The
  logp_const captures the q-independent normalizer α·log β - lgamma(α).
  """
  def gamma do
    %FamilySpec{
      name: "gamma",
      push_fields: """
      float alpha;
      float beta_param;
      float logp_const;\
      """,
      grad_block: """
      float grad_q = in_bounds ? pc.alpha - pc.beta_param * exp(qi) : 0.0;\
      """,
      grad_block_n: """
      float grad_qn = in_bounds ? pc.alpha - pc.beta_param * exp(qi) : 0.0;\
      """,
      logp_block: """
      float lp_i = in_bounds ? pc.alpha * qi - pc.beta_param * exp(qi) : 0.0;\
      """,
      logp_final: "partial[0] + float(pc.n) * pc.logp_const"
    }
  end

  @doc """
  Push for Gamma: same 24-byte layout as Beta.

  Caller computes `logp_const = α·log(β) - lgamma(α)` and passes it
  explicitly — keeps `lgamma` out of `nx_vulkan`'s dependency surface.
  """
  def gamma_push(n, k, eps, alpha, beta, logp_const)
      when is_number(alpha) and is_number(beta) and is_number(logp_const) do
    <<n::little-32, k::little-32, eps::little-float-32, alpha::little-float-32,
      beta::little-float-32, logp_const::little-float-32>>
  end

  @doc """
  Lognormal(μ, σ) on log-unconstrained space.

  Unconstrained: q_uc = log(q), q = exp(q_uc) ∈ (0, ∞).

  log p(q | μ, σ) = -log q - 0.5 log(2π σ²) - (log q - μ)² / (2σ²)
  log p(q_uc) = log p(q) + q_uc
              = -0.5 log(2π σ²) - (q_uc - μ)² / (2σ²)

  This is **literally Normal(μ, σ) on q_uc**. The `-log q = -q_uc`
  term and the Jacobian `+q_uc` cancel exactly. The grad and logp
  expressions below are identical to what a Normal-on-q_uc spec
  would produce.

  d/dq_uc log p(q_uc) = -(q_uc - μ) / σ²

  Shipped as a working template-driven shader to demonstrate the
  template handles this family. In production use the hand-written
  Normal shader (binary-identical math, vendored .spv).
  """
  def lognormal do
    %FamilySpec{
      name: "lognormal",
      push_fields: """
      float mu;
      float sigma;
      float logp_const;\
      """,
      grad_block: """
      float inv_sigma2 = 1.0 / (pc.sigma * pc.sigma);
      float grad_q = in_bounds ? -(qi - pc.mu) * inv_sigma2 : 0.0;\
      """,
      grad_block_n: """
      float inv_sigma2_n = 1.0 / (pc.sigma * pc.sigma);
      float grad_qn = in_bounds ? -(qi - pc.mu) * inv_sigma2_n : 0.0;\
      """,
      logp_block: """
      float inv_sigma2_lp = 1.0 / (pc.sigma * pc.sigma);
      float diff_lp = qi - pc.mu;
      float lp_i = in_bounds ? -0.5 * diff_lp * diff_lp * inv_sigma2_lp : 0.0;\
      """,
      logp_final: "partial[0] + float(pc.n) * pc.logp_const"
    }
  end

  @doc """
  Push for Lognormal: 24 bytes, layout {n, K, eps, mu, sigma, logp_const}.
  logp_const = -0.5·log(2π σ²) (per-element constant absorbed at host).
  """
  def lognormal_push(n, k, eps, mu, sigma) when is_number(mu) and is_number(sigma) do
    logp_const = -0.5 * :math.log(2.0 * :math.pi() * sigma * sigma)

    <<n::little-32, k::little-32, eps::little-float-32, mu::little-float-32,
      sigma::little-float-32, logp_const::little-float-32>>
  end
end
