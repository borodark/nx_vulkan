defmodule Nx.Vulkan.Fast do
  @moduledoc """
  Named fused kernels for MCMC hot paths.

  Each function is a composition of standard Nx ops that produces
  a mathematically-equivalent result to the fused shader that
  `Nx.Vulkan.VulkanoBackend` would dispatch. Cross-backend correctness
  is guaranteed (EXLA, BinaryBackend, VulkanoBackend).

  ## Note on Nx 0.12 migration

  Prior to Nx 0.12, each function emitted Nx.Defn.Expr.optional/3
  IR nodes (an internal Nx API, so not linked here) for backend-specific
  fused dispatch. That API was removed in Nx 0.12. The functions now call
  the fallback Nx ops directly.
  The VulkanoBackend's per-op dispatch is fast enough that the
  fused-kernel optimization is not critical — the chain shader path
  (where performance matters) bypasses this module entirely.

  ## How to use

  Inside a `defn` or any Nx.Defn.jit-traced function:

      defn leapfrog_step(q, eps, p, grad) do
        q_new = Nx.Vulkan.Fast.leapfrog_position(q, eps, p)
        p_new = Nx.Vulkan.Fast.momentum_step(p, eps, grad)
        {q_new, p_new}
      end
  """

  @doc """
  Position update: `q + eps * p`. The dominant elementwise body in
  every NUTS leapfrog.

  ## Examples

      iex> q = Nx.tensor([1.0, 2.0])
      iex> eps = Nx.tensor([0.5, 0.5])
      iex> p = Nx.tensor([2.0, 4.0])
      iex> Nx.Vulkan.Fast.leapfrog_position(q, eps, p) |> Nx.to_flat_list()
      [2.0, 4.0]
  """
  @spec leapfrog_position(Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  def leapfrog_position(q, eps, p) do
    Nx.add(q, Nx.multiply(eps, p))
  end

  @doc """
  Half-step momentum update: `p + half_eps * grad`. Used at the start
  and end of every leapfrog iteration in the standard symplectic
  integrator. `half_eps` is `eps / 2` precomputed by the caller.
  """
  @spec leapfrog_momentum_half(Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  def leapfrog_momentum_half(p, half_eps, grad) do
    Nx.add(p, Nx.multiply(half_eps, grad))
  end

  @doc """
  Full-step momentum update: `p + eps * grad`. Same shape as the
  half-step but kept distinct to signal the caller's intent.
  """
  @spec momentum_step(Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  def momentum_step(p, eps, grad) do
    Nx.add(p, Nx.multiply(eps, grad))
  end

  @doc """
  Apply diagonal mass-matrix inverse: `p * inv_mass`. Trivial as a
  fused kernel (one binary op), but named for symmetry — a future
  shader could combine it with adjacent ops in the leapfrog without
  changing call sites.
  """
  @spec inv_mass_apply(Nx.t(), Nx.t()) :: Nx.t()
  def inv_mass_apply(p, inv_mass) do
    Nx.multiply(p, inv_mass)
  end

  @doc """
  Kinetic energy: `0.5 * sum(p² * inv_mass)`. Reduces to a scalar.
  Used in NUTS for the joint log-probability:
  `joint_logp = log_prob - kinetic_energy(p, inv_mass)`.
  """
  @spec kinetic_energy(Nx.t(), Nx.t()) :: Nx.t()
  def kinetic_energy(p, inv_mass) do
    p
    |> Nx.pow(2)
    |> Nx.multiply(inv_mass)
    |> Nx.sum()
    |> Nx.multiply(0.5)
  end

  @doc """
  Normal log-density: `-0.5*((x-mu)/sigma)² - log(sigma) - 0.5*log(2π)`.
  Output shape matches `x`. The MCMC distribution-density hot path.
  """
  @log_sqrt_2pi 0.91893853320467274178

  @spec normal_logpdf(Nx.t(), Nx.t(), Nx.t()) :: Nx.t()
  def normal_logpdf(x, mu, sigma) do
    z = Nx.divide(Nx.subtract(x, mu), sigma)
    z2 = Nx.multiply(z, z)
    log_sigma = Nx.log(sigma)

    # `@log_sqrt_2pi` MUST be materialised at the computation's type.
    #
    # As a bare Elixir float it went through `Nx.tensor/1`, which defaults to
    # {:f, 32}: 0.91893853320467274 becomes 0.9189385175704956, losing eight
    # significant digits. Subtracting that from an f64 tensor produces an f64
    # RESULT TYPE carrying an f32-precision VALUE — so this function answered
    # -0.9189385175704956 for normal_logpdf(0, 0, 1) where the exact value is
    # -0.9189385332046727, an absolute error of 1.6e-8 that no type signature
    # showed.
    #
    # It is a log-density feeding a Metropolis acceptance ratio across thousands
    # of leapfrog steps, so silent f32 in an f64 chain is the wrong kind of
    # cheap. `Nx.type(z2)` keeps f32 callers exactly as they were.
    #
    # -0.5 needs no such care: it is exact in binary at any width.
    Nx.subtract(
      Nx.subtract(Nx.multiply(z2, -0.5), log_sigma),
      Nx.tensor(@log_sqrt_2pi, type: Nx.type(z2))
    )
  end
end
