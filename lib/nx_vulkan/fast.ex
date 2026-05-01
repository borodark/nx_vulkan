defmodule Nx.Vulkan.Fast do
  @moduledoc """
  Named fused kernels for MCMC hot paths.

  Each function emits an `Nx.Defn.Expr.optional/3` IR node whose name
  matches a callback on `Nx.Vulkan.Backend`. Under Nx.Vulkan the
  evaluator dispatches one fused shader; under any other backend the
  defn fallback runs and produces a mathematically-equivalent result.
  Same pattern as `Emily.Fast`.

  ## Why this exists

  We previously built `Nx.Vulkan.Compiler` to walk the IR and detect
  fusable patterns automatically. That works for narrow cases but
  doesn't scale: each new pattern is more compiler code, false
  negatives are silent, and the matched shapes drift from real exmc
  usage. Naming the kernels at call sites makes the intent explicit
  and the dispatch deterministic. The fallback ensures cross-backend
  correctness (EXLA, BinaryBackend, EMLX).

  ## How to use

  Inside a `defn` or any Nx.Defn.jit-traced function:

      defn leapfrog_step(q, eps, p, grad) do
        q_new = Nx.Vulkan.Fast.leapfrog_position(q, eps, p)
        p_new = Nx.Vulkan.Fast.momentum_step(p, eps, grad)
        {q_new, p_new}
      end

  Under Nx.Vulkan.Backend each Fast call collapses to one
  `Nx.Vulkan.fused_chain_4` dispatch (single shader). Under any other
  backend the defn fallback runs the composed primitives.

  ## Adding kernels

  Each kernel is two functions:

    1. The public entry — emits `Nx.Defn.Expr.optional/3`.
    2. A private `_fallback` — defn-style composed Nx ops.

  Plus one matching callback in `Nx.Vulkan.Backend`. Total ~30 lines
  per kernel; compare to ~100 lines + tests for an IR-detector
  pattern.
  """

  alias Nx.Defn.Expr

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
  @spec leapfrog_position(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def leapfrog_position(q, eps, p) do
    Expr.optional(:fast_leapfrog_position, [q, eps, p, []],
                  &leapfrog_position_fallback/4)
  end

  defp leapfrog_position_fallback(q, eps, p, _opts) do
    Nx.add(q, Nx.multiply(eps, p))
  end

  @doc """
  Half-step momentum update: `p + half_eps * grad`. Used at the start
  and end of every leapfrog iteration in the standard symplectic
  integrator. `half_eps` is `eps / 2` precomputed by the caller.
  """
  @spec leapfrog_momentum_half(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def leapfrog_momentum_half(p, half_eps, grad) do
    Expr.optional(:fast_leapfrog_momentum_half, [p, half_eps, grad, []],
                  &leapfrog_momentum_half_fallback/4)
  end

  defp leapfrog_momentum_half_fallback(p, half_eps, grad, _opts) do
    Nx.add(p, Nx.multiply(half_eps, grad))
  end

  @doc """
  Full-step momentum update: `p + eps * grad`. Same shape as the
  half-step but kept distinct to signal the caller's intent.
  """
  @spec momentum_step(Nx.Tensor.t(), Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def momentum_step(p, eps, grad) do
    Expr.optional(:fast_momentum_step, [p, eps, grad, []],
                  &momentum_step_fallback/4)
  end

  defp momentum_step_fallback(p, eps, grad, _opts) do
    Nx.add(p, Nx.multiply(eps, grad))
  end

  @doc """
  Apply diagonal mass-matrix inverse: `p * inv_mass`. Trivial as a
  fused kernel (one binary op), but named for symmetry — a future
  shader could combine it with adjacent ops in the leapfrog without
  changing call sites.
  """
  @spec inv_mass_apply(Nx.Tensor.t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def inv_mass_apply(p, inv_mass) do
    Expr.optional(:fast_inv_mass_apply, [p, inv_mass, []],
                  &inv_mass_apply_fallback/3)
  end

  defp inv_mass_apply_fallback(p, inv_mass, _opts) do
    Nx.multiply(p, inv_mass)
  end
end
