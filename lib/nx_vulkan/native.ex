defmodule Nx.Vulkan.Native do
  @moduledoc """
  Rustler NIF bindings for the Vulkan compute backend.

  All functions in this module are NIF stubs that fail with
  `:nif_not_loaded` if the native library wasn't compiled. They get
  replaced at module-load time by the real Rust implementations.

  Don't call these directly from application code — use `Nx.Vulkan`
  or the `Nx.Vulkan.Backend` module instead. This module exists
  only to give Rustler a place to bind into.
  """

  use Rustler, otp_app: :nx_vulkan, crate: :nx_vulkan_native

  @doc false
  def init(), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def device_name(), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def has_f64(), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def upload_binary(_data), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def download_binary(_tensor, _n_bytes), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def byte_size(_tensor), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def apply_binary(_a, _b, _op, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def apply_unary(_a, _op, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def reduce_scalar(_a, _op, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def matmul(_a, _b, _m, _n, _k, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def random(_n, _seed, _dist, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def transpose(_a, _m, _n, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def cast(_a, _n, _out_elem_bytes, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def reduce_axis(_a, _outer, _reduce, _inner, _op, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def fused_chain(_a, _b, _ops, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def fused_chain_4(_a, _b, _c, _d, _ops, _buf_idx, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def kinetic_energy(_p, _inv_mass, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def normal_logpdf(_x, _mu, _sigma, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def apply_binary_broadcast(_a, _b, _op, _ndim, _out_shape, _a_strides, _b_strides, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def matmul_v(_a, _b, _m, _n, _k, _tile_m, _tile_n, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def apply_binary_f64(_a, _b, _op, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def apply_unary_f64(_a, _op, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def reduce_axis_f64(_a, _outer, _reduce, _inner, _op, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def apply_binary_broadcast_f64(_a, _b, _op, _ndim, _out_shape, _a_strides, _b_strides, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def logsumexp(_a, _outer, _reduce, _inner, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def pool_clear(), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def pool_stats(), do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_normal(_q, _p, _inv_mass, _eps, _mu, _sigma, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_chain_normal(_q, _p, _inv_mass, _k, _eps, _mu, _sigma, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_chain_normal_lg(_q, _p, _inv_mass, _k, _eps, _mu, _sigma, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_chain_exponential(_q, _p, _inv_mass, _k, _eps, _lambda, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_chain_studentt(_q, _p, _inv_mass, _k, _eps, _mu, _sigma, _nu, _logp_const, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_chain_cauchy(_q, _p, _inv_mass, _k, _eps, _loc, _scale, _log_pi_scale, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_chain_halfnormal(_q, _p, _inv_mass, _k, _eps, _sigma, _log_const, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_chain_normal_f64(_q, _p, _inv_mass, _k, _eps, _mu, _sigma, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc false
  def leapfrog_chain_weibull(_q, _p, _inv_mass, _k, _eps, _weibull_k, _lambda, _logp_const, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)
end
