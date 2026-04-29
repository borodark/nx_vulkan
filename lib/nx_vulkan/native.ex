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
end
