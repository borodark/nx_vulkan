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
end
