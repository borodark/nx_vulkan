defmodule Nx.Vulkan.NativeV do
  @moduledoc """
  Rustler NIF for the pure-Rust (vulkano) compute backend.

  Sibling of `Nx.Vulkan.Native` (the C++/spirit-backed NIF). Same
  chain-shader dispatch contract, but resource lifetimes are
  managed by Rust ownership (Arc<Buffer>) rather than opaque
  `VkBuf*` pointers — so the stale-handle bug class that
  surfaced as `ArgumentError` in `Nx.Vulkan.Backend.to_binary`
  (Mission II R4) is structurally absent.

  This module is the spike landing zone — for now it only
  exposes `leapfrog_chain_synth/6`, taking bytes in and bytes
  out (no persistent ResourceArc tensor handles). When the
  vulkano backend gets feature-parity with the C++ path, this
  expands to cover `apply_binary`, `reduce`, etc.

  ## Compatibility

  - Loads any SPV the existing pipeline emits (verified
    byte-for-byte equivalence against `Nx.Vulkan.Native.leapfrog_chain_synth`
    on the regime-model R3 fixture; see
    `nx_vulkan/spike/vulkano_synth/README.md`).
  - Builds on Linux + FreeBSD 15.0 with vulkano 0.34.
  """

  use Rustler, otp_app: :nx_vulkan, crate: :nx_vulkan_vulkano

  @doc """
  Dispatch a K-step leapfrog chain against the synthesised SPV.

  All inputs are binaries:

  - `q_init`, `p_init`: d * 4 bytes each (little-endian f32)
  - `extras`: (n_obs + d) * 4 bytes — obs followed by inv_mass
    in the R2.2.3 packed layout
  - `push`: 20–128 bytes, the synth template's push block
    (`K, n_obs, d, _pad, eps`)
  - `k`: leapfrog steps per dispatch (must match push.K)
  - `spv_path`: filesystem path to the cached SPV blob

  Returns `{:ok, {q_chain_bin, p_chain_bin, grad_chain_bin,
  logp_chain_bin}}` on success — same shape as the C++ NIF's
  return after `download_binary_batch4/4`.
  """
  def leapfrog_chain_synth(_q, _p, _extras, _push, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)
end
