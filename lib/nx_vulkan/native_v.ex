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

  @doc """
  Plan A* — boundary-cast f64 variant of `leapfrog_chain_synth/6`.

  Same dispatch contract but all binaries are little-endian f64
  (8 bytes per element). Push block is 24+ bytes (eps is f64 at byte
  offset 16). SPV at `spv_path` must be the f64-compiled synth shader
  (uses `GL_ARB_gpu_shader_fp64` for storage; transcendentals via
  emitter-generated `double(log(float(x)))` wrappers).

  Returns `{:ok, {q_chain_bin, p_chain_bin, grad_chain_bin,
  logp_chain_bin}}` as little-endian f64 binaries — each `q/p/grad`
  is `K * d * 8` bytes; `logp` is `K * 8` bytes.

  See `docs/EXMC_VULKAN_DOS_AND_DONTS.md` for why this two-variant
  split exists (GLSL.std.450 has no f64 transcendentals at the Khronos
  spec layer; boundary-cast is the workaround).
  """
  def leapfrog_chain_synth_f64(_q, _p, _extras, _push, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  # -- Buffer lifecycle ---------------------------------------------------

  @doc """
  Allocate a device buffer + upload `data` to it. Returns
  `{:ok, ref}`. The ref is a Rustler resource that owns the
  underlying `Arc<Buffer>` — when the BEAM GCs it, vulkano's Drop
  runs and the GPU memory is freed.
  """
  def buf_upload(_data), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Allocate a zero-initialised device buffer of `n_bytes`. Returns `{:ok, ref}`."
  def buf_alloc(_n_bytes), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Read a device buffer back to a host binary. Returns `{:ok, binary}`."
  def buf_download(_ref), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Buffer size in bytes (returns integer, never crashes on a valid resource)."
  def buf_byte_size(_ref), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Overwrite an existing device buffer with new host data.
  Returns `:ok` or `{:error, :size_mismatch}` when sizes disagree.
  """
  def buf_upload_into(_ref, _data), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Concatenate N device buffers into a single fresh buffer via
  vkCmdCopyBuffer (no shader). Inputs are copied in list order
  into the destination starting at offset 0; total output size =
  Σ inputs[i].n_bytes. Returns `{:ok, output_ref}`.

  Tier 2 step 1 of SHAPE_C_PLAN.md — keeps the result on the
  device so downstream ops don't pay the download+upload round
  trip that the host-fallback path imposed.
  """
  def concat_buffers(_inputs), do: :erlang.nif_error(:nif_not_loaded)

  # -- Compute ops ---------------------------------------------------------

  @doc """
  Elementwise binary op. `op_code` selects which operation the
  shader executes via a specialisation constant:

      0=add  1=mul  2=sub  3=div  4=pow  5=max  6=min

  Buffers must all be the same byte size. Returns `:ok` or
  `{:error, :size_mismatch}` / `{:error, :dispatch_failed, msg}`.
  """
  def apply_binary(_out, _a, _b, _n, _op_code, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Elementwise unary op. `op_code` selects:

      0=exp  1=log  2=sqrt  3=abs  4=neg  5=sigmoid  6=tanh  7=relu
      8=ceil  9=floor  10=sign  11=reciprocal  12=square

  Buffers must be the same byte size.
  """
  def apply_unary(_out, _a, _n, _op_code, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Per-axis reduction. `op_code`: 0=sum, 1=max, 2=min.

  Input shape is interpreted as a virtual (outer, reduce_size, inner)
  tensor; output shape is (outer, inner) — i.e. the reduction
  collapses the middle axis. For full reductions use
  `outer=1, reduce_size=n, inner=1`.

  Buffers: out has `outer * inner` elements; a has
  `outer * reduce_size * inner` elements.
  """
  def reduce_axis(_out, _a, _outer, _reduce_size, _inner, _op_code, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  2D transpose. Input A is M×N row-major; output is N×M row-major.
  Buffers: a (`m*n*4` bytes), out (`m*n*4` bytes).
  """
  def transpose_2d(_out, _a, _m, _n, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  2D matmul. C = A · B where A is M×K row-major, B is K×N row-major,
  C is M×N row-major. All f32. Buffers: a (m*k*4), b (k*n*4), out
  (m*n*4).
  """
  def matmul(_out, _a, _b, _m, _n, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)
end
