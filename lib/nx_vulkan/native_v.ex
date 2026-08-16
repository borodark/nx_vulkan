defmodule Nx.Vulkan.NativeV do
  @moduledoc """
  Rustler NIF for the vulkano compute backend.

  Resource lifetimes are managed by Rust ownership (Arc<Buffer>).
  When the BEAM GCs the Elixir reference, vulkano's `Drop` runs
  and the GPU memory is freed.

  Builds on Linux + FreeBSD 15.0 with vulkano 0.34.
  """

  use Rustler, otp_app: :nx_vulkan, crate: :nx_vulkan_vulkano

  @doc """
  Boundary-cast f64 leapfrog chain dispatch.

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
  Submit any recorded-but-unsubmitted dispatches and wait for them.

  Dispatches are batched into one command buffer and one fence wait rather
  than submitted per op (T1 of `PLAN_AFTER_BACKWARD_PASS.md`); the batch is
  flushed automatically at every host boundary — `buf_download/1`,
  `buf_upload_into/2`, `concat_buffers/1`, `fft/*` — so correctness never
  depends on calling this.

  It matters for **measurement**. Timing a loop without a readback now times
  the recording, not the work, and charges the whole cost to whichever
  iteration happens to trip the batch cap. Call this before stopping the
  clock, or read a result back. `NXV_BATCH_MAX=0` disables batching entirely
  and is the A/B control.

  Returns `:ok`, or `{:error, :dispatch_failed, msg}` if a queued dispatch
  failed to record or the submission failed.
  """
  def flush, do: :erlang.nif_error(:nif_not_loaded)

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
  Broadcasting elementwise binary op (rank <= 4). Bindings a/b/out/params;
  `params` is [rank, out[4], a[4], b[4]] int32; `n` = output element count.
  Keeps bias-add / scaling / relu-via-max on the GPU. op codes: 0=add 1=mul
  2=sub 3=div 5=max 6=min (pow excluded — fp64 has no pow).
  """
  def apply_binary_broadcast(_out, _a, _b, _params, _n, _rank, _op_code, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Elementwise dtype cast (in binding 0 -> out binding 1). The shader defines the
  source/dest types; `n` = element count. Used for f32<->f64 on the GPU.
  """
  def cast(_out, _a, _n, _spv_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Strided slice (type-generic u32-word copy). Bindings in/out/params; params is
  [rank, ews, S[4], O[4], start[4], stride[4]] int32 (ews = element_bytes/4);
  `n` = output element count. Keeps static-start slices on the GPU.
  """
  def apply_slice(_out, _in, _params, _n, _rank, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  put_slice overlay (type-generic u32-word copy). Bindings tensor/slice/out/
  params; params is [rank, ews, T[4], S[4], start[4]] int32 (ews =
  element_bytes/4, T = tensor dims, S = slice dims, start already clamped to
  [0, T - S]); `n` = output element count. Each output element reads the slice
  inside the window and the tensor outside it.
  """
  def apply_put_slice(_out, _in, _slice, _params, _n, _rank, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Pad (type-generic copy). Bindings in/out/params/pad-value; params is
  [rank, ews, S[4], O[4], low[4], interior[4]] int32 (ews = element_bytes/4);
  `pad_value` buffer is one element (ews u32 words); `n` = output element count.
  Elements landing in an edge pad, interior gap, or outside the source get the
  pad value. Keeps pad on the GPU.
  """
  def apply_pad(_out, _in, _params, _padval, _n, _rank, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Gather (leading-prefix axes). Bindings in/out/indices/params; params is
  [K, ews, idx_words, count, stride[4]] int32 (ews = value_bytes/4, idx_words =
  index_bytes/4, count = product of trailing non-indexed dims); `n` = output
  element count, `k` = number of indexed leading axes. Keeps the common gather
  (default all-axes / leading-prefix axes) on the GPU.
  """
  def apply_gather(_out, _in, _idx, _params, _n, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Generic JIT-shader dispatch (thrust 3 fusion compiler). Runs a runtime-
  generated shader: inputs at bindings 0..k-1 (in the order of `in_refs`),
  output at binding k, push {n = element count}. One dispatch replaces a whole
  fused elementwise chain generated by `Nx.Vulkan.Codegen`.
  """
  def dispatch_generated(_out, _in_refs, _n, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Generic JIT fused-reduce dispatch (thrust 3). Runtime-generated shader that
  fuses an elementwise chain into a reduction: inputs at bindings 0..k-1, output
  at k, push {outer, reduce_size, inner}. One invocation per output slot; the
  reduce op (sum f64-acc / max / min) is baked into the generated shader.
  """
  def dispatch_generated_reduce(_out, _in_refs, _outer, _reduce_size, _inner, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Broadcasting select: out = pred ? t : f. Bindings pred/t/f/out/params; params
  is [rank, out[4], pred[4], t[4], f[4]] int32; `n` = output element count. pred
  is u8 (read as u32 words in the shader). Keeps masking / where / relu-grad on
  the GPU.
  """
  def apply_select(_out, _pred, _t, _f, _params, _n, _rank, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Broadcasting comparison -> u8 (packed as u32). Bindings a/b/out/params; op codes
  0=eq 1=ne 2=lt 3=le 4=gt 5=ge. `out` buffer must be padded to a 4-byte multiple
  (ceil(n/4) u32 words). Keeps mask-producing ops on the GPU.
  """
  def apply_compare(_out, _a, _b, _params, _n, _rank, _op_code, _spv_path),
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
  Generic permuted transpose for rank <= 4. Semantics match
  `Nx.transpose(t, axes: perm)`: output axis `d` is input axis `perm[d]`.

  Buffers: a, out (`n * element_bytes` each), params — an int32 buffer laid out
  as `[rank, in[4], out[4], perm[4]]` with shapes left-aligned and padded to 4.
  """
  def transpose_nd(_out, _a, _params, _n, _rank, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Reverse along a set of axes, rank <= 4. Semantics match
  `Nx.reverse(t, axes: axes)`.

  Buffers: a, out (`n * element_bytes` each), params — an int32 buffer laid out
  as `[rank, shape[4], rev[4]]`, where `rev[d]` is 1 when axis `d` is reversed.
  Input and output share a shape.
  """
  def reverse_nd(_out, _a, _params, _n, _rank, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Explicit broadcast, rank <= 4. Semantics match
  `Nx.broadcast(t, shape, axes: axes)` — input axis `i` lands on output axis
  `axes[i]`, and an input axis of size 1 repeats.

  Buffers: a, out, params — an int32 buffer laid out as
  `[out_rank, in_rank, out[4], in[4], axes[4]]`.
  """
  def broadcast_nd(_out, _a, _params, _n, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Windowed max/min, rank <= 4. `op_code`: 0=max, 1=min.

  Buffers: a, out, params — an int32 buffer laid out as
  `[rank, in[4], out[4], win[4], strides[4]]`. One thread per output element;
  padding and window dilation are not handled and must be filtered by the
  caller.
  """
  def window_reduce(_out, _a, _params, _n, _rank, _op_code, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Scatter source values to each window's argmax (pooling backward), rank <= 4,
  NON-OVERLAPPING windows only.

  Buffers: a (original input), src (one value per window), init (1 element),
  out, params — `[rank, in[4], win_grid[4], win[4], strides[4]]`. Ties go to the
  LAST maximum in row-major order, matching `Nx.BinaryBackend`.
  """
  def window_scatter_max(_out, _a, _src, _init, _params, _n, _rank, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  2D matmul. C = A · B where A is M×K row-major, B is K×N row-major,
  C is M×N row-major. All f32. Buffers: a (m*k*4), b (k*n*4), out
  (m*n*4).
  """
  def matmul(_out, _a, _b, _m, _n, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Register-blocked matmul dispatch (32-wide output tiles) for the *_rb32 shaders.
  Benchmark-only — the register-blocked kernels regressed on Kepler; used by
  examples/matmul_rb_race.exs to evaluate them on other GPUs. See F32_PLAN.md.
  """
  def matmul32(_out, _a, _b, _m, _n, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Radix-2 Cooley-Tukey FFT (power-of-two, last-axis, batched) in f64.

  `out` is a complex buffer of `batch*n*16` bytes (interleaved re/im f64).
  `in` is either real f64 (`batch*n*8` bytes, `is_complex = 0`) or complex
  (`batch*n*16`, `is_complex = 1`). `logn = log2(n)`. `inverse = 1` applies the
  1/n normalisation and the conjugate twiddle. Two shaders: bit-reversed load
  then log2(n) butterfly stages.
  """
  def fft(_out, _in, _n, _logn, _batch, _is_complex, _inverse, _bitrev_spv, _stage_spv),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  im2col unfold for conv (spatial rank <= 3, groups == 1), f64. Fills the
  column matrix `col` (M×K = N*O_total × Cin*K_total) from `in`. `params` is a
  21-int buffer of per-dim [D,O,K,stride,pad_lo,input_dil,kernel_dil].
  """
  def conv_im2col(_col, _in, _params, _n, _cin, _o_total, _k_total, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Conv GEMM: out{N,Cout,O_total} = im2col(A){M,K} · kernel{Cout,K}, written in
  canonical output layout. f64.
  """
  def conv_gemm(_out, _col, _kernel, _n, _cout, _o_total, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Physical Vulkan device name + type, e.g.
  `{:ok, "NVIDIA GeForce GT 650M", "DiscreteGpu"}`. Used to label
  benchmark/parity reports per host.
  """
  def device_name, do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  Whether the physical device advertises `shaderFloat64` — `{:ok, boolean}`.
  The `_f64.spv` shaders and any generated f64 kernel require it; on a device
  without it, pipeline creation for those fails at dispatch time, so callers
  must gate on this and take a host fallback instead.
  """
  def device_supports_f64, do: :erlang.nif_error(:nif_not_loaded)

  # Chain shader NIFs — registered in Rust, must have stubs here for NIF loading.
  # f32 variant kept as stub only (unused); f64 is the active path.
  def leapfrog_chain_synth(_q, _p, _extras, _push, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)

  # leapfrog_chain_synth_f64/6 is the documented f64 variant declared above
  # (see the @doc near the top of this module) — not redeclared here.

  def leapfrog_chain_synth_batch(_q, _p, _extras, _push, _k, _spv_path),
    do: :erlang.nif_error(:nif_not_loaded)
end
