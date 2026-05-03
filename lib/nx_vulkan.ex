defmodule Nx.Vulkan do
  @moduledoc """
  Nx tensor backend on Vulkan compute.

  Wraps Spirit's Vulkan compute kernels (elementwise, reductions,
  matmul, random) as an `Nx.Backend`. Works on FreeBSD with NVIDIA
  hardware where CUDA does not. Same backend code runs on Linux,
  macOS (via MoltenVK), and any Vulkan-capable GPU.

  ## Status

  v0.0.1 — bootstrap. The plan in `PLAN.md` lays out the 10-milestone
  path to v0.1. This release just initializes Vulkan and reports
  which physical device was selected; tensor types and operators
  land in subsequent commits.

  ## Usage (target)

      iex> Nx.Vulkan.init()
      :ok

      iex> Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
      ...

      iex> Nx.Defn.default_options(default_backend: Nx.Vulkan.Backend)

  ## Files

    * `lib/nx_vulkan.ex`           - this module (top-level API)
    * `lib/nx_vulkan/native.ex`    - Rustler NIF binding (skeleton)
    * `lib/nx_vulkan/backend.ex`   - `Nx.Backend` impl (TBD)
    * `native/nx_vulkan_native/`   - Rust NIF crate
    * `c_src/`                     - extern "C" shim into spirit's
                                     C++ Vulkan backend
  """

  @doc """
  Initialize the Vulkan compute context. Call once at startup.
  Returns `:ok` on success, `{:error, reason}` if no Vulkan-capable
  device is found.
  """
  defdelegate init(), to: Nx.Vulkan.Native

  @doc """
  Returns the name of the selected physical device, or `nil` if
  `init/0` has not been called.
  """
  defdelegate device_name(), to: Nx.Vulkan.Native

  @doc "Returns true if the selected device supports f64 (double precision)."
  defdelegate has_f64?(), to: Nx.Vulkan.Native, as: :has_f64

  # ------------------------------------------------------------------
  # v0.0.2 — tensor lifetime + round-trip
  # ------------------------------------------------------------------

  @doc """
  Upload a list of f32 values to a freshly-allocated GPU buffer.
  Returns `{:ok, tensor_ref}` where `tensor_ref` is an opaque
  `ResourceArc` whose underlying VkBuf is freed when GC'd.

      iex> Nx.Vulkan.init()
      :ok
      iex> {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      iex> {:ok, [1.0, 2.0, 3.0, 4.0]} = Nx.Vulkan.download_f32(t, 4)
  """
  def upload_f32(list) when is_list(list) do
    bin =
      list
      |> Enum.flat_map(fn x -> [<<x::float-32-native>>] end)
      |> IO.iodata_to_binary()

    Nx.Vulkan.Native.upload_binary(bin)
  end

  @doc "Upload a raw binary (already packed f32 little-endian) to GPU memory."
  def upload_binary(bin) when is_binary(bin) do
    Nx.Vulkan.Native.upload_binary(bin)
  end

  @doc """
  Download a GPU buffer back into a list of f32 values. `n_elements`
  must match what was uploaded.

  Non-finite values (NaN, +Inf, -Inf) are returned as the atoms
  `:nan`, `:infinity`, `:neg_infinity`. Erlang's float pattern
  `<<x::float-32-native>>` rejects these bit patterns; we decode
  the raw 32-bit pattern and check the IEEE 754 exponent/mantissa
  to recover them.
  """
  def download_f32(tensor, n_elements) when is_integer(n_elements) and n_elements >= 0 do
    case Nx.Vulkan.Native.download_binary(tensor, n_elements * 4) do
      {:ok, bin} -> {:ok, decode_f32_list(bin)}
      err -> err
    end
  end

  defp decode_f32_list(<<>>), do: []

  defp decode_f32_list(<<bits::32-native, rest::binary>>) do
    [decode_f32(bits) | decode_f32_list(rest)]
  end

  # IEEE 754 binary32: 1 sign bit, 8 exponent, 23 mantissa.
  # We only need the exponent==255 branch for special values; finite
  # values are pulled directly via the float pattern.
  defp decode_f32(bits) do
    <<f::float-32-native>> = <<bits::32-native>>
    f
  rescue
    MatchError -> decode_f32_special(bits)
  end

  # Big-endian view for the spec-defined breakdown — the field
  # extraction is byte-order agnostic when we operate on the integer.
  defp decode_f32_special(bits) do
    sign = Bitwise.band(Bitwise.bsr(bits, 31), 1)
    exponent = Bitwise.band(Bitwise.bsr(bits, 23), 0xFF)
    mantissa = Bitwise.band(bits, 0x7FFFFF)

    cond do
      exponent == 255 and mantissa == 0 and sign == 0 -> :infinity
      exponent == 255 and mantissa == 0 and sign == 1 -> :neg_infinity
      exponent == 255 -> :nan
      true -> raise MatchError, term: bits
    end
  end

  @doc "Download as a raw binary (caller does the unpack)."
  def download_binary(tensor, n_bytes) do
    Nx.Vulkan.Native.download_binary(tensor, n_bytes)
  end

  import Kernel, except: [byte_size: 1, max: 2, min: 2]

  @doc "Byte size of an uploaded tensor (in bytes)."
  defdelegate byte_size(tensor), to: Nx.Vulkan.Native

  # ------------------------------------------------------------------
  # v0.0.3 — elementwise binary ops
  # ------------------------------------------------------------------

  @ops_binary %{
    add: 0,
    multiply: 1,
    subtract: 2,
    divide: 3,
    pow: 4,
    max: 5,
    min: 6,
    equal: 7,
    less: 8,
    greater: 9
  }

  for {name, op_const} <- @ops_binary do
    @doc """
    Elementwise `#{name}` of two GPU tensors of equal length.
    Returns `{:ok, tensor}` or `{:error, reason}`.
    """
    def unquote(name)(a, b) do
      Nx.Vulkan.Native.apply_binary(
        a,
        b,
        unquote(op_const),
        shader_path("elementwise_binary.spv")
      )
    end
  end

  @doc false
  def shader_path(name) do
    :nx_vulkan
    |> :code.priv_dir()
    |> Path.join("shaders")
    |> Path.join(name)
  end

  # ------------------------------------------------------------------
  # v0.0.4 — elementwise unary ops
  # ------------------------------------------------------------------

  @ops_unary %{
    exp: 0,
    log: 1,
    sqrt: 2,
    abs: 3,
    negate: 4,
    sigmoid: 5,
    tanh: 6,
    relu: 7,
    ceil: 8,
    floor: 9,
    sign: 10,
    reciprocal: 11,
    square: 12,
    erf: 13,
    expm1: 14
  }

  for {name, op_const} <- @ops_unary do
    @doc "Elementwise `#{name}` of a GPU tensor."
    def unquote(name)(a) do
      Nx.Vulkan.Native.apply_unary(a, unquote(op_const), shader_path("elementwise_unary.spv"))
    end
  end

  # ------------------------------------------------------------------
  # f64 elementwise — Day 6 / step 2c
  # ------------------------------------------------------------------

  @doc "f64 elementwise binary; dispatches elementwise_binary_f64.spv."
  def apply_binary_f64(a, b, op) do
    code = Map.fetch!(@ops_binary, op)
    Nx.Vulkan.Native.apply_binary_f64(a, b, code,
                                       shader_path("elementwise_binary_f64.spv"))
  end

  @doc "f64 elementwise unary; dispatches elementwise_unary_f64.spv."
  def apply_unary_f64(a, op) do
    code = Map.fetch!(@ops_unary, op)
    Nx.Vulkan.Native.apply_unary_f64(a, code,
                                      shader_path("elementwise_unary_f64.spv"))
  end

  @doc "f64 per-axis reduction; dispatches reduce_axis_f64.spv."
  def reduce_axis_f64(a, outer, reduce_size, inner, op) do
    Nx.Vulkan.Native.reduce_axis_f64(a, outer, reduce_size, inner, op,
                                      shader_path("reduce_axis_f64.spv"))
  end

  @doc "f64 broadcast elementwise binary; dispatches elementwise_binary_broadcast_f64.spv."
  def apply_binary_broadcast_f64(a, b, op, ndim, out_shape, a_strides, b_strides) do
    op_const = Map.fetch!(@ops_binary, op)
    Nx.Vulkan.Native.apply_binary_broadcast_f64(
      a, b, op_const, ndim,
      pad4(out_shape), pad4(a_strides), pad4(b_strides),
      shader_path("elementwise_binary_broadcast_f64.spv")
    )
  end

  @doc """
  Numerically-stable logsumexp over a single virtual reduce axis.
  `log(sum(exp(x - max(x))))`-shape inside one shader dispatch via the
  two-pass shader. f32 only.
  """
  def logsumexp(a, outer, reduce_size, inner) do
    Nx.Vulkan.Native.logsumexp(a, outer, reduce_size, inner,
                                shader_path("logsumexp.spv"))
  end

  # ------------------------------------------------------------------
  # v0.0.5 — reductions (return host-side scalar)
  # ------------------------------------------------------------------

  @doc "Sum of all elements (returns a host-side f32)."
  def sum(t), do: Nx.Vulkan.Native.reduce_scalar(t, 0, shader_path("reduce.spv"))

  @doc "Min of all elements."
  def reduce_min(t), do: Nx.Vulkan.Native.reduce_scalar(t, 1, shader_path("reduce.spv"))

  @doc "Max of all elements."
  def reduce_max(t), do: Nx.Vulkan.Native.reduce_scalar(t, 2, shader_path("reduce.spv"))

  @doc "Mean of all elements (sum + host-side divide)."
  def mean(t) do
    case sum(t) do
      {:ok, s} ->
        n = div(Nx.Vulkan.Native.byte_size(t), 4)
        {:ok, s / n}

      err ->
        err
    end
  end

  # ------------------------------------------------------------------
  # v0.0.6 — matmul (naive)
  # ------------------------------------------------------------------

  @doc """
  Matrix multiply: `C[M*N] = A[M*K] · B[K*N]`. All row-major f32.
  Returns `{:ok, c_tensor}`.

  Auto-selects the best shader variant based on `(M, N, K)`:

    * Tiny (M*N*K < 4096): naive `matmul.spv` — dispatch overhead
      dominates; tiling adds no win.
    * Medium (4096 ≤ M*N*K < 256³): `matmul_tiled.spv` (16×16 shared-
      memory tiles) — good cache behavior, modest GPU occupancy.
    * Large (M*N*K ≥ 256³ ≈ 16M): `matmul_tiled16x2.spv` — each thread
      computes 2 output rows; mac-248 measured **4.2× win at 1024×1024**
      vs the naive variant.

  `matmul_tiled32.spv` exists in spirit too but only wins on Ampere+
  (1024 threads/SM); on Kepler/Maxwell it loses to 16x2 due to shared
  memory pressure (8 KB tile vs 3 KB). Not auto-selected; reachable
  via `matmul_variant/6`.
  """
  def matmul(a, b, m, n, k) do
    {shader, tile_m, tile_n} = pick_matmul(m, n, k)
    matmul_variant(a, b, m, n, k, shader, tile_m, tile_n)
  end

  @doc """
  Matrix multiply with explicit shader variant. Use when you know
  better than the heuristic, or to benchmark.

      :matmul                 # naive, gx=ceil(N/16), gy=ceil(M/16)
      :matmul_tiled           # 16×16 shared-mem tiles
      :matmul_tiled32         # 32×32 tiles (Ampere wins)
      :matmul_tiled16x2       # 32×16 output (2 rows per thread)
  """
  def matmul_variant(a, b, m, n, k, variant)
      when variant in [:matmul, :matmul_tiled, :matmul_tiled32, :matmul_tiled16x2] do
    {tile_m, tile_n} = variant_tiles(variant)
    matmul_variant(a, b, m, n, k, "#{variant}.spv", tile_m, tile_n)
  end

  @doc false
  def matmul_variant(a, b, m, n, k, shader_name, tile_m, tile_n) do
    Nx.Vulkan.Native.matmul_v(a, b, m, n, k, tile_m, tile_n, shader_path(shader_name))
  end

  @doc """
  Picks the best matmul shader for a given `(M, N, K)` shape. Returns
  `{shader_name, tile_m, tile_n}`. Public so benchmarks can introspect
  the heuristic.
  """
  def pick_matmul(m, n, k) do
    flops = m * n * k

    cond do
      # Below 4K total ops: dispatch + descriptor write costs dominate.
      flops < 4_096 -> {"matmul.spv", 16, 16}
      # 256³ = 16 777 216. mac-248's bench shows the 16x2 variant
      # taking the lead from this size onward.
      flops >= 16_777_216 -> {"matmul_tiled16x2.spv", 32, 16}
      # Middle ground: classic 16×16 tile.
      true -> {"matmul_tiled.spv", 16, 16}
    end
  end

  defp variant_tiles(:matmul), do: {16, 16}
  defp variant_tiles(:matmul_tiled), do: {16, 16}
  defp variant_tiles(:matmul_tiled32), do: {32, 32}
  defp variant_tiles(:matmul_tiled16x2), do: {32, 16}

  # ------------------------------------------------------------------
  # v0.0.7 — random
  # ------------------------------------------------------------------

  @doc "Generate `n` uniform [0,1) f32 values, deterministic via `seed`."
  def uniform(n, seed \\ 42) when is_integer(n) and is_integer(seed) do
    Nx.Vulkan.Native.random(n, seed, 0, shader_path("random_philox.spv"))
  end

  @doc "Generate `n` standard-normal N(0,1) f32 values via Box-Muller."
  def normal(n, seed \\ 42) when is_integer(n) and is_integer(seed) do
    Nx.Vulkan.Native.random(n, seed, 1, shader_path("random_philox.spv"))
  end

  # ------------------------------------------------------------------
  # v0.1 phase 1.1 — comparisons via composition
  # ------------------------------------------------------------------

  @doc """
  Branchless select: `cond_true_or_false ? t : f`. `cond` is a 0/1
  tensor (typically the output of `equal/2`, `less/2`, `greater/2`).

  Implemented compositionally as `cond * t + (1 - cond) * f`. Once
  the v0.1 broadcast shader supports scalar broadcast we'll switch
  to a 3-input shader; today this composition is the right shape
  and adds two dispatches' worth of overhead per select.
  """
  def select(cond, t, f) do
    n_bytes = Nx.Vulkan.Native.byte_size(cond)
    n = div(n_bytes, 4)

    with {:ok, ones} <- upload_constant(1.0, n),
         {:ok, inv} <- subtract(ones, cond),
         {:ok, on_t} <- multiply(cond, t),
         {:ok, on_f} <- multiply(inv, f),
         {:ok, out} <- add(on_t, on_f) do
      {:ok, out}
    end
  end

  @doc """
  Clip every element to `[low, high]`. Implemented as
  `max(low, min(high, a))` with broadcasted scalar tensors.
  Replaceable with a single-shader `clip.comp` once the broadcast
  story matures (currently materializes scalars to N-element
  buffers).
  """
  def clip(a, low, high) when is_number(low) and is_number(high) do
    n_bytes = Nx.Vulkan.Native.byte_size(a)
    n = div(n_bytes, 4)

    with {:ok, low_t} <- upload_constant(low, n),
         {:ok, high_t} <- upload_constant(high, n),
         {:ok, capped} <- min(a, high_t),
         {:ok, floored} <- max(low_t, capped) do
      {:ok, floored}
    end
  end

  defp upload_constant(value, n) when is_number(value) do
    f = value / 1.0
    bin = :binary.copy(<<f::float-32-native>>, n)
    Nx.Vulkan.Native.upload_binary(bin)
  end

  # ------------------------------------------------------------------
  # v0.1 phase 1.2 — reshape (metadata-only) + broadcast (Backend) +
  # transpose (new shader)
  # ------------------------------------------------------------------

  @doc """
  2D transpose: `c = a^T` where `a` is M×N and `c` is N×M, both
  row-major f32. Returns `{:ok, c_tensor}`.
  """
  def transpose_2d(a, m, n) do
    Nx.Vulkan.Native.transpose(a, m, n, shader_path("transpose.spv"))
  end

  # ------------------------------------------------------------------
  # v0.1 phase 1.8 GPU path — f32↔f64 cast
  # ------------------------------------------------------------------

  @doc "Cast f32 tensor → f64 (allocates 8-byte output)."
  def cast_f32_to_f64(a, n) do
    Nx.Vulkan.Native.cast(a, n, 8, shader_path("cast_f32_to_f64.spv"))
  end

  @doc "Cast f64 tensor → f32 (allocates 4-byte output)."
  def cast_f64_to_f32(a, n) do
    Nx.Vulkan.Native.cast(a, n, 4, shader_path("cast_f64_to_f32.spv"))
  end

  # ------------------------------------------------------------------
  # v0.1 phase 1.4 GPU path — per-axis reduce
  # ------------------------------------------------------------------

  @doc """
  Per-axis reduction over a virtual 3-D layout (outer, reduce, inner).
  `op`: 0=sum, 1=max, 2=min. Output is (outer * inner) f32.
  """
  def reduce_axis(a, outer, reduce_size, inner, op) do
    Nx.Vulkan.Native.reduce_axis(a, outer, reduce_size, inner, op, shader_path("reduce_axis.spv"))
  end

  # ------------------------------------------------------------------
  # Path A — fused elementwise chain (FUSION_RESEARCH.md)
  # ------------------------------------------------------------------

  @op_codes %{
    # Binary ops — second operand is always buffer `b`.
    add: 0,
    multiply: 1,
    subtract: 2,
    divide: 3,
    pow: 4,
    max: 5,
    min: 6,
    # Unary ops — operate on the running register only.
    exp: 100,
    log: 101,
    sqrt: 102,
    abs: 103,
    negate: 104,
    sigmoid: 105,
    tanh: 106,
    relu: 107,
    ceil: 108,
    floor: 109,
    sign: 110,
    reciprocal: 111,
    square: 112,
    erf: 113,
    expm1: 114
  }

  @doc """
  Run a chain of up to 8 elementwise ops in a single shader dispatch.

  Replaces N separate dispatches with one. Each binary step combines the
  running register with `b`; each unary step transforms the register only.

      iex> {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      iex> {:ok, b} = Nx.Vulkan.upload_f32([0.5, 0.5, 0.5])
      iex> # (a * b) + b → exp
      iex> {:ok, c} = Nx.Vulkan.fused_chain(a, b, [:multiply, :add, :exp])
      iex> {:ok, vals} = Nx.Vulkan.download_f32(c, 3)
      iex> vals  # exp((a*b)+b) = exp(1.0), exp(1.5), exp(2.0)
      [2.71828..., 4.48168..., 7.38905...]

  Op atoms supported:

    * Binary (combine register with `b`): `:add`, `:multiply`, `:subtract`,
      `:divide`, `:pow`, `:max`, `:min`
    * Unary (transform register): `:exp`, `:log`, `:sqrt`, `:abs`,
      `:negate`, `:sigmoid`, `:tanh`, `:relu`, `:ceil`, `:floor`,
      `:sign`, `:reciprocal`, `:square`

  Note: `:erf` (113) and `:expm1` (114) became fully functional in
  the chain after spirit `161296d1` — `apply_unary` switched in cases
  13 and 14. Earlier versions of the fused shader passed them through
  unchanged.

  Chains longer than 8 ops should be split: dispatch fused_chain twice
  with the running tensor used as `a` for the second call.
  """
  def fused_chain(a_ref, b_ref, ops) when is_list(ops) do
    codes = Enum.map(ops, &Map.fetch!(@op_codes, &1))
    Nx.Vulkan.Native.fused_chain(a_ref, b_ref, codes, shader_path("fused_elementwise.spv"))
  end

  @doc """
  4-input fused chain. `ops_with_buf` items are either `{op_atom, idx}`
  for binary (idx ∈ {1, 2, 3} for b/c/d) or plain `op_atom` for unary.
  All 4 buffers must be the same byte size; up to 8 ops.
  """
  def fused_chain_4(a_ref, b_ref, c_ref, d_ref, ops_with_buf)
      when is_list(ops_with_buf) do
    {codes, buf_idx} =
      ops_with_buf
      |> Enum.map(fn
        {op, idx} -> {Map.fetch!(@op_codes, op), idx}
        op when is_atom(op) -> {Map.fetch!(@op_codes, op), 1}
      end)
      |> Enum.unzip()

    Nx.Vulkan.Native.fused_chain_4(
      a_ref, b_ref, c_ref, d_ref, codes, buf_idx,
      shader_path("fused_elementwise_4in.spv")
    )
  end

  @doc """
  Fused kinetic-energy primitive: `0.5 * sum(p² * inv_mass)` reduced
  per workgroup. Returns a buffer of `ceil(n/256)` partial f32 sums;
  caller does the final reduction (typically via `Nx.Vulkan.sum/1` or
  on the host).
  """
  def kinetic_energy(p_ref, inv_mass_ref) do
    Nx.Vulkan.Native.kinetic_energy(p_ref, inv_mass_ref,
                                     shader_path("kinetic_energy.spv"))
  end

  @doc """
  Fused Normal log-density primitive:
  `-0.5*((x-mu)/sigma)² - log(sigma) - 0.5*log(2π)`.
  Output shape matches `x`. f32 only.
  """
  def normal_logpdf(x_ref, mu_ref, sigma_ref) do
    Nx.Vulkan.Native.normal_logpdf(x_ref, mu_ref, sigma_ref,
                                    shader_path("normal_logpdf.spv"))
  end

  @doc """
  Fused NUTS leapfrog step for a univariate Normal log-density model.
  One Vulkan dispatch per leapfrog step instead of ~12 elementwise
  dispatches via the IR walker. Returns `{q_new_ref, p_new_ref}`.

  `q_ref`, `p_ref`, `inv_mass_ref` are f32 buffers of identical size.
  `eps`, `mu`, `sigma` are scalars (f32 in the shader push constants;
  f64 here for caller convenience). f32 only.

  Closed-form gradient:
  `grad_q log N(q | mu, sigma) = -(q - mu) / sigma²` — no autodiff
  machinery in the shader.
  """
  def leapfrog_normal(q_ref, p_ref, inv_mass_ref, eps, mu, sigma) do
    Nx.Vulkan.Native.leapfrog_normal(
      q_ref, p_ref, inv_mass_ref,
      eps, mu, sigma,
      shader_path("leapfrog_normal.spv")
    )
  end

  # ------------------------------------------------------------------
  # Phase 2 — Nx.Defn JIT integration
  # ------------------------------------------------------------------

  @doc """
  JIT-compile a function so each op dispatches through the Vulkan backend.

  Symmetric counterpart of `EXLA.jit/2` and `EMLX.jit/2`. There's no
  kernel fusion in v0.1 — each `Nx.*` call inside the defn becomes one
  shader dispatch via `Nx.Defn.Evaluator`. Combined-shader fusion is the
  v0.2 work (see FUSION_RESEARCH.md).

  Sets `Nx.Vulkan.Backend` as the global default if it isn't already, so
  scalars and tensors created inside the defn land on the GPU. Calls
  `Nx.Vulkan.init/0` (idempotent).

      iex> Nx.Vulkan.init()
      :ok
      iex> f = fn x -> Nx.add(x, x) end
      iex> Nx.Vulkan.jit(f).(Nx.tensor([1.0, 2.0]))
      #Nx.Tensor<f32[2] [2.0, 4.0]>
  """
  def jit(fun, opts \\ []) do
    ensure_default_backend!()
    compiler = Keyword.get(opts, :compiler, Nx.Vulkan.Compiler)
    Nx.Defn.jit(fun, [{:compiler, compiler} | Keyword.delete(opts, :compiler)])
  end

  defp ensure_default_backend! do
    case Nx.default_backend() do
      {Nx.Vulkan.Backend, _} ->
        :ok

      _ ->
        :ok = init()
        Nx.global_default_backend(Nx.Vulkan.Backend)
        :ok
    end
  end

  # ------------------------------------------------------------------
  # Broadcast elementwise binary
  # ------------------------------------------------------------------

  @doc """
  Dispatch the broadcast variant of an elementwise binary op. `op` is
  one of the binary atom keys in `@ops_binary`, ndim ≤ 4. Stride of 0
  on an axis means broadcast on that axis. `out_shape`, `a_strides`,
  `b_strides` are lists; the helper pads to length 4.

  Use `Nx.Vulkan.broadcast_strides/2` to compute strides from a source
  shape against the output shape.
  """
  def apply_binary_broadcast(a, b, op, ndim, out_shape, a_strides, b_strides) do
    op_const = Map.fetch!(@ops_binary, op)

    Nx.Vulkan.Native.apply_binary_broadcast(
      a,
      b,
      op_const,
      ndim,
      pad4(out_shape),
      pad4(a_strides),
      pad4(b_strides),
      shader_path("elementwise_binary_broadcast.spv")
    )
  end

  @doc """
  Per-axis strides for broadcasting `src_shape` to `out_shape`.

  Returns a length-4 list (zero-padded). Stride is 0 on a broadcast axis
  (size 1 in `src` but >1 in `out`); otherwise it's the row-major
  product of trailing source dims.

      iex> Nx.Vulkan.broadcast_strides({1, 4}, {3, 4})
      [0, 1, 0, 0]
      iex> Nx.Vulkan.broadcast_strides({2, 1}, {2, 4})
      [1, 0, 0, 0]
  """
  def broadcast_strides(src_shape, out_shape) do
    src = Tuple.to_list(src_shape)
    out = Tuple.to_list(out_shape)
    rank = length(out)

    # Right-align: pad src with 1s on the left so the trailing dims align.
    pad = rank - length(src)
    src_aligned = List.duplicate(1, pad) ++ src

    {strides, _} =
      Enum.zip(src_aligned, out)
      |> Enum.reverse()
      |> Enum.reduce({[], 1}, fn {sd, od}, {acc, running} ->
        cond do
          sd == od ->
            {[running | acc], running * sd}

          sd == 1 ->
            {[0 | acc], running}

          true ->
            raise ArgumentError,
                  "shapes don't broadcast: #{inspect(src_shape)} → #{inspect(out_shape)}"
        end
      end)

    pad4(strides)
  end

  defp pad4(list) do
    # Kernel.max because Kernel.max/2 is excluded at the top of this
    # module (Nx.Vulkan.max/2 is the GPU op, not the integer max).
    n_pad = Kernel.max(0, 4 - length(list))
    Enum.take(list, 4) ++ List.duplicate(0, n_pad)
  end

  # ------------------------------------------------------------------
  # Buffer pool — Week 1 step 1a (PATH_TO_FULL_PASS.md)
  # ------------------------------------------------------------------

  @doc """
  Release every pooled VkBuf back to the device. Call at idle time to
  reclaim memory; otherwise the pool grows to working-set size and stays
  there. Idempotent.
  """
  defdelegate pool_clear(), to: Nx.Vulkan.Native

  @doc """
  Buffer pool stats. Returns `{:ok, %{hits, misses, freed,
  size_classes, total_pooled}}`. `hits/misses` count alloc requests
  served from / missed by the pool; `freed` counts buffers actually
  vkFreeMemory'd (pool-overflow or explicit clear); `size_classes` is
  the number of distinct sizes currently held; `total_pooled` is the
  total VkBuf count waiting for reuse.

      iex> Nx.Vulkan.init()
      iex> Nx.Vulkan.pool_stats()
      {:ok, %{hits: _, misses: _, freed: _, size_classes: _, total_pooled: _}}
  """
  defdelegate pool_stats(), to: Nx.Vulkan.Native
end
