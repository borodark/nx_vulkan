defmodule Nx.Vulkan.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` for the Vulkan backend — thrust 3, the fusion compiler.

  EXLA's structural advantage over an eager backend is whole-graph
  compilation: it fuses a chain of elementwise ops into a single kernel
  instead of dispatching each op separately (each with its own launch +
  intermediate buffer). This compiler does the same for the cases it
  supports.

  At `__compile__` time it traces the `defn` to an `Nx.Defn.Expr` tree. If the
  single output is a same-shape f32 elementwise chain (see `Nx.Vulkan.Codegen`),
  it generates one GLSL shader for the whole chain, compiles it once (cached by
  source hash), and returns a function that uploads the inputs, issues ONE
  `dispatch_generated` call, and hands back a GPU-resident result. Anything it
  can't fuse — tuple outputs, reductions, dot/conv, broadcasting between
  differing shapes, non-f32 — falls through to `Nx.Defn.Evaluator`, so results
  are always correct; the worst case is "no fusion, same as eager."

  ## Usage

      Nx.Defn.jit(&my_fun/2, compiler: Nx.Vulkan.Compiler).(a, b)

  Set `NXV_FUSE_DEBUG=1` to log which path each `defn` takes.

  ## Reductions

  An elementwise chain feeding a reduction (`sum`/`reduce_max`/`reduce_min`) is
  fused into a single **parallel workgroup-per-slot shared-memory tree reduce**
  (`Codegen.emit_fused_reduce` + `dispatch_generated_reduce`), which grid-strides
  over output slots so one launch handles any slot count. It beats even the eager
  path, whose own `reduce_axis` is one-thread-per-slot serial. Enabled by default
  for a contiguous reduce (`inner_stride == 1`) with FEW output slots — full
  reductions and small-output reductions — which win across the fleet: ~8-27x
  over eager on the GT 650M and ~2.8-6.7x on the RTX 3060 Ti (the case where EXLA
  had out-run the eager backend). The many-slot wide-reduce regime is grid-
  stride-capable and wins on the weak Kepler eager path (~4.4x) but REGRESSES on
  the much stronger Ampere eager path (0.44x), so it is hardware-dependent: it is
  auto-enabled only on GPUs `Nx.Vulkan.Device` classifies `:weak` (integrated /
  software / older low-end discrete), and stays off on strong GPUs. Force it on
  any GPU with `NXV_FUSE_REDUCE=1`. Non-contiguous, short-axis and mid-slot
  reductions fall back to the already-parallel eager path — no regressions. `=0`
  disables all reduce fusion. See `reduce_beneficial?/3` and `Nx.Vulkan.Device`.
  """

  @behaviour Nx.Defn.Compiler

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr
  alias Nx.Vulkan.{Codegen, VulkanoBackend, NativeV}

  @impl true
  def __partitions_options__(opts) do
    List.duplicate(opts, Keyword.get(opts, :max_concurrency, 1))
  end

  @impl true
  def __to_backend__(_opts) do
    Nx.default_backend()
  end

  @impl true
  def __jit__(key, vars, fun, args_list, opts) do
    __compile__(key, vars, fun, opts).(args_list)
  end

  @impl true
  def __compile__(key, vars, fun, opts) do
    result = fun.(vars)

    case try_fuse(result) do
      {:ok, spv_path, param_order, template} ->
        debug(fn -> "FUSED elementwise: root=#{op_of(result)} n_in=#{length(param_order)}" end)
        fn [params] -> [run_fused(spv_path, param_order, template, params)] end

      {:ok_reduce, spv_path, param_order, {outer, rsize, inner}, template} ->
        debug(fn -> "FUSED reduce: root=#{op_of(result)} n_in=#{length(param_order)} o/r/i=#{outer}/#{rsize}/#{inner}" end)

        fn [params] ->
          [run_fused_reduce(spv_path, param_order, {outer, rsize, inner}, template, params)]
        end

      :fallback ->
        debug(fn -> "fallback (evaluator): root=#{inspect(op_of(result))}" end)
        Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)
    end
  end

  @impl true
  def __shard_jit__(_key, _mesh, _vars, _fun, _args_list, _opts) do
    raise "sharding is not supported by Nx.Vulkan.Compiler"
  end

  # ---- fusion decision -------------------------------------------------

  # A reduce root (sum / product / reduce_max / reduce_min) over a fusable
  # elementwise inner → one fused shader that evaluates the chain and reduces in
  # a single parallel dispatch (see `fuse_reduce/5` and `Codegen`).
  defp try_fuse(%T{data: %Expr{op: op, args: [inner, red_opts]}} = result)
       when op in [:sum, :product, :reduce_max, :reduce_min] do
    fuse_reduce(result, inner, op, red_opts[:axes], nil)
  end

  # `mean` lowers to `divide(sum(inner, axes), n)` — fuse it as a `sum` with a
  # `/n` post-scale baked into the shader. Any other divide falls through to the
  # general elementwise path.
  defp try_fuse(
         %T{
           data: %Expr{
             op: :divide,
             args: [
               %T{data: %Expr{op: :sum, args: [inner, red_opts]}},
               %T{data: %Expr{op: :constant, args: [n]}}
             ]
           }
         } = result
       )
       when is_number(n) do
    case fuse_reduce(result, inner, :sum, red_opts[:axes], n) do
      :fallback -> fuse_elementwise(result)
      ok -> ok
    end
  end

  defp try_fuse(%T{data: %Expr{}} = result), do: fuse_elementwise(result)

  # Non-tensor (tuple / container) output — not fused yet.
  defp try_fuse(_), do: :fallback

  # Compile a fused reduce for `reduce_op(inner, axes)` with an optional `/scale`
  # post-op (for mean). Falls back unless the inner is a same-shape f32
  # elementwise chain, the reduce is contiguous, and `reduce_beneficial?/3` says
  # the parallel reduce wins on this GPU.
  defp fuse_reduce(result, inner, reduce_op, axes, scale) do
    with true <- match?(%T{type: {:f, 32}}, inner),
         true <- Codegen.fusable?(inner),
         {:ok, outer, rsize, inner_stride} <- reduce_dims(inner.shape, axes),
         true <- reduce_beneficial?(outer * inner_stride, rsize, inner_stride) do
      {glsl, meta} = Codegen.emit_fused_reduce(inner, reduce_op, scale)

      case Codegen.compile_cached(glsl) do
        {:ok, spv_path} ->
          {:ok_reduce, spv_path, meta.param_order, {outer, rsize, inner_stride},
           Nx.to_template(result)}

        {:error, _} ->
          :fallback
      end
    else
      _ -> :fallback
    end
  end

  defp fuse_elementwise(result) do
    if Codegen.fusable?(result) do
      {glsl, meta} = Codegen.emit_elementwise(result)

      case Codegen.compile_cached(glsl) do
        {:ok, spv_path} -> {:ok, spv_path, meta.param_order, Nx.to_template(result)}
        {:error, _} -> :fallback
      end
    else
      :fallback
    end
  end

  # When to use the parallel fused reduce. The fused shader gives each output
  # slot a 256-thread workgroup doing a coalesced tree reduce, and grid-strides
  # over slots so it handles any slot count. It requires a contiguous reduce
  # (inner_stride == 1 — full reduction or last-axis; inner > 1 is uncoalesced,
  # measured 0.23x).
  #
  # DEFAULT REGIME — few slots (slots <= @few_slots, reduce >= @min_reduce_few):
  # eager's `reduce_axis` is one-thread-per-slot, so with few slots it is badly
  # under-parallelised. Fused wins here on BOTH the weak Kepler and the strong
  # Ampere GPUs — a full reduction (slots=1) is ~8-27x over eager on the GT 650M
  # and ~2.8-6.7x on the RTX 3060 Ti (the case where EXLA had out-run eager).
  #
  # The many-slot wide-reduce regime (slots >= @many_slots, reduce >=
  # @min_reduce_many) is grid-stride-capable and wins ~4.4x on Kepler, but it
  # REGRESSES on Ampere (0.44x): a 3060 Ti's eager one-thread-per-slot path is
  # already well-fed by thousands of slots, leaving the fused kernel no headroom,
  # only overhead. So it is hardware-dependent — auto-enabled only on GPUs
  # `Nx.Vulkan.Device` classifies `:weak` (integrated / software / older
  # low-end discrete), where it's measured to win. Force on any GPU with
  # NXV_FUSE_REDUCE=1; `=0` disables all reduce fusion.
  @few_slots 256
  @min_reduce_few 64
  @many_slots 2048
  @min_reduce_many 256

  defp reduce_beneficial?(slots, reduce_size, inner_stride) do
    force = System.get_env("NXV_FUSE_REDUCE")

    cond do
      force == "0" -> false
      inner_stride != 1 -> false
      slots < 1 -> false
      force == "1" -> true
      slots <= @few_slots -> reduce_size >= @min_reduce_few
      slots >= @many_slots -> reduce_size >= @min_reduce_many and Nx.Vulkan.Device.weak?()
      true -> false
    end
  end

  # Map a reduction to the (outer, reduce_size, inner) view the fused shader
  # loops over. Supports full reduction (axes nil / all) and a single axis;
  # multi-axis reductions fall back.
  defp reduce_dims({}, _axes), do: {:ok, 1, 1, 1}

  defp reduce_dims(in_shape, axes) do
    dims = Tuple.to_list(in_shape)
    rank = length(dims)
    all = Enum.to_list(0..(rank - 1))

    cond do
      axes == nil or axes == all ->
        {:ok, 1, Enum.reduce(dims, 1, &*/2), 1}

      is_list(axes) and length(axes) == 1 ->
        ax = hd(axes)
        ax = if ax < 0, do: ax + rank, else: ax
        outer = dims |> Enum.take(ax) |> Enum.reduce(1, &*/2)
        rsize = Enum.at(dims, ax)
        inner = dims |> Enum.drop(ax + 1) |> Enum.reduce(1, &*/2)
        {:ok, outer, rsize, inner}

      true ->
        :error
    end
  end

  # ---- runtime dispatch ------------------------------------------------

  defp run_fused(spv_path, param_order, template, params) do
    in_refs =
      Enum.map(param_order, fn pidx ->
        # Defn passes each argument as a zero-arg thunk (see Evaluator).
        tensor = params |> Enum.at(pidx) |> then(& &1.()) |> Nx.devectorize()

        %T{data: %VulkanoBackend{ref: ref}} = Nx.backend_transfer(tensor, VulkanoBackend)
        ref
      end)

    n = Nx.size(template)
    {:ok, out_ref} = NativeV.buf_alloc(n * 4)
    :ok = NativeV.dispatch_generated(out_ref, in_refs, n, spv_path)

    data = %VulkanoBackend{ref: out_ref, shape: template.shape, type: template.type}
    %{template | data: data}
  end

  defp run_fused_reduce(spv_path, param_order, {outer, rsize, inner}, template, params) do
    in_refs =
      Enum.map(param_order, fn pidx ->
        tensor = params |> Enum.at(pidx) |> then(& &1.()) |> Nx.devectorize()
        %T{data: %VulkanoBackend{ref: ref}} = Nx.backend_transfer(tensor, VulkanoBackend)
        ref
      end)

    n_out = max(Nx.size(template), 1)
    {:ok, out_ref} = NativeV.buf_alloc(n_out * 4)
    :ok = NativeV.dispatch_generated_reduce(out_ref, in_refs, outer, rsize, inner, spv_path)

    data = %VulkanoBackend{ref: out_ref, shape: template.shape, type: template.type}
    %{template | data: data}
  end

  # ---- debug -----------------------------------------------------------

  defp op_of(%T{data: %Expr{op: op}}), do: op
  defp op_of(_), do: :not_a_tensor

  defp debug(fun) do
    if System.get_env("NXV_FUSE_DEBUG") == "1", do: IO.puts("[Nx.Vulkan.Compiler] " <> fun.())
    :ok
  end
end
