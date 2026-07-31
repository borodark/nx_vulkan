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

  Fusing an elementwise chain into a reduction (`sum`/`reduce_max`/`reduce_min`)
  is implemented (`Codegen.emit_fused_reduce` + `dispatch_generated_reduce`) but
  **off by default** — enable with `NXV_FUSE_REDUCE=1`. The current fused-reduce
  shader runs one invocation per output slot and loops the reduce axis serially,
  which has far fewer threads than the eager path's fully-parallel elementwise
  stage; it measured 0.3–0.6x (i.e. slower) across the Kepler fleet. Reductions
  therefore stay on the faster eager path until a shared-memory parallel tree
  reduction lands. See the note on `try_fuse/1`.
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

  # Reduce root (sum / reduce_max / reduce_min) over a fusable elementwise inner
  # → one fused shader that evaluates the chain and reduces in a single dispatch.
  #
  # DISABLED BY DEFAULT (opt in with NXV_FUSE_REDUCE=1). The current shader does
  # one invocation per output slot and loops the reduce axis serially with an
  # f64 accumulator; that has `reduce_size`x fewer threads than the eager path's
  # fully-parallel elementwise stage and re-evaluates the body in the loop, so
  # it measured *slower* than eager (elementwise + reduce) in every regime on
  # the Kepler fleet — 0.3-0.6x. The correct fix is a shared-memory parallel
  # tree reduction; until then reductions stay on the faster eager path. The
  # codegen (`emit_fused_reduce`) and `dispatch_generated_reduce` NIF are kept
  # as the scaffold for that work.
  defp try_fuse(%T{data: %Expr{op: op, args: [inner, red_opts]}} = result)
       when op in [:sum, :reduce_max, :reduce_min] do
    with true <- fuse_reduce_enabled?(),
         true <- match?(%T{type: {:f, 32}}, inner),
         true <- Codegen.fusable?(inner),
         {:ok, outer, rsize, inner_stride} <- reduce_dims(inner.shape, red_opts[:axes]) do
      {glsl, meta} = Codegen.emit_fused_reduce(inner, op, Tuple.to_list(inner.shape))

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

  defp try_fuse(%T{data: %Expr{}} = result) do
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

  # Non-tensor (tuple / container) output — not fused yet.
  defp try_fuse(_), do: :fallback

  defp fuse_reduce_enabled?, do: System.get_env("NXV_FUSE_REDUCE") == "1"

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
