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
  (`Codegen.emit_fused_reduce` + `dispatch_generated_reduce`). This is enabled
  by default for the regime where it reliably wins — a full reduction or a
  contiguous last-axis reduce (`inner_stride == 1`) with few output slots. There
  it beats even the eager path, whose own `reduce_axis` is one-thread-per-slot
  serial: a full `sum` measured ~16x over eager on the GT 650M, which is exactly
  the case where EXLA out-ran the eager backend. Reductions with many output
  slots, a non-contiguous axis, or a short reduce axis stay on the eager path
  (already parallel enough). `NXV_FUSE_REDUCE=1` forces fusion for any
  contiguous reduce; `=0` disables it. See `reduce_beneficial?/3`.
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
  # it uses a parallel workgroup-per-slot shared-memory tree reduce (256 threads
  # cooperate on each output slot). That beats even the eager path, whose own
  # `reduce_axis` is one-thread-per-slot serial — which is exactly why EXLA wins
  # `sum`. Valid only when the number of output slots fits the 1-D workgroup
  # count limit (@max_wg_slots) and the reduce axis is large enough to amortise
  # the workgroup (@min_reduce_for_fuse); otherwise the eager path (fully-
  # parallel elementwise + reduce) is already fine, so fall back.
  defp try_fuse(%T{data: %Expr{op: op, args: [inner, red_opts]}} = result)
       when op in [:sum, :reduce_max, :reduce_min] do
    with true <- match?(%T{type: {:f, 32}}, inner),
         true <- Codegen.fusable?(inner),
         {:ok, outer, rsize, inner_stride} <- reduce_dims(inner.shape, red_opts[:axes]),
         true <- reduce_beneficial?(outer * inner_stride, rsize, inner_stride) do
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

  # When to use the parallel fused reduce. The fused shader gives each output
  # slot a whole 256-thread workgroup; that only beats eager when eager's own
  # reduce is under-parallelised, i.e. when there are FEW output slots. The
  # canonical case is a full reduction (slots = 1): eager's `reduce_axis` runs
  # it single-threaded, so fused wins ~16x on the GT 650M — exactly the case
  # where EXLA out-ran the eager backend. With many slots eager already has
  # enough threads and the extra tree-reduce overhead makes fused a wash or a
  # small loss, so we fall back. Constraints:
  #   * inner_stride == 1 — a full reduction or a contiguous LAST-axis reduce;
  #     inner > 1 makes each thread's reads strided/uncoalesced (measured 0.23x).
  #   * slots <= @max_fuse_slots — the small-output regime where the win is real
  #     and reproducible; above it, benchmarks were noisy/neutral so eager wins.
  #   * reduce axis long enough to fill the workgroup, else eager wins.
  # NXV_FUSE_REDUCE=1 forces on for any contiguous reduce within the hard
  # workgroup-count limit (still requires inner == 1); =0 forces off.
  @hard_wg_limit 65535
  @max_fuse_slots 256
  @min_reduce_for_fuse 64

  defp reduce_beneficial?(slots, reduce_size, inner_stride) do
    force = System.get_env("NXV_FUSE_REDUCE")

    cond do
      force == "0" -> false
      inner_stride != 1 -> false
      slots < 1 or slots > @hard_wg_limit -> false
      force == "1" -> true
      true -> slots <= @max_fuse_slots and reduce_size >= @min_reduce_for_fuse
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
