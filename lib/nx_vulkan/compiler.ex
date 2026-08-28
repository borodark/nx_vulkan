defmodule Nx.Vulkan.Compiler do
  @moduledoc """
  `Nx.Defn.Compiler` for the Vulkan backend — thrust 3, the fusion compiler.

  EXLA's structural advantage over an eager backend is whole-graph
  compilation: it fuses a chain of elementwise ops into a single kernel
  instead of dispatching each op separately (each with its own launch +
  intermediate buffer). This compiler does the same for the cases it
  supports.

  At `__compile__` time it traces the `defn` to an `Nx.Defn.Expr` tree. If the
  single output is a same-shape elementwise chain (see `Nx.Vulkan.Codegen`), it
  generates one GLSL shader for the whole chain, compiles it once (cached by
  source hash), and returns a function that uploads the inputs, issues ONE
  `dispatch_generated` call, and hands back a GPU-resident result. Anything it
  can't fuse — tuple outputs, batched dot, broadcasting between differing
  shapes, integer dtypes — falls through to `Nx.Defn.Evaluator`, so results are
  always correct; the worst case is "no fusion, same as eager."

  ## Dtypes

  `{:f, 32}` and `{:f, 64}` both fuse; the generated shaders are parameterised
  on the element type and every buffer is sized by it. A graph must be of ONE
  float dtype throughout — mixed precision would need a cast stage this compiler
  does not emit, so it falls back.

  Two f64 caveats:

    * **No f64 transcendentals exist in SPIR-V** (GLSL.std.450 defines
      exp/log/pow/tanh for 16- and 32-bit floats only), so an f64 tree
      containing one falls back whole-graph rather than silently boundary-casting
      through f32. f64 arithmetic, `sqrt`, `abs`, `sign`, `min`/`max`,
      `floor`/`ceil`/`round` all fuse — GLSL.std.450 covers 64-bit for those.
    * f64 fusion requires the device to advertise `shaderFloat64`
      (`Nx.Vulkan.Device.f64?/0`, override `NXV_F64=0|1`). glslangValidator
      compiles fp64 GLSL regardless of the device, so the failure would
      otherwise surface only at dispatch.

  f64 matmul is ~1/32 rate on consumer NVIDIA cards, so f64 fusion is about
  coverage, not speed.

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
  alias Nx.Defn.{Expr, Composite}
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
        debug(fn ->
          "FUSED reduce: root=#{op_of(result)} n_in=#{length(param_order)} o/r/i=#{outer}/#{rsize}/#{inner}"
        end)

        fn [params] ->
          [run_fused_reduce(spv_path, param_order, {outer, rsize, inner}, template, params)]
        end

      :fallback ->
        case try_multistage(result) do
          {:ok_plan, stages, out_sid, template} ->
            debug(fn -> "MULTISTAGE: #{length(stages)} stages, root=#{op_of(result)}" end)
            fn [params] -> [run_plan(stages, out_sid, template, params)] end

          {:ok_plan_multi, stages, out_refs, template} ->
            debug(fn ->
              "MULTISTAGE(multi): #{length(stages)} stages, #{length(out_refs)} outputs"
            end)

            fn [params] -> [run_plan_multi(stages, out_refs, template, params)] end

          :fallback ->
            debug(fn -> "fallback (evaluator): root=#{inspect(op_of(result))}" end)
            Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)
        end
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
    with true <- float_tensor?(inner),
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
    if float_tensor?(result) and Codegen.fusable?(result) do
      {glsl, meta} = Codegen.emit_elementwise(result)

      case Codegen.compile_cached(glsl) do
        {:ok, spv_path} -> {:ok, spv_path, meta.param_order, Nx.to_template(result)}
        {:error, _} -> :fallback
      end
    else
      :fallback
    end
  end

  # ---- multi-stage split (dot / conv boundaries) -----------------------
  #
  # When a graph isn't a single fusable region (it contains a `dot` or `conv`),
  # split it into a schedule of stages that run on-device with GPU-resident
  # intermediates: each `dot` becomes a matmul stage, each `conv` an im2col+GEMM
  # stage, and each maximal fusable elementwise region becomes ONE generated
  # shader whose leaf inputs may be earlier stages' output buffers. This fuses
  # whole NN layers — e.g. `relu(x @ W + b)` runs as a matmul stage + a single
  # fused `max(dot + b, 0)` stage instead of matmul + broadcast-add + relu as
  # three separate eager dispatches; a CNN layer `relu(conv(x, k) + b)` fuses the
  # same way around the conv.

  # f32 matmul uses the f64-accumulator variant (the eager default — matches an
  # f64 reference to f32 round-off); f64 uses the f64 GEMM. Both are tiled.
  @matmul_f32_spv Path.expand("../../priv/shaders/matmul_f32_f64acc.spv", __DIR__)
  @matmul_f64_spv Path.expand("../../priv/shaders/matmul_f64.spv", __DIR__)

  defp matmul_spv({:f, 32}), do: @matmul_f32_spv
  defp matmul_spv({:f, 64}), do: @matmul_f64_spv

  # Transpose is NOT a view like reshape/squeeze — it permutes axes, so it moves
  # data and needs a real dispatch producing a new buffer. Reuse the eager
  # backend's 2-D shaders (same envelope: 2-D, [1,0], f32/f64).
  @transpose_f32_spv Path.expand("../../priv/shaders/transpose_f32.spv", __DIR__)
  @transpose_f64_spv Path.expand("../../priv/shaders/transpose_f64.spv", __DIR__)

  defp transpose_spv({:f, 32}), do: @transpose_f32_spv
  defp transpose_spv({:f, 64}), do: @transpose_f64_spv

  defp ebytes({:f, 32}), do: 4
  defp ebytes({:f, 64}), do: 8

  defp try_multistage(%T{type: type} = result) do
    if supported_type?(type) and has_boundary?(result) do
      try do
        {ref, state} = plan_node(result, new_plan_state(result))

        case ref do
          {:stage, sid} -> {:ok_plan, Enum.reverse(state.stages), sid, Nx.to_template(result)}
          _ -> :fallback
        end
      catch
        :unschedulable -> :fallback
      end
    else
      :fallback
    end
  end

  # non-float single tensor: not schedulable
  defp try_multistage(%T{}), do: :fallback

  # Composite output (a tuple / container of tensors): plan every leaf through ONE
  # shared stage schedule, so subexpressions common to several outputs are computed
  # once (memoised) and each output is a distinct buffer. Enables e.g. `{mean, var}`
  # layernorm stats or `{softmax, logsumexp}` to fuse in a single graph instead of
  # falling back to the Evaluator. Requires every leaf to be f32 and at least one to
  # carry a boundary; any unschedulable leaf drops the whole tuple to the Evaluator.
  defp try_multistage(result) do
    leaves = Composite.flatten_list([result])

    if leaves != [] and Enum.all?(leaves, &float_tensor?/1) and
         Enum.any?(leaves, &has_boundary?/1) do
      try do
        {template, {refs_rev, state}} =
          Composite.traverse(
            result,
            {[], new_plan_state(result)},
            fn leaf, {refs, st} ->
              {ref, st} = plan_node(leaf, st)
              {Nx.to_template(leaf), {[ref | refs], st}}
            end
          )

        {:ok_plan_multi, Enum.reverse(state.stages), Enum.reverse(refs_rev), template}
      catch
        :unschedulable -> :fallback
      end
    else
      :fallback
    end
  end

  # Fresh plan state for `result`, with the cross-stage-CSE hoist set precomputed.
  defp new_plan_state(result) do
    %{stages: [], memo: %{}, counter: 0, shared: hoist_ids(result)}
  end

  @doc false
  # Test hook: the cross-stage-CSE hoist ANALYSIS for a result expression
  # (ungated — exercises the sharing detection regardless of the NXV_CSE default).
  def __hoist_ids__(result), do: do_hoist_ids(result)

  # Boundary ops — a reference through one of these definitely crosses a stage.
  @boundary_ops [:sum, :product, :reduce_max, :reduce_min, :dot, :conv, :transpose]

  # Cross-stage CSE: the set of node ids worth materialising once and reading as a
  # buffer, rather than re-inlining into every consumer. A fusable node qualifies
  # when it is referenced by >= 2 distinct consumers (distinct parents, plus being
  # an output counts as one) AND at least one of those references CROSSES a stage
  # boundary — a boundary-op parent (reduce/dot/conv) or being reused as an output.
  # Sharing purely among elementwise parents is skipped: those fuse into one region
  # where the in-shader CSE (emit_dag) already computes the node once, so hoisting
  # there would only add a needless dispatch.
  defp hoist_ids(result) do
    # Cross-stage CSE (hoisting a boundary-crossing shared subexpr into its own
    # stage) is DEFAULT-OFF: the fleet race (bench_results/CSE_SOFTMAX_RACE.md)
    # found it never wins — it regresses ~0.72-0.87x on small/medium softmax and
    # is at best neutral (~1.0x) at large sizes, on BOTH Kepler (GT 650M) and
    # Ampere (RTX 3060 Ti). Hoisting swaps a cheap recompute for an extra dispatch
    # + a global-memory round-trip, which dominates. Opt in with NXV_CSE=1 for the
    # rare graph with a genuinely expensive shared subexpr. NOTE: this only gates
    # the *hoisting*; the always-beneficial reuse of a stage already materialised
    # for another tuple output (via `memo`) is independent and stays on.
    if System.get_env("NXV_CSE") == "1" do
      do_hoist_ids(result)
    else
      MapSet.new()
    end
  end

  defp do_hoist_ids(result) do
    roots = Composite.flatten_list([result])
    out_ids = MapSet.new(Enum.map(roots, fn %T{data: %Expr{id: id}} -> id end))

    {parents, _} =
      Enum.reduce(roots, {%{}, MapSet.new()}, fn r, {p, s} -> walk_parents(r, p, s) end)

    parents
    |> Enum.filter(fn {cid, %{ids: pids, crosses: crosses}} ->
      is_out = MapSet.member?(out_ids, cid)
      consumers = MapSet.size(pids) + if(is_out, do: 1, else: 0)
      consumers >= 2 and (crosses or is_out)
    end)
    |> Enum.map(&elem(&1, 0))
    |> MapSet.new()
  end

  # Build child_id -> %{ids: distinct-parent-ids, crosses: has a boundary parent}.
  defp walk_parents(%T{data: %Expr{id: id, op: op, args: args}}, parents, seen) do
    if MapSet.member?(seen, id) do
      {parents, seen}
    else
      seen = MapSet.put(seen, id)
      boundary? = op in @boundary_ops

      Enum.reduce(args, {parents, seen}, fn
        %T{data: %Expr{id: cid}} = child, {p, s} ->
          entry = Map.get(p, cid, %{ids: MapSet.new(), crosses: false})
          entry = %{ids: MapSet.put(entry.ids, id), crosses: entry.crosses or boundary?}
          walk_parents(child, Map.put(p, cid, entry), s)

        _, acc ->
          acc
      end)
    end
  end

  # plan_node returns {ref, state} where ref is {:param, pidx} | {:stage, sid}.
  defp plan_node(%T{data: %Expr{id: id}} = node, state) do
    case Map.get(state.memo, id) do
      nil -> plan_new(node, state)
      ref -> {ref, state}
    end
  end

  defp plan_new(%T{data: %Expr{op: :parameter, args: [pidx]}} = node, state) do
    ref = {:param, pidx}
    {ref, memoize(state, node, ref)}
  end

  # reshape / squeeze are pure row-major views: same bytes, a new logical shape,
  # no dispatch. Alias the input buffer and let downstream stages read it with the
  # reshaped dims (every stage's dims come from the Expr tree, not the buffer).
  # This lets a CNN classifier head — conv -> flatten (reshape) -> dense (dot) —
  # fuse end-to-end. Safe because every buffer we schedule (matmul/conv/fused/param)
  # is contiguous row-major, so a reshape over it really is just a relabel.
  defp plan_new(%T{data: %Expr{op: op, args: [inp | _]}} = node, state)
       when op in [:reshape, :squeeze] do
    {ref, state} = plan_node(inp, state)
    {ref, memoize(state, node, ref)}
  end

  defp plan_new(%T{data: %Expr{op: :dot, args: [a, ca, ba, b, cb, bb]}} = node, state) do
    dot_2d!(node, a, ca, ba, b, cb, bb)
    {a_ref, state} = plan_node(a, state)
    {b_ref, state} = plan_node(b, state)
    {m, k} = {elem(a.shape, 0), elem(a.shape, 1)}
    n = elem(b.shape, 1)
    {sid, state} = new_sid(state)

    state =
      add_stage(
        state,
        {:dot, sid, a_ref, b_ref, m, n, k, ebytes(node.type), matmul_spv(node.type)}
      )

    ref = {:stage, sid}
    {ref, memoize(state, node, ref)}
  end

  # 2-D transpose as a stage: unlike reshape/squeeze (row-major views that alias
  # the input buffer) a transpose permutes axes, so it must materialise a new
  # contiguous buffer via the eager backend's transpose shader. Covering it lets
  # `x @ Wᵀ` — the standard dense-layer form — schedule as transpose stage +
  # matmul stage instead of dropping the whole graph to the Evaluator. The
  # envelope matches `VulkanoBackend.transpose/3` exactly (2-D, `[1,0]`,
  # f32/f64); higher rank or any other permutation throws :unschedulable.
  defp plan_new(%T{data: %Expr{op: :transpose, args: [inp, axes]}} = node, state) do
    unless tuple_size(inp.shape) == 2 and axes == [1, 0] and supported_type?(node.type),
      do: throw(:unschedulable)

    {in_ref, state} = plan_node(inp, state)
    {m, n} = {elem(inp.shape, 0), elem(inp.shape, 1)}
    {sid, state} = new_sid(state)

    state =
      add_stage(
        state,
        {:transpose, sid, in_ref, m, n, ebytes(node.type), transpose_spv(node.type)}
      )

    ref = {:stage, sid}
    {ref, memoize(state, node, ref)}
  end

  defp plan_new(%T{data: %Expr{op: :conv, args: [inp, kernel, opts]}} = node, state) do
    conv_schedulable!(node, inp, kernel, opts)
    {in_ref, state} = plan_node(inp, state)
    {k_ref, state} = plan_node(kernel, state)
    plan = VulkanoBackend.conv_plan(node.type, inp.shape, kernel.shape, node.shape, opts)
    {sid, state} = new_sid(state)
    state = add_stage(state, {:conv, sid, in_ref, k_ref, plan})
    ref = {:stage, sid}
    {ref, memoize(state, node, ref)}
  end

  # A reduction (sum / product / reduce_max / reduce_min) materialised as its own
  # stage: the parallel tree reduce over the fusable inner chain writes a small
  # GPU buffer that downstream stages consume (broadcast-aware). This lets
  # `x - mean(x)` (layernorm / softmax) fuse — the sum becomes a reduce stage and
  # the surrounding subtract/divide fuse as a region reading it. `mean` needs no
  # special case: it lowers to `divide(sum, n)`, the divide fuses in the consuming
  # region and the `sum` falls here as a leaf. Gated by the SAME fleet-validated
  # `reduce_beneficial?` as the standalone reduce, so multi-stage never forces a
  # reduce the fleet says regresses on this GPU — it falls back whole-graph to the
  # Evaluator instead (still correct), keeping behaviour consistent with try_fuse.
  defp plan_new(%T{data: %Expr{op: op, args: [inner, red_opts]}} = node, state)
       when op in [:sum, :product, :reduce_max, :reduce_min] do
    unless float_tensor?(inner), do: throw(:unschedulable)

    {outer, rsize, inner_stride} =
      case reduce_dims(inner.shape, red_opts[:axes]) do
        {:ok, o, r, i} -> {o, r, i}
        :error -> throw(:unschedulable)
      end

    unless reduce_beneficial?(outer * inner_stride, rsize, inner_stride),
      do: throw(:unschedulable)

    # If the reduced `inner` is itself a stage boundary (already materialised, or
    # hoisted for CSE, or non-fusable) it becomes the reduce's single buffer input
    # — the reduce reads it instead of re-emitting its arithmetic. Otherwise its
    # fusable chain inlines into the reduce shader (with shared descendants read
    # from their buffers).
    leaves =
      if stage_leaf?(inner, state.memo, state.shared),
        do: [inner],
        else: region_leaves(inner, state.memo, state.shared)

    {inputs, input_refs, state} =
      Enum.reduce(leaves, {[], [], state}, fn leaf, {ins, refs, st} ->
        {key, lref, st} = plan_leaf(leaf, st)
        {ins ++ [{key, leaf.shape}], refs ++ [lref], st}
      end)

    {glsl, _} = Codegen.emit_reduce_region(inner, op, inputs)
    spv = compile!(glsl)
    n_out = max(Nx.size(node), 1)
    {sid, state} = new_sid(state)

    state =
      add_stage(
        state,
        {:reduce, sid, spv, input_refs, {outer, rsize, inner_stride}, n_out, ebytes(node.type)}
      )

    ref = {:stage, sid}
    {ref, memoize(state, node, ref)}
  end

  defp plan_new(%T{data: %Expr{op: op}} = node, state) do
    unless fusable_elementwise?(node), do: throw(:unschedulable)
    # A maximal fusable region; its leaves (params + non-fusable / materialised /
    # hoisted nodes) become this stage's inputs. Materialise each non-param leaf
    # as an earlier stage.
    _ = op
    leaves = region_leaves(node, state.memo, state.shared)

    {inputs, input_refs, state} =
      Enum.reduce(leaves, {[], [], state}, fn leaf, {ins, refs, st} ->
        {key, lref, st} = plan_leaf(leaf, st)
        {ins ++ [{key, leaf.shape}], refs ++ [lref], st}
      end)

    {glsl, _} = Codegen.emit_region(node, inputs)
    spv = compile!(glsl)
    {sid, state} = new_sid(state)
    state = add_stage(state, {:fused, sid, spv, input_refs, Nx.size(node), ebytes(node.type)})
    ref = {:stage, sid}
    {ref, memoize(state, node, ref)}
  end

  # A region leaf: a parameter binds directly; anything else is materialised as
  # its own stage and referenced by the region via its node id.
  defp plan_leaf(%T{data: %Expr{op: :parameter, args: [pidx]}}, state),
    do: {{:param, pidx}, {:param, pidx}, state}

  defp plan_leaf(%T{data: %Expr{id: id}} = leaf, state) do
    {lref, state} = plan_node(leaf, state)
    {{:stage, id}, lref, state}
  end

  # The leaf inputs of a fusable region rooted at `root`, in first-encounter
  # order, de-duped; constants inline (skipped). A descendant is a leaf iff it is
  # a `stage_leaf?` — a parameter/boundary, an already-materialised node (`memo`),
  # or a node hoisted for cross-stage CSE (`shared`). `root` itself is always
  # expanded (it is the region being built), so it is never treated as its own
  # leaf. This is where cross-stage CSE takes effect: a subexpression shared
  # across a stage boundary is read from its buffer here instead of re-inlined.
  defp region_leaves(root, memo, shared) do
    root |> expand_children(memo, shared, [], MapSet.new()) |> elem(0)
  end

  defp expand_children(%T{data: %Expr{args: args}}, memo, shared, acc, seen) do
    Enum.reduce(args, {acc, seen}, fn
      %T{data: %Expr{}} = child, {a, s} -> collect_leaf(child, memo, shared, a, s)
      _, as -> as
    end)
  end

  defp collect_leaf(%T{data: %Expr{op: :constant}}, _memo, _shared, acc, seen), do: {acc, seen}

  defp collect_leaf(%T{data: %Expr{id: id}} = node, memo, shared, acc, seen) do
    cond do
      MapSet.member?(seen, id) -> {acc, seen}
      stage_leaf?(node, memo, shared) -> {acc ++ [node], MapSet.put(seen, id)}
      true -> expand_children(node, memo, shared, acc, MapSet.put(seen, id))
    end
  end

  # A node is a stage boundary (materialised as its own buffer input to a region)
  # if it is non-fusable (param / dot / conv / reduce), already planned (memo), or
  # hoisted for cross-stage CSE (shared: referenced across a stage boundary).
  defp stage_leaf?(%T{data: %Expr{id: id}} = node, memo, shared) do
    not fusable_elementwise?(node) or Map.has_key?(memo, id) or MapSet.member?(shared, id)
  end

  defp fusable_elementwise?(%T{data: %Expr{op: op}, type: t}),
    do: supported_type?(t) and Codegen.fusable_op?(op, t)

  defp fusable_elementwise?(_), do: false

  # f32 and f64 are the two dtypes the generated shaders cover; everything else
  # (integers, f16, u8 masks) stays on the Evaluator. f64 additionally requires
  # the device to advertise shaderFloat64 — without it, pipeline creation for an
  # f64 kernel fails at DISPATCH time (glslangValidator compiles fp64 GLSL
  # happily regardless), which would crash rather than degrade. Gating here
  # keeps the fallback whole-graph and correct.
  defp float_tensor?(%T{type: type}), do: supported_type?(type)
  defp float_tensor?(_), do: false

  defp supported_type?({:f, 32}), do: true
  defp supported_type?({:f, 64}), do: Nx.Vulkan.Device.f64?()
  defp supported_type?(_), do: false

  defp has_boundary?(%T{data: %Expr{op: op}}) when op in @boundary_ops, do: true

  defp has_boundary?(%T{data: %Expr{args: args}}) do
    Enum.any?(args, fn
      %T{data: %Expr{}} = child -> has_boundary?(child)
      _ -> false
    end)
  end

  defp has_boundary?(_), do: false

  # A conv is schedulable as a stage under the same envelope the eager backend's
  # GPU conv path covers: f32 in/kernel/out, spatial rank 1..3, no feature/batch
  # grouping, identity permutations. Anything else throws :unschedulable and the
  # whole graph falls back to the Evaluator (still correct).
  defp conv_schedulable!(
         %T{type: t},
         %T{type: it, shape: ishape},
         %T{type: kt, shape: kshape},
         opts
       ) do
    rank = tuple_size(ishape)
    sr = rank - 2

    ok =
      supported_type?(t) and it == t and kt == t and
        sr >= 1 and sr <= 3 and
        Keyword.get(opts, :feature_group_size, 1) == 1 and
        Keyword.get(opts, :batch_group_size, 1) == 1 and
        identity_perm?(opts[:input_permutation], rank) and
        identity_perm?(opts[:kernel_permutation], tuple_size(kshape)) and
        identity_perm?(opts[:output_permutation], rank)

    unless ok, do: throw(:unschedulable)
  end

  defp identity_perm?(nil, _rank), do: true
  defp identity_perm?(perm, rank), do: perm == Enum.to_list(0..(rank - 1)//1)

  # A plain 2-D `a @ b` with all three tensors the same float dtype. Mixed
  # precision would need a cast stage we don't emit, so it falls back.
  defp dot_2d!(%T{type: t}, %T{shape: as, type: at} = _a, ca, ba, %T{shape: bs, type: bt}, cb, bb) do
    ok =
      supported_type?(t) and at == t and bt == t and
        tuple_size(as) == 2 and tuple_size(bs) == 2 and
        ca == [1] and cb == [0] and ba == [] and bb == []

    unless ok, do: throw(:unschedulable)
  end

  defp compile!(glsl) do
    case Codegen.compile_cached(glsl) do
      {:ok, spv} -> spv
      {:error, _} -> throw(:unschedulable)
    end
  end

  defp memoize(state, %T{data: %Expr{id: id}}, ref),
    do: %{state | memo: Map.put(state.memo, id, ref)}

  defp add_stage(state, instr), do: %{state | stages: [instr | state.stages]}

  defp new_sid(state), do: {state.counter, %{state | counter: state.counter + 1}}

  # ---- multi-stage runtime executor ------------------------------------

  defp run_plan(stages, out_sid, template, params) do
    values = Enum.reduce(stages, %{}, &exec_stage(&1, &2, params))
    out_ref = Map.fetch!(values, {:stage, out_sid})
    %{template | data: %VulkanoBackend{ref: out_ref, shape: template.shape, type: template.type}}
  end

  # Multi-output: run the shared schedule once, then rebuild the output container
  # by walking its template in the same leaf order the plan collected `out_refs`,
  # binding each leaf to its resolved buffer (a stage output, or a passthrough
  # param). A ref shared by several outputs (memoised) aliases the same buffer.
  defp run_plan_multi(stages, out_refs, template, params) do
    values = Enum.reduce(stages, %{}, &exec_stage(&1, &2, params))

    {result, {[], _}} =
      Composite.traverse(template, {out_refs, values}, fn leaf, {[ref | rest], vals} ->
        {buf, vals} = resolve(ref, vals, params)
        data = %VulkanoBackend{ref: buf, shape: leaf.shape, type: leaf.type}
        {%{leaf | data: data}, {rest, vals}}
      end)

    result
  end

  defp exec_stage({:dot, sid, a_ref, b_ref, m, n, k, eb, spv}, values, params) do
    {a, values} = resolve(a_ref, values, params)
    {b, values} = resolve(b_ref, values, params)
    {:ok, out} = NativeV.buf_alloc(m * n * eb)
    :ok = NativeV.matmul(out, a, b, m, n, k, spv)
    Map.put(values, {:stage, sid}, out)
  end

  # transpose stage: 2-D axis permutation into a fresh contiguous buffer. `m`/`n`
  # are the INPUT dims (the shader reads a[m][n], writes out[n][m]), matching
  # `VulkanoBackend.transpose/3`.
  defp exec_stage({:transpose, sid, in_ref, m, n, eb, spv}, values, params) do
    {a, values} = resolve(in_ref, values, params)
    {:ok, out} = NativeV.buf_alloc(m * n * eb)
    :ok = NativeV.transpose_2d(out, a, m, n, spv)
    Map.put(values, {:stage, sid}, out)
  end

  defp exec_stage({:fused, sid, spv, input_refs, n_elems, eb}, values, params) do
    {in_refs, values} = Enum.map_reduce(input_refs, values, &resolve(&1, &2, params))
    {:ok, out} = NativeV.buf_alloc(n_elems * eb)
    :ok = NativeV.dispatch_generated(out, in_refs, n_elems, spv)
    Map.put(values, {:stage, sid}, out)
  end

  # conv stage: im2col unfold + GEMM, reusing the eager backend's shaders and the
  # precomputed geometry (VulkanoBackend.conv_plan). Input/kernel may be earlier
  # stages' GPU buffers, so `conv(relu(x), k)` fuses the relu as an input stage.
  defp exec_stage({:conv, sid, in_ref, k_ref, p}, values, params) do
    {in_buf, values} = resolve(in_ref, values, params)
    {k_buf, values} = resolve(k_ref, values, params)
    {:ok, params_ref} = NativeV.buf_upload(p.params_bin)
    {:ok, col_ref} = NativeV.buf_alloc(p.m * p.k_cols * p.ebytes)

    :ok =
      NativeV.conv_im2col(
        col_ref,
        in_buf,
        params_ref,
        p.n,
        p.cin,
        p.o_total,
        p.k_total,
        p.k_cols,
        p.im2col_spv
      )

    {:ok, out} = NativeV.buf_alloc(p.n * p.cout * p.o_total * p.ebytes)
    :ok = NativeV.conv_gemm(out, col_ref, k_buf, p.n, p.cout, p.o_total, p.k_cols, p.gemm_spv)
    Map.put(values, {:stage, sid}, out)
  end

  # reduce stage: parallel workgroup-per-slot tree reduce (dispatch_generated_reduce)
  # over the fusable inner chain, writing outer*inner slots. Inputs may be earlier
  # stages' buffers.
  defp exec_stage(
         {:reduce, sid, spv, input_refs, {outer, rsize, inner}, n_out, eb},
         values,
         params
       ) do
    {in_refs, values} = Enum.map_reduce(input_refs, values, &resolve(&1, &2, params))
    {:ok, out} = NativeV.buf_alloc(n_out * eb)
    :ok = NativeV.dispatch_generated_reduce(out, in_refs, outer, rsize, inner, spv)
    Map.put(values, {:stage, sid}, out)
  end

  # Resolve an input ref to a device buffer; param buffers are transferred once
  # and cached (a param feeding several stages uploads only once).
  defp resolve({:stage, sid}, values, _params), do: {Map.fetch!(values, {:stage, sid}), values}

  defp resolve({:param, pidx} = key, values, params) do
    case Map.get(values, key) do
      nil ->
        ref = param_ref(params, pidx)
        {ref, Map.put(values, key, ref)}

      ref ->
        {ref, values}
    end
  end

  # Device buffer for runtime argument `pidx`. Defn passes each argument as a
  # zero-arg thunk (see Evaluator).
  #
  # An argument that is ALREADY GPU-resident binds its existing buffer. This
  # matters a lot: `VulkanoBackend.backend_transfer/3` delegates to
  # `backend_copy/3`, which round-trips through host memory (`to_binary` then
  # `from_binary`) even when the target backend is VulkanoBackend itself. Calling
  # it on an already-on-device param would cost a full download + upload per
  # param PER INVOCATION — for a 512x512 f64 weight, 2 MB each way — which
  # swamps the kernel being fed and made fused graphs measurably slower than the
  # eager path they replace.
  defp param_ref(params, pidx) do
    tensor = params |> Enum.at(pidx) |> then(& &1.()) |> Nx.devectorize()

    case tensor.data do
      %VulkanoBackend{ref: ref} ->
        ref

      _ ->
        %T{data: %VulkanoBackend{ref: ref}} = Nx.backend_transfer(tensor, VulkanoBackend)
        ref
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
    in_refs = Enum.map(param_order, &param_ref(params, &1))

    n = Nx.size(template)
    {:ok, out_ref} = NativeV.buf_alloc(n * ebytes(template.type))
    :ok = NativeV.dispatch_generated(out_ref, in_refs, n, spv_path)

    data = %VulkanoBackend{ref: out_ref, shape: template.shape, type: template.type}
    %{template | data: data}
  end

  defp run_fused_reduce(spv_path, param_order, {outer, rsize, inner}, template, params) do
    in_refs = Enum.map(param_order, &param_ref(params, &1))

    n_out = max(Nx.size(template), 1)
    {:ok, out_ref} = NativeV.buf_alloc(n_out * ebytes(template.type))
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
