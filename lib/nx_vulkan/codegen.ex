defmodule Nx.Vulkan.Codegen do
  @moduledoc """
  JIT GLSL codegen from `Nx.Defn.Expr` trees — the heart of the thrust-3
  fusion compiler.

  `Nx.Vulkan.Compiler` traces a `defn` to an expression tree; this module
  turns a fully-fusable elementwise subtree into a single GLSL compute shader
  (one thread per element, the whole chain inlined) and compiles it to SPIR-V,
  cached by source hash. One generated shader + one `dispatch_generated` call
  replaces the N per-op dispatches the eager backend would issue.

  Scope today (increment 1): same-shape, f32 elementwise chains over unary and
  binary arithmetic ops plus scalar constants. Comparisons (which change dtype
  to u8), broadcasting between differing tensor shapes, reductions and library
  ops (dot/conv) are not fused yet — `Nx.Vulkan.Compiler` falls back to the
  Evaluator for those, so behaviour stays correct.

  Ported from the dropped `577baf9`/`9a9e3ad` codegen, retargeted to Nx 0.13 and
  the current `Nx.Vulkan.NativeV` dispatch primitives.
  """

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  # op -> GLSL template, `r` is the (parenthesised) operand.
  @unary_ops %{
    exp: "exp(r)",
    log: "log(r)",
    sqrt: "sqrt(r)",
    rsqrt: "(1.0 / sqrt(r))",
    abs: "abs(r)",
    negate: "(-r)",
    sigmoid: "(1.0 / (1.0 + exp(-r)))",
    tanh: "tanh(r)",
    ceil: "ceil(r)",
    floor: "floor(r)",
    round: "round(r)",
    sign: "sign(r)",
    reciprocal: "(1.0 / r)",
    square: "(r * r)",
    erf: "erf_approx(r)",
    expm1: "expm1_approx(r)"
  }

  # op -> {kind, glsl}. `:infix` emits `(l op r)`; `:call` emits `f(l, r)`.
  @binary_ops %{
    add: {:infix, "+"},
    subtract: {:infix, "-"},
    multiply: {:infix, "*"},
    divide: {:infix, "/"},
    pow: {:call, "pow"},
    max: {:call, "max"},
    min: {:call, "min"}
  }

  @doc "Set of ops that can be fused into one elementwise shader (excludes params/constants)."
  def fusable_op?(op), do: is_map_key(@unary_ops, op) or is_map_key(@binary_ops, op)

  @doc """
  True when the whole expression tree is a same-shape f32 elementwise chain
  (unary/binary ops, parameters, scalar constants) that `emit_elementwise/1`
  can compile into a single shader.
  """
  def fusable?(%T{type: {:f, 32}, shape: out_shape} = expr) do
    fusable_node?(expr, out_shape)
  end

  def fusable?(_), do: false

  # In a valid elementwise tree every node's shape broadcasts to the root shape
  # (NumPy rules), so nodes may be smaller than `out_shape` — a param `{n}` added
  # to `{m, n}`, a scalar-tensor scale, etc. The codegen loads each param at its
  # broadcast-mapped index; interior nodes are unchanged (see `emit_loads`).
  defp fusable_node?(%T{data: %Expr{op: :parameter}, type: {:f, 32}, shape: s}, out_shape),
    do: broadcasts_to?(s, out_shape)

  defp fusable_node?(%T{data: %Expr{op: :constant, args: [c]}, shape: {}}, _out_shape),
    do: is_number(c)

  defp fusable_node?(%T{data: %Expr{op: op, args: [a]}, type: {:f, 32}, shape: s}, out_shape)
       when is_map_key(@unary_ops, op),
       do: broadcasts_to?(s, out_shape) and fusable_node?(a, out_shape)

  defp fusable_node?(%T{data: %Expr{op: op, args: [a, b]}, type: {:f, 32}, shape: s}, out_shape)
       when is_map_key(@binary_ops, op),
       do: broadcasts_to?(s, out_shape) and operand_fusable?(a, out_shape) and operand_fusable?(b, out_shape)

  defp fusable_node?(_, _), do: false

  # Binary operands may be scalar constants (shape {}) even when the op output
  # is a full tensor; those are fine (emitted as literals).
  defp operand_fusable?(%T{data: %Expr{op: :constant, args: [c]}, shape: {}}, _out),
    do: is_number(c)

  defp operand_fusable?(node, out_shape), do: fusable_node?(node, out_shape)

  # NumPy broadcast: `s` right-aligns to `o`, each dim 1 or equal.
  defp broadcasts_to?(s, o) do
    sl = Tuple.to_list(s)
    ol = Tuple.to_list(o)

    length(sl) <= length(ol) and
      (List.duplicate(1, length(ol) - length(sl)) ++ sl)
      |> Enum.zip(ol)
      |> Enum.all?(fn {sd, od} -> sd == 1 or sd == od end)
  end

  @doc """
  Emit a GLSL compute shader for a fusable elementwise expression.

  Returns `{glsl, %{param_order: [param_index, ...], n_inputs: k}}` where
  `param_order[b]` is the runtime argument index bound to input binding `b`.
  """
  def emit_elementwise(%T{} = expr) do
    inputs = param_inputs(expr)
    {glsl, _meta} = emit_region(expr, inputs)
    {glsl, %{param_order: Enum.map(inputs, fn {{:param, pidx}, _} -> pidx end), n_inputs: length(inputs)}}
  end

  @doc """
  Emit an elementwise shader for a fusion region whose leaf inputs are given by
  `inputs` — an ordered list of `{{:param, pidx} | {:stage, node_id}, shape}`.
  Parameters and stage-materialised buffers are both loaded from input bindings
  (broadcast-aware); constants inline. Used by both the single-region compile
  and the multi-stage split (where a leaf may be a prior stage's output buffer).
  Returns `{glsl, %{n_inputs: k}}`.
  """
  def emit_region(%T{shape: out_shape} = root, inputs) do
    ctx = build_ctx(inputs)
    {temp_lines, root_ref} = emit_dag(root, ctx)
    k = length(inputs)
    loads = emit_loads(inputs, out_shape, "i", "    ")
    temps = Enum.map_join(temp_lines, "\n", &("    " <> &1))

    glsl = """
    #version 450

    layout(local_size_x = 256) in;

    layout(push_constant) uniform Push { uint n; } pc;

    #{input_decls(k)}
    layout(std430, binding = #{k}) writeonly buffer Out { float out_buf[]; };

    #{helper_functions()}

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= pc.n) return;
    #{loads}
    #{temps}
        out_buf[i] = #{root_ref};
    }
    """

    {glsl, %{n_inputs: k}}
  end

  # All distinct parameters of `expr` as region inputs, in ascending index order.
  defp param_inputs(expr) do
    shapes = collect_params(expr, %{})

    shapes
    |> Map.keys()
    |> Enum.sort()
    |> Enum.map(fn pidx -> {{:param, pidx}, Map.fetch!(shapes, pidx)} end)
  end

  # Reduce ops we fuse: op -> {glsl accumulator init, combine}. `sum`/`product`
  # accumulate in f64 for precision; `mean` fuses as `sum` with a post-scale.
  @reduce_ops [:sum, :product, :reduce_max, :reduce_min]

  @doc "True if `op` is a reduction this module can fuse an elementwise inner into."
  def reduce_op?(op), do: op in @reduce_ops

  @wg_size 256

  @doc """
  Emit a GLSL shader that fuses an elementwise `inner` chain into a reduction
  over the (outer, reduce_size, inner_stride) view, using a **parallel
  workgroup-per-slot shared-memory tree reduce**: each output slot gets one
  workgroup of #{@wg_size} threads that stride the reduce axis, accumulate into
  a shared array, then tree-reduce to a single value. `sum` accumulates in f64
  to match BinaryBackend.

  This is #{@wg_size}x more parallel than a serial per-slot loop and beats even
  the eager path (whose `reduce_axis` is itself one-thread-per-slot). It is only
  valid when the number of slots fits the one-dimensional workgroup-count limit
  (`maxComputeWorkGroupCount[0]`, typically 65535) — the caller gates on that.
  Dispatch **`outer*inner` workgroups** (one per slot), NOT `ceil(slots/256)`.

  `scale` (a number or nil) applies a final `/ scale` to each output slot — this
  is how `mean` fuses: `divide(sum(...), n)` becomes a fused sum scaled by `1/n`.

  Returns `{glsl, %{param_order: [...], n_inputs: k}}`.
  """
  def emit_fused_reduce(%T{} = inner, reduce_op, scale \\ nil)
      when reduce_op in @reduce_ops do
    inputs = param_inputs(inner)
    {glsl, meta} = emit_reduce_region(inner, reduce_op, inputs, scale)
    param_order = Enum.map(inputs, fn {{:param, pidx}, _} -> pidx end)
    {glsl, Map.put(meta, :param_order, param_order)}
  end

  @doc """
  Like `emit_fused_reduce/3` but for a reduction region whose leaf inputs are
  given explicitly as `inputs` (`{{:param, pidx} | {:stage, node_id}, shape}`) —
  so the reduced `inner` chain may read earlier stages' output buffers, not just
  parameters. Used by the multi-stage split to materialise a reduce as a stage
  (e.g. `mean(x)` in an `x - mean(x)` layernorm graph). Returns `{glsl, %{n_inputs: k}}`.
  """
  def emit_reduce_region(%T{shape: in_shape} = inner, reduce_op, inputs, scale \\ nil)
      when reduce_op in @reduce_ops do
    ctx = build_ctx(inputs)
    {temp_lines, root} = emit_dag(inner, ctx)
    k = length(inputs)

    decls = input_decls(k)

    # load each input at the running reduce index `idx` (broadcast-aware: the
    # pre-reduction shape `in_shape` is the coordinate space)
    loads = emit_loads(inputs, in_shape, "idx", "            ")

    temps = Enum.map_join(temp_lines, "\n", &("            " <> &1))

    %{acc_type: acc_type, init: init, shared_init: shared_init, accumulate: accumulate,
      combine: combine, store: base_store} = reduce_kind(reduce_op, root)

    store = if scale, do: "(#{base_store}) / #{glsl_float(scale)}", else: base_store

    glsl = """
    #version 450
    #extension GL_ARB_gpu_shader_fp64 : require

    layout(local_size_x = #{@wg_size}) in;

    layout(push_constant) uniform Push {
        uint outer;
        uint reduce_size;
        uint inner;
        uint op;
    } pc;

    #{decls}
    layout(std430, binding = #{k}) writeonly buffer Out { float out_buf[]; };

    shared #{acc_type} sdata[#{@wg_size}];

    #{helper_functions()}

    void main() {
        uint tid = gl_LocalInvocationID.x;
        uint slots = pc.outer * pc.inner;
        // Grid-stride over output slots so one launch handles any slot count,
        // not just <= maxComputeWorkGroupCount[0]. Each workgroup fully reduces
        // one slot (coalesced: its 256 threads stride consecutive elements),
        // then moves to the next slot gl_NumWorkGroups.x away.
        for (uint slot = gl_WorkGroupID.x; slot < slots; slot += gl_NumWorkGroups.x) {
            uint outr = slot / pc.inner;
            uint inr = slot % pc.inner;
            uint base = outr * pc.reduce_size * pc.inner + inr;

            #{acc_type} acc = #{init};
            for (uint r = tid; r < pc.reduce_size; r += #{@wg_size}u) {
                uint idx = base + r * pc.inner;
    #{loads}
    #{temps}
                #{accumulate}
            }
            sdata[tid] = acc;
            barrier();

            for (uint s = #{div(@wg_size, 2)}u; s > 0u; s >>= 1u) {
                if (tid < s) sdata[tid] = #{combine};
                barrier();
            }

            if (tid == 0u) out_buf[slot] = #{store};
            barrier();
        }
    }
    """

    _ = shared_init
    {glsl, %{n_inputs: k}}
  end

  # Per-op GLSL fragments for the parallel tree reduce. `root` is the GLSL ref to
  # the fused elementwise value at the current reduce index (a temp / load).
  defp reduce_kind(:sum, root) do
    %{
      acc_type: "double",
      init: "0.0lf",
      shared_init: "0.0lf",
      accumulate: "acc += double(#{root});",
      combine: "sdata[tid] + sdata[tid + s]",
      store: "float(sdata[0])"
    }
  end

  defp reduce_kind(:product, root) do
    %{
      acc_type: "double",
      init: "1.0lf",
      shared_init: "1.0lf",
      accumulate: "acc *= double(#{root});",
      combine: "sdata[tid] * sdata[tid + s]",
      store: "float(sdata[0])"
    }
  end

  defp reduce_kind(:reduce_max, root), do: minmax_kind(root, "max", "-1.0/0.0")
  defp reduce_kind(:reduce_min, root), do: minmax_kind(root, "min", "1.0/0.0")

  defp minmax_kind(root, fun, init) do
    %{
      acc_type: "float",
      init: init,
      shared_init: init,
      accumulate: "acc = #{fun}(acc, #{root});",
      combine: "#{fun}(sdata[tid], sdata[tid + s])",
      store: "sdata[0]"
    }
  end

  # ---- expression DAG -> GLSL (with CSE) --------------------------------

  # Linearise the elementwise DAG into SSA-style temporaries so a node used by
  # several parents (fan-out) is computed ONCE, not re-inlined at every use —
  # naive inlining is exponential for deep DAGs (e.g. 8 chained squarings ->
  # 255 multiplies vs 8 with CSE). Returns `{temp_lines, root_ref}`: `temp_lines`
  # are `float tN = <expr>;` in dependency order (each node in terms of earlier
  # temps / param loads / literals), and `root_ref` is the GLSL reference to the
  # whole expression's value (a temp, a param load `vB`, or a constant literal).
  # `ctx` = %{params: %{pidx => binding}, stages: %{node_id => binding}}. A leaf
  # input is a parameter OR a node materialised by an earlier stage (multi-stage
  # split); both load from a buffer `vB`. Constants inline as literals.
  defp emit_dag(%T{} = expr, ctx) do
    order = expr |> topo_order(ctx, [], MapSet.new()) |> elem(0) |> Enum.reverse()
    temp = order |> Enum.with_index() |> Map.new(fn {n, i} -> {n.data.id, i} end)

    lines =
      Enum.map(order, fn n ->
        "float t#{Map.fetch!(temp, n.data.id)} = #{node_expr(n, ctx, temp)};"
      end)

    {lines, ref(expr, ctx, temp)}
  end

  # A node is a leaf input (no temp, stop recursion) if it's a parameter, a
  # constant, or the output of an earlier stage bound to an input buffer.
  defp leaf?(%T{data: %Expr{op: op}}, _ctx) when op in [:parameter, :constant], do: true
  defp leaf?(%T{data: %Expr{id: id}}, ctx), do: Map.has_key?(ctx.stages, id)

  # Post-order DFS collecting interior nodes de-duped by id; each node is placed
  # AFTER its children, so reversing the accumulator gives dependency order.
  defp topo_order(node, ctx, acc, seen) do
    cond do
      leaf?(node, ctx) ->
        {acc, seen}

      MapSet.member?(seen, node.data.id) ->
        {acc, seen}

      true ->
        {acc, seen} =
          Enum.reduce(node.data.args, {acc, MapSet.put(seen, node.data.id)}, fn
            %T{data: %Expr{}} = child, {a, s} -> topo_order(child, ctx, a, s)
            _, as -> as
          end)

        {[node | acc], seen}
    end
  end

  # GLSL for one node in terms of its children's refs (no recursion into them).
  defp node_expr(%T{data: %Expr{op: op, args: [a]}}, ctx, temp)
       when is_map_key(@unary_ops, op) do
    # Replace only the standalone `r` operand token — a plain "r" replace would
    # also clobber the `r` inside op names like sqrt/round/reciprocal/erf.
    String.replace(Map.fetch!(@unary_ops, op), ~r/\br\b/, "(#{ref(a, ctx, temp)})")
  end

  defp node_expr(%T{data: %Expr{op: op, args: [a, b]}}, ctx, temp)
       when is_map_key(@binary_ops, op) do
    l = ref(a, ctx, temp)
    r = ref(b, ctx, temp)

    case Map.fetch!(@binary_ops, op) do
      {:infix, sym} -> "(#{l} #{sym} #{r})"
      {:call, f} -> "#{f}(#{l}, #{r})"
    end
  end

  # Reference to a node's already-computed value.
  defp ref(%T{data: %Expr{op: :constant, args: [c]}}, _ctx, _temp), do: glsl_float(c)

  defp ref(%T{data: %Expr{op: :parameter, args: [pidx]}}, ctx, _temp),
    do: "v#{Map.fetch!(ctx.params, pidx)}"

  defp ref(%T{data: %Expr{id: id}}, ctx, temp) do
    case Map.get(ctx.stages, id) do
      nil -> "t#{Map.fetch!(temp, id)}"
      b -> "v#{b}"
    end
  end

  # Build a `ctx` from an ordered input list of `{key, _shape}` where key is
  # `{:param, pidx}` or `{:stage, node_id}`; binding = position in the list.
  defp build_ctx(inputs) do
    {params, stages, _} =
      Enum.reduce(inputs, {%{}, %{}, 0}, fn {key, _shape}, {p, s, b} ->
        case key do
          {:param, pidx} -> {Map.put(p, pidx, b), s, b + 1}
          {:stage, id} -> {p, Map.put(s, id, b), b + 1}
        end
      end)

    %{params: params, stages: stages}
  end

  # `layout(...) buffer InB { float bufB[]; };` for each of `k` input bindings.
  defp input_decls(k) do
    0..(k - 1)
    |> Enum.map_join("\n", fn b ->
      "layout(std430, binding = #{b}) readonly buffer In#{b} { float buf#{b}[]; };"
    end)
  end

  defp glsl_float(c) do
    s = to_string(c / 1.0)
    # GLSL wants a decimal point; "1.0e10"/"2.0" from Float already have one.
    if String.contains?(s, [".", "e", "E"]), do: s, else: s <> ".0"
  end

  # Map each distinct parameter index -> its shape (for broadcast-aware loads).
  defp collect_params(%T{data: %Expr{op: :parameter, args: [pidx]}, shape: shape}, acc),
    do: Map.put(acc, pidx, shape)

  defp collect_params(%T{data: %Expr{args: args}}, acc) do
    Enum.reduce(args, acc, fn
      %T{data: %Expr{}} = child, a -> collect_params(child, a)
      _, a -> a
    end)
  end

  defp collect_params(_, acc), do: acc

  # ---- broadcast-aware parameter loads ---------------------------------

  # Emit `float vB = bufB[<index>];` for each input binding (in `inputs` order).
  # Inputs whose shape equals `out_shape` load at the flat index `ivar`
  # directly; broadcast inputs (scalar, row/col vector, any dim == 1) load at
  # their NumPy-broadcast source index, computed from `ivar` with the
  # (compile-time-constant) shapes baked in.
  defp emit_loads(inputs, out_shape, ivar, indent) do
    inputs
    |> Enum.with_index()
    |> Enum.map(fn {{_key, shape}, b} ->
      idx = if shape == out_shape, do: ivar, else: broadcast_index(shape, out_shape, ivar)
      "#{indent}float v#{b} = buf#{b}[#{idx}];"
    end)
    |> Enum.join("\n")
  end

  # GLSL uint expression for the flat source index of a broadcast input of shape
  # `s` when the output flat index is `ivar` and the output shape is `o`. Only
  # dims where `s` is not 1 contribute (row-major suffix strides, baked in).
  defp broadcast_index(s, o, ivar) do
    ol = Tuple.to_list(o)
    r = length(ol)
    sl = Tuple.to_list(s)
    aligned = List.duplicate(1, r - length(sl)) ++ sl

    terms =
      for d <- 0..(r - 1), Enum.at(aligned, d) != 1 do
        out_suffix = ol |> Enum.drop(d + 1) |> Enum.product()
        in_suffix = aligned |> Enum.drop(d + 1) |> Enum.product()
        coord = "((#{ivar} / #{out_suffix}u) % #{Enum.at(ol, d)}u)"
        if in_suffix == 1, do: coord, else: "#{coord} * #{in_suffix}u"
      end

    case terms do
      [] -> "0u"
      _ -> "(" <> Enum.join(terms, " + ") <> ")"
    end
  end

  # ---- SPIR-V compile + cache ------------------------------------------

  @doc """
  Compile GLSL to a cached `.spv`, keyed by source hash. Returns
  `{:ok, spv_path}` or `{:error, reason}`. Reuses an existing `.spv` on hit,
  so a given fused kernel is compiled by glslangValidator exactly once.
  """
  def compile_cached(glsl) do
    hash = :erlang.phash2(glsl, 0xFFFFFFFF)
    cache_dir = Path.join(:code.priv_dir(:nx_vulkan), "shader_cache")
    File.mkdir_p!(cache_dir)
    spv_path = Path.join(cache_dir, "gen_#{Integer.to_string(hash, 16)}.spv")

    if File.exists?(spv_path) do
      {:ok, spv_path}
    else
      comp_path = spv_path <> ".comp"
      File.write!(comp_path, glsl)

      try do
        case System.cmd("glslangValidator", ["-V", comp_path, "-o", spv_path],
               stderr_to_stdout: true
             ) do
          {_, 0} -> {:ok, spv_path}
          {out, _} -> {:error, out}
        end
      after
        File.rm(comp_path)
      end
    end
  end

  defp helper_functions do
    """
    float erf_approx(float x) {
        float a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
        float a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
        float s = sign(x), ax = abs(x);
        float t = 1.0 / (1.0 + p * ax);
        float y = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * exp(-ax*ax);
        return s * y;
    }

    float expm1_approx(float x) {
        if (abs(x) < 0.5) {
            float x2 = x * x;
            return x + x2*0.5 + x2*x*(1.0/6.0) + x2*x2*(1.0/24.0) + x2*x2*x*(1.0/120.0);
        }
        return exp(x) - 1.0;
    }
    """
  end
end
