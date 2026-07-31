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

  defp fusable_node?(%T{data: %Expr{op: :parameter}, type: {:f, 32}, shape: s}, out_shape),
    do: s == out_shape

  defp fusable_node?(%T{data: %Expr{op: :constant, args: [c]}, shape: {}}, _out_shape),
    do: is_number(c)

  defp fusable_node?(%T{data: %Expr{op: op, args: [a]}, type: {:f, 32}, shape: s}, out_shape)
       when is_map_key(@unary_ops, op),
       do: s == out_shape and fusable_node?(a, out_shape)

  defp fusable_node?(%T{data: %Expr{op: op, args: [a, b]}, type: {:f, 32}, shape: s}, out_shape)
       when is_map_key(@binary_ops, op),
       do: s == out_shape and operand_fusable?(a, out_shape) and operand_fusable?(b, out_shape)

  defp fusable_node?(_, _), do: false

  # Binary operands may be scalar constants (shape {}) even when the op output
  # is a full tensor; those are fine (emitted as literals).
  defp operand_fusable?(%T{data: %Expr{op: :constant, args: [c]}, shape: {}}, _out),
    do: is_number(c)

  defp operand_fusable?(node, out_shape), do: fusable_node?(node, out_shape)

  @doc """
  Emit a GLSL compute shader for a fusable elementwise expression.

  Returns `{glsl, %{param_order: [param_index, ...], n_inputs: k}}` where
  `param_order[b]` is the runtime argument index bound to input binding `b`.
  """
  def emit_elementwise(%T{} = expr) do
    param_order = expr |> collect_param_indices(MapSet.new()) |> Enum.sort()
    binding_of = param_order |> Enum.with_index() |> Map.new()
    body = emit_expr(expr, binding_of)
    k = length(param_order)

    decls =
      param_order
      |> Enum.with_index()
      |> Enum.map(fn {_pidx, b} ->
        "layout(std430, binding = #{b}) readonly buffer In#{b} { float buf#{b}[]; };"
      end)
      |> Enum.join("\n")

    loads =
      param_order
      |> Enum.with_index()
      |> Enum.map(fn {_pidx, b} -> "    float v#{b} = buf#{b}[i];" end)
      |> Enum.join("\n")

    glsl = """
    #version 450

    layout(local_size_x = 256) in;

    layout(push_constant) uniform Push { uint n; } pc;

    #{decls}
    layout(std430, binding = #{k}) writeonly buffer Out { float out_buf[]; };

    #{helper_functions()}

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= pc.n) return;
    #{loads}
        out_buf[i] = #{body};
    }
    """

    {glsl, %{param_order: param_order, n_inputs: k}}
  end

  # ---- expression -> GLSL string ---------------------------------------

  defp emit_expr(%T{data: %Expr{op: :parameter}} = t, binding_of) do
    %T{data: %Expr{args: [pidx]}} = t
    "v#{Map.fetch!(binding_of, pidx)}"
  end

  defp emit_expr(%T{data: %Expr{op: :constant, args: [c]}}, _binding_of) do
    glsl_float(c)
  end

  defp emit_expr(%T{data: %Expr{op: op, args: [a]}}, binding_of)
       when is_map_key(@unary_ops, op) do
    inner = emit_expr(a, binding_of)
    # Replace only the standalone `r` operand token — a plain "r" replace would
    # also clobber the `r` inside op names like sqrt/round/reciprocal/erf.
    String.replace(Map.fetch!(@unary_ops, op), ~r/\br\b/, "(#{inner})")
  end

  defp emit_expr(%T{data: %Expr{op: op, args: [a, b]}}, binding_of)
       when is_map_key(@binary_ops, op) do
    l = emit_expr(a, binding_of)
    r = emit_expr(b, binding_of)

    case Map.fetch!(@binary_ops, op) do
      {:infix, sym} -> "(#{l} #{sym} #{r})"
      {:call, f} -> "#{f}(#{l}, #{r})"
    end
  end

  defp glsl_float(c) do
    s = to_string(c / 1.0)
    # GLSL wants a decimal point; "1.0e10"/"2.0" from Float already have one.
    if String.contains?(s, [".", "e", "E"]), do: s, else: s <> ".0"
  end

  defp collect_param_indices(%T{data: %Expr{op: :parameter, args: [pidx]}}, acc),
    do: MapSet.put(acc, pidx)

  defp collect_param_indices(%T{data: %Expr{args: args}}, acc) do
    Enum.reduce(args, acc, fn
      %T{data: %Expr{}} = child, a -> collect_param_indices(child, a)
      _, a -> a
    end)
  end

  defp collect_param_indices(_, acc), do: acc

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
