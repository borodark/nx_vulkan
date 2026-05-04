defmodule Nx.Vulkan.Codegen do
  @moduledoc """
  Compiles Nx.Defn expression trees into GLSL compute shaders.

  Walks the expression DAG, partitions into fusable subgraphs, and
  emits one GLSL compute shader per subgraph. Non-fusable ops (matmul,
  conv) dispatch to pre-compiled library shaders.

  ## Architecture

      Nx.Defn.Expr tree
          ↓  partition/1
      [Subgraph, ...]          (fusable groups + boundary ops)
          ↓  emit_glsl/1
      GLSL source string
          ↓  compile_spv/1
      SPIR-V binary            (via shaderc or glslangValidator)
          ↓  cache by hash
      Vulkan dispatch

  ## Usage

      # Inside Nx.Vulkan.Compiler.__compile__/4:
      expr = trace(fun, vars)
      {glsl, metadata} = Nx.Vulkan.Codegen.emit(expr, var_ids)
      spv = Nx.Vulkan.Codegen.compile_spv(glsl)
      # dispatch with spv...
  """

  alias Nx.Tensor, as: T
  alias Nx.Defn.Expr

  # ----------------------------------------------------------------
  # Op classification
  # ----------------------------------------------------------------

  @unary_ops %{
    exp: "exp(r)",
    log: "log(r)",
    sqrt: "sqrt(r)",
    abs: "abs(r)",
    negate: "(-r)",
    sigmoid: "(1.0 / (1.0 + exp(-r)))",
    tanh: "tanh(r)",
    ceil: "ceil(r)",
    floor: "floor(r)",
    sign: "sign(r)",
    rsqrt: "(1.0 / sqrt(r))",
    erf: "erf_approx(r)",
    expm1: "expm1_approx(r)"
  }

  @binary_ops %{
    add: "+",
    subtract: "-",
    multiply: "*",
    divide: "/",
    pow: "pow",
    max: "max",
    min: "min"
  }

  @compare_ops %{
    equal: "==",
    less: "<",
    greater: ">",
    less_equal: "<=",
    greater_equal: ">=",
    not_equal: "!="
  }

  # Ops that can be fused into a single shader (element-wise, same shape)
  @fusable_ops Map.keys(@unary_ops) ++ Map.keys(@binary_ops) ++ Map.keys(@compare_ops)

  # Ops that need dedicated library shaders (not generated)
  @library_ops [:dot, :conv, :window_sum, :window_max, :sort, :argsort]

  # Ops that need shared-memory reduction patterns
  @reduce_ops [:sum, :product, :reduce_max, :reduce_min, :all, :any]

  @doc """
  Analyze an expression tree and return a compilation plan.

  Returns `{:fused, glsl, bindings}` for fusable subgraphs,
  or `{:mixed, stages}` for graphs that need multiple dispatches.
  """
  @spec analyze(Expr.t(), [non_neg_integer()]) ::
          {:fused, String.t(), map()} | {:mixed, [stage()]} | :unsupported
  def analyze(%T{data: %Expr{}} = expr, var_ids) do
    dag = linearize(expr, %{})
    groups = partition(dag, var_ids)

    case groups do
      [{:elementwise, _ops}] ->
        {glsl, bindings} = emit_elementwise(expr, var_ids)
        {:fused, glsl, bindings}

      stages when is_list(stages) ->
        {:mixed, stages}
    end
  end

  @doc """
  Emit a GLSL compute shader for a fully-fusable elementwise expression.

  Returns `{glsl_source, %{bindings: [...], push_size: N}}`.
  """
  @spec emit_elementwise(Expr.t(), [non_neg_integer()]) :: {String.t(), map()}
  def emit_elementwise(%T{data: %Expr{}} = expr, _var_ids) do
    # Collect all parameter references and assign buffer bindings
    params = collect_params(expr, %{})
    bindings = params |> Map.keys() |> Enum.sort() |> Enum.with_index()

    # Generate the shader body by walking the expression tree
    {body_code, _} = emit_expr(expr, bindings)

    n_inputs = length(bindings)
    n_buffers = n_inputs + 1  # inputs + output

    glsl = """
    #version 450

    layout (local_size_x = 256) in;

    layout (push_constant) uniform Push {
        uint n;
    } pc;

    #{emit_buffer_declarations(bindings, :readonly)}
    layout (std430, binding = #{n_inputs}) writeonly buffer Output { float out_buf[]; };

    #{emit_helper_functions()}

    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i >= pc.n) return;

    #{emit_input_loads(bindings)}

        float result = #{body_code};
        out_buf[i] = result;
    }
    """

    metadata = %{
      bindings: bindings,
      n_buffers: n_buffers,
      push_size: 4,
      type: :f32
    }

    {glsl, metadata}
  end

  @doc """
  Emit a GLSL compute shader for an expression that includes a
  single-axis reduction. The reduction uses shared memory.
  """
  @spec emit_reduce(Expr.t(), [non_neg_integer()], keyword()) :: {String.t(), map()}
  def emit_reduce(%T{data: %Expr{op: reduce_op}} = expr, _var_ids, _opts) do
    pre_reduce_expr = get_pre_reduce_expr(expr)

    # Generate the element-wise pre-reduction code
    params = collect_params(pre_reduce_expr, %{})
    bindings = params |> Map.keys() |> Enum.sort() |> Enum.with_index()
    {body_code, _} = emit_expr(pre_reduce_expr, bindings)

    n_inputs = length(bindings)
    init_val = reduce_init(reduce_op)
    combine_op = reduce_combine(reduce_op)

    glsl = """
    #version 450

    layout (local_size_x = 256) in;

    layout (push_constant) uniform Push {
        uint outer;
        uint reduce_size;
        uint inner;
    } pc;

    #{emit_buffer_declarations(bindings, :readonly)}
    layout (std430, binding = #{n_inputs}) writeonly buffer Output { float out_buf[]; };

    shared float partial[256];

    #{emit_helper_functions()}

    void main() {
        uint slot = gl_GlobalInvocationID.x;
        uint n_slots = pc.outer * pc.inner;
        if (slot >= n_slots) return;

        uint o = slot / pc.inner;
        uint ii = slot % pc.inner;
        uint base = o * pc.reduce_size * pc.inner + ii;

        float acc = #{init_val};
        for (uint k = 0u; k < pc.reduce_size; ++k) {
            uint idx = base + k * pc.inner;
    #{emit_indexed_loads(bindings, "idx")}
            float val = #{body_code};
            acc = #{combine_op};
        }

        out_buf[slot] = acc;
    }
    """

    metadata = %{
      bindings: bindings,
      n_buffers: n_inputs + 1,
      push_size: 12,
      type: :f32,
      pattern: :reduce_axis
    }

    {glsl, metadata}
  end

  @doc """
  Compile a GLSL source string to SPIR-V binary.

  Uses `glslangValidator` CLI. For production, replace with shaderc
  library binding (in-process, ~10x faster).
  """
  @spec compile_spv(String.t()) :: {:ok, binary()} | {:error, String.t()}
  def compile_spv(glsl_source) do
    tmp_comp = System.tmp_dir!() |> Path.join("nx_vulkan_#{:erlang.phash2(glsl_source, 1_000_000)}.comp")
    tmp_spv = tmp_comp <> ".spv"

    File.write!(tmp_comp, glsl_source)

    case System.cmd("glslangValidator", ["-V", tmp_comp, "-o", tmp_spv], stderr_to_stdout: true) do
      {_, 0} ->
        spv = File.read!(tmp_spv)
        File.rm(tmp_comp)
        File.rm(tmp_spv)
        {:ok, spv}

      {error, _} ->
        File.rm(tmp_comp)
        {:error, error}
    end
  end

  @doc """
  Compile and cache. Returns a path to the cached .spv file.
  Uses the GLSL source hash as the cache key.
  """
  @spec compile_cached(String.t()) :: {:ok, String.t()} | {:error, String.t()}
  def compile_cached(glsl_source) do
    hash = :erlang.phash2(glsl_source, 0xFFFFFFFF)
    cache_dir = Path.join(:code.priv_dir(:nx_vulkan), "shader_cache")
    File.mkdir_p!(cache_dir)
    spv_path = Path.join(cache_dir, "gen_#{Integer.to_string(hash, 16)}.spv")

    if File.exists?(spv_path) do
      {:ok, spv_path}
    else
      case compile_spv(glsl_source) do
        {:ok, spv_binary} ->
          File.write!(spv_path, spv_binary)
          {:ok, spv_path}

        error ->
          error
      end
    end
  end

  @doc """
  Split a mixed expression tree into stages at non-fusable boundaries.

  Returns a list of `{:elementwise | :library | :reduce, expr, deps}`
  tuples in execution order. Each stage can be compiled independently;
  the dispatcher executes them sequentially, passing intermediate
  results.
  """
  @spec split_stages(Expr.t()) :: [stage()]
  def split_stages(%T{data: %Expr{}} = expr) do
    dag = linearize(expr, %{})

    # Walk the DAG bottom-up, grouping consecutive fusable ops.
    # A non-fusable op (dot, conv, reduce) creates a stage boundary.
    dag
    |> Enum.reduce([], fn {_id, {op, tensor}}, stages ->
      cond do
        op in @fusable_ops or op == :parameter or op == :constant ->
          add_to_current_stage(stages, :elementwise, {op, tensor})

        op in @reduce_ops ->
          [{:reduce, [{op, tensor}]} | stages]

        op in @library_ops ->
          [{:library, [{op, tensor}]} | stages]

        true ->
          [{:unsupported, [{op, tensor}]} | stages]
      end
    end)
    |> Enum.reverse()
  end

  defp add_to_current_stage([{:elementwise, ops} | rest], :elementwise, entry) do
    [{:elementwise, [entry | ops]} | rest]
  end

  defp add_to_current_stage(stages, kind, entry) do
    [{kind, [entry]} | stages]
  end

  # ----------------------------------------------------------------
  # Expression tree walking
  # ----------------------------------------------------------------

  # Emit GLSL expression string for an Nx.Defn.Expr node.
  defp emit_expr(%T{data: %Expr{op: :parameter, id: id}}, bindings) do
    case List.keyfind(bindings, id, 0) do
      {_, idx} -> {"v#{idx}", %{}}
      nil -> {"0.0", %{}}
    end
  end

  defp emit_expr(%T{data: %Expr{op: op, args: [arg]}}, bindings)
       when is_map_key(@unary_ops, op) do
    {inner, deps} = emit_expr(arg, bindings)
    glsl_template = Map.fetch!(@unary_ops, op)
    code = String.replace(glsl_template, "r", "(#{inner})")
    {code, deps}
  end

  defp emit_expr(%T{data: %Expr{op: op, args: [left, right]}}, bindings)
       when is_map_key(@binary_ops, op) do
    {l_code, _} = emit_expr(left, bindings)
    {r_code, _} = emit_expr(right, bindings)

    code =
      case Map.fetch!(@binary_ops, op) do
        sym when sym in ["+", "-", "*", "/"] ->
          "(#{l_code} #{sym} #{r_code})"

        func ->
          "#{func}(#{l_code}, #{r_code})"
      end

    {code, %{}}
  end

  defp emit_expr(%T{data: %Expr{op: op, args: [left, right]}}, bindings)
       when is_map_key(@compare_ops, op) do
    {l_code, _} = emit_expr(left, bindings)
    {r_code, _} = emit_expr(right, bindings)
    sym = Map.fetch!(@compare_ops, op)
    {"((#{l_code} #{sym} #{r_code}) ? 1.0 : 0.0)", %{}}
  end

  defp emit_expr(%T{data: %Expr{op: :constant, args: [val]}}, _bindings) do
    {"#{val / 1.0}", %{}}
  end

  defp emit_expr(%T{data: %Expr{op: op}}, _bindings) do
    raise "Nx.Vulkan.Codegen: unsupported op #{inspect(op)} in fusable expression"
  end

  # ----------------------------------------------------------------
  # DAG analysis
  # ----------------------------------------------------------------

  defp linearize(%T{data: %Expr{id: id, op: op, args: args}} = t, acc) do
    if Map.has_key?(acc, id) do
      acc
    else
      acc = Map.put(acc, id, {op, t})

      Enum.reduce(args, acc, fn
        %T{data: %Expr{}} = child, a -> linearize(child, a)
        _, a -> a
      end)
    end
  end

  defp partition(dag, _var_ids) do
    # Classify each node
    ops = Enum.map(dag, fn {_id, {op, _tensor}} -> op end)

    all_fusable =
      Enum.all?(ops, fn op ->
        op in @fusable_ops or op == :parameter or op == :constant
      end)

    has_reduce = Enum.any?(ops, fn op -> op in @reduce_ops end)
    has_library = Enum.any?(ops, fn op -> op in @library_ops end)

    cond do
      all_fusable -> [{:elementwise, ops}]
      has_reduce and not has_library -> [{:reduce_fused, ops}]
      true -> [{:mixed, ops}]
    end
  end

  defp collect_params(%T{data: %Expr{op: :parameter, id: id}}, acc) do
    Map.put(acc, id, true)
  end

  defp collect_params(%T{data: %Expr{args: args}}, acc) do
    Enum.reduce(args, acc, fn
      %T{data: %Expr{}} = child, a -> collect_params(child, a)
      _, a -> a
    end)
  end

  defp get_pre_reduce_expr(%T{data: %Expr{op: op, args: [inner | _]}})
       when op in @reduce_ops do
    inner
  end

  # ----------------------------------------------------------------
  # GLSL code generation helpers
  # ----------------------------------------------------------------

  defp emit_buffer_declarations(bindings, mode) do
    qualifier = if mode == :readonly, do: "readonly", else: ""

    bindings
    |> Enum.map(fn {_id, idx} ->
      "layout (std430, binding = #{idx}) #{qualifier} buffer Input#{idx} { float buf#{idx}[]; };"
    end)
    |> Enum.join("\n")
  end

  defp emit_input_loads(bindings) do
    bindings
    |> Enum.map(fn {_id, idx} ->
      "    float v#{idx} = buf#{idx}[i];"
    end)
    |> Enum.join("\n")
  end

  defp emit_indexed_loads(bindings, index_var) do
    bindings
    |> Enum.map(fn {_id, idx} ->
      "        float v#{idx} = buf#{idx}[#{index_var}];"
    end)
    |> Enum.join("\n")
  end

  defp emit_helper_functions do
    """
    float erf_approx(float x) {
        float a1 =  0.254829592;
        float a2 = -0.284496736;
        float a3 =  1.421413741;
        float a4 = -1.453152027;
        float a5 =  1.061405429;
        float p  =  0.3275911;
        float s  = sign(x);
        float ax = abs(x);
        float t  = 1.0 / (1.0 + p * ax);
        float y  = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * exp(-ax * ax);
        return s * y;
    }

    float expm1_approx(float x) {
        if (abs(x) < 0.5) {
            float x2 = x * x;
            return x + x2 * 0.5 + x2 * x * (1.0 / 6.0)
                 + x2 * x2 * (1.0 / 24.0) + x2 * x2 * x * (1.0 / 120.0);
        } else {
            return exp(x) - 1.0;
        }
    }
    """
  end

  defp reduce_init(:sum), do: "0.0"
  defp reduce_init(:product), do: "1.0"
  defp reduce_init(:reduce_max), do: "-1.0/0.0"
  defp reduce_init(:reduce_min), do: "1.0/0.0"

  defp reduce_combine(:sum), do: "acc + val"
  defp reduce_combine(:product), do: "acc * val"
  defp reduce_combine(:reduce_max), do: "max(acc, val)"
  defp reduce_combine(:reduce_min), do: "min(acc, val)"

  # ----------------------------------------------------------------
  # Type helpers
  # ----------------------------------------------------------------

  @type stage :: {:elementwise, [atom()]} | {:reduce_fused, [atom()]} | {:mixed, [atom()]}
end
