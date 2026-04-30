defmodule Nx.Vulkan.Compiler do
  @moduledoc """
  Path A.2 v2 (partial) — `Nx.Defn.Compiler` that auto-detects fusable
  elementwise chains and dispatches `Nx.Vulkan.fused_chain/3` instead
  of N separate shader calls.

  ## What it does today

  At `__jit__` time, calls `fun.(vars)` once to materialize the IR.
  Walks the result looking for a chain of supported elementwise ops
  whose only inputs are the two function arguments. If matched: skip
  Evaluator entirely and dispatch a single `fused_chain` call.

  ## What it doesn't do yet

    * **Multi-output**: only single-output chains. A defn that returns
      a tuple falls through to `Nx.Defn.Evaluator`.
    * **Branched chains**: only linear chains. A node that's used
      twice falls through.
    * **More than 2 vars**: 2-arg functions only (matches the shader's
      two-input layout). Wider arities fall through.
    * **Chains > 8 ops**: shader limit; longer chains fall through.
    * **Non-elementwise ops**: any reduce/reshape/dot in the chain
      falls through.

  All fall-through cases delegate to `Nx.Defn.Evaluator` so behavior
  stays correct — the worst case is "no fusion, same speed as before."

  ## Configuration

      config :exmc, :compiler, :vulkan

  `Exmc.JIT` then routes to `Nx.Vulkan.jit/2`, which uses this compiler
  when available (defaults to `Nx.Defn.Evaluator` if unsupported).
  """

  @behaviour Nx.Defn.Compiler

  alias Nx.Defn.Expr
  alias Nx.Tensor, as: T

  @binary_ops %{
    add: 0,
    multiply: 1,
    subtract: 2,
    divide: 3,
    pow: 4,
    max: 5,
    min: 6
  }

  # Commutative binary ops — if first arg is the b var and second is
  # not, we can swap and still produce the correct result. Subtract,
  # divide, pow are NOT commutative and so the chain bails on the
  # reversed pattern.
  @commutative_ops [:add, :multiply, :max, :min]

  @unary_ops [
    :exp,
    :log,
    :sqrt,
    :abs,
    :negate,
    :sigmoid,
    :tanh,
    :relu,
    :ceil,
    :floor,
    :sign,
    :reciprocal,
    :square,
    :erf,
    :expm1
  ]

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
    expr = trace(fun, vars)
    var_ids = collect_var_ids(vars)

    case detect_chain(expr, var_ids) do
      {:ok, ops, a_var_id, b_var_id} ->
        # Build a closure that bypasses Evaluator entirely.
        compile_fused(ops, a_var_id, b_var_id, vars, expr)

      :no_match ->
        # Fall through: evaluator handles the rest.
        Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)
    end
  end

  @impl true
  def __shard_jit__(_key, _mesh, _vars, _fun, _args_list, _opts) do
    raise "sharding is not supported by Nx.Vulkan.Compiler"
  end

  # --- Tracing ---------------------------------------------------------

  defp trace(fun, vars) do
    # Trace under the Nx.Defn.Expr backend so calls produce IR nodes
    # rather than executing on whichever backend is current.
    previous = Process.put(Nx.Shared.backend_pdict_key(), {Nx.Defn.Expr, []})

    try do
      vars
      |> fun.()
    after
      if previous,
        do: Process.put(Nx.Shared.backend_pdict_key(), previous),
        else: Process.delete(Nx.Shared.backend_pdict_key())
    end
  end

  defp collect_var_ids(vars) do
    vars
    |> Enum.map(fn
      %T{data: %Expr{op: :parameter, args: [i]}} = t ->
        # Capture {position, id, shape, type} for matching during walk.
        {i, t.data.id, t.shape, t.type}

      _ ->
        nil
    end)
    |> Enum.reject(&is_nil/1)
  end

  # --- Chain detection -------------------------------------------------

  # A chain is recognizable iff the result IR is a single-output tensor
  # whose op is elementwise, recursing into a chain that ultimately
  # bottoms out at one of the input vars (the "a" var). Every binary
  # step's second operand must be the "b" var.
  defp detect_chain(%T{data: %Expr{}} = root, var_ids) when length(var_ids) == 2 do
    [{0, a_id, _shape_a, _type_a}, {1, b_id, _shape_b, _type_b}] = var_ids

    case walk_chain(root, a_id, b_id, []) do
      {:ok, ops} when length(ops) >= 1 and length(ops) <= 8 ->
        {:ok, ops, a_id, b_id}

      _ ->
        :no_match
    end
  end

  # 1-arg defn — pass `a` as both buffers. Only unary chains can fuse
  # this way (binary ops would degenerate to op(a, a) which is
  # rarely what the user wrote). The shader still reads b unconditionally
  # but ignores its value when no binary op fires.
  defp detect_chain(%T{data: %Expr{}} = root, var_ids) when length(var_ids) == 1 do
    [{0, a_id, _shape, _type}] = var_ids

    case walk_unary_only(root, a_id, []) do
      {:ok, ops} when length(ops) >= 1 and length(ops) <= 8 ->
        {:ok, ops, a_id, a_id}

      _ ->
        :no_match
    end
  end

  defp detect_chain(_, _), do: :no_match

  # 1-arg variant: only unary ops; bottom out at the single var.
  defp walk_unary_only(%T{data: %Expr{id: id, op: :parameter}}, a_id, acc)
       when id == a_id do
    {:ok, acc}
  end

  defp walk_unary_only(%T{data: %Expr{op: op, args: [arg]}}, a_id, acc)
       when op in @unary_ops do
    walk_unary_only(arg, a_id, [op | acc])
  end

  defp walk_unary_only(_, _, _), do: :no_match

  # Reached `a` — bottom of chain.
  defp walk_chain(%T{data: %Expr{id: id, op: :parameter}}, a_id, _b_id, acc)
       when id == a_id do
    {:ok, acc}
  end

  # Unary fusable op — record and recurse into single arg.
  defp walk_chain(%T{data: %Expr{op: op, args: [arg]}}, a_id, b_id, acc)
       when op in @unary_ops do
    walk_chain(arg, a_id, b_id, [op | acc])
  end

  # Binary fusable op — second arg must be `b`. If first is `b` and the
  # op is commutative, swap and continue.
  defp walk_chain(%T{data: %Expr{op: op, args: [first, second]}}, a_id, b_id, acc) do
    cond do
      not Map.has_key?(@binary_ops, op) ->
        :no_match

      var_id(second) == b_id ->
        walk_chain(first, a_id, b_id, [op | acc])

      var_id(first) == b_id and op in @commutative_ops ->
        # Reversed-order commutative pattern (e.g., Nx.add(b, expr)).
        # Swap and proceed as if it were Nx.add(expr, b).
        walk_chain(second, a_id, b_id, [op | acc])

      true ->
        :no_match
    end
  end

  defp walk_chain(_, _, _, _), do: :no_match

  defp var_id(%T{data: %Expr{op: :parameter, id: id}}), do: id
  defp var_id(_), do: nil

  # --- Compilation -----------------------------------------------------

  defp compile_fused(ops, _a_var_id, _b_var_id, _vars, expr) do
    out_shape = expr.shape
    out_type = expr.type

    fn [params] ->
      # Per Nx.Defn.Compiler protocol, params is a list of zero-arity
      # thunks (`[(-> Nx.Tensor.t())]`). For 1-arg defns we pass the
      # same buffer as both a and b to the shader (which reads b
      # unconditionally but ignores it for unary-only chains).
      case params do
        [a_thunk] ->
          a_tensor = a_thunk.()
          [run_fused(a_tensor, a_tensor, ops, out_shape, out_type)]

        [a_thunk, b_thunk | _] ->
          [run_fused(a_thunk.(), b_thunk.(), ops, out_shape, out_type)]
      end
    end
  end

  defp run_fused(a_tensor, b_tensor, ops, out_shape, out_type) do
    case {a_tensor.data, b_tensor.data} do
      {%Nx.Vulkan.Backend{ref: a_ref}, %Nx.Vulkan.Backend{ref: b_ref}} ->
        {:ok, ref} = Nx.Vulkan.fused_chain(a_ref, b_ref, ops)

        %T{
          data: %Nx.Vulkan.Backend{ref: ref, shape: out_shape, type: out_type},
          shape: out_shape,
          type: out_type,
          names: List.duplicate(nil, tuple_size(out_shape))
        }

      _ ->
        # Operands aren't both on the Vulkan backend — fall through to
        # ordinary execution. The simplest "fall through" here is to
        # just rebuild the chain on whatever backend the inputs are on.
        Enum.reduce(ops, a_tensor, &apply_chain_op(&1, &2, b_tensor))
    end
  end

  # Helper for the fall-through path: apply each op in order.
  defp apply_chain_op(op, acc, _b_tensor) when op in @unary_ops do
    apply(Nx, op, [acc])
  end

  defp apply_chain_op(op, acc, b_tensor) do
    apply(Nx, op, [acc, b_tensor])
  end
end
