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
      {:ok, outer_ops, b_pre_ops, a_var_id, b_var_id} ->
        if System.get_env("NXV_FUSE_DEBUG") == "1" do
          tag = if b_pre_ops == [], do: "FUSED", else: "FUSED+pre"
          IO.puts("[Nx.Vulkan.Compiler] #{tag}: pre=#{inspect(b_pre_ops)} outer=#{inspect(outer_ops)}")
        end

        compile_fused(outer_ops, b_pre_ops, a_var_id, b_var_id, vars, expr)

      :no_match ->
        if System.get_env("NXV_FUSE_DEBUG") == "1" do
          IO.puts("[Nx.Vulkan.Compiler] no_match — vars=#{length(var_ids)} root_op=#{inspect_root(expr)}")
        end

        # Fall through: evaluator handles the rest.
        Nx.Defn.Evaluator.__compile__(key, vars, fun, opts)
    end
  end

  defp inspect_root(%T{data: %Expr{op: op, args: args}}) do
    arg_ops =
      Enum.map(args, fn
        %T{data: %Expr{op: o}} -> o
        other -> "#{inspect(other)}"
      end)

    "#{op} args=#{inspect(arg_ops)}"
  end

  defp inspect_root(other), do: inspect(other)

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
      {:ok, ops, b_pre_ops}
      when length(ops) >= 1 and length(ops) <= 8 and length(b_pre_ops) <= 8 ->
        {:ok, ops, b_pre_ops, a_id, b_id}

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
      {:ok, ops, []} when length(ops) >= 1 and length(ops) <= 8 ->
        {:ok, ops, [], a_id, a_id}

      _ ->
        :no_match
    end
  end

  defp detect_chain(_, _), do: :no_match

  # 1-arg variant: only unary ops; bottom out at the single var.
  defp walk_unary_only(%T{data: %Expr{id: id, op: :parameter}}, a_id, acc)
       when id == a_id do
    {:ok, acc, []}
  end

  defp walk_unary_only(%T{data: %Expr{op: op, args: [arg]}}, a_id, acc)
       when op in @unary_ops do
    walk_unary_only(arg, a_id, [op | acc])
  end

  defp walk_unary_only(_, _, _), do: :no_match

  # Walks a sub-expression that bottoms out at b. Recognized shapes:
  #   - parameter b → empty pre-chain (just use b directly)
  #   - unary(sub) → recurse, prepend the unary
  #   - multiply(b, b) → :square peephole
  defp walk_b_subchain(%T{data: %Expr{op: :parameter, id: id}}, b_id, acc)
       when id == b_id do
    {:ok, acc}
  end

  defp walk_b_subchain(%T{data: %Expr{op: op, args: [arg]}}, b_id, acc)
       when op in @unary_ops do
    walk_b_subchain(arg, b_id, [op | acc])
  end

  # mult(b, b) ⇒ square(b) peephole
  defp walk_b_subchain(
         %T{
           data: %Expr{
             op: :multiply,
             args: [%T{data: %Expr{op: :parameter, id: id1}}, %T{data: %Expr{op: :parameter, id: id2}}]
           }
         },
         b_id,
         acc
       )
       when id1 == b_id and id2 == b_id do
    {:ok, [:square | acc]}
  end

  defp walk_b_subchain(_, _, _), do: :no_match

  # walk_chain returns {:ok, ops, b_pre_ops} or :no_match.
  # b_pre_ops is the unary chain to apply to b BEFORE the outer chain.
  # Most paths return [] (b used directly); right-folded patterns return
  # the pre-eval ops for b's sub-expression.

  # Reached `a` — bottom of chain. No b pre-eval.
  defp walk_chain(%T{data: %Expr{id: id, op: :parameter}}, a_id, _b_id, acc)
       when id == a_id do
    {:ok, acc, []}
  end

  # Unary fusable op — record and recurse. Propagates b_pre_ops from below.
  defp walk_chain(%T{data: %Expr{op: op, args: [arg]}}, a_id, b_id, acc)
       when op in @unary_ops do
    walk_chain(arg, a_id, b_id, [op | acc])
  end

  # Binary fusable op — second arg must be `b`. If first is `b` and the
  # op is commutative, swap and continue. If neither but first is the
  # `a` parameter and second is a chain on b, switch to right-folded
  # mode (pre-eval the second-arg sub-chain on b once, then dispatch
  # the outer chain with that temp as b).
  defp walk_chain(%T{data: %Expr{op: op, args: [first, second]}}, a_id, b_id, acc) do
    cond do
      not Map.has_key?(@binary_ops, op) ->
        :no_match

      var_id(second) == b_id ->
        walk_chain(first, a_id, b_id, [op | acc])

      var_id(first) == b_id and op in @commutative_ops ->
        walk_chain(second, a_id, b_id, [op | acc])

      var_id(first) == a_id ->
        # Right-folded: chain ends here on the a side; pre-eval the
        # sub-chain on b and combine with op.
        case walk_b_subchain(second, b_id, []) do
          {:ok, b_pre_ops} -> {:ok, [op | acc], b_pre_ops}
          :no_match -> :no_match
        end

      var_id(second) == a_id and op in @commutative_ops ->
        case walk_b_subchain(first, b_id, []) do
          {:ok, b_pre_ops} -> {:ok, [op | acc], b_pre_ops}
          :no_match -> :no_match
        end

      true ->
        :no_match
    end
  end

  defp walk_chain(_, _, _, _), do: :no_match

  defp var_id(%T{data: %Expr{op: :parameter, id: id}}), do: id
  defp var_id(_), do: nil

  # --- Compilation -----------------------------------------------------

  defp compile_fused(outer_ops, b_pre_ops, _a_var_id, _b_var_id, _vars, expr) do
    out_shape = expr.shape
    out_type = expr.type

    fn [params] ->
      case params do
        [a_thunk] ->
          # 1-arg path — no b_pre_ops possible.
          a_tensor = a_thunk.()
          [run_fused(a_tensor, a_tensor, outer_ops, out_shape, out_type)]

        [a_thunk, b_thunk | _] ->
          a_tensor = a_thunk.()
          b_tensor = b_thunk.()

          # If b_pre_ops is non-empty, evaluate that unary chain on b
          # first to produce a temp buffer used as b in the outer
          # fused chain. One pre-dispatch + one fused dispatch.
          b_eff =
            if b_pre_ops == [] do
              b_tensor
            else
              run_fused(b_tensor, b_tensor, b_pre_ops, b_tensor.shape, b_tensor.type)
            end

          [run_fused(a_tensor, b_eff, outer_ops, out_shape, out_type)]
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
