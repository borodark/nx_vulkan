defmodule Nx.Vulkan.Fuse do
  @moduledoc """
  Path A.2 — compile-time fusion of elementwise op chains.

  Given a 2-arg function whose body is a chain of `Nx.*` calls, rewrites
  it at macro time into a single `Nx.Vulkan.fused_chain/3` dispatch.
  Replaces N shader dispatches with one.

  ## Example

      import Nx.Vulkan.Fuse

      f = fuse(fn a, b -> Nx.exp(Nx.add(Nx.multiply(a, b), b)) end)
      {:ok, c} = f.(a_ref, b_ref)
      # one fused dispatch instead of three

  ## Recognized pattern

  The macro recognizes function bodies of the form:

      Nx.<op_n>(Nx.<op_{n-1}>(... Nx.<op_1>(a, b) ...))

  where:

    * `a` and `b` are the two function arguments (`b` may be reused for
      every binary op).
    * Each `op_k` is from the supported elementwise set:
      - Binary (combine register with `b`): `:add`, `:subtract`,
        `:multiply`, `:divide`, `:pow`, `:max`, `:min`
      - Unary (transform register): `:exp`, `:log`, `:sqrt`, `:abs`,
        `:negate`, `:sigmoid`, `:tanh`, `:relu`, `:ceil`, `:floor`,
        `:sign`, `:reciprocal`, `:square`

  Chains longer than 8 ops fall back to non-fused composition (one
  shader dispatch per op).

  ## Limitations (v0.2 work)

    * No autograd integration — `fuse` returns a `Nx.Vulkan` ref tuple,
      not a defn-traceable value.
    * Binary ops only see `b` as the second operand. `Nx.add(a, c)`
      where `c` is a third tensor doesn't fuse.
    * No reshape/broadcast fusion. Only same-shape elementwise.
    * Auto-detection inside `defn` blocks requires a real
      `Nx.Defn.Compiler`. That's the v0.2 follow-up that recognizes
      chains in any defn body without macro opt-in.
  """

  @binary_ops %{
    add: :add,
    subtract: :subtract,
    multiply: :multiply,
    divide: :divide,
    pow: :pow,
    max: :max,
    min: :min
  }

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
    :square
  ]

  @doc """
  Macro entry point. See module docs.

  Returns the original function unchanged if the body doesn't fit the
  recognized chain pattern — graceful fallback, never a compile error.
  """
  defmacro fuse({:fn, _, [{:->, _, [[a_ast, b_ast], body]}]} = orig) do
    a_name = var_name(a_ast)
    b_name = var_name(b_ast)

    case build_chain(body, a_name, b_name, []) do
      {:ok, ops} when length(ops) > 0 and length(ops) <= 8 ->
        quote do
          fn unquote(a_ast), unquote(b_ast) ->
            Nx.Vulkan.fused_chain(unquote(a_ast), unquote(b_ast), unquote(ops))
          end
        end

      _ ->
        orig
    end
  end

  defmacro fuse(other), do: other

  # --- AST walker ----------------------------------------------------

  # Leaf: the function's first arg. End of chain (innermost op input).
  defp build_chain({var_name, _, ctx}, a_name, _b_name, acc)
       when is_atom(var_name) and is_atom(ctx) and var_name == a_name do
    {:ok, acc}
  end

  # Unary Nx.<op>(inner) — collect op, recurse into inner.
  defp build_chain(
         {{:., _, [{:__aliases__, _, [:Nx]}, op]}, _, [inner]},
         a_name,
         b_name,
         acc
       )
       when is_atom(op) do
    cond do
      op in @unary_ops ->
        build_chain(inner, a_name, b_name, [op | acc])

      true ->
        :no_chain
    end
  end

  # Binary Nx.<op>(inner, b) — collect op, recurse into inner.
  defp build_chain(
         {{:., _, [{:__aliases__, _, [:Nx]}, op]}, _, [inner, second]},
         a_name,
         b_name,
         acc
       )
       when is_atom(op) do
    cond do
      Map.has_key?(@binary_ops, op) and var_name(second) == b_name ->
        build_chain(inner, a_name, b_name, [Map.fetch!(@binary_ops, op) | acc])

      true ->
        :no_chain
    end
  end

  defp build_chain(_, _, _, _), do: :no_chain

  # Pull a var name out of an AST node, returning nil for non-vars.
  defp var_name({name, _, ctx}) when is_atom(name) and is_atom(ctx), do: name
  defp var_name(_), do: nil
end
