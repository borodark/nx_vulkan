defmodule Nx.Vulkan.CompilerTest do
  @moduledoc """
  Thrust 3 — the `Nx.Defn.Compiler` fusion compiler. A fusable same-shape f32
  elementwise chain JIT-compiles to ONE generated GLSL shader dispatched in a
  single GPU call; anything unsupported (reductions, tuple outputs, dot/conv,
  non-f32) falls through to `Nx.Defn.Evaluator` and stays correct.

  Every fused result is checked against the eager BinaryBackend computation.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.{VulkanoBackend, Codegen}

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp jit(fun), do: Nx.Defn.jit(fun, compiler: Nx.Vulkan.Compiler)

  defp bin(list), do: Nx.tensor(list, type: :f32, backend: Nx.BinaryBackend)

  defp close?(got, ref) do
    Nx.to_flat_list(got)
    |> Enum.zip(Nx.to_flat_list(ref))
    |> Enum.all?(fn {x, y} -> abs(x - y) <= 1.0e-5 end)
  end

  describe "fused elementwise chains run on the GPU in one dispatch" do
    setup do
      %{a: bin([1.0, 2.0, 3.0, 4.0]), b: bin([0.5, 1.5, 2.5, 3.5])}
    end

    test "tanh(a*b + a)", %{a: a, b: b} do
      got = jit(fn x, y -> Nx.tanh(Nx.add(Nx.multiply(x, y), x)) end).(a, b)
      ref = Nx.tanh(Nx.add(Nx.multiply(a, b), a))
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "a * 2.0 + b (scalar constant folded into the shader)", %{a: a, b: b} do
      got = jit(fn x, y -> Nx.add(Nx.multiply(x, 2.0), y) end).(a, b)
      ref = Nx.add(Nx.multiply(a, 2.0), b)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "relu via max(a - b, 0)", %{a: a, b: b} do
      got = jit(fn x, y -> Nx.max(Nx.subtract(x, y), 0.0) end).(a, b)
      ref = Nx.max(Nx.subtract(a, b), 0.0)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "sigmoid(a) * b", %{a: a, b: b} do
      got = jit(fn x, y -> Nx.multiply(Nx.sigmoid(x), y) end).(a, b)
      ref = Nx.multiply(Nx.sigmoid(a), b)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "single-argument deep unary chain sqrt(exp(negate(a)))", %{a: a} do
      got = jit(fn x -> Nx.sqrt(Nx.exp(Nx.negate(x))) end).(a)
      ref = Nx.sqrt(Nx.exp(Nx.negate(a)))
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "2D same-shape chain", %{} do
      a = Nx.reshape(bin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), {2, 3})
      b = Nx.reshape(bin([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), {2, 3})
      got = jit(fn x, y -> Nx.multiply(Nx.add(x, y), Nx.subtract(x, y)) end).(a, b)
      ref = Nx.multiply(Nx.add(a, b), Nx.subtract(a, b))
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.shape(got) == {2, 3}
      assert close?(got, ref)
    end
  end

  describe "unsupported graphs fall back to the Evaluator (still correct)" do
    test "reduction (sum) falls back" do
      a = bin([1.0, 2.0, 3.0, 4.0])
      b = bin([1.0, 1.0, 1.0, 1.0])
      got = jit(fn x, y -> Nx.sum(Nx.multiply(x, y)) end).(a, b)
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.to_number(got) == 10.0
    end

    test "tuple output falls back" do
      a = bin([1.0, 2.0])
      got = jit(fn x -> {Nx.negate(x), Nx.exp(x)} end).(a)
      {n, e} = got
      assert close?(n, Nx.negate(a))
      assert close?(e, Nx.exp(a))
    end

    test "integer (non-f32) chain falls back" do
      a = Nx.tensor([1, 2, 3], type: :s32, backend: Nx.BinaryBackend)
      got = jit(fn x -> Nx.add(x, x) end).(a)
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.to_flat_list(got) == [2, 4, 6]
    end
  end

  describe "Codegen unit" do
    test "fusable?/1 accepts an f32 elementwise tree, rejects a reduction" do
      alias Nx.Defn.Expr
      p0 = Expr.parameter(Nx.template({4}, :f32), :root, 0)
      p1 = Expr.parameter(Nx.template({4}, :f32), :root, 1)
      assert Codegen.fusable?(Nx.tanh(Nx.add(Nx.multiply(p0, p1), p0)))
      refute Codegen.fusable?(Nx.sum(Nx.multiply(p0, p1)))
    end

    test "emit_elementwise assigns bindings in ascending parameter order" do
      alias Nx.Defn.Expr
      p0 = Expr.parameter(Nx.template({4}, :f32), :root, 0)
      p1 = Expr.parameter(Nx.template({4}, :f32), :root, 1)
      # Use p1 before p0 in the source order; binding order must still be [0, 1].
      {glsl, meta} = Codegen.emit_elementwise(Nx.subtract(p1, p0))
      assert meta.param_order == [0, 1]
      assert meta.n_inputs == 2
      assert glsl =~ "out_buf[i] = (v1 - v0);"
    end
  end
end
