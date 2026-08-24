defmodule Nx.Vulkan.FloatRoundRemainderTest do
  @moduledoc """
  Two op codes the float shaders were missing, and the traps in both.

  Neither needed a new shader — only an arm and a widened selector. What they
  did need was picking the right formula, because in both cases GLSL's obvious
  built-in implements a DIFFERENT rule than `Nx.BinaryBackend`, and the
  difference only shows on inputs a casual test would not include.

    * `round` — Nx rounds HALF AWAY FROM ZERO. GLSL's `round()` is
      implementation-defined at a tie and may round to even; `roundEven()`
      definitely does. All three agree on 3.7 and disagree on 2.5.
    * `remainder` — Nx takes the sign of the DIVIDEND (`remainder(-5, 3)` is
      -2). GLSL's `mod()` takes the sign of the divisor and answers 1. They
      agree whenever the operands share a sign, which is half the cases.

  Both are asserted against `Nx.BinaryBackend` at every sign combination, and
  both assert residency — the values alone cannot tell a fallback from a fix.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp check(build) do
    got = build.(VulkanoBackend)
    assert match?(%VulkanoBackend{}, got.data), "expected the result to stay on the GPU"
    ref = build.(Nx.BinaryBackend)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  describe "round" do
    for type <- [{:f, 32}, {:f, 64}] do
      type = Macro.escape(type)

      test "#{inspect(type)} rounds half AWAY FROM ZERO" do
        # The ties are the whole test. -0.5 -> -1 and 2.5 -> 3 are what
        # separate away-from-zero from round-to-even, which would give -0 and 2.
        vals = [-3.5, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.5, 3.7, -3.7, 0.4999]
        check(fn b -> Nx.round(Nx.tensor(vals, type: unquote(type), backend: b)) end)
      end
    end

    test "the ties, spelled out" do
      t = Nx.tensor([-2.5, -0.5, 0.5, 2.5], type: {:f, 32}, backend: VulkanoBackend)
      assert Nx.to_flat_list(Nx.round(t)) == [-3.0, -1.0, 1.0, 3.0]
    end
  end

  describe "remainder" do
    for type <- [{:f, 32}, {:f, 64}] do
      type = Macro.escape(type)

      test "#{inspect(type)} takes the sign of the DIVIDEND, at all four sign pairs" do
        xs = [5.0, -5.0, 5.0, -5.0, 7.5, -7.5]
        ys = [3.0, 3.0, -3.0, -3.0, 2.0, 2.0]

        check(fn b ->
          Nx.remainder(
            Nx.tensor(xs, type: unquote(type), backend: b),
            Nx.tensor(ys, type: unquote(type), backend: b)
          )
        end)
      end

      test "#{inspect(type)} broadcasting against a scalar divisor" do
        check(fn b ->
          Nx.remainder(
            Nx.tensor([5.0, -5.0, 8.0], type: unquote(type), backend: b),
            Nx.tensor(3.0, type: unquote(type), backend: b)
          )
        end)
      end
    end

    test "the sign pairs, spelled out — GLSL's mod would answer 1.0 for two of them" do
      x = Nx.tensor([5.0, -5.0, 5.0, -5.0], type: {:f, 32}, backend: VulkanoBackend)
      y = Nx.tensor([3.0, 3.0, -3.0, -3.0], type: {:f, 32}, backend: VulkanoBackend)
      assert Nx.to_flat_list(Nx.remainder(x, y)) == [2.0, -2.0, 2.0, -2.0]
    end
  end

  describe "the arms that were removed to make room" do
    # elementwise_binary_f64.comp defined codes 7/8/9 as equal/less/greater,
    # left over from before compare_f64.comp existed. They were unreachable
    # under the old `code <= 6` cap — but they gave code 8 a different MEANING
    # in that one file than in @binary_ops, so widening the cap by one (which
    # is exactly what `remainder` did) would have returned a comparison mask
    # and looked entirely plausible.
    #
    # This asserts the removal took nothing with it: f64 comparisons go through
    # compare_f64.comp and always did.
    test "f64 comparisons are unaffected and still resident" do
      a = Nx.tensor([1.0, 2.0, 3.0], type: {:f, 64}, backend: VulkanoBackend)
      b = Nx.tensor([1.0, 3.0, 2.0], type: {:f, 64}, backend: VulkanoBackend)

      for {op, expected} <- [
            {:equal, [1, 0, 0]},
            {:less, [0, 1, 0]},
            {:greater, [0, 0, 1]}
          ] do
        got = apply(Nx, op, [a, b])
        assert match?(%VulkanoBackend{}, got.data), "#{op} left the GPU"
        assert Nx.to_flat_list(got) == expected
      end
    end
  end
end
