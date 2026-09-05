defmodule Nx.Vulkan.ChainBoundaryTest do
  @moduledoc """
  Where the f64 chain path stops being f64.

  GLSL.std.450 has no double-precision transcendentals, so every f64 family that
  needs one casts down through f32 and back. That cast has the IEEE f32 range,
  and past it the shader returns Inf — silently, in an otherwise-f64 result.

  Measured on the RTX 3060 Ti, first |q| producing a non-finite logp or grad:

      normal_f64        none to 700     no boundary cast at all
      cauchy_f64        none to 700     log(1 + z^2) grows too slowly
      studentt_f64      none to 700     same
      exponential_f64   88.72           exp(float(q)),  ln(f32_max) = 88.7228
      halfnormal_f64    44.36           exp(float(2q)), half of it
      weibull_f64       44.36           same

  These are pinned loosely — well inside and well outside — because the point is
  the CLASS, not the third decimal: a driver with different denormal handling
  may move the exact edge, but no conforming one moves it from 44 to 400.

  Two ways this test earns its place. If someone gives a family a native f64
  path, its row here starts passing where it should fail and the pin says so.
  And it documents, executably, that a sampler driving a scale parameter toward
  zero can reach these magnitudes during warmup while every fixed-point test in
  the suite stays finite.
  """
  use ExUnit.Case, async: true
  @moduletag :gpu

  alias Nx.Vulkan.{ChainShaderSpecsF64, NativeV, Synthesis}

  @d 2
  @k 1

  defp f64(l), do: for(v <- l, into: <<>>, do: <<v::little-float-64>>)
  defp push(k, d, eps), do: <<k::little-32, 0::little-32, d::little-32, 0::little-32, eps::little-float-64>>

  # Arity-guarded: a binary comprehension SKIPS an Inf/NaN segment rather than
  # raising, so counting is the only way to notice one. See PROPERTY_TESTING.md §8.
  defp all_finite?(bin) do
    length(for <<v::little-float-64 <- bin>>, do: v) == div(byte_size(bin), 8)
  end

  defp probe(spec, q) do
    {:ok, spv} = Synthesis.compile(spec)
    qs = f64(List.duplicate(q, @d))
    zero = f64(List.duplicate(0.0, @d))
    ones = f64(List.duplicate(1.0, @d))

    {:ok, {_q, _p, g, l}} =
      NativeV.leapfrog_chain_synth_f64(qs, zero, ones, push(@k, @d, 1.0e-8), @k, spv)

    all_finite?(g) and all_finite?(l)
  end

  @unbounded ~w(normal cauchy studentt)a
  @exp_q ~w(exponential)a
  @exp_2q ~w(halfnormal weibull)a

  defp spec_for(name) do
    Enum.find(ChainShaderSpecsF64.all(), &(&1.name == "#{name}_f64")) ||
      flunk("no f64 family named #{name}")
  end

  describe "families with no boundary cast survive anything reachable" do
    for fam <- @unbounded do
      test "#{fam}_f64 stays finite at |q| = 700" do
        assert probe(spec_for(unquote(fam)), 700.0),
               "#{unquote(fam)}_f64 went non-finite; it has no f32 cast that should do that"
      end
    end
  end

  describe "families that cast through f32 overflow at the f32 range" do
    for fam <- @exp_q do
      test "#{fam}_f64 is finite at 80 and not at 100 — exp(float(q))" do
        assert probe(spec_for(unquote(fam)), 80.0), "overflowed EARLY, below ln(f32_max)"
        refute probe(spec_for(unquote(fam)), 100.0), "did NOT overflow — has the cast been removed?"
      end
    end

    for fam <- @exp_2q do
      test "#{fam}_f64 is finite at 40 and not at 50 — exp(float(2q))" do
        assert probe(spec_for(unquote(fam)), 40.0), "overflowed EARLY, below ln(f32_max)/2"
        refute probe(spec_for(unquote(fam)), 50.0), "did NOT overflow — has the cast been removed?"
      end
    end
  end
end
