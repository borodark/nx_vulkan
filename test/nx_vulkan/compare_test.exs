defmodule Nx.Vulkan.CompareTest do
  @moduledoc """
  GPU comparison ops -> u8 (thrust 2). equal/not_equal/less/less_equal/greater/
  greater_equal now run a shader that packs u8 results into u32 words (no 8-bit
  storage needed) when both operands share an f32/f64 type. With select, this
  puts the full relu-grad mask chain (x > 0 -> select) on the GPU.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp check(build) do
    got = build.(VulkanoBackend)
    ref = build.(Nx.BinaryBackend)
    assert match?(%VulkanoBackend{}, got.data), "expected on-GPU comparison"
    assert Nx.type(got) == {:u, 8}
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  describe "comparison ops on GPU (f32)" do
    test "greater x>0",
      do:
        check(fn b ->
          Nx.greater(Nx.subtract(Nx.iota({4, 5}, type: {:f, 32}, backend: b), 10.0), 0.0)
        end)

    test "less",
      do:
        check(fn b ->
          Nx.less(
            Nx.iota({7}, type: {:f, 32}, backend: b),
            Nx.tensor(3.0, type: {:f, 32}, backend: b)
          )
        end)

    test "greater_equal, non-multiple-of-4 length",
      do: check(fn b -> Nx.greater_equal(Nx.iota({5}, type: {:f, 32}, backend: b), 2.0) end)

    test "not_equal",
      do: check(fn b -> Nx.not_equal(Nx.iota({3, 3}, type: {:f, 32}, backend: b), 4.0) end)

    test "row-broadcast greater {4,5} > {4,1}" do
      check(fn b ->
        t = Nx.iota({4, 5}, type: {:f, 32}, backend: b)
        Nx.greater(t, Nx.reduce_max(t, axes: [1], keep_axes: true))
      end)
    end
  end

  test "full relu-grad mask chain stays on GPU (greater -> select)" do
    b = VulkanoBackend
    x = Nx.subtract(Nx.iota({4, 5}, type: {:f, 32}, backend: b), 10.0)
    dy = Nx.add(Nx.iota({4, 5}, type: {:f, 32}, backend: b), 1.0)
    mask = Nx.greater(x, 0.0)
    grad = Nx.select(mask, dy, Nx.tensor(0.0, type: {:f, 32}, backend: b))
    assert match?(%VulkanoBackend{}, mask.data)
    assert match?(%VulkanoBackend{}, grad.data)

    xb = Nx.backend_copy(x, Nx.BinaryBackend)
    dyb = Nx.backend_copy(dy, Nx.BinaryBackend)

    ref =
      Nx.select(
        Nx.greater(xb, 0.0),
        dyb,
        Nx.tensor(0.0, type: {:f, 32}, backend: Nx.BinaryBackend)
      )

    assert Nx.to_flat_list(Nx.backend_copy(grad, Nx.BinaryBackend)) == Nx.to_flat_list(ref)
  end
end
