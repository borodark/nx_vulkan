defmodule Nx.Vulkan.BroadcastTest do
  @moduledoc """
  Broadcasting elementwise binary ops on the GPU (thrust 2 — kill host-fallback
  round-trips). Bias-add, relu-via-max, softmax subtract, row/col/4D broadcast
  now dispatch a broadcast shader instead of transferring to BinaryBackend.
  Verified against a BinaryBackend reference; f32 (the DL path) stays on-GPU.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp maxdiff(a, b) do
    Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend), Nx.backend_copy(b, Nx.BinaryBackend))
    |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  # run builder on both backends; assert exact-ish match + on-GPU (f32)
  defp check(build, tol) do
    got = build.(VulkanoBackend)
    ref = build.(Nx.BinaryBackend)
    assert match?(%VulkanoBackend{}, got.data), "expected on-GPU broadcast dispatch"
    assert Nx.type(got) == Nx.type(ref)
    assert maxdiff(got, ref) <= tol
  end

  describe "f32 broadcast on GPU (the DL path)" do
    test "bias add {8,16}+{16}" do
      check(fn b -> Nx.add(Nx.iota({8, 16}, type: {:f, 32}, backend: b), Nx.iota({16}, type: {:f, 32}, backend: b)) end, 0.0)
    end

    test "relu max(x, 0.0)" do
      check(fn b -> Nx.max(Nx.subtract(Nx.iota({8, 16}, type: {:f, 32}, backend: b), 40.0), 0.0) end, 0.0)
    end

    test "softmax subtract {8,16}-{8,1}" do
      check(fn b ->
        t = Nx.iota({8, 16}, type: {:f, 32}, backend: b)
        Nx.subtract(t, Nx.reduce_max(t, axes: [1], keep_axes: true))
      end, 0.0)
    end

    test "scalar multiply {5,5}*3.0" do
      check(fn b -> Nx.multiply(Nx.iota({5, 5}, type: {:f, 32}, backend: b), 3.0) end, 0.0)
    end

    test "col divide {8,16}/{8,1}" do
      check(fn b ->
        Nx.divide(Nx.add(Nx.iota({8, 16}, type: {:f, 32}, backend: b), 1.0), Nx.add(Nx.iota({8, 1}, type: {:f, 32}, backend: b), 1.0))
      end, 1.0e-5)
    end

    test "4D bias {2,3,4,5}+{5}" do
      check(fn b -> Nx.add(Nx.iota({2, 3, 4, 5}, type: {:f, 32}, backend: b), Nx.iota({5}, type: {:f, 32}, backend: b)) end, 0.0)
    end
  end

  test "f64 same-type broadcast on GPU" do
    check(fn b -> Nx.add(Nx.iota({8, 16}, type: {:f, 64}, backend: b), Nx.iota({16}, type: {:f, 64}, backend: b)) end, 0.0)
  end
end
