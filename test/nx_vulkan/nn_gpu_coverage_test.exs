defmodule Nx.Vulkan.NnGpuCoverageTest do
  @moduledoc """
  Thrust-2 milestone: an f32 NN forward pass now stays entirely on the GPU — no
  host round-trips. Before the broadcast wiring, bias-add / relu / softmax-sub /
  softmax-div all fell back to BinaryBackend. This guards that every op in an
  mlp + softmax forward (and clip) returns a VulkanoBackend tensor, and that the
  result matches a BinaryBackend reference.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp gpu?(t), do: match?(%VulkanoBackend{}, t.data)

  defp mlp_softmax(backend) do
    x = Nx.iota({8, 20}, type: {:f, 32}, backend: backend)
    w1 = Nx.divide(Nx.iota({20, 16}, type: {:f, 32}, backend: backend), 100.0)
    b1 = Nx.iota({16}, type: {:f, 32}, backend: backend)
    w2 = Nx.divide(Nx.iota({16, 3}, type: {:f, 32}, backend: backend), 50.0)
    b2 = Nx.iota({3}, type: {:f, 32}, backend: backend)

    h = Nx.dot(x, w1) |> Nx.add(b1) |> Nx.max(0.0)
    logits = Nx.dot(h, w2) |> Nx.add(b2)
    m = Nx.reduce_max(logits, axes: [1], keep_axes: true)
    e = Nx.exp(Nx.subtract(logits, m))
    Nx.divide(e, Nx.sum(e, axes: [1], keep_axes: true))
  end

  test "full mlp + softmax forward stays on GPU and matches BinaryBackend" do
    got = mlp_softmax(VulkanoBackend)
    assert gpu?(got), "NN forward leaked to host — a broadcast/elementwise op fell back"

    ref = mlp_softmax(Nx.BinaryBackend)

    diff =
      Nx.subtract(Nx.backend_copy(got, Nx.BinaryBackend), ref)
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff < 1.0e-6
  end

  test "clip stays on GPU (f32) and matches" do
    v = Nx.subtract(Nx.iota({4, 5}, type: {:f, 32}, backend: VulkanoBackend), 8.0)
    b = Nx.subtract(Nx.iota({4, 5}, type: {:f, 32}, backend: Nx.BinaryBackend), 8.0)
    got = Nx.clip(v, 0.0, 5.0)
    assert gpu?(got)

    diff =
      Nx.subtract(Nx.backend_copy(got, Nx.BinaryBackend), Nx.clip(b, 0.0, 5.0))
      |> Nx.abs()
      |> Nx.reduce_max()
      |> Nx.to_number()

    assert diff == 0.0
  end
end
