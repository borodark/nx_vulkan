defmodule Nx.Vulkan.GatherTest do
  @moduledoc """
  GPU gather (thrust 2). A type-generic u32-word-copy shader handles the common
  gather where the indexed axes are a leading prefix [0..K-1] (includes the
  default all-axes case): each index row selects a leading multi-coord and
  copies the contiguous trailing block. Runs for 4/8-byte value + index dtypes,
  rank 1..4; non-prefix axes, sub-word dtypes and rank > 4 host-fall-back.
  Verified vs BinaryBackend.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp check(build, on_gpu) do
    got = build.(VulkanoBackend)
    ref = build.(Nx.BinaryBackend)
    if on_gpu, do: assert(match?(%VulkanoBackend{}, got.data), "expected on-GPU gather")
    assert Nx.shape(got) == Nx.shape(ref)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  for vt <- [{:f, 32}, {:f, 64}, {:s, 32}], it <- [{:s, 64}, {:s, 32}] do
    vt = Macro.escape(vt)
    it = Macro.escape(it)

    describe "gather value=#{inspect(vt)} index=#{inspect(it)} on GPU" do
      test "1d full", do: check(fn b -> Nx.gather(Nx.iota({6}, type: unquote(vt), backend: b), Nx.tensor([[0], [3], [5], [1]], type: unquote(it), backend: b)) end, true)
      test "2d full (scalar picks)", do: check(fn b -> Nx.gather(Nx.iota({3, 4}, type: unquote(vt), backend: b), Nx.tensor([[0, 1], [2, 3], [1, 0]], type: unquote(it), backend: b)) end, true)
      test "2d rows axes[0] (block copy)", do: check(fn b -> Nx.gather(Nx.iota({3, 4}, type: unquote(vt), backend: b), Nx.tensor([[2], [0]], type: unquote(it), backend: b), axes: [0]) end, true)
      test "3d full batched index", do: check(fn b -> Nx.gather(Nx.iota({2, 3, 4}, type: unquote(vt), backend: b), Nx.tensor([[[0, 0, 0], [1, 2, 3]], [[0, 1, 2], [1, 0, 1]]], type: unquote(it), backend: b)) end, true)
    end
  end

  test "non-prefix axes ([1]) falls back but stays correct" do
    check(fn b -> Nx.gather(Nx.iota({3, 4}, type: {:f, 32}, backend: b), Nx.tensor([[1], [3]], type: {:s, 64}, backend: b), axes: [1]) end, false)
  end

  test "u8 value dtype falls back but stays correct" do
    check(fn b -> Nx.gather(Nx.iota({6}, type: {:u, 8}, backend: b), Nx.tensor([[0], [2]], type: {:s, 64}, backend: b)) end, false)
  end
end
