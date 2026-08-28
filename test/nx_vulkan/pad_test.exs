defmodule Nx.Vulkan.PadTest do
  @moduledoc """
  GPU pad (thrust 2). A type-generic u32-word-copy shader maps each output
  element back through the per-axis {low, high, interior} config: edge pads,
  interior gaps and out-of-source positions get the pad value. Runs for
  4/8-byte dtypes (f32/f64/s32/s64), rank 1..4, scalar same-type pad value;
  sub-word dtypes and rank > 4 host-fall-back. Verified vs BinaryBackend.
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
    if on_gpu, do: assert(match?(%VulkanoBackend{}, got.data), "expected on-GPU pad")
    assert Nx.shape(got) == Nx.shape(ref)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  defp pv(type, b), do: Nx.tensor(0, type: type, backend: b)

  for type <- [{:f, 32}, {:f, 64}, {:s, 32}] do
    t = Macro.escape(type)

    describe "pad #{inspect(type)} on GPU" do
      test "1d low/high edge",
        do:
          check(
            fn b ->
              Nx.pad(Nx.iota({4}, type: unquote(t), backend: b), pv(unquote(t), b), [{2, 1, 0}])
            end,
            true
          )

      test "2d asymmetric",
        do:
          check(
            fn b ->
              Nx.pad(Nx.iota({2, 3}, type: unquote(t), backend: b), pv(unquote(t), b), [
                {1, 0, 0},
                {0, 2, 0}
              ])
            end,
            true
          )

      test "1d interior",
        do:
          check(
            fn b ->
              Nx.pad(Nx.iota({3}, type: unquote(t), backend: b), pv(unquote(t), b), [{0, 0, 2}])
            end,
            true
          )

      test "1d negative crop",
        do:
          check(
            fn b ->
              Nx.pad(Nx.iota({5}, type: unquote(t), backend: b), pv(unquote(t), b), [{-1, -1, 0}])
            end,
            true
          )

      test "3d edge + interior mix",
        do:
          check(
            fn b ->
              Nx.pad(Nx.iota({2, 2, 2}, type: unquote(t), backend: b), pv(unquote(t), b), [
                {1, 0, 0},
                {0, 1, 1},
                {0, 0, 0}
              ])
            end,
            true
          )

      test "nonzero pad value",
        do:
          check(
            fn b ->
              Nx.pad(
                Nx.iota({3}, type: unquote(t), backend: b),
                Nx.tensor(7, type: unquote(t), backend: b),
                [{1, 1, 0}]
              )
            end,
            true
          )
    end
  end

  @tag :host_fallback_expected
  test "u8 pad falls back but stays correct" do
    check(
      fn b ->
        Nx.pad(Nx.iota({3}, type: {:u, 8}, backend: b), Nx.tensor(0, type: {:u, 8}, backend: b), [
          {1, 1, 0}
        ])
      end,
      false
    )
  end
end
