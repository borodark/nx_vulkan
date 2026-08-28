defmodule Nx.Vulkan.SliceTest do
  @moduledoc """
  GPU strided slice (thrust 2). A type-generic u32-word-copy shader handles
  static-start slices of 4/8-byte dtypes (f32/f64/s32/s64) on the GPU; dynamic
  (tensor) starts, sub-word dtypes and rank > 4 host-fall-back. Verified vs
  BinaryBackend.
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
    if on_gpu, do: assert(match?(%VulkanoBackend{}, got.data), "expected on-GPU slice")
    assert Nx.shape(got) == Nx.shape(ref)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  for type <- [{:f, 32}, {:f, 64}, {:s, 32}] do
    describe "slice #{inspect(type)} on GPU" do
      test "1d contiguous",
        do:
          check(
            fn b ->
              Nx.slice(Nx.iota({8}, type: unquote(Macro.escape(type)), backend: b), [2], [4])
            end,
            true
          )

      test "2d block",
        do:
          check(
            fn b ->
              Nx.slice(Nx.iota({5, 6}, type: unquote(Macro.escape(type)), backend: b), [1, 2], [
                3,
                3
              ])
            end,
            true
          )

      test "1d strided",
        do:
          check(
            fn b ->
              Nx.slice(Nx.iota({10}, type: unquote(Macro.escape(type)), backend: b), [1], [6],
                strides: [2]
              )
            end,
            true
          )

      test "3d",
        do:
          check(
            fn b ->
              Nx.slice(
                Nx.iota({2, 3, 4}, type: unquote(Macro.escape(type)), backend: b),
                [0, 1, 1],
                [2, 2, 2]
              )
            end,
            true
          )
    end
  end

  @tag :host_fallback_expected
  test "u8 slice falls back but stays correct" do
    check(fn b -> Nx.slice(Nx.iota({6}, type: {:u, 8}, backend: b), [1], [4]) end, false)
  end

  @tag :host_fallback_expected
  test "dynamic (tensor) start falls back but stays correct" do
    check(
      fn b ->
        Nx.slice(Nx.iota({6}, type: {:f, 32}, backend: b), [Nx.tensor(2, backend: b)], [3])
      end,
      false
    )
  end
end
