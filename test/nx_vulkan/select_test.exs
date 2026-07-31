defmodule Nx.Vulkan.SelectTest do
  @moduledoc """
  GPU broadcast select (thrust 2 — masking / where / relu-grad). When pred is u8
  and on_true/on_false/out share an f32/f64 type, select runs a shader instead of
  host round-tripping. Verified vs BinaryBackend.
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

  defp check(build) do
    got = build.(VulkanoBackend)
    ref = build.(Nx.BinaryBackend)
    assert match?(%VulkanoBackend{}, got.data), "expected on-GPU select"
    assert maxdiff(got, ref) == 0.0
  end

  for type <- [{:f, 32}, {:f, 64}] do
    describe "select #{inspect(type)}" do
      @tag type: type
      test "relu-grad select(x>0, dy, 0)" do
        check(fn b ->
          x = Nx.subtract(Nx.iota({4, 5}, type: unquote(Macro.escape(type)), backend: b), 10.0)
          dy = Nx.add(Nx.iota({4, 5}, type: unquote(Macro.escape(type)), backend: b), 1.0)
          Nx.select(Nx.greater(x, 0.0), dy, Nx.tensor(0.0, type: unquote(Macro.escape(type)), backend: b))
        end)
      end

      test "boolean mask, both branches full" do
        check(fn b ->
          m = Nx.tensor([1, 0, 1, 1, 0, 1], type: {:u, 8}, backend: b)
          Nx.select(m, Nx.iota({6}, type: unquote(Macro.escape(type)), backend: b), Nx.negate(Nx.iota({6}, type: unquote(Macro.escape(type)), backend: b)))
        end)
      end

      test "scalar on_false broadcast" do
        check(fn b ->
          m = Nx.greater(Nx.iota({3, 4}, type: unquote(Macro.escape(type)), backend: b), 5.0)
          Nx.select(m, Nx.iota({3, 4}, type: unquote(Macro.escape(type)), backend: b), Nx.tensor(-1.0, type: unquote(Macro.escape(type)), backend: b))
        end)
      end
    end
  end

  test "mixed-type branches fall back but stay correct" do
    m = Nx.tensor([1, 0, 1], type: {:u, 8}, backend: VulkanoBackend)
    got = Nx.select(m, Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}, backend: VulkanoBackend), Nx.tensor([9, 9, 9], type: {:s, 64}, backend: VulkanoBackend))
    ref = Nx.select(Nx.backend_copy(m, Nx.BinaryBackend), Nx.tensor([1.0, 2.0, 3.0], type: {:f, 32}, backend: Nx.BinaryBackend), Nx.tensor([9, 9, 9], type: {:s, 64}, backend: Nx.BinaryBackend))
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end
end
