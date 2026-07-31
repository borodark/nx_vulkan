defmodule Nx.Vulkan.CastTest do
  @moduledoc """
  f32<->f64 dtype casts on the GPU (thrust 2). as_type between f32 and f64 now
  runs a cast shader instead of host round-tripping; other dtype pairs still
  fall back. Verified against BinaryBackend.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp maxdiff(a, b) do
    Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend) |> Nx.as_type({:f, 64}), Nx.backend_copy(b, Nx.BinaryBackend) |> Nx.as_type({:f, 64}))
    |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  @data [1.5, -2.25, 3.125, 4.0, -0.5, 100.0, -0.0001]

  test "f32 -> f64 on GPU, exact" do
    v = Nx.tensor(@data, type: {:f, 32}, backend: VulkanoBackend) |> Nx.as_type({:f, 64})
    b = Nx.tensor(@data, type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.as_type({:f, 64})
    assert match?(%VulkanoBackend{}, v.data)
    assert Nx.type(v) == {:f, 64}
    assert maxdiff(v, b) == 0.0
  end

  test "f64 -> f32 on GPU, exact" do
    v = Nx.tensor(@data, type: {:f, 64}, backend: VulkanoBackend) |> Nx.as_type({:f, 32})
    b = Nx.tensor(@data, type: {:f, 64}, backend: Nx.BinaryBackend) |> Nx.as_type({:f, 32})
    assert match?(%VulkanoBackend{}, v.data)
    assert Nx.type(v) == {:f, 32}
    assert maxdiff(v, b) == 0.0
  end

  test "non-f32/f64 cast still falls back but stays correct" do
    v = Nx.tensor([1.7, 2.3, 3.9], type: {:f, 32}, backend: VulkanoBackend) |> Nx.as_type({:s, 32})
    b = Nx.tensor([1.7, 2.3, 3.9], type: {:f, 32}, backend: Nx.BinaryBackend) |> Nx.as_type({:s, 32})
    assert Nx.to_flat_list(v) == Nx.to_flat_list(b)
  end
end
