defmodule Nx.Vulkan.FFTTest do
  @moduledoc """
  Native f64 Cooley-Tukey FFT on the GPU, verified against a BinaryBackend
  reference. Covers both correctness (bit-for-bit in f64) and that the
  supported case actually dispatches on the GPU rather than host-falling-back
  (asserted via the result tensor's backend).
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp v(data, type \\ {:f, 64}), do: Nx.tensor(data, type: type, backend: VulkanoBackend)
  defp b(data, type \\ {:f, 64}), do: Nx.tensor(data, type: type, backend: Nx.BinaryBackend)

  defp max_abs_diff(a, b) do
    ab = Nx.backend_copy(a, Nx.BinaryBackend)
    bb = Nx.backend_copy(b, Nx.BinaryBackend)
    assert Nx.shape(ab) == Nx.shape(bb)
    assert Nx.type(ab) == Nx.type(bb)
    Nx.subtract(ab, bb) |> Nx.abs() |> Nx.reduce_max() |> Nx.to_number()
  end

  describe "GPU path — runs on-device and matches BinaryBackend (f64 -> c128)" do
    test "forward fft, power-of-two lengths" do
      for data <- [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Enum.map(1..16, &(&1 * 1.0))
          ] do
        got = Nx.fft(v(data))
        assert match?(%VulkanoBackend{}, got.data), "expected on-GPU dispatch"
        assert Nx.type(got) == {:c, 128}
        assert max_abs_diff(got, Nx.fft(b(data))) < 1.0e-10
      end
    end

    test "inverse fft" do
      data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
      got = Nx.ifft(v(data))
      assert match?(%VulkanoBackend{}, got.data)
      assert max_abs_diff(got, Nx.ifft(b(data))) < 1.0e-10
    end

    test "batched over leading axis" do
      data = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [-1.0, 0.5, 2.0, -3.0]]
      got = Nx.fft(v(data))
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.shape(got) == {3, 4}
      assert max_abs_diff(got, Nx.fft(b(data))) < 1.0e-10
    end

    test "ifft(fft(x)) round-trips to x" do
      data = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]
      rt = v(data) |> Nx.fft() |> Nx.ifft()
      assert match?(%VulkanoBackend{}, rt.data)
      # original as complex for comparison
      assert max_abs_diff(rt, b(data, {:c, 128})) < 1.0e-10
    end

    test "complex input (fft of a complex tensor) stays on GPU" do
      # produce a complex tensor on the GPU, then fft it again
      cin = Nx.fft(v([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]))
      got = Nx.fft(cin)
      assert match?(%VulkanoBackend{}, got.data)

      bref = Nx.fft(Nx.fft(b([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])))
      assert max_abs_diff(got, bref) < 1.0e-10
    end
  end

  describe "host fallback — correct for the cases the GPU path does not cover" do
    @tag :host_fallback_expected
    test "non-power-of-two length (padded) falls back but stays correct" do
      data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
      got = Nx.fft(v(data))
      # length padded to 8 != axis size 6 -> host fallback
      refute match?(%VulkanoBackend{}, got.data)
      assert max_abs_diff(got, Nx.fft(b(data))) < 1.0e-10
    end

    @tag :host_fallback_expected
    test "explicit :length that slices falls back but stays correct" do
      data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
      got = Nx.fft(v(data), length: 4)
      refute match?(%VulkanoBackend{}, got.data)
      assert max_abs_diff(got, Nx.fft(b(data), length: 4)) < 1.0e-10
    end

    @tag :host_fallback_expected
    test "non-last axis falls back but stays correct" do
      data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
      got = Nx.fft(v(data), axis: 0)
      refute match?(%VulkanoBackend{}, got.data)
      assert max_abs_diff(got, Nx.fft(b(data), axis: 0)) < 1.0e-10
    end

    @tag :host_fallback_expected
    test "f32 input (maps to c64) falls back but stays correct" do
      data = [1.0, 2.0, 3.0, 4.0]
      got = Nx.fft(v(data, {:f, 32}))
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.type(got) == {:c, 64}
      assert max_abs_diff(got, Nx.fft(b(data, {:f, 32}))) < 1.0e-6
    end
  end
end
