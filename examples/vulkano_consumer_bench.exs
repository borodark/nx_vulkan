# Consumer-of-host-result bench. Tier 1 of SHAPE_C_PLAN.md skips the
# upload-back step in host-fallback callbacks — the result stays on
# BinaryBackend. The original op-only bench (vulkano_ops_bench.exs)
# doesn't see this benefit because it discards the result on every
# iteration. Real consumers (the eXMC trial's `signal_params` calling
# `Nx.to_flat_list` on a stored trace) DO see it: skipping the
# upload-back means skipping a subsequent download.
#
# This script measures four flows per op:
#
#   A. BinaryBackend.op alone                 — host-host baseline
#   B. BinaryBackend.op + to_flat_list        — host-host with consume
#   C. Vulkano.op alone                       — current Tier 1
#   D. Vulkano.op + to_flat_list              — Tier 1 in the real flow
#   E. Vulkano.op + transfer-back + to_flat_list  — pre-Tier-1 simulated
#
# D vs E is the headline: how much wall time did skipping the
# upload-back actually save when the consumer reads the result?
#
# Focuses on Shape C ops where Tier 1 was supposed to pay off:
# concatenate, pad, put_slice, indexed_put, broadcast, gather, take.

defmodule VulkanoConsumerBench do
  alias Nx.Vulkan.VulkanoBackend

  @sizes [1024, 16384, 262144, 1_048_576]
  @reps 30

  def main do
    {host, _} = System.cmd("hostname", ["-s"])
    host = String.trim(host)

    IO.puts("=== consumer-of-host-result bench ===")
    IO.puts("host: #{host}")
    IO.puts("flow legend:")
    IO.puts("  A bin only     | B bin+consume | C vulk only | D vulk+consume | E vulk+reupload+consume")
    IO.puts("")
    IO.puts("speedup column = bin_consume / vulk_consume   (D vs B)")
    IO.puts("                 vulk_consume / vulk_reup     (D vs E — the Tier 1 win)")
    IO.puts("")

    _ = Nx.iota({4}, type: :f32, backend: VulkanoBackend)

    Enum.each([:concatenate, :pad, :put_slice, :indexed_put, :broadcast, :gather, :take], fn op ->
      IO.puts("\n# #{op}")

      :io.format("~12s ~12s ~12s ~12s ~12s ~12s ~12s ~12s~n",
        ['size', 'A bin', 'B bin+rd', 'C vulk', 'D vk+rd', 'E vk+up+rd', 'D/B', 'D/E'])

      Enum.each(@sizes, fn n ->
        {a, b, c, d, e} = measure_op(op, n)
        d_vs_b = if b > 0, do: b / d, else: 0.0
        d_vs_e = if e > 0, do: e / d, else: 0.0

        :io.format("~12s ~10.1fus ~10.1fus ~10.1fus ~10.1fus ~10.1fus ~10.2fx ~10.2fx~n",
          [Integer.to_string(n), a, b, c, d, e, d_vs_b, d_vs_e])
      end)
    end)
  end

  defp measure_op(:concatenate, n) do
    a_b = mk_bin({n})
    b_b = mk_bin({n})
    a_v = Nx.backend_transfer(a_b, VulkanoBackend)
    b_v = Nx.backend_transfer(b_b, VulkanoBackend)

    a = time(fn -> Nx.concatenate([a_b, b_b]) end)
    b = time(fn -> Nx.concatenate([a_b, b_b]) |> Nx.to_flat_list() end)
    c = time(fn -> Nx.concatenate([a_v, b_v]) end)
    d = time(fn -> Nx.concatenate([a_v, b_v]) |> Nx.to_flat_list() end)
    e = time(fn ->
      Nx.concatenate([a_v, b_v])
      |> Nx.backend_transfer(VulkanoBackend)
      |> Nx.to_flat_list()
    end)

    {a, b, c, d, e}
  end

  defp measure_op(:pad, n) do
    # pad needs a 2D-ish shape; use {row, col} with col=8 (NUTS-ish)
    cols = 8
    rows = max(div(n, cols), 1)
    a_b = mk_bin({rows, cols})
    a_v = Nx.backend_transfer(a_b, VulkanoBackend)
    pv_b = Nx.tensor(0.0, type: :f32, backend: Nx.BinaryBackend)
    pv_v = Nx.backend_transfer(pv_b, VulkanoBackend)
    cfg = [{1, 1, 0}, {1, 1, 0}]

    a = time(fn -> Nx.pad(a_b, pv_b, cfg) end)
    b = time(fn -> Nx.pad(a_b, pv_b, cfg) |> Nx.to_flat_list() end)
    c = time(fn -> Nx.pad(a_v, pv_v, cfg) end)
    d = time(fn -> Nx.pad(a_v, pv_v, cfg) |> Nx.to_flat_list() end)
    e = time(fn ->
      Nx.pad(a_v, pv_v, cfg) |> Nx.backend_transfer(VulkanoBackend) |> Nx.to_flat_list()
    end)

    {a, b, c, d, e}
  end

  defp measure_op(:put_slice, n) do
    cols = 8
    rows = max(div(n, cols), 1)
    a_b = mk_bin({rows, cols})
    a_v = Nx.backend_transfer(a_b, VulkanoBackend)
    slice_b = mk_bin({1, cols})
    slice_v = Nx.backend_transfer(slice_b, VulkanoBackend)
    row_idx = div(rows, 2)

    a = time(fn -> Nx.put_slice(a_b, [row_idx, 0], slice_b) end)
    b = time(fn -> Nx.put_slice(a_b, [row_idx, 0], slice_b) |> Nx.to_flat_list() end)
    c = time(fn -> Nx.put_slice(a_v, [row_idx, 0], slice_v) end)
    d = time(fn -> Nx.put_slice(a_v, [row_idx, 0], slice_v) |> Nx.to_flat_list() end)
    e = time(fn ->
      Nx.put_slice(a_v, [row_idx, 0], slice_v) |> Nx.backend_transfer(VulkanoBackend) |> Nx.to_flat_list()
    end)

    {a, b, c, d, e}
  end

  defp measure_op(:indexed_put, n) do
    a_b = mk_bin({n})
    a_v = Nx.backend_transfer(a_b, VulkanoBackend)
    idx = Nx.tensor([[0], [div(n, 2)], [n - 1]], type: :s64, backend: Nx.BinaryBackend)
    upd = Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: Nx.BinaryBackend)

    a = time(fn -> Nx.indexed_put(a_b, idx, upd) end)
    b = time(fn -> Nx.indexed_put(a_b, idx, upd) |> Nx.to_flat_list() end)
    c = time(fn -> Nx.indexed_put(a_v, idx, upd) end)
    d = time(fn -> Nx.indexed_put(a_v, idx, upd) |> Nx.to_flat_list() end)
    e = time(fn ->
      Nx.indexed_put(a_v, idx, upd) |> Nx.backend_transfer(VulkanoBackend) |> Nx.to_flat_list()
    end)

    {a, b, c, d, e}
  end

  defp measure_op(:broadcast, n) do
    a_b = Nx.tensor(1.0, type: :f32, backend: Nx.BinaryBackend)
    a_v = Nx.backend_transfer(a_b, VulkanoBackend)

    a = time(fn -> Nx.broadcast(a_b, {n}) end)
    b = time(fn -> Nx.broadcast(a_b, {n}) |> Nx.to_flat_list() end)
    c = time(fn -> Nx.broadcast(a_v, {n}) end)
    d = time(fn -> Nx.broadcast(a_v, {n}) |> Nx.to_flat_list() end)
    e = time(fn ->
      Nx.broadcast(a_v, {n}) |> Nx.backend_transfer(VulkanoBackend) |> Nx.to_flat_list()
    end)

    {a, b, c, d, e}
  end

  defp measure_op(:gather, n) do
    a_b = mk_bin({n})
    a_v = Nx.backend_transfer(a_b, VulkanoBackend)
    n4 = div(n, 4)
    idx = Nx.tensor(for(i <- 0..(n4 - 1), do: [i * 4]), type: :s64, backend: Nx.BinaryBackend)

    a = time(fn -> Nx.gather(a_b, idx) end)
    b = time(fn -> Nx.gather(a_b, idx) |> Nx.to_flat_list() end)
    c = time(fn -> Nx.gather(a_v, idx) end)
    d = time(fn -> Nx.gather(a_v, idx) |> Nx.to_flat_list() end)
    e = time(fn ->
      Nx.gather(a_v, idx) |> Nx.backend_transfer(VulkanoBackend) |> Nx.to_flat_list()
    end)

    {a, b, c, d, e}
  end

  defp measure_op(:take, n) do
    # take needs a 2D source; reuse the put_slice shape
    cols = 8
    rows = max(div(n, cols), 1)
    a_b = mk_bin({rows, cols})
    a_v = Nx.backend_transfer(a_b, VulkanoBackend)
    idx = Nx.tensor(Enum.take_every(0..(rows - 1), 2), type: :s64, backend: Nx.BinaryBackend)

    a = time(fn -> Nx.take(a_b, idx, axis: 0) end)
    b = time(fn -> Nx.take(a_b, idx, axis: 0) |> Nx.to_flat_list() end)
    c = time(fn -> Nx.take(a_v, idx, axis: 0) end)
    d = time(fn -> Nx.take(a_v, idx, axis: 0) |> Nx.to_flat_list() end)
    e = time(fn ->
      Nx.take(a_v, idx, axis: 0) |> Nx.backend_transfer(VulkanoBackend) |> Nx.to_flat_list()
    end)

    {a, b, c, d, e}
  end

  defp mk_bin(shape) do
    n = shape |> Tuple.to_list() |> Enum.reduce(1, &*/2)

    Nx.iota({n}, type: :f32, backend: Nx.BinaryBackend)
    |> Nx.divide(Nx.tensor(n * 1.0))
    |> Nx.add(Nx.tensor(0.01))
    |> Nx.reshape(shape)
  end

  defp time(fun) do
    _ = fun.()
    _ = fun.()

    samples =
      for _ <- 1..@reps do
        {us, _} = :timer.tc(fun)
        us
      end
      |> Enum.sort()

    Enum.at(samples, div(@reps, 2)) * 1.0
  rescue
    _ -> -1.0
  end
end

VulkanoConsumerBench.main()
