# Comprehensive per-op bench for Nx.Vulkan.VulkanoBackend.
# Covers every callback the backend implements, across a ~10-shape grid
# per op. Compares VulkanoBackend vs Nx.BinaryBackend at each shape.
# Writes CSV to bench_results/<hostname>_<YYYY-MM-DD>.csv (one row per
# op×shape) so cross-machine, cross-time comparison is a diff away.
#
# Usage:
#   mix run examples/vulkano_ops_bench.exs                  # all ops
#   OP_FILTER=add mix run examples/vulkano_ops_bench.exs    # just add
#
# Output schema:
#   op_class, op_name, shape, n_reps, vulkano_us_median, vulkano_us_p95,
#   binary_us_median, binary_us_p95, speedup
#
# `speedup = binary_us_median / vulkano_us_median`. >1 means vulkano wins.

defmodule VulkanoOpsBench do
  @bench_results_dir Path.expand("../bench_results", __DIR__)

  # Shape grids
  @elementwise_sizes [256, 1024, 4096, 16384, 65536, 262144, 1_048_576]
  @reduction_sizes [256, 1024, 4096, 16384, 65536, 262144, 1_048_576]
  @matmul_sizes [16, 32, 64, 128, 256, 512, 1024]
  # Sampler-shape grid: (n_samples, d) — typical NUTS leapfrog scratch.
  @sampler_shapes [{16, 1}, {32, 4}, {64, 8}, {200, 8}, {200, 16}, {1024, 32}]
  # Pad / slice / put_slice work on 2D matrices; vary both dims.
  @matrix_shapes [{8, 8}, {64, 64}, {256, 256}, {1024, 1024}]

  @op_filter System.get_env("OP_FILTER")
  @reps_default 30
  @reps_heavy 8

  def main do
    {host, _} = System.cmd("hostname", ["-s"])
    host = String.trim(host)
    date = Date.utc_today() |> Date.to_iso8601()

    File.mkdir_p!(@bench_results_dir)
    csv_path = Path.join(@bench_results_dir, "#{host}_#{date}.csv")

    IO.puts("\n=== VulkanoBackend full-op bench ===")
    IO.puts("host:     #{host}")
    IO.puts("date:     #{date}")
    IO.puts("output:   #{csv_path}")
    IO.puts("filter:   #{@op_filter || "(all)"}\n")

    # Touch a vulkano tensor so the device init line prints up-front and
    # the first real bench doesn't pay init cost in its measurement.
    _ = Nx.iota({4}, type: :f32, backend: Nx.Vulkan.VulkanoBackend)

    rows = []

    rows = rows ++ bench_binary_ops()
    rows = rows ++ bench_unary_ops()
    rows = rows ++ bench_reduction_ops()
    rows = rows ++ bench_matmul()
    rows = rows ++ bench_storage_ops()
    rows = rows ++ bench_movement_ops()
    rows = rows ++ bench_comparison_ops()
    rows = rows ++ bench_misc_host_ops()
    rows = rows ++ bench_sampler_path_ops()

    write_csv(csv_path, rows)
    print_summary(rows)

    IO.puts("\nwrote #{length(rows)} rows -> #{csv_path}")
  end

  # ---- Binary SPV ops ----

  defp bench_binary_ops do
    ops = [:add, :multiply, :subtract, :divide, :pow, :max, :min]

    for op <- ops,
        filter_match?(op),
        n <- @elementwise_sizes do
      a_v = make_tensor_vulkano({n})
      b_v = make_tensor_vulkano({n})
      a_b = make_tensor_binary({n})
      b_b = make_tensor_binary({n})

      v = time(fn -> apply(Nx, op, [a_v, b_v]) end, @reps_default)
      b = time(fn -> apply(Nx, op, [a_b, b_b]) end, @reps_default)

      row("binary", to_string(op), inspect({n}), @reps_default, v, b)
    end
  end

  # ---- Unary SPV ops ----

  defp bench_unary_ops do
    ops = [:exp, :log, :sqrt, :abs, :negate, :sigmoid, :tanh, :floor, :ceil, :sign]

    for op <- ops,
        filter_match?(op),
        n <- @elementwise_sizes do
      a_v = make_tensor_vulkano({n}, positive: op in [:log, :sqrt])
      a_b = make_tensor_binary({n}, positive: op in [:log, :sqrt])

      v = time(fn -> apply(Nx, op, [a_v]) end, @reps_default)
      b = time(fn -> apply(Nx, op, [a_b]) end, @reps_default)

      row("unary", to_string(op), inspect({n}), @reps_default, v, b)
    end
  end

  # ---- Reductions ----

  defp bench_reduction_ops do
    ops = [:sum, :reduce_max, :reduce_min]

    for op <- ops,
        filter_match?(op),
        n <- @reduction_sizes do
      a_v = make_tensor_vulkano({n})
      a_b = make_tensor_binary({n})

      v = time(fn -> apply(Nx, op, [a_v]) end, @reps_default)
      b = time(fn -> apply(Nx, op, [a_b]) end, @reps_default)

      row("reduction", "#{op} all-axes", inspect({n}), @reps_default, v, b)
    end
  end

  # ---- Matmul (dot rank-2) ----

  defp bench_matmul do
    if filter_match?(:dot) or filter_match?(:matmul) do
      for m <- @matmul_sizes do
        reps = if m >= 256, do: @reps_heavy, else: @reps_default
        a_v = make_tensor_vulkano({m, m})
        b_v = make_tensor_vulkano({m, m})
        a_b = make_tensor_binary({m, m})
        b_b = make_tensor_binary({m, m})

        v = time(fn -> Nx.dot(a_v, b_v) end, reps)
        b =
          if m <= 256 do
            time(fn -> Nx.dot(a_b, b_b) end, reps)
          else
            # CPU matmul above 256x256 dominates total run time. Skip it
            # but keep a placeholder so the row exists.
            %{median_us: -1.0, p95_us: -1.0, n: 0}
          end

        row("linalg", "dot rank-2 f32", inspect({m, m}), reps, v, b)
      end
    else
      []
    end
  end

  # ---- Storage / creation ----

  defp bench_storage_ops do
    rows = []

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:from_binary) do
          bin = :binary.copy(<<0::float-32-native>>, n)
          v = time(fn -> Nx.from_binary(bin, :f32, backend: Nx.Vulkan.VulkanoBackend) end, @reps_default)
          b = time(fn -> Nx.from_binary(bin, :f32, backend: Nx.BinaryBackend) end, @reps_default)
          row("storage", "from_binary", inspect({n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:to_binary) do
          a_v = make_tensor_vulkano({n})
          a_b = make_tensor_binary({n})
          v = time(fn -> Nx.to_binary(a_v) end, @reps_default)
          b = time(fn -> Nx.to_binary(a_b) end, @reps_default)
          row("storage", "to_binary", inspect({n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:iota) do
          v = time(fn -> Nx.iota({n}, type: :f32, backend: Nx.Vulkan.VulkanoBackend) end, @reps_default)
          b = time(fn -> Nx.iota({n}, type: :f32, backend: Nx.BinaryBackend) end, @reps_default)
          row("storage", "iota", inspect({n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- [16, 64, 256, 1024], filter_match?(:eye) do
          v = time(fn -> Nx.eye(n, type: :f32, backend: Nx.Vulkan.VulkanoBackend) end, @reps_default)
          b = time(fn -> Nx.eye(n, type: :f32, backend: Nx.BinaryBackend) end, @reps_default)
          row("storage", "eye", inspect({n, n}), @reps_default, v, b)
        end

    rows
  end

  # ---- Movement (reshape, squeeze, transpose) ----

  defp bench_movement_ops do
    rows = []

    rows =
      rows ++
        for {m, n} <- @matrix_shapes, filter_match?(:transpose) do
          a_v = make_tensor_vulkano({m, n})
          a_b = make_tensor_binary({m, n})
          v = time(fn -> Nx.transpose(a_v) end, @reps_default)
          b = time(fn -> Nx.transpose(a_b) end, @reps_default)
          row("movement", "transpose 2D", inspect({m, n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:reshape) do
          a_v = make_tensor_vulkano({n})
          a_b = make_tensor_binary({n})
          # Reshape to a square-ish 2D shape.
          d = trunc(:math.sqrt(n))
          if d * d == n do
            v = time(fn -> Nx.reshape(a_v, {d, d}) end, @reps_default)
            b = time(fn -> Nx.reshape(a_b, {d, d}) end, @reps_default)
            row("movement", "reshape", "#{inspect({n})} -> #{inspect({d, d})}", @reps_default, v, b)
          end
        end
        |> Enum.reject(&is_nil/1)

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:squeeze) do
          a_v = make_tensor_vulkano({1, n})
          a_b = make_tensor_binary({1, n})
          v = time(fn -> Nx.squeeze(a_v, axes: [0]) end, @reps_default)
          b = time(fn -> Nx.squeeze(a_b, axes: [0]) end, @reps_default)
          row("movement", "squeeze", inspect({1, n}), @reps_default, v, b)
        end

    rows
  end

  # ---- Comparisons (all host fallback) ----

  defp bench_comparison_ops do
    ops = [:equal, :not_equal, :less, :less_equal, :greater, :greater_equal]

    for op <- ops,
        filter_match?(op),
        n <- @elementwise_sizes do
      a_v = make_tensor_vulkano({n})
      b_v = make_tensor_vulkano({n})
      a_b = make_tensor_binary({n})
      b_b = make_tensor_binary({n})
      v = time(fn -> apply(Nx, op, [a_v, b_v]) end, @reps_default)
      b = time(fn -> apply(Nx, op, [a_b, b_b]) end, @reps_default)
      row("compare", to_string(op), inspect({n}), @reps_default, v, b)
    end
  end

  # ---- Misc host ops ----

  defp bench_misc_host_ops do
    rows = []

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:select) do
          pred_v = Nx.greater(make_tensor_vulkano({n}), Nx.tensor(0.5, backend: Nx.Vulkan.VulkanoBackend))
          on_t_v = make_tensor_vulkano({n})
          on_f_v = make_tensor_vulkano({n})
          pred_b = Nx.greater(make_tensor_binary({n}), Nx.tensor(0.5))
          on_t_b = make_tensor_binary({n})
          on_f_b = make_tensor_binary({n})
          v = time(fn -> Nx.select(pred_v, on_t_v, on_f_v) end, @reps_default)
          b = time(fn -> Nx.select(pred_b, on_t_b, on_f_b) end, @reps_default)
          row("host", "select", inspect({n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- @elementwise_sizes, op <- [:all, :any], filter_match?(op) do
          a_v = Nx.greater(make_tensor_vulkano({n}), Nx.tensor(0.5, backend: Nx.Vulkan.VulkanoBackend))
          a_b = Nx.greater(make_tensor_binary({n}), Nx.tensor(0.5))
          v = time(fn -> apply(Nx, op, [a_v]) end, @reps_default)
          b = time(fn -> apply(Nx, op, [a_b]) end, @reps_default)
          row("host", to_string(op), inspect({n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for {m, n} <- @matrix_shapes, filter_match?(:slice) do
          a_v = make_tensor_vulkano({m, n})
          a_b = make_tensor_binary({m, n})
          # Slice the central quarter.
          m4 = max(div(m, 4), 1)
          n4 = max(div(n, 4), 1)
          v = time(fn -> Nx.slice(a_v, [m4, n4], [m4 * 2, n4 * 2]) end, @reps_default)
          b = time(fn -> Nx.slice(a_b, [m4, n4], [m4 * 2, n4 * 2]) end, @reps_default)
          row("host", "slice 2D centre", inspect({m, n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:as_type) do
          a_v = make_tensor_vulkano({n})
          a_b = make_tensor_binary({n})
          v = time(fn -> Nx.as_type(a_v, :s32) end, @reps_default)
          b = time(fn -> Nx.as_type(a_b, :s32) end, @reps_default)
          row("host", "as_type f32->s32", inspect({n}), @reps_default, v, b)
        end

    rows
  end

  # ---- Sampler-path ops (newly added host fallbacks) ----

  defp bench_sampler_path_ops do
    rows = []

    rows =
      rows ++
        for shape <- @matrix_shapes, filter_match?(:pad) do
          {m, n} = shape
          a_v = make_tensor_vulkano({m, n})
          a_b = make_tensor_binary({m, n})
          pv_v = Nx.tensor(0.0, type: :f32, backend: Nx.Vulkan.VulkanoBackend)
          pv_b = Nx.tensor(0.0, type: :f32)
          # Add 1 row + 1 col padding on each side, no interior pad.
          cfg = [{1, 1, 0}, {1, 1, 0}]
          v = time(fn -> Nx.pad(a_v, pv_v, cfg) end, @reps_default)
          b = time(fn -> Nx.pad(a_b, pv_b, cfg) end, @reps_default)
          row("sampler-host", "pad", inspect(shape), @reps_default, v, b)
        end

    rows =
      rows ++
        for shape <- @sampler_shapes, filter_match?(:put_slice) do
          {n_samples, d} = shape
          a_v = make_tensor_vulkano({n_samples, d})
          a_b = make_tensor_binary({n_samples, d})
          slice_v = make_tensor_vulkano({1, d})
          slice_b = make_tensor_binary({1, d})
          # Insert at a middle row.
          row_idx = div(n_samples, 2)
          v = time(fn -> Nx.put_slice(a_v, [row_idx, 0], slice_v) end, @reps_default)
          b = time(fn -> Nx.put_slice(a_b, [row_idx, 0], slice_b) end, @reps_default)
          row("sampler-host", "put_slice 1-row", inspect(shape), @reps_default, v, b)
        end

    rows =
      rows ++
        for shape <- @sampler_shapes, op <- [:indexed_put, :indexed_add], filter_match?(op) do
          {n_samples, d} = shape
          a_v = make_tensor_vulkano({n_samples * d})
          a_b = make_tensor_binary({n_samples * d})
          idx = Nx.tensor([[0], [div(n_samples * d, 2)], [n_samples * d - 1]],
                  type: :s64, backend: Nx.BinaryBackend)
          upd = Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: Nx.BinaryBackend)
          v = time(fn -> apply(Nx, op, [a_v, idx, upd]) end, @reps_default)
          b = time(fn -> apply(Nx, op, [a_b, idx, upd]) end, @reps_default)
          row("sampler-host", to_string(op), inspect(shape), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:broadcast) do
          a_v = Nx.tensor(1.0, type: :f32, backend: Nx.Vulkan.VulkanoBackend)
          a_b = Nx.tensor(1.0, type: :f32)
          v = time(fn -> Nx.broadcast(a_v, {n}) end, @reps_default)
          b = time(fn -> Nx.broadcast(a_b, {n}) end, @reps_default)
          row("sampler-host", "broadcast scalar->{n}", inspect({n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- @elementwise_sizes, filter_match?(:concatenate) do
          a_v = make_tensor_vulkano({n})
          b_v = make_tensor_vulkano({n})
          a_b = make_tensor_binary({n})
          b_b = make_tensor_binary({n})
          v = time(fn -> Nx.concatenate([a_v, b_v]) end, @reps_default)
          b = time(fn -> Nx.concatenate([a_b, b_b]) end, @reps_default)
          row("sampler-host", "concatenate 2", inspect({n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for n <- [256, 1024, 4096, 16384, 65536], filter_match?(:gather) do
          a_v = make_tensor_vulkano({n})
          a_b = make_tensor_binary({n})
          # Gather every 4th element.
          n4 = div(n, 4)
          idx_list = for i <- 0..(n4 - 1), do: [i * 4]
          idx = Nx.tensor(idx_list, type: :s64, backend: Nx.BinaryBackend)
          v = time(fn -> Nx.gather(a_v, idx) end, @reps_default)
          b = time(fn -> Nx.gather(a_b, idx) end, @reps_default)
          row("sampler-host", "gather every-4th", inspect({n}), @reps_default, v, b)
        end

    rows =
      rows ++
        for {m, n} <- @matrix_shapes, filter_match?(:take) do
          a_v = make_tensor_vulkano({m, n})
          a_b = make_tensor_binary({m, n})
          # Take every other row.
          idx_list = Enum.take_every(0..(m - 1), 2)
          idx = Nx.tensor(idx_list, type: :s64, backend: Nx.BinaryBackend)
          v = time(fn -> Nx.take(a_v, idx, axis: 0) end, @reps_default)
          b = time(fn -> Nx.take(a_b, idx, axis: 0) end, @reps_default)
          row("sampler-host", "take rows-stride-2", inspect({m, n}), @reps_default, v, b)
        end

    rows
  end

  # ---- helpers ----

  defp filter_match?(op) when @op_filter == nil, do: true
  defp filter_match?(op), do: String.contains?(to_string(op), @op_filter)

  defp make_tensor_vulkano(shape, opts \\ []) do
    base = make_tensor_binary(shape, opts)
    Nx.backend_transfer(base, Nx.Vulkan.VulkanoBackend)
  end

  defp make_tensor_binary(shape, opts \\ []) do
    n = shape |> Tuple.to_list() |> Enum.reduce(1, &*/2)
    base =
      Nx.iota({n}, type: :f32, backend: Nx.BinaryBackend)
      |> Nx.divide(Nx.tensor(n * 1.0))
      |> Nx.add(Nx.tensor(0.01))
      |> Nx.reshape(shape)

    if Keyword.get(opts, :positive, false) do
      Nx.add(Nx.abs(base), Nx.tensor(1.0e-3))
    else
      base
    end
  end

  defp time(fun, n_iter) do
    # Warmup x2 — first call may pay JIT / pipeline init.
    _ = fun.()
    _ = fun.()

    samples =
      for _ <- 1..n_iter do
        {us, _} = :timer.tc(fun)
        us
      end
      |> Enum.sort()

    %{
      median_us: Enum.at(samples, div(n_iter, 2)) * 1.0,
      p95_us: Enum.at(samples, min(round(n_iter * 0.95), n_iter - 1)) * 1.0,
      n: n_iter
    }
  rescue
    e ->
      IO.puts("  TIME FAILED: #{Exception.message(e)}")
      %{median_us: -1.0, p95_us: -1.0, n: 0}
  catch
    kind, reason ->
      IO.puts("  TIME CAUGHT #{kind}: #{inspect(reason)}")
      %{median_us: -1.0, p95_us: -1.0, n: 0}
  end

  defp row(class, op, shape, n_reps, vulkano, binary) do
    speedup =
      if vulkano.median_us > 0 and binary.median_us > 0 do
        binary.median_us / vulkano.median_us
      else
        0.0
      end

    pretty(class, op, shape, vulkano, binary, speedup)

    [class, op, shape, n_reps,
     vulkano.median_us, vulkano.p95_us,
     binary.median_us, binary.p95_us,
     speedup]
  end

  defp pretty(class, op, shape, v, b, speedup) do
    # Wider fields — Erlang's ~N.Df truncates with *** when the value
    # exceeds the field width. ~12.2f handles up to ~1e9 us cleanly.
    v_str = if v.median_us > 0, do: :io_lib.format("~12.2fus", [v.median_us]), else: "       failed"
    b_str = if b.median_us > 0, do: :io_lib.format("~12.2fus", [b.median_us]), else: " skipped/failed"
    sp_str = if speedup > 0, do: :io_lib.format("~9.2fx", [speedup]), else: "       -- "

    IO.puts(
      "  [#{String.pad_trailing(class, 12)}] #{String.pad_trailing(op, 28)} " <>
        "#{String.pad_trailing(shape, 22)} " <>
        "vulkano=#{IO.iodata_to_binary(v_str)}  " <>
        "binary=#{IO.iodata_to_binary(b_str)}  " <>
        "speedup=#{IO.iodata_to_binary(sp_str)}"
    )
  end

  defp write_csv(path, rows) do
    header = "op_class,op_name,shape,n_reps,vulkano_us_median,vulkano_us_p95,binary_us_median,binary_us_p95,speedup\n"

    body =
      rows
      |> Enum.map(fn [class, op, shape, n, v_med, v_p95, b_med, b_p95, sp] ->
        # Quote shape in case it contains commas.
        ~s(#{class},#{op},"#{shape}",#{n},#{f(v_med)},#{f(v_p95)},#{f(b_med)},#{f(b_p95)},#{f(sp)})
      end)
      |> Enum.join("\n")

    File.write!(path, header <> body <> "\n")
  end

  defp f(-1.0), do: ""
  defp f(0.0), do: "0"
  defp f(v) when is_float(v), do: :erlang.float_to_binary(v, decimals: 3)
  defp f(v), do: to_string(v)

  defp print_summary(rows) do
    IO.puts("\n=== summary ===")
    by_class = Enum.group_by(rows, fn [c | _] -> c end)

    Enum.each(by_class, fn {class, group} ->
      ratios = group |> Enum.map(&Enum.at(&1, 8)) |> Enum.filter(&(&1 > 0))
      n_wins = Enum.count(ratios, &(&1 >= 1.0))
      n_total = length(ratios)
      med = if n_total > 0, do: median_of(ratios), else: 0.0

      IO.puts(
        "  #{String.pad_trailing(class, 14)} " <>
          "#{n_wins}/#{n_total} vulkano wins, median speedup #{Float.round(med, 2)}x"
      )
    end)
  end

  defp median_of([]), do: 0.0
  defp median_of(list) do
    sorted = Enum.sort(list)
    Enum.at(sorted, div(length(sorted), 2))
  end
end

VulkanoOpsBench.main()
