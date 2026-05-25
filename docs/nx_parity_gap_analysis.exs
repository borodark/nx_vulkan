# Parity gap analysis: enumerate Nx.Backend @callbacks, diff against
# VulkanoBackend implementations, categorize missing callbacks by op
# family + complexity + EXMC usage. Writes a structured CSV for the
# research doc.

defmodule ParityGap do
  @callbacks_file "/home/io/projects/learn_erl/nx_vulkan/deps/nx/lib/nx/backend.ex"
  @vulkano_file "/home/io/projects/learn_erl/nx_vulkan/lib/nx_vulkan/vulkano_backend.ex"
  @exmc_lib_dir "/home/io/projects/learn_erl/pymc/exmc/lib"
  @output "/tmp/parity_gap.csv"

  # Op family categorization (used for prioritization)
  @families %{
    # linalg
    cholesky: "linalg",
    determinant: "linalg",
    eigh: "linalg",
    lu: "linalg",
    qr: "linalg",
    svd: "linalg",
    solve: "linalg",
    triangular_solve: "linalg",
    # fft / spectral
    fft: "fft",
    fft2: "fft",
    ifft: "fft",
    # convolution
    conv: "conv",
    # reduction
    all: "reduction",
    all_close: "reduction",
    any: "reduction",
    product: "reduction",
    reduce: "reduction",
    cumulative_sum: "reduction-cumulative",
    cumulative_max: "reduction-cumulative",
    cumulative_min: "reduction-cumulative",
    cumulative_product: "reduction-cumulative",
    # sort / order
    argsort: "sort",
    sort: "sort",
    top_k: "sort",
    take_along_axis: "sort",
    # shape / move
    reverse: "shape",
    to_batched: "shape",
    bitcast: "shape",
    # window
    window_max: "window",
    window_min: "window",
    window_product: "window",
    window_reduce: "window",
    window_sum: "window",
    window_scatter_max: "window",
    window_scatter_min: "window",
    # pointer / system
    from_pointer: "pointer",
    to_pointer: "pointer",
    optional: "system",
    # logic / phase
    logical_not: "logic",
    phase: "complex"
  }

  # Implementation complexity estimate
  @complexity %{
    "linalg" => "medium (host fallback via BinaryBackend; GPU shader = hard)",
    "fft" => "skip (out of exmc scope; GPU shader = very hard)",
    "conv" => "skip (out of exmc scope; CNN territory)",
    "reduction" => "easy (host fallback via Nx.<op>)",
    "reduction-cumulative" => "easy-medium (host fallback fine; GPU scan possible)",
    "sort" => "medium (host fallback; GPU sort = hard but doable)",
    "shape" => "easy (host fallback, mostly free)",
    "window" => "medium (host fallback; rarely needed)",
    "pointer" => "skip (FFI surface; not needed for exmc)",
    "system" => "n/a (optional callback, no impl needed)",
    "logic" => "easy (host fallback)",
    "complex" => "skip (no complex tensors in exmc)"
  }

  def run do
    callbacks_text = File.read!(@callbacks_file)
    vulkano_text = File.read!(@vulkano_file)

    callbacks =
      Regex.scan(~r/^  @callback ([a-z_]+)\(([^)]*)\)/m, callbacks_text)
      |> Enum.map(fn [_, name, args] -> {name, String.trim(args)} end)
      |> Enum.uniq()

    vulkano_impls =
      Regex.scan(~r/^  def ([a-z_]+)\(/m, vulkano_text)
      |> Enum.map(fn [_, name] -> name end)
      |> MapSet.new()

    missing =
      callbacks
      |> Enum.reject(fn {name, _} -> MapSet.member?(vulkano_impls, name) end)
      |> Enum.sort_by(fn {name, _} -> name end)

    IO.puts("Total Nx.Backend @callbacks: #{length(callbacks)}")
    IO.puts("VulkanoBackend implementations: #{MapSet.size(vulkano_impls)}")
    IO.puts("Missing callbacks: #{length(missing)}")
    IO.puts("")

    # Tally exmc usage for each missing op
    rows =
      Enum.map(missing, fn {name, args} ->
        family = Map.get(@families, String.to_atom(name), "uncategorized")
        complexity = Map.get(@complexity, family, "n/a")
        exmc_uses = count_exmc_uses(name)
        {name, args, family, complexity, exmc_uses}
      end)

    # Write CSV
    header = "callback,args,family,complexity,exmc_usage_count\n"
    csv =
      Enum.map_join(rows, "\n", fn {n, a, f, c, u} ->
        ~s|#{n},"#{a}",#{f},"#{c}",#{u}|
      end)

    File.write!(@output, header <> csv <> "\n")
    IO.puts("wrote #{@output} (#{length(rows)} rows)")

    # Print summary by family
    IO.puts("\n=== by family (callback count + total EXMC usage) ===")
    rows
    |> Enum.group_by(fn {_, _, f, _, _} -> f end)
    |> Enum.map(fn {f, fs} ->
      total_uses = Enum.map(fs, fn {_, _, _, _, u} -> u end) |> Enum.sum()
      complexity = Map.get(@complexity, f, "n/a")
      {f, length(fs), total_uses, complexity}
    end)
    |> Enum.sort_by(fn {_, _, uses, _} -> -uses end)
    |> Enum.each(fn {f, count, uses, c} ->
      IO.puts("  #{String.pad_trailing(f, 22)} #{String.pad_leading("#{count}", 3)} cb  #{String.pad_leading("#{uses}", 3)} uses  — #{c}")
    end)

    # Show top-10 by EXMC usage
    IO.puts("\n=== top 10 missing callbacks by EXMC usage ===")
    rows
    |> Enum.sort_by(fn {_, _, _, _, u} -> -u end)
    |> Enum.take(10)
    |> Enum.each(fn {n, _, f, _, u} ->
      IO.puts("  #{String.pad_leading("#{u}", 4)} uses — #{String.pad_trailing(n, 24)} (#{f})")
    end)
  end

  defp count_exmc_uses(callback_name) do
    # Find Nx.<callback> calls in exmc lib/
    pattern = "Nx\\.#{callback_name}\\b"

    case System.cmd("grep", ["-rE", "--include=*.ex", pattern, @exmc_lib_dir], stderr_to_stdout: true) do
      {output, 0} -> output |> String.split("\n", trim: true) |> length()
      {_, _} -> 0
    end
  end
end

ParityGap.run()
