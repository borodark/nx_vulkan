defmodule Nx.Vulkan.ParityTest do
  @moduledoc """
  Cross-backend parity suite for `Nx.Vulkan.VulkanoBackend`.

  Two modes:

  - **Parity mode** (when EXLA is available): runs each fixture on EXLA
    as reference and on VulkanoBackend, compares within tolerance.
  - **Cross-host mode** (Vulkano-only): runs each fixture on VulkanoBackend,
    writes results to JSON. Diff across hosts catches non-determinism.

  Run modes:

      # full parity check (super-io with EXLA)
      mix test test/nx_vulkan/parity_test.exs --only parity

      # vulkano-only (mac-247, mac-248 — no EXLA on FreeBSD)
      mix test test/nx_vulkan/parity_test.exs --only vulkano_only

  Output:
      /tmp/parity_report_<hostname>_<commit>.json

  See `docs/NX_PARITY_RESEARCH.md` for the full methodology + per-callback
  expectation matrix.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  @moduletag :parity

  # Tolerance per dtype — values larger than these fail the parity check.
  @tolerance %{
    {:f, 64} => 1.0e-10,
    {:f, 32} => 1.0e-6,
    {:f, 16} => 1.0e-3,
    {:s, 64} => 0,
    {:s, 32} => 0,
    {:s, 16} => 0,
    {:s, 8} => 0,
    {:u, 64} => 0,
    {:u, 32} => 0,
    {:u, 16} => 0,
    {:u, 8} => 0
  }

  # --- Fixtures: per-callback test specifications. ---
  #
  # Each entry maps a callback name to a list of test cases. Each case is
  # `%{name: string, inputs: [tensor_spec], opts: [], expected_status: atom}`.
  #
  # tensor_spec is `{shape, dtype, seed}` — fixtures are generated
  # deterministically from the seed.
  #
  # expected_status one of:
  #   :pass — should pass on both EXLA and VulkanoBackend (within tolerance)
  #   :pass_host_fallback — VulkanoBackend falls back to BinaryBackend
  #   :skip_unimplemented — VulkanoBackend doesn't implement; should raise
  #   :skip — intentionally out of scope (fft, conv, etc)
  #
  # This list grows as Tier 1 implementations land. Start with a small
  # smoke-set covering already-implemented ops + the first batch (reduction
  # family) that lands this sprint.

  @fixtures [
    # --- Already-implemented (smoke validation that suite + tolerance work) ---
    %{
      callback: :add,
      cases: [
        %{
          name: "elementwise f32 vector",
          inputs: [{{4}, {:f, 32}, 42}, {{4}, {:f, 32}, 7}],
          opts: [],
          expected_status: :pass
        }
      ]
    },
    %{
      callback: :sum,
      cases: [
        %{
          name: "axis-0 reduction on 3x4 f64",
          inputs: [{{3, 4}, {:f, 64}, 42}],
          opts: [axes: [0]],
          expected_status: :pass
        }
      ]
    },
    %{
      callback: :dot,
      cases: [
        %{
          name: "8x8 matmul f32",
          inputs: [{{8, 8}, {:f, 32}, 42}, {{8, 8}, {:f, 32}, 7}],
          opts: [],
          expected_status: :pass
        },
        %{
          name: "8x8 matmul f64",
          inputs: [{{8, 8}, {:f, 64}, 42}, {{8, 8}, {:f, 64}, 7}],
          opts: [],
          expected_status: :pass
        },
        %{
          name: "16x32 @ 32x8 matmul f64",
          inputs: [{{16, 32}, {:f, 64}, 99}, {{32, 8}, {:f, 64}, 13}],
          opts: [],
          expected_status: :pass
        }
      ]
    },

    # --- Tier 1 reduction family (lands next batch) ---
    %{
      callback: :all,
      cases: [
        %{
          name: "boolean reduce-AND",
          inputs: [{{10}, {:u, 8}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :any,
      cases: [
        %{
          name: "boolean reduce-OR",
          inputs: [{{10}, {:u, 8}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :all_close,
      cases: [
        %{
          name: "two equal f32 vectors",
          inputs: [{{8}, {:f, 32}, 42}, {{8}, {:f, 32}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :product,
      cases: [
        %{
          name: "multiplicative reduction",
          inputs: [{{4}, {:f, 32}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :cumulative_sum,
      cases: [
        %{
          name: "cumulative sum f64",
          inputs: [{{8}, {:f, 64}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },

    # --- Tier 1 round 1 (2026-05-26) — explicit host-fallback impls ---
    %{
      callback: :cumulative_max,
      cases: [
        %{
          name: "cumulative max f64",
          inputs: [{{8}, {:f, 64}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :cumulative_min,
      cases: [
        %{
          name: "cumulative min f64",
          inputs: [{{8}, {:f, 64}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :cumulative_product,
      cases: [
        %{
          name: "cumulative product f32",
          inputs: [{{6}, {:f, 32}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :logical_not,
      cases: [
        %{
          name: "boolean negation",
          inputs: [{{8}, {:u, 8}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },

    # --- Sort family (composes today; promote to pass after explicit impl lands) ---
    %{
      callback: :sort,
      cases: [
        %{
          name: "sort 100 f32 values",
          inputs: [{{100}, {:f, 32}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :argsort,
      cases: [
        %{
          name: "argsort 100 f32 values",
          inputs: [{{100}, {:f, 32}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },
    %{
      callback: :reverse,
      cases: [
        %{
          name: "reverse along axis 0",
          inputs: [{{10}, {:s, 32}, 42}],
          opts: [axes: [0]],
          expected_status: :pass_host_fallback
        }
      ]
    },

    # --- Round 2 fixtures: linalg (via Nx.LinAlg module) ---
    %{
      callback: :determinant,
      module: Nx.LinAlg,
      cases: [
        %{
          name: "determinant 4x4 f64",
          inputs: [{{4, 4}, {:f, 64}, 42}],
          opts: [],
          expected_status: :pass_host_fallback
        }
      ]
    },

    # --- Intentionally out-of-scope (skip set) ---
    %{
      callback: :fft,
      cases: [
        %{
          name: "fft skip",
          inputs: [{{64}, {:f, 32}, 42}],
          opts: [length: 64],
          expected_status: :skip
        }
      ]
    },
    %{
      callback: :conv,
      cases: [
        %{
          name: "conv skip",
          inputs: [{{1, 1, 8, 8}, {:f, 32}, 42}, {{1, 1, 3, 3}, {:f, 32}, 7}],
          opts: [],
          expected_status: :skip
        }
      ]
    }
  ]

  # --- Test runner ---

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  describe "parity check" do
    @tag :vulkano_only
    test "vulkano: run all non-skip fixtures, write JSON report" do
      results = Enum.flat_map(@fixtures, fn entry ->
        module = Map.get(entry, :module, Nx)
        Enum.map(entry.cases, fn case_spec ->
          run_vulkano_case(module, entry.callback, case_spec)
        end)
      end)

      report = build_report("vulkano-only", results)
      out_path = report_path("vulkano-only")
      File.write!(out_path, Jason.encode!(report, pretty: true))
      IO.puts("wrote #{out_path}")

      # Pass if no regression on EXPECTED-PASS callbacks. Failures on
      # :pass_host_fallback (Tier 1 todo) are tracked-but-not-blocking
      # — they become :pass_host_fallback when the impl lands and the
      # parity score climbs.
      regressions =
        Enum.filter(results, fn r ->
          r.actual_status == :fail and r.expected_status == :pass
        end)

      assert regressions == [],
        "regressions on expected-pass callbacks: #{inspect(Enum.map(regressions, & &1.callback))}"
    end

    @tag :parity
    test "parity: run all non-skip fixtures against EXLA reference (super-io only)" do
      if exla_available?() do
        results = Enum.flat_map(@fixtures, fn entry ->
          module = Map.get(entry, :module, Nx)
          Enum.map(entry.cases, fn case_spec ->
            run_parity_case(module, entry.callback, case_spec)
          end)
        end)

        report = build_report("parity-vs-exla", results)
        out_path = report_path("parity-vs-exla")
        File.write!(out_path, Jason.encode!(report, pretty: true))
        IO.puts("wrote #{out_path}")

        failed = Enum.filter(results, fn r -> r.actual_status == :fail end)

        assert failed == [],
          "parity failures: #{inspect(Enum.map(failed, & &1.callback))}"
      else
        IO.puts("EXLA not available — skipping parity-vs-exla")
      end
    end
  end

  # --- Helpers ---

  defp run_vulkano_case(module, callback, case_spec) do
    inputs = Enum.map(case_spec.inputs, &generate_tensor(&1, VulkanoBackend))
    apply_and_capture(module, callback, inputs, case_spec)
  end

  defp run_parity_case(module, callback, case_spec) do
    inputs_exla = Enum.map(case_spec.inputs, &generate_tensor(&1, EXLA.Backend))
    inputs_vulk = Enum.map(case_spec.inputs, &generate_tensor(&1, VulkanoBackend))

    {exla_status, exla_result} = safe_apply(module, callback, inputs_exla, case_spec.opts)
    {vulk_status, vulk_result} = safe_apply(module, callback, inputs_vulk, case_spec.opts)

    actual_status =
      cond do
        case_spec.expected_status in [:skip, :skip_unimplemented] ->
          :skipped
        exla_status == :ok and vulk_status == :ok ->
          if within_tolerance?(exla_result, vulk_result), do: :ok, else: :fail
        true ->
          :error
      end

    %{
      callback: callback,
      case: case_spec.name,
      expected_status: case_spec.expected_status,
      actual_status: actual_status,
      exla_status: exla_status,
      vulkano_status: vulk_status,
      max_err: if(actual_status == :ok, do: max_abs_diff(exla_result, vulk_result), else: nil)
    }
  end

  defp apply_and_capture(module, callback, inputs, case_spec) do
    {status, result} = safe_apply(module, callback, inputs, case_spec.opts)

    actual_status =
      cond do
        case_spec.expected_status == :skip -> :skipped
        case_spec.expected_status == :skip_unimplemented and status == :error -> :ok
        status == :ok -> :ok
        true -> :fail
      end

    %{
      callback: callback,
      case: case_spec.name,
      expected_status: case_spec.expected_status,
      actual_status: actual_status,
      result_summary: if(status == :ok, do: summarize(result), else: inspect(result, limit: 2))
    }
  end

  defp safe_apply(module, callback, inputs, opts) do
    try do
      result = apply(module, callback, inputs ++ wrap_opts(opts))
      {:ok, result}
    rescue
      e -> {:error, Exception.message(e)}
    end
  end

  defp wrap_opts([]), do: []
  defp wrap_opts(opts), do: [opts]

  defp generate_tensor({shape, type, seed}, backend) do
    :rand.seed(:exsss, {seed, seed + 1, seed + 2})

    size = Tuple.to_list(shape) |> Enum.reduce(1, &(&1 * &2))

    values =
      case type do
        {:f, _} -> for _ <- 1..size, do: :rand.normal()
        {:s, _} -> for _ <- 1..size, do: :rand.uniform(100) - 50
        {:u, _} -> for _ <- 1..size, do: :rand.uniform(2) - 1
      end

    values
    |> Nx.tensor(type: type, backend: Nx.BinaryBackend)
    |> Nx.reshape(shape)
    |> Nx.backend_transfer(backend)
  end

  defp within_tolerance?(a, b) do
    tol = Map.get(@tolerance, Nx.type(a), 1.0e-6)

    if Nx.shape(a) != Nx.shape(b), do: false,
    else: max_abs_diff(a, b) <= tol
  end

  defp max_abs_diff(a, b) do
    a_bin = Nx.backend_copy(a, Nx.BinaryBackend)
    b_bin = Nx.backend_copy(b, Nx.BinaryBackend)

    Nx.subtract(a_bin, b_bin)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  defp summarize(%Nx.Tensor{} = t) do
    %{
      shape: Nx.shape(t) |> Tuple.to_list(),
      type: Nx.type(t) |> Tuple.to_list() |> Enum.map(&inspect/1) |> Enum.join("/"),
      backend: inspect(t.data.__struct__)
    }
  end

  defp summarize(other), do: inspect(other, limit: 2)

  defp build_report(mode, results) do
    {:ok, hostname} = :inet.gethostname()
    hostname = to_string(hostname)

    pass = Enum.count(results, &(&1.actual_status == :ok))
    fail = Enum.count(results, &(&1.actual_status == :fail))
    skipped = Enum.count(results, &(&1.actual_status == :skipped))
    error = Enum.count(results, &(&1.actual_status == :error))
    total_runnable = pass + fail + error

    %{
      hostname: hostname,
      mode: mode,
      timestamp: DateTime.utc_now() |> DateTime.to_iso8601(),
      vulkano_commit: vulkano_commit(),
      device: device_info(),
      summary: %{
        total: length(results),
        pass: pass,
        fail: fail,
        skipped: skipped,
        error: error,
        parity_score: if(total_runnable > 0, do: pass / total_runnable, else: 0.0)
      },
      results: results
    }
  end

  defp report_path(mode) do
    {:ok, hostname} = :inet.gethostname()
    "/tmp/parity_report_#{hostname}_#{mode}.json"
  end

  defp vulkano_commit do
    case System.cmd("git", ["-C", Application.app_dir(:nx_vulkan, ".."), "log", "-1", "--format=%h"], stderr_to_stdout: true) do
      {sha, 0} -> String.trim(sha)
      _ -> "unknown"
    end
  end

  defp device_info do
    if Code.ensure_loaded?(Nx.Vulkan.NativeV) do
      "vulkano (via NativeV)"
    else
      "unknown"
    end
  end

  defp exla_available?, do: Code.ensure_loaded?(EXLA.Backend)
end
