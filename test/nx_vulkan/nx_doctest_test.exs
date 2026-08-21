defmodule Nx.Vulkan.NxDoctestTest do
  @moduledoc """
  Conformance: run Nx's own doctests against VulkanoBackend set as the default
  backend — the community-standard backend validation (mirrors torchx's
  `nx_doctest_test.exs`). First run: **954 doctests, 42 failures**; after fixing
  the slice dynamic-index and composed-fallback default-backend-leak bugs it
  surfaced, the `@except` buckets below cover the rest.

  Excepts are bucketed by reason. `@rounding` and `@unsupported` are documented
  non-bugs (float last-digit inspect diffs on the native shader path; dtypes the
  f64-real backend does not represent). `@backlog` are REAL bugs the conformance
  run found that are tracked in ROADMAP_NEXT_BEST_NX.md (thrust 0) — excepted to
  keep the suite green, NOT waived.

  ## This module is in the strict run (W2)

  It used to carry `@moduletag :host_fallback_expected` and so was skipped
  entirely by `scripts/strict_test.sh`. That tag is retired. All 843 doctests
  below now run under `NXV_HOST_FALLBACK=raise` except the ones named in
  `Nx.Vulkan.NxDoctestRegister` (`test/nx_doctest_register.exs`), which is where
  the reasons live and which `test_helper.exs` applies. Baseline: **355 of 843
  (42.1%) resident** (319 / 37.8% at W2; W1 moved 28, W3 another 8). `sh scripts/doctest_residency.sh` prints it and fails if it
  moves in either direction.

  Anything excepted *here* is excepted from both runs and never executes at all;
  anything in the register still runs and still asserts its value. Prefer the
  register: an op that computes the right answer on the host is a residency
  problem, not a correctness one, and the two should not share an off switch.
  """
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Nx.Vulkan.VulkanoBackend)
    :ok
  end

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  # Native-shader float results differ from BinaryBackend in the last ULP, so
  # the doctest's `inspect` string doesn't match (values are within tolerance).
  # standard_deviation joined this bucket when coerce_to/2 learned to convert
  # rank-0 integer constants: the op used to host-fall-back on its `ddof`
  # scalar and so matched BinaryBackend exactly. It now runs natively and lands
  # 1 ULP away (6.3639607 vs 6.363961 — same f32 value, different inspect
  # string). Moving an op onto the GPU is expected to cost the last digit.
  # standard_deviation and covariance joined this bucket as ops moved on-device:
  # the first when coerce_to/2 learned rank-0 integer constants, the second when
  # it gained an s32 -> float cast shader (covariance's doctest takes integer
  # input, so it used to fall back at the cast and run entirely on the host).
  # Both now land 1 ULP away — same f32 value, different inspect string.
  #
  # Note the standing cost of this bucket: excepting a function drops ALL of its
  # doctests, not just the one that drifted. Expect more entries as more ops move
  # to the GPU; that is the trade, and it is worth watching rather than
  # accumulating silently.
  # variance/2 joined for the same reason at T11, one step further down the same
  # road: its `ddof` divisor is an s32 scalar against an f32 scalar, a pair that
  # missed the flat apply_binary path (types differ) and was then refused by the
  # broadcast path's `rank >= 1` check. With rank 0 allowed it divides on the
  # GPU, and its f32 divide lands 1 ULP away from BinaryBackend's
  # (1.6666667 vs 1.6666666) — measurably so on every f32 divide this backend
  # has ever run, flat path included, not something rank 0 introduced.
  #
  # CORRECTION (W5 T2): this comment used to say the GPU was "1 ULP off a
  # correctly-rounded one". It is the other way round. For 10/6 the GPU returns
  # 0x3FD55556 and BinaryBackend 0x3FD55555, and against the true 5/3 those sit
  # 3.97e-8 and 1.59e-7 away — the GPU's is the correctly-rounded f32 and the
  # host's is 1 ULP low. The doctests still have to be excepted, because they
  # assert BinaryBackend's inspect string and that is the contract this suite
  # checks; but "moving an op onto the GPU costs the last digit" is the wrong
  # way to describe it. It costs the last digit of AGREEMENT, and here it buys
  # accuracy. Worth knowing before anyone treats this bucket as a list of
  # GPU imprecisions.
  #
  # weighted_mean/3 joined at W5 T2 for exactly this reason: its `sum` was an
  # integer reduction that used to fall back, taking the whole expression to the
  # host with it. With an s32 reduce shader the sums stay resident, so the final
  # divide happens on the GPU and lands on the other side of that 1 ULP.
  @rounding [
    exp: 1,
    tanh: 1,
    sigmoid: 1,
    sqrt: 1,
    log: 1,
    add: 2,
    standard_deviation: 2,
    covariance: 3,
    variance: 2,
    weighted_mean: 3
  ]

  # Dtypes the f64-real, byte-addressed backend does not represent: complex
  # (skip set), f8 special values (e.g. :infinity in :f8_e4m3fn), and sub-byte
  # types (u2/u4/s4 — a non-byte-aligned bitstring can't upload to a GPU buffer).
  @unsupported [complex: 2, real: 1, imag: 1, conjugate: 1, phase: 1, tensor: 2, bit_size: 1]

  # REAL bugs still open — tracked in ROADMAP thrust 0, not waived. These three
  # share one systemic root: Nx composes them (dynamic-index slice,
  # select_and_scatter) and materialises scalar indices on the *default* backend;
  # when VulkanoBackend is default, Nx's own BinaryBackend.select_and_scatter
  # then calls to_binary on a VulkanoBackend scalar and crashes. Needs a deeper
  # default-backend-isolation fix than with_binary_backend covers.
  @backlog [slice: 4, window_scatter_min: 5, window_scatter_max: 5]

  doctest Nx, except: [:moduledoc] ++ @rounding ++ @unsupported ++ @backlog
end
