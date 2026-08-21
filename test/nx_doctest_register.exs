defmodule Nx.Vulkan.NxDoctestRegister do
  @moduledoc """
  The residency register for `Nx.Vulkan.NxDoctestTest` — every one of Nx's own
  doctests that still leaves the GPU, named, with the reason it does.

  ## Why this exists

  `nx_doctest_test.exs` used to carry `@moduletag :host_fallback_expected`,
  which took it out of the strict run wholesale. That was one line standing in
  for 843 unmeasured decisions: nothing distinguished a doctest that runs
  entirely on the GPU from one that quietly computed on `Nx.BinaryBackend`, and
  a host fallback returns a **bit-identical** result, so no assertion in that
  file could ever tell them apart. The tag is retired (W2). This register
  replaces it, one line per op, bucketed by reason.

  Measured on `main` @ W1, mac-247 / GT 650M — and confirmed identical on
  super-io / RTX 3060 Ti at W2, so these gates are dtype/shape logic and not
  hardware-conditioned:
  **570 of 835 doctests (68.3%) run with host fallbacks refused** as of W5 T2.
  Note the denominator: 835, not 843 — `weighted_mean/3`'s eight doctests joined
  `@rounding` in `nx_doctest_test.exs` when T2 made its sums resident, so they
  no longer execute at all. **Every ordinal below was renumbered by that**, which
  is the fragility the moduledoc warns about, happening for real. Only the super-io figure is re-measured at this point; the Kepler
  has not been re-run since W4.

  ## W5 T2 — integer reductions, and the bucket names stop meaning much

  529/843 -> 570/835. `reduce_axis_s32` and `window_reduce_s32` landed, and
  `product`/`window_sum`/`window_product` — unconditional host fallbacks at
  EVERY dtype, f32 included — became op codes on the shared reduce and window
  paths rather than separate transfers.

  Two things this tier exposed that are worth more than its count:

    * `@integer_dtype` is now badly named. What is left in it is mostly not
      dtype-gated at all — `window_reduce_op/6` refusing PADDED and DILATED
      windows (23), `indexed_put/5` with no scatter shader at any dtype (22),
      `argmax`/`argmin` (22), `reduce/5`'s arbitrary fun (11). These sit here
      because Nx's doctests for them are s32, not because s32 is the problem.
    * **`exp/2` at f32 appeared from nowhere (9 doctests) and is not a
      regression.** Those are `logsumexp`, which used to fail earlier at `sum`;
      with `sum` resident they get further and stop at `exp` of an INTEGER
      input. The unary gate requires `a_v.type == out.type`, and Nx types
      `exp(s32)` as f32, so the operand is never coerced — even though
      `cast_s32_to_f32.spv` has existed all along and the binary path already
      does exactly this coercion via `coerce_to/2`. A narrow gate (skill §1b),
      not a missing kernel.

  ## W5 T1 — 134 entries left, and all 134 moved

  The largest single movement this register has recorded, and unlike W4's it
  needs no asterisk: every one of the 134 is genuinely device-resident, not
  merely permitted to leave. `@integer_dtype` went 357 -> 223 and **no other
  bucket moved at all**, which is the shape a dtype fix should have.

  What landed: five new shaders (`elementwise_binary_s32`,
  `elementwise_binary_bcast_s32`, `elementwise_unary_s32`, `compare_s32`,
  `select_s32`), the two float compare shaders extended with op codes 6-10, and
  `@host_fallback_binary_ops` emptied down to `atan2` alone.

  The `@integer_dtype` name is now doing less work than it looks. What is left
  in it is mostly NOT dtype-gated: reductions (T2), `dot` (T3), and the ops with
  no GPU path at any dtype — `indexed_put/5`, `argmax`/`argmin`, `reduce/5`,
  `all`/`any`, `stack/3`. See NEXT.md §1.2 for the three-way split.

  ## The concat_nd fold — 13 entries left, and all 13 moved

  Unlike W4 below, this one needs no asterisk. The axis > 0 concatenate shader
  took the register from 458 to 445, and every one of those 13 doctests now
  runs on the device rather than merely being permitted to leave it:

    * `Nx.concatenate/2 (728, 730, 731)` — the shader itself;
    * `Nx.take_along_axis/3 (709-713)` and `Nx.take/3 (706, 707)` — they
      compose through concatenate;
    * `Nx.gather/3 (721, 722)` — the off-prefix axes W4's census named;
    * `Nx.top_k/2 (746)`.

  `Nx.take/3 (705)` is deliberately still listed. It went resident mid-session
  and then back to falling back while the take path was being edited; it is the
  one entry in this neighbourhood that is not settled. Re-measure it before
  assuming either way.

  W4 predicted exactly this set from its census, which is the census earning
  its keep: twelve opaque blocks became three named gaps, and closing one of
  the three moved all five ops that shared it.

  ## Read the W4 movement carefully — 30 entries left, but only 5 moved

  W4 took this register from 488 entries to 458, and the two halves of that 30
  are not the same kind of progress:

    * **5 genuinely reached the device** — `Nx.take/3 (699)`,
      `Nx.logical_not/1 (315)`, `Nx.pad_outer/3 (161)` and `Nx.top_k/2 (745,
      747)`. Their blocks are routed on-device now and the work runs on
      shaders.
    * **25 are FFT** — `fft2`, `ifft2`, `rfft`, `irfft`. Those did not move an
      inch. They are allowlisted in `Nx.Vulkan.Fallback` as a permanent complex
      -dtype limitation, and an allowlisted fallback is *permitted* rather than
      *refused*, so the doctest stops failing under `:raise` and leaves this
      register by the script's rules.

  That is the same convention `Nx.Block.Phase` and the seven `Nx.LinAlg` blocks
  have always been under, so the number stays comparable across W2/W1/W3/W4 —
  but it does mean this rate answers "how much is refused-clean", not "how much
  runs on the GPU". **Device-resident-only, W4 scores 360/843 (42.7%).** Quote
  whichever you mean, and say which. After the concat_nd fold above — all 13 of
  which are genuinely resident — the two readings are 398/843 (47.2%)
  refused-clean and 373/843 (44.2%) device-resident.

  ## How it is enforced

  `test_helper.exs` turns `filters/0` into ExUnit `:test`-name excludes, but
  only when `Nx.Vulkan.Fallback.mode/0` is `:raise`. In a normal `mix test` run
  nothing is excluded: all 843 doctests run and assert their values exactly as
  before, which is what keeps this an API-completeness suite. Under
  `NXV_HOST_FALLBACK=raise` the 265 listed ones step aside so the remaining 570
  can assert *where* they computed.

  `sh scripts/doctest_residency.sh` prints the rate and fails two ways:

    * a doctest not listed here falls back — a residency **regression**;
    * a doctest listed here stops falling back — a **stale** entry, and the
      rate is understating the truth.

  So the register is exact in both directions and the rate moves only when
  someone edits this file on purpose.

  ## The ordinals renumber

  ExUnit names a doctest `doctest Nx.add/2 (37)`, where 37 is its ordinal
  **after** the `:except` filtering in `nx_doctest_test.exs`. Adding an entry to
  that file's `@rounding` / `@unsupported` / `@backlog`, or bumping the `nx`
  dependency, shifts every later ordinal and invalidates this register wholesale.
  That fails loudly rather than silently: `doctest_residency.sh` reports the
  mismatch and prints the doctests that actually fall back today, so the repair
  is a paste and not an investigation. ExUnit offers no stabler handle than the
  test name.

  ## How the buckets were assigned

  By measurement, not judgement. Each doctest is filed under the dtype and op of
  the **first** `Nx.Vulkan.HostFallbackError` it raises under
  `NXV_HOST_FALLBACK=raise`. Format: `{"Nx.fun/arity", [ordinals]}`.
  """

  # 169 doctests, down from 357 before W5 (226 after T1 + the pow correction).
  # The name no longer fits: most of what is left is shape- or capability-gated
  # rather than dtype-gated, and is s32 only because Nx's doctests are. See the
  # T2 note in the moduledoc. This WAS a float backend (MISSION §3.1): the integer
  # elementwise, compare, select and reduce callbacks had no shader, and Nx's
  # own doctests are written almost entirely in {:s, 32}.
  #
  # **W5 T1 took 134 of these** — integer elementwise binary and unary, compare,
  # select, and the logical/bitwise/shift families that ride the same kernels.
  # What remains is NOT one bucket: T2 (integer axis- and window-reduce) and T3
  # (integer `dot`) are still dtype-gated, but ~71 of these have no GPU path at
  # ANY dtype and writing an integer shader will not close them. NEXT.md §1.2
  # has the three-way split; do not read this bucket's size as W5's remaining
  # work. W1 took 28 out of it and W3 another
  # 8 (all `Nx.all_close/3`, whose block body stopped leaking onto the GPU) — the
  # index-remap family went word-generic, so transpose/reverse/broadcast and
  # everything composing from them (tile, fill, revectorize, iota, eye,
  # put_slice, slice_along_axis, broadcast_vectors) now run on integers.
  #
  # The concat_nd shader then took 12 more: all three `Nx.concatenate/2`, the
  # two off-prefix `Nx.gather/3`, two `Nx.take/3` and every
  # `Nx.take_along_axis/3`. Those were integer-dtype doctests only incidentally
  # — what actually gated them was the axis > 0 concatenate, not the dtype, so
  # they left this bucket without W5 touching it.
  @integer_dtype [
    {"Nx.all/2", [441, 442, 443, 444, 445, 446]},
    {"Nx.all_close/3", [460]},
    {"Nx.any/2", [447, 448, 449, 450, 451]},
    {"Nx.argmax/2", [527, 528, 529, 530, 531, 532, 533, 534, 535, 536, 537]},
    {"Nx.argmin/2", [538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548]},
    {"Nx.as_type/2", [87, 90, 91]},
    {"Nx.bitcast/2", [95]},
    {"Nx.bitwise_not/1", [418, 419]},
    {"Nx.count_leading_zeros/1", [430, 431, 432, 433]},
    {"Nx.dot/2", [626, 629, 632, 633, 635, 636]},
    {"Nx.dot/4", [639, 641]},
    {"Nx.dot/6", [642, 643, 644, 645, 646]},
    {"Nx.fill/3", [823]},
    {"Nx.gather/3", [711, 712]},
    {"Nx.indexed_add/4", [347, 348, 351, 352]},
    {"Nx.indexed_put/4", [356, 357, 358, 359, 360, 361, 364, 365]},
    {"Nx.is_infinity/1", [409]},
    {"Nx.is_nan/1", [406]},
    {"Nx.linspace/3", [804, 805, 806]},
    {"Nx.make_diagonal/2", [39, 40, 41, 42, 43, 44]},
    {"Nx.max/2", [270]},
    {"Nx.min/2", [276]},
    {"Nx.mode/2", [494, 495, 496, 497, 499, 500]},
    {"Nx.multiply/2", [239]},
    {"Nx.negate/1", [414]},
    {"Nx.pad_outer/3", [156, 158, 160, 162]},
    {"Nx.population_count/1", [424]},
    {"Nx.pow/2", [241, 242, 244]},
    {"Nx.product/2", [505]},
    {"Nx.put_diagonal/3", [46, 47, 48, 49, 50, 51]},
    {"Nx.quotient/2", [260, 261]},
    {"Nx.reduce/4", [607, 608, 610, 611, 612, 613, 614, 615, 616, 617]},
    {"Nx.reflect/2", [814]},
    {"Nx.remainder/2", [249]},
    {"Nx.select/3", [336, 337, 338, 339, 344, 345, 346]},
    {"Nx.slice_along_axis/4", [683]},
    {"Nx.stack/2", [725, 726, 727, 728, 729]},
    {"Nx.subtract/2", [233]},
    {"Nx.sum/2", [466, 467]},
    {"Nx.take/3", [692, 693, 694, 695, 696, 697]},
    {"Nx.tril/2", [23]},
    {"Nx.triu/2", [27]},
    {"Nx.window_max/3", [563, 565, 566]},
    {"Nx.window_mean/3", [557, 559, 560, 561]},
    {"Nx.window_min/3", [568, 570, 571]},
    {"Nx.window_product/3", [573, 575, 576]},
    {"Nx.window_reduce/5", [618, 619, 620, 621, 622]},
    {"Nx.window_sum/3", [550, 552, 553, 554, 555]}
  ]

  # 37 doctests. GLSL.std.450 defines its transcendentals for 32-bit floats
  # only — there is no f64 `Sin`, `Log1p`, `Erf` or `Atan2`. This is the same
  # constraint that puts {:pow, 3} on Nx.Vulkan.Fallback's allowlist, and it is
  # a Vulkan/SPIR-V fact rather than a gap in this repo: closing it means
  # hand-writing double-precision polynomial approximations per op, which is a
  # project and not a task. The f32 forms of all 19 run on the GPU.
  @f64_transcendental [
    {"Nx.acos/1", [369, 387]},
    {"Nx.acosh/1", [370, 388]},
    {"Nx.asin/1", [371, 389]},
    {"Nx.asinh/1", [372, 390]},
    {"Nx.atan/1", [373, 391]},
    {"Nx.atan2/2", [265]},
    {"Nx.atanh/1", [374, 392]},
    {"Nx.cbrt/1", [375, 393]},
    {"Nx.cos/1", [376, 394]},
    {"Nx.cosh/1", [377, 395]},
    {"Nx.erf/1", [378, 396]},
    {"Nx.erf_inv/1", [379, 397]},
    {"Nx.erfc/1", [380, 398]},
    {"Nx.expm1/1", [381, 399]},
    {"Nx.log1p/1", [382, 400]},
    {"Nx.rsqrt/1", [383, 401]},
    {"Nx.sin/1", [384, 402]},
    {"Nx.sinh/1", [385, 403]},
    {"Nx.tan/1", [386, 404]}
  ]

  # 20 doctests. Complex is not representable on a byte-addressed f64-REAL
  # backend, and the whole FFT family produces or consumes it. There is no FFT
  # shader either, so both halves of the reason hold independently. Related:
  # `complex/2`, `real/1`, `imag/1`, `conjugate/1`, `phase/1` are in
  # @unsupported above — excepted outright, because they cannot even produce a
  # value here. These can: they fall back and are correct.
  @complex_and_fft [
    {"Nx.as_type/2", [88]},
    {"Nx.conv/3", [673]},
    {"Nx.fft/2", [752, 753, 754, 755, 756, 757, 758, 759]},
    {"Nx.ifft/2", [761, 762, 763, 764, 765, 766, 767, 768]}
  ]

  # 31 doctests — the interesting bucket, and the one to read before picking up
  # W1 or W8. These are float ops on a float backend that still left the GPU,
  # i.e. gates narrower than the capability behind them. The patterns:
  #
  #   rank 0        `dot/7`, `product/3`, `reduce/5`, `divide/3` refusing a {}
  #                 output. T11 widened several rank-0 gates; these are what it
  #                 did not reach.
  #   dot shapes    `dot/7` at {} and {1, 1, 2, 2} — W8's "beyond rank-2 ×
  #                 rank-2", visible here as four doctests.
  #   rank-3 window `window_sum/4`, `window_product/4`, `window_reduce_op/6` at
  #                 {2, 2, 5}. One shape, five doctests, four callbacks.
  #   log with base `Nx.log2/1`, `Nx.log10/1` and `Nx.log/2` compose to the
  #                 backend's `log/2`, which refuses at f32 — while `Nx.log/1`
  #                 on the same dtype runs natively (it is in @rounding above
  #                 precisely because the GPU answer differs in the last ULP).
  #                 Two paths to one op, one of them gated shut.
  #   scatter       `indexed_add/5`, `indexed_put/5` at {1}.
  #
  # Every line here is a candidate W1 item with a reproducer already written.
  @float_residency_gap [
    {"Nx.as_type/2", [86, 89, 92]},
    {"Nx.atan2/2", [262, 263, 264]},
    {"Nx.bitcast/2", [94]},
    {"Nx.concatenate/2", [719]},
    {"Nx.divide/2", [254]},
    {"Nx.dot/2", [627, 628, 634]},
    {"Nx.dot/4", [640]},
    {"Nx.indexed_add/4", [349, 350]},
    {"Nx.indexed_put/4", [362, 363]},
    {"Nx.log/2", [819, 820]},
    {"Nx.log10/1", [817, 818]},
    {"Nx.log2/1", [815, 816]},
    {"Nx.logsumexp/2", [827, 828, 829, 830, 831, 832, 833, 834, 835]},
    {"Nx.reduce/4", [609]},
    {"Nx.remainder/2", [247]},
    {"Nx.round/1", [440]},
    {"Nx.select/3", [343]},
    {"Nx.window_max/3", [564]},
    {"Nx.window_mean/3", [558]},
    {"Nx.window_min/3", [569]},
    {"Nx.window_product/3", [574]},
    {"Nx.window_sum/3", [551]}
  ]

  @doc """
  The whole register, flattened to `{"Nx.fun/arity", [ordinals]}` entries.
  """
  def all do
    @integer_dtype ++ @f64_transcendental ++ @complex_and_fft ++ @float_residency_gap
  end

  @doc """
  The register as ExUnit `:test`-name exclude filters.
  """
  def filters do
    for {fun, ordinals} <- all(), n <- ordinals do
      {:test, "doctest #{fun} (#{n})"}
    end
  end

  @doc """
  How many doctests the register excuses. 445 as measured; the number to watch.
  """
  def count, do: all() |> Enum.map(fn {_, ordinals} -> length(ordinals) end) |> Enum.sum()
end
