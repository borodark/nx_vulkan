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

  Measured on `main` @ `7b0e23f`, super-io / RTX 3060 Ti:
  **319 of 843 doctests (37.8%) run with host fallbacks refused.**

  ## How it is enforced

  `test_helper.exs` turns `filters/0` into ExUnit `:test`-name excludes, but
  only when `Nx.Vulkan.Fallback.mode/0` is `:raise`. In a normal `mix test` run
  nothing is excluded: all 843 doctests run and assert their values exactly as
  before, which is what keeps this an API-completeness suite. Under
  `NXV_HOST_FALLBACK=raise` the 524 listed ones step aside so the remaining 319
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

  # 409 doctests. This is a float backend (MISSION §3.1): the integer
  # elementwise, compare, select and reduce callbacks have no shader, and Nx's
  # own doctests are written almost entirely in {:s, 32}. Nothing here is a gate
  # bug — the capability does not exist yet. **W5 retires this bucket wholesale**,
  # and it is the single largest reason the rate is 38% and not 80%; these same
  # 409 ordinals are its acceptance test.
  @integer_dtype [
    {"Nx.abs/1", [416]},
    {"Nx.all/2", [441, 442, 443, 444, 445, 446]},
    {"Nx.all_close/3", [452, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462]},
    {"Nx.any/2", [447, 448, 449, 450, 451]},
    {"Nx.argmax/2", [535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545]},
    {"Nx.argmin/2", [546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556]},
    {"Nx.as_type/2", [87, 90, 91]},
    {"Nx.bitcast/2", [95]},
    {"Nx.bitwise_and/2", [278, 279, 280, 281]},
    {"Nx.bitwise_not/1", [417, 418, 419]},
    {"Nx.bitwise_or/2", [283, 284, 285, 286]},
    {"Nx.bitwise_xor/2", [288, 289, 290, 291]},
    {"Nx.broadcast/3", [131, 132, 133, 136, 137, 138, 139]},
    {"Nx.broadcast_vectors/2", [225]},
    {"Nx.clip/3", [682]},
    {"Nx.concatenate/2", [728, 730, 731]},
    {"Nx.count_leading_zeros/1", [426, 427, 428, 429, 430, 431, 432, 433]},
    {"Nx.cumulative_max/2", [603, 604, 605, 606, 607, 608]},
    {"Nx.cumulative_min/2", [597, 598, 599, 600, 601, 602]},
    {"Nx.cumulative_product/2", [591, 592, 593, 594, 595, 596]},
    {"Nx.cumulative_sum/2", [585, 586, 587, 588, 589, 590]},
    {"Nx.diff/2", [609, 610, 611, 612]},
    {"Nx.dot/2", [631, 634, 637, 640, 641, 643, 644, 645]},
    {"Nx.dot/4", [647, 649]},
    {"Nx.dot/6", [650, 651, 652, 653, 654]},
    {"Nx.equal/2", [301, 302, 303]},
    {"Nx.eye/2", [19]},
    {"Nx.fill/3", [829, 830, 831]},
    {"Nx.gather/3", [719, 720, 721, 722]},
    {"Nx.greater/2", [320, 321, 322]},
    {"Nx.greater_equal/2", [328, 329, 330]},
    {"Nx.indexed_add/4", [347, 348, 351, 352]},
    {"Nx.indexed_put/4", [356, 357, 358, 359, 360, 361, 364, 365]},
    {"Nx.iota/2", [14]},
    {"Nx.is_infinity/1", [408, 409, 410]},
    {"Nx.is_nan/1", [405, 406, 407]},
    {"Nx.left_shift/2", [293, 294, 295]},
    {"Nx.less/2", [324, 325, 326]},
    {"Nx.less_equal/2", [332, 333, 334]},
    {"Nx.linspace/3", [812, 813, 814]},
    {"Nx.logical_and/2", [305, 306, 307]},
    {"Nx.logical_not/1", [314, 315]},
    {"Nx.logical_or/2", [308, 309, 310]},
    {"Nx.logical_xor/2", [311, 312, 313]},
    {"Nx.logsumexp/2", [835, 836, 837, 838, 839, 840, 841, 842, 843]},
    {"Nx.make_diagonal/2", [39, 40, 41, 42, 43, 44]},
    {"Nx.max/2", [266, 267, 269, 270]},
    {"Nx.mean/2", [473, 474, 475, 477, 478, 479, 480, 481]},
    {"Nx.min/2", [272, 273, 275, 276]},
    {"Nx.mode/2", [502, 503, 504, 505, 507, 508]},
    {"Nx.multiply/2", [235, 236, 238, 239]},
    {"Nx.negate/1", [411, 412, 414]},
    {"Nx.not_equal/2", [316, 317, 318, 319]},
    {"Nx.outer/2", [661, 662, 663, 664]},
    {"Nx.pad_outer/3", [155, 156, 157, 158, 159, 160, 161, 162]},
    {"Nx.population_count/1", [421, 422, 423, 424]},
    {"Nx.product/2", [509, 510, 512, 513, 514, 515, 516, 517]},
    {"Nx.put_diagonal/3", [46, 47, 48, 49, 50, 51]},
    {"Nx.put_slice/3", [698]},
    {"Nx.quotient/2", [256, 257, 258, 259, 260, 261]},
    {"Nx.reduce/4", [615, 616, 618, 619, 620, 621, 622, 623, 624, 625]},
    {"Nx.reduce_max/2", [519, 521, 522, 523, 524, 525, 526]},
    {"Nx.reduce_min/2", [527, 529, 530, 531, 532, 533, 534]},
    {"Nx.reflect/2", [821, 822]},
    {"Nx.remainder/2", [245, 246, 248, 249]},
    {"Nx.revectorize/3", [226, 227]},
    {"Nx.reverse/2", [672, 673, 674, 675, 676, 678]},
    {"Nx.right_shift/2", [297, 298, 299]},
    {"Nx.select/3", [336, 337, 338, 339, 340, 341, 342, 344, 345, 346]},
    {"Nx.sign/1", [415]},
    {"Nx.slice_along_axis/4", [691, 693]},
    {"Nx.stack/2", [733, 734, 735, 736, 737]},
    {"Nx.subtract/2", [229, 230, 232, 233]},
    {"Nx.sum/2", [463, 464, 466, 467, 468, 469, 470, 471]},
    {"Nx.take/3", [699, 700, 701, 702, 703, 704, 705, 706, 707]},
    {"Nx.take_along_axis/3", [709, 710, 711, 712, 713]},
    {"Nx.tile/2", [111, 112, 113]},
    {"Nx.top_k/2", [745]},
    {"Nx.transpose/2", [666, 668, 669]},
    {"Nx.tri/3", [29, 30]},
    {"Nx.tril/2", [21, 22, 23]},
    {"Nx.triu/2", [25, 26, 27]},
    {"Nx.vectorize/2", [211, 212]},
    {"Nx.weighted_mean/3", [482, 483, 484, 485, 486, 487, 488, 489]},
    {"Nx.window_max/3", [570, 571, 573, 574]},
    {"Nx.window_mean/3", [564, 565, 567, 568, 569]},
    {"Nx.window_min/3", [575, 576, 578, 579]},
    {"Nx.window_product/3", [580, 581, 583, 584]},
    {"Nx.window_reduce/5", [626, 627, 628, 629, 630]},
    {"Nx.window_sum/3", [557, 558, 560, 561, 562, 563]}
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

  # 45 doctests. Complex is not representable on a byte-addressed f64-REAL
  # backend, and the whole FFT family produces or consumes it. There is no FFT
  # shader either, so both halves of the reason hold independently. Related:
  # `complex/2`, `real/1`, `imag/1`, `conjugate/1`, `phase/1` are in
  # @unsupported above — excepted outright, because they cannot even produce a
  # value here. These can: they fall back and are correct.
  @complex_and_fft [
    {"Nx.as_type/2", [86, 88, 92]},
    {"Nx.conv/3", [681]},
    {"Nx.fft/2", [760, 761, 762, 763, 764, 765, 766, 767]},
    {"Nx.fft2/2", [778, 779, 780, 781, 782, 783, 784]},
    {"Nx.ifft/2", [769, 770, 771, 772, 773, 774, 775, 776]},
    {"Nx.ifft2/2", [787, 788, 789, 790, 791, 792, 793]},
    {"Nx.irfft/2", [804, 805, 806, 807, 808]},
    {"Nx.rfft/2", [796, 797, 798, 799, 800, 801]}
  ]

  # 33 doctests — the interesting bucket, and the one to read before picking up
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
    {"Nx.as_type/2", [89]},
    {"Nx.atan2/2", [262, 263, 264]},
    {"Nx.bitcast/2", [94]},
    {"Nx.concatenate/2", [727]},
    {"Nx.divide/2", [254]},
    {"Nx.dot/2", [635, 636, 642]},
    {"Nx.dot/4", [648]},
    {"Nx.indexed_add/4", [349, 350]},
    {"Nx.indexed_put/4", [362, 363]},
    {"Nx.log/2", [827, 828]},
    {"Nx.log10/1", [825, 826]},
    {"Nx.log2/1", [823, 824]},
    {"Nx.product/2", [511]},
    {"Nx.reduce/4", [617]},
    {"Nx.remainder/2", [247]},
    {"Nx.round/1", [440]},
    {"Nx.select/3", [343]},
    {"Nx.top_k/2", [746, 747]},
    {"Nx.window_max/3", [572]},
    {"Nx.window_mean/3", [566]},
    {"Nx.window_min/3", [577]},
    {"Nx.window_product/3", [582]},
    {"Nx.window_sum/3", [559]}
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
  How many doctests the register excuses. 524 as measured; the number to watch.
  """
  def count, do: all() |> Enum.map(fn {_, ordinals} -> length(ordinals) end) |> Enum.sum()
end
