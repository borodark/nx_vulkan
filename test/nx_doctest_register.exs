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
  **714 of 833 doctests (85.7%) run with host fallbacks refused**, of which
  **703 (84.4%) are genuinely device-resident**. **`dot/7` and `select/4` are
  both entirely closed.** The 11-doctest gap is
  `Nx.reduce/4`, newly allowlisted — see the asterisk below. Quote whichever
  reading you mean, and say which. Note the
  denominator: 833, not 843. `weighted_mean/3` and `Nx.log/2` joined `@rounding`
  in `nx_doctest_test.exs` as their operands went resident and their f32
  arithmetic stopped matching BinaryBackend's inspect string, so those ten
  doctests no longer execute at all. **Every ordinal below was renumbered
  twice by that**, which is the fragility the moduledoc warns about, happening
  for real. Only the super-io figure is re-measured at this point; the Kepler
  has not been re-run since W4.

  ## `window_reduce/6` — the fold that WON (+5)

  709/833 -> 714/833, and the opposite answer to `reduce/5` from the same
  starting position: an arbitrary user fun that no shader can express.

  The difference is fold LENGTH, and it was measured rather than argued.
  `reduce/5` folds over the reduced axis — thousands of steps — and lost by 12x
  at reduce_size 4096. `window_reduce` folds over the WINDOW: 4 or 9 or 25 steps
  whatever the tensor size. Both host arms scale with the DATA, and the data is
  much bigger than the window, so this wins by 45x to 1800x
  (`bench/window_reduce_fold_vs_host.exs`, confirmed on the GT 650M).

  Padding needed no special case. BinaryBackend starts with
  `Nx.pad(tensor, acc, ...)` — it pads with the ACCUMULATOR, not a per-op
  identity — so doing the same reduces every padded window to the valid case.

  Two things the tests pin that `add` alone would never catch: the fold order is
  row-major with `fun.(element, acc)` — element FIRST — which only a
  non-commutative fun such as `subtract` can distinguish; and a fun that is not
  elementwise (one that reduces) would silently change the shape, so the fold
  checks and falls back rather than answering wrongly.

  ## `as_type` float -> integer (+2), and it is three rules

  707/833 -> 709/833. The cast matrix had eight entries and every one went TO a
  float; nothing converted to an integer. `cast_f32_to_w32.comp` and
  `cast_f32_to_u8.comp` are the first that do.

  It is not the one-line `int(x)` it looks like. `Nx.BinaryBackend` applies
  THREE rules depending on the value, and GLSL's own float->int conversion is
  UNDEFINED for most of them:

    * `NaN` -> 0;
    * the infinities SATURATE to the destination's limits;
    * everything else truncates toward zero and WRAPS modulo 2^width — `300.0`
      becomes 44 at u8, not 255.

  Saturating and wrapping in the same conversion is the trap: an implementation
  that clamped everything would pass the infinity tests and fail on `300.0`, and
  one that wrapped everything would do the reverse. Both directions are pinned.

  `int(1.0e10)` is itself undefined behaviour, so the modulo happens in floating
  point BEFORE the conversion and has to be exact. Above 2^55 an f32 is already
  a multiple of 2^32 so the answer is 0; below it the double arithmetic is
  exact. `1.0e15` is the case that would break a naive version — large enough to
  need the wrap, small enough that the answer is not 0.

  Integer-to-integer casts still fall back; only the float sources are covered.

  ## `select/4` takes any numeric predicate (+8)

  699/833 -> 707/833, no new shader. The gate demanded `{:u, 8}` because that is
  what the compare family emits — but `Nx.select/3` accepts ANY numeric
  predicate and treats nonzero as true, and its own doctests pass `1`, `0` and
  `Nx.tensor([0, 1, 0])`: all s32. `Nx.not_equal(pred, 0)` is that
  normalisation, and it has been a GPU op here since W5 T1.

  Note `!= 0`, not `== 1`. A normalisation that compared against 1 would pass
  every ordinary case and fail on a negative or fractional predicate; both are
  pinned.

  ## batched matmul closes `dot/7` completely (+5)

  694/833 -> 699/833, and `dot/7` disappears from this register entirely. Three
  shaders (`matmul_batched_{f32,f64,s32}`) and one NIF — the first Rust since
  `scatter`, and the last this backend needed for the dot path.

  Nx guarantees batch axes are "successive dimensions starting from 0" on both
  operands, so the batch is always a leading prefix and needs no rotation — only
  the flatten to `{B, M, K}` and `{B, K, N}`. The batch index rides the THIRD
  dispatch dimension rather than being looped in the caller, because dispatching
  once per matrix would pay the launch cost per batch element: the exact
  overhead that made the vectorised `reduce/5` fold lose to the host. Capped at
  65535 by `maxComputeWorkGroupCount[2]`, above which it falls back rather than
  looping.

  **Half of what this closed was never about batching as the user wrote it.**
  `Nx.vectorize` turns a vectorised axis into a leading batch axis, so vectorised
  `dot` came along as a side effect.

  ## `dot` generalised to any UNBATCHED contraction (+5)

  689/833 -> 694/833, and still no new kernel. `dot_orient/6` rotated the rank-2
  cases; the gate now flattens ANY unbatched contraction into the matmul the
  shader already does:

      a -> transpose to [free_a..., contracted_a...] -> reshape {M, K}
      b -> transpose to [contracted_b..., free_b...] -> reshape {K, N}

  M, K and N are the PRODUCTS of those dim groups, so a rank-4 contraction over
  two axes is the same dispatch as a rank-2 one over a single axis. Three things
  make it correct rather than plausible: `axes_a[i]` pairs with `axes_b[i]`
  POSITIONALLY and flattening in the given order preserves that; Nx's output is
  a's free dims then b's free dims in original order, which is exactly what
  `{M, N}` unrolls to, so no output permutation is needed; and an empty
  contraction falls out as the empty product K = 1 rather than needing a case.

  What is LEFT is all batched — six doctests, and half of them are batched only
  because an operand is `Nx.vectorize`d, which Nx turns into a leading batch
  axis. Closing them needs a batched matmul: a new kernel, a new NIF, and the
  first Rust since `scatter`.

  ## `bitcast/2` is a relabel — one line (+2)

  Nx raises on mismatched bit widths before dispatch, so this backend only ever
  sees a same-width reinterpretation of the same bytes. That is metadata,
  exactly like `reshape/2`. It had been transferring the whole tensor to the
  host in order to do nothing to it.

  Found while SIZING `as_type/2` rather than while doing it, which is the
  argument for sizing. Third of its species after `stack/3` and the rank-1
  `dot` promotion: an op that never asked for a capability it already had.

  ## `gather/4` rotates its axes instead of refusing them (+12)

  675/833 -> 687/833, again with no new shader. The gate required the indexed
  axes to be a leading prefix `[0..K-1]`; it now transposes the source when they
  are not, which is `dot_orient/6`'s normalise-then-dispatch applied to a
  different kernel.

  **The output needs no rotation back**, and that is what makes it two lines.
  Nx defines a gather's result as the index batch dims followed by the
  non-indexed source dims IN THEIR ORIGINAL RELATIVE ORDER, and a transpose that
  only moves the indexed axes to the front leaves that order untouched.

  The transpose is available exactly where the gather is — `transpose_nd` is a
  word copy for rank <= 4 and 4-byte-divisible dtypes, which the gate already
  required — so this costs one extra dispatch against a host round trip for the
  whole tensor. Closed `Nx.take/3` at axis > 0 (5), `Nx.gather/3` off-prefix (2),
  `Nx.pad_outer/3` (4) and `Nx.reflect/2` (1).

  ## `stack/3` routed to `concatenate/3` — no kernel at all (+5)

  670/833 -> 675/833 for two lines of routing. `Nx.BinaryBackend` implements
  stack as `Tuple.insert_at(shape, axis, 1)` followed by `bin_concatenate`, and
  this backend already had both halves: `reshape/2` is metadata only, and
  `concatenate/3` has had a shader since `concat_nd`. The callback arrives with
  the ORIGINAL tensors, so inserting the axis is the backend's job — and that was
  the entire gap.

  Worth remembering when scoring the rest of §1.3: the cheapest item on the board
  was not a missing capability but an op that never asked for the one it had.

  ## `dot` — the s32 matmul, and why W5 T3 was worth 4 rather than 17 (+6)

  664/833 -> 670/833. `glsl/matmul_s32.comp` is the integer twin of
  matmul_f32_f32acc — same 16x16 tiling, same dispatch, reusing `matmul/7`.
  Note there is no accumulator POLICY for integers: f32 offers :f32 and :f64
  because both are defensible approximations of an exact sum, whereas on
  integers only one answer matches the reference. Tiling changes the summation
  ORDER, which for floats is a precision question and for integers is not one
  at all — wrapping addition is associative.

  **The census over-counted this one badly, and it is worth understanding why.**
  `dot/7` showed 17 first-fallbacks, which MISSION §7 scored as W5's T3. Only
  FOUR of them were rank-2 x rank-2; the rest were rank-1, batched, multi-axis
  or higher-rank contractions that no dtype port touches. First-fallback
  attribution names the OP, not the gap — the same trap `window_sum` and
  `window_product` sprang in T2, where the register's `@integer_dtype` bucket
  claimed two ops that had no GPU path at any dtype.

  The other 2 of the 6 came from promoting rank-1 operands to rank 2 with a
  length-1 axis (vec·vec, mat·vec, vec·mat), which needs no shader at all and
  helps FLOATS as much as integers — `Nx.dot/2` on two f32 vectors was going to
  the host with the matmul shader sitting right there.

  11 `dot` doctests remain and are not dtype-gated: batched and higher-rank
  contractions need a real tensordot generalisation.

  ## `reduce/5` is allowlisted, and that +11 is PERMISSION not residency

  This is the second time the headline has moved for a reason other than work
  reaching the GPU — the first was W4's 25 FFT doctests. `Nx.reduce/4` takes an
  arbitrary user fun and now carries an allowlist entry, so its 11 doctests stop
  being refused and leave this register by the script's rules. **Not one of them
  runs on the device.** Refused-clean is 664/833 (79.7%); device-resident is
  653/833 (78.4%).

  The entry is a decision backed by measurement, not a shrug. Vectorising the
  fold — one dispatch per step along the reduced axis, fun evaluated on resident
  tensors — was prototyped and raced against the host path it would replace:

  | reduce_size | on-device fold | host fallback |
  |---:|---:|---:|
  | 8 | 0.97 ms | 0.19 ms |
  | 64 | 6.12 ms | 3.02 ms |
  | 512 | 39.81 ms | 22.01 ms |
  | 4096 | 440.62 ms | 37.40 ms |

  It is slower at every size and the gap WIDENS with the axis, because the cost
  is per-dispatch launch overhead. Nothing removes that without assuming the fun
  is associative (a log2-step tree reduce), which `Nx.reduce` does not guarantee
  — it is a left fold. Probing the fun to recognise `add` is the other tempting
  shortcut and is unsound for the same reason a probe always is.

  Trading +11 residency for a 12x regression is the 0.2.0 mistake in miniature.

  ## `allany` — all and any (+10)

  643/833 -> 653/833. Four more shaders that reuse `reduce_axis/7` verbatim, so
  still no new NIF. The output is `{:u, 8}` whatever the input, written as
  packed u32 words with ONE THREAD PER FOUR OUTPUT SLOTS — which makes the
  dispatch geometry deliberately loose, since the NIF launches four times the
  threads this needs and the surplus return at the bounds guard. Reusing the
  NIF is worth the idle threads.

  The `{:u, 8}` INPUT entry is the one that earns its keep:
  `Nx.all(Nx.greater(a, b))` is the natural idiom and `greater` already emits a
  u8 mask on the GPU, so without it the mask went back to the host purely to be
  summarised. Same lesson as T12's `{:u, 8} -> {:u, 32}` sum entry — the dtype a
  gate refuses is usually one the backend itself produced.

  NaN needs no special case here, unlike argreduce: `NaN != 0.0` is true and
  BinaryBackend agrees, so NaN is truthy on both sides.

  ## `argreduce` — argmax and argmin (+14)

  629/833 -> 643/833. `glsl/argreduce_{f32,f64,s32}.comp` reuse `reduce_axis/7`
  verbatim — same bindings, same (outer, reduce_size, inner, op) push layout —
  so three shaders landed with no new NIF. The output is an INDEX rather than a
  value, so unlike `reduce_spv/2` the selector is keyed on the input type alone.

  Two semantics traps, and neither is visible on ordinary data:

    * **`:tie_break`.** `:low` (the default) keeps the FIRST index holding the
      extreme; `:high` the last. A strict comparison gives `:low` for free, and
      `:high` needs the tie to count as a win. Any input without duplicates
      passes either way.
    * **NaN is absorbing, and last-NaN-wins.** BinaryBackend's rule is one line
      (`x == :nan or comparator.(x, cur)`), and IEEE comparison gets BOTH halves
      wrong on its own because `v < best` and `v > best` are false for a NaN
      operand. A NaN candidate always replaces the incumbent — including another
      NaN, so `argmax([nan, 5, nan])` is 2 even at `:low` — and a NaN incumbent
      is unbeatable by any number. Two of Nx's own doctests pin this and nothing
      else in the suite would have.

  Infinities need no special case: IEEE ordering already matches.

  ## `scatter` — indexed_put and indexed_add (+23)

  606/833 -> 629/833. `glsl/scatter.comp` is the inverse of `gather.comp` —
  same index arithmetic, same params layout, source and destination swapped —
  and it closes the largest non-decided residual W5's census named. It is also
  where `Nx.LinAlg.invert/1` died (MISSION §3.3); invert now reports only its
  two allowlisted LinAlg blocks.

  The two ops share a shader and differ in exactly one way, which decides their
  dtype gates:

    * `indexed_put` DOCUMENTS its race — "in case of repeating indices, the
      result is non-deterministic, since the operation happens in parallel when
      running on devices such as the GPU". A plain word write is therefore the
      SPECIFIED behaviour rather than a tolerated approximation, and every
      4/8-byte dtype gets it.
    * `indexed_add` must accumulate duplicates deterministically, which needs an
      atomic. Integer `atomicAdd` is core GLSL 4.30 and exact on the
      two's-complement bit pattern. Float `indexed_add` stays on the host for
      the same reason overlapping pooling backward does — an f32 atomic needs
      `GL_EXT_shader_atomic_float`, which the Kepler fleet does not guarantee.

  Two things it needed beyond the gather template: the output is seeded with a
  COPY of the target (a scatter writes only what the indices name, so a zeroed
  buffer is wrong), and both target and updates are coerced, because Nx promotes
  them — which is the narrow gate invert was actually hitting.

  ## Two narrow gates, worth 36 between them — and neither was a missing kernel

  570/835 -> 606/833. Both were `if`s that were narrower than the shader behind
  them (skill §1b), and both were found by T2 rather than by reading:

    * **The unary path never coerced its operand** (+13). `Nx.exp(s32_tensor)`
      has an f32 output template against an s32 operand, so `a_v.type == type`
      was false and the whole thing went to the host — though
      `cast_s32_to_f32.spv` had existed since T11 and the BINARY path had been
      coercing via `coerce_to/2` all along. Cost: `Nx.log/2` excepted, because
      `log(t)/log(base)` computed wholly in f32 gives 2.9999998 for
      log(27)/log(3) where BinaryBackend's f64 gives 3.0.
    * **The window gate refused padded and dilated windows** (+23). Nx pads a
      window reduction with the OP'S IDENTITY, and for all four ops skipping an
      out-of-bounds element equals combining with that identity — so no `inf`
      literals were needed, and the integer shader stayed identical in shape to
      the float ones. One edge does need them: a window that is ENTIRELY
      padding has nothing to seed from, and max/min must return -inf/+inf
      (INT_MIN/INT_MAX on s32). Every value assertion passed without that
      handling, because every window they used touched a real element.

  Negative padding still falls back: it crops rather than pads, and
  skip-out-of-bounds cannot express it.

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
  `NXV_HOST_FALLBACK=raise` the 119 listed ones step aside so the remaining 714
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

  # 51 doctests, down from 357 before W5.
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
    {"Nx.all/2", [446]},
    {"Nx.all_close/3", [460]},
    {"Nx.argmax/2", [532, 534, 535, 536]},
    {"Nx.argmin/2", [543, 545, 546, 547]},
    {"Nx.as_type/2", [87, 90]},
    {"Nx.bitwise_not/1", [418, 419]},
    {"Nx.count_leading_zeros/1", [430, 431, 432, 433]},
    {"Nx.fill/3", [821]},
    {"Nx.indexed_add/4", [351]},
    {"Nx.indexed_put/4", [361, 364]},
    {"Nx.is_infinity/1", [409]},
    {"Nx.is_nan/1", [406]},
    {"Nx.linspace/3", [804, 806]},
    {"Nx.max/2", [270]},
    {"Nx.min/2", [276]},
    {"Nx.mode/2", [494, 495, 496, 497, 499, 500]},
    {"Nx.multiply/2", [239]},
    {"Nx.negate/1", [414]},
    {"Nx.population_count/1", [424]},
    {"Nx.pow/2", [241, 242, 244]},
    {"Nx.product/2", [505]},
    {"Nx.quotient/2", [260, 261]},
    {"Nx.remainder/2", [249]},
    {"Nx.slice_along_axis/4", [683]},
    {"Nx.subtract/2", [233]},
    {"Nx.sum/2", [466, 467]},
    {"Nx.take/3", [697]},
    {"Nx.tril/2", [23]},
    {"Nx.triu/2", [27]}
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
    {"Nx.concatenate/2", [719]},
    {"Nx.divide/2", [254]},
    {"Nx.dot/2", [634]},
    {"Nx.indexed_add/4", [349, 350]},
    {"Nx.remainder/2", [247]},
    {"Nx.round/1", [440]}
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
