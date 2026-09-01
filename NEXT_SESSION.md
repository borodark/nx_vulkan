# Next session — state, open items, and one live bug

**HEAD `ab2e779`**, pushed to `origin` (private). Nothing pushed to `upstream`.
Rewritten 2026-08-31, after a session on super-io plus a Jetson verification.

Two commits landed: `d7b5f08` (the `buf_upload` heap fix, and the DVFS
correction below) and `ab2e779` (folding the staging copies into the caller's
command buffer, which repairs a regression `d7b5f08` introduced).

---

## First: the reboot never happened

The previous edition of this file was written expecting a reboot and spent a
section on what `/tmp` would cost. Uptime is 13 days and `/tmp` is intact —
`ew.exs`, `lim.exs`, `vram.exs`, `dpf_probe.sh` and ~70 other probes are all
still there. Nothing was lost. `scripts/staged/jetson_run_trace.sh` is still
staged and still unrun.

super-io's driver is healthy: `NVIDIA GeForce RTX 3060 Ti (DiscreteGpu)`,
`unified memory: false (staging path: ON)`. Not llvmpipe.

---

## THE HEADLINE: `buf_upload` had the bug `alloc_buffer` had — FIXED on super-io, unverified elsewhere

Last session fixed `alloc_buffer`, which asked for
`PREFER_DEVICE | HOST_RANDOM_ACCESS` — a preference and a requirement, the
requirement winning, so every output buffer lived in system RAM.

**`upload_buffer` asked for `PREFER_DEVICE | HOST_SEQUENTIAL_WRITE`.** Same
shape, same defect, never migrated. Every tensor that enters the GPU from the
host goes through it.

The consequence is not a constant tax — it is a cliff, and the cliff is
**cumulative BAR1 pressure**, not buffer size. super-io has BAR1 = 256 MiB
(resizable BAR is OFF). Uploaded buffers land in that host-visible device-local
window until it fills, then silently fall back to plain system RAM.

Measured with ten 32 MiB uploads held live, each read by the flat elementwise
shader, at boost clock:

    N   cumulative   ms      3n GB/s   BAR1 used
    1       32 MiB   0.554   181.7        69 MiB
    2       64 MiB   0.500   201.3       101 MiB
    3       96 MiB   0.593   169.8       133 MiB
    4      128 MiB   0.544   185.0       165 MiB
    5      160 MiB   0.695   144.8       197 MiB
    6      192 MiB   0.606   166.1       229 MiB
    7      224 MiB   3.063    32.9       229 MiB   <- window full
    8      256 MiB   3.086    32.6       229 MiB
    9      288 MiB   3.079    32.7       229 MiB
   10      320 MiB   3.077    32.7       229 MiB

BAR1 stops climbing at 229 MiB and the throughput falls 5.5x, at **constant
buffer size**. Backing the device-resident operands out of the 3n figure puts
the PCIe leg at ~10.9 GB/s — the same sustained 10.8 GB/s the previous session
measured directly. It is the same road.

The first sighting was accidental: a control comparing `buf_alloc`'d operands
against uploaded ones read 1.29 ms vs 11.16 ms at 128 MiB (two 128 MiB uploads
= 256 MiB, over the window) and matched exactly at 48 and 64 MiB (96 and
128 MiB, under it).

### The fix, as applied

Both halves were already in the tree — `buf_download` uses `staging_read`,
`buf_upload_into` uses `staging_write`. `upload_buffer` now allocates through
`alloc_buffer` (DEVICE_LOCAL) and writes through staging, in two variants:

* `upload_buffer` blocks until the copy has executed. Required by the three
  leapfrog synth NIFs and `fft`, which build and submit their OWN command
  buffers; a deferred copy would leave them reading an uninitialised buffer.
* `upload_buffer_deferred` ENQUEUES the copy into the pending batch, and is
  used only by the `buf_upload` NIF. The blocking variant would put a submit
  and fence wait on a path that previously did neither, and that path is not
  only for big tensors — `gpu_bcast_binary` uploads a 52-byte params buffer
  through it on every broadcast op. This is the same trap `alloc_buffer_zeroed`
  fell into and had to undo. Safe because every consumer of a `buf_upload`
  result either goes through `enqueue_dispatch` (same queue, replayed in order)
  or calls `flush_pending()` first.

Verified on super-io, same probe, boost clock:

    N   cumulative   ms      3n GB/s   BAR1 used
    1       32 MiB   0.528   190.7        37 MiB
    6      192 MiB   0.543   185.4        37 MiB
    7      224 MiB   0.631   159.5        37 MiB
   10      320 MiB   0.627   160.5        37 MiB

BAR1 no longer moves at all — 37 MiB is the desktop's own baseline — and the
cliff is gone. `mix test`: **833 doctests, 871 tests, 0 failures**, matching the
recorded baseline exactly.

### `d7b5f08` introduced a regression, and `ab2e779` fixes it

Staging cost a submit and fence wait per upload, which was the wrong trade for
the four callers that build and submit their own command buffer. `Nx.fft` makes
exactly one `upload_buffer` call per invocation, so it is a clean handle:

    n       twiddle   6b38aee (pre)   d7b5f08 (post)   delta
    1024      8 KiB   0.527 / 0.469   0.713 / 0.829    +0.27 ms
    4096     32 KiB   0.741 / 0.666   0.851 / 0.874    +0.16 ms
   16384    128 KiB   1.380 / 1.363   1.537 / 1.536    +0.17 ms

The table grows 16x and the delta does not move — submission cost, not staging
bandwidth. `leapfrog_chain_synth_f64` makes THREE such calls per chain dispatch,
and exmc runs one dispatch per chain per draw: ~6000 fence waits on a 4-chain
500-draw run, for copies already ordered ahead of a dispatch the NIF submits
itself.

`ab2e779`: `upload_buffer_staged` prepares the staging buffer without
submitting, `record_upload` folds the copy into the caller's own command buffer,
vulkano's automatic sync inserts the barrier. A fence becomes a barrier. The
blocking `upload_buffer` had no callers left and is deleted.

Verified with three arms, three rounds, arm order rotated so no arm always eats
the cold start; 30 fft calls per timed sample. Per-call ms:

    n       pre                   broken                fixed
    1024    0.424 0.407 0.437     0.616 0.559 0.468     0.438 0.398 0.402
    4096    0.631 0.485 0.610     0.747 0.684 0.638     0.547 0.543 0.565

Two conditions, both required: broken must separate from pre (else the harness
cannot see the effect and proves nothing), and fixed must sit with pre. At
n=1024 broken and pre do not overlap; fixed lies inside pre at both sizes.

### What the heap fix cost

A/B at boost clock, per-call ms, before against after:

    quantity   48 MiB          64 MiB          128 MiB
    flat_re    0.716 -> 0.700  0.821 -> 0.822  1.296 -> 1.296
    bcast_re   0.567 -> 0.555  0.660 -> 0.639  1.025 -> 1.032
    alloc      0.557 -> 0.577  0.589 -> 0.646  0.894 -> 0.911
    nx_mul     1.151 -> 1.270  1.449 -> 1.466  2.257 -> 2.274

Shader and allocation paths are unchanged within noise. The full Nx elementwise
path costs roughly **+0.08 ms per call**, a fixed cost independent of size,
attributed directly:

    dispatch only (params reused)    0.667 ms
      + buf_upload(52 B params)      0.743 ms   (+0.076)
      + buf_alloc(64 MiB) output     1.331 ms   (+0.588)

That is the 52-byte params buffer, which used to be a bare host write and is now
a real device copy command. It is ~9% at 48 MiB and inside the noise at 128 MiB,
paid to remove a 5.5x cliff. Two ways to get it back, neither attempted: the
params block is 13 int32s = 52 bytes and would fit in a 128-byte push constant,
or the buffer could be cached on shape. Allocation, at +0.588 ms, remains seven
times larger and is the better target — see open item 2.

Why no test caught it: the host path returns bit-identical results, residency is
unchanged (the tensor *is* on the backend, just in the wrong heap), and the
fallback census cannot see it either. Nothing in the suite asserts which heap a
buffer landed in.

---

## Item 1 is answered, and the answer is that the number was wrong

The previous edition asked, of `Nx.multiply` at 33-38 GB/s against 448 GB/s of
VRAM: "is it the shader or the dispatch geometry?"

Neither. **It was the GPU clock.** That measurement was taken at 210 MHz, the
card's idle floor.

`/tmp/ew.exs` warms with `Nx.multiply` + `flush` + `garbage_collect` in a loop.
Each dispatch is ~1 ms with host work between, which the driver reads as idle,
so the clock never leaves the floor. Its `pin.()` is three 512x512 matmuls —
far too small to boost an Ampere card — and it runs *before* the 600 ms warm
that lets the clock decay again.

Reproduced and then broken, 64 MiB f32, median of 9:

    warm                              ms      2n GB/s   clocks.sm
    ew.exs warm, verbatim shape      3.381      39.7      210 MHz
    sustained-dispatch preamble      1.778      75.5     1935 MHz
    ew.exs warm, clock already up    1.277     105.1     1935 MHz

39.7 GB/s at 210 MHz reproduces the 33-38 GB/s that went into the last planning
document as a hardware finding. The same op boosted is 2.5-2.7x faster
(`nx_mul` at 64 MiB: 3.913/3.746 ms idle against 1.449/1.510 ms boost).

**The shader was never the problem.** Slope across 48/64/128 MiB at verified
boost, so the fixed per-call cost drops out:

    quantity     rep1        rep2      traffic
    flat_re    431 GB/s    429 GB/s      3n     <- 96% of the card's 448
    bcast_re   367 GB/s    378 GB/s      2n
    nx_mul     156 GB/s    134 GB/s      2n

The flat elementwise kernel is memcpy-class. The broadcast kernel — the one the
old measurement actually exercised, since `Nx.multiply(x, 1.0)` takes
`gpu_bcast_binary`, not `apply_binary` — is within ~15% of it, so its
per-element `%`/`/` index decomposition costs approximately nothing; it is
fully hidden behind memory. **Do not write the hand-rolled saturating kernel
the last edition proposed.** There is no headroom there to find.

---

## GPU clock is now a dimension of the test, not an unrecorded condition

Every number in the previous document was taken at an unrecorded clock. This is
the fourth DVFS incident in this project (`a6dcfa7` was the third, and
`examples/unified_vs_discrete_race.exs` already carries `pin_clock` and a long
comment block about exactly this — the throwaway probe simply did not use it).

The harness now induces three states rather than assuming one. There is no root
on this box, so `nvidia-smi -lgc` is unavailable; the states are induced by
workload shape and *verified* afterwards, never requested:

* `idle` — sleep before each sample; the GPU settles to its 210 MHz floor
* `partial` — a short burst before each sample; catches the ramp
* `boost` — sustained dispatch; the state a real workload actually runs in

### The instrument lesson, which cost a whole run

**Do not call `nvidia-smi` from inside the measuring process.** It takes
50-100 ms, and that is by itself enough GPU idle to drop the boost. A run
guarded that way rejected all 12 of its own rows as "low clock" while the
timings were plainly boosted, and reported flat-shader slopes of 593 and
804 GB/s — above the card's physical 448 GB/s peak, which is how the corruption
announced itself. The clock is now sampled by an external
`nvidia-smi -lms 50` logger writing `epoch_ms clock`, and each measurement
window emits begin/end marks that are joined against that log afterwards.

A second, quieter trap in the same run: `Nx.iota` at 33M elements is minutes of
host-side work with the GPU idle throughout, which dominated the wall time and
held the card at 210 MHz for 73% of the samples. Build large resident test
tensors with `Nx.from_binary(:binary.copy(...))` — a host memcpy and one
upload — not `Nx.iota`.

### Clock sensitivity separates driver work from GPU work

Ratio of idle-clock time to boost-clock time, two replicates. Above 1 means the
quantity is GPU-clock-bound:

    MiB  quantity    rep1    rep2
     48  flat_re     1.358   1.678
     48  bcast_re    4.042   3.527
     48  alloc       1.381   1.077
     48  nx_mul      2.613   2.347
     64  flat_re     1.815   1.757
     64  bcast_re    4.197   3.605
     64  alloc       1.256   1.242
     64  nx_mul      2.700   2.481
    128  flat_re     1.909   1.821
    128  bcast_re    4.709   4.694
    128  alloc       1.186   1.269
    128  nx_mul      2.850   2.492

**`buf_alloc` is nearly clock-invariant (1.08-1.38x) while shader work moves
3.5-4.7x.** That is the discriminator the previous four race designs were
reaching for and never isolated: allocation is driver and host bookkeeping, not
GPU work, and it therefore does *not* shrink when the card boosts. At idle it
hides inside a slow measurement; at boost it is exposed as a first-order cost.

Read the `flat_re` rows with suspicion — its windows are first in each state
block and several straddle the decay (`clk_med 1755`, `clk_min 255`), so its
idle figures are contaminated toward the boost end and its sensitivity is
understated. `bcast_re` is the clean signal.

### At boost, allocation is as expensive as the computation

64 MiB, boost, median of 9: `alloc` 0.589 ms against `bcast_re` 0.660 ms, and
`nx_mul` 1.449 ms. The accounting closes — allocation is most of the gap
between the raw dispatch and the full Nx call, with ~0.2 ms left for the params
upload and Nx plumbing.

The comment at `native/nx_vulkan_vulkano/src/lib.rs:1285` says `buf_alloc`
"does not initialise, which is why it is roughly free". **That is no longer
true.** It does not initialise, but above the 32 MiB cliff it takes a dedicated
allocation per call, and the cost scales with size (slope 240 and 366 GB/s at
boost across two replicates — the spread is wide and this quantity is the
noisiest of the four, so treat it as order-of-magnitude). Every
elementwise op allocates and drops one output buffer. A buffer pool or free-list
keyed on size is the obvious move and has not been tried.

---

## Open items, in priority order

### 1. Verify across the Keplers — the Jetson is DONE

**Jetson: verified.** 833 doctests / 871 tests / 0 failures at `d7b5f08` on a
cross-compiled NIF, on the `unified memory: true (staging path: OFF)` branch.
The no-op claim is now measured rather than reasoned. `ab2e779` has NOT been
run there yet, though its unified path is the same no-op by construction.

**mac-247 and mac-248: untouched.** No clean A/B for either commit. The exmc
session observed exmc suite time rising 8.8% on 247 (883 -> 960s) and 10.4% on
248 (521 -> 576s) across `2617e5e -> 6b38aee` while super-io stayed flat, with a
narrower chain-dispatch probe on 247 moving 3.0 -> 4.0/4.1/4.1s. Their
before-reading is n=1 and these are suite times, so it is a direction, not a
magnitude. Re-measure on `ab2e779`, never on `d7b5f08` — that commit alone
slows the chain path.

**Amplify any Kepler measurement.** FreeBSD reports `[N/A]` for `clocks.sm` on
those cards, so the DVFS confound that swung a super-io number 2.6x cannot even
be observed there. Use many dispatches per timed sample; a per-call effect of
~0.16 ms will not survive single-call timing. Not committed yet.

Worth adding at the same time: a probe that asserts which heap a buffer landed
in, since nothing in the suite can currently see this class of bug. The BAR1
column in `scripts/staged/bar1_cliff.exs` is the whole of the current
instrumentation.

### 1b. The upload/readback work is measured — on mac-248, not here

All three changes were measured by the exmc session driving the f64 chain NIF
directly on **mac-248 (GT 750M, headless)**, N=3000/sample, 6000-dispatch
warmup, 6 replicates. Per-dispatch cost:

    nx_vulkan ab2e779   365 us
    nx_vulkan 096d7bd   238 us   fast path OFF
                        224 us   fast path ON

Decomposing:

    365 -> 238   four download fences becoming one (8cce91c)   -127 us   -35%
    238 -> 224   small-upload fast path (b59c4a7)               -14 us    -6%
    365 -> 224   both                                          -141 us   -39%

The readback batching is nine times the fast path, which is what the byte
asymmetry predicts: `3*K*d*8` down against `2*d*8` up.

Caveat kept deliberately: the 365 arm came from a separate build rather than an
interleaved one, so the 35% carries a build boundary. The ON/OFF pairs are the
rigorous half — same binary, runtime knob, order balanced, non-overlapping
(highest ON 226.7 us, lowest OFF 237.1 us).

**Neither number could have been produced here.** super-io's ~900 us noise band
is wider than both effects, and nothing in this repo could drive the chain path
until `15abc96`. That is now fixed, so the next such measurement can be taken
locally — on a headless box.

### 1c. SIXTEEN unallowlisted fallbacks, and strict mode cannot see them

I claimed `pow` was "the one real gap". It was not — it was the one my
leapfrog-shaped census happened to touch, and it was already an `@allowlist`
entry, so the repo knew. A wider sweep of the op surface finds **17 fallbacks,
identical on f32 and f64**:

    rsqrt  sin  cos  tan  asin  acos  atan  sinh  cosh
    erf  erfc  cbrt  expm1  log1p        (14 unary)
    atan2 (both scalar and same-shape forms)
    sort                                  (allowlisted, deliberate)

**Only `sort` is on the allowlist. The other 16 are not.** Strict mode would
raise on every one of them — it passes because the test suite never calls them.
That is the day's pattern once more: a green run over an unexercised path is not
evidence, and `scripts/strict_test.sh` returning 0 failures means "no unlisted
fallback in the tested paths", never "no unlisted fallbacks".

Covered already, for contrast: `exp`, `log`, `sqrt`, `tanh`, `logistic`, `abs`,
`negate`, `sign`, `floor`, `ceil`, `round`, and every binary/reduction/shape op
swept. So the unary shader has the easy transcendentals and is missing the
trigonometric, hyperbolic and error families.

**Most of these are cheap for f32.** GLSL.std.450 provides `Sin`, `Cos`, `Tan`,
`Asin`, `Acos`, `Atan`, `Atan2`, `Sinh`, `Cosh`, `InverseSqrt`, `Log1p`-able
forms and `Pow` — so the f32 arms are op-code additions to
`elementwise_unary_f32.comp`, the same shape of change as the `pow` fix in
`cf7b689`. `erf`/`erfc` are NOT in GLSL.std.450 and need a polynomial
approximation or a documented host path. f64 has no transcendentals at all and
must boundary-cast, exactly as `pow_f64` does.

**RESOLVED for f32, deliberately NOT for f64.** Twelve arms added to
`elementwise_unary_f32.comp` at codes 17-28 — sin, cos, tan, asin, acos, atan,
sinh, cosh, rsqrt, cbrt, expm1, log1p. f32 fallbacks go **17 -> 5**: `erf`,
`erfc`, `atan2` (both forms), `sort`.

f64 keeps the host path and stays at 17, which is a decision rather than an
omission. The f64 shader HAS those arms and routing to them works, but:

    Nx.sin(Nx.tensor(1, type: :f64))
      host  0.8414709848078965      full f64, ~1e-16
      GPU   0.8414708971977234      f32 boundary cast, ~1e-7

Nx documents the first. Admitting f64 turned **22 of Nx's own doctests red**,
and excepting them would cost an f64 caller nine digits to save a host round
trip on ops nothing here calls in a hot loop — while renumbering every later
doctest and invalidating the 78-entry residency register for ops that never
moved. `exp/log/sqrt/sigmoid/tanh` keep their f64 boundary cast because that is
a standing decision with `grad_test` tolerances calibrated around it; new ops do
not inherit it by default.

Still open, and each needs to reach the GPU or the allowlist: `erf`/`erfc` (not
in GLSL.std.450 — needs a polynomial that agrees with `:math.erf`, and note the
f64 shader's old `erf_approx` was deleted as unreachable), `atan2` (GLSL has
`Atan2`; needs a new binary op code across four shaders), and the twelve f64
forms above.

### 2. Pool or free-list the output buffers

Allocation is ~0.59 ms at 64 MiB, clock-invariant, and paid per op. Above the
32 MiB cliff it is a dedicated allocation every time. This is now the largest
identified cost in the elementwise path that is not physics.

### 3. Re-examine anything else measured without a clock record

Four designs are recorded below as failed, and the estimator faults found in
them were real. But the throughput figures they were reasoning about were
taken the same way `ew.exs` took its number. The Race 1c disagreement between
mac-247 and mac-248, in particular, deserves a re-read now that clock state is
recordable — it was never a controlled variable.

### 4. Jetson's below-cliff `alloc_buffer` is bimodal, not slower

24 and 28 MiB read 0.109-0.122 ms; 26 and 30 MiB read 0.150-0.342. Two
populations at the same size. Unchanged from last session, and now more
interesting: if allocation is clock-invariant on Ampere, a bimodal allocation
cost on the Tegra is unlikely to be DVFS either.

### 5. The chain path: the blocker is NOT f32-vs-f64, and the fix is local

This doc has recorded the blocker as "the templated path in `ShaderTemplate`
emits f32 while the active NIF wants f64". Both are f32-capable. The real
situation, established by reading all three sides:

**The NIF pushes a FIXED-SIZE parsed struct, not the caller's bytes.**
`push_constants(layout, 0, push_block)` sends `sizeof(PushBlock)` = 20 bytes
(f32) or `sizeof(PushBlockF64)` = 24 bytes (f64), laid out
`{k_steps, n_obs, d, _pad, eps}`. Anything a caller puts in the push block
beyond that header is silently dropped.

So the contract these NIFs actually implement is: **a fixed header in push
constants, everything else in buffers.**

* **exmc obeys it.** Its synthesised shaders declare exactly that 24-byte block
  and carry priors in the `extras` buffer. It has always been self-consistent,
  and it dispatches its OWN shaders — never `glsl/leapfrog_chain_*_f64.spv`.
* **nx_vulkan's own shaders did not.** `glsl/leapfrog_chain_normal_f64.comp`
  declared `{uint n; uint K; double eps; double mu; double sigma}` = 32 bytes
  with family parameters INLINE. `mu` and `sigma` were never forwarded, and the
  header disagreed besides — the shader's `n` was the DIMENSION at offset 0,
  where the NIF writes `k_steps`. **RESOLVED**: all six ported onto the
  templated path as `Nx.Vulkan.ChainShaderSpecsF64` and verified bit-exact
  against a host leapfrog; the hand-written `.comp` and their shipped `.spv`
  were deleted on 2026-09-01 rather than repaired, because a baked shader is
  parameter-specific and there is no static artifact to replace them with.
* **`ShaderTemplate` and `ChainShaderSpecs` have the same shape**:
  `{uint n; uint K; float eps; <family fields>}`, family params inline,
  `beta_push/6` packing to match.

This is not an offset bug. It is a design difference, and it means the shipped
chain shaders and the whole templated path are **structurally undriveable** by
the NIFs in this repo. Nothing here calls those NIFs — the `synthesis.ex:33` and
`node.ex:44` references are inside `@moduledoc` blocks — so nothing ever tripped
it.

**The fix is local and breaks nothing downstream.** Move family parameters out
of the push block and into a buffer, matching the design exmc already proves
works, and align the header. That is a change to `ShaderTemplate`,
`ChainShaderSpecs` and the six shipped `.comp` files — all inside nx_vulkan,
with no contract change for exmc, which never touches any of them.

Doing that would also give this repo its first working chain caller, which is
what would let it verify its own chain NIFs. `8cce91c`'s batched readback had to
be measured by exmc for exactly this reason.

**Landed meanwhile:** `5693ddf` bounds `d` by `q_init.len()` in all three chain
NIFs, so a layout disagreement returns `:size_mismatch` instead of requesting a
multi-gigabyte allocation. It picks no layout.

**Rejected, for two independent reasons:** deriving `d` from `q_init.len()`
instead of reading it from the push block.

1. It does not make the templated path work. The NIF would still forward only
   the fixed header and drop the inline family params. Found by reading the
   `push_constants` call after the compiler rejected the edit.
2. **It would be a regression for padded callers.** The parsed `d` is not just
   used for sizing — it is pushed to the shader, which indexes with it
   (`q_chain[k * pc.d + i]`). Sizing from `push_block.d` therefore agrees with
   the shader BY CONSTRUCTION. Deriving from the buffer instead would decouple
   the two: a caller that legitimately pads `q_init` gets outputs sized for the
   padded width while the shader writes the unpadded one — over-long binaries
   with trailing garbage, a silent wrong answer. Today that caller is correct.

exmc confirmed empirically at d=1 and d=3 that its `q_init` is exactly `d*8`
bytes, and green-lit the derivation. It is still the wrong change: their
confirmation removes the risk to *them*, not the reason the design is worse.

**So the correct action here was no code change beyond `5693ddf`'s guard**, and
`<=` is the right bound — it catches a misread `d` (which is wild, ~1.2e9) while
permitting padding, which is legitimate and works today. Tightening to `==`
would reject a currently-correct caller for no gain.

**Verified while in there:** `logp_chain` is sized `K` (and `n_instances * K`),
never `K*d`, in all three NIFs. exmc flagged this as a place where a `K*d`
assumption would silently over-allocate by a factor of d, invisible at d=1. The
code is already right; noted so nobody 'fixes' it.

### 5b. Race 5 (MCMC) has never run

Nothing in this repo calls `leapfrog_chain_synth_f64`. The shipped chain shaders
are consumed by eXMC downstream, and the templated path in `ShaderTemplate`
emits f32 while the active NIF wants f64. The call convention has to be
established on one box before four boxes run it. The design is still worth
keeping: the chain shader returns the whole trajectory — `3*K*d*8` bytes down
against `2*d*8` up.

### 6. Race 1c voids on the Jetson even on a QUIET box

Re-raced there at `d7b5f08` in a window with load 0.40-0.73 across 14 samples
taken every 10s during the run, thermal control compute drift 0.0% and
allocation drift 0.7%. It still voided: **estimator divergence 30.7% against a
30% gate.** A contended run earlier the same evening read 177.8% divergence and
32.7% allocation drift, so contention inflates it ~5.8x — but clean, it sits a
few points over the threshold.

That is a statement about the instrument, not about any commit: the gate and the
method are within a few points of each other on integrated hardware. It is the
evidence `scripts/staged/jetson_run_trace.sh` was staged to gather, and it now
argues the trace is worth running rather than merely staged.

**Harness gap found doing this:** a VOID run still writes
`bench_results/unified_vs_discrete_<host>.json`, and the file carries no void
marker — the verdict goes to stdout only. `load_after` is recorded but nothing
says the run was rejected, so a later reader takes it as a result. The harness
should refuse to write on a void, or stamp the file. I backed up the baseline
before racing and restored it; the next person will not necessarily think to.

### 7. The Jetson can now be built for in ~2 minutes, not ~47

`.claude/skills/jetson-cross-build/` (untracked as of this writing) cross-builds
the aarch64 NIF in a container on super-io: **1m55s against ~47 min native** on
that 2-core board. Validated end to end at `d7b5f08` — deployed, loaded
(`NVIDIA Tegra X1 (nvgpu) (IntegratedGpu)`), and passed the full suite.

Deploy rules, all of them load-bearing:
* `priv` is a symlink in both `_build/dev` and `_build/test`, so overwriting
  `priv/native/libnx_vulkan_vulkano.so` covers every environment.
* Run with `--no-compile` or Rustler rebuilds and silently replaces it.
* Checksum before AND after; that is the only proof the artifact under test is
  what ran.
* The real ABI bar is **GLIBC** (max symbol 2.25 against the box's 2.27), not
  LSE atomics. The artifact contains 12 LSE instructions in
  `compiler_builtins`' outline-atomics helpers behind a HWCAP guard that reads 0
  on that board — and the Jetson's own native build has MORE of them (20) while
  passing today. No stable-Rust build can avoid them.

### 7. super-io's poison flip rate is still unmeasured

Two samples, both 20/20, and one showing 19/40 effectiveness with 20/20 padding
— which breaks the lockstep claim. The cross-box table is already retired.
Closing this properly needs ~8 super-io runs.

---

## What is settled, so nobody re-litigates it

* **The 32 MiB allocation cliff is vulkano's**, not any memory architecture's —
  reproduced 6/6 across every box and commit.
* **The BAR1 cliff is a different cliff.** 256 MiB, cumulative across live
  uploaded buffers, host-visible heap exhaustion. Do not conflate the two: one
  is per-allocation and vulkano's, the other is a whole-process budget and the
  driver's.
* **The elementwise shaders are not a bottleneck on Ampere.** 431 GB/s of 448.
* **`buf_alloc` is not clock-bound; shader work is.** 1.1-1.4x against 3.5-4.7x.
* **The unified-vs-discrete question is unanswered and four designs failed.**
  Race 1 measured a submission-to-throughput ratio while claiming arithmetic
  intensity; `a` was a mixture; `s` was an ill-conditioned intercept; PRICE
  tracks GPU speed because its two terms are different physics.
* **The control pair is not a pair.** mac-247 and mac-248 differ 1.39x on
  submission cost, in the opposite direction from their per-dispatch GPU work.
* **DVFS is on all three architectures**, and pstate cannot certify a cold
  measurement in either direction. What it CAN detect is a foreign client:
  6 MiB resident + P8 sustained means someone else has the card.
* **The poison-control rate is not a cross-box observable.** It moves with the
  commit.

---

## Harness invariants worth not breaking

Learned expensively, all of them:

* **Record the GPU clock for every timed quantity, from outside the process.**
  Fourth incident. An unrecorded clock turned a 210 MHz reading into a hardware
  finding that survived into a planning document.
* **Never poll `nvidia-smi` inline.** The instrument idles the GPU it measures.
* **Sanity-check every derived figure against a physical bound.** The corrupted
  run announced itself by reporting 804 GB/s on a 448 GB/s card. Without that
  check it would have read as a merely noisy result.
* A control must re-measure the **kind** of work it certifies. A matmul cannot
  vouch for an allocation; that cost three near-misses.
* Warm the clock before timing, and warm **both sides** of the 32 MiB cliff.
* GC inside any loop that allocates — a retained-allocation leak has now
  appeared three times.
* Thresholds must match the measured noise floor of their own quantity: compute
  10%, allocation 40%, estimator divergence 30%.
* Prefer slopes to intercepts, and robust estimators to hand-rolled rules.
* Replicate. Every estimator defect in the last four rounds was visible only in
  a second run.
* **Build large test tensors with `from_binary`, not `Nx.iota`.** Host-side
  tensor construction is GPU idle time and will silently unboost your card.
* **Amplify an effect until it clears the noise floor, then verify BOTH
  directions.** A ~0.16 ms per-call cost measured one call at a time sat inside
  34% process-to-process variance on the SAME binary. Putting 30 calls in each
  timed sample fixed it. And when verifying a fix, the broken arm must still
  separate — "fixed looks like pre" proves nothing if the harness cannot
  resolve broken from pre either.
* **Rotate arm order in an A/B.** Whichever arm runs first after a binary swap
  eats the cold start; a fixed order silently charges it to one arm.
* **Pick the measurement host by CONTENTION, not by clock observability — and
  they are anti-correlated.** super-io is the only box where `clocks.sm` reads,
  which is why four DVFS incidents pushed every measurement onto it. It is also
  a DESKTOP: Firefox and Cinnamon composite on that GPU throughout, and
  `nvidia-smi --query-compute-apps` shows nothing, so the contention is
  invisible to the check this project uses. The exmc session measured the same
  chain benchmark on both boxes, N=3000/sample, 6000-dispatch warmup, 6
  replicates:

      super-io  RTX 3060 Ti   822 .. 1741 us/dispatch   57% of median
      mac-248   GT 750M       364.5 .. 365.4 us          0.3% of median

  The headless Kepler is faster in absolute terms AND three orders of magnitude
  tighter. super-io's noise band is ~900 us wide, so anything under ~1 ms is
  unmeasurable there. **Use a headless box.** The clock-observability argument
  that put measurements on super-io selected for the worst available host.
* **A control that should show nothing must actually show nothing.** The
  small-buffer fast path A/B put its over-threshold size in both arms, where the
  change cannot act — and that control reproduced 4.6% of the 7.2% "treatment"
  effect. Without it the result would have read as a win. Build the null arm
  into the experiment, not the interpretation.
* **Warmup can look exactly like a leak.** The exmc chain benchmark read
  625 -> 902 -> 969 us across its first three replicates at a 300-dispatch
  warmup, which reads as a retained-allocation leak — a false alarm this project
  has now had three times. At 6000 dispatches it settles. Discard early
  replicates before diagnosing a trend.
* **`pgrep -f "foo"` matches the shell running it.** A wait loop built that way
  never fires. Key on a pid via `/proc`, or use `pgrep -x`. Documented in
  `scripts/staged/jetson_run_trace.README` and walked into anyway.
* **`mix test` reads stdin.** Inside `ssh 'bash -s'` with a heredoc it swallows
  the rest of the script, so trailing verification lines never run. Redirect
  with `< /dev/null`.
* **A swapped-in `.so` needs `--no-compile` AND a checksum.** Rustler rebuilds
  on the next `mix compile` and would silently replace the artifact under test,
  handing you a green suite that proves nothing about it.
* **`mix run` and `mix test` use DIFFERENT `_build` trees.** `mix test` compiles
  into `_build/test`; `mix run` reads `_build/dev`. So `mix run --no-compile`
  after a green `mix test` runs STALE code, confidently. This produced a census
  claiming f64 ops were resident when the source said otherwise and the test
  env agreed with the source — caught only because the two disagreed. Run
  `mix compile` before any `mix run --no-compile` probe, or drop `--no-compile`.

### Disclosure for every number in this document

Taken on super-io with a live desktop session on the same card — firefox and
cinnamon, 2.1-2.6 GiB resident, P0 throughout. No foreign *compute* client (the
6 MiB + P8 signature was absent), but this is not a clean box, and the previous
document's own rule about checking `pstate,memory.used` reads dirty here by
construction. The clock-state separation was verified out-of-process and is
unambiguous: idle windows held 210 MHz, boost windows held 1920-1935 MHz.

**This caveat turned out to be the headline, not a footnote.** The desktop
contention is worth ~900 us of noise (see the host-selection invariant above),
which is wider than most of the per-call effects measured here. Read every
number in this document accordingly:

* The elementwise slope figures (431 GB/s etc.) are large-signal and survive.
* The DVFS finding survives — a 2.6x swing is far outside the band.
* **~0.16 ms per submit-and-fence is an estimate taken inside noise of
  comparable width, not a measurement.** It separated because the effect was
  ~0.135 ms and consistent, but do not size anything from its magnitude. It was
  used to predict a chain-path cost and the prediction was out by 3x.
* The small-buffer fast path was re-run on a headless box and IS established:
  **~14.5 us/dispatch, 6.1%**, non-overlapping across order-balanced replicates
  (see below).

---

## Publishing (operator decision, not done)

* `~/projects/learn_erl/pymc/www.dataalienist.com` — two posts committed and
  **not deployed**: "An Absence Mistaken for a Discovery" (the correction) and
  the correction banner on "The Copy That Wasn't There".
* `mix hex.retire nx_vulkan 0.2.0`, `upstream/main` publishing, consumer pin
  bump — all still outstanding from before this sequence.
