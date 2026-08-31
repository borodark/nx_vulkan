# Next session — state, open items, and one live bug

**HEAD `6b38aee`**, pushed to `origin` (private). Nothing pushed to `upstream`.
Working tree clean apart from this file. Rewritten 2026-08-31, after a session
that ran on super-io.

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

### What it cost

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

### 1. Verify the `buf_upload` fix across the fleet

Done and verified on super-io only. **The Jetson is the one that matters**: it
takes the `unified` branch, where both `staging_write` and
`enqueue_staging_write` write in place and queue nothing, so it should be a
no-op there — but that is reasoning, not measurement, and this project has been
wrong about the Jetson's memory behaviour before. The Keplers need a plain
re-verify. Not committed yet.

Worth adding at the same time: a probe that asserts which heap a buffer landed
in, since nothing in the suite can currently see this class of bug. The BAR1
column in `scripts/staged/bar1_cliff.exs` is the whole of the current
instrumentation.

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

### 5. Race 5 (MCMC) has never run

Nothing in this repo calls `leapfrog_chain_synth_f64`. The shipped chain shaders
are consumed by eXMC downstream, and the templated path in `ShaderTemplate`
emits f32 while the active NIF wants f64. The call convention has to be
established on one box before four boxes run it. The design is still worth
keeping: the chain shader returns the whole trajectory — `3*K*d*8` bytes down
against `2*d*8` up.

### 6. The Race 1c clock trace, staged and unrun

`scripts/staged/jetson_run_trace.sh`. Decides whether Race 1c is structurally
unable to measure submission on an integrated part.

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

### Disclosure for every number in this document

Taken on super-io with a live desktop session on the same card — firefox and
cinnamon, 2.1-2.6 GiB resident, P0 throughout. No foreign *compute* client (the
6 MiB + P8 signature was absent), but this is not a clean box, and the previous
document's own rule about checking `pstate,memory.used` reads dirty here by
construction. The clock-state separation was verified out-of-process and is
unambiguous: idle windows held 210 MHz, boost windows held 1920-1935 MHz.

---

## Publishing (operator decision, not done)

* `~/projects/learn_erl/pymc/www.dataalienist.com` — two posts committed and
  **not deployed**: "An Absence Mistaken for a Discovery" (the correction) and
  the correction banner on "The Copy That Wasn't There".
* `mix hex.retire nx_vulkan 0.2.0`, `upstream/main` publishing, consumer pin
  bump — all still outstanding from before this sequence.
