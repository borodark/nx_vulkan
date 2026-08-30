# Unified memory vs dedicated GPU RAM — Races 1, 1b and 4

**Status: NO ANSWER YET, and the control arm says why.**
Runs at `d4ca422`, four boxes, replicated. Harness `examples/unified_vs_discrete_race.exs`.

---

## The result that decides whether there is a result

The experiment's design rests on the two Keplers agreeing. They are the same
architecture, both discrete, both idle, both holding P0 throughout — so the
spread between them is the noise floor against which any Jetson difference must
be judged. They do not agree.

| box | s (submission, ms) | c (per output element, ns) | within-box spread on s |
|---|---|---|---|
| mac-247  GT 650M | 0.1603 / 0.1496 | 3.611 / 3.645 | 6.9% |
| mac-248  GT 750M | 0.0814 / 0.0788 | 3.559 / 3.543 | 3.2% |
| jetson   Tegra X1 | 0.6804 / 0.7767 | 22.30 / 22.25 | 14.2% |
| super-io RTX 3060 Ti | upper bound only | 0.38–1.06 | — |

**The two controls differ on `s` by 1.9x while agreeing on `c` to 1.7%.** Their
total dispatch floors differ by only ~12% (a = 0.56 vs 0.50). So the *sum* is
consistent across the pair and the GPU-work term is consistent, but the split
between submission and fixed GPU work is not.

The Jetson's `s` is 5–9x the Keplers', which looks like a large effect. It is
not reportable, for three independent reasons, any one of which is sufficient:

1. **The control pair disagrees by 1.9x**, so there is no scale on which to
   judge a difference.
2. **Submission is host work**, and the Jetson's host is `--disable-jit` on 2
   cores at 5 W. A slow host inflates `s` with identical memory architecture.
3. **`s` is an extrapolated intercept**, biased upward by residual quantisation
   floor by 10–20% on mac-248 — and by a *different* amount on each box, since
   the floor's extent is a property of that box's tile geometry.

---

## What the sweep can and cannot reach

Doubling n quadruples the output, so a point in the true n² regime is 4.00x the
one below it. Anything well under 4 is still on the tile-quantisation plateau.

    box        64->128  128->256  256->512  512->1024  1024->2048
    mac-248      1.41      2.17      3.21       3.67       3.98
    super-io     1.03      1.20      1.24       2.68       2.60

mac-248 reaches the regime near n=2048. **super-io never reaches it at all** —
its `s` fit exceeds its own smallest measured dispatch, which is impossible
since submission is a component of every dispatch, so the harness now reports it
as an upper bound rather than a measurement.

The fast box would need n >= 4096. Its 64 MiB output sits **above the 32 MiB
allocator cliff**, so Race 1b cannot simply be extended into it without
confounding Race 1b with Race 4's regime. That is a design problem, not a
tuning problem.

---

## The control-pair test: half answered

`s_flush` — per-submission cost, measured as the median of adjacent marginal
slopes over a fixed 32 dispatches split across a varying flush count. A slope,
not an intercept, so it does not depend on extrapolating to zero, on the
quantisation floor, or on which points enter a fit.

    mac-247   4f23e59 A   532.9 us
              4f23e59 B   522.1 us
              6848e19 P   549.6 us
              mean        534.9 us, 5.1% spread over three valid runs

Against `s`, which disagreed 1.9x between the two Keplers, and against OLS on
the same three runs (532.8 / 412.5 / 518.0), that is a usable quantity.

**mac-248's `s_flush` is the missing half.** If it lands near 535 us the control
pair agrees and the Jetson becomes judgeable for the first time. If it lands at
half or double, the disagreement is deeper than the estimator and Race 1c has
not fixed it either.

super-io is out as a reference: 355 / 622 / 767 us across three runs at load
3-5.

### VERDICT: the pair disagrees, and it is not an artifact of the estimator

    mac-247   532.9 / 522.1 / 549.6 us   mean 534.9
    mac-248   396.5 / 372.0 us           mean 384.2
    gap       1.39x — far outside 247's 5.1% and 248's 6.4%

Race 1c did not rescue the control pair. It did something more useful: it made
the disagreement diagnosable.

**mac-248's discriminator settles what `s_flush` is measuring.** `base` is the
per-dispatch record and GPU work at a fixed dispatch count, so if `s_flush` were
still a mixture the box with more GPU work would carry the higher `s_flush`.
The opposite holds:

    box       base/dispatch    s_flush
    mac-247      0.585 ms       535 us
    mac-248      0.730 ms       384 us

248 does **1.25x more GPU work per dispatch and has 1.39x LOWER submission
cost**. The two move in opposite directions, so `s_flush` is separating
something real from GPU work — the disagreement lives in the submission path
itself, not in a residual mixture. That is the decomposition working and the
control pair failing, at the same time.

**"Same architecture" was never "same hardware", and that assumption was mine.**
248 is right to flag it: a GT 650M and a GT 750M are different SKUs on different
Mac hosts, and `s_flush` is a host-and-driver-dominated quantity. Agreement to a
few percent was an assumption stated as a null hypothesis.

**The pstate difference is real and is a candidate mechanism.** Under an
identical 20-minute idle protocol, 41 consecutive samples, thermally settled at
56 C by t+9:30, mac-248's GT 750M **never leaves P0** while mac-247's GT 650M
reaches a stable P8. If 247's card sits in P8 between measurements and 248's
never does, the two are not interchangeable controls — 247's clocks may still be
ramping through part of its measurement. Sampling pstate *during* Race 1c rather
than at idle would settle it.

Excluding F=1 (below) tightens both boxes' within-run spread but does **not**
close the gap: at F>=2 the pair reads ~528 vs ~366, a 1.44x ratio. The
disagreement is robust to the estimator.

### The criterion, stated before 248's number is known

mac-247 proposed this while its own result was the only one in hand, and it is
recorded here before the comparison exists so the threshold cannot be chosen to
suit the answer:

* **Within ~10% of 535 us → the control pair is consistent**, and the Jetson
  becomes judgeable for the first time.
* **Outside ~10% → do not conclude the pair disagrees yet.** Take a fourth
  mac-247 run on a verified-quiet window first.

The asymmetry is deliberate and is 247's own reasoning: its 5.1% is three runs
rather than a distribution, and two of them came from the same session minutes
apart. It also now knows first-hand how easily third-party GPU work hides inside
a nominally idle reading — its second pstate error was *more* confident than its
first precisely because it had fresh data and did not ask where the data came
from. A disagreement is therefore the claim that needs the extra run, not the
agreement.

## What IS established

**The 32 MiB allocator cliff is vulkano's, not any memory architecture's.**
Reproduced 6/6 across every commit and every box: `buf_alloc` flat at
0.006–0.14 ms below it with a slope of 0.0000, then a step of 226x–3400x at
exactly 32 MiB. Discrete PCIe and unified LPDDR4 alike, two operating systems,
three GPU generations.

**Race 4 below-cliff is now reportable; above-cliff is not.** At 25 reps
`zeroed_below` replicates to 1.9% (it was 73% at 9 reps). `zeroed_above` did
**not** improve — 1.88x between two quiet consecutive runs at 25 reps — so the
variance is not sampling noise but something about allocator state above the
dedicated-allocation threshold. It is excluded from all conclusions.

**DVFS on Kepler: UNKNOWN, and pstate cannot settle it. This section has now
been wrong twice in opposite directions — both corrections are below.**

**Correction 2 (current).** The retraction below is also wrong. With the
third-party job gone, the GPU at 0 MiB and no clients, mac-247's Kepler reads a
stable **P0** for 54+ seconds at 56 C. Its resting state with no client is P0.
The P8 readings that motivated correction 1 were recorded while a third-party
job was cycling the GPU — it was observed driving P8→P0→P1→P5→P8 with
temperature swinging 59→69→59 C and 6 MiB resident — and were misattributed to
"the GPU left alone for 15 minutes". The box was busy.

So neither previous statement holds. The field is not inert (it takes P0, P1,
P5, P8), but because the no-client resting state IS P0, **observing P0 tells you
nothing about whether the GPU was boosted or cold**. Pstate on this
Kepler/470/FreeBSD combination cannot certify a cold measurement in either
direction. The original caution was right for the wrong reason.

My "an 80-second idle sample is not an idle sample" framing does not hold
either: a genuine ramp-down was captured twice inside the contaminated window
and runs P0→P1→P5→P8 in ~10-15 s. When it happens it is fast — it just requires
a client present.

Anyone re-sampling pstate must first verify no other process is touching the
GPU, or they will reproduce this mistake.

**Correction 1 (superseded, kept because the error is instructive).**

The claim was that Kepler holds P0 through 80 s of idle sampling and never
enters a low-power state, so the boosting parts were Ampere and Tegra and the
one that did not was Kepler — cutting across the discrete/unified split. That
was wrong, and mac-247 retracted it against its own earlier evidence.

Left alone for ~15 minutes, that Kepler reads a stable **P8**, sampled five
times. Every prior reading was taken within a couple of minutes of a GPU
workload, i.e. inside the ramp-down window, which is why they were all P0. An
80-second idle sample is not an idle sample on this hardware. The field was
never inert; a null was over-read.

Consequences:

* Clock pinning is **load-bearing on Kepler too**, not the no-op predicted.
* The −10.8% that pinning moved on that box's small-k plateau (against +0.11% on
  its throughput region) was attributed to a warm-pipeline effect on the grounds
  that "a clock change would scale the throughput region too". With DVFS
  present that is no longer the only candidate — a P8→P0 transition plausibly
  hits the dispatch floor harder, since a long compute-bound dispatch has time
  to ramp within itself and a 1 ms one does not. **Unresolved.**
* What survives is the operational conclusion, on better grounds: prefer `s` and
  `s_flush` over any empirical floor, because the floor is what moves — not
  because the pstate field is useless, but because these boxes genuinely change
  power state and cold floors are therefore untrustworthy.

mac-248's P0 readings carry the same defect and want re-sampling after a long
idle before its pstate data is used for anything.

**Neither discrete box is near a roofline** — 1.3% and 0.4% of peak f32. The
k-sweep does not walk an arithmetic-intensity roofline; "arithmetic intensity"
oversells what it isolates.

---

## Eleven harness defects, and what each would have produced

Every one biased the answer. Three were introduced while fixing the previous one.

| defect | what it would have shown |
|---|---|
| Single warmup vs GPU clock idling to 10% | 288% "thermal drift" on an idle box; unified memory winning short bursts because the Ampere was asleep |
| Warm loop never GC'd | Race 4 dead on both Keplers; on an 8 GB card, silent contamination that still printed RACE: OK |
| Rule 6 — "normalising cancels a slow host" | Host was 50–61% of measurement below k=32; the left of every curve was Nx frontend |
| Endpoint slope estimator | Kepler/Ampere zeroed ratio 9.9x vs 4.8x — a factor of two from the estimator |
| Difference of medians on a bimodal host | Up to the whole 0.6 ms mode gap, 50% of GPU time at k=4 |
| Tile quantisation unflagged | The low-k end — where unified memory should appear — is measuring padding |
| `System.cmd` raises on missing binary | Harness could not run on the treatment box at all |
| Negative-slope check on an O(1) series | Voided two healthy runs |
| Adaptive floor detector | Flipped its own answer between replicates minutes apart |
| Host normaliser as median of bimodal, n-dependent draws | 33% step from the draw alone; bias the size of the effect |
| n=2048 added "for leverage" | Top point dominates OLS leverage; steeper bandwidth-bound slope extrapolated to zero inflates the intercept |

---

## RACE 2 RESULT: the within-box ratio does not isolate memory architecture either

Three boxes, replicated, with the DVFS confound measured directly on the two
that could show it:

    box                    memory     c ns/el   PRICE >=1MiB   low-to-mid rise
    super-io RTX 3060 Ti   discrete      1.06        3.05           2.34x
    mac-247  GT 650M       discrete      3.61        1.22           0.74x
    jetson   Tegra X1      UNIFIED      22.30        0.97           0.86x

**mac-247 is discrete and it groups with the unified Jetson, not with the
discrete Ampere** — on both discriminators. Two unrelated discrete cards do not
agree on the shape, so the shape belongs to super-io rather than to paying a bus.
That is the control doing its job, and the answer is negative.

### Dividing out GPU throughput does not change the answer — and shows why

Both Race 2 terms move the same bytes (the crossing is upload+download of
`size`; one op is read+write of `size`), so expressing each as a **bandwidth**
divides out GPU compute throughput by construction. That is the only
normalisation available which does not compare a bandwidth against a FLOPs rate.
At 16 MiB:

    box                    memory    cross GB/s   "dev" GB/s   ratio
    super-io RTX 3060 Ti   discrete      5.63        16.01      0.35
    mac-247  GT 650M       discrete      6.64         6.05      1.10
    mac-248  GT 750M       discrete      7.91         6.31      1.25
    jetson   Tegra X1      UNIFIED       2.07         2.02      1.03

The ordering is unchanged: the two discrete Keplers still sit with the unified
Jetson and super-io is still the outlier. Normalising properly did not rescue
the comparison.

**And the reason is now measurable. The denominator is not a bandwidth.** A
direct probe of the elementwise path on super-io, GC'd per iteration:

    4 MiB   0.633 ms/op   12.3 GB/s
   16 MiB   1.907 ms/op   16.4 GB/s
   64 MiB  40.382 ms/op    3.1 GB/s   <- output crosses the 32 MiB cliff

It plateaus at **16.4 GB/s against a 448 GB/s card — 27x off memory
bandwidth** — and collapses above the allocation cliff. So `dev GB/s` is a
dispatch-bound quantity wearing bandwidth units, and it differs across boxes for
reasons that have nothing to do with memory architecture.

That is precisely why the Keplers read ~1.0. It is a **coincidence**: their
elementwise path happens to be about as slow as their PCIe link, so crossing and
compute land together. The Jetson's ~1.0 is meaningful — same physical RAM — but
the measurement cannot distinguish "same memory" from "compute as slow as the
bus", and on this fleet three of four boxes are in the second category.

**So the blocker is not the experimental design any more. It is that
nx_vulkan's elementwise path runs 27x below memory bandwidth.** Until the
denominator saturates, no ratio built on it can separate a memory architecture
from a slow shader. That is a finding about the backend, and it is actionable in
a way the race never was.

### mac-248 confirms it independently, and sharpens the mechanism

    box                    64 KiB   1 MiB   16 MiB   growth
    super-io  discrete       1.24    2.90     2.84     2.30x
    mac-247   discrete       1.36    1.00     1.34     0.68/1.29x
    mac-248   discrete       1.21    0.65     0.79     0.65x  (replicates 0.0%)
    jetson    UNIFIED        1.18    1.01     0.94     0.75-0.85x

Its growth factor lands on **0.65x twice**, so the statistic is solid and the
disagreement with super-io's 2.30x is not noise. Two discrete Keplers, measured
independently, both contradict the discrete Ampere.

**The striking detail is where they agree.** At 64 KiB every box reads ~1.2 —
super-io 1.24, mac-248 1.21, mac-247 1.36, jetson 1.18 — and they diverge
monotonically from there. 248's explanation: at 64 KiB both terms are dominated
by fixed dispatch and submission overhead, which is similar everywhere. As size
grows `boundary` becomes bus-bandwidth-bound while `compute` becomes
GPU-bandwidth-bound, so the boxes separate according to their own bus-to-GPU
ratio — not according to whether a bus exists.

Its algebra states the flaw exactly:

    PRICE_1 / PRICE_2 = (bus_1 / bus_2) x (op_2 / op_1)

The denominator is GPU strength, undivided. Its Kepler runs 9.3 GFLOP/s against
super-io's 73 — 7.8x slower — which alone pushes its PRICE far below, and does.

**PRICE remains a genuinely useful WITHIN-box number.** It answers "should I
chunk transfers on this machine" directly, and on mac-248 the answer is that a
crossing costs less than one op at every size above 64 KiB. It is simply not a
box-independent property of a bus.

### Why: PRICE inherits the same flaw as `a` and as rule 6

PRICE is `boundary / compute`, and the cancellation argument was that host, GPU,
driver and SKU appear in both terms so they divide out. **They do not, because
the two terms are different physics.** The boundary term is bus- or
memcpy-bandwidth-bound; the compute term is GPU-throughput-bound. A fast GPU
shrinks the denominator without shrinking the numerator, so it raises PRICE with
no change in memory architecture whatsoever.

The data says exactly that. Compute per element spans **21x** across the fleet
(1.06 → 22.30 ns) and PRICE spans **3.1x in the opposite direction**. PRICE
tracks GPU speed, and the discrete/unified split does not predict it.

This is the third instance of the same error, all mine:

* **rule 6** — "normalising to a box's own baseline cancels a slow host". False:
  host cost is additive, not multiplicative.
* **`a`** — assumed to be submission cost. Actually a mixture of submission and
  fixed GPU work, which is why it scaled at 14x where throughput scaled at 23x.
* **PRICE** — assumed to cancel because both terms share a machine. They share a
  machine and not a mechanism.

Each time the mistake was asserting that terms appearing on both sides cancel,
without checking they were the same kind of quantity.

### What the boxes established along the way

* **The per-arm clock confound is real on discrete hardware.** mac-247 measured
  its round-trip arm spending 65-75% of its time in P1 at 1 MiB while the
  resident arm never left P0, reproducible across two probes. At 16 MiB both
  arms hold P0. The Jetson measured a flat 614.4 MHz in every arm at every size.
  Direction is conservative in both cases: a sagging clock in the low-duty arm
  inflates PRICE, so both boxes' low values are upper bounds.
* **End-to-end growth is not a usable statistic.** mac-247's two largest sizes
  replicate at 57% and 64%, with one large point corrupted per run in opposite
  positions — end-to-end growth read 0.68x and 1.29x on the same box. The 64 and
  256 KiB points replicate to 2.2% and 4.9%, so the low-to-mid rise is the part
  that survives. The harness now reports both, labelled.

## The pstate question: BOTH Keplers have DVFS, and both agents over-read a null

mac-248 has withdrawn its own "this card never leaves P0" claim. It found the
card at P8 at 21:54 and caught a P5 excursion during the Race 2 runs. Its
20-minute idle protocol was too short — **the same category of error as the
80-second sample, made with more confidence than the evidence supported**, and
it says so in those terms.

So both Keplers have DVFS; 248's simply holds P0 longer. It also withdraws the
DVFS explanation it offered for the 247 `s_flush` gap, which leaves that gap
without a mechanism.

It makes one further point worth keeping: **reproducibility is not evidence
against a clock confound**, because a systematic clock difference between the
two arms would itself reproduce. Its 0.0% growth agreement therefore says
nothing about whether the arms sat at different clocks. Settling that needs
pstate sampled *inside* each arm, which means instrumenting the harness rather
than sampling alongside it.

## The pstate question, resolved on the third attempt (superseded by the above)

This document was wrong about Kepler DVFS twice, in opposite directions. The
coherent version is that **pstate alone cannot certify anything, but pstate
together with `memory.used` detects a foreign GPU client**:

    0 MiB resident, P0, no beam.smp at high CPU   -> genuinely free
    6 MiB resident, P8, sustained                 -> another client is attached

That is the signature mac-247 misread as "a long idle produced P8". It was a
third-party job holding the card. The card's own resting state with no client is
P0, so observing P0 tells you nothing about whether a measurement was warm — but
observing P8 *with memory resident* reliably tells you someone else is there.

**Check `nvidia-smi --query-gpu=pstate,memory.used` before any timing on this
fleet.** A quiet CPU does not mean a free card.

## Race 2's cancellation argument has a limit, and it is not symmetric

PRICE cancels host speed, GPU speed, driver overhead and SKU because they appear
in both arms — but that holds only if contamination is **symmetric**. A foreign
GPU client is not: it shares the PCIe path, the driver submission queues and the
DMA staging, all of which the round-trip arm uses far more than the resident arm.
So interference inflates `boundary` while leaving `compute` comparatively
intact, and since PRICE is `boundary/compute` the contamination multiplies
straight into the headline instead of cancelling.

mac-247 demonstrated this shape already: its `6848e19` run Q had allocation and
submission both ~2x slow while Race 1's matmuls stayed clean at 1.6% drift. A
one-sided inflation of exactly that kind would produce a plausible PRICE curve
with a wrong growth factor — and the growth factor is the robust part the
prediction rests on.

**So Race 2 needs a genuinely free GPU more than the earlier races did, not
less.** Its within-box design removes the cross-box confounds; it does not remove
this one.

## Staged and unrun: the Race 1c clock trace

`/tmp/run_trace.sh` and `/tmp/run_trace.README` on the Jetson (192.168.0.250),
staged and never fired — the box was held for 95+ minutes by another user's
`mix test` in `~/exmc_oss`, which exercises `nx_vulkan` and therefore contends
for the GPU, the exact resource the question depends on. **These are /tmp files
and will not survive a reboot.**

It samples devfreq at 250 ms with every harness output line wall-clock stamped,
so the clock trace can be sliced into Race 1c's per-F windows (each `F total_ms`
row prints after that F is measured, so F[i]'s window runs from the F[i-1] stamp
to the F[i] stamp). It carries a self-abort guard that re-checks for competing
work immediately before measuring and exits 9 rather than produce a contended
table.

**What it would decide:**

* Clock RISES with F → Race 1c is DVFS-confounded on an integrated part and
  needs redesign: larger n to raise duty, or interleaved saturating work to hold
  the clock across all F.
* Clock FLAT and the marginals still bend → the cause is something else, and
  that is the more interesting answer, because it would constrain the Kepler
  disagreement too.

Reference clock behaviour already measured on that box: idle 76.8 MHz, cap 614.4
MHz; 2.25 s to full clock at 50-65% duty against ~500 ms at 99.7%; and Race 1b
holding 614.4 MHz on every sample at 70-80% duty — which is precisely why Race
1c's lower duty is the open question.

The limitation stands without the trace. A negative marginal is impossible, so
Race 1c cannot be trusted to measure submission on that box at these sizes. The
trace would name the mechanism, not establish the fault.

## What would make this answerable

1. **Fix the control pair first.** Until two same-architecture boxes agree on `s`
   to better than their own noise floors, no cross-box number means anything.
   The 1.9x Kepler disagreement is the blocking issue, not the Jetson.
2. **Reach the n² regime on the fast box** without crossing the 32 MiB cliff.
   Raising `k` rather than `n` makes each output element cost more, so the
   regime starts at smaller n — but changes what is held fixed, and larger
   operands may themselves cost more to submit. Needs thought before it is run.
3. **Race 5 (MCMC) is untouched.** Nothing in this repo calls
   `leapfrog_chain_synth_f64`, and the templated path emits f32 while the active
   NIF wants f64. The call convention has to be established before four boxes
   run it.

**The honest summary: the plan's Race 1 measured a submission-to-throughput
ratio while claiming to measure arithmetic intensity; Race 1b is the right
construct and is not yet precise enough to use; Race 4's one solid result says
the thing everyone assumed was a unified-memory story is a vulkano suballocator
threshold.**
