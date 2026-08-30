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
