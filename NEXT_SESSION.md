# Next session — state, open items, and what a reboot costs

**HEAD `c166c13`**, pushed to `origin` (private). Nothing pushed to `upstream`.
Working tree clean. Written 2026-08-31, before a reboot.

---

## Where things stand

The backend has a real fix in it. `alloc_buffer` requested
`PREFER_DEVICE | HOST_RANDOM_ACCESS` — a preference and a requirement, and the
requirement wins — so every output buffer lived in system RAM and **every store
the shader executed crossed PCIe**, measured directly at a sustained 10.8 GB/s
during computations that transfer nothing.

Compute buffers are now `DEVICE_LOCAL`, host access goes through staging, and
staging is **gated on a unified-memory probe** because it costs the Jetson
47-152% per crossing for zero gain.

    box        elementwise 64 MiB    allocation above cliff
    super-io   3.1 -> 35.1 GB/s      11-22 ms -> 0.64 ms
    mac-247    1.61 -> 4.75 GB/s     23-30 ms -> 4.9 ms   (nothing below cliff)
    mac-248    2.05 -> 5.00 GB/s     ~20 ms -> 4.0 ms     (nothing below cliff)
    jetson     unchanged (correctly — staging OFF there)

All four boxes: **833 doctests / 871 tests / 0 failures, strict 163 excluded,
residency 755/833 (90.6%)** — unchanged throughout, across five NIF revisions.

---

## THE REBOOT COSTS

`/tmp` is wiped. Rescued into `scripts/staged/` already:

* `jetson_run_trace.sh` + `.README` — the Race 1c devfreq trace, staged on the
  Jetson and **never run** because the box was contended for 95 minutes. The
  README carries the box's setup traps and the `pgrep -f` self-matching hazard.

Not rescued and probably not worth it: per-box probe scripts on the Keplers
(`/tmp/ew.exs`, `/tmp/lim.exs`, `/tmp/vram.exs` on 248; `/tmp/dpf_probe.sh` on
the Jetson). All are reconstructable from the reports.

**If super-io is the box rebooting**, check the driver afterwards. There is a
recorded history of an nvidia kernel-module/userspace version mismatch dropping
it to llvmpipe, which gives *wrong u8 answers*. Confirm the device before
trusting any result:

    mix run -e 'IO.inspect Nx.Vulkan.NativeV.device_name()'

---

## Open items, in priority order

### 1. The 27x is now ~14x. Find the rest.

`Nx.multiply` went 16.4 -> 33-38 GB/s on super-io, against 448 GB/s of VRAM.
The PCIe tax was the dominant term, not the only one. Chaining N ops under one
flush gives a flat marginal cost, so it is per-op work rather than submission.
The shader is unremarkable — 256 threads, one element each, coalesced, one
bounds check. **Next step: is it the shader or the dispatch geometry?** Compare
against a hand-written saturating kernel to establish what the card can actually
do through this backend.

### 2. Jetson's below-cliff `alloc_buffer` is bimodal, not slower

24 and 28 MiB read 0.109-0.122 ms (baseline); 26 and 30 MiB read 0.150-0.342.
Two populations at the same size, below-cliff slope noise in both signs. Not
blocking at 0.15 ms absolute, but "bimodal at one size" is a much better lead
than the flat 1.6x reading it replaces.

### 3. super-io's poison flip rate is still unmeasured

I got 2 samples before timeouts, both 20/20. **And one showed 19/40
effectiveness with 20/20 padding, which breaks my own claim that the two move in
lockstep.** The cross-box poison table is already retired — 248 showed its rate
moves with the *commit*, so it tracks allocator behaviour, not hardware — but if
anyone wants it closed properly it needs ~8 super-io runs to establish a rate,
against the Jetson's 8/8 at 20/20 and the Keplers' stable 1-3/20.

### 4. Race 5 (MCMC) has never run

Nothing in this repo calls `leapfrog_chain_synth_f64`. The shipped chain shaders
are consumed by eXMC downstream, and the templated path in `ShaderTemplate`
emits f32 while the active NIF wants f64. **The call convention has to be
established on one box before four boxes run it.**

The design is worth keeping: the chain shader returns the whole trajectory —
`3*K*d*8` bytes down against `2*d*8` up — so MCMC is transfer-shaped in one
direction, and the effect should grow with fusion depth. Now that the PCIe tax
is gone, this is a different measurement than it would have been.

### 5. The Race 1c clock trace, staged and unrun

`scripts/staged/jetson_run_trace.sh`. Decides whether Race 1c is structurally
unable to measure submission on an integrated part: clock rises with F means
DVFS-confounded and needs redesign; clock flat with the marginals still bending
means something else, which constrains the Kepler disagreement too.

---

## What is settled, so nobody re-litigates it

* **The 32 MiB allocation cliff is vulkano's**, not any memory architecture's —
  reproduced 6/6 across every box and commit, discrete PCIe and unified LPDDR4
  alike.
* **The unified-vs-discrete question is unanswered and four designs failed.**
  Race 1 measured a submission-to-throughput ratio while claiming arithmetic
  intensity; `a` was a mixture; `s` was an ill-conditioned intercept; PRICE
  tracks GPU speed because its two terms are different physics. The fleet's
  GPUs differ 21x in throughput, which swamps the effect.
* **The control pair is not a pair.** mac-247 and mac-248 differ 1.39x on
  submission cost, and it is not the estimator: 248 does 1.25x more GPU work per
  dispatch and has 1.39x LOWER submission cost, so the two move in opposite
  directions. Different SKUs, different hosts. "Same architecture" was never
  "same hardware".
* **DVFS is on all three architectures**, and pstate cannot certify a cold
  measurement in either direction — the resting state with no client is P0. What
  it CAN detect is a foreign client: **6 MiB resident + P8 sustained means
  someone else has the card**. Check `pstate,memory.used` before any timing.
* **The poison-control rate is not a cross-box observable.** It moves with the
  commit.

---

## Harness invariants worth not breaking

Learned expensively, all of them:

* A control must re-measure the **kind** of work it certifies. A matmul cannot
  vouch for an allocation; that cost three near-misses.
* Warm the clock before timing, and warm **both sides** of the 32 MiB cliff.
* GC inside any loop that allocates — a retained-allocation leak has now
  appeared three times, once in a throwaway diagnostic written to investigate
  the previous two.
* Thresholds must match the measured noise floor of their own quantity: compute
  10%, allocation 40%, estimator divergence 30%.
* Prefer slopes to intercepts, and robust estimators to hand-rolled rules. `c`
  replicated to 0.44% where `s` went 8.8% on the same data.
* Replicate. Every estimator defect in the last four rounds was visible only in
  a second run.

---

## Publishing (operator decision, not done)

* `~/projects/learn_erl/pymc/www.dataalienist.com` — two posts committed and
  **not deployed**: "An Absence Mistaken for a Discovery" (the correction) and
  the correction banner on "The Copy That Wasn't There".
* `mix hex.retire nx_vulkan 0.2.0`, `upstream/main` publishing, consumer pin
  bump — all still outstanding from before this sequence.
