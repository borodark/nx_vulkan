# Plan — unified memory vs dedicated GPU RAM: what to race, and why

**Revision 2**, 2026-08-29, against `main` @ `eb775ea`.
Revision 1 was written 2026-08-28 @ `85cc566`. **Status:** plan. Nothing run yet.

Two things changed since revision 1, and one of them changes the experimental
design rather than its parameters.

---

## The question this is NOT asking

**Which box is faster.** An RTX 3060 Ti beats a Tegra X1 at 5W on any
compute-bound workload by a margin nobody needs a benchmark to predict.

The question worth the expense:

> **Does unified memory change the SHAPE of the cost curve, and if so, where is
> the crossover — and does an MCMC sampler ever sit on the far side of it?**

If it does, the backend should behave differently on a unified device: batching
policy, fusion depth, whether a host round trip is ever the cheaper option. If it
does not, that is worth establishing once and never re-testing.

---

## What changed since revision 1

### 1. Race 4's hypothesis is largely answered — narrow it

Revision 1 proposed Race 4 to test whether the 32 MiB allocator cliff is a
vulkano suballocator threshold rather than a Tegra artifact. **Four boxes have
since answered that**, via `scripts/poison_control.exs`:

    box                              6 x 32 MiB    sub-cliff schemes
    mac-247   GT 650M   Kepler          0/40            40/40
    mac-248   GT 750M   Kepler          0/40            40/40
    super-io  RTX 3060  Ampere          0/40            40/40
    jetson    Tegra X1  unified         0/40            40/40

At and above 32 MiB, allocations come back freshly zeroed on every box — the
signature of a dedicated `vkAllocateMemory` returned to the driver on free. Same
size, no gradient, across two operating systems, three GPU generations, discrete
PCIe and unified LPDDR4 alike.

That is a *behavioural* confirmation, not a timing one, so it does not fully
subsume Race 4 — but it collapses it from "is the cliff real and shared?" to
"does the post-cliff **slope** differ?". Race 4 shrinks accordingly.

### 2. The Keplers are not optional — they are what makes this interpretable

Revision 1 raced two boxes. **Every difference it could find is confounded**,
because unified-vs-discrete is perfectly correlated with weak-vs-strong and
old-vs-new:

| box | memory | GPU strength | era |
|---|---|---|---|
| Jetson Tegra X1 | **unified** | weak | 2015 |
| super-io RTX 3060 Ti | discrete | strong | 2021 |

Any curve that differs between those two has three candidate explanations and no
way to choose. Adding the Keplers breaks the correlation, because a GT 650M is
discrete, weak and old — it varies strength and era while holding memory
architecture at *discrete*:

| box | memory | strength | era | role |
|---|---|---|---|---|
| Jetson Tegra X1 | **unified** | weak | 2015 | treatment |
| mac-247 GT 650M | discrete | weak | 2012 | **control for weak+old** |
| mac-248 GT 750M | discrete | weak | 2013 | control replicate |
| super-io RTX 3060 Ti | discrete | strong | 2021 | control for strong |

**The inference rule this buys:**

- Jetson differs from **all three** discrete boxes in the same direction
  → memory architecture. This is the only pattern that supports the claim.
- Jetson looks like the **Keplers** and both differ from Ampere
  → weak-GPU effect. Nothing to do with unified memory.
- Jetson differs from everything, Keplers and Ampere agree
  → Tegra-specific or 5W-specific; suspect the confound table below before
  believing it.

Without the Kepler leg, the first and second patterns are indistinguishable, and
this project has already published one hardware finding that was really a
contention artifact. **mac-248 is newly viable for this**: its clickhouse,
postgres and ten hello_beam jails were stopped 2026-08-29, and it idles at ~0.2.

Two Keplers also give a same-architecture replicate, which is the cheapest
available estimate of between-box noise. mac-248 is known to swing ±11–13% on
repeated samples; any effect smaller than the 247-vs-248 spread is not an effect.

---

## What we already know, so we do not re-measure it

| fact | source |
|---|---|
| Every Tegra Vulkan memory type is `DEVICE_LOCAL`; no host-visible-but-not-device-local type exists | fleet §1.4 |
| There is no staging copy on any box — `HOST_SEQUENTIAL_WRITE` writes into mapped memory | `alloc_buffer` audit |
| `buf_upload` 16 MiB: **6.0 ms** Jetson vs **3.64 ms** Ampere — Jetson memory IS slower | fleet A/B |
| Zero-fill 16 MiB: **3.71 ms** Jetson vs **5.08 ms** Ampere — a host WRITE is faster there | fleet A/B |
| Allocator cliff at exactly 32 MiB on all four boxes | poison_control, 4 boxes |
| Submission floor ~170 µs, 75% in `vkQueueWaitIdle` | DTrace, `bench_results/BATCHED_DISPATCH.md` |
| Removing the zero-fill won **least** on the Jetson (37× vs Ampere's 635×) | fleet §1.4a |

**The one-line theory:** the Jetson's advantage is the absence of a bus, not fast
memory. It should close the gap exactly where PCIe was being paid, and nowhere
else. The zero-fill result is the first evidence *for* this theory — unified
memory had already made the old path cheap, so there was less to reclaim.

---

## The confound that will ruin this if ignored

**The Jetson is a correctness box and has been called one all along. Racing it
at all requires separating which caveats bite which measurement.**

| caveat | affects HOST work | affects GPU work |
|---|---|---|
| OTP built `--disable-jit` | **yes, severely** — all BEAM/Elixir execution | no |
| Rust NIF built with relaxed LTO | marginally — NIF entry/exit | no |
| `nvpmodel` 5W, 2 of 4 cores online | **yes** — dispatch recording, tensor construction | indirectly (submission latency) |
| 3.9 GB shared DRAM | sizing limit | sizing limit |
| Thermal throttling under sustained load | yes | **yes** |

**Consequence: any benchmark whose timer encloses host-side tensor construction
measures the JIT-less OTP, not the GPU.**

**Rules for every measurement below:**

1. Build inputs **on the device** (`backend: VulkanoBackend`), outside the timer.
2. `NativeV.flush()` inside the timer; `:erlang.garbage_collect()` outside it.
3. Report **medians of ≥ 9**, plus min and max — a single mean hides throttling.
4. `uptime` before AND during; on mac-248 also `doas jls`, because `ps` alone
   gives a false all-clear on a jail host.
5. **Thermal control:** repeat the first measurement last. If they differ by more
   than ~10%, the Jetson throttled and the run is void.
6. **Report every number as a ratio to the same box's own baseline**, never as a
   cross-box absolute. This is what makes the JIT-less host cancel out.

Rule 6 is the one that makes a Jetson timing defensible at all. We are not
comparing milliseconds across boxes; we are comparing the *shape* of each box's
own curve.

---

## The races

### Race 1 — Arithmetic intensity sweep (primary)

**Hypothesis:** the Jetson's relative disadvantage shrinks as bytes-moved per
FLOP rises, and there is a crossover ratio where it stops mattering.

Matmul `{n,k} × {k,n}` does `2·n²·k` FLOPs and moves `~(2nk + n²)` elements, so
sweeping `k` at fixed `n` walks arithmetic intensity across an order of magnitude
without changing the kernel.

    n = 512 fixed;  k ∈ {4, 16, 64, 256, 1024}
    per box: GFLOP/s, effective GB/s, and each box's curve NORMALISED to its own k=1024 point

**Read it as:** if all four normalised curves have the same shape, memory
architecture changes nothing. If the Jetson's curve flattens where the three
discrete curves keep falling, that is the effect, with a Kepler control.

**Cost:** ~20 min per box. **Run first; stop if the shapes match.**

### Race 2 — Round trip vs resident

**Hypothesis:** a host round trip costs proportionally less on unified memory, so
the threshold at which a host fallback beats a GPU dispatch is *lower* there.

Three variants over sizes 4 KiB → 16 MiB:

    (a) resident:    upload once, N ops on device, download once
    (b) round-trip:  upload -> 1 op -> download, N times
    (c) host-only:   BinaryBackend throughout

Report `(b)/(a)` per box — **the price of a round trip in units of compute**, and
the number a per-device fallback policy would be keyed on.

**Cost:** ~30 min per box.

### Race 3 — Batching policy under unified memory

**Hypothesis:** `NXV_BATCH_MAX` (default 64) was tuned on discrete hardware. The
~170 µs submission floor is a driver/queue cost, not a bus cost.

Sweep `NXV_BATCH_MAX ∈ {0, 1, 4, 16, 64, 256}` over ~200 small elementwise
dispatches. Report time per dispatch; find the knee per box.

**Note the trap:** `Nx.Vulkan.Device.class/0` is `:weak | :strong`, and it
classifies **GT-line Keplers and the Tegra alike as `:weak`**. So if the knee
splits Jetson from Keplers, the existing class machinery *cannot express the
policy* — it would need a memory-architecture predicate, not a strength one.
That is a finding about the abstraction, not just the constant.

**Only run if Race 1 shows differing shapes. Cost:** ~40 min per box.

### Race 4 — Allocation cliff slope (narrowed)

The cliff's existence and location are settled on four boxes. What remains: does
the **post-cliff slope** differ? Fine sweep 24–40 MiB in 2 MiB steps,
`buf_alloc` and `buf_alloc_zeroed`.

**Actionable if:** the slope differs enough that large outputs should be chunked
below the cliff on one box and not another.

**Cost:** ~15 min per box. Cheap; run alongside Race 1.

---

## Race 5 — Do MCMC samplers benefit? (the product question)

Races 1–4 characterise the machine. This one asks whether any of it reaches
eXMC's actual workload, and it is the only race whose null result is *also* a
product decision.

### Why MCMC is not a bandwidth workload

A sampler's state is tiny. `d` parameters in f64 is `8d` bytes — even `d = 10,000`
is 80 KB, orders of magnitude below any bandwidth regime. Naively, memory
architecture should be irrelevant and the ~170 µs submission floor should
dominate everything.

**The chain shader breaks that reasoning, and this is the part worth racing.**
`leapfrog_chain_synth_f64/6` does not return an endpoint. It returns the whole
trajectory:

    upload   per dispatch:  q, p            = 2·d·8 bytes          (+ extras, small)
    download per dispatch:  q, p, grad chains = 3·K·d·8 bytes
                            logp chain        =   K·8 bytes

The download is `K`-proportional and the upload is not. At `d = 100, K = 100`
that is **240 KB down against 1.6 KB up — a 150:1 asymmetry, paid every
dispatch.** MCMC turns out to be transfer-shaped after all, in one direction
only, and precisely because of the fusion this project already built.

### The interaction that makes it interesting

Over a run of `N` total leapfrog steps at fusion depth `K`:

| cost | scales as | present on |
|---|---|---|
| submissions | `N/K` × ~170 µs | both — driver/queue, not bus |
| trajectory bytes | `3·N·d·8` — **independent of K** | both, but *cheaper on unified* |

Fusion buys down submissions and does nothing about bytes. So both boxes become
transfer-bound at large `K`, and that is exactly where unified memory should
show up.

**Predictions, in falsifiable form:**

1. **The Ampere/Tegra ratio narrows as `K` rises.** At small `K` both boxes are
   submission-bound and unified memory is invisible; at large `K` the discrete
   box pays PCIe for the trajectory and the Jetson does not.
2. **Optimal fusion depth `K*` is higher on unified memory** — the discrete box
   stops benefiting from deeper fusion sooner, because it hits the transfer
   floor first.
3. **The Keplers should track Ampere, not the Jetson**, on both. If they track
   the Jetson instead, prediction 1 was measuring weak-GPU dispatch overhead and
   the memory story is dead.

**Design:** sweep `K ∈ {1, 4, 16, 64, 256}` × `d ∈ {16, 128, 1024}` against a
fixed total step budget `N = 4096`, one of the shipped f64 chain shaders
(`leapfrog_chain_normal_f64` — the hand-written one, not a synthesised spec, to
keep glslang out of the timer). Report per box: total wall time, and time
normalised to that box's own `K = 1` point.

**Cost:** ~45 min per box. Run only after Race 1, and only on the two required
boxes plus at least one Kepler.

### The decision this feeds

If predictions 1 and 2 hold, the actionable output is **not** a tuning constant.
It is that an **endpoint-only chain variant** — returning `q, p, logp` at the
final step instead of the full trajectory — would cut the download by a factor of
`K` and is therefore worth building **for discrete hardware and barely worth it
for unified**. That is a concrete, differently-valued piece of work on different
boxes, which is the strongest form this question can produce.

The caveat that makes it honest: **NUTS needs the trajectory** for the U-turn
criterion and multinomial sampling, so endpoint-only is a plain-HMC option, not a
universal one. If eXMC is NUTS-only in practice, prediction 2 still stands but
the actionable half evaporates — and that is worth knowing before building
anything.

---

## Sizing for 3.9 GB

The Jetson has ~3.3 GB available in practice. **Cap any single tensor at 64 MiB**
and any working set at ~512 MiB. Race 1's `n=512, k=1024` f32 matmul is ~2 MiB in
and 1 MiB out. Race 2's 16 MiB ceiling is deliberate: it sits below the 32 MiB
cliff so Race 2 is not accidentally measuring Race 4. Race 5's largest case
(`d=1024, K=256`) downloads `3·256·1024·8` = 6 MiB per dispatch — comfortable,
and also below the cliff.

---

## What would make each race a null result

Stated in advance, so a null is reportable rather than a disappointment:

- **Race 1:** normalised curve shapes agree within ±15% across all four boxes.
  → unified memory does not change the shape; close the question.
- **Race 2:** `(b)/(a)` within ±20% between boxes. → no per-device fallback policy.
- **Race 3:** knee at the same `NXV_BATCH_MAX`. → the default is fine as a constant.
- **Race 4:** post-cliff slopes within ±25%. → suballocator, document and move on.
- **Race 5:** Ampere/Tegra ratio flat in `K`. → **the strongest null available.**
  It would mean the trajectory download never dominates at realistic sizes, the
  submission floor rules everything, and no endpoint-only variant is worth
  building on any box. That closes a design question this project cannot
  currently answer either way.

**Four of five nulls would still be worth the expense**, because "unified memory
does not change our policy" is a decision that is currently unmade rather than
made.

---

## Order of operations

1. **Race 1 + Race 4** on all four boxes (~35 min each). Decide from Race 1's
   normalised shapes whether to continue.
2. **Race 5** regardless of Race 1's outcome — it is the product question, and a
   flat result there is independently valuable.
3. If Race 1 non-flat → **Race 2**, where a policy change would come from.
4. If Race 2 shows a gap → **Race 3**, how the policy would be expressed.

Write results to `bench_results/UNIFIED_VS_DISCRETE.md`, with load average and
the thermal control alongside **every** table, per the standing rule. Follow
`examples/w5_kernels_race.exs`, which already records `box_was_busy` in its JSON.

---

## A note on who runs this

The Jetson and Kepler legs should run as fleet agents, not interactively — the
runs are long and the boxes have a history of being contended by unrelated jobs.

**The agent brief must carry the confound table and rule 6.** The single most
likely way to waste this is to time host-side tensor construction on a JIT-less
OTP and report the result as a memory-architecture finding. The second most
likely is to run mac-248 without checking `doas jls` first.
