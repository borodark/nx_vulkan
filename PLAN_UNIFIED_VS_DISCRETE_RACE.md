# Plan — Jetson (unified memory) vs super-io (discrete): what to race, and why

**Written:** 2026-08-28, against `main` @ `85cc566`.
**Status:** plan. Nothing run yet.

---

## The question this is NOT asking

**Which box is faster.** An RTX 3060 Ti beats a Tegra X1 at 5W on any
compute-bound workload by a margin nobody needs a benchmark to predict, and a
table confirming it would be a waste of two days.

The question worth the expense is narrower and has an actionable answer:

> **Does unified memory change the SHAPE of the cost curve, and if so, where is
> the crossover?**

If it does, the backend should behave differently on a unified device — batching
policy, when to keep intermediates resident, whether a host round trip is ever
the cheaper option. If it does not, that is worth knowing once and never
re-testing.

`bench_results/` already holds the absolute-throughput story
(`AMPERE_SUPER_IO_RESULTS.md`, `MODEL_SCALING.md`). This is the other axis.

---

## What we already know, so we do not re-measure it

Established 2026-08-28 and not in dispute:

| fact | source |
|---|---|
| Every Tegra Vulkan memory type is `DEVICE_LOCAL`; no host-visible-but-not-device-local type exists | fleet §1.4 |
| There is no staging copy on any box — `HOST_SEQUENTIAL_WRITE` writes into mapped memory | `alloc_buffer` |
| `buf_upload` 16 MiB: **6.0 ms** Jetson vs **3.64 ms** Ampere — Jetson memory IS slower | fleet A/B |
| Zero-fill 16 MiB: **3.71 ms** Jetson vs **5.08 ms** Ampere — but a host WRITE is faster there | fleet A/B |
| Allocator cliff at exactly 32 MiB on both | two independent sweeps |
| Submission floor ~170 µs, 75% in `vkQueueWaitIdle`; batching amortises it | DTrace, `bench_results/BATCHED_DISPATCH.md` |

**The one-line theory to test:** the Jetson's advantage is the absence of a bus,
not fast memory. So it should close the gap exactly where PCIe was being paid,
and nowhere else.

---

## The confound that will ruin this if ignored

**The Jetson is a correctness box, and I have said so repeatedly. Racing it
requires separating which caveats bite which measurement.**

| caveat | affects HOST work | affects GPU work |
|---|---|---|
| OTP built `--disable-jit` | **yes, severely** — all BEAM/Elixir execution | no |
| Rust NIF built with relaxed LTO | marginally — NIF entry/exit | no |
| `nvpmodel` 5W, 2 of 4 cores online | **yes** — dispatch recording, tensor construction | indirectly (submission latency) |
| 3.9 GB shared DRAM | sizing limit | sizing limit |
| Thermal throttling under sustained load | yes | **yes** |

**Consequence: any benchmark whose timer encloses host-side tensor construction
measures the JIT-less OTP, not the GPU.** The earlier
`Nx.BinaryBackend.iota` incident — 3 GB of host allocation inside a test setup —
is exactly this failure mode, and it killed the whole VM there.

**Rules for every measurement below:**

1. Build inputs **on the device** (`backend: VulkanoBackend`), outside the timer.
2. `NativeV.flush()` inside the timer; `:erlang.garbage_collect()` outside it.
3. Report **medians of ≥ 9**, plus min and max — a single mean hides throttling.
4. `uptime` and `free -h` **before and during**. This project has withdrawn one
   contended timing table already; do not add a second.
5. Run a **thermal control**: the same measurement first and last in the session.
   If they differ by more than ~10%, the Jetson throttled and the run is void.

---

## The four races

### Race 1 — Arithmetic intensity sweep (the primary experiment)

**Hypothesis:** the Jetson's relative disadvantage shrinks as bytes-moved per
FLOP rises, and there is a crossover ratio where it stops mattering.

Hold total FLOPs roughly constant and vary how much data moves. A matmul of
`{n,k} × {k,n}` does `2·n²·k` FLOPs and moves `~(2nk + n²)` elements, so sweeping
`k` at fixed `n` walks arithmetic intensity across an order of magnitude without
changing the kernel.

    n = 512 fixed;  k ∈ {4, 16, 64, 256, 1024}
    report GFLOP/s and effective GB/s per box, and the RATIO between boxes

**What makes this informative:** the ratio, not the absolute. If the Ampere/Tegra
ratio is flat across `k`, unified memory changes nothing and we stop. If it
narrows as `k` falls (transfer-dominated), the theory holds and Race 3 becomes
worth running.

**Cost:** ~20 min per box. Cheap. **Run this first and stop if the ratio is flat.**

### Race 2 — Round trip vs resident

**Hypothesis:** a host round trip costs proportionally less on unified memory, so
the threshold at which a host fallback beats a GPU dispatch is *lower* on the
Jetson.

Three variants of the same computation over sizes 4 KiB → 16 MiB:

    (a) resident:    upload once, N ops on device, download once
    (b) round-trip:  upload → 1 op → download, N times
    (c) host-only:   BinaryBackend throughout

Report `(b)/(a)` per box. **That ratio is the price of a round trip in units of
compute**, and it is the number the fallback policy should be keyed on.

**Why it matters:** the backend's host-fallback decisions are currently uniform
across devices. If (b)/(a) differs materially, they should not be.

**Cost:** ~30 min per box.

### Race 3 — Batching policy under unified memory

**Hypothesis:** `NXV_BATCH_MAX` (default 64) was tuned on discrete hardware. The
~170 µs submission floor is a *driver and queue* cost, not a bus cost, so it may
not scale with the box the way batching assumes.

Sweep `NXV_BATCH_MAX ∈ {0, 1, 4, 16, 64, 256}` over a fixed chain of ~200 small
elementwise dispatches. Report time per dispatch.

**The finding to look for:** whether the knee is at the same value on both. If
the Jetson's optimum is materially different, the default should be
device-class-dependent rather than a constant — and `Nx.Vulkan.Device` already
has the `weak?`/`class` machinery to express that.

**Only run this if Race 1 shows a non-flat ratio.**

**Cost:** ~40 min per box.

### Race 4 — Allocation cliff, generalised

**Hypothesis:** the 32 MiB cliff is a vulkano suballocator threshold, identical
on both, and independent of unified memory.

Fine sweep 24–40 MiB in 2 MiB steps on both boxes, `buf_alloc` and
`buf_alloc_zeroed`. Two boxes already found the edge at exactly 32 MiB; this
confirms it is the allocator rather than a coincidence, and establishes whether
the *post-cliff slope* (~0.83 ms/MiB on Tegra) differs.

**Actionable if:** the slope differs enough that large-output ops should be
chunked below the cliff on one box and not the other.

**Cost:** ~15 min per box. Cheap; run alongside Race 1.

---

## Sizing for 3.9 GB

The Jetson has ~3.3 GB available in practice. **Cap any single tensor at
64 MiB** and any working set at ~512 MiB. Race 1's `n=512, k=1024` f32 matmul is
~2 MiB of input and 1 MiB of output — comfortable. Race 2's 16 MiB ceiling is
deliberate: it sits below the 32 MiB allocator cliff so Race 2 is not
accidentally measuring Race 4.

---

## What would make each race a null result

Stated in advance, so a null is reportable rather than a disappointment:

- **Race 1:** Ampere/Tegra ratio flat within ±15% across `k`. → unified memory
  does not change the shape; record it and close the question.
- **Race 2:** `(b)/(a)` within ±20% between boxes. → round-trip cost scales with
  the machine; no per-device fallback policy needed.
- **Race 3:** knee at the same `NXV_BATCH_MAX` on both. → the default is fine;
  leave it a constant.
- **Race 4:** same cliff size and slope within ±25%. → it is the suballocator,
  document and move on.

**Three of four nulls would still be worth the two days**, because "unified
memory does not change our policy" is a decision this project currently cannot
make either way.

---

## Order of operations

1. **Race 1 + Race 4** on both boxes (cheap, ~35 min each). Decide from Race 1's
   ratio whether to continue.
2. If non-flat → **Race 2**, which is where a policy change would come from.
3. If Race 2 shows a gap → **Race 3**, which is how the policy would be expressed.

Write results to `bench_results/UNIFIED_VS_DISCRETE.md` with the load average and
the thermal control alongside every table, per the standing rule.

---

## A note on who runs this

The Jetson legs should run as a fleet agent, not interactively — the runs are
long and the box has a history of being contended by unrelated jobs. The agent
brief must carry the confound table above, because the single most likely way to
waste this is to time host-side tensor construction on a JIT-less OTP and report
the result as a GPU number.
