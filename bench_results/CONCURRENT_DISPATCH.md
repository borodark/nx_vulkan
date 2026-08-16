# Batched dispatch under concurrency — a negative result

**Date:** 2026-08-15 · **Commit:** `6ab64ac` · **Harness:**
[`examples/concurrent_dispatch_bench.exs`](../examples/concurrent_dispatch_bench.exs),
driven by [`scripts/concurrency_race.sh`](../scripts/concurrency_race.sh)
**Hosts:** mac-247 (GT 650M, Kepler, FreeBSD) and mac-248 (GT 750M, Kepler,
FreeBSD), both idle. super-io was excluded: three agents were using its GPU, and
it inflated an established measurement by 43% while they did.

## The question

[`BATCHED_DISPATCH.md`](BATCHED_DISPATCH.md) measures 1.45–1.71× — in **one BEAM
process**, on three hosts. The queue producing that is a single
`OnceLock<Mutex<Vec<RecordFn>>>` static in the NIF, i.e. one queue per BEAM VM
shared by every process, and `submit_and_wait` ends in a device-wide
`queue.wait_idle()`. Neither costs anything with one dispatcher. The deployments
this backend targets have many: exmc runs a GenServer per instrument and
[`LIMITATIONS.md`](../LIMITATIONS.md) §7 puts that queue 67 jobs deep.

Three effects were predicted, none measured: the batch is a shared bucket (no
graph gets a batch of its own once N rises); a readback by any process flushes
everyone's work and waits on the whole device; descriptor-set pressure scales
with N.

## The answer: none of them shows up

`NXV_BATCH_MAX=64` throughput relative to the `=0` submit-per-dispatch control.
**Five interleaved replicates per cell**, 30–40 reps per worker.

| N | GT 650M | GT 750M |
|---:|---:|---:|
| 1 | 1.60× | 1.36× |
| 4 | 1.17× | 1.01× |
| 8 | 1.22× | 1.14× |
| 16 | 1.20× | 1.06× |
| 32 | **1.33×** | 1.15× |

Batching wins at every N on both cards. The shared bucket is not costing
measurable throughput up to 32 concurrent dispatchers.

**So the GPU-node follow-ups are not filed as work.** Routing batches through
`Nx.Vulkan.Node`'s `with_node/2` at the graph boundary, owner-keyed pending
queues, and per-submission fences in place of `wait_idle` were all sketched
against a contention cost that this race looked for and did not find. T1's
decision to ship batching on by default, with no `Device.class/0` gate, survives
into the concurrent regime.

The tail says the same. p95/p50 on the GT 750M is 2.19 control against 2.19
batched at N=8 — the wide tail there is a property of *that host*, present
identically in both arms. It is not convoy, and attributing it to batching was
an error of exactly the kind this document exists to prevent.

## Harness validation

At N=1 the harness reproduces the published single-process figures from an
independent implementation (p50 ms, control → batched):

| | this harness | `BATCHED_DISPATCH.md` |
|---|---|---|
| GT 650M | 13.8 → 8.7 | 14.6 → 8.8 |
| GT 750M | 10.5 → 6.3 | 13.3 → 9.1 |

The 650M lands almost exactly on the published numbers. That agreement is what
makes the concurrency columns above worth reading.

## Two things this established that it was not looking for

### 1. One process under-feeds these GPUs

Throughput roughly doubles from N=1 to N=8 on both cards before saturating —
GT 650M 108 → 190 steps/s, GT 750M 95 → 118. A single dispatcher does not keep
either card busy. **Concurrency is worth having; the shared queue is simply not
the obstacle to it.** Any future work on multi-tenant throughput should start
from that, rather than from the assumption that the queue needs unsharing.

### 2. The GT 750M is a poor host for timing work; the GT 650M is a good one

Standard deviation across five replicates of the same cell, same session:

| host | control | batched |
|---|---|---|
| GT 650M | ±0.6–7.0 (**2–4%**) | ±2.2–19.8 |
| GT 750M | ±9.4–14.3 (**11–13%**) | ±10.0–14.9 |

This is recorded nowhere else in the repo and it invalidates single-run cells on
mac-248: **a 15% effect measured there once is a coin flip.**

It is not a hypothetical. The first pass of this race — one run per cell —
produced a clean, monotone, entirely convincing hardware crossover: batching
apparently regressing to 0.84× / 0.80× / 0.88× on the GT 750M at N ≥ 4 while the
GT 650M never regressed, with a widening p95/p50 that supplied a mechanism. A
two-replicate confirmation reproduced two of the three cells and looked like
corroboration. Five replicates erased the whole thing.

The lesson is not "be careful". It is specific: **race on the GT 650M, or
replicate.** Direction agreeing twice on a host with 13% noise is worth nothing,
and the failure mode is not a wrong number — it is a coherent false mechanism
that survives the first attempt to check it.

## Method notes

- Workers each build their **own** parameter set. Sharing one tensor across N
  processes would be a kinder workload than the deployment it stands in for, and
  would let the driver reuse residency the real thing cannot.
- All workers warm up, then block on a barrier, then start together. Without the
  barrier, early starters finish while late ones are still compiling pipelines,
  and staggered work gets reported as concurrent work — understating exactly the
  contention being measured.
- The timed path reads back **only the scalar loss**. `buf_download` calls
  `flush_pending`, which submits every queued dispatch including the gradients,
  and `submit_and_wait` blocks on the whole command buffer, so the scalar
  accounts for all the work. The first version also transferred a `{784,128}`
  gradient and summed it on the host each step — ~400 KB over PCIe plus a
  100k-element `BinaryBackend` reduction — which added a large constant to every
  arm equally and flattened cap 0/4/64 to within noise of each other. It hid the
  effect being measured and disagreed with this repo's own published 1.45×. If a
  future change makes the timed path transfer anything but a scalar, that is a
  bug.
- `NXV_BATCH_MAX` is read into a `OnceLock` on first dispatch and fixed for the
  OS process lifetime, so the cap sweep is a shell loop over separate `mix run`
  invocations. The process-count sweep does work inside one run.
- Every guard from [`mnist_mlp_step_bench.exs`](../examples/mnist_mlp_step_bench.exs)
  is carried over: global default backend, per-tensor residency assertions, a
  NaN check on the loss, and a fallback census printed before timing.

## Not measured

- **N > 32**, and hosts other than the two Keplers. super-io was contended
  throughout and contributes nothing here.
- **Heterogeneous workers.** Every worker runs the same graph. exmc's real
  shape is instruments of differing model sizes, where a large graph's flush
  could plausibly convoy small ones in a way uniform workers cannot show. That
  is the remaining way the shared bucket could still cost something, and this
  race does not rule it out.
- **Descriptor-pool pressure at high N**, which was the going-in Kepler concern.
  Nothing failed, but nothing instrumented it either.
