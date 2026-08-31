# The elementwise path writes its output across PCIe

**Measured 2026-08-30 on super-io (RTX 3060 Ti, discrete).** Found while chasing
why `Nx.multiply` runs 27x below memory bandwidth.

## The number

    Nx.multiply(x, 1.0), x = 16 MiB f32, resident, no fallbacks

     4 MiB   0.633 ms/op   12.3 GB/s
    16 MiB   1.907 ms/op   16.4 GB/s
    64 MiB  40.382 ms/op    3.1 GB/s   <- output crosses the 32 MiB alloc cliff

16.4 GB/s against a card with **448 GB/s** of VRAM.

It is not dispatch overhead. Chaining N ops under a single flush gives a flat
marginal cost — 1.84, 1.79, 1.81, 1.96 ms per op at N = 2, 4, 8, 16 — so the
cost is per-op work, not per-submission.

It is not a fallback. `Fallback.count/1` reports `%{}` and the result is
`%VulkanoBackend{}` at `{:f, 32}`.

The shader is not obviously wrong either: `local_size_x = 256`, one element per
thread, coalesced indexing, a single bounds check.

## The cause, measured directly

`nvidia-smi dmon -s t` during a pure compute loop — no uploads, no downloads,
one resident input, GC'd every iteration:

    # gpu  rxpci  txpci      (MB/s)
        0      4  10802
        0     21  10802
        0      4   9903
        ...sustained for the whole 20 s loop

**~10.8 GB/s of sustained GPU-to-host PCIe traffic during a computation that
transfers nothing.** The write bandwidth implied by timing is 8.7 GB/s (16 MiB
per op at 1.8 ms), which matches.

`rxpci` stays at 4-21 MB/s — and it stays there with a **two-tensor**
`Nx.multiply(x, y)` as well, where 32 MiB of input is read per op. So the inputs
are being read from VRAM and only the freshly allocated **output** lives in host
memory.

## Where it comes from

`alloc_buffer` in `native/nx_vulkan_vulkano/src/lib.rs`:

    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
        | MemoryTypeFilter::HOST_RANDOM_ACCESS,

`PREFER_DEVICE` is a preference. `HOST_RANDOM_ACCESS` is a **requirement** —
the memory must be host-visible. On a discrete NVIDIA card the host-visible,
host-cached types are not device-local, so the requirement wins and the output
buffer is placed in system RAM. Every store the shader executes then crosses
PCIe.

## This is the cost of "the copy that wasn't there"

The `alloc_buffer` audit established that this backend performs no staging copy
— `HOST_SEQUENTIAL_WRITE` writes straight into mapped memory — and that was
written up as a happy finding.

It is true, and this is why it is true: **buffers are host-visible precisely so
that no staging copy is needed.** Upload writes directly into the buffer the
shader will read; download reads directly out of the buffer the shader wrote.
No copy, by construction.

The bill arrives on the compute side. On a discrete card that design trades one
transfer for *every store the shader makes*, which is a 52x write-bandwidth tax.
On the Jetson it costs nothing at all, because there is only one pool of memory
— which is also why the Jetson looked good in every unified-vs-discrete
comparison. **That advantage is an artifact of this defect, not a property of
unified memory.**

## The fix

The standard Vulkan pattern the backend currently skips:

* allocate compute buffers `DEVICE_LOCAL` only, with no host-visibility
  requirement;
* keep a host-visible staging buffer for upload and download;
* `vkCmdCopyBuffer` between them at the boundary.

That reintroduces exactly the staging copy the audit noted was absent — one
copy per transfer, in exchange for full VRAM bandwidth on every store. Since
transfers are already the rare operation relative to compute in any resident
workload, the trade is heavily favourable on discrete hardware and neutral on
unified.

It touches `buf_alloc`, `buf_upload_into`, `buf_download` and the output
allocation in every dispatch path, so it is an architectural change rather than
a tweak. It is also independently verifiable: `nvidia-smi dmon -s t` should show
`txpci` fall to near zero during a compute loop.

## FIXED — measured after the change

`alloc_buffer` now requests `PREFER_DEVICE` alone. Host access goes through
`staging_read` / `staging_write`: a host-visible staging buffer plus a
`vkCmdCopyBuffer`. `alloc_buffer_zeroed` allocates device-local and zeroes with
`fill_buffer` on the device instead of writing n_bytes of zeros from the host.

Clock-pinned, median of 9, on super-io:

    MiB     before        after     speedup
      4   12.3 GB/s   37.9 GB/s       3.1x
     16   16.4 GB/s   33.0 GB/s       2.0x
     64    3.1 GB/s   35.1 GB/s      11.3x

**The PCIe traffic is gone.** `txpci` during the same compute loop falls from a
sustained 10,800 MB/s to single digits.

**The size-dependent collapse is gone too.** Throughput was 12.3 / 16.4 / 3.1
across 4 / 16 / 64 MiB — falling off a cliff once the output crossed 32 MiB.
It is now flat at 33-38 GB/s across the whole range. That collapse was the same
defect compounded: above the dedicated-allocation threshold the host-visible
placement got worse still.

Correctness is unchanged: 833 doctests / 871 tests / 0 failures, strict 0
failures / 163 excluded, residency 755/833 (90.6%) — identical to before.

**It is still 13x off the card's 448 GB/s.** The PCIe tax was the dominant term,
not the only one. What remains is worth a separate investigation.

### The speedup is NOT general — mac-247 A/B'd it

mac-247 built `4e271e0` and `1c575cc` in turn in one session on a verified-free
card, two replicates each, identical probe. Its GT 650M on the 470/FreeBSD
stack:

    MiB    before      after    speedup
      4   5.94 GB/s  5.99 GB/s    1.01x
     16   5.80 GB/s  5.63 GB/s    0.97x
     64   1.61 GB/s  4.61 GB/s    2.85x

**Below the cliff it gained nothing.** The whole win on that card is the
above-cliff path: 64 MiB ran at 0.28x the 16 MiB rate before and 0.82x after,
75-93 ms down to 28-30 ms.

So on the 470 driver the `HOST_RANDOM_ACCESS` requirement was evidently
satisfiable from device-local memory for small allocations and not for large
ones, while on super-io's current driver it displaced everything. **The defect
and the fix are both real; the magnitude is driver- and size-dependent, and
"2.0-11.3x" describes super-io, not the fleet.**

Its larger gain was somewhere I did not predict — **allocation**:

    buf_alloc above cliff     23-30 ms  ->  5.6-6.7 ms
    zeroed    above cliff     39-53 ms  ->  6.4-7.4 ms   (~7x)
    zeroed    below cliff      1.21 ms  ->  0.57-0.70 ms (~2x)
    fitted alloc_above slope   0.75-1.91 ->  0.1241
    fitted zeroed_above slope  1.39-1.79 ->  0.0591

The 32 MiB cliff still exists and is now about an order of magnitude cheaper to
cross. Device-local allocation is simply cheaper than host-visible allocation,
and `alloc_buffer_zeroed` no longer writes n_bytes of zeros from the host.

Two more from that box. `nvidia-smi dmon` is **not supported** on a GT 650M under
the 470 driver, so no direct `txpci` confirmation is available there — the timing
is the only evidence. And `s_flush` reads 538.0 us against its established
522-550 band, so submission cost did not move, which is the expected result and
worth having.

### mac-248 confirms the pattern — and measures a cost I had not

It built the pre-fix commit too rather than trusting a remembered number, and
used a marginal method (1 op against 9, difference over 8) so dispatch cost is
removed:

    MiB    pre-fix   post A   post B    change
      4      6.26     6.23     6.27      none
     16      6.45     6.17     6.53      none
     64      2.05     4.86     4.85      2.37x

Same shape as mac-247: **nothing below the cliff, a real win above it**,
reproducible to 0.2% and about 40x outside its noise. Two independent Keplers
now agree that the defect was confined to above-cliff allocations on the 470
stack while it displaced everything on super-io's.

It also corrects something I told it. I had explained its Race 2 `~1.0`
coincidence by saying its compute was running over PCIe like everything else.
**It was not** — making its 16 MiB buffers device-local changed nothing there,
so that explanation was wrong for its box.

### THE STAGING COPY COSTS 40% OF CROSSING BANDWIDTH

This is the part I did not measure before claiming the change was good.

    at 16 MiB      pre-fix   post A   post B
    dev_GB/s          6.47     6.05     5.87
    cross_GB/s        7.92     4.77     4.86
    boundary_ms      3.948    6.557    6.436
    PRICE             0.80     1.27     1.21

My prediction was that `dev_GB/s` would rise while `cross_GB/s` held. Neither
happened. The `~1.0` coincidence broke because **the crossing got worse**, not
because the device got better.

So the honest net for that box: a real win on above-cliff compute, nothing below
it, and a **40% tax on every host-device crossing**. Whether that is positive
depends entirely on the workload — compute-heavy above the cliff wins,
transfer-heavy loses. Race 2's sweep stops at 16 MiB and so never reaches the
cliff where the win lives, which is why Race 2 read as a pure regression while
the elementwise probe showed a 2.37x gain. Both are true.

**This strengthens the case for making the staging path conditional** rather than
unconditional, which was already flagged for the Jetson and now has a second
motivation on discrete hardware.

### Two things 248's control stopped it reporting wrongly

It found uploads failing at 256 MiB (`upload buffer: a non-validation error
occurred`) and was ready to file it as a regression. Pre-fix behaviour is
**byte-identical** — same ceiling, same error. Pre-existing and unrelated.

It also found odd VRAM accounting — ~192 MiB of live device tensors while
`nvidia-smi` reports 70 MiB used — which would be suggestive that not all
compute buffers are device-local there, consistent with both the below-cliff
non-improvement and the still-poisonable padding leg. It offered it as a lead
rather than a finding, on the grounds that the same tool reports `[N/A]` for
clocks on that build and it does not trust the memory accounting. That is the
right call and the lead is worth chasing.

### The batched fill's real payoff was Race 4, not the drift

I batched the fill to fix an allocation drift, and the Jetson showed that
mechanism was wrong — it was DVFS. The change was still correct, for a reason
neither of us gave at the time. mac-248 at `a6dcfa7`, three runs:

    quantity                  before        after        factor
    buf_alloc @32 MiB      ~19.8-20.2 ms   3.97-4.12      ~5x
    buf_alloc_zeroed @32   ~33.2-48.2 ms   4.68-4.82    ~7-10x
    alloc_above slope        0.69-1.10     0.059-0.087   ~10x
    zeroed_above slope       0.80-2.46     0.095         ~10-20x
    zeroed_below @24 MiB   ~1.09-1.11 ms   0.513-0.527    ~2x

**And it repaired a measurement, not just a cost.** That box had flagged
`zeroed_above` repeatedly as unusable — 1.88x to 2.24x spread across quiet
consecutive runs even at 25 reps — and recommended dropping it from any
conclusion. It now reproduces to **1.4%** (0.0952 / 0.0943 / 0.0956).

So the variance was never sampling noise. It was per-allocation submission
overhead, which is exactly what the batching removed. **The recommendation to
drop `zeroed_above` is withdrawn; the quantity is reportable now.**

### RETIRED: the poison-control rate is not a property of the card

mac-248 settles this, and not in the direction the cross-box table suggested.
Four fresh processes gave 3/20, 2/20, 3/20, 3/20 — a narrow low band with no
flipping, where super-io alternates between the extremes 20/20 and 0/20 and the
Jetson pins at 20/20. Three boxes, three distinct behaviours, which looks like a
finding.

It is not, and 248 supplied the disqualifying evidence itself: **its own rate
moved with the commit.** It read 7/20 at `1c575cc` and 2-3/20 at `a6dcfa7`. The
number tracks allocator and submission behaviour, changing when the code
changes, rather than sitting where the hardware puts it.

A quantity that is stable within a build and moves between builds is not
describing the card. The cross-box comparison is withdrawn entirely — not
merely unproven, as I had it after the super-io flip, but measuring the wrong
thing. The UNPROVEN branch remains valuable as an honest self-report; its
numeric rate is not a cross-box observable.

### Earlier retraction, superseded by the above

I reported that small device-local allocations had stopped being poisonable on
super-io and not on the Keplers, and treated the difference as a property of the
boxes. Three consecutive runs on super-io, same commit, same box:

    padding-size probes:   20/20      POISON CONTROL: PASS
    padding-size probes:    0/20      PASS with the padding leg UNPROVEN
    padding-size probes:   20/20      PASS

**It is bimodal run to run, and the whole poisoning behaviour flips together** —
the large-scheme effectiveness reads 40/40 alongside 20/20, and 19/40 alongside
0/20. That is per-process allocator state, not hardware.

So the earlier claim rested on two runs that both happened to land on the same
mode, and the cross-box table (super-io 0/20, mac-247 2/20, mac-248 7/20,
jetson 20/20) is not evidence of anything architectural. It may still be real —
the Keplers and the Jetson have not been sampled enough times to say — but it is
not established, and I asserted it as though it were.

Same error as everything else in this sequence: a two-sample result read as a
property. The UNPROVEN branch is working correctly either way; it fires on
roughly half of super-io's runs and reports honestly when it does.

### The poison control differs across boxes (unconfirmed, see retraction above)

super-io reports `PASS with the padding leg UNPROVEN` (0/20 dirty at 4 B / 8 B).
**mac-247 reports a plain PASS with 2/20 and mac-248 with 7/20** — the padding
leg is still proven on both Keplers. Three boxes, three different answers. Small device-local allocations stopped being poisonable on one card and
did not on the other, which is a real divergence in the new allocator's
behaviour and should be understood before that branch is relied upon.

### One honest consequence: the poison control now reports UNPROVEN

`poison_control.exs` came back `PASS with the padding leg UNPROVEN` — 0/20 dirty
at 4 B / 8 B, where before the change it was 20/20 on every box. Small
device-local allocations are no longer poisonable by the existing scheme, so the
zero-padding claim can no longer be established this way and the harness says so
instead of reporting clean.

**This is the first time that branch has fired anywhere.** It was written on the
argument that a control which cannot detect the defect must not report its
absence, and no box had ever exercised it. The concat checks are unaffected —
the 8 MiB scheme still dirties at 19/40 — but the padding leg needs a new
mechanism.

## VERIFIED: the branch restores the unified box exactly

The Jetson at `fad28e9` logs `unified memory: true (staging path: OFF)` and:

    KiB   PRICE base  regressed   fixed     fixed/base
     64        1.18       2.32     1.34        1.14x
    256        1.83       2.49     1.74        0.95x
   1024        1.01       2.77     1.04        1.03x
   4096        0.96       2.14     0.95        0.98x
  16384        0.94       1.45     0.94        1.00x

Everything at >=1 MiB is back within 5%, and 16 MiB is exact to two decimals.

**The acceptance test was the shape, not the timing**, and the shape returned:

    baseline   0.87x, 0.84x        (falling — the unified signature)
    regressed  1.22x, 1.15x, 1.21x (rising, toward the discrete shape)
    fixed      0.78x, 0.78x, 0.78x (falling, replicating to under 2%)

The allocator is no longer manufacturing the discrete signature on that box.
Variance recovered too, and beat baseline: PRICE spread at 16 MiB went
6.6% -> 38.8% -> **2.2%**.

Correctness exact on all three suites. `poison_control` unchanged at 20/20 —
still the opposite of super-io's 0/20, so that divergence is unrelated to the
branch and remains unexplained.

## OPEN: the device-side zero fill drifts upward and trips the VOID gate

The half of the change that stayed unconditional has a cost of its own, and the
Jetson found it while verifying the half that did not.

    commit     zero-fill method      24 MiB zeroed   allocation drift
    4e271e0    host-side write            5.24 ms    0.8%, 0.5%
    1c575cc    device fill_buffer      3.87 ms       14.0, 1.7, 5.8, 9.8%
    fad28e9    device fill_buffer      3.9-4.3 ms    12.1, 13.3, 9.3%

The device-side fill is genuinely ~23% faster on the mean, which is why it was
kept unconditional. But its **within-run drift went from under 1% to 9-13%**,
which sits on the harness's 10% VOID threshold and trips it about two runs in
three. Compute drift over the same runs is 0.0-0.2%, so the GPU is steady and
Race 2 is unaffected — the instability is confined to `alloc_buffer_zeroed`.

It drifts *upward* over a run, which is the opposite of the cold-start effect
mac-247 found in the first size measured, so it is not the same phenomenon.

A plausible mechanism, not yet tested: `alloc_buffer_zeroed` now builds a
command buffer and does a full `submit_and_wait` **per allocation**, where it
previously did a host-side memset and no submission at all. That is a queue
submit and a fence wait added to every zeroed allocation, and command-pool state
accumulating across a run would produce exactly an upward walk.

If that is the cause, the fix is to record the fill into the pending batch
rather than submitting synchronously — ordering is preserved because the fill
and the dispatch land in the same command buffer in order. `poison_control` is
the right check, since the four `allany_*` shaders are precisely the ones that
depend on the zeroing being real.

**Until this is understood, that box will VOID most runs for a reason unrelated
to whatever is being tested**, which makes future work there harder to read.

## Not yet checked

* Whether the Keplers and the Jetson show the same `txpci` signature. FreeBSD's
  nvidia-smi may not support `dmon`; the Jetson has no nvidia-smi at all and
  would need a different counter.
* Why inputs land in device-local memory while outputs do not, given both go
  through the same filter. `Buffer::from_iter` (upload) and `Buffer::new_slice`
  (output) may be satisfied differently by vulkano's allocator.
* Whether the 64 MiB collapse to 3.1 GB/s is the same effect compounded by the
  dedicated-allocation cliff, or something additional.
