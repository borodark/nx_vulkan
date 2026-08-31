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

## Not yet checked

* Whether the Keplers and the Jetson show the same `txpci` signature. FreeBSD's
  nvidia-smi may not support `dmon`; the Jetson has no nvidia-smi at all and
  would need a different counter.
* Why inputs land in device-local memory while outputs do not, given both go
  through the same filter. `Buffer::from_iter` (upload) and `Buffer::new_slice`
  (output) may be satisfied differently by vulkano's allocator.
* Whether the 64 MiB collapse to 3.1 GB/s is the same effect compounded by the
  dedicated-allocation cliff, or something additional.
