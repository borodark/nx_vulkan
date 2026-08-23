# DTrace for Vulkan Compute Profiling on FreeBSD

> **STALE BELOW THE QUICK START — corrected 2026-08-23.** Every probe name this
> document originally gave is dead, and its flagship "profile the full dispatch
> stack" script does not compile. The advice was written against a C++ shim that
> no longer exists; the NIF is Rust (Rustler + vulkano) now. Section
> **"Probes that actually work"** immediately below is the current truth. The
> rest is kept for the technique, not the symbol names.

## Probes that actually work (verified on mac-247, 2026-08-23)

**`nxv_*` is gone.** `nm priv/native/libnx_vulkan_vulkano.so | grep -c nxv_`
returns 0. So is `timing_get/0` — no such function anywhere in `lib/` or
`native/`, so this doc's "use it for routine benchmarking" advice is
unactionable.

**Loader-level Vulkan probes look usable and are not.** `vkQueueSubmit` and
`vkWaitForFences` exist in `libvulkan.so.1` and appear in `dtrace -l`, so they
seem probeable — but ash/vulkano resolves driver pointers through
`vkGetDeviceProcAddr` and calls the ICD directly, bypassing the loader. Probing
them records **zero events**. They also have no `:return` probes (tail-call
trampolines), which is why the script further down fails to compile at its first
line. And the code never waits on a fence at all: it uses `queue.wait_idle()`,
i.e. `vkQueueWaitIdle`.

Use these instead. All are local `t` symbols — **this works only because the
release build is unstripped**; the `.so` exports just `nif_init` dynamically, so
adding `strip = true` to the release profile would delete every probe here.

| phase | probe glob |
|---|---|
| record one dispatch | `_ZN17nx_vulkan_vulkano16enqueue_dispatch17h*` (6 monomorphisations) |
| one submission | `_ZN17nx_vulkan_vulkano15submit_and_wait17h*` |
| submit + wait pair | `_ZN17nx_vulkan_vulkano17finish_and_disarm17h*` |
| **submit** | `*CommandBufferExecFuture*5flush17h*` |
| **queue wait** | `*QueueState9wait_idle17h*` |

Both of the last two have `:entry` and `:return`. `doas dtrace` is passwordless
on the fleet.

## What the breakdown actually says

Measured on the GT 650M while running `bench/window_reduce_fold_vs_host.exs`:

| per submission | batch=64, 512² 3×3 | batch=0, 512² 3×3 | batch=0, 64² 2×2 |
|---|---|---|---|
| dispatches / submission | **18.0** | 1.0 | 1.0 |
| build cmdbuf | 148.9 µs | 16.8 µs | 15.6 µs |
| **submit** | 46.7 µs | 11.6 µs | **10.9 µs** |
| **queue wait** | **7281.6 µs (96.7%)** | **489.8 µs (92.3%)** | **127.2 µs (75.1%)** |
| total `submit_and_wait` | 7531 µs | 531 µs | **169.5 µs** |

Recording a dispatch *without* submitting costs **4.5 µs**.

**The fixed floor is ~170 µs per submission and it is three-quarters queue-wait.
But it is charged per SUBMISSION, not per dispatch** — `enqueue_dispatch`
batches up to `NXV_BATCH_MAX` (64) into one command buffer, and a 3×3 window
fold's 18 dispatches were measured riding in a single queue wait. Disabling
batching entirely costs 1.4×, not the order of magnitude a pure launch-overhead
model predicts.

**That matters because it corrects a recorded belief.** `{:reduce, 5}`'s
allowlist entry attributes its loss to "per-dispatch launch overhead". At
`reduce_size` 4096 that would be 4096/64 = 64 submissions ≈ 11 ms of queue wait
against a measured 441 ms — under 3%. The conclusion stands (the host wins) but
the mechanism does not: what scales is per-dispatch WORK — descriptor sets,
buffer churn, GPU execution — not launch cost.

**Two cautions.** DTrace is not free at this rate: probing all six
`enqueue_dispatch` monomorphisations plus submit and wait took 512² 3×3 from
10.4 ms to 16.9 ms per fold, about 60%. Aggregates stay sound; wall-clock taken
*during* tracing does not. And profiling a benchmark that interleaves two arms
can mislead — see the GC note in `bench/window_reduce_fold_vs_host.exs`.

---

## Why DTrace

FreeBSD ships DTrace in the base system — no packages, no kernel
modules, no recompilation. It traces userland functions in live
processes with zero instrumentation overhead when idle. For profiling
Vulkan dispatch paths (NIF → C++ shim → Vulkan driver), DTrace gives
per-call latency without modifying the code.

We used two complementary approaches during the nx_vulkan performance
investigation:

1. **Built-in timing NIFs** (`timing_get/0`) — atomic counters in
   the C++ shim, exposed via Rustler. Cheap, always available, but
   only measures what we instrumented.
2. **DTrace** — traces any function in the process, including
   Vulkan driver internals we don't control. Needs root/doas.

## Quick Start

### Count NIF calls during a sampling run

```sh
# Terminal 1: start the BEAM
cd ~/exmc/exmc
NX_VULKAN_PATH=$HOME/nx_vulkan EXMC_COMPILER=vulkan \
  iex -S mix

# In iex:
IO.puts(System.pid())   # e.g., "42543"
# ... start sampling ...
```

```sh
# Terminal 2: attach DTrace (needs root)
doas dtrace -p 42543 \
  -n 'pid$target::nxv_*:entry { @[probefunc] = count(); }' \
  -n 'tick-10s { printa(@); exit(0); }'
```

Output:

```
  nxv_buf_alloc                  47
  nxv_buf_download               12
  nxv_buf_free                   35
  nxv_buf_upload                 18
  nxv_apply_binary               84
  nxv_apply_unary                42
  nxv_leapfrog_chain_normal     200
  nxv_reduce                     14
```

### Measure per-call latency of Vulkan fence waits

```sh
doas dtrace -p $BEAM_PID -n '
pid$target::submit_and_wait:entry { self->ts = timestamp; }
pid$target::submit_and_wait:return /self->ts/ {
    @latency = quantize(timestamp - self->ts);
    self->ts = 0;
}
tick-10s { printa(@latency); exit(0); }
'
```

Output (power-of-2 histogram, nanoseconds):

```
           value  ------------- Distribution ------------- count
          131072 |                                         0
          262144 |@@@@@@@@@@@@@@@@@@@@@@@@@@@              189
          524288 |@@@@@@@@@@@@@                            91
         1048576 |                                         3
         2097152 |                                         0
```

Reads: most `submit_and_wait` calls complete in 262-524 µs on the
GT 750M. Compare to Linux NVIDIA where the same call takes 1-2 ms.

### Trace Vulkan driver calls

The FreeBSD NVIDIA driver exposes symbols that DTrace can probe:

```sh
# List available Vulkan driver probes
doas dtrace -l -p $BEAM_PID | grep -i vk | head -20
```

```sh
# Time vkQueueSubmit specifically
doas dtrace -p $BEAM_PID -n '
pid$target::vkQueueSubmit:entry { self->ts = timestamp; }
pid$target::vkQueueSubmit:return /self->ts/ {
    @submit = avg(timestamp - self->ts);
    self->ts = 0;
}
tick-10s { printa(@submit); exit(0); }
'
```

### Profile the full dispatch stack

> **This script does NOT compile.** `vkQueueSubmit:return` matches no probes
> (tail-call trampoline), and the `nxv_*` and loader-level entry probes never
> fire. Kept to show the shape; substitute the working globs from the top of
> this file.

```sh
doas dtrace -p $BEAM_PID -n '
/* Measure each phase of a dispatch call */

pid$target::nxv_leapfrog_chain_normal:entry {
    self->disp_start = timestamp;
}

pid$target::vkQueueSubmit:entry /self->disp_start/ {
    self->submit_start = timestamp;
}

pid$target::vkQueueSubmit:return /self->submit_start/ {
    @submit_us = avg((timestamp - self->submit_start) / 1000);
    self->submit_start = 0;
}

pid$target::vkWaitForFences:entry /self->disp_start/ {
    self->wait_start = timestamp;
}

pid$target::vkWaitForFences:return /self->wait_start/ {
    @wait_us = avg((timestamp - self->wait_start) / 1000);
    self->wait_start = 0;
}

pid$target::nxv_leapfrog_chain_normal:return /self->disp_start/ {
    @total_us = avg((timestamp - self->disp_start) / 1000);
    self->disp_start = 0;
}

tick-15s {
    printf("\n--- Dispatch breakdown (µs, averages) ---\n");
    printa("  submit:  %@d\n", @submit_us);
    printa("  wait:    %@d\n", @wait_us);
    printa("  total:   %@d\n", @total_us);
    exit(0);
}
'
```

## What We Measured (2026-05-05)

### R3.2: Per-fence timing via built-in NIFs

Normal(0,1) d=1, 100+100 iterations, 352 dispatches:

| Metric | FreeBSD GT 750M | Linux RTX 3060 Ti | Ratio |
|--------|-----------------|-------------------|-------|
| submit | 11.6 µs | 138 µs | 12× |
| wait | 406 µs | 1,130 µs | 2.8× |
| record | 4.3 µs | 19 µs | 4.4× |
| **total** | **422 µs** | **1,287 µs** | **3.1×** |

### Interpretation

The 3.1× per-fence gap explains the entire wall-time difference
between FreeBSD and Linux Vulkan runs. The chain shader compute
is identical (same SPIR-V, same algorithm); the difference is
how long the CPU blocks waiting for the GPU to signal fence
completion.

FreeBSD's NVIDIA driver (470.256.02) appears to use a tighter
polling loop or lower-latency fence signaling path than Linux's
NVIDIA driver on the same driver version. This is consistent
with FreeBSD's historically lower-latency syscall path and the
fact that the GT 750M's PCIe bus is Gen2 x16 (shorter round-trip
than the RTX 3060 Ti's Gen4 x16, though Gen4 should be faster —
suggesting the driver implementation, not the bus, is the
bottleneck on Linux).

## DTrace vs Built-in Timing

| Aspect | DTrace | timing_get/0 NIF |
|--------|--------|-----------------|
| Setup | needs root/doas | none |
| Overhead | ~100ns per probe | ~20ns per atomic increment |
| Scope | any function in process | only instrumented points |
| Vulkan internals | yes (vkQueueSubmit etc) | no |
| Production safe | yes (disable = zero cost) | yes (always on) |
| Cross-platform | FreeBSD, illumos, macOS | anywhere the NIF compiles |

**Recommendation**: use `timing_get/0` for routine benchmarking,
DTrace for deep investigation of driver-level behavior.

## Prerequisites

```sh
# DTrace is in FreeBSD base — no install needed
dtrace -V    # Sun D 1.13

# For userland tracing, the BEAM must NOT be stripped
# (pkg-installed Erlang is fine; custom builds need --enable-dtrace)
file $(which beam.smp) | grep "not stripped"

# doas config (user must be in wheel group)
pw groupmod wheel -m $USER
echo 'permit persist :wheel' > /usr/local/etc/doas.conf
```

## Caveats

1. **DTrace pid provider attaches per-process.** Start the BEAM
   first, get its PID, then attach. DTrace cannot trace a process
   that hasn't started yet (use `-c` flag to launch and trace
   simultaneously, but this is awkward with `mix run`).

2. **Symbol visibility.** The Rustler NIF `.so` and the Vulkan
   driver `.so` must not be stripped. FreeBSD's pkg-installed
   NVIDIA driver retains symbols; custom builds may not.

3. **Overhead on hot paths.** Tracing every `submit_and_wait` at
   2000 calls/second adds ~200 µs total — negligible. Tracing
   every `vkCmdDispatch` in a tight loop may add measurable
   overhead; use `tick-Ns` aggregation instead of per-call printf.

4. **No DTrace on Linux.** Linux has `bpftrace` (similar syntax,
   different backend). The equivalent of the scripts above in
   bpftrace:
   ```
   bpftrace -p $PID -e 'uprobe:/path/to/libnx_vulkan_native.so:submit_and_wait {
       @start[tid] = nsecs;
   }
   uretprobe:/path/to/libnx_vulkan_native.so:submit_and_wait /@start[tid]/ {
       @latency = hist(nsecs - @start[tid]);
       delete(@start[tid]);
   }'
   ```
