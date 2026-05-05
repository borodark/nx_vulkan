# Hypothesis: Linux Vulkan NUTS overhead is outside the chain shader

## Evidence from FreeBSD GT 750M audit (2026-05-05)

| Metric | FreeBSD GT 750M | Expected Linux RTX 3060 Ti |
|--------|-----------------|---------------------------|
| Raw chain dispatch (K=32) | 617 µs | ~500 µs (measured earlier) |
| Per leapfrog step | 19.3 µs | ~15 µs |
| Fused NUTS iter (50+50) | 3.3 ms/iter | should be ~2-3 ms/iter |
| Unfused NUTS iter | 283 ms/iter | — |
| Fused/unfused speedup | **86.7×** | — |
| Full race (1000+1000) | 1.4-1.9s | **32-55s** (20-30× slower) |

The chain shader dispatches are comparable across hosts (~500-617µs).
The full NUTS sampling is **20-30× slower on Linux** despite faster
raw dispatch. The overhead is not in the shader.

## Where to look

### H1: speculative path re-dispatching

The NUTS tree builder's speculative path (`ensure_available/3` in
`lib/exmc/nuts/tree.ex`) calls the chain shader to pre-compute K
steps. If the tree builder is **re-requesting already-computed steps**
(cache miss in the speculative buffer), dispatch count inflates.

**Test**: add a counter in `do_dispatch/10`:

```elixir
# In lib/exmc/nuts/tree.ex, top of do_dispatch:
n = Process.get(:dispatch_count, 0)
Process.put(:dispatch_count, n + 1)
```

After a 100+100 run, read `Process.get(:dispatch_count)`. If it's
>> 200 (warmup + sample iterations), the speculative path is
over-dispatching.

**Expected on FreeBSD**: ~200-400 dispatches for 100+100.
**If Linux shows**: 5000+ → speculative re-dispatch is the culprit.

### H2: step-size adaptation thrashing

`find_reasonable_epsilon` and the dual-averaging warmup call the
log-prob + gradient repeatedly. If these calls go through the
**unfused** per-op path (because `fused_leapfrog_meta` isn't
consulted during adaptation), the warmup phase dominates.

**Test**: time warmup vs sampling separately:

```elixir
{us_warmup, _} = :timer.tc(fn ->
  Sampler.sample(ir, %{}, num_warmup: 100, num_samples: 0, seed: 42)
end)
{us_sample, _} = :timer.tc(fn ->
  Sampler.sample(ir, %{}, num_warmup: 0, num_samples: 100, seed: 42)
end)
IO.puts("warmup: #{div(us_warmup, 1000)}ms")
IO.puts("sample: #{div(us_sample, 1000)}ms")
```

If warmup >> sample: adaptation is the bottleneck, not the chain shader.

### H3: Evaluator fallback on non-chain defns

The NUTS tree builder has multiple `defn` functions beyond the
leapfrog body (kinetic energy, joint log-prob, U-turn check). If
these don't hit the fused path, each evaluates via per-op dispatch
(~500µs × N ops per call × M calls per iteration).

**Test**: `NXV_FUSE_DEBUG=1` during a 10+10 run. Count how many
`no_match` lines appear. Each `no_match` is a defn that falls
through to Evaluator (per-op dispatch).

**FreeBSD result**: 3 `no_match` calls during init, then the
chain shader handles the hot path. If Linux shows continuous
`no_match` during sampling, those defns aren't being routed
through the chain shader.

### H4: BEAM scheduling / GC pressure

Erlang's scheduler + GC can introduce pauses. With persistent
Vulkan buffers, each tensor is a NIF resource — the BEAM GC
must trace and potentially finalize them. High churn of short-lived
tensors (from non-fused ops in the tree builder) creates GC
pressure.

**Test**: run with `+MIscs 256` (increase scheduler-specific
carrier size) and `ERL_FULLSWEEP_AFTER=0` to see if GC pauses
contribute.

### H5: pipeline cache miss

Spirit's `get_or_create_pipe` caches pipelines by
`(spv_path, spec_constant, n_buffers)`. If the cache key
computation is slow or the cache is being evicted, each dispatch
recreates the pipeline (~22ms on RTX 3060 Ti per the
RESULTS doc).

**Test**: add a hit/miss counter in `get_or_create_pipe` in
`c_src/nx_vulkan_shim.cpp`. After a 100+100 run, report hit
rate. Should be >99% after warmup.

## Recommended investigation order

1. **H1** (dispatch counter) — 5 min, most likely culprit
2. **H2** (warmup vs sample split) — 5 min
3. **H3** (NXV_FUSE_DEBUG count) — already have the env var
4. **H5** (pipeline cache hit rate) — 10 min C++ edit
5. **H4** (GC pressure) — last resort

## What FreeBSD's numbers prove

The chain shader architecture is sound. At 19.3 µs/step on a 2013
GPU, the fused leapfrog is **faster than EXLA's per-call CUDA
overhead** for small d. The 86.7× speedup over unfused Vulkan
confirms the dispatch-count-reduction thesis from
RESEARCH_FAST_KERNELS.md.

The Linux gap is an integration issue, not a shader issue.
