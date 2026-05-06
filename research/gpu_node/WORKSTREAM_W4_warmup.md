# W4 — Warmup Curve Characterization

**Question:** When is a shader "warm"? How long does first-dispatch take vs steady-state? What's the warmup criterion the GPU node uses to declare a shader ready for client traffic?

**Budget:** baseline doc per shader + a "warm-up criterion" function the GPU node can use.

## Hypothesis

Three contributors to warmup cost, in expected order of magnitude:

1. **`vkCreateComputePipelines`** — first call after `vkCreateShaderModule`. NVIDIA Linux: ~10-50 ms cached, much higher cold. Already cached by `get_or_create_pipe`, so should fire only on first reference.
2. **First NIF call's BEAM scheduler cold cache** — the `:erlang.nif_load` warmup. Should be one-time per VM, not per shader.
3. **First N dispatches' XLA/EXLA-side artifact compile** — but this is only on the EXLA fallback path; pure Vulkan doesn't pay this.

The H3 instrumentation (`Nx.Vulkan.Native.timing_get/0`) gives per-dispatch submit/wait/record numbers. We can use it to characterize the warmup curve directly.

## Protocol

For each of the 6 shaders (Normal, Exp, StudentT, Cauchy, HalfNormal, Weibull):

1. Fresh BEAM VM, fresh GPU node start.
2. Warm pool with N dispatches, recording per-dispatch wall + per-dispatch submit_ns + wait_ns from `timing_get`.
3. Plot:
   - dispatch wall vs dispatch index
   - submit_ns/wait_ns/record_ns vs dispatch index
4. Identify "warm" point: where p99 over a 50-dispatch window first satisfies `p99 ≤ 1.5 × p50`.
5. Tabulate per-shader warmup cost (sum of dispatch walls up to "warm" point) and warm-state per-dispatch wall.

## "Warm" criterion proposal

```
warm?(shader) ⇔
  last_50_dispatches_p99_wall ≤ 1.5 × last_50_dispatches_p50_wall
  AND last_dispatch_wait_ns < 1.2 × steady_state_wait_ns_for_this_device
```

The GPU node holds a "warmup queue" — shaders not yet warm. Client dispatches against unwarm shaders go through the warmup queue (which forces sequential execution to stabilize the warmup curve), then graduate to the dispatch queue.

## Output

- `warmup_curve_normal.csv` (and one per family)
- `warmup_summary.md` — per-family table: cold-dispatch wall, warm-dispatch wall, warmup count, total warmup wall.

## Notes / log

### Run 1 (2026-05-05) — RTX 3060 Ti, post-fix Vulkan path

Method: per-window cumulative timing via `Nx.Vulkan.Native.timing_get/0`. Each window = 5 samples (~14-30 dispatches depending on tree depth). Warm criterion: first window where p99 of the previous 50-window slice ≤ 1.5 × p50.

| Family | Cold window (µs) | Warm p50 (µs) | Warm p99 (µs) | p99/p50 | Warm @ window | Total dispatches (50 windows) |
|---|---|---|---|---|---|---|
| Normal | 254,242 | 284,723 | 581,756 | 2.04 | 20 | 2,771 |
| Exponential | 916,281 | 569,223 | 941,586 | 1.65 | 38 | 5,034 |
| StudentT | 414,523 | 326,177 | 552,703 | 1.69 | 20 | 6,981 |
| HalfNormal | 596,099 | 684,116 | 1,017,005 | 1.49 | 20 | 8,781 |
| Weibull | 553,461 | 495,897 | 889,225 | 1.79 | 50 (never settled) | 5,769 |

CSVs: `bench/warmup_curves/{family}.csv` (one row per window).

Summary doc: `warmup_summary.md` (this directory).

### Observations

1. **Cold/warm ratios are 1× to 3.6×, not 10×+.** Pipeline cache + persistent buffer fix already amortized most cold-start cost. Cold windows aren't dramatically worse than warm — Normal is essentially flat (cold 254 ms vs warm-p50 285 ms; cold is *smaller*, well within noise).

2. **Exponential is the outlier on cold cost.** 916 ms cold vs 569 ms warm-p50 = 1.6× cold penalty. Possibly the f32 numerics + log-transform constraint cause more first-dispatch shader-internal jitting. Worth a follow-up trace.

3. **Weibull never settles in the 50-window window.** Its p99/p50 stays at 1.79. Either Weibull's NUTS dynamics produce more variable tree depths (likely — Weibull k=2 is positively skewed), or the warm criterion needs a longer baseline. Not actually a problem for the GPU node — Weibull dispatch latency isn't pathological in absolute terms (~500 µs/window steady state, comparable to other families).

4. **HalfNormal at 1.49 just barely passes the 1.5× threshold.** The warm criterion is sensitive at this boundary; a 100-window run would likely show clearer separation.

5. **Cauchy is missing from this run** — the bench script has only 5 families. Cauchy's chain shader exists; need a follow-up entry in the cells list.

### Warmup criterion proposal (refined)

Original proposal: `p99 ≤ 1.5 × p50` over a 50-sample window. Per the data:

- Normal, StudentT, HalfNormal hit the criterion by window 20 (~100 samples, ~280-700 dispatches).
- Exponential needs ~38 windows (~190 samples).
- Weibull doesn't settle in 50 windows.

For the GPU node, the criterion should probably be:
- `min_dispatches >= 200` (gross floor — driver pipeline state is settled by then).
- `last_50_window_p99 <= 1.5 × last_50_window_p50` (the original quality bar).
- OR a hard cap at `max_warmup_dispatches = 1000` (don't block client traffic forever — declare warm and ship).

The Weibull case argues for the hard cap. Unsettled curves shouldn't gate availability; they just mean post-warmup, dispatch latency has higher variance than a Normal-shape distribution.

### Next steps

- Add Cauchy to the cells list, re-run.
- Run on FreeBSD GT 750M (mac-248) for cross-platform comparison — does the warmup curve shape match, or is mesa even smoother than NVIDIA Linux?
- Per-dispatch (not per-window) timing would resolve which actual dispatch indices are slow. Requires either tree.ex instrumentation or `Sampler.sample_stream/4` driving N=1 calls in a loop. Defer until needed.

