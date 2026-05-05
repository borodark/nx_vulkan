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

(empty)
