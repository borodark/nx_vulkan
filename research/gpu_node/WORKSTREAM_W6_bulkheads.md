# W6 — Bulkheads + Recovery

**Status:** BLOCKED on W3 (need GPUNode.Server to bulkhead).

**Question:** What does failure look like, and how do clients survive a GPU-node crash without dropping inference work on the floor?

## Failure modes to characterize

1. **Bad shader — compile failure.** `glslc` rejects synthesized GLSL. Easy: surface error to client.
2. **Bad shader — validation failure.** Shader compiles + runs but its output distribution doesn't match the EXLA reference (W2 detects this). Easy: don't register the shader.
3. **Bad shader — infinite loop.** Shader compiles + validates on small inputs but loops forever on a different push-constant value. Hard: NVIDIA Linux driver may not be cancellable. Verify behavior of `vkQueueWaitIdle` cancellation, `vkResetCommandPool` mid-dispatch.
4. **GPU OOM.** Pool fragmentation or trial-mode overload. Pool already returns NULL on alloc failure; need clean error path through the dispatch chain.
5. **Driver crash / GPU hang.** TDR (Linux equivalent: GPU resets, all submissions die, vkDeviceLost from every API call). The GPU node must die and restart cleanly. All clients fall back to EXLA.
6. **GPU node OOM (RAM, not VRAM).** BEAM heap grows. Standard supervisor restart.

## Watchdog design

Per-dispatch:

```
Task.Supervisor.async_nolink(GPUWatchdog, fn ->
  do_dispatch(...)
end)
|> Task.yield(@dispatch_timeout_ms)
|> case do
  {:ok, result} -> result
  nil -> 
    # Timeout. Try to cancel.
    Task.shutdown(task, :brutal_kill)
    {:error, :gpu_dispatch_timeout}
end
```

Where `@dispatch_timeout_ms` is set per-shader from W4's warmup characterization (e.g., `5 × p99_warm_dispatch_wall`).

On timeout:
- Mark the shader as :suspect.
- Fail the client request (client falls back to EXLA path).
- After N consecutive timeouts on the same shader, evict it.
- After M consecutive timeouts on different shaders within a window, suicide the GPU node (driver is probably hosed).

## Client fallback contract

```elixir
case Exmc.GPUNode.dispatch(spec, args, timeout: 5_000) do
  {:ok, result} -> result
  {:error, :gpu_unavailable} -> exla_fallback(args)
  {:error, :gpu_dispatch_timeout} -> exla_fallback(args)
  {:error, _other} -> exla_fallback(args)
end
```

The fallback IS the EXLA path that already exists in `tree.ex`'s catch-all dispatch clause:

```elixir
defp do_dispatch(_meta, _compiler, spec_buf, q, p, grad, eps_t, n_t, _k, _dir_sign) do
  spec_buf.multi_step_fn.(q, p, grad, eps_t, spec_buf.inv_mass_diag, n_t)
end
```

Good — we don't need new fallback logic, just a way to opt out of the chain-shader path on a per-call basis.

## Chaos test

A test that:
- Synthesizes a deliberately bad shader (push-constant-controlled infinite loop).
- Submits it via the watchdog path.
- Verifies the timeout fires.
- Submits 10 more dispatches against the same shader.
- Verifies the shader gets evicted, GPU node continues serving other shaders.
- Submits 100 dispatches against an unrelated good shader.
- Verifies they all succeed.

## Output

- `failure_modes.md` — per-mode, what we observe and how we recover.
- `chaos_test.exs` — runnable harness.

## Notes / log

### Phase 0 W6 (2026-05-05) — minimal timeout + EXLA fallback

Shipped on `pymc@feat/gpu-node`:

- `Exmc.GPUNode.Server.chain_dispatch/9` now wraps `GenServer.call`
  with a try/catch on `:exit`. Reads `Application.get_env(:exmc,
  :gpu_node_timeout_ms, :infinity)`. Returns `{:error,
  :gpu_dispatch_timeout}` on timeout, `{:error, :gpu_node_dead}`
  on server-not-found.
- `Exmc.NUTS.Tree.route_chain/12` accepts the full `spec_buf` as the
  fallback handle. On error tuple from the GPU node, calls
  `spec_buf.multi_step_fn.(q, p, grad, eps_t, inv_mass, n_t)` —
  which is the existing EXLA path used by the catch-all dispatch
  clause.
- Test: `test/exmc/gpu_node/bulkhead_test.exs`:
  - Timeout case: `:gpu_node_timeout_ms = 1` triggers the watchdog,
    returns the error tuple.
  - Dead-server case: server stopped → `{:error, :gpu_node_dead}`.
  - End-to-end: with 1 ms timeout active, sampler completes 100 warmup
    + 100 samples via the EXLA fallback. Posterior mean within ±0.5 of
    target (Normal(0,1) → mean ≈ 0).

3 tests, 0 failures, 13.8 s on RTX 3060 Ti.

### What's NOT shipped yet (Phase 1 W6)

- **Per-shader :suspect tracking + eviction.** Today every timeout
  triggers a fallback but doesn't blacklist the shader. A genuinely
  bad shader will time out forever, paying the timeout penalty
  every dispatch. Phase 1: track consecutive timeouts per shader,
  evict after N (e.g. 3) consecutive failures.
- **GPU node suicide on M timeouts across shaders.** If the entire
  driver is hosed, the right answer is restart, not per-shader
  eviction. Need a cross-shader window counter.
- **Chaos test.** No deliberately-bad-shader test yet. The Phase 0
  test uses a 1 ms timeout on a healthy shader to simulate the
  watchdog firing — sufficient to verify the fallback contract,
  insufficient to verify driver recovery semantics under genuine
  bad inputs.
- **GenServer process unstuck after timeout.** Today, when the
  watchdog fires, the GenServer is still blocked on the NIF call
  until that actually returns. Subsequent calls queue behind it.
  For Phase 1: Task.Supervisor.async_nolink + Task.shutdown(:brutal_kill)
  to actually kill the in-flight dispatch — but this leaks the GPU
  buffer state and may hang the driver. Investigation needed.
- **Driver-level recovery** (`vkResetCommandPool`, `vkQueueWaitIdle`
  cancellation). All Phase 2+.

### Files

- `pymc/exmc/lib/exmc/gpu_node/server.ex` — `chain_dispatch/9` with timeout.
- `pymc/exmc/lib/exmc/nuts/tree.ex` — `route_chain/12` with fallback.
- `pymc/exmc/test/exmc/gpu_node/bulkhead_test.exs` — 3 cases.

### Refined client fallback contract

Working contract from the test:

```elixir
case Exmc.GPUNode.Server.chain_dispatch(...) do
  {:error, :gpu_dispatch_timeout} -> exla_fallback()
  {:error, :gpu_node_dead}        -> exla_fallback()
  {:error, _other}                 -> exla_fallback()
  result                           -> result
end
```

The fallback IS `spec_buf.multi_step_fn` — the same EXLA path the
catch-all dispatch clause uses. No new fallback logic needed.

