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

(empty — workstream blocked until W3 lands GPUNode.Server)
