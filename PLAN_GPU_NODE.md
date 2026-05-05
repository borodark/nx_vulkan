# PLAN — Long-Lived GPU Node for On-Demand Shader Synthesis

**Status:** research / scoping
**Authors:** io@octanix.com (Linux dev box, super-io)
**Cross-references:**
- `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md` — chain-shader history (Phase 2 closed)
- `~/projects/learn_erl/nx_vulkan/248_TODO.md` — R1/R2/R3 hypothesis arc + measurement protocol
- `~/projects/learn_erl/nx_vulkan/FAIR_RACE_FREEBSD.md` — FreeBSD GT 750M baseline
- `~/projects/learn_erl/pymc/exmc/bench/fair_race_results_linux.md` — Linux RTX 3060 Ti pre/post-fix race
- `~/.claude/projects/-home-io-projects-learn-erl-pymc/memory/zed_project.md` — Zed declarative BEAM deploy plan (must come AFTER this plan lands)

---

## Context: how we got here

### The starting position

Six hand-written GLSL chain shaders for Normal, Exponential, StudentT, Cauchy, HalfNormal, and Weibull. Compiled to SPIR-V at build time via `glslc`, vendored into `nx_vulkan/priv/shaders/`. Each new distribution requires a developer to:

1. Hand-derive `dlogp/dq` for the family.
2. Write GLSL implementing K=32 leapfrog steps with that gradient.
3. Compile to SPIR-V offline.
4. Vendor the binary.
5. Add an Elixir dispatch clause in `exmc/lib/exmc/nuts/tree.ex` matching a tagged tuple like `{:beta, alpha, beta}`.

All six families work, all pass the fair race against EXLA, and Vulkan beats EXLA on Normal d=8/d=50 + StudentT (with `+sbt tnnps`). But the model has a hard ceiling: any new distribution is a multi-step manual effort.

### The R1-R3 arc proved out long-lived state inside one process

Mac-248's FreeBSD GT 750M finished the same chain-shader race in 1.4-1.9 sec (1000W + 1000S). Linux RTX 3060 Ti — a physically much faster GPU — was taking 32-55 sec on the same code. The 18-30× gap was physically impossible without an external confound, so we walked five hypotheses:

| H | Hypothesis | Result |
|---|---|---|
| H1 | Live trial GPU contention (pid 82641 holding CUDA context for 15 days) | RULED OUT — 5.7% delta after kill, within seed noise |
| H2 | Warmup vs sample split (compile cost concentrated in warmup) | RULED OUT — 59% warmup / 41% sample, proportional to dispatch count |
| H3 | Per-fence wait latency (NVIDIA Linux blocking-fence-wait floor) | CONFIRMED — 1.13 ms wait : 138 µs submit per `submit_and_wait`. Wait is the floor |
| H4 | BEAM scheduler interaction with NIF latency | CONFIRMED — `+sbt tnnps` gives +28% on Vulkan, +25% on EXLA |
| H5 | Pipeline cache miss per dispatch | RULED OUT — `get_or_create_pipe` already caches by SPV path |

The H3 instrumentation (atomic `vkQueueSubmit` / `vkWaitForFences` counters in `Backend_par_vulkan.cpp`) revealed the actual root cause: **8 round-trips per chain dispatch**:

```
vulkan_upload(q)              → alloc + upload + submit_and_wait    [1]
vulkan_upload(p)              → alloc + upload + submit_and_wait    [2]
vulkan_upload(inv_mass)       → alloc + upload + submit_and_wait    [3]
Nx.Vulkan.leapfrog_chain_*    → submit_and_wait                     [4]
vulkan_to_tensor(q_chain)     → cmd_copy + submit_and_wait          [5]
vulkan_to_tensor(p_chain)     → cmd_copy + submit_and_wait          [6]
vulkan_to_tensor(grad_chain)  → cmd_copy + submit_and_wait          [7]
vulkan_to_tensor(logp_chain)  → cmd_copy + submit_and_wait          [8]
```

8 fences × 1.27 ms = ~10.2 ms/dispatch on Linux. Matches the H2 measurement almost exactly.

### The fix — persistent state at the *process* level

Two commits landed:

- `nx_vulkan@b2fc47d` — `upload_binary_into_batch2` (q+p batched), `download_binary_batch4` (4-buffer chain output batched), `timing_reset/get` instrumentation NIFs.
- `pymc/feat/dsl-shader-codegen@152da19eb` — Persistent q/p/inv_mass GPU buffers cached in process dict, batched IO across all 6 chain shader families.

**Result (Normal d=1, 1000/1000, 5 seeds):**

| Configuration | Wall (ms) | vs FreeBSD GT 750M |
|---|---|---|
| R1 (pre-fix, pre-trial-kill, default scheduler) | 32,260 | 19.5× |
| R3 (post-fix, default scheduler) | 18,009 | 10.9× |
| R3 + `+sbt tnnps` | **12,722** | **7.7×** |

Total Linux improvement: **2.5× wall-time**. Vulkan now beats EXLA on 4 of 7 cells (was 1 of 7).

### What's left on the table

The remaining 7.7× gap is the per-fence blocking-wait floor in the NVIDIA Linux Vulkan driver (~1.13 ms vs ~150 µs on mesa-radv). No host-side batching can reduce it. Next levers — `VK_KHR_synchronization2` semaphores, polling fence waits, or different driver entirely — are speculative.

But there's a more important next layer: **extend the persistent-state lifetime beyond one sampler process**.

---

## The vision

A long-lived GPU node — separate BEAM process or service — that owns the pipeline cache, owns the buffer pool, and synthesizes new shaders on demand. Multiple client BEAM processes (sampler workers, trading instruments, any consumer) talk to it.

```
client BEAM node                        GPU node (long-lived)
─────────────────                       ──────────────────────
 submit IR + dispatch spec  ─────────►   synthesize GLSL from spec
                                          glslc → SPIR-V (cached)
                                          vkCreateShaderModule
                                          vkCreateComputePipelines
                                          warm: N throwaway dispatches
                                          register {spec_hash, pipeline_id}
        ◄──────────────────────────  return pipeline_id
 dispatch(pipeline_id, q, p)  ────────►  bind cached pipeline + buffers
                                          submit + wait
        ◄──────────────────────────  return chain output
```

The persistent-buffer fix proved out **state-per-process**. The GPU node extends that to **state-per-machine**, with the additional capability that *new shaders can be requested at runtime*.

---

## Open research questions

The plan is to *enumerate then measure*, the same shape that worked for H1-H5.

| # | Question | Why it matters | How to answer |
|---|---|---|---|
| Q1 | What's the right shader codegen substrate? | Determines flexibility vs reliability — handwritten templates vs GLSL synthesis vs naga IR vs rspirv | Build three small prototypes: `(a)` parameterized GLSL templates with text substitution, `(b)` Elixir IR → GLSL transpiler, `(c)` direct SPIR-V via rspirv. Measure per-shader synthesis time, correctness, line count |
| Q2 | What's the validation contract? | A wrong shader silently corrupts inference. Need a gate before "warm" | Reference impl in EXLA. Statistical comparison: KS test on 10k draws, mean/var within 3σ. Reject if shader output distribution drifts. Property: posterior on known conjugate models matches analytic |
| Q3 | What's the dispatch protocol? | Distributed Erlang adds cookie/auth for free but couples versions. TCP+binary is leaner but reinvents framing | Prototype both: D-Erlang via `:rpc.call`, vs TCP with a single binary message format. Measure round-trip latency under load. Distributed Erlang likely wins for our case |
| Q4 | How do we discover the GPU node? | The `zed` plan uses mDNS. We could reuse that infrastructure | Reuse `mdns_lite` advertisement (`_exmc_gpu._tcp.local`). Each GPU node advertises {device, vram, compiler}. Client picks via affinity rules |
| Q5 | What's the eviction policy? | Pipeline cache grows unbounded; VRAM is finite | Track `last_used_at` per pipeline. LRU eviction when VRAM use crosses 80%. Persistent buffers tied to pipeline lifetime — destroyed on evict |
| Q6 | How does warmup work? | First dispatch on any pipeline pays compile + cache-miss cost. Need to *prove* it's warm before declaring ready | Run N throwaway dispatches per shader at registration time. Measure per-dispatch p50/p99 latency. Declare warm when p99 ≤ 1.5× p50 across 50 samples |
| Q7 | What's the failure model? | Bad shader → GPU hang → entire node down. Need bulkheads | Each pipeline runs under a watchdog: timeout the dispatch, kill the queue submission via VK_KHR_external_semaphore signal. Crash the GPU node process; supervisor restarts. Client retries with reference EXLA fallback |
| Q8 | Multi-tenancy: one node, many clients | Trading trial has 67 instruments, each could be a "client". Currently they share GPUScheduler permits at process level | Per-client virtual queues with weighted fair scheduling. Track GPU time per client. Hard cap on per-client VRAM. Same pattern as `GPUScheduler` extended cross-process |
| Q9 | Persistence across restarts | Pipeline cache rebuild is expensive. SPIR-V compile is expensive | Two layers: `vkPipelineCache` serialized to disk (driver-specific binary, opaque to us); SPIR-V cached as a content-addressed blob (`{spec_hash}.spv`). Both restored on startup |

---

## Research workstreams (parallel)

### W1 — Codegen substrate (highest unknown)

Pick the shader synthesis approach. Prototype Q1's three options against the existing 6 chain shaders. Goal: synthesize a Beta or Gamma chain shader in **<1000 ms end-to-end** (template render + glslc compile + vkCreateShaderModule + vkCreateComputePipelines + first-call statistical validation against a reference EXLA impl). The 1-second budget is realistic given `glslc` alone is 50-200 ms and validation needs ~100 throwaway dispatches.

**Risk:** GLSL synthesis is harder than it looks once derivatives enter the picture. The existing shaders hand-derived `dlogp/dq` for each family; an automated synthesizer needs either a closed-form gradient table per family, or symbolic differentiation, or autodiff in the shader (expensive).

**Inspiration:** TVM, Halide, Triton (CUDA), wgpu's wgsl-shader-builder. PyMC's PyTensor compile-to-C.

### W2 — Validation harness

Build the EXLA-reference vs synthesized-shader statistical comparison framework. Apply to existing 6 shaders first to validate the harness against known-good code. Then use it as the gate for W1's synthesized shaders.

**Reuse:** the `proper_statem` accumulator pattern from the NUTS bug story (statistical postcondition for distributional bugs — see `~/projects/learn_erl/pymc/www.dataalienist.com/blog-nuts-statem.html`). A shader that produces correct mean but wrong variance is the same shape of bug as a NUTS sampler with capped log-weights.

### W3 — Process model + protocol

Spike a `Exmc.GPUNode.Server` GenServer that holds the pipeline cache, exposes `register_shader/2`, `dispatch/3`, `evict/1`. Distributed Erlang transport. Reuse `mdns_lite` from the `zed` plan for discovery. Run client + server on same machine first; cross-machine in phase 2.

**Decision point:** does the GPU node serve via `:rpc.call`, or via a `:gen_statem` with explicit message passing? The latter is more expressive (back-pressure, request priorities) but more code.

### W4 — Warmup characterization

Instrument first-dispatch through 200th dispatch latency for each existing chain shader. Find the warmup curve. Determine "warm" criterion empirically.

Cross-correlate with H3 timing: how much of warmup is `vkCreateComputePipelines`, how much is the first NIF call's cold cache, how much is BEAM scheduler-state? The H3 instrumentation (`Nx.Vulkan.Native.timing_get/0`) is already in main and produces these numbers cheaply.

### W5 — Pipeline cache persistence

Investigate `vkPipelineCache` serialization. NVIDIA Linux supports it. If a cold restart can warm-start from a 10 MB pickled cache file, restart cost goes from ~30 s to ~3 s.

Two-layer cache:
- `~/.exmc/gpu_node/pipeline_cache/{device_uuid}.bin` — opaque vkPipelineCache blob
- `~/.exmc/gpu_node/spv/{spec_hash}.spv` — content-addressed compiled SPIR-V (re-usable across restarts and across devices of the same family)

### W6 — Bulkheads + recovery

What does a watchdog timeout look like in practice? How long does the NVIDIA driver take to recover from a `vkQueueWaitIdle` cancellation? Can we even cancel?

Spike with deliberately bad shaders (infinite loops via push-constant inputs the shader trusts). Find the failure modes before production does. Confirm clients fall back to EXLA cleanly.

---

## Phased prototyping plan

### Phase 0 (week 1) — Single-process spike

Existing `nx_vulkan` already has the pipeline cache (`get_or_create_pipe` in `c_src/nx_vulkan_shim.cpp`). Wrap it in a GenServer. Move it from "implicit per-process state via dlopen" to "explicit named GenServer with a state map." No new shaders yet; just port the existing 6.

**Deliverable:** `Exmc.GPUNode.Server` running locally, `tree.ex` dispatching through it instead of through direct NIF calls. Fair race numbers unchanged or marginally faster (the GenServer hop is ~1 µs, negligible vs the 1 ms fence wait).

### Phase 1 (weeks 2-3) — Synthesis prototype

W1 + W2: pick a substrate (likely template-based GLSL with text substitution to start, since the existing shaders are already templated by push constants). Implement Beta + Gamma chain shaders via synthesis. Validate.

**Deliverable:** `Exmc.GPUNode.Server.register_shader({:beta, alpha, beta_param})` synthesizes, compiles, validates, registers. Subsequent `dispatch/3` calls work. Fair race extended with Beta and Gamma cells.

### Phase 2 (week 4) — Warmup + caching

W4 + W5: characterize warmup, persist cache. The fair-race bench becomes the warmth benchmark.

**Deliverable:** Warmup curve doc per shader. `~/.exmc/gpu_node/` populated. Cold restart time < 3 s with cache vs ~30 s without.

### Phase 3 (week 5+) — Multi-client + protocol

W3 + W8: distributed Erlang transport, multi-tenant scheduling. Trial-mode workload (67 instruments) is the natural stress test.

**Deliverable:** GPU node serves multiple clients on same machine; per-client throughput measurements; fair scheduling under contention.

### Phase 4 — Bulkheads + production gates

W6: timeouts, fallback to EXLA, OOM handling. Driver crash recovery.

**Deliverable:** Chaos test: feed deliberately bad shaders, kill the GPU node mid-dispatch, exhaust VRAM. Clients should fall back to EXLA without crashing.

---

## Risks and unknowns

- **Driver instability under arbitrary shaders.** The NVIDIA Linux driver doesn't gracefully recover from bad submissions; expect to crash and restart the GPU node frequently in early phases. This is acceptable as long as clients fall back to EXLA.
- **GLSL → SPIR-V compile latency.** `glslc` is ~50-200 ms per shader. If we synthesize per-request, that's a noticeable warm-up delay. Mitigation: SPIR-V cache by `spec_hash`.
- **Cross-architecture validity.** SPIR-V cached on Linux NVIDIA may not run on FreeBSD GT 750M; pipeline cache is definitely device-specific. Cache layout: `{platform}/{device_uuid}/{spec_hash}.{spv,pipeline}`.
- **Memory pool fragmentation.** Long-lived process accumulates fragmented VRAM. Need periodic compaction or full pool reset. Trial run at 67 instruments × 8 hours will be the test.
- **The `zed` overlap.** GPU-node lifecycle (start, health-check, restart on crash, BE restart on driver crash) is exactly what `zed` will manage. **This research must land BEFORE `zed` is written** so the GPU node design informs `zed`'s container/process abstractions, not after.
- **Synthesis correctness.** Hand-written shaders had hand-derived gradients vetted by the authors. Synthesized shaders need an automated correctness gate strong enough to catch the same class of errors. The `proper_statem` accumulator pattern is the model.

---

## What this research produces

- A working `Exmc.GPUNode.Server` that holds the existing 6 chain shaders, plus 1-2 synthesized shaders (Beta and Gamma), demonstrated to converge on conjugate models.
- A characterization document: warmup curve per shader, eviction policy parameters, recovery semantics under bad inputs.
- A clear answer to "do we ship a long-lived GPU node?" — the prototype either demonstrates it or it doesn't, and the measurements tell us why.
- An informed input to the `zed` plan: what the GPU-node lifecycle and supervision tree need to look like at the deployment-tool level.

The shape mirrors what worked for H1-H5: **enumerate the questions, instrument the system, let the numbers force the architecture.** No design-docs-without-evidence; every decision backed by a measurement.

---

## Why this matters now (vs later)

Three forcing functions:

1. **The `zed` plan needs to know what it's deploying.** GPU node lifecycle is the most complex BEAM process `zed` will manage. Designing `zed`'s primitives without knowing the GPU node's needs risks designing the wrong primitives.

2. **The trial workload is real.** 67 instruments running periodic NUTS samplers, each compiling its own shaders, each holding its own pipeline cache. The current per-process model leaks ~25 MB VRAM per instrument's HLO cache (memory note #64). At trial-mode scale, a shared GPU node would eliminate the duplication entirely.

3. **Custom shader requests are the natural next API.** Today new distributions require manual shader work. Researchers want to try fat-tailed Cauchy mixtures, censored Gaussians, von Mises — and they want to do it in Elixir, not write GLSL. The GPU node turns "add a distribution" from a multi-day developer task into a runtime API call.

The research arc that closed the H1-H5 gap took roughly four hours and produced a 2.5× speedup on Linux. The GPU-node research arc is bigger (5+ weeks of phased prototypes), but the productivity payoff — *new distributions become a runtime API call* — is one of the few changes that genuinely unlocks new science instead of speeding up existing work.
