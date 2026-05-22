# Plan: proper context lifecycle in `nx_vulkan_vulkano`

The exmc-side tier-3 BEAM split
(`exmc/docs/TEST_INFRA_TIER_3_PLAN.md`) is a workaround for a real
problem: the vulkano-side NIF accumulates GPU state — buffer
allocator high-water marks, descriptor-set pool exhaustion,
pipeline cache growth — that contaminates long-running test
sessions. This document plans the *cure*: explicit context
lifecycle in `nx_vulkan_vulkano`.

When this lands, the BEAM split becomes optional (still useful for
parallelism reasons; no longer required for correctness).

## What "context" means here

In `native/nx_vulkan_vulkano/src/lib.rs:ctx()`, the `Context`
struct holds:

```rust
pub struct Context {
    pub device:        Arc<Device>,
    pub queue:         Arc<Queue>,
    pub mem_allocator: Arc<StandardMemoryAllocator>,
    pub cmd_allocator: Arc<StandardCommandBufferAllocator>,
    pub set_allocator: Arc<StandardDescriptorSetAllocator>,
    // ... plus a process-wide PIPELINE_CACHE: OnceLock<Mutex<HashMap<...>>>
}
```

The whole thing is created once (`OnceLock<Context>`) and lives for
the lifetime of the BEAM process. The four allocators grow
monotonically:

- `mem_allocator`: every `upload_buffer` / `alloc_buffer` reserves
  device memory. vulkano's `StandardMemoryAllocator` recycles inside
  a single allocation block but doesn't shrink between dispatches.
- `cmd_allocator`: command buffer storage, one per dispatch. Frees
  on Drop, but the pool that backed them stays.
- `set_allocator`: descriptor sets. Capped at `set_count=32` per
  pool per `PipelineLayout` identity. When the cap is hit, a new
  pool is allocated. Pools are never released.
- `PIPELINE_CACHE`: every unique `(spv_path, op_code)` produces one
  `(PipelineLayout, ComputePipeline)`. Grows with the number of
  shaders dispatched.

Over ~700 tests' worth of churn, each of these reaches a state
where the next test's behavior depends on the leftover. That's the
state pollution.

## Two approaches

### Approach A — `reset_context/0` NIF (~3 days)

Single NIF that destroys and rebuilds the entire `Context`. Equivalent
to "BEAM restart but for vulkano only."

```elixir
# Elixir side:
:ok = Nx.Vulkan.NativeV.reset_context()

# Test pattern:
setup do
  on_exit(fn -> Nx.Vulkan.NativeV.reset_context() end)
end
```

Rust side:

```rust
#[rustler::nif]
fn reset_context(env: Env) -> NifResult<Atom> {
    // 1. Lock the pipeline cache, drop it (drops Pipeline +
    //    PipelineLayout Arcs).
    if let Some(cache) = pipeline_cache().lock().ok() {
        cache.clear();
    }

    // 2. Replace CTX. The old Context's Arc<Device>, Arc<Queue>,
    //    Arc<*Allocator> all drop, which in turn drop the
    //    underlying Vulkan resources via vulkano's Drop chain.
    //
    // The challenge: CTX is a OnceLock. OnceLock has no `take()`
    // in stable Rust. Need to either switch to OnceCell + Mutex,
    // or to ArcSwap, or to a pure Mutex<Option<Arc<Context>>>.
    //
    // ArcSwap is the cleanest: lock-free reads, atomic swap.
    //
    let new_ctx = build_context()?;
    CTX.store(Arc::new(new_ctx));

    Ok(atoms::ok())
}
```

**Pros**
- Single-shot reset; semantics obvious to callers.
- Test code can opt in via `on_exit` without thinking about which
  resources to reset.

**Cons**
- Have to refactor `CTX` from `OnceLock` to `ArcSwap` (or
  equivalent). Every read site (`ctx()`) becomes a `.load()` call;
  acceptable but touches every NIF entry point.
- Reset takes ~50-100ms (device teardown + reinit). Per-test cost
  if used heavily.
- Concurrent test workers calling `reset_context` mid-dispatch
  hit a race: dispatch holds an `Arc<Context>` snapshot; reset
  replaces the global; old context's Drop runs after dispatch
  finishes. ArcSwap handles this correctly via reference counting,
  but worth verifying with a stress test.

### Approach B — fine-grained recycling (~1 week)

Don't tear down the whole context; recycle the parts that grow
monotonically.

**B.1: switch `set_allocator` to a per-call pool.** Currently
`StandardDescriptorSetAllocator` keeps a per-`PipelineLayout` pool
of 32 sets that's never released. Replacing it with a pool that
drains its pending allocations between dispatches caps the high-water
mark. vulkano's `SubbufferAllocator` does this for memory; the
equivalent for descriptor sets needs custom plumbing.

**B.2: bound the pipeline cache.** Currently the HashMap grows
unboundedly. Add an LRU bound (e.g. 128 entries) that evicts
least-recently-used pipelines. Memory savings are real on long
sessions where many distinct shaders are tried (synth-shader
testing in particular generates ~10-20 SPV variants per session).

**B.3: SubbufferAllocator integration.** The current code uses
`upload_buffer` / `alloc_buffer` for each dispatch. Switch to
vulkano's `SubbufferAllocator` for transient buffers — it keeps
a freelist of recently-released chunks and reuses them. Permanent
buffers (the `q_buf`, `p_buf` used across leapfrog chains) stay
on direct allocation.

**Pros**
- No global tear-down; lock-free reads stay lock-free.
- Per-dispatch cost stays where it is (no 50ms reset penalty).
- Memory profile improves for long-running production traders too,
  not just tests.

**Cons**
- Three sub-projects, each smaller than approach A but more
  individually scoped. Easier to ship incrementally; longer
  total wall time.
- Doesn't address pipeline cache pollution from `op_code`-keyed
  entries (specialization constants can map distinct workloads
  to distinct pipeline objects).
- More invasive on the dispatch hot path. Higher chance of
  micro-regressions.

## Recommended path

**Ship A first; consider B for performance** later.

Reasons:
1. A is bounded in scope (~3 days). It solves the test-pollution
   problem outright. The BEAM split workaround can stay as
   defense-in-depth.
2. B is genuinely good engineering but most of the benefit accrues
   to long-running production traders (mac-247's eXMC trial), not
   to test correctness. Sequence it after A unless the trader
   shows symptoms first.
3. A's main cost (the `ArcSwap` refactor) is also a prerequisite
   for B — both need the ability to atomically replace allocators
   under live readers. Doing A first paves the way.

## Concrete plan for approach A

### Step 1 — refactor `CTX` to `ArcSwap` (~half a day)

```rust
use arc_swap::ArcSwap;

static CTX: OnceLock<ArcSwap<Context>> = OnceLock::new();

pub fn ctx() -> Result<Arc<Context>, String> {
    let swap = CTX.get_or_try_init(|| -> Result<_, String> {
        Ok(ArcSwap::new(Arc::new(build_context()?)))
    })?;
    Ok(swap.load_full())
}
```

Every `ctx()?` callsite still returns `Arc<Context>` — they're
just one extra atomic load deeper. Add `arc-swap = "1.7"` to
`Cargo.toml`.

### Step 2 — extract `build_context` (~1 hour)

Pull the existing initialization out of `ctx()` into a free
function `build_context() -> Result<Context, String>` so it can
be called both at first init and on reset.

### Step 3 — `reset_context` NIF (~half a day)

```rust
#[rustler::nif(schedule = "DirtyIo")]
fn reset_context() -> NifResult<rustler::Atom> {
    // Clear the pipeline cache first (drops pipeline Arcs, which
    // hold device references).
    if let Ok(mut cache) = pipeline_cache().lock() {
        cache.clear();
    }
    // Swap in a fresh context.
    let new_ctx = build_context()
        .map_err(|e| rustler::Error::Term(Box::new(e)))?;
    let swap = CTX.get().expect("ctx() never called");
    swap.store(Arc::new(new_ctx));
    Ok(atoms::ok())
}
```

Wire to Elixir:

```elixir
def reset_context, do: :erlang.nif_error(:nif_not_loaded)
```

Dirty IO scheduler because device teardown can block for tens of
milliseconds; we don't want to stall the BEAM normal scheduler.

### Step 4 — concurrent-dispatch race test (~1 day)

Stress test that:
1. Spawns 8 worker processes each looping `apply_binary` on a
   shared input.
2. From a 9th process, calls `reset_context` every 100ms.
3. Asserts no panics or InvalidDevice errors from the workers
   for 60 seconds.

`ArcSwap::store` is wait-free but the *Vulkan side* — i.e., the
device referenced by an old context — needs to survive any
in-flight dispatch. Since dispatches hold their `Arc<Context>`
via `ctx()?`, they keep their device alive. The reset's new
device is separate. No collision; verifying with a stress test
just pins the contract.

### Step 5 — expose to Elixir as a behaviour-level reset (~1 hour)

`Nx.Vulkan.VulkanoBackend.reset/0` calls
`Nx.Vulkan.NativeV.reset_context/0`. Then exmc's `on_exit` hooks
in :gpu_state-tagged modules can opt in:

```elixir
defmodule Exmc.NUTSTest do
  use ExUnit.Case
  @moduletag :gpu_state

  setup do
    on_exit(fn -> Nx.Vulkan.VulkanoBackend.reset() end)
  end
  # ...
end
```

### Step 6 — verify against the failing batch (~1 hour)

Re-run the exmc full sweep (`mix test.all`). Acceptance: no
failures from state pollution after 3 consecutive runs.

## Out of scope but worth noting

- **EXLA-side equivalents.** EXLA has its own client / device /
  buffer cache lifecycle that could pollute similarly. Not in
  scope for nx_vulkan but the same pattern could apply.

- **Spirit-backend lifecycle.** `Nx.Vulkan.Backend` (C++ spirit
  path) has its own state machine that has caused the
  Mission-II-era crashes. Less relevant now that vulkano is the
  preferred path, but if spirit stays as a fallback, the same
  reset capability is welcome.

- **Per-test BEAM (alternative).** Some Elixir projects spawn a
  fresh BEAM per test file via a custom Mix task. Most reliable
  isolation, ~6× wall-time penalty. Mentioned for completeness;
  approach A is enough.

## Effort estimate

| Step | Effort |
|------|--------|
| ArcSwap refactor | 4h |
| extract build_context | 1h |
| reset_context NIF | 4h |
| stress test | 1d |
| Elixir behaviour-level reset | 1h |
| verification + doc | 1h |

Total: **~3 days focused**. Half a week elapsed in practice.

## When this becomes the priority

Defer until:
- The eXMC trial on mac-247 shows symptoms attributable to
  vulkano memory growth (descriptor pool exhaustion, "Invalid
  device" errors, OOM); or
- The tier-3 BEAM split proves insufficient (e.g. the GPU suite
  itself becomes too flaky to use as CI gate); or
- A long-running model (multi-day Bayesian inference) hits
  resource limits during a single sample.

Until then, the BEAM split holds, and this plan stays as the
known cure.
