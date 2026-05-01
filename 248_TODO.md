# mac-248 — Iter 4: fence reuse + Iter 5 stretch: pre-recorded command buffers

**Goal**: reduce per-dispatch overhead in spirit's `dispatch()` so
NUTS sampling under `:vulkan` becomes tractable. The 17 timeout
failures in v11 are all NUTS sampler tests where ~30 ops × ~1000
steps × N chains × ~280µs/dispatch = 8+ seconds per chain. Auto-fusion
has reduced dispatch *count* as much as a 2-input shader allows; the
remaining lever is per-dispatch *cost*.

PERSISTENT_BUFFERS_PLAN.md scopes the work; the parts that matter for
nx_vulkan are iter 4 (fence reuse) and a follow-up not in the current
plan: pre-recorded command buffers.

## Background — where the 280µs/dispatch goes

Measured on Linux RTX 3060 Ti (raw single-op `Nx.Vulkan.add`):

| Phase | Approx cost | Notes |
|---|---|---|
| BEAM → NIF crossing | 10 µs | Erlang resource decode |
| `vkCreateFence` | ~25 µs | per-submit, driver call |
| Command buffer allocate + begin | 30–50 µs | recording-only |
| `vkCmdBindPipeline` + descriptor write | 20 µs | bound for each dispatch today |
| `vkCmdDispatch` | µs | recording the actual workload is tiny |
| `vkQueueSubmit` + `vkWaitForFences` | 50–100 µs | the real GPU round-trip |
| `vkDestroyFence` | ~25 µs | matched to create |
| NIF → BEAM return | 10 µs | resource encode |

**Total ~280 µs of which ~150 µs is fence + recording overhead** —
both removable.

## Layout note

Mac-248 uses the flat layout: `~/spirit/`, `~/nx_vulkan/`. Paths below
assume that.

## Iter 4 — Fence reuse (highest ROI)

**Effort**: half a day. Direct ~50µs per dispatch savings. Reasonable
chance of a 1.2–1.4× speedup on the NUTS hot path.

### Strategy

One reusable fence per worker thread, stashed in `g_vk_ctx`. Created
at `vk_init`, destroyed at `vk_destroy`. Reset (`vkResetFences`)
between submits.

### Steps

```
cd ~/spirit
git pull
git checkout -b feat/fence-reuse
```

Edit `core/src/engine/Backend_par_vulkan.cpp`:

1. Add a thread-local fence to `g_vk_ctx`:

   ```cpp
   // In Backend_par_vulkan.hpp's Context struct:
   thread_local static VkFence reusable_fence = VK_NULL_HANDLE;
   ```

   Or simpler: a single `g_vk_ctx.reusable_fence` initialized at
   `vk_init`, since current spirit/nx_vulkan use one queue with a
   global SUBMIT_LOCK on the Rust side.

2. In `vk_init`, create the fence:

   ```cpp
   VkFenceCreateInfo fci{};
   fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
   vkCreateFence(g_vk_ctx.device, &fci, nullptr, &g_vk_ctx.reusable_fence);
   ```

3. In `vk_destroy`, destroy it.

4. In `dispatch()` (and the inlined dispatch dance in `nxv_matmul`,
   `nxv_transpose`, `nxv_apply_binary_broadcast`), replace the
   per-call fence create/destroy with reset + reuse:

   ```cpp
   // Old: VkFence fence; vkCreateFence(...); ...; vkDestroyFence(...)
   // New:
   vkResetFences(g_vk_ctx.device, 1, &g_vk_ctx.reusable_fence);
   vkQueueSubmit(g_vk_ctx.compute_queue, 1, &si, g_vk_ctx.reusable_fence);
   vkWaitForFences(g_vk_ctx.device, 1, &g_vk_ctx.reusable_fence, VK_TRUE, UINT64_MAX);
   ```

   Spirit's existing dispatch code handles SUBMIT_LOCK at the Rust
   level — no thread safety concern inside the C++ shim.

### Acceptance

- `bench_gpu_add` (dispatch-only at N=1K) drops by ≥30 µs.
- Existing 3-test correctness suite passes unchanged.
- nx_vulkan's `mix test` still 137/0 after merge + recompile.

### Push

```
git add core/include/engine/Backend_par_vulkan.hpp \
        core/src/engine/Backend_par_vulkan.cpp
git commit -m "Backend_par_vulkan: reuse a single VkFence (Iter 4)"
git push origin feat/fence-reuse
```

## Iter 5 (stretch) — Pre-recorded command buffers

**Effort**: 1–2 days. Cumulative 1.5–2× on the dispatch hot path
combined with iter 4. **Only do this if iter 4 numbers leave significant
overhead on the table.**

### Strategy

`vkCmdBindPipeline` + `vkCmdBindDescriptorSets` + `vkCmdDispatch` is
the same recording for every call to a given (pipeline, n) pair. Cache
the recorded command buffer keyed on `(pipe, push_value_hash)` and
re-submit instead of re-recording.

The catch: push constants are baked into the command buffer when
recorded. So we'd cache one command buffer per distinct push value
(typically `n` for elementwise — modest cardinality). Or pull the push
out of the recorded buffer and inject via `vkCmdPushConstants` at
submit time using a small primary command buffer that wraps the pre-
recorded secondary. The simpler approach is to cache by push value
and accept the cardinality.

### API addition

```cpp
// Records {bind pipe, bind desc, push, dispatch} into a fresh
// command buffer keyed on (pipe, push_data). Returns the buffer.
VkCommandBuffer record_dispatch(VkPipe* p, VkBuffer* bufs, int n_buffers,
                                uint32_t group_count_x,
                                uint32_t push_size, const void* push_data);

// Re-submit a previously recorded command buffer.
int replay_dispatch(VkCommandBuffer cmd);
```

The cache key is `(spv_path_pipe_id, push_data_bytes)`. For our typical
shaders (push = `n` in 4 bytes), cache cardinality is bounded by the
number of distinct shapes the workload sees. For NUTS at d=4..50 and
N=1..1000 elements, that's a few dozen entries.

### Acceptance

- Single-op cost at N=1K drops further by ≥50 µs after iter 4.
- nx_vulkan tests still pass.

### Push

Separate branch; this is the harder change.

## After your push

Linux side will:

1. Merge the iter 4 branch.
2. Rebuild + run the leapfrog bench (`mix run bench/leapfrog_bench.exs`)
   to confirm per-op cost drops.
3. Re-run the full exmc Vulkan suite (v12). Expected: timeouts drop
   from 17 to ~10 with iter 4 alone, possibly further with iter 5.
4. If timeouts cleared: declare Week 2 done at the structural level.
5. If still timing out: pick which path is the next bottleneck (most
   likely will be the 2-input shader limitation that prevents
   leapfrog bodies like `q + eps * p` from fusing).

## What this DOES NOT do

- **Doesn't help startup** — pipeline cache is already in place.
- **Doesn't reduce the BEAM↔NIF cost** — that's a Rustler/Erlang
  fundamental. ~20 µs/call floor.
- **Doesn't lift the 2-input shader constraint** — leapfrog bodies
  with 3+ unique input tensors still fall through to per-op dispatch.
  Lifting that needs a 4-input fused shader (separate work).
