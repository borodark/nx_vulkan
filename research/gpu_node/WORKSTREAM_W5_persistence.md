# W5 — Pipeline Cache Persistence

**Question:** Can we eliminate cold-start cost by serializing `vkPipelineCache` to disk + caching SPIR-V by content hash?

**Budget:** cold-start time < 3 seconds with cache, vs ~30 seconds without.

## Two-layer design

```
~/.exmc/gpu_node/
├── spv/
│   ├── {spec_hash_1}.spv         # content-addressed compiled SPIR-V
│   ├── {spec_hash_2}.spv         # — portable across same-arch devices
│   └── ...
└── pipeline_cache/
    ├── {device_uuid_1}.bin       # opaque vkPipelineCache blob
    └── {device_uuid_2}.bin       # — device-specific
```

### Layer 1 — SPIR-V CAS

- `spec_hash = sha256(canonical_glsl_source || compiler_flags)`
- Persists the output of `glslc`. Re-usable across BEAM restarts and across machines of the same architecture (Linux x86_64 → Linux x86_64 is fine; cross-platform may not be).
- Cost: a sha256 + a fopen. Saves the 50-200 ms `glslc` call.

### Layer 2 — vkPipelineCache

- Each `vkCreatePipelineCache` call can take a `pInitialData` blob from a previous `vkGetPipelineCacheData` call.
- Driver-specific: NVIDIA stores compiled GPU machine code keyed by SPIR-V hash. Cache hit = no compile, just reuse.
- Cost: ~10 MB blob per device. Read on startup, write on shutdown (or periodically).
- Risk: driver may invalidate cache on driver upgrade, GPU swap, OS upgrade. Always-validate-then-fallback.

## Protocol

1. Spike: at GPU node startup, read `~/.exmc/gpu_node/pipeline_cache/{uuid}.bin` if present, pass to `vkCreatePipelineCache(pInitialData)`.
2. At shutdown (or per-N-shader-loads), call `vkGetPipelineCacheData`, write blob to disk.
3. Measure: cold-start wall (`mix run -e "Exmc.GPUNode.Server.start_link(); :ok"` time-to-ready) with and without the cache.
4. Stress-test: corrupt the cache file by 1 byte, verify graceful fallback (driver detects corruption, returns VK_ERROR_INITIALIZATION_FAILED, we discard and re-compile).

## Open questions

- Q5.a — Does NVIDIA Linux support `VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT`? If yes, multi-threaded shader load is safe.
- Q5.b — What's the actual cache hit rate during a normal trial-mode session? Trace `vkCreateComputePipelines` calls and log cache-hit vs cache-miss. (Vulkan extension `VK_EXT_pipeline_creation_cache_control` exposes this.)
- Q5.c — Cross-machine cache sharing? Trial-mode could pre-warm caches on the build server and ship them to deployment targets. Probably out of scope for Phase 2.

## Output

- `cache_design.md` — final design including file layout, atomicity (write-temp-then-rename), corruption handling.
- Spike code in `nx_vulkan/c_src/nx_vulkan_shim.cpp` (extend pipeline cache to accept external init data).

---

## vkPipelineCache mechanics

### What `vkPipelineCache` actually is

A driver-private, opaque blob keyed by (driver build, physical device, SPIR-V module). Holds the result of the back-end compile from SPIR-V → device ISA. On a cache hit, `vkCreateComputePipelines` skips the back-end compile entirely — the only cost left is descriptor / layout / pipeline-object construction.

There is no API to inspect or merge content other than `vkMergePipelineCaches`. Treat it as a pickle.

### Header format

The Vulkan spec mandates a fixed prefix on every blob produced by `vkGetPipelineCacheData`:

| Offset | Size | Field |
|--------|------|-------|
| 0      | 4    | `headerSize` (uint32, little-endian; usually 32 for V1) |
| 4      | 4    | `headerVersion` (uint32; `VK_PIPELINE_CACHE_HEADER_VERSION_ONE = 1`) |
| 8      | 4    | `vendorID` (uint32; e.g. NVIDIA = 0x10DE) |
| 12     | 4    | `deviceID` (uint32; e.g. RTX 3060 Ti = 0x2486) |
| 16     | 16   | `pipelineCacheUUID` (16 raw bytes from `VkPhysicalDeviceProperties.pipelineCacheUUID`) |

Everything after `headerSize` bytes is opaque.

When the application passes `pInitialData` into `vkCreatePipelineCache`, the implementation **must** validate every field; on mismatch it ignores the data and returns a fresh empty cache. **No error is returned in that case** — it is silent. So we cannot distinguish "no init data" from "init data was rejected" via the create call alone. We can sniff the header ourselves before calling, or compare the on-disk header against `VkPhysicalDeviceProperties` to detect rejection up front.

### Cache invalidation rules

The `pipelineCacheUUID` changes (and therefore invalidates persisted blobs) on:

- Driver upgrade — NVIDIA, AMD, Mesa all bump the UUID per driver build.
- GPU model change (`vendorID`/`deviceID` changes too).
- Sometimes on OS / kernel upgrade if the driver's user-mode component bumps.

It does NOT change on:

- BEAM restart.
- Machine reboot (assuming same driver, same GPU).
- vkInstance recreation within the same process.

So the right strategy is: **trust the header**. If `pipelineCacheUUID` matches `VkPhysicalDeviceProperties.pipelineCacheUUID`, load. Else delete the file and start fresh.

### `VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT`

Promoted to core in Vulkan 1.3 (was `VK_EXT_pipeline_creation_cache_control` 0x00000001 in 1.2). Without it the implementation internally serializes any concurrent `vkCreateComputePipelines` calls that share a `VkPipelineCache` (a small mutex inside the driver). With it, the application promises external sync and the driver skips the lock.

**Support landscape (early 2026):**

- NVIDIA proprietary 535+ on Linux: yes.
- Mesa RADV / ANV (Intel): yes since Mesa 22.x.
- MoltenVK on macOS: yes.
- Older NVIDIA (<460) and old AMDGPU-PRO: partial — the bit is silently ignored, fallback is internal sync. Safe to set unconditionally.

For our use we don't actually need it: every `get_or_create_pipe` call is already serialized via the same `g_pipe_cache` lookup we hold a mutex around (or will hold). Setting the bit lets us drop that mutex if we ever want concurrent shader loads. Keep it as a low-priority optimization, not part of the v1 spike.

### `VK_EXT_pipeline_creation_cache_control` — telemetry

This extension (also core in 1.3) chains a `VkPipelineCreationFeedbackCreateInfo` onto every `vkCreateComputePipelines` call. After the call, the driver fills in:

- `VkPipelineCreationFeedback.flags`:
  - `VK_PIPELINE_CREATION_FEEDBACK_VALID_BIT` — feedback is meaningful
  - `VK_PIPELINE_CREATION_FEEDBACK_APPLICATION_PIPELINE_CACHE_HIT_BIT` — **the hit/miss bit we want**
  - `VK_PIPELINE_CREATION_FEEDBACK_BASE_PIPELINE_ACCELERATION_BIT` — irrelevant for compute
- `VkPipelineCreationFeedback.duration` — nanoseconds spent in pipeline creation.

This is the right signal for Q5.b. Plumb it into our shim, log per-pipeline (path, hit, duration_ns) at INFO. That gives us the hit-rate diagnostic for free, without timing the wrapping NIF call.

Caveat: the extension also adds `VK_PIPELINE_CREATE_FAIL_ON_PIPELINE_COMPILE_REQUIRED_BIT`, which lets you query a cache without compiling on miss. Not useful here — we always want to compile on miss.

### Cross-process and cross-machine notes

- **Cross-process on the same machine**: every BEAM process gets its own `VkInstance` / `VkDevice`. The cache file on disk is shared, but only one process writes it. With our single-writer model below, this is safe.
- **Cross-machine, same arch + same driver build**: legal but UUID will differ between machines (the UUID has a machine-stable component on some drivers). Our deployment story is "ship the SPV CAS, let each box build its own pipeline cache on first run." The pipeline cache file is a runtime convenience, not a deployment artifact.

### The rough patches

Three sharp edges worth flagging early:

1. **NVIDIA Linux silently rejects caches written by a process running under a different `LD_LIBRARY_PATH` than the writer.** The driver UUID embeds a hash of certain driver components; if you shimmed a different libGLX or libnvidia-glcore at startup, you get a different UUID even on the same machine. Fix: pin the driver path in the deploy.
2. **`vkGetPipelineCacheData` is not free.** It serializes the entire driver-side cache (~5-15 MB on our workload, more if many shader variants). Calling it after every shader registration would dominate. Per-N-shader-loads or shutdown-only.
3. **Mesa swrast (the FreeBSD GT 750M fallback path) returns a 32-byte all-zero header from `vkGetPipelineCacheData` if the cache is empty.** Validation: check returned size > headerSize, not just != 0.

---

## Survey: similar-domain projects

### DXVK / VKD3D-Proton (Steam Vulkan layer for D3D9-12 games)

- Maintains its own state-tracking cache **separately** from the Vulkan pipeline cache: `*.dxvk-cache` files, on disk, keyed per game.
- Pre-warms the cache by running the game once and recording every pipeline state ever observed; subsequent launches replay them in a worker thread at startup so the actual gameplay frame doesn't stall on `vkCreateGraphicsPipelines`.
- Steam ships a "Fossilize" layer that records pipeline creation across the install base, normalizes them, and ships pre-built caches to every user. **This is the closest model to our "trial-mode pre-warm on the build server" idea (Q5.c).**
- Lesson for us: separate the **shader/pipeline state** cache (replayable, portable) from the **driver bytecode** cache (opaque, device-specific). Our SPV CAS = state cache; vkPipelineCache = bytecode cache. Same two-layer split they evolved into.

### wgpu (Rust gfx-rs)

- `wgpu` exposes `Device::create_pipeline_cache` as of 0.20. Takes optional initial bytes, returns a handle that can be passed to `compute_pipeline_descriptor.cache`.
- `pipeline_cache.get_data()` returns the opaque blob to persist.
- Their docs explicitly call out: "the application is responsible for invalidating stale data; the validator only checks the header."
- Lesson: their API surface is exactly what we should expose to Elixir. Three calls: `cache_load(path) -> handle`, `cache_save(handle, path)`, `dispatch_with_cache(handle, ...)`.

### Mesa shader cache (`MESA_SHADER_CACHE_DIR`)

- Different beast. This is a **GLSL → NIR → backend-IR cache** that lives below the Vulkan API. `~/.cache/mesa_shader_cache/`. Driver-managed, not application-visible.
- Operates orthogonally to `vkPipelineCache`: even with our cache empty, Mesa hits its own NIR cache on second run.
- We don't interact with it. But: if a user reports "cold start is fast even without our cache," Mesa's shader cache is likely why. Document this so we don't get confused by misleading numbers on AMD/Intel.

### Stan / cmdstanpy (parallel-domain analog)

- Compiles `.stan` → C++ → executable, caches the compiled binary in `~/.cmdstan/<model>/<hash>`.
- Hash key: source text + compiler flags + cmdstan version.
- Single-writer (the compile step), multi-reader (parallel sampling).
- Atomic write: temp file + `os.rename`.
- Lesson: this is the same shape as our SPV CAS. The directory layout is well-trodden, no surprises expected.

---

## Concrete design for the exmc cache

### File layout

```
~/.exmc/gpu_node/
├── spv/
│   ├── {sha256_64bit_prefix}.spv       # 16 hex chars; full hash in side-file
│   └── {sha256}.meta                   # JSON: source text, compile flags, glslc version, ts
├── pipeline_cache/
│   ├── {device_uuid_hex}.bin           # opaque vkPipelineCache blob
│   └── {device_uuid_hex}.meta          # JSON: vendor, device, driver_version, ts, blob_bytes
└── lock                                 # advisory flock for single-writer; see below
```

Resolved via `Application.get_env(:exmc, :gpu_node_cache_dir)`, falling back to `${HOME}/.exmc/gpu_node`. Override per-host via env var `EXMC_GPU_NODE_CACHE_DIR` (matches the trial-mode override pattern from `trial/start_trial.sh`).

The 16-hex-prefix on SPV filenames keeps `ls` readable; the chance of a collision in a 67-instrument trial is ~1 in 2^32 even at 10^4 distinct shaders. The `.meta` side-file holds the full hash for hard verification.

`device_uuid_hex` is 32 hex chars, derived from `VkPhysicalDeviceProperties.pipelineCacheUUID` (16 raw bytes). Allows multiple GPUs on one box to coexist (already a real case: super-io has the RTX 3060 Ti + onboard Intel).

### Atomicity strategy

Standard temp-rename:

```
write tmp_path = path + ".tmp." + os.getpid() + "." + monotonic_ns
fsync(tmp_fd)
rename(tmp_path, final_path)
```

`rename(2)` on ext4 / xfs is atomic on the same filesystem (POSIX guarantee). The `.tmp.PID.NS` suffix means a crashed concurrent writer can't collide. Stale `.tmp.*` files are garbage-collected on next start (older than 10 min → unlink).

We do **not** fsync the parent directory after rename — we accept that an OS crash within ~30 seconds of write may revert. Cache rebuild is cheap relative to the cost of a fsync per write.

For the SPV files, write-once-read-many: once written, never modified. So no race after the rename completes.

### Corruption handling

Three layers:

1. **Header sniff before passing to driver.** Read first 32 bytes, compare `pipelineCacheUUID` field against `VkPhysicalDeviceProperties.pipelineCacheUUID`. On mismatch: delete the file, log a one-line WARN (`pipeline cache UUID mismatch — driver upgraded? rebuilding`), proceed with empty cache.
2. **Trust the driver.** If the header sniff passes but the body is corrupt, `vkCreatePipelineCache` returns either `VK_SUCCESS` with a silently-empty cache, or `VK_ERROR_INITIALIZATION_FAILED`. We treat both the same: try again with `pInitialData = NULL`, log a WARN, delete the file.
3. **SPV files are validated by `vkCreateShaderModule`.** A truncated or mangled SPV gets `VK_ERROR_INVALID_SHADER_NV` or similar. On any non-`VK_SUCCESS`, delete the file and re-invoke `glslc` to recompile from the canonical source (the canonical source must be available — recorded in the `.meta` side-file or recoverable from the codegen IR).

### Concurrency model

Single-writer, multi-reader, advisory file-lock:

- Each GPU node process tries `flock(LOCK_EX | LOCK_NB)` on `~/.exmc/gpu_node/lock` at startup.
  - Success → this process is the canonical writer for the duration of its lifetime.
  - Failure → another GPU node is running. Operate read-only against the cache. Skip writebacks at shutdown.
- Reads are unlocked. The `rename`-based write atomicity means a reader either sees the old or new file, never a half-written one.

Multi-process write contention is rare in our workload — trial-mode runs one GPU node per box. The lock is an insurance policy, not an everyday code path.

If we ever do want true multi-writer (e.g. two BEAM nodes sharing a GPU on the same box), the answer is `vkMergePipelineCaches`: load each writer's blob into a separate `VkPipelineCache`, merge into a fresh one, persist the merged result. Out of scope for v1.

### Eviction

For the SPV CAS:

- Soft cap: 500 MB. On startup, if `du -s ~/.exmc/gpu_node/spv` exceeds the cap, walk by atime, delete oldest until under.
- Atime-driven LRU; relies on filesystem `relatime` (default on Linux).
- No mid-run eviction; the working set per BEAM lifetime is small enough to ignore (≤100 shaders for the foreseeable workload).

For the pipeline cache blob:

- No eviction. The blob is a single file per device. Driver-managed internal LRU.
- On restart, if `vkGetPipelineCacheData` returns > 100 MB (driver pathological), log it, persist it anyway. The driver chose that size; trust it.

### Cache invalidation triggers

Trust the driver via the header check (above). Do NOT proactively invalidate on:

- BEAM restart — desired behavior is to use the warm cache.
- Source code changes to non-shader code — no impact on shaders.

Do invalidate on:

- `pipelineCacheUUID` mismatch — automatic, see corruption handling.
- Manual: `Exmc.GPUNode.Cache.purge!/0` for testing.

We do NOT track:

- Driver version string (the UUID covers it).
- glslc version (the SPV CAS hash includes it via compile flags).
- OS version (driver UUID covers any ABI-relevant change).

---

## C++ extension sketch (Backend_par_vulkan + nx_vulkan_shim)

### State additions to `Engine::Backend::vulkan::VkContext`

```cpp
struct VkContext {
    /* ... existing ... */
    VkPipelineCache pipeline_cache = VK_NULL_HANDLE;  /* shared by all create_pipeline calls */
    bool pipeline_cache_dirty       = false;          /* set after each successful pipeline create */
    std::string pipeline_cache_path;                  /* resolved at vk_init via env */
};
```

### Init: load cache file, pass into `vkCreatePipelineCache`

```cpp
/* Backend_par_vulkan.cpp, end of vk_init(), after device + queues exist */

static int load_pipeline_cache(VkContext& ctx) {
    /* resolve path: $EXMC_GPU_NODE_CACHE_DIR or ~/.exmc/gpu_node */
    std::string dir = resolve_cache_dir();  /* mkdir -p as needed */
    char uuid_hex[33] = {0};
    for (int i = 0; i < 16; i++)
        snprintf(uuid_hex + 2*i, 3, "%02x",
                 ctx.device_props.pipelineCacheUUID[i]);
    ctx.pipeline_cache_path = dir + "/pipeline_cache/" + uuid_hex + ".bin";

    /* read file if present; sniff header */
    std::vector<uint8_t> blob;
    if (read_file_if_exists(ctx.pipeline_cache_path, blob) == 0) {
        if (!header_matches_device(blob, ctx.device_props)) {
            unlink(ctx.pipeline_cache_path.c_str());
            blob.clear();
            fprintf(stderr, "spirit-vulkan: pipeline cache UUID mismatch, rebuilding\n");
        }
    }

    VkPipelineCacheCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    ci.initialDataSize = blob.size();
    ci.pInitialData    = blob.empty() ? nullptr : blob.data();

    VkResult r = vkCreatePipelineCache(ctx.device, &ci, nullptr, &ctx.pipeline_cache);
    if (r != VK_SUCCESS) {
        /* retry empty */
        ci.initialDataSize = 0;
        ci.pInitialData    = nullptr;
        unlink(ctx.pipeline_cache_path.c_str());
        return vkCreatePipelineCache(ctx.device, &ci, nullptr, &ctx.pipeline_cache)
               == VK_SUCCESS ? 0 : -1;
    }
    return 0;
}
```

### Create: pass `ctx.pipeline_cache` (currently `VK_NULL_HANDLE`)

In `create_pipeline`, replace the existing call:

```cpp
/* was: */
VK_CHECK(vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1,
         &pipe_ci, nullptr, &p->pipeline), "compute pipeline");

/* becomes: */
VkPipelineCreationFeedback fb{};
VkPipelineCreationFeedbackCreateInfo fb_ci{};
fb_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATION_FEEDBACK_CREATE_INFO;
fb_ci.pPipelineCreationFeedback = &fb;
fb_ci.pipelineStageCreationFeedbackCount = 0;  /* compute = single stage, can omit per-stage */
pipe_ci.pNext = &fb_ci;  /* only if VK_EXT_pipeline_creation_cache_control enabled */

VK_CHECK(vkCreateComputePipelines(ctx.device, ctx.pipeline_cache, 1,
         &pipe_ci, nullptr, &p->pipeline), "compute pipeline");

ctx.pipeline_cache_dirty = true;
log_pipeline_creation(spv_path, fb);  /* hit/miss + ns */
```

### Save: shutdown + periodic

```cpp
/* In vk_destroy(), before vkDestroyDevice */
if (ctx.pipeline_cache != VK_NULL_HANDLE && ctx.pipeline_cache_dirty) {
    save_pipeline_cache(ctx);
    vkDestroyPipelineCache(ctx.device, ctx.pipeline_cache, nullptr);
    ctx.pipeline_cache = VK_NULL_HANDLE;
}

static int save_pipeline_cache(VkContext& ctx) {
    size_t sz = 0;
    vkGetPipelineCacheData(ctx.device, ctx.pipeline_cache, &sz, nullptr);
    if (sz == 0) return 0;
    std::vector<uint8_t> blob(sz);
    if (vkGetPipelineCacheData(ctx.device, ctx.pipeline_cache, &sz, blob.data())
        != VK_SUCCESS) return -1;
    /* write tmp + rename */
    std::string tmp = ctx.pipeline_cache_path + ".tmp." +
                      std::to_string(getpid()) + "." +
                      std::to_string(monotonic_ns());
    if (write_file_atomic(tmp, blob) != 0) return -1;
    if (rename(tmp.c_str(), ctx.pipeline_cache_path.c_str()) != 0) {
        unlink(tmp.c_str());
        return -1;
    }
    ctx.pipeline_cache_dirty = false;
    return 0;
}
```

Periodic save (e.g. every N=64 pipeline creations): driven from the shim, not the backend, so we can tune from Elixir without a recompile.

### Shim functions (extern "C" surface to Rust)

Additions to `nx_vulkan_shim.h`:

```c
/* Pipeline cache lifecycle ------------------------------------------- */

/* Force the in-memory vkPipelineCache to be flushed to its file. Call
 * periodically (e.g. every N shader registrations) or at clean shutdown.
 * Returns 0 on success, -1 on I/O error, -2 if no cache active. */
int nxv_pipeline_cache_save(void);

/* Stats: how many pipelines have been created, hit/miss split. */
void nxv_pipeline_cache_stats(unsigned long* total_creates,
                              unsigned long* cache_hits,
                              unsigned long* cache_miss_compile_ns_total,
                              unsigned long* blob_bytes_on_disk);

/* Manually purge: deletes the cache file from disk. Next nxv_init starts
 * cold. Used for testing. Returns 0 on success or if no file. */
int nxv_pipeline_cache_purge(void);

/* SPV CAS — content-addressed SPIR-V store --------------------------- */

/* Load SPV bytes for a given hash. Returns NULL if not in cache.
 * Caller must call nxv_spv_release(handle) to free.
 * out_bytes / out_len are filled when found. */
void* nxv_spv_load(const char* hash_hex, const unsigned char** out_bytes,
                   unsigned long* out_len);
void  nxv_spv_release(void* handle);

/* Store SPV bytes under a hash. Atomic write-rename. Returns 0 on success. */
int   nxv_spv_store(const char* hash_hex, const unsigned char* bytes,
                    unsigned long len);

/* List hashes currently present in the SPV CAS, comma-separated into
 * out_buf (caller-provided, max out_buf_len bytes). Returns count. */
unsigned long nxv_spv_list(char* out_buf, unsigned long out_buf_len);
```

The Rust NIF then exposes these to Elixir as:

- `Nx.Vulkan.Native.pipeline_cache_save() :: :ok | {:error, term}`
- `Nx.Vulkan.Native.pipeline_cache_stats() :: %{creates: integer, hits: integer, ...}`
- `Nx.Vulkan.Native.spv_load(hash_hex) :: {:ok, binary} | :miss`
- `Nx.Vulkan.Native.spv_store(hash_hex, binary) :: :ok | {:error, term}`

Elixir-side codegen (W1) calls `spv_load` first, falls through to `glslc + spv_store` on miss. Pipeline creation is automatic via the existing dispatch path; only the save/stats/purge surface needs explicit calls.

### What to leave for v2

- `VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT` — not needed until parallel shader registration arrives.
- Cross-process `vkMergePipelineCaches` — single-writer is enough.
- `Fossilize`-style replay-from-recording for true cold-start warm-up — only relevant if first-build cost ever becomes user-visible.

## Notes / log

(empty)
