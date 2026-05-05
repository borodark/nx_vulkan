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

## Notes / log

(empty)
