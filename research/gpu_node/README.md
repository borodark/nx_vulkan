# GPU Node Research — feat/gpu-node

Workstream output directory. See `../../PLAN_GPU_NODE.md` for the master plan.

Each WORKSTREAM_*.md is a living document for one of the six parallel workstreams. Each starts as a stub with the question + protocol; gets updated as evidence lands.

| File | Workstream | Owner | Status |
|---|---|---|---|
| WORKSTREAM_W1_codegen.md | Shader synthesis substrate (3-prototype bake-off) | open | scaffold |
| WORKSTREAM_W2_validation.md | Statistical validation harness (lives in pymc/exmc/research/gpu_node) | open | scaffold |
| WORKSTREAM_W3_gpunode_server.md | GPUNode.Server GenServer (lives in pymc/exmc/research/gpu_node) | open | scaffold |
| WORKSTREAM_W4_warmup.md | Warmup curve characterization | open | scaffold |
| WORKSTREAM_W5_persistence.md | vkPipelineCache + SPIR-V CAS | open | scaffold |
| WORKSTREAM_W6_bulkheads.md | Watchdog + bad-shader recovery | open | blocked on W3 |

W2/W3 live under `pymc/exmc/research/gpu_node/` because they touch sampler/test code.
W1/W4/W5/W6 live here under `nx_vulkan/research/gpu_node/` because they touch the GPU layer.

## Branches

- `nx_vulkan@feat/gpu-node`
- `pymc@feat/gpu-node` (forked off `feat/dsl-shader-codegen`)

## Cross-references

- `nx_vulkan/PLAN_GPU_NODE.md` — full plan
- `nx_vulkan/248_TODO.md` — R3 ask to mac-248 (FreeBSD verification)
- `pymc/exmc/bench/fair_race_results_linux.md` — current Linux baseline
- `nx_vulkan/FAIR_RACE_FREEBSD.md` — FreeBSD baseline
