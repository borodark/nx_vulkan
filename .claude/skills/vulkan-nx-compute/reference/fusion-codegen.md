# JIT fusion codegen (thrust 3)

The runtime GLSL generator, distinct from the hand-written static kernels. Lives
in `lib/nx_vulkan/codegen.ex` (`Nx.Vulkan.Codegen`), driven by
`lib/nx_vulkan/compiler.ex` (`Nx.Vulkan.Compiler`, an `Nx.Defn.Compiler`).

Use it with `Nx.Defn.jit(&fun/2, compiler: Nx.Vulkan.Compiler).(a, b)`. A fusable
same-shape f32 elementwise chain becomes ONE generated shader + one dispatch,
replacing N per-op eager dispatches. Non-fusable graphs fall back to
`Nx.Defn.Evaluator` (always correct). Multi-stage graphs split at `dot`/`conv`
boundaries: each matmul/conv is a stage (reusing the eager `matmul`/`conv`
shaders), each maximal elementwise region is one generated shader whose leaf
inputs may be a prior stage's on-device buffer.

## What the codegen emits

- **`emit_region/2`** (and `emit_elementwise/1`) — a one-thread-per-element
  shader: `layout(local_size_x = 256)`, inputs at bindings `0..k-1`, output at
  `k`, push `{n}`. Dispatched by the `dispatch_generated` NIF (`n.div_ceil(256)`
  groups). Same std430 / inputs-first-output-last convention as static kernels.
- **`emit_fused_reduce/3`** — fuses an elementwise inner into a reduction as a
  **256-thread workgroup-per-slot shared-memory tree reduce** (grid-strided over
  slots to pass the 65535 workgroup-count limit). f64 accumulator for
  sum/product. `sum`/`product`/`reduce_max`/`reduce_min`; `mean` = sum with a
  baked `/n` post-scale. Dispatched by `dispatch_generated_reduce` (launch
  `outer*inner` workgroups, capped at 65535).
- **`emit_dag/2`** — linearises the Expr DAG into SSA `float tN = ...;` temps via
  a post-order id-deduped topo sort, so a fan-out node is computed ONCE. Naive
  inlining is exponential (8 chained squarings → 255 mults vs 8 with CSE).
- **`emit_loads/4` + `broadcast_index/3`** — NumPy-broadcast-aware parameter
  loads: only leaf parameter loads need broadcast handling (in a valid
  elementwise tree every node broadcasts to the root shape; Nx carries
  mismatched-shape operands directly, no `:broadcast` node). Shapes are
  compile-time constants baked into the GLSL index math.

## Compile + cache

`compile_cached/1` hashes the GLSL (`:erlang.phash2`), writes
`priv/shader_cache/gen_<hex>.spv` (gitignored) via
`System.cmd("glslangValidator", ["-V", comp_path, "-o", spv_path])`, and reuses
on hash hit — each generated kernel compiles exactly once.

## Correctness & tuning knobs

- Fusion output must match `Nx.BinaryBackend` to f32 eps (sum in f64, the
  `~r/\br\b/` operand-substitution rule — see gotchas.md).
- `reduce_beneficial?/3` gates the fused reduce: contiguous (`inner_stride == 1`),
  `slots <= 256`, `reduce_size >= 64` — the few-output regime where eager's
  serial `reduce_axis` is under-parallelised. The many-slot path wins on Kepler
  but regresses Ampere → auto-enabled only when `Nx.Vulkan.Device.weak?()`.
- `NXV_FUSE_DEBUG=1` logs FUSED/fallback per defn; `NXV_FUSE_REDUCE=1|0`
  forces/disables reduce fusion; `NXV_GPU_CLASS=weak|strong` overrides device
  class.

See `test/nx_vulkan/compiler_test.exs` and `examples/fused_jit_bench.exs`, and the
`thrust3-fusion-compiler` project memory for the full increment history.
