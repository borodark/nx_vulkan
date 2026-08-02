---
name: vulkan-nx-compute
description: Author or extend a GLSL Vulkan compute kernel wired into Elixir Nx through the Rust/Vulkano NIF in this repo (nx_vulkan). Use when adding a new GPU op, a fused JIT shader, or a matmul/conv/reduction kernel, or when debugging numerical parity or dispatch geometry for one.
---

# Vulkan compute kernels for Nx (this repo's playbook)

How numerical compute actually flows in `nx_vulkan`:

```
Nx op → Nx.Vulkan.VulkanoBackend (lib/nx_vulkan/vulkano_backend.ex)
      → Nx.Vulkan.NativeV stub    (lib/nx_vulkan/native_v.ex)
      → Rustler NIF               (native/nx_vulkan_vulkano/src/lib.rs)
      → vulkano dispatch of a .spv compiled from glsl/*.comp
```

Two kinds of shaders exist and follow different rules:

- **Static kernels** — hand-written `glsl/*.comp`, compiled ahead of time to
  `priv/shaders/*.spv`, dispatched by a dedicated NIF (`matmul`, `reduce_axis`,
  `conv_im2col`, `apply_binary`, …). Used by the eager backend.
- **JIT-fused kernels** — GLSL generated at runtime from an `Nx.Defn.Expr` tree
  by `lib/nx_vulkan/codegen.ex` (`Nx.Vulkan.Codegen`), compiled on demand to
  `priv/shader_cache/gen_<hash>.spv`, dispatched by the generic
  `dispatch_generated` / `dispatch_generated_reduce` NIFs. This is the thrust-3
  fusion compiler (`Nx.Vulkan.Compiler`). See `reference/fusion-codegen.md`.

Read `reference/recipe.md` for the full add-a-static-op checklist and
`reference/gotchas.md` before trusting any perf or correctness claim.

## 1. Shader vs host fallback: when to write a kernel at all

Every backend op has the same shape: try the GPU, else `Nx.backend_transfer` to
`Nx.BinaryBackend`, compute, transfer back. See
`binary_op_host_fallback/4`, `unary_op_host_fallback/3`, `reduce_op_host_fallback/4`
in `vulkano_backend.ex`. A kernel is worth writing when:

- The op is **bandwidth-bound** (elementwise add/mul/relu, cast, pad, slice,
  broadcast) — moving data to the host and back costs more than the compute, so
  even a trivial kernel wins by keeping the tensor on-device. These are the big
  wins here (add ~4-5x, reductions ~1.9x with f32; see `gpu-fleet-and-f32` memory).
- The op is **compute-bound and reused** (matmul, conv-as-GEMM) — worth tiling.
- The **dtype is supported**: f32 and f64 storage. f64 needs
  `GL_ARB_gpu_shader_fp64` (the NIF enables `shader_float64` only if the device
  advertises it — `lib.rs` `ctx()`). **GLSL.std.450 has no f64 transcendentals**
  (log/exp/pow/sin…): for f64 those must be boundary-cast `double(log(float(x)))`,
  or the op stays on the host. Integer dtypes generally host-fall-back.

Do NOT write a kernel for control-flow-heavy, host-cheap, or rarely-hit ops —
the fallback is always correct and the round trip is negligible there
(concat, sort, LU/QR/SVD, argmax…). Prefer growing the fallback list.

## 2. Anatomy of a static kernel (`glsl/*.comp`)

The simplest real one, `glsl/elementwise_binary_f32.comp`:

```glsl
#version 450
layout(local_size_x = 256) in;
layout(constant_id = 0) const int OP = 0;          // spec constant picks the op

layout(std430, binding = 0) readonly  buffer A { float a[]; };  // inputs first
layout(std430, binding = 1) readonly  buffer B { float b[]; };
layout(std430, binding = 2) writeonly buffer O { float o[]; };  // output LAST

layout(push_constant) uniform Push { uint n; } p;   // element count in push `n`

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;                            // bounds guard — REQUIRED
    // ... write o[i]
}
```

Repo conventions (follow them — the NIF layer assumes them):

- **std430** on every SSBO. Bindings: **inputs at 0..k-1, output last at k**
  (see `dispatch_generated` in `lib.rs`, which writes inputs then output).
- **Element count `n` in a push constant**, not a buffer. Wider ops pack a
  struct: matmul pushes `{m, n, k}`, reduce pushes `{outer, reduce_size, inner, op}`.
- A **spec constant at `constant_id = 0`** selects a variant (op code). The NIF
  passes it via `get_or_create_pipeline(spv_path, Some(op_code))`; a shader with
  no spec constant passes `None` (sentinel `-1` in the pipeline cache key).
- `#extension GL_ARB_gpu_shader_fp64 : require` at the top of any f64 shader.

## 3. Compiling `.comp` → `.spv`

Static kernels are precompiled with the Vulkan SDK's `glslangValidator`:

```sh
glslangValidator -V glsl/foo.comp -o priv/shaders/foo.spv
```

`-V` = compile GLSL to Vulkan SPIR-V. Commit the `.spv` (they ship in the hex
package — see the `priv/shaders` entry in `mix.exs` `package/0`). There is no mix
task — run the command above directly. (`scripts/build_and_test.sh` loops
glslangValidator over a shader dir, but it points at a **machine-local**
`/home/io/spirit/shaders`, not the repo's `glsl/` — don't rely on it to compile a
kernel you added under `glsl/`.) JIT kernels are compiled the same way but at
runtime by `Codegen.compile_cached/1`
(`System.cmd("glslangValidator", ["-V", comp_path, "-o", spv_path])`) into the
gitignored `priv/shader_cache/`.

## 4. Wiring a new static op end-to-end

See `reference/recipe.md` for the copy-pasteable skeleton. The four edits:

1. **`glsl/foo.comp`** → compile to `priv/shaders/foo.spv` (step 3).
2. **NIF in `native/nx_vulkan_vulkano/src/lib.rs`**: a `#[rustler::nif(schedule = "DirtyIo")]`
   fn taking `ResourceArc<VulkanoTensor>` buffers + scalars + `spv_path: String`;
   build the descriptor set (inputs then output), a `#[repr(C)] BufferContents`
   push struct, and dispatch via the shared `run_single_dispatch(context, &cached, set, push, groups)`.
   Add the fn name to the `rustler::init!` list at the bottom. Get the pipeline
   from `get_or_create_pipeline` (NEVER build a fresh pipeline per call except the
   legacy `matmul` — see gotchas: the descriptor-pool exhaustion bug).
3. **Stub in `lib/nx_vulkan/native_v.ex`**: `def foo(...), do: :erlang.nif_error(:nif_not_loaded)`
   with arg arity matching the NIF exactly. Required or the NIF won't load.
4. **`lib/nx_vulkan/vulkano_backend.ex`**: resolve the spv path
   (`Path.expand("../../priv/shaders/foo.spv", __DIR__)` as a module attr),
   `buf_alloc`/`buf_upload` the buffers, guard the GPU path (dtype + shape +
   `match?(%__MODULE__{}, t.data)`), dispatch, and **host-fall-back otherwise**.

Buffer lifecycle NIFs you'll reuse: `buf_upload/1` (host binary → device),
`buf_alloc/1` (zeroed device buffer of N bytes), `buf_download/1`,
`buf_upload_into/2`. Lifetimes are Rust-owned (`Arc<Buffer>` behind a Rustler
`ResourceArc`) — freed when the BEAM GCs the ref, no manual free.

## 5. std430 layout & param-buffer convention

- Scalars/small config → push constants. But push blocks are size-limited (the
  synth path caps at 128 bytes); **variable-length shape metadata goes in a
  dedicated `params` SSBO** of `int32`. The repo's packing convention, e.g.
  broadcast binary: `params = [rank, out[4], a[4], b[4]]`; slice:
  `[rank, ews, S[4], O[4], start[4], stride[4]]` where `ews = element_bytes/4`.
  Ranks are padded to 4 (`pad4/1` in the backend). Encode little-endian:
  `for v <- [...], into: <<>>, do: <<v::signed-32-little>>`.
- std430 array-of-scalar SSBOs (`float x[]`, `int x[]`) are tightly packed, so
  Elixir binaries map 1:1. Avoid `vec3`/nested structs in SSBOs (std430 pads
  them) unless you match the padding exactly.
- A u8 mask tensor (compare output / select pred) is read as **u32 words**
  (`ceil(n/4)`), so allocate the output padded to a 4-byte multiple, and the
  device needs `robust_buffer_access` for the tail (enabled in `ctx()`).

## 6. Dispatch geometry

- **Elementwise / one-thread-per-element**: `local_size_x = 256`,
  `groups = ceil(n / 256)` (`n.div_ceil(256)` in Rust). Guard `if (i >= p.n) return;`.
- **2D matmul**: `local_size = 16x16`, `groups = (ceil(N/16), ceil(M/16), 1)`,
  x over N (col), y over M (row). Register-blocked `*_rb32` variant uses a 32-wide
  output tile → `ceil(N/32), ceil(M/32)`.
- **Per-slot reductions** (`reduce_axis`): one thread per output slot,
  `groups = ceil(outer*inner / 256)`.
- **Workgroup-per-slot tree reduce** (fused reduce, `dispatch_generated_reduce`):
  **one whole workgroup per output slot**, dispatch `outer*inner` workgroups,
  NOT `ceil(/256)`.
- **The 65535 limit**: `maxComputeWorkGroupCount[0]` is typically 65535. If slot
  count can exceed it, the shader must **grid-stride**
  (`for (uint slot = gl_WorkGroupID.x; slot < slots; slot += gl_NumWorkGroups.x)`)
  and the NIF caps the launch (`n_slots.min(65535)`), as `dispatch_generated_reduce`
  does. A plain `ceil(n/256)` elementwise grid can also blow past 65535 for very
  large n — grid-stride there too if you expect >~16M elements.

## 7. Numerical correctness (must match BinaryBackend exactly)

The bar is: **bit-for-bit agreement with `Nx.BinaryBackend`** (the host
fallback), to f32/f64 round-off. Concretely, from this repo:

- **Sum accumulates in f64** even for f32 inputs — `Nx.BinaryBackend` sums in
  f64. `reduce_axis_f32.comp` and the fused reduce both do `double acc`,
  `acc += double(a[...])`, store `float(acc)`. A naive f32 accumulator diverges.
- **Matmul accumulator policy**: default `:f64` accumulator even for f32 storage
  (`matmul_f32_f64acc.comp` — `double acc`, `acc += double(A)*double(B)`), which
  matches an f64 reference to f32 round-off. `:f32` (`matmul_f32_f32acc.comp`) is
  faster (fewer f64 MACs, big on f64-rate-limited GPUs) but only accurate enough
  when opted in. Conv-as-GEMM has the same f32acc/f64acc split.
- **f64 transcendentals don't exist in SPIR-V** — boundary-cast through f32 (see
  §1). Document the precision cost.
- **Contiguity**: kernels assume row-major contiguous buffers. The reduce path
  only fast-paths axis patterns that map to contiguous `(outer, reduce_size,
  inner)` slabs (`classify_reduce_axes`); anything else host-falls-back. Don't
  feed a kernel a non-contiguous view.
- **Codegen `\br\b` bug** (JIT path): unary GLSL templates substitute the operand
  for the token `r` with a **word-boundary regex** `~r/\br\b/`, not
  `String.replace(_, "r", _)` — a plain replace clobbers the `r` inside
  `sqrt`/`round`/`reciprocal`/`erf`. See `codegen.ex` `node_expr/3`.

## 8. Performance lessons (real, and hardware-specific)

From the `thrust3-fusion-compiler` and `gpu-fleet-and-f32` memories:

- **16x16 shared-memory tiling** for GEMM is the baseline win — each global load
  is reused 16x; it fixed a 1024³ perf cliff. All GEMMs here are tiled.
- **Register blocking (32x32)** helps Ampere (RTX 3060 Ti, headroom) but
  **regresses both Kepler cards** (GT 650M, GT 750M). Kept as benchmark-only
  `*_rb32.spv` + the `matmul32` NIF, NOT the default.
- **Parallelism beats fusion for reductions**: the first fused reduce was
  one-thread-per-slot serial and regressed 0.3-0.6x everywhere. The win came
  from the **256-thread workgroup-per-slot tree reduce** out-parallelising eager's
  serial `reduce_axis` when slots are few (full sum 256² → 9.9x, 1024² → 27x on
  Kepler).
- **Win/loss crossovers are HARDWARE-SPECIFIC.** The many-slot fused reduce wins
  ~4.4x on Kepler but **regresses ~0.44x on Ampere** (a strong GPU's eager reduce
  is already well-fed by many slots). This is why `Nx.Vulkan.Device.class/0`
  (`:weak | :strong`, override `NXV_GPU_CLASS`) gates it: auto-enable only on weak
  GPUs. **CRITICAL: validate every perf heuristic across the fleet** (247 Kepler
  GT 650M / 248 Kepler GT 750M / 249 Ampere RTX 3060 Ti — see the
  `gpu-fleet-and-f32` memory for SSH access), never just the local box.

## 9. Verifying a kernel

1. **Confirm the real GPU is active, not llvmpipe** (the software Vulkan CPU
   fallback — correct but not a perf signal, and a couple of `select` tests fail
   only on it):
   ```elixir
   Nx.Vulkan.NativeV.device_name()   # {:ok, "NVIDIA GeForce GT 650M", "DiscreteGpu"}
   ```
   The NIF also prints `[nx_vulkan_vulkano] device: ...` on first init.
2. **Correctness = compare to `Nx.BinaryBackend`.** Run the op both ways and
   assert equality within dtype eps (this is exactly what the test suite and the
   host-fallback do). f32 sum/matmul must match the f64-accumulated reference.
3. `mix test` for the suite (recompile any changed `.spv` with the §3 command
   first). Benchmarks live in `examples/*_race.exs`; `sh scripts/race.sh`.
4. For a fused JIT kernel, `NXV_FUSE_DEBUG=1` logs FUSED/fallback per defn;
   `NXV_FUSE_REDUCE=1|0` forces/disables the reduce fusion.
