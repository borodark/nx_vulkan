# PLAN — f64 matmul in nx_vulkan (mac-248 implementation target)

Filed: 2026-07-05. Cross-refs:
[[exmc/research/PLAN_FREEBSD_FLEET.md]],
[[exmc memory: nx_vulkan_deficits.md]],
[[exmc DECISIONS.md D87/D88]].

## Why this task

The regime model (D88 fleet target) does not use matmul at the sampler
level, so f64 matmul does NOT block Stage 2 of the FreeBSD trial. It
DOES block:

- Any fleet model with `Nx.dot`/matmul in the log-density (linear
  regression posteriors, GP means, matrix-normal likelihoods, learned
  encoders inside custom `logpdf` fns).
- The `dense_mass: true` sampler option under Vulkan — dense mass
  matrix updates involve `Nx.dot` on covariance / inv-mass matrices.
  Currently silently host-transfers to BinaryBackend on the f64
  default (bandwidth cliff on every leapfrog step).

Filing now while the vulkano-f64 evidence is fresh, not later when a
model needs it and blocks a fleet Stage.

## Current state — what exists

- **Rust NIF `matmul`**: f32 only, at
  `native/nx_vulkan_vulkano/src/lib.rs:1624`. Uses `matmul.spv`.
  Push block `{m: u32, n: u32, k: u32}` (12 bytes). Workgroup 16×16,
  dispatch `ceil(N/16)×ceil(M/16)`. No dtype in push — element size
  is implicit in the shader.
- **Shipped SPVs**: `priv/shaders/matmul.spv`,
  `priv/shaders/matmul_tiled.spv`, `matmul_tiled16x2.spv`,
  `matmul_tiled32.spv`. All f32. No GLSL sources in the repo — SPVs
  are shipped precompiled.
- **VulkanoBackend `dot/7`**: `lib/nx_vulkan/vulkano_backend.ex:927+`.
  `fast_path` requires `type == {:f, 32}` for a, b, out. Anything
  else → `Nx.backend_transfer` to BinaryBackend and back (slow).
- **Device features**: `shader_float64` enabled in
  `native/nx_vulkan_vulkano/src/lib.rs:212-219` when supported.
  Confirmed on GT 650M / GT 750M / RTX 3060 Ti per #177 findings.

## Scope

Three surfaces, in order of implementation:

### Surface 1 — GLSL source and SPV

Write `glsl/matmul_f64.comp` (new directory — the repo currently
ships only SPVs). Contents:

```glsl
#version 450
#extension GL_ARB_gpu_shader_fp64 : require

layout(local_size_x = 16, local_size_y = 16) in;

layout(std430, binding = 0) readonly buffer A { double a[]; };
layout(std430, binding = 1) readonly buffer B { double b[]; };
layout(std430, binding = 2) writeonly buffer O { double o[]; };

layout(push_constant) uniform Push {
    uint m;
    uint n;
    uint k;
} push;

void main() {
    uint row = gl_GlobalInvocationID.y;
    uint col = gl_GlobalInvocationID.x;
    if (row >= push.m || col >= push.n) return;

    double acc = 0.0lf;
    for (uint i = 0u; i < push.k; ++i) {
        acc += a[row * push.k + i] * b[i * push.n + col];
    }
    o[row * push.n + col] = acc;
}
```

Compile with `glslangValidator -V glsl/matmul_f64.comp -o
priv/shaders/matmul_f64.spv`. Verify with
`spirv-val priv/shaders/matmul_f64.spv`.

Note: keep the naive path first (no tile). Tiling comes as a
follow-up if profiling shows need. f64 memory bandwidth is 2× the
f32 case; tile params for f64 differ from f32.

### Surface 2 — Rust NIF `matmul_f64`

Mirror `matmul` at `native/nx_vulkan_vulkano/src/lib.rs:1624`.
Push block is the same `{m, n, k}` — no doubles in push, only
u32 shape. Element size in buffer is `sizeof<f64> = 8`.

Register in `rustler::init!` list at line ~1739 next to `matmul`.

### Surface 3 — Elixir wrapper + backend routing

- `lib/nx_vulkan/native_v.ex`: add `matmul_f64(out_ref, a_ref, b_ref,
  m, n, k, spv_path)` NIF stub.
- `lib/nx_vulkan/vulkano_backend.ex:919`: add
  `@matmul_f64_spv Path.expand("../../priv/shaders/matmul_f64.spv",
  __DIR__)`.
- `lib/nx_vulkan/vulkano_backend.ex:927+` `dot/7`: split
  `fast_path` predicate by dtype. Route `{:f, 32}` to existing
  `matmul` + `@matmul_spv`; route `{:f, 64}` to new `matmul_f64` +
  `@matmul_f64_spv`. `element_bytes/1` already returns 8 for f64.

## Why mac-248 is the implementation host

- Primary glslang host per
  `[[exmc memory: reference_dev_machines.md]]`. Has
  `glslangValidator` and `spirv-tools` already installed and
  working (mac-248 built the f32 SPVs shipped in the repo).
- Confirmed `shader_float64` on Kepler GK106 (GT 750M) per #177
  three-host bit-exact evidence.
- FreeBSD 15 host — matches D88 production target. Any FreeBSD-
  specific f64 matmul quirk (driver rounding, SPIR-V validator
  version) surfaces here first.
- 247_TODO.md pattern already exists for mac-247 handoffs — a
  parallel 248 handoff is the established workflow.

super-io is a valid fallback (Linux + glslang + Ampere), but doing
Surface 1 on super-io hides FreeBSD-driver differences until Stage 2
integration.

## Testing plan

Land tests alongside the code, not after.

- **Unit** in `test/vulkano_backend_test.exs`: two new cases
  mirroring the f32 tests:
  1. f64 3×4 matmul vs BinaryBackend `Nx.dot`, abs diff < 1e-12.
  2. f64 512×512 matmul vs BinaryBackend `Nx.dot`, max rel diff
     < 1e-13 (accumulator precision check).
- **Parity** in `test/parity_test.exs`: add f64 matmul to the
  existing parity sweep.
- **Smoke** — end-to-end model with `dense_mass: true` on a small
  hierarchical model, compare posteriors vs EXLA-f64 within
  reference SDs (proves the dot() call actually routes through the
  f64 shader instead of falling back). Fixture: 3-RV hierarchical
  Normal, d=6, 500 warmup + 500 samples.
- **spirv-val** on the new SPV, checked into CI once CI catches up.

## Success criteria

1. `matmul_f64.spv` in `priv/shaders/`, spirv-val clean.
2. Rust NIF compiles on FreeBSD + Linux via `mix compile` in
   nx_vulkan.
3. All existing vulkano tests still pass (`mix test` in nx_vulkan).
4. New f64 matmul tests pass on mac-248 and super-io.
5. Hierarchical model with `dense_mass: true` on Vulkan matches
   EXLA reference within reference SDs, WITHOUT triggering any
   BinaryBackend host-transfer (measurable via
   `Exmc.NUTS.Vulkan.Dispatch.dispatch_count/0` and a Nx binary-
   transfer counter — add one if missing).

## Non-goals for this task

- **Tiled f64 matmul.** Naive path first. Profile after Surface 3
  lands; add tiled variants only if measured throughput matters
  and is worth the shader-compile complexity.
- **f64 batched matmul.** `dot/7`'s `batched_a`/`batched_b` route
  stays on BinaryBackend for now. Fleet models don't hit batched
  matmul at the sampler level.
- **f64 transpose_2d.** Separate deficit — file its own task if
  needed. `dot/7` doesn't require transpose currently (contracting
  axes handled shader-side).

## Handoff to mac-248

Concrete session-start checklist:

1. Read this plan.
2. Read the three-host findings for context:
   `~/exmc/exmc/research/177_grad_diff_findings.md` on mac-248.
3. Confirm `glslangValidator --version` and `spirv-val --version`
   are on PATH.
4. Confirm `Nx.Vulkan.NativeV.matmul/7` exists in the current
   nx_vulkan checkout (`grep -n matmul lib/nx_vulkan/native_v.ex`).
5. Start on Surface 1: write `glsl/matmul_f64.comp`, compile SPV,
   verify.
6. Commit each surface separately. Push to origin
   (`git@192.168.0.249:/home/git/repos/nx_vulkan.git` — mirror the
   #177 workflow).
7. Report results back with the same "per-host results" table
   format used in `177_grad_diff_findings.md`.

## Blocking dependencies

None. All prerequisites are on main (`shader_float64` device
feature, f64 elementwise NIFs, `Nx.Vulkan.has_f64?/0` present-but-
stale — see nx_vulkan_deficits.md).

## Cross-references

- [[exmc/DECISIONS.md D88]] — nx_vulkan strategic promotion for
  FreeBSD fleet.
- [[exmc/research/PLAN_FREEBSD_FLEET.md]] — the parent fleet plan.
  This task is post-Stage-2 (only unblocks fleet models that need
  matmul), not on Stage 2's critical path.
- [[exmc/research/175_f64_leapfrog_plan.md]] — the sister task.
  #175 established that vulkano-f64 works end-to-end at the
  elementwise level. This task extends that guarantee to matmul.
- [[exmc memory: nx_vulkan_deficits.md]] — where this task's
  motivation lives.
- `247_TODO.md` — mac-247 handoff pattern to mirror.
