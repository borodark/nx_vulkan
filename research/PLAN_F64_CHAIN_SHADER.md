# PLAN — f64 chain shader surface (mac-248 implementation target)

Filed: 2026-07-05. Cross-refs:
[[exmc/DECISIONS.md D87 D88]],
[[exmc/research/175_f64_leapfrog_plan.md]] (Surface 3, non-goal there),
[[nx_vulkan/research/PLAN_F64_MATMUL.md]] (commit `423d258`, sister task),
[[nx_vulkan/248_TODO.md]].

Surprise from the survey: `leapfrog_chain_normal_f64.spv` already
exists (spirit C++ path, `Nx.Vulkan.Native`), and
`leapfrog_chain_synth_f64` is fully wired end-to-end on the vulkano
Rust side (`native/nx_vulkan_vulkano/src/lib.rs:590`, wrapper
`native_v.ex:66`) — both have no caller. The matrix is partial, not
empty. This plan finishes it.

## Why this task

The regime model does NOT hit this path — Custom likelihood, so
`ChainShaderCodegen.detect_meta/1` returns `:unsupported` (Plan B'
guard, `compiler.ex:99`), routing through `Evaluator +
VulkanoBackend` per-op which is already precision-portable per #175.
This task does NOT block D88 Stage 2.

It DOES block, silently, on the f64 default:

- **Plan B fast path** (spirit C++) — Single-family Vulkan models
  Normal/Exponential/StudentT/Cauchy/HalfNormal/Weibull dispatch
  through the f32 SPVs. Same silent-collapse class as #177's regime
  failure (D87 root cause: f32 accumulation, emitter correct).
- **Plan A\* synth path** (vulkano) — Custom-likelihood models
  whose emitter output goes through the vulkano
  `leapfrog_chain_synth` NIF. Synth-f64 exists but is unreachable —
  dispatch.ex routes to the f32 sibling unconditionally.

## Current state — what exists

- **Spirit C++ (`nx_vulkan_native`, Plan B fast path):** 6 built-in
  families shipped f32 — `normal`, `exponential`, `studentt`,
  `cauchy`, `halfnormal`, `weibull` (SPVs in `priv/shaders/`, NIFs
  `native.ex:144-172`). Only Normal has an f64 sibling:
  `leapfrog_chain_normal_f64.spv` + NIF (`native.ex:168`) + wrapper
  (`nx_vulkan.ex:702`) — currently unused.
- **Vulkano (`nx_vulkan_vulkano`, Plan A\*):** `leapfrog_chain_synth`
  (f32, `src/lib.rs:424`) + `leapfrog_chain_synth_f64` (f64,
  `src/lib.rs:590`, wrapper `native_v.ex:66`). Rust NIFs both
  registered. f64 NIF is unused because dispatch.ex never calls it.
- **exmc dispatch (`Exmc.NUTS.Vulkan.Dispatch`):** every call site
  hardcodes `Nx.as_type(:f32)` at the NIF boundary — load-bearing
  per the moduledoc "Precision boundary" (lines 29-50), which
  explicitly flags this task as future work.
- **GLSL sources:** NOT in the repo. SPVs shipped precompiled (same
  as matmul).

## Scope — surfaces in implementation order

Each surface is independently commitable. Land tests alongside.

### Surface 1 — GLSL sources for the 5 missing spirit families

Write `glsl/leapfrog_chain_{exponential,studentt,cauchy,halfnormal,weibull}_f64.comp`.
Match the existing f32 shader's math bit-for-bit; only change is
`float`→`double` on buffers/locals and
`#extension GL_ARB_gpu_shader_fp64 : require`. If f32 GLSL sources
aren't in the repo, disassemble via
`spirv-cross --output tmp.comp priv/shaders/leapfrog_chain_<fam>.spv`
as a starting point.

### Surface 2 — `log_d` / `exp_d` boundary-cast helpers

GLSL.std.450 has no f64 transcendentals. Emit as inline helpers in
each shader that needs them (StudentT, Weibull, HalfNormal, Cauchy,
LogNormal-inside-synth):

```glsl
double log_d(double x) { return double(log(float(x))); }
double exp_d(double x) { return double(exp(float(x))); }
```

Precision loss is bounded — cast down at f32 mantissa (23-bit),
back up to f64 storage.

### Surface 3 — SPV compilation

`glslangValidator -V glsl/<name>.comp -o priv/shaders/<name>.spv`,
then `spirv-val`. 5 new SPVs (Normal-f64 already shipped) + 1
synth-f64 SPV emitted by the codegen at Surface 7.

### Surface 4 — Rust NIFs (spirit path only)

`native/nx_vulkan_native/src/lib.rs`: mirror
`leapfrog_chain_normal_f64` for the 5 missing families. Push blocks:
same layout as f32 sibling, `double` scalars where the f32 sibling
has `float`. Register in the C shim and the `rustler::init!` list.

Vulkano: no per-family NIFs needed — synth-f64 already registered.

### Surface 5 — Elixir wrappers

`nx_vulkan/lib/nx_vulkan/native.ex` + `nx_vulkan.ex`: mirror
`leapfrog_chain_normal_f64` (`native.ex:168`, `nx_vulkan.ex:702`)
for the 5 missing families. `shader_path("leapfrog_chain_<fam>_f64.spv")`.

### Surface 6 — `Exmc.NUTS.Vulkan.Dispatch` routing

At each family call site in `dispatch.ex`, branch on
`Exmc.JIT.precision()`:

- `:f32` → existing NIF + f32 SPV path + `Nx.as_type(:f32)` casts.
- `:f64` → `_f64` NIF + f64 SPV + `Nx.as_type(:f64)` casts +
  `bin_to_tensor/2` reads `:f64` instead of `:f32`.

Update the moduledoc "Precision boundary" section — remove the
"Future work" note; this task closes it.

### Surface 7 — `custom_synth.ex` codegen for f64

`MultiRvCustomSpec.render/1` currently emits f32 GLSL. Add a
`precision: :f64` mode: swap `float`→`double`, add the
`GL_ARB_gpu_shader_fp64` extension line, inject `log_d`/`exp_d`
helpers when the captured decls reference log/exp. Dispatch picks
the f64 shader when `precision() == :f64`.

Task #154's `@batched_template` (multi_rv_custom_spec.ex:684) stays
f32 — batched f64 is a Non-goal per that file's line 699-701.

## Non-changes to keep

- f32 SPVs stay side-by-side with f64 for hardware without
  `shaderFloat64`. Fleet has it; keep the escape hatch.
- `force_precision: :f32` env override remains the fallback signal.
- Batched synth (`leapfrog_chain_synth_batch`) stays f32.

## Testing plan

Land tests alongside the code.

- **Per-shader unit** in `test/exmc/nuts/vulkan/dispatch_test.exs`:
  one call per family, f64, vs BinaryBackend `Evaluator`. Tolerance:
  `< 1e-12` rel for polynomial-dominated (Normal, Cauchy); `< 1e-8`
  for transcendental-heavy (StudentT, Weibull, HalfNormal,
  Exponential) because of `log_d`/`exp_d` boundary noise.
- **Full sampling** in `test/exmc/regression/`: one model per
  family, 500 warmup + 500 samples, posterior mean/sd match EXLA-f64
  within reference SDs. Assert `Dispatch.dispatch_count/0 > 0` to
  catch silent Evaluator fallback (dispatch moduledoc lines 17-28).
- **Silent-collapse smoke test** — Normal(mu, sigma), `sigma` prior
  scale `0.01` (regime failure mimic per D87). Under f64 default:
  posterior recovers within reference SDs. Under `force_precision:
  :f32`: sampler collapses (assertion inverted, expects failure).
  Fixture in `test/exmc/regression/f64_chain_collapse_test.exs`.
- **spirv-val** on each new SPV in CI.

## Success criteria

1. All 5 missing spirit-family f64 SPVs in `priv/shaders/`,
   spirv-val clean.
2. Synth-f64 codegen path emits a working SPV; dispatch.ex routes
   to `leapfrog_chain_synth_f64` NIF when `precision() == :f64`.
3. `mix test` in nx_vulkan and exmc both pass on mac-248 and super-io.
4. Every family's sampling test matches EXLA-f64 within reference SDs.
5. Silent-collapse smoke test passes: sigma=0.01 Normal samples
   correctly under f64 default, collapses under `force_precision: :f32`.

## Non-goals for this task

- **Tiled f64 chain shaders.** Naive path first; profile after
  Surface 6 lands. f64 memory bandwidth is 2× f32 — tile params
  will differ.
- **Batched f64 chain shader** (Task #154 multi-instance path).
  `@batched_template` stays f32 per its own comment.
- **New families** (Beta, Gamma, LogNormal, Exponential-power).
  Not currently in the fast-path family set; file a separate task
  if any fleet model needs them.
- **f16 / int8 anything.**
- **Converting spirit C++ `Native` synth wrapper.** Spirit path is
  legacy; #175's D87 update deferred it. Vulkano-only for the
  synth-f64 route.

## Blocking dependencies

None. `shaderFloat64` device feature enabled per #175 evidence
across the three-host fleet; `leapfrog_chain_synth_f64` Rust NIF
already merged; Normal-f64 SPV already shipped as the reference
for GLSL shape.

## Handoff to mac-248

Concrete session-start checklist:

1. Read this plan and `PLAN_F64_MATMUL.md` (commit `423d258`).
2. Confirm `glslangValidator --version` and `spirv-val --version`
   on PATH.
3. Confirm the shipped `leapfrog_chain_normal_f64.spv` disassembles
   cleanly via `spirv-cross --output tmp.comp
   priv/shaders/leapfrog_chain_normal_f64.spv` — use as the GLSL
   template for the 5 siblings.
4. Surface 1 → Surface 3 first (all 5 spirit-family SPVs). Commit
   each family separately (small blast radius).
5. Surface 4 → 5 (Rust NIFs + Elixir wrappers) in one commit per
   family, mirroring `leapfrog_chain_normal_f64`'s existing shape.
6. Surface 6 (exmc dispatch routing) in a single commit — touch
   only `dispatch.ex`.
7. Surface 7 (synth-f64 codegen) last — biggest exmc-side blast
   radius; land after the fast path is proven.
8. Push each commit to `super-io` remote
   (`git@192.168.0.249:/home/git/repos/nx_vulkan.git` and the exmc
   mirror), same workflow as #177/matmul.
9. Report per-host smoke-test results in the
   `177_grad_diff_findings.md` table format.

Scope estimate: ~1.5 days at matmul work-rate. 10 SPVs (5 spirit +
1 synth + Normal-f64 verification + 3 helper compiles) vs matmul's
1, but the shader math is copy-diff from existing f32, not novel.
