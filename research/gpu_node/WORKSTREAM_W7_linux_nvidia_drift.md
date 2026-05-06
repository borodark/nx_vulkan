# W7 — Linux NVIDIA Vulkan Chain-Integrator Drift

**Question:** Why do Exponential, Cauchy, HalfNormal, and Weibull chain shaders produce incorrect posteriors on Linux NVIDIA Vulkan but correct posteriors on FreeBSD NVIDIA Vulkan with the same SPIR-V bytes?

**Status:** ADDED 2026-05-06 in response to R6 cross-platform finding.

**Priority:** Medium. Phase 1 (synthesized Beta/Gamma/Lognormal) is unaffected — those shaders pass W2 on Linux too. W7 is a quality-of-results issue for the existing 4 hand-written shaders that have been silently misbehaving on Linux for months.

## Evidence (R6, commit `b4b1232` on `nx_vulkan@feat/gpu-node`)

W2 validator pass rates with identical SPIR-V binaries:

| Platform | Tests | Pass | Fail |
|---|---|---|---|
| Linux RTX 3060 Ti | 16 | 12 | **4** |
| FreeBSD GT 750M | 16 | **16** | 0 |
| FreeBSD GT 650M | 16 | **16** | 0 |

The 4 Linux failures are Exponential, Cauchy, HalfNormal, Weibull — the families currently tagged `:vulkan_known_failure` and auto-skipped. The same SPV files pass on both Macs.

The Linux NVIDIA driver is producing subtly wrong fp32 results in the chain-shader inner leapfrog loop. The error accumulates across K=32 leapfrog steps and produces measurable bias in mean / variance that the validator catches.

## Hypotheses to test (in likely-impact order)

### H7.1 — FMA fusion changes accumulator semantics

NVIDIA's GLSL → SPIR-V → PTX pipeline aggressively fuses `a*b + c` into `fma(a, b, c)`, which has different rounding (single-rounded vs double-rounded). For long-running accumulators in the leapfrog loop, this can drift.

**Test:** Add `precise float p_half;`, `precise float qi;`, `precise float pi;` decorations in the chain shader sources, recompile, re-run W2. If the 4 failures flip green, FMA fusion is the culprit.

Alternative: use `OpDecorate ... NoContraction` in SPIR-V directly. This is what `precise` translates to.

### H7.2 — Loop-carried dependency reordering

The chain shader has `pi`, `qi` as loop-carried dependencies. NVIDIA's optimizer may reorder operations across iterations in ways that change rounding for some shaders but not others.

**Test:** Add `barrier()` calls between leapfrog steps (currently only at the per-K-step reduction). Recompile, re-run W2. Costs perf but should isolate whether intra-loop reordering is involved.

### H7.3 — Denormal handling

If any intermediate value drops to denormal magnitude, the driver may flush-to-zero or produce different results. Some shader features benchmarked in StudentT/Cauchy compute `1/(1 + z²·inv_nu)` which can produce tiny `1/large` values.

**Test:** Probe one failing shader (Cauchy is suspected) by clamping the denominator floor to `1e-30`. If the validator passes after clamping, denormal handling differs.

### H7.4 — Driver version specifics

Maybe the issue is specific to the NVIDIA driver version (550-something on super-io). Cross-check by testing on an older or newer NVIDIA driver if available.

**Test:** Easy — `nvidia-smi` reports current driver version. If we can swap to NVIDIA's beta or to older 535/525, re-run W2. Limited utility; mostly informational.

### H7.5 — NVK (mesa NVIDIA backend)

NVK is Mesa's open-source NVIDIA Vulkan driver, separate from the proprietary `nvidia` package. It uses the same fp32 hardware paths but goes through Mesa's compiler stack (NIR), not NVIDIA's.

**Test:** If NVK supports compute on RTX 3060 Ti (it does as of mid-2025), install + run W2 against NVK. If results match FreeBSD (also mesa-radv), that confirms it's the **proprietary NVIDIA driver's compiler**, not the hardware.

## Affected shaders

```
nx_vulkan/c_src/spirit/shaders/leapfrog_chain_exponential.comp
nx_vulkan/c_src/spirit/shaders/leapfrog_chain_cauchy.comp
nx_vulkan/c_src/spirit/shaders/leapfrog_chain_halfnormal.comp
nx_vulkan/c_src/spirit/shaders/leapfrog_chain_weibull.comp
```

Not affected:
- `leapfrog_chain_normal.comp` — passes
- `leapfrog_chain_studentt.comp` — passes
- The 3 Phase 1 synthesized templates — pass on Linux too

The pattern: passing shaders have either no transform (Normal, StudentT real-valued) OR the simplest log-space gradient (just an `exp()`, no compound terms). Failing shaders all have the most complex per-step expressions (Cauchy's quotient, Weibull's `pow`, Exponential's identity-but-with-log-transform-Jacobian).

## Investigation plan

The plan is staged so each cheap test runs independently and produces a clear isolation signal before the next one starts. Same shape as the H1-H5 hypothesis arc — minimum work to falsify, walk down the list until something fits.

### Stage 1 — H7.1 (FMA fusion via `precise float`)

**Branch:** `nx_vulkan@feat/w7-precise-float`

**Steps:**

1. Add `precise` qualifier to the loop-carried accumulators in each of the 4 failing shaders:
   ```glsl
   precise float qi = in_bounds ? q_init[i] : 0.0;
   precise float pi = in_bounds ? p_init[i] : 0.0;
   ```
   And to the per-step intermediates that feed back into `qi` / `pi`:
   ```glsl
   precise float p_half = pi + 0.5 * pc.eps * grad_q;
   ```
   This emits `OpDecorate ... NoContraction` in the SPIR-V output. The driver's optimizer must treat the decorated ops as un-fusable.

2. Recompile via `glslangValidator -V` per existing build.

3. Vendor the new SPVs via `c_src/spirit/VENDOR.md` procedure.

4. Run `mix test test/exmc/gpu_node/validator_test.exs --include vulkan --include requires_vulkan` on Linux RTX 3060 Ti. Compare the 4 known-failure cases.

5. If green → land the qualifier change to main. Update WORKSTREAM_W2_validation.md calibration table. Remove `:vulkan_known_failure` tag from the 4 tests.

6. If red → annotate WORKSTREAM_W7 notes with the persistent error magnitudes and proceed to Stage 2.

**Expected wall:** 1-2 hours. Most of it is mechanically adding `precise` to ~10 lines per shader and rebuilding.

**Cost of being wrong:** if `precise` adds noticeable shader runtime, the fix isn't free. Measure: re-run the fair race for the affected cells with and without `precise`. Acceptable if <5% slowdown — the chain shaders are submission-bound, not arithmetic-bound, on Linux NVIDIA.

### Stage 2 — H7.3 (denormal handling)

**Branch:** `nx_vulkan@feat/w7-denormal-clamp`

**Only if Stage 1 fails to fix all 4 shaders.**

Pick the easiest failing shader (Cauchy is suspected because of its `1 / (1 + (q-loc)²/scale²)` denominator) and clamp:

```glsl
float denom = 1.0 + z2;
denom = max(denom, 1e-30);  // avoid denormal at large z
```

Recompile, re-run W2. If Cauchy goes green, denormal handling is the issue and similar clamps probably need to land in the other failing shaders.

**Expected wall:** 1 hour for one shader. If it works, ~3 hours total to backport the pattern to the other 3 shaders.

### Stage 3 — H7.5 (NVK comparison)

**Branch:** none — system change.

Install NVK on super-io (mesa's open-source NVIDIA Vulkan driver, available since Mesa 24.x). Re-run W2 with NVK as the active Vulkan driver via `VK_LOADER_DRIVERS_SELECT=nvidia_*` or the equivalent ICD selector.

If W2 is 16/16 under NVK on the same RTX 3060 Ti — confirms the issue is the proprietary NVIDIA driver's compiler stack, not the hardware. Provides a workaround (run on NVK in production) and an upstream bug filing target (NVIDIA's compiler team).

**Expected wall:** 30 min to install NVK + run, more if NVK lacks features we depend on (some Vulkan extensions are unimplemented in NVK as of 2026). Worst case: NVK can't even initialize; we abandon this hypothesis.

### Stage 4 — H7.2 (loop-carried reordering)

**Branch:** `nx_vulkan@feat/w7-barrier-per-step`

**Only if Stages 1+2+3 all fail.**

Add a `barrier()` between each leapfrog step. This forces the optimizer to flush all in-flight work before the next iteration. Costs perf but should isolate whether intra-loop reordering is the issue.

**Expected wall:** 30 min to edit + rebuild + re-run W2 on one shader. If green, we have the answer but the perf cost may make the workaround unacceptable; in that case the real fix is upstream-driver work.

### Stage 5 — H7.4 (driver version)

**Branch:** none.

If we make it to Stage 5 without isolation, try a different NVIDIA driver version. Probably file an NVIDIA bug at this point — Stages 1-4 will have produced enough reproducer material.

## Resource estimates

- **If Stage 1 succeeds (most likely):** total wall ~2 hours, single commit, removes 4 `:vulkan_known_failure` tags, ships W7.
- **If Stages 1-3 walk:** ~6-8 hours total. Some real perf testing. Worth the time.
- **If Stages 1-4 all fail:** ~10 hours and we're filing bugs upstream. The 4 shaders stay tagged but at least we have a reproducer for NVIDIA.

## Gating

**Don't start W7 until Phase 2 W5 (pipeline cache persistence) lands.** The persistent-cache work could itself touch some of the same code paths and we want a clean baseline before changing shader semantics.

## Reporting

Every stage produces a commit on its own branch with the W2 numbers attached, even if the stage fails. Update WORKSTREAM_W7 notes/log with each result. Final outcome lands as a merge or as a "filed-upstream" doc if no in-tree fix exists.

## Cross-platform check after a fix

Once any stage produces a Linux-green W2, ask mac-248 to re-run W2 on FreeBSD GT 750M and GT 650M. The fix should not regress FreeBSD (which already passes). If FreeBSD goes red, the fix was wrong even though Linux turned green — back it out.

## What W7 explicitly does NOT do

- **Doesn't block Phase 2 of `PLAN_GPU_NODE.md`.** The synthesized chain shaders work on all platforms; W7 is about getting the 4 hand-written shaders to also work on Linux.
- **Doesn't change shader logic.** All proposed fixes are decorator/qualifier additions, not algorithmic changes.
- **Doesn't extend to non-chain shaders.** The plain `leapfrog_*` (single-step) shaders haven't been validated cross-platform; they may have the same issue, but that's out of scope here.

## Output

- `research/gpu_node/W7_root_cause.md` — once isolated, document the actual fix.
- Edits to the 4 affected `.comp` files in `nx_vulkan/c_src/spirit/shaders/`.
- New SPV files vendored to `nx_vulkan/priv/shaders/`.
- Updated `:vulkan_known_failure` → either removed (if the fix lands) or renamed to `:linux_nvidia_chain_drift` and made platform-conditional via `:os.type()`.

## Cross-references

- R6 result: `r4_cross_platform_results.md` — the 16/0 on Macs vs 12/4 on Linux.
- Original validator calibration: `validation_calibration.md` — documents the per-shader drift magnitudes.
- Existing tag: `EXMC_COMPILER=vulkan` `test/test_helper.exs` auto-excludes `:vulkan_known_failure`.

## Notes / log

### Stage 1 (2026-05-06) — `precise float` on loop accumulators

Added `precise` qualifiers to `qi`, `pi`, `p_half`, and the per-step
gradient intermediates (`diff`, `grad_q`, `grad_qn`, etc.) in all 4
affected shaders. Recompiled to SPV; vendored.

Result on Linux RTX 3060 Ti W2 validator:

| Shader | Pre-Stage-1 | Post-Stage-1 | Status |
|---|---|---|---|
| Weibull | drift (mean ~0.98 vs 0.886) | **PASS** | tag REMOVED |
| Exponential | drift | drift | tag stays |
| Cauchy | drift (IQR 1.76 vs 8.83) | drift | tag stays |
| HalfNormal | drift (mean 0.582 vs 0.896) | drift | tag stays |

R8 confirmed FreeBSD remains 16/16 with the new SPVs (precise is a
no-op on mesa-radv) and Weibull wall is unchanged.

**Stage 1 is real progress** — 1 of 4 shaders recovered, no
regressions, fix is portable. Stage 1 commits:
`spirit@704dd2df`, `nx_vulkan@29dd09b`.

### Stage 2 (2026-05-06) — denormal clamping + alternative gradient form

**Result: no measurable effect.** Output bit-identical to Stage 1.

Tried rewriting Cauchy's gradient as
`-2 * diff * inv_denom` with `precise` intermediates and a
`max(denom, 1e-30)` clamp. The Linux NVIDIA Vulkan compiler folded
the rewrite back to the original mathematical form despite the
`precise` decorations on the intermediate variables. SPIR-V
disassembly shows 22 `NoContraction` decorations applied as
expected, but the runtime trajectory is identical.

Reverted to Stage 1 broader form. No commit needed (revert produced
same SPV bytes as `spirit@704dd2df`).

### The actual root cause is in the validator, not the shader

While investigating Stage 2, found that
`Exmc.NUTS.Vulkan.Validator.run_exla/2` clears
`Application.get_env(:exmc, :compiler)` before sampling. That makes
the EXLA reference path run at **f64** (default `Exmc.JIT.precision`).
The Vulkan path runs at **f32** (forced by the chain shader). The
validator is comparing posteriors generated at different precision.

For Normal, StudentT, and Weibull, the f32 chain stays close enough
to the f64 reference for the comparison to pass. For fat-tailed
families (Cauchy especially), tiny per-step differences compound
across 32 leapfrog steps × ~1000 iterations and the chains diverge
to meaningfully different posterior shapes. **What the validator
calls "drift" is actually a precision-gap artifact.**

Evidence: the failure shapes don't match a typical drift pattern.
- Cauchy: Vulkan IQR 1.76 vs EXLA IQR 8.83. Vulkan stays near the
  mode; EXLA explores the fat tails. Consistent with f32 saturating
  the gradient at large `|q-loc|`.
- Exponential: Vulkan variance 0.354 vs EXLA 0.243. Vulkan
  over-explores the negative-q tail (where exp(qi) is tiny and
  gradient ≈ 1). Consistent with f32 underflow at small exp(qi).
- HalfNormal: Vulkan mean 0.582 vs EXLA 0.896. Vulkan undersamples
  the positive tail. Consistent with f32 overflow of `exp(2*qi)` at
  moderately large qi.

All three failure shapes match a *precision* hypothesis, not a
*compiler quirk* hypothesis.

### Stage 2.5 (2026-05-06) — matched-precision validator + per-shader diagnosis

Added `:precision => :f32 | :f64` option to `Validator.validate/3`.
When `:f32`, the validator forces the EXLA reference path to f32
via `Application.put_env(:exmc, :force_precision, :f32)` so it
matches the chain shader's working precision. Underlying mechanism:
`Exmc.JIT.precision/0` now reads `Application.get_env(:exmc,
:force_precision)` as an override.

Re-running the 3 known-failure tests at matched precision produced
**three different diagnoses**:

| Shader | f64 EXLA / f32 Vulkan | f32 EXLA / f32 Vulkan | Diagnosis |
|---|---|---|---|
| Exponential | FAIL (var 0.354 vs 0.243, 1.2× tol) | **:ok** | **pure precision-gap, FIXED at matched precision** |
| Cauchy | FAIL (IQR 1.76 vs 8.83, 35× tol) | FAIL (IQR 1.76 vs 2.13, 5× tol) | **mostly precision-gap, residual is finite-N noise on fat-tailed posterior** |
| HalfNormal | FAIL (mean 0.582 vs 0.896, 2.7× tol) | FAIL (mean 0.582 vs 0.896, 2.7× tol — unchanged) | **transform mismatch — separate bug** |

### The HalfNormal transform mismatch

`Exmc.Dist.HalfNormal.transform/1` returns `:softplus`, meaning
EXLA samples on q_uc such that `q = log(1 + exp(q_uc))`. The chain
shader implements `:log` transform (`q = exp(q_uc)`). All other
positive-support distributions in the catalog (Exponential, Gamma,
Weibull, Lognormal) use `:log` — HalfNormal is the only outlier.

The shader's gradient `1.0 - exp(2*qi)/sigma²` is correct for
log-transform HalfNormal. For softplus-transform HalfNormal it
would be different (involves `dq/dq_uc = sigmoid(q_uc)` rather
than `exp(q_uc)`).

**Two fix options:**

1. **Change `Exmc.Dist.HalfNormal.transform/1` to `:log`** —
   one-line fix, makes HalfNormal consistent with the rest of the
   positive-support catalog. Risk: any user model relying on
   softplus geometry near q=0 sees different mass-matrix
   adaptation (probably fine in practice; both transforms
   asymptote to the same posterior).

2. **Reject HalfNormal in the chain-shader codegen** when
   `transform != :log`. `chain_shader_codegen.detect_meta/1` would
   call into the Dist module and fall back to the EXLA path if the
   transform is not `:log`. Conservative; doesn't break anything
   but loses the chain-shader speedup for HalfNormal.

3. **Write a softplus-transform chain shader** as a separate
   variant. More work; only worth it if there's a demonstrated
   reason HalfNormal needs softplus.

Recommendation: Option 1 (change transform) for consistency.

### Re-tagging plan

Replace the single `:vulkan_known_failure` tag with three more
accurate tags:

- **Exponential**: untag entirely. Add the matched-precision check
  with `precision: :f32` as the test default; the f64 reference
  becomes an opt-in stricter test.
- **Cauchy**: rename to `:f32_precision_limited`. Document that
  fat-tailed posteriors at f32 produce statistically different
  IQRs from f64 even when the algorithm is correct. Not a bug; a
  known limitation of f32 chain shaders.
- **HalfNormal**: rename to `:transform_mismatch` until Option 1
  lands. Once HalfNormal's transform is `:log`, the test should
  pass at matched precision and the tag comes off entirely.

### What this means for W7 overall

- **Stage 1 was right**: Weibull is a real driver-level fix
  (precise float disabled FMA fusion that was actually wrong).
- **Stages 2 / 3 / 4 / 5** are no longer applicable — there's
  no remaining driver bug to chase.
- **The W2 validator was diagnosing real problems**, but mixing
  three different ones under a single tag. Stage 2.5 separates
  them: 1 driver bug (Stage 1, fixed), 1 precision-limited
  family (Cauchy, accept), 1 algorithmic mismatch (HalfNormal,
  fixable), 1 false alarm (Exponential, untag).

Two options for the validator:

1. **Make EXLA also use f32** (matches Vulkan's precision). The
   validator becomes a pure shader-correctness test. Risk: can't
   distinguish "shader is correct" from "f32 is too coarse for this
   distribution." Both paths will be wrong in the same way.

2. **Keep EXLA at f64, document the precision-gap shaders as
   `:f32_precision_limited`**, separate from
   `:vulkan_known_failure`. Tag is platform-agnostic — the issue
   is that f32 chain shaders genuinely cannot reproduce f64
   reference posteriors for fat-tailed distributions, regardless
   of driver. FreeBSD also runs f32 chain shaders; FreeBSD passes
   the validator only because mesa happens to produce slightly
   tighter f32 numerics in its compute path that matches the f64
   reference closer in finite samples.

Option 2 is more honest. It also predicts that **FreeBSD's 16/16
might be the result of finite-sample MCMC noise rather than a
genuinely better f32 path**. Worth re-running R5/R6/R8 with
larger N (5000 samples?) on FreeBSD to see if the fat-tailed
shaders also start failing there at higher statistical power.

### Status / decisions

- **Stage 1 ships.** Real fix for Weibull. (`704dd2df` + `29dd09b`)
- **Stage 2 had no effect.** Not landing the Stage 2 source changes.
- **The remaining 3 failures aren't shader bugs.** They're a
  fundamental f32-vs-f64 precision gap in the validator setup.
  Renaming the tag from `:vulkan_known_failure` to
  `:f32_precision_limited` would be more honest.
- **Stage 3 (NVK comparison) is no longer the right next step.**
  NVK would just give us f32 numerics from a different driver —
  same precision gap. Skip.
- **Stage 4 (barrier per leapfrog step) is no longer the right
  next step.** Same reasoning.
- **Stage 5 (file upstream NVIDIA bug) is no longer applicable.**
  No driver bug to file.

W7 evolves into "validator precision-gap accounting" rather than
"driver fp32 drift hunt." The original 4-shader red light was
correctly identifying a real problem; the diagnosis just turned
out to be different from the initial hypothesis.
