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

(empty — workstream just opened)
