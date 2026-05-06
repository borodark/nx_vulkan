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

**Phase 1** (1-2 hours): test H7.1 only. Add `precise` qualifiers to the 4 failing shaders, recompile, re-run W2. This is the highest-leverage cheap test.

**Phase 2** (2-4 hours, only if H7.1 doesn't fix it): test H7.3 (denormal clamping). Then H7.5 (NVK). H7.2 only if those don't isolate it.

**Phase 3** (broader): if none of H7.1-H7.5 narrows it, escalate. File a Khronos / NVIDIA driver bug report with a minimal reproducer extracted from one failing shader.

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
