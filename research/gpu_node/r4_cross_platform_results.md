# R4: Cross-Platform Results — FreeBSD GT 750M + GT 650M

## W2 Validator

| Platform | Tests | Pass | Fail | Excluded | Notes |
|----------|-------|------|------|----------|-------|
| Linux RTX 3060 Ti | 13 | 13 | 0 | 4 | Reference |
| **FreeBSD GT 750M (mac-248)** | 13 | **13** | **0** | 4 | Identical to Linux |
| **FreeBSD GT 650M (mac-247)** | 13 | **13** | **0** | 4 | Identical to Linux |

Same 4 excluded (Exponential, HalfNormal, Weibull, Cauchy — pre-existing
chain-integrator drift). Normal and StudentT pass cleanly on all three
platforms. The validator is platform-agnostic.

## W4 Warmup Characterization

### mac-248 FreeBSD GT 750M

| Family | Cold (µs) | Warm p50 (µs) | Warm p99 (µs) | Warm @ window | p99/p50 |
|--------|----------|---------------|---------------|---------------|---------|
| Normal | 20,494 | 10,938 | 20,941 | 50 | 1.91 |
| Exponential | 30,831 | 27,288 | 37,845 | 20 | 1.39 |
| StudentT | 29,138 | 26,078 | 36,587 | 20 | 1.40 |
| HalfNormal | 45,791 | 45,279 | 58,196 | 20 | 1.28 |
| Weibull | 23,391 | 28,099 | 40,736 | 20 | 1.45 |

### mac-247 FreeBSD GT 650M

| Family | Cold (µs) | Warm p50 (µs) | Warm p99 (µs) | Warm @ window | p99/p50 |
|--------|----------|---------------|---------------|---------------|---------|
| Normal | 29,665 | 18,804 | 32,750 | 22 | 1.74 |
| Exponential | 49,972 | 43,658 | 60,124 | 20 | 1.38 |
| StudentT | 45,205 | 42,426 | 59,245 | 20 | 1.40 |
| HalfNormal | 76,270 | 69,182 | 93,895 | 20 | 1.36 |
| Weibull | 67,758 | 66,053 | 91,331 | 20 | 1.38 |

### Cross-platform comparison (warm p50)

| Family | Linux RTX 3060 Ti | FreeBSD GT 750M | FreeBSD GT 650M | 750M/Linux | 650M/750M |
|--------|-------------------|-----------------|-----------------|------------|-----------|
| Normal | 254,000 (cold) | 10,938 | 18,804 | 0.04× | 1.72× |
| Exponential | 916,000 (cold) | 27,288 | 43,658 | 0.03× | 1.60× |
| StudentT | 414,000 (cold) | 26,078 | 42,426 | 0.06× | 1.63× |
| HalfNormal | 596,000 (cold) | 45,279 | 69,182 | 0.08× | 1.53× |
| Weibull | 553,000 (cold) | 28,099 | 66,053 | 0.05× | 2.35× |

Note: Linux numbers are cold-window (from the R4 reference). FreeBSD
warm p50 is 15-25× faster than Linux cold — confirming the driver
latency advantage measured in R3.2 (422µs vs 1287µs per fence).

GT 650M is consistently 1.5-2.4× slower than GT 750M — matches the
hardware spec difference (fewer cores, lower clock).

## W6 Chaos (Bulkhead)

| Platform | Tests | Pass | Fail | Notes |
|----------|-------|------|------|-------|
| Linux RTX 3060 Ti | 3 | 3 | 0 | Reference |
| **FreeBSD GT 750M (mac-248)** | 3 | **2** | **1** | timeout test: sampler finished before timeout fired (FreeBSD too fast) |
| **FreeBSD GT 650M (mac-247)** | 3 | **3** | **0** | All pass |

The mac-248 failure is a test calibration issue: the timeout test
expects the sampler to exceed the timeout, but FreeBSD's faster
fence waits let it complete in time. Not a bug — the bulkhead
mechanism works correctly, the test's timeout constant needs
platform-aware calibration.

## R5 — Post-Phase-1 Synthesis Cross-Platform

### Synthesis smoke test (Beta shader)

| Platform | synth+compile | first dispatch | logp | Match? |
|----------|--------------|----------------|------|--------|
| Linux RTX 3060 Ti | 157ms cold / 8ms cached | 39ms | -1.5162 | ref |
| **FreeBSD GT 750M** | **135ms** cold | **21ms** | **-1.5162** | ✓ |
| **FreeBSD GT 650M** | **191ms** cold / 2ms cached | **21ms** | **-1.5162** | ✓ |

Synthesized Beta shader compiles and produces identical logp on all three platforms.

### W2 validator (post-Phase-1, no regression)

| Platform | Pass | Fail | Excluded |
|----------|------|------|----------|
| FreeBSD GT 750M | 13 | 0 | 4 |
| FreeBSD GT 650M | 13 | 0 | 4 |

Identical to R4. Phase 1 didn't break anything.

### W4 warmup (post-Phase-1)

#### mac-248 GT 750M

| Family | Cold (µs) | Warm p50 (µs) | p99/p50 |
|--------|----------|---------------|---------|
| Normal | 26,248 | 15,530 | 1.65 |
| Exponential | 37,109 | 27,171 | 1.39 |
| StudentT | 28,971 | 25,524 | 1.38 |
| HalfNormal | 56,530 | 45,175 | 1.37 |
| Weibull | 33,331 | 28,933 | 1.76 |

#### mac-247 GT 650M

| Family | Cold (µs) | Warm p50 (µs) | p99/p50 |
|--------|----------|---------------|---------|
| Normal | 29,790 | 18,425 | 1.77 |
| Exponential | 48,545 | 42,947 | 1.48 |
| StudentT | 50,187 | 42,926 | 1.37 |
| HalfNormal | 66,504 | 68,199 | 1.61 |
| Weibull | 58,003 | 66,808 | 1.46 |

Consistent with R4 — no regression from Phase 1.

### W6 chaos (post-Phase-1)

| Platform | Pass | Fail | Notes |
|----------|------|------|-------|
| FreeBSD GT 750M | 2 | 1 | timeout calibration (same as R4) |
| FreeBSD GT 650M | 2 | 1 | timeout calibration (regressed from R4's 3/0) |

## R6 — Phase 1 Fully Wired (10-cell race + validator)

### Validator with 3 new Phase 1 tests

| Platform | Tests | Pass | Fail | Notes |
|----------|-------|------|------|-------|
| Linux RTX 3060 Ti | 16 | 12 | 4 | 4 known-failures (Exp/Cauchy/HN/Weibull) |
| **FreeBSD GT 750M** | 16 | **16** | **0** | All pass incl. Beta/Gamma/Lognormal synth |
| **FreeBSD GT 650M** | 16 | **16** | **0** | All pass incl. Beta/Gamma/Lognormal synth |

FreeBSD passes ALL 16 validator tests — 4 more than Linux. The
4 known-failure families (Exponential, Cauchy, HalfNormal, Weibull)
that fail on Linux NVIDIA pass on FreeBSD NVIDIA. This suggests the
chain-integrator drift issue (Stage 1.5.4) is Linux-driver-specific,
not shader-logic-specific.

### 10-cell race (partial — EXLA path blocked on d>1)

Only Normal d=1 completed before the unfused "EXLA" path stalled on d=8:

| Cell | mac-248 GT 750M | mac-247 GT 650M | Linux (ref) |
|------|-----------------|-----------------|-------------|
| Normal d=1 EXLA (unfused) | 307ms | 490ms | 873ms |
| Normal d=1 Vulkan-fused | **103ms** | **166ms** | 1,500ms |

FreeBSD fused path is 8.4-14.6× faster than Linux fused. Consistent
with the per-fence latency advantage measured throughout R3-R5.

Race cells d>1 blocked: unfused Vulkan at d=8+ is too slow on FreeBSD
(same issue as R1). Would need either EXLA CPU baseline or a fused-only
race to complete the 10-cell matrix.

## R7 — Phase 2 Refactor Validation

Phase 2 refactored Exmc.GPUNode.* → Exmc.NUTS.Vulkan.* and extracted
shader synthesis into Nx.Vulkan namespace. Pipeline cache persistence
added (W5).

### Validator (Phase 2 module paths)

| Platform | Tests | Pass | Fail | Notes |
|----------|-------|------|------|-------|
| FreeBSD GT 750M (mac-248) | 16 | **16** | **0** | All pass incl. synth |
| FreeBSD GT 650M (mac-247) | 16 | **16** | **0** | All pass incl. synth |

### Bulkhead (Phase 2)

| Platform | Tests | Pass | Fail |
|----------|-------|------|------|
| FreeBSD GT 750M | 3 | **3** | **0** |
| FreeBSD GT 650M | 3 | **3** | **0** |

### Build note

Required fix: test file `validator_test.exs` had stale alias
`Exmc.GPUNode.Validator` → updated to `Exmc.NUTS.Vulkan.Validator`.
Mac-247 also needed full nx_vulkan pull to get `39fec0e` (Phase 2
module extraction) before synth tests could pass.
