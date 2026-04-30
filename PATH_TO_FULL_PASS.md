# Path to Full exmc Test Pass on Nx.Vulkan

**Date:** 2026-04-30
**Baseline:** 622 tests, 29 failures (95.3% pass rate)
**Target:** 622 tests, 0 failures
**Machine:** FreeBSD 15.0, GT 750M (Vulkan), OTP 27 + Elixir 1.18.4

---

## Failure analysis (29 failures)

| Category | Count | Files | Root cause |
|---|---|---|---|
| Timeouts | 13 | trading (8), distributed (5) | BinaryBackend too slow for NUTS |
| ArithmeticError | 8 | bhm_benchmark (3), integration (3), trading (2) | f64 overflow on wide priors |
| License NIF | 1 | license_nif_test | Build env mismatch |
| Downstream | 7 | trading, compiler, stan, level_set | Cascade from above |

**Zero test bugs.** All failures expose real backend limitations.

---

## The three gaps

### Gap 1: Speed (13 timeouts → 0)

**Problem:** NUTS sampling on BinaryBackend is 30-100x slower than
EXLA-CUDA. Complex models (5+ parameters, hierarchical) timeout at
60s or 300s.

**Fix path (ordered by impact):**

| Step | What | Expected speedup | Effort |
|---|---|---|---|
| 1a | **Persistent buffers** | 100-700x on alloc overhead | 1.5 d |
| 1b | **Pipeline caching** | ~22ms/dispatch → ~0 on warm | 0.5 d |
| 1c | **f32 shader path for all ops** | Eliminate host fallback round-trips | Already done (117/0 nx_vulkan) |
| 1d | **Fused elementwise chains** | 1.6-4x on op chains | Shader done, Elixir wiring 1 d |
| 1e | **Per-axis reduce shader** | Eliminate host-materialize for Welford | Shader done, Elixir wiring 1 d |
| 1f | **Tiled matmul auto-select** | 4.2x on 1024x1024 (16x2 variant) | 0.5 d |

**Total effort:** ~5.5 days
**Expected outcome:** 5-20x speedup over current. Timeouts at 60s
become 3-12s completions. The 300s timeouts become 15-60s.

**If still timing out after 1a-1f:** increase test timeouts for the
GPU path (legitimate — GPU cold-start is slower, but steady-state
is faster). OR gate on `@tag timeout: 600_000` for sampling-heavy
tests.

### Gap 2: Numerical stability (8 overflows → 0)

**Problem:** BinaryBackend's f64 arithmetic overflows on wide priors
(`Normal(0, 100)`). EXLA handles this via log-space ops internally.
The ArithmeticError is `1.003e180 * -1.003e150` — intermediate
values exceed f64 range during NUTS leapfrog adaptation.

**Fix path:**

| Step | What | How | Effort |
|---|---|---|---|
| 2a | **Log-space log_prob** | Compute `log(exp(x))` as `x` (cancel), use `logsumexp` for normalizing | 1 d |
| 2b | **Clamped step sizes** | Clamp NUTS step_size adaptation to `[1e-6, 1.0]` — prevents the integrator from taking huge steps that overflow | 0.5 d |
| 2c | **f64 shader dispatch** | Use f64 shaders (GT 750M supports it) for the mass matrix accumulator path | 1 d |
| 2d | **logsumexp shader** | Fused `log(sum(exp(x - max(x))))` — numerically stable | Shader: 0.5 d, wiring: 0.5 d |

**Total effort:** ~3.5 days
**Expected outcome:** All 8 arithmetic overflows eliminated. The
models that overflow on wide priors converge cleanly.

**Alternative (faster but weaker):** Tighten priors in the failing
tests (e.g., `Normal(0, 10)` instead of `Normal(0, 100)`). This
changes test semantics — the tests are right to use wide priors
because production models do. Prefer fixing the backend.

### Gap 3: Build env (1 NIF mismatch → 0)

**Problem:** License NIF fingerprint compiled on a different
OTP/architecture doesn't match.

**Fix:** Rebuild the NIF on this machine. One command:

```sh
mix deps.compile exmc_license --force
```

**Effort:** 0 days (it's a build step, not a code change).

---

## Execution order

```
Week 1: Speed
├── Day 1-2: Persistent buffers (1a) + pipeline cache (1b)
│   → Re-run tests, measure timeout reduction
├── Day 3: Fused chain wiring (1d) + reduce_axis wiring (1e)
│   → Re-run tests
└── Day 4: Tiled matmul auto-select (1f) + assess remaining timeouts
    → If still timing out: add @tag timeout for sampling tests

Week 2: Stability
├── Day 5: Log-space log_prob (2a) + step size clamping (2b)
│   → Re-run bhm_benchmark + integration tests
├── Day 6: f64 shader dispatch (2c)
│   → Re-run all tests
└── Day 7: logsumexp shader (2d) + NIF rebuild (3)
    → Final full run: target 622/622

Week 3: Buffer
└── Day 8: Fix any stragglers, document, commit
```

**Total: ~9 days to 622/0.**

---

## Where each fix lives

| Fix | Repo | Machine |
|---|---|---|
| Persistent buffers (1a) | nx_vulkan (Rust NIF) | Linux |
| Pipeline caching (1b) | nx_vulkan (Rust NIF) | Linux |
| Fused chain wiring (1d) | nx_vulkan (Elixir) | Linux |
| Reduce_axis wiring (1e) | nx_vulkan (Elixir + Rust) | Linux |
| Tiled matmul auto-select (1f) | nx_vulkan (Elixir) | Linux |
| Log-space log_prob (2a) | exmc | Any |
| Step size clamping (2b) | exmc | Any |
| f64 dispatch (2c) | nx_vulkan (Rust) + spirit (shaders) | Both |
| logsumexp shader (2d) | spirit (shader) → nx_vulkan (wiring) | mac-248 + Linux |

**Mac-248 shader work:** logsumexp shader (2d). Everything else is
Elixir/Rust on the Linux side or exmc code changes on any machine.

---

## Decision points

**After Week 1 (speed fixes):**
- If timeouts drop from 13 to ≤3: proceed to Week 2
- If timeouts drop to 0: skip to stability, may finish in 6 days

**After Week 2 (stability fixes):**
- If overflows drop from 8 to ≤2: investigate remaining as edge cases
- If 0: done, run final full pass

**622/0 means:** exmc runs its full test suite on FreeBSD + Vulkan GPU
with zero failures. The MCMC trader works on FreeBSD without CUDA.

---

## What this does NOT include

- **Performance parity with EXLA-CUDA.** Tests pass ≠ fast. That's
  FUSION_RESEARCH.md Path A (4 more weeks after 622/0).
- **Multi-GPU.** Single device.
- **Autograd on Vulkan.** Currently falls back to BinaryBackend for
  `value_and_grad`. Works but slow. Vulkan autograd is post-622/0.
- **Training loops.** exmc doesn't train neural nets; it samples
  posteriors. Axon integration is a separate effort.

---

## Cross-references

- [EXMC_PORT_PLAN.md](EXMC_PORT_PLAN.md) — the original port plan
- [FUSION_RESEARCH.md](FUSION_RESEARCH.md) — post-622/0 performance work
- [LIMITATIONS.md](LIMITATIONS.md) — known gaps in Nx.Vulkan
- `~/spirit/PERSISTENT_BUFFERS_PLAN.md` — buffer optimization roadmap
- `~/spirit/shaders/` — all compiled shaders
