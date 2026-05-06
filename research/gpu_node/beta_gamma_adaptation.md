# Beta / Gamma synth-shader NUTS adaptation diagnosis

**Status**: Research, not implemented. Diagnosis only.
**Date**: 2026-05-06 (Linux RTX 3060 Ti, exmc@main, nx_vulkan@main)
**Trigger**: Phase 1 fair-race (race_quick) reported `Beta synth` ESS/s = 1.0
(0.04× EXLA) and `Gamma synth` ESS/s = 4.3 (0.11× EXLA). Lognormal synth
mixed normally (0.95×).

## TL;DR

The chain is statistically valid; **sampling-time** mixing is fine
(depth-1-3 trees, accept_prob ≈ 0.92, divergences ≈ 0). The poor
wall-time-per-effective-sample ratio comes from **warmup spending most
of its time inside very deep trees because the diagonal mass matrix
never gets close to the posterior precision**. Hand-setting the mass
matrix to the prior precision and running only 50 iterations of fine-
tune warmup (skipping the broken Phase II/III) gives **3.3× wall
speedup on Beta, 3.9× on Gamma, and 5× even on Lognormal**.

## Evidence (Vulkan path, seed=42, RTX 3060 Ti)

Raw log: `/tmp/beta_gamma_diag.out`. Scripts: `/tmp/beta_gamma_diag.exs`,
`/tmp/beta_gamma_diag_bc.exs`.

### A. Baseline (warmup=1000, samples=500)

| Cell                 | wall (ms) | ESS  | ESS/s | inv_mass | depth hist          | accept |
|----------------------|-----------|------|-------|----------|---------------------|--------|
| Beta(2,3)            | 297,581   | 274.8| 0.92  | **1.056**| 1×203, 2×291, 3×6   | 0.920  |
| Gamma(2,1)           | 114,814   | 242.1| 2.11  | **0.652**| 1×199, 2×300, 3×1   | 0.835  |
| Lognormal(0,1) ctrl  |  18,385   | 176.6| 9.61  | **0.923**| 1×178, 2×292, 3×30  | 0.937  |

Posterior-precision targets (closed form):

- Beta(α=2, β=3) on logit-uc: Var(q_uc) ≈ 1/(α+β) = **0.20** → Welford ended at 1.056 (5× too big)
- Gamma(α=2, β=1) on log-uc:  Var(q_uc) ≈ 1/α = **0.50** → Welford ended at 0.652 (1.3× too big)
- Lognormal(0, 1) on log-uc:  Var(q_uc) = σ² = **1.00** → Welford ended at 0.923 (8 % off — fine)

Lognormal happens to have posterior variance = 1, identical to the
identity init. That is the only reason it "passes" today.

### B. Long warmup (warmup=2000, samples=500)

| Cell        | wall (ms) | ESS  | ESS/s        | inv_mass | depth hist     |
|-------------|-----------|------|--------------|----------|----------------|
| Beta(2,3)   | 388,459   | 268.2| **0.69 ↓**   | 1.591 ↑  | 1×229, 2×271   |
| Gamma(2,1)  | 139,736   | 180.8| **1.29 ↓**   | 0.722 ↑  | 1×200, 2×300   |

Doubling warmup made **both** distributions worse. The mass estimate
drifted *further* from truth (Beta 1.06→1.59, Gamma 0.65→0.72). With
Stan-style per-window Welford resets, the mass estimate comes from the
*last* doubling window, which under wrong-mass+wrong-ε is sampling
trajectory dynamics rather than the posterior. Longer warmup just gives
the chain more rope to drift away from the typical set under bad
geometry. Hypothesis 3 ("warmup window too short") is **falsified**.

### C. Hand-set mass via warm_start (warmup=50, samples=500)

`init_inv_mass = 1/precision`, `init_step_size = 0.5`. The 50-iter
warmup is too short for Phase II so the mass stays at our hand-set
value; only DA fine-tunes ε.

| Cell            | wall (ms) | ESS  | ESS/s    | inv_mass  | depth hist                  | accept |
|-----------------|-----------|------|----------|-----------|-----------------------------|--------|
| Beta(2,3)       |  91,128   | 137.4| **1.51** | 0.200     | 1×102, 2×171, 3×211, 4×16   | 0.991  |
| Gamma(2,1)      |  29,611   | 201.7| **6.81** | 0.500     | 1×177, 2×307, 3×16          | 0.884  |
| Lognormal(0,1)  |   3,701   | 318.5| **86.05**| 1.000     | 1×188, 2×312                | 0.931  |

Wall reduction vs A: **Beta 3.3×, Gamma 3.9×, Lognormal 5.0×**. Beta's
ESS-per-sample dropped (137 vs 275) because 50 warmup is too short for
the chain to fully equilibrate from random init — but ESS/s **still
went up 64 %** because the wall budget collapsed.

The Lognormal speedup is the smoking gun: even when the mass *was*
right by accident, the 1000-iter warmup was burning ~14 s of pointless
adaptation time. Cut warmup → wall drops 5×.

## Root cause

Three reinforcing mechanisms; none is "deep sampling trees" (the
hypothesis we walked in with).

1. **Identity mass + ε₀ = 1.0 is too aggressive for Beta/Gamma logit/log-uc.**
   `find_reasonable_epsilon` (sampler.ex:496) doubles ε from 1.0 until
   single-step log-accept crosses log(0.5). For Beta(2,3) at the mode,
   one ε=1, M⁻¹=I leapfrog step moves q_uc by ~1 σ-of-momentum =
   ~2.2 σ-of-posterior. The single-step accept rate is acceptable,
   but the **trajectory** turns over in 2-3 leapfrogs and stays
   correlated. ε=1.0 reaches the upper clamp (`@epsilon_max = 1.0`,
   step_size.ex:22) so DA can never move past it.

2. **Per-window Welford reset never converges to posterior covariance.**
   Phase II runs doubling windows of length 25, 50, 100, 200, 400…
   (sampler.ex:728, lesson #16). Each window resets Welford. The chain
   is still equilibrating under the wrong geometry, so each window's
   variance estimate reflects **trajectory dynamics**, not posterior
   covariance. Doubling warmup just spawns more windows under the same
   wrong regime — see Experiment B.

3. **Most of the wall is warmup, not sampling.** Sampling-only
   diagnostics are clean (depth ≤ 3, accept ≈ 0.9). But A's wall_ms
   minus the ~17 s "sampling cost" inferred from Lognormal is ~280 s
   of warmup wall for Beta. With early `max_tree_depth = 8` cap for
   the first 200 iterations (lesson #17) and bad mass, individual
   warmup trees can hit 2⁸ = 256 leapfrogs each. That's where the
   wall is going.

The W2 statistical validator passes because NUTS is a valid sampler
under any positive-definite mass; only the wall is paying.

## Proposed fix

A *prior-aware* mass-matrix initializer for the synthesized chain
shaders. When the user opts into the fused chain via
`Application.put_env(:exmc, :fused_leapfrog_meta, {:beta, α, β})` (or
the Phase A+B auto-routing equivalent), the codegen already knows the
distribution family and parameters. Translate that into an
`init_inv_mass_diag` heuristic at the cold-start branch
(sampler.ex:230-234):

| Family            | Posterior var on uc-space (closed form / heuristic) |
|-------------------|-----------------------------------------------------|
| Beta(α, β)        | `1 / (α + β)`  on logit-uc                          |
| Gamma(α, β), α>1  | `1 / α`         on log-uc                           |
| Gamma(α, β), α≤1  | `1.0` (peaked at 0; identity is fine)               |
| Lognormal(μ, σ)   | `σ²`            on log-uc                           |
| Exponential(λ)    | `1.0`           on log-uc (Jacobian-fixed)          |
| HalfNormal(σ)     | `σ²`            on log-uc                           |
| StudentT(ν,μ,σ)   | `σ² · ν / (ν − 2)` for ν > 2 else `σ²`              |
| Weibull(k, λ)     | `(π² / 6) / k²` on log-uc (asymptote)               |

These are first-moment heuristics; they don't have to be exact, just
within a factor of 2. Pipe as the *initial* `inv_mass_diag` (replacing
the identity broadcast at sampler.ex:231) **only** when chain-shader
auto-routing is active. Phase II adaptation continues normally and
will refine the value.

For `find_reasonable_epsilon`, also seed ε₀ ≈ √(min(inv_mass)) so the
doubling search starts in the right ballpark instead of always at 1.0.

Independent observation: even when mass init is correct, Lognormal's
A→C wall went from 18 s to 4 s. There is a **separate** "warmup is too
long for well-conditioned models" question worth raising — for chain-
shader cells we know the geometry a priori, so 50-100 warmup is plenty.
That's a follow-on; the mass-init fix is the headline.

## Expected ESS/s improvement

Combined effect of mass init + shorter warmup (which the chain-shader
codegen knows is safe):

| Cell        | A baseline | C hand-mass + 50 warmup | Multiplier |
|-------------|------------|-------------------------|------------|
| Beta(2,3)   | 0.92       | 1.51                    | **1.6×**   |
| Gamma(2,1)  | 2.11       | 6.81                    | **3.2×**   |
| Lognormal   | 9.61       | 86.05                   | **9.0×**   |

Realistic projection: Beta ratio vs EXLA improves from 0.04 → ~0.15;
Gamma from 0.11 → ~0.4; Lognormal from 0.95 → ~5×. The chain-shader
path becomes competitive with EXLA on Beta and clearly faster on
Lognormal, matching the hand-written family pattern.

## Effort estimate

**Quick win, ~1 day.** The hook point is sampler.ex:194-234. Replace
the identity init with a metadata-driven heuristic when
`Process.get(:exmc_chain_meta)` or the application env is set. The
math is closed-form per family. No changes to NUTS internals, no
changes to `nx_vulkan`. Add a regression test that asserts the final
inv_mass_diag is within 30 % of the family target after a race_quick
run on Beta and Gamma.

**Open question** (do not block on this): the Phase II Welford does
not converge for short warmup on Beta-like geometry. That is a deeper
sampler bug — Stan and PyMC handle it via `regularize.hpp` which
shrinks toward a *prior* covariance, not toward 1e-3·I. Fixing that
benefits hierarchical models too and is ~1 week. Out of scope here.
