# mac-248 — Re-run exmc Vulkan suite with full Week 1 + Day 5 + Day 6 stack

**Goal**: cross-host parity check. Linux is shaking out the bug-fixes
iteratively (v3 → v10, failures roughly 174 → ~25). FreeBSD GT 750M's
prior baseline was **622/29 (95.3%)** before any of today's Week 1
work. We want a clean post-everything number on FreeBSD to compare
against.

## What's in scope since your prior 622/29 run

| Step | Where | Win |
|---|---|---|
| **Day 1a** persistent buffer pool | nx_vulkan | 1.5–1.7× per-op |
| **Day 1b** pipeline cache | already in `g_pipe_cache` | preexisting |
| **Day 1c** f32 path for all ops | nx_vulkan | 134/0 |
| **Day 1d** `Nx.Vulkan.Compiler` auto-fusion | nx_vulkan + exmc test_helper | 1 dispatch per chain |
| **Day 1e** per-axis reduce shader (single-axis) | nx_vulkan | GPU instead of host |
| **Day 1f** tiled matmul auto-select | nx_vulkan | 4.2× at d=1024 (your bench) |
| **Day 5/2a** stable inverse-softplus | exmc/lib/exmc/{compiler,log_prob,model_comparison,point_map}.ex | kills f64 overflow on wide priors |
| **Day 5/2b** clamped step sizes | exmc/lib/exmc/nuts/step_size.ex | epsilon clamped to [1e-6, 1.0] |
| **Day 6/2c** f64 elementwise GPU dispatch | spirit + nx_vulkan | f64 ops at GPU speed |
| Bug fixes | nx_vulkan | 6 plumbing classes (shape/type drift, atom-floats, list recursion, dot axes, host_via_nx opts, bitcast arity) |

## Step 1 — Sync three repos

```
cd ~/spirit
git checkout feature/vulkan-backend
git pull

cd ~/nx_vulkan
git pull       # main is at ee9ac75 (your last to_binary tweak)

cd ~/phd       # or wherever exmc lives
git pull       # main is at 499e6d685 (Day 5 fixes)
```

## Step 2 — Verify nx_vulkan tests first

```
cd ~/nx_vulkan
mix compile --force      # picks up all NIF + Elixir changes since the prior run
mix test
```

Expected: **134 tests, 0 failures**. Tests added during today's work
include f64 GPU dispatch (3 tests), bitcast (host fallback), commutative
swap in compiler, 1-arg auto-fusion, plus mac-248-side ones for the
to_binary truncate behavior.

If anything fails here, send the log — don't proceed to exmc.

## Step 3 — Re-run the full exmc suite

```
cd ~/phd/exmc
EXMC_COMPILER=vulkan mix test 2>&1 | tee ~/exmc_vulkan_w1_w5_w6.log
```

**Expected vs your 622/29 baseline:**

| Metric | Baseline (you) | After Week 1+5+6 |
|--------|---------|---------------------|
| Total tests | 622 | 622 |
| Failures | 29 | **target ≤ 15** |
| Wall time | (you didn't report — please do this time) | likely faster (auto-fusion, f64 GPU, pool) |

Linux's v10 partial breakdown (test killed at 30-min hard timeout):
- ~18 timeouts (60s/120s/300s) — these are NUTS sampling tests where
  the leapfrog is still too slow for the test budget. Same class as
  your 13 timeouts in 622/29. Day 1d (auto-fusion) and Day 6 f64 GPU
  should reduce these but won't eliminate them.
- ~5 incompatible-tensor errors — exmc-side defn closure issue.
- ~5 misc small failures (bitcast was one, fixed).

If your full-suite passes are >607 (i.e., failures < 15), we're at
the "Week 3 buffer day" stage. If timeouts dominate similar to v10,
the next investigation is whether to push on persistent-buffers
iterations (PERSISTENT_BUFFERS_PLAN.md iter 2–4) or accept higher
test timeouts.

## Step 4 — Send back

1. The summary: `N tests, M failures, K excluded`.
2. Wall time from `Finished in X seconds`.
3. Failure classification:

   ```
   grep -E "^     \*\* " ~/exmc_vulkan_w1_w5_w6.log | sed 's/^     //' \
     | awk -F'(' '{print $1 "(" $2 ")"}' \
     | sort | uniq -c | sort -rn | head -10
   ```

4. Whichever specific tests are still failing (the
   `^  [0-9]+\) test ...` lines).

## Optional stretches

If you want to keep cycling shader work:

- **f64 `reduce_axis_f64.comp`** — same recipe as today's f64
  elementwise (copy `reduce_axis.comp`, add the f64 extension, change
  `float` → `float64_t`). Helps mass-matrix Welford updates that
  currently host-fallback on f64 inputs.
- **f64 broadcast shader** — copy `elementwise_binary_broadcast.comp`,
  same f64 transformation.
- **`logsumexp.comp`** — Day 7/2d. Performance-only; Nx.logsumexp
  already composes from primitives that work. Lower priority.

None of those are blocking; the Linux side will keep fixing whatever
falls out of your re-run.

## Cross-host comparison

| Host | OS | GPU | nx_vulkan | exmc full (prior) | exmc full (post) | Wall time |
|------|----|-----|-----------|------|------|------|
| Linux | Linux 6.8 | RTX 3060 Ti | 134/0 | hung @ 60min then fixes | partial v10 | TBD |
| mac-248 | FreeBSD 15.0 | GT 750M | TBD | **622/29** | TBD | TBD |
| mac-247 | FreeBSD 15.0 | GT 650M | TBD | (skip) | (skip) | — |
