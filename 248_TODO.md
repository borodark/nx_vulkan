# mac-248 — Re-run exmc Vulkan suite to measure Week 1 gains

**Goal**: confirm Week 1's speed work moves the needle. Your previous
run was **622/29 (95.3%)**; expect timeouts to drop now that the
chain-shaped paths fuse and the alloc/free overhead is amortized.

## What changed since your last run

| # | Step | Win | Where |
|---|---|---|---|
| 1a | Persistent buffer pool | 1.5–1.7× per-op (measured) | nx_vulkan |
| 1b | Pipeline cache | already in place | nx_vulkan |
| 1c | f32 path for all ops | 124/0 nx_vulkan | nx_vulkan |
| 1d | **Auto-fusion compiler** | N elementwise dispatches → 1 fused | nx_vulkan + exmc |
| 1e | Per-axis reduce shader | single-axis on GPU | nx_vulkan |
| 1f | Tiled matmul auto-select | 4.2× at 1024×1024 (mac-248's bench) | nx_vulkan |

Auto-fusion (1d) is the big lever: every defn body in exmc now passes
through `Nx.Vulkan.Compiler` which walks the IR and replaces fusable
elementwise chains with one `fused_chain` dispatch instead of N. The
NUTS leapfrog should benefit most.

## Step 1 — Sync three repos

```
cd ~/spirit
git checkout feature/vulkan-backend
git pull

cd ~/nx_vulkan
git pull       # main is at d3a3c7b (Day 1f)

cd ~/phd       # or wherever exmc lives
git pull       # main is at e569fac6 (uses Nx.Vulkan.Compiler)
```

## Step 2 — Verify nx_vulkan tests first

```
cd ~/nx_vulkan
mix compile --force      # picks up new shaders + Rust NIF additions
mix test
```

Expected: **124 tests, 0 failures**. The 3 new from Day 1f cover
matmul auto-select; the 2 from Day 1d cover the auto-fusion compiler;
the 2 from Day 1a cover the buffer pool.

If anything fails here, **stop**. The exmc run will be unreliable.

## Step 3 — Benchmark the buffer pool (optional sanity)

```
cd ~/nx_vulkan
mix run bench/pool_bench.exs
```

Should show 1.5–1.7× speedup on chain-shaped workloads vs forced-clear.
Linux measured 97.5% pool hit rate. If your numbers differ wildly,
flag it — the GT 750M's allocator may behave differently.

## Step 4 — Run the targeted subset (sanity)

```
cd ~/phd/exmc          # or wherever
EXMC_COMPILER=vulkan mix test test/exmc_test.exs test/dist_test.exs \
                              test/diagnostics_test.exs test/compiler_test.exs
```

Previous Linux number: **63 + 11 doctests, ~6 failures = 91.9%**. With
auto-fusion, expect this to stay similar (these tests aren't NUTS-heavy)
or improve slightly.

## Step 5 — Run the FULL suite

```
cd ~/phd/exmc
EXMC_COMPILER=vulkan mix test 2>&1 | tee ~/exmc_vulkan_w1.log
```

**Compare to your prior run (622/29):**

| Metric | Last run | Expected after Week 1 |
|--------|---------|---------------------|
| Total tests | 622 | 622 |
| Failures | 29 | **15–22** (some timeouts convert to passes) |
| Wall time | unknown — please report | substantially less (auto-fusion + pool) |
| Timeouts in failure breakdown | 13 | **3–8** (chain ops fuse, leapfrog faster) |
| ArithmeticError | 8 | 8 (Week 2 fixes these — overflow on wide priors) |

If timeouts drop to ≤3, Week 1 met its target.
If timeouts drop to 0, we can skip step 1b/1c discussion in Week 2 and
   go straight to overflow fixes.
If timeouts stay at 13: investigate. Either the auto-fusion isn't
   firing in the hot path, or there's a different bottleneck. Send
   the log and I'll triage.

## Step 6 — Report

Send back:

1. `~/exmc_vulkan_w1.log` — full test output.
2. The summary line: `N tests, M failures, K excluded`.
3. Wall time from `Finished in X seconds`.
4. Failure classification:

   ```
   grep -E "^     \*\* " ~/exmc_vulkan_w1.log | sed 's/^     //' \
     | awk -F'(' '{print $1 "(" $2 ")"}' \
     | sort | uniq -c | sort -rn | head -10
   ```

5. Pool stats at end of run (curious how the pool behaves under MCMC):

   ```
   # In iex, after the mix test process exits and you're somewhere
   # with nx_vulkan loaded:
   Nx.Vulkan.init()
   Nx.Vulkan.pool_stats()
   ```

   Or just skip this — it's optional.

## Notes

- The auto-fusion compiler (1d) replaces `Nx.Defn.Evaluator` for
  `:vulkan` paths in `Exmc.JIT`. If anything is mysteriously slower
  than before, set `EXMC_COMPILER_FALLBACK=evaluator` (not yet wired,
  but easy: edit `test/test_helper.exs` to use `Nx.Defn.Evaluator`
  instead of `Nx.Vulkan.Compiler` and re-run as a control).
- If a test starts failing with `KeyError :data` or similar tracing
  errors, that's the auto-fusion compiler having trouble with a defn
  shape it doesn't expect — falls through to Evaluator on most paths,
  but if a tensor has unusual structure it can crash. Report the
  test name; the fix is mechanical.
- After your run, Linux side will compare numbers and decide whether
  to start Week 2 (overflow fixes) or close any unexpected regressions
  first.

## Cross-host comparison

| Host | OS | GPU | nx_vulkan | exmc full (prior) | exmc full (w1) | Wall time |
|------|----|-----|-----------|------|------|------|
| Linux | Linux 6.8 | RTX 3060 Ti | 124/0 | hung @ 60min | TBD | TBD |
| mac-248 | FreeBSD 15.0 | GT 750M | TBD | **622/29** | TBD | TBD |
| mac-247 | FreeBSD 15.0 | GT 650M | TBD | (skip) | (skip) | — |

Linux full-suite was hanging on host-fallback heavy paths previously;
with Week 1 Day 1d (auto-fusion) it should now also complete. I'll
re-run on Linux in parallel.
