# mac-248 — Run exmc test suite under `:vulkan`

**Goal**: cross-host validation of the EXMC port. Linux already has
117/0 on `nx_vulkan` and is timing the full `exmc` suite under
`EXMC_COMPILER=vulkan`. Get the same numbers on FreeBSD GT 750M.

Two branches contain everything needed:

| Repo | Branch | Contains |
|------|--------|----------|
| `nx_vulkan` | `feat/exmc-port-phase3` | Phase 2 + Phase 3 backfill + Path A fused_chain + Path A.2 macro + broadcast wiring |
| `phd` (parent of exmc) | `feat/exmc-vulkan-port` | Phase 2 wiring in `Exmc.JIT`, `EXMC_COMPILER` env, `mix.exs` nx_vulkan dep |
| `spirit` | `feature/vulkan-backend` | All shaders (broadcast w/compare ops, fused_elementwise w/erf/expm1, transpose, reduce_axis, cast, matmul-tuning variants) |

## Step 1 — Sync the three repos

Mac-248 layout: `~/spirit/`, `~/nx_vulkan/`, and exmc lives somewhere
under your phd checkout — the `phd` repo has it as `exmc/`. Adjust
paths if your layout differs.

```
cd ~/spirit
git checkout feature/vulkan-backend
git pull

cd ~/nx_vulkan
git fetch
git checkout feat/exmc-port-phase3
```

For exmc: if you don't already have the `phd` repo cloned, clone it
where convenient (e.g., `~/phd`). exmc is at `phd/exmc/`.

```
# If not yet cloned:
git clone <phd-repo-url> ~/phd

cd ~/phd
git fetch
git checkout feat/exmc-vulkan-port
```

## Step 2 — Verify nx_vulkan tests pass first

Three-host parity baseline before running anything bigger:

```
cd ~/nx_vulkan
mix test
```

Expected: **117 tests, 0 failures**. If this fails, stop here and
diagnose — the broadcast wiring or Path A.2 macro probably needs a
fresh `mix compile --force` after the branch switch.

## Step 3 — Set the NX_VULKAN_PATH env var

`exmc/mix.exs` references `nx_vulkan` as an optional path dep, with
`NX_VULKAN_PATH` env override. Set it once per shell:

```
export NX_VULKAN_PATH=$HOME/nx_vulkan
```

Or add to your `.profile` / `.shrc`.

## Step 4 — Resolve exmc deps

```
cd ~/phd/exmc        # or wherever exmc lives in your layout
mix deps.get
```

This should pull in `nx_vulkan` from the path dep. If it complains
about `probnik_qr` or other internal deps, you may need to
`PROBNIK_QR_PATH=...` similar to NX_VULKAN_PATH.

## Step 5 — Run the targeted subset first (sanity)

```
cd ~/phd/exmc
EXMC_COMPILER=vulkan mix test test/exmc_test.exs test/dist_test.exs \
                              test/diagnostics_test.exs test/compiler_test.exs
```

Expected on Linux: **63 tests + 11 doctests, ~6 failures = 91.9%
pass.** If your number is significantly different (>10pp), there's a
host-specific issue worth flagging before the full suite.

## Step 6 — Run the full suite

```
cd ~/phd/exmc
EXMC_COMPILER=vulkan mix test 2>&1 | tee ~/exmc_vulkan_full.log
```

**Expectations**:

- Pre-broadcast wiring, the Linux full suite **hung at 60+ minutes**
  (the host-fallback round-trip on shape-mismatched ops dominated).
- Post-broadcast wiring, much fewer host fallbacks. The suite should
  complete in a tractable time — Linux is currently running this and
  we'll have a number to compare against. Likely 10-30 minutes on a
  fast host; potentially longer on the GT 750M.
- Pass rate target: ~85% (the EXMC_PORT_PLAN projected 80%+ once
  Phase 1 + 2 + Phase 3 broadcast landed).

If the suite hangs past 60 minutes again, kill it (`Ctrl-C`) and
report the test name it was stuck on. That points to whichever
operator is still hitting host-fallback at scale.

## Step 7 — Report numbers

Once `mix test` finishes, send back:

1. The summary line: `N tests, M failures, K excluded`.
2. Wall time from `Finished in X seconds`.
3. The `~/exmc_vulkan_full.log` file (or just the failure
   classifications via `grep -E "^     \*\* "
   ~/exmc_vulkan_full.log | sort | uniq -c | sort -rn | head -10`).

Three-host comparison:

| Host | OS | GPU | nx_vulkan | exmc full | Wall time |
|------|-----|-----|---|---|---|
| Linux | Linux 6.8 | RTX 3060 Ti | 117/0 | TBD (running now) | TBD |
| mac-248 | FreeBSD 15.0 | GT 750M | TBD | TBD | TBD |
| mac-247 | FreeBSD 15.0 | GT 650M | TBD | (skip — same results expected) | — |

## Notes

- The `propcheck` property tests inside exmc's test suite can take a
  long time per test under Vulkan's per-op dispatch model. If a
  property test specifically times out, it's not necessarily wrong —
  it's the same number of statistical iterations running through the
  GPU instead of CPU.
- If you see `:size_mismatch` failures, that means a broadcast case
  the shader doesn't yet handle (rank > 4, or non-broadcastable
  shapes). Capture the test name; the missing case is documented in
  `LIMITATIONS.md` §2.
- If the BEAM crashes (segfault, DEVICE_LOST), that's a different
  class of bug worth reporting immediately — the SUBMIT_LOCK in
  Rust is meant to prevent it but mass concurrent dispatch under
  property tests is a stress case we haven't fully exercised.

## After your run completes

Linux side will:

1. Compare Linux vs mac-248 numbers (looking for cross-host
   regressions, not just absolute pass rate).
2. If both finish: update `LIMITATIONS.md` §6 with real numbers.
3. If mac-248 surfaces failures Linux didn't, triage the first one
   together.
4. Decide whether the next priority is matmul-variant selection
   (route through tiled-32 or tiled-16x2 vs naive matmul) or the
   long tail of remaining missing-op callbacks.
