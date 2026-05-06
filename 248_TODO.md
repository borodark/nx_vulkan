# mac-248 — R1: replay the fair race on FreeBSD (**READY — multivariate IR fix landed**)

> **Status update 2026-05-04**: R1 is now **READY**. The blocking
> multivariate-IR bug was a missing `Nx.sum` on free-RV per-element
> logp tensors — fixed on `pymc/main` at `3b17d8e40`. Linux race
> re-collected Normal d=8 + d=50; predicted scaling confirmed:
> Vulkan crossover happens around d=20-30, Vulkan **wins at d=50
> with ratio 1.45×**. Race subjects: 7 cells (Normal at d=1/8/50,
> Exponential, StudentT, HalfNormal, Weibull). See
> `~/projects/learn_erl/pymc/exmc/bench/fair_race_results_linux.md`
> for the Linux numbers to compare against.
>
> One additional ask landed since: the Linux side just shipped
> Phases A+B of "auto-route to chain shader from IR" on branch
> `pymc/feat/dsl-shader-codegen` — `Sampler.sample/3` now
> auto-engages the chain shader for any of the 6 supported
> single-RV families with no `Application.put_env` hack required.
> The race script doesn't depend on this (it sets the env
> explicitly), but the auto-route is what makes the chain shader
> useful in real usage.

## Linux race result (your reference)

| Model         |  d | EXLA wall (ms) | Vulkan wall (ms) | EXLA ESS/s | Vulkan ESS/s | ratio |
|---------------|----|----------------|------------------|------------|--------------|-------|
| Normal        |  1 |          7,884 |           32,260 |       54.3 |         13.3 |  0.24 |
| Normal        |  8 |          2,029 |            2,749 |       32.5 |         24.0 |  0.74 |
| Normal        | 50 |          3,554 |            4,329 |        6.4 |          9.3 | **1.45** |
| Exponential   |  1 |         15,740 |           42,691 |       30.6 |         13.8 |  0.45 |
| StudentT df=3 |  1 |         15,606 |           36,057 |       12.0 |          6.7 |  0.56 |
| HalfNormal    |  1 |         16,238 |           55,235 |       28.0 |          4.7 |  0.17 |
| Weibull k=2   |  1 |         15,749 |           40,570 |       25.4 |          9.8 |  0.39 |

(d=1 cells from full 1000/1000; d=8/d=50 from quick race
post-fix at 100/100. Full d=8/d=50 numbers will be re-collected
on Linux and pushed to the same file before you start; pull then.)

Crossover: somewhere around d=20-30 on Linux RTX 3060 Ti. The
chain shader's per-thread parallelism scales linearly with `n`;
EXLA's per-call CUDA overhead doesn't.

Your job: replay this race on FreeBSD GT 750M and report whether
the FreeBSD column matches Linux ratios within run-to-run
variance, or shows substrate-specific divergence.

---

## Recent state — Y4 closed

- ✅ Y4 `leapfrog_chain_weibull.spv` shipped (`spirit/67531dca`),
  vendored, wired, smoke-tested on Linux. Math agrees to f32
  epsilon (q[0,0]=0.05, p[0,0]=0.4895, logp[0]=-1.2481 at K=1).
  Linux side at `nx_vulkan/56fc1cc`, eXMC dispatch + new
  `:requires_vulkan` chain test on `pymc/8d265a2eb`.
- All 6 chain shaders complete: Normal, Exponential, Student-t,
  Cauchy, HalfNormal, Weibull. Plus `_lg` and f64 Normal siblings,
  `reduce_full_f64`, canonical Random123 Philox.

The original hierarchical Weibull test stays `:vulkan_known_failure`
because it's multi-RV with observed data — chain dispatch can't
engage there. That's a separate problem class (codegen op
coverage on `feat/vulkan-codegen` or chain-shader generalization
to multi-RV models). Not on your queue.

---

## R1 — Race on FreeBSD GT 750M

A fair race ran on the Linux RTX 3060 Ti (2026-05-04) — Vulkan
fused chain vs EXLA reference across 7 single-distribution
models. The natural sequel: re-run the same race on mac-248's
FreeBSD GT 750M to measure how the chain shader scales across
GPU generations and OS substrates.

This isn't a "is Vulkan fast" benchmark — that's already
answered on Linux. The FreeBSD race tests **portability of the
speedup ratio**: does the fused chain win on FreeBSD/Kepler
the same way it wins on Linux/Ampere, or is something
substrate-specific eating the gain?

### Backend baseline — pick whichever applies

| Option | When |
|--------|------|
| **A. EXLA host client** (CPU-only, no CUDA on FreeBSD) | If EXLA builds on FreeBSD via `mix deps.compile exla` |
| **B. Vulkan unfused** (per-op IR walker) | If EXLA doesn't build — same Vulkan substrate, different dispatch strategy |
| **C. BinaryBackend** (pure Elixir) | Last resort — informational only, not a real comparison |

Try Option A first (`cd ~/projects/learn_erl/pymc/exmc && mix deps.compile exla 2>&1 | tail -5`). If it builds clean, you have an EXLA CPU baseline; if it errors out within 30 seconds, fall back to Option B. Don't sink hours into making EXLA build on FreeBSD — that's a separate engineering project.

### Subjects (same as the Linux race)

| # | Model | d |
|---|-------|---|
| 1 | `x ~ Normal(0, 1)` | 1 |
| 2 | `x ~ Normal(0, 1)` | 8 |
| 3 | `x ~ Normal(0, 1)` | 50 |
| 4 | `x ~ Exponential(2)` | 1 |
| 5 | `x ~ StudentT(df=3, ...)` | 1 |
| 6 | `x ~ HalfNormal(σ=1)` | 1 |
| 7 | `x ~ Weibull(k=2, λ=1)` | 1 |

7 subjects. Cauchy excluded (variance undefined → ESS noisy).

### Protocol

| Knob | Value |
|------|-------|
| `num_warmup` | 1000 |
| `num_samples` | 1000 |
| seeds per cell | 5 (medians + IQR) |
| chain count | 1 |

5 seeds × 7 models × 2 backends = **70 runs total**.

### Runtime expectation

GT 750M is Kepler-class (2013), substantially slower than the
Ampere RTX 3060 Ti for compute. Expect Vulkan-fused per-step
to be ~3-5× slower in absolute terms. Total wall-clock for
the full race: budget **2-3 hours**. If a single cell exceeds
10 minutes (Option B with K=32 at d=50 might), abort that cell
and report partial.

### Implementation

A Mix script lives at the dev box at `/tmp/fair_race.exs`. It
hasn't been written yet — Linux side will write it as part of
the Linux race execution and push to a shared location. For
now, here's the schematic per cell:

```elixir
# Per (model, seed, backend):
#   1. Build the IR
#   2. Set the right Application env vars for the backend
#   3. Set fused_leapfrog_meta if the backend is Vulkan + cell is single-RV
#   4. {trace, stats} = Sampler.sample(ir, %{}, num_warmup: 1000, num_samples: 1000, seed: seed)
#   5. Compute ESS via Exmc.Diagnostics.ess (per-parameter min if d > 1)
#   6. Record: wall_ms, ess_min, mean, var, divergences
# Report median across the 5 seeds per (model, backend) cell.
```

The `fused_leapfrog_meta` shape per cell:

| Cell | Meta |
|------|------|
| Normal d=1/8/50 | `{:normal, 0.0, 1.0}` |
| Exponential | `{:exponential, 2.0}` |
| StudentT | `{:studentt, 0.0, 1.0, 3.0, logp_const_t}` |
| HalfNormal | `{:halfnormal, 1.0, log_const_h}` |
| Weibull | `{:weibull, 2.0, 1.0, n*(log(2) - 2*log(1)) = n*0.6931}` |

### Output format

Single Markdown table. Add a column for the Linux numbers (from
the prior race) so the FreeBSD column reads in context:

```
| Model     | d  | Linux Vk ms | FreeBSD Vk ms | Linux ratio | FreeBSD ratio | substrate Δ |
|-----------|----|-------------|---------------|-------------|---------------|-------------|
| Normal    |  1 | …           | …             | …           | …             | …           |
…
```

Substrate Δ = (FreeBSD ratio / Linux ratio). If close to 1.0,
the speedup is portable. If ≠ 1.0, something substrate-specific
is at play.

Plus a one-line headline: "GT 750M Vulkan fused chain on
FreeBSD: matches/beats/lags Linux ratio on N/7 cells."

### What this race answers

- **Is the chain shader's win portable across GPUs?** If FreeBSD
  ratios match Linux ratios, the win is fundamental to the
  per-dispatch-amortization architecture, not a property of any
  particular driver or hardware generation.
- **Does FreeBSD's nvidia driver have any Vulkan-overhead quirks
  vs Linux's?** A divergence in ratios surfaces this.
- **Validates the walkable-path post's cross-platform claim
  with real measurements.** Not just "runs" but "runs at a
  predictable speed."

### What it does NOT answer

- **Multi-chain scaling on FreeBSD** — single chain only here.
- **Hierarchical models** — chain dispatch doesn't engage; out
  of scope.
- **Long-chain (5000+ samples) behavior** — 1000+1000 budget.

### Risks (read carefully before starting)

1. **EXLA does not build on FreeBSD.** XLA needs Bazel + a
   heavy C++ build chain that has never been first-class on
   FreeBSD. `mix deps.compile exla` will likely fail within
   30-60 seconds with a Bazel-related error. **What to do**:
   note the exact error in your follow-up commit (one paragraph
   is enough — don't try to debug the XLA build), fall back
   to Option B (Vulkan unfused) for the baseline. If even
   `deps.get` fails for EXLA on FreeBSD (lockfile mismatch),
   skip EXLA in mix.exs locally for the duration of the race
   (`{:exla, "..."` line commented out), set
   `EXMC_COMPILER=vulkan`, and run BOTH the fused and unfused
   Vulkan paths against each other.

2. **GT 750M VRAM is 1-2 GB; chain output buffers can grow.**
   At the largest race cell (Normal d=50, K=32 batch in
   speculative path), each chain dispatch allocates 4 output
   buffers of `K × n × 4 = 6.4 KB` each — trivial. Even
   pessimistic with d=50, K=128 (extension batch): 25.6 KB ×
   4 = 100 KB per dispatch. Should not pressure VRAM. **If
   you see `VK_ERROR_OUT_OF_DEVICE_MEMORY`** during a cell:
   confirm with `nvidia-smi` (FreeBSD nvidia tools) that the
   compositor or X server isn't reserving most of the VRAM,
   then reduce `num_warmup` to 500 for that cell.

3. **f32 numerical drift over long chains may diverge between
   Linux and FreeBSD even at the same seed.** Both backends are
   f32, but different GPU silicon executes f32 arithmetic with
   slightly different fused-multiply-add behavior, denormal
   handling, etc. Over 1000 warmup × ~16 leapfrog steps =
   16,000 f32 leapfrogs, drift accumulates. **Expected:**
   posterior moments (mean, var) match across hosts within
   MCMC noise (|Δm| < 0.3, |Δv| < 0.5). **What's NOT
   expected:** bit-identical traces. Don't compare per-sample
   trajectories; compare the Markov-chain summary statistics.
   If a cell's mean/var diverges past those bounds, that's a
   real finding worth flagging in the report — could be a
   FreeBSD-specific f32 quirk worth investigating.

4. **Per-cell wall-clock blow-up.** A bad cell can take
   hours under Option B (unfused Vulkan, which is the path
   the chain shaders exist to fix). The Normal d=50 cell
   under Option B specifically risks 30+ minutes per seed ×
   5 seeds = 2.5 hours for that single cell. **Mitigation**:
   per-cell hard timeout of 10 minutes (Mix script enforces
   via `Task.async_stream` with `:timeout`). On timeout, the
   cell records `:dnf` instead of numbers. Race continues to
   the next cell. Report any DNFs explicitly; don't pretend
   they didn't happen.

5. **Compositor / X server interference.** If mac-248 has X
   running and a compositor, the GT 750M is also rendering
   the desktop, which contends with compute dispatches. Most
   FreeBSD workstation configs leave the GPU mostly idle for
   compute, but if you see wildly variable per-seed wall
   times (e.g., one seed 5s, the next 30s), this is the
   first thing to suspect. **Mitigation**: run the race
   from a TTY (Ctrl+Alt+F1) or after stopping the X server
   for the duration. Document the answer in the report —
   "X stopped" or "X running, using GPU N% per nvidia-smi".

6. **Tagged meta typos.** The fused_leapfrog_meta shapes are
   distribution-specific. If a meta's tag doesn't match a
   `do_dispatch` clause, the race silently falls through to
   the unfused Vulkan path — and the cell measures the WRONG
   thing without raising. **Mitigation**: enable
   `Application.put_env(:exmc, :fused_dispatch_debug, true)`
   if the diagnostics module supports it (TBD), or add a
   single `IO.inspect` at the top of `do_dispatch/10` for the
   duration of the race so you can confirm each cell hit the
   right clause. Remove before commit.

7. **GLSL not installed on FreeBSD post-bring-up.** The chain
   shaders are pre-compiled SPIR-V vendored in
   `nx_vulkan/priv/shaders/` — you don't need glslang at
   race time. But if the codegen branch is somehow active
   (`feat/vulkan-codegen` checked out instead of `main`), it
   tries to JIT-compile GLSL via `glslangValidator`. **Mitigation**:
   confirm `git -C ~/nx_vulkan branch --show-current` returns
   `main`, not `feat/vulkan-codegen`, before starting the race.

8. **Result reproducibility.** The race uses `seed: seed` for
   each (model, seed) pair. The Erlang `:rand` PRNG is
   deterministic given a seed, but if mac-248 has been
   running other Erlang processes that consumed entropy from
   the same global state, results may differ slightly across
   runs. **Mitigation**: each race iteration calls
   `:rand.seed_s(:exsss, {seed, ...})` explicitly via
   `Sampler.sample(seed: seed)` — no shared state.

### What does count as a successful race

- All 7 cells complete without `:dnf` under the hard
  per-cell timeout.
- All 7 cells' posterior summaries (mean, var) match within
  the MCMC-noise tolerance described in risk #3.
- A clean ratio table (FreeBSD vs Linux) — even if the
  ratio is 0.5 or 2.0, that's a real measurement worth
  publishing. Only "I couldn't get cell N to run" is a
  failure outcome.

### Coordination with Linux side

I (Linux dev box) will run the same race in parallel. Once both
results are in, the combined table goes into the *walkable-path*
blog as the cross-platform measurement that makes the original
post's "runs on FreeBSD via Vulkan" claim quantitative.

If the Mix script ends up shared (it should — same race, same
Elixir), I'll push it to `~/projects/learn_erl/pymc/exmc/bench/fair_race.exs`
once the Linux side is done. Pull, run, report.

## What this TODO is NOT

- Not asking for shader changes — your 6 chain shaders are
  done.
- Not asking for FreeBSD-specific fixes — the FreeBSD bring-up
  already validated everything builds + runs. This is just
  measurement.

## Cross-reference

- `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md` —
  full chain-shader history; Phase 2 done.
- `~/projects/learn_erl/pymc/www.dataalienist.com/blog-walkable-path.html`
  — the post the cross-platform measurement updates.
- Linux race results (when ready): `~/projects/learn_erl/pymc/exmc/bench/fair_race_linux.csv`
- Send your FreeBSD CSV to the same dir (`fair_race_freebsd.csv`)
  via the same nas path the project uses. Or paste the table
  inline in a 248 follow-up commit.

---

## R2 (2026-05-04) — H1 confound test on Linux side, optional sibling on FreeBSD

### What just happened on Linux

Re-reading the R1 race table above, the Linux Vulkan numbers are
catastrophically bad next to your FreeBSD GT 750M numbers from
`FAIR_RACE_FREEBSD.md` (1.4-1.9 s wall on the cells where Linux
Vulkan sits at 32-55 s). A 20-30× gap in favor of an older,
weaker GPU on a slower platform is physically impossible without
an external confound. We ranked five hypotheses; H1 was **live
trial GPU contention** — pid `82641` on `super-io` was a 15-day
uptime trading-trial BEAM node holding CUDA context with 67
instruments running periodic NUTS samplers and burning 31% CPU.

R2 on the Linux side, in progress now:

1. Killed pid 82641 (the trial) — confirmed dead, GPU compute
   apps clear, 6.2 GB free.
2. Re-running `RACE_QUICK=1 mix run bench/fair_race.exs`
   against the cleaned GPU.
3. Comparing the new wall_ms numbers against:
   - The R1 Linux table above (32-55 s on Vulkan)
   - Your `FAIR_RACE_FREEBSD.md` numbers (1.4-1.9 s)

If Linux Vulkan now lands near FreeBSD Vulkan, **trial
contention was the entire gap**. If it stays slow, the trial
was a red herring and we keep walking down the list (H2 warmup
vs sample split, H3 Evaluator fallback, H4 BEAM GC, H5 pipeline
cache).

The post-trial-kill race results land in
`~/projects/learn_erl/pymc/exmc/bench/fair_race_results_linux.md`
under a new section "R2 — post-trial-kill". I will commit that
file with a note pointing back to this TODO so you can
git-pull and read.

### What this asks of you (optional but useful)

The clean-run on Linux is the primary measurement. FreeBSD has
no equivalent confound to remove (no live trial, no 15-day
BEAM hog), so you do not need to re-do R1.

But there is one **optional sibling experiment** on FreeBSD that
would harden the H1 conclusion either way:

> Spin up a small artificial load on the GT 750M while running
> one Vulkan cell of R1 (whichever cell ran fastest in your
> baseline — probably `Normal d=1`). Measure wall_ms with the
> load present, divide by your baseline wall_ms, and report the
> contention multiplier.
>
> If the contention multiplier on FreeBSD GT 750M is also ~10-20×,
> we know the gap is *contention sensitivity* (a property of the
> driver / scheduler) and not Linux-specific. If FreeBSD gets a
> ~2× contention slowdown, the gap is Linux-driver-specific and
> H1 only partially explains things.

### Suggested artificial load on FreeBSD

Anything that holds a Vulkan queue submission active is fine.
The simplest: a separate `mix run` on the same machine that
loops `Sampler.sample/3` on `Normal d=1` with the chain shader
in the background. Or `vkmark` / `vkcube` if you have it
installed. The point is to reproduce the *shape* of what the
Linux trial was doing — repeated GPU submissions over a long
elapsed time — not the magnitude.

If you don't want to build artificial load, **skip this**. The
Linux R2 measurement is sufficient on its own.

### What R2 explicitly does NOT ask

- No FreeBSD-side code changes.
- No R1 re-run if the GT 750M numbers haven't shifted.
- No trial of your own — the trial was a Linux-side artifact.

### Coordination

I will land R2 results on Linux first. If the Vulkan numbers
drop to ~2-3 s wall (FreeBSD-class), R2 is closed and the gap
is fully explained. If they only drop to ~10-15 s, we have a
partial explanation and H2-H5 stay open.

Either way, the answer goes back into the *walkable-path* blog
as the cross-platform measurement that R1 set up.

---

## R3 (2026-05-04 evening) — H1 ruled out, H2 split done, persistent-buffer fix landed

### R2 outcome: H1 (live trial GPU contention) is dead

Killed pid 82641. Re-ran the same 1000/1000 Normal d=1 Vulkan
cell on cleaned GPU. 5-seed median:

- pre-trial-kill:  32,260 ms
- post-trial-kill: 30,404 ms

5.7% delta — within seed noise. The 18× gap is *not* the trial.

### H2 split: cost is per-dispatch steady-state, not warmup or compile

Same Normal d=1 cell, 5-seed median, split via two runs (full +
warmup-only):

- full   (1000W + 1000S): 34,879 ms,  2,614 dispatches → 13.34 ms/disp
- warmup-only (1000W):    20,619 ms,  1,313 dispatches → 15.70 ms/disp
- sample-only (delta):    14,260 ms,  1,297 dispatches → 10.99 ms/disp

Warmup share 59%, sample share 41% — essentially proportional
to dispatch count. Not a JIT/compile spike, not warmup-specific.
Steady-state per-dispatch overhead, ~13 ms on Linux RTX 3060 Ti
vs ~0.6 ms on your FreeBSD GT 750M (reading from FAIR_RACE_FREEBSD.md).

### H3 instrumentation: per-fence wait latency is the floor

Added atomic counters around vkQueueSubmit, vkWaitForFences,
and command-buffer recording in spirit's Backend_par_vulkan.cpp.
Exposed via `Nx.Vulkan.Native.timing_get/0` (returns
`{count, dispatch_ns, submit_ns, wait_ns, record_ns}`).

Linux NVIDIA driver per submit_and_wait:
- submit:  ~138 µs
- wait:   ~1130 µs   ← 8× the submit cost
- record:   ~19 µs

The wait is the blocking fence wait. ~1 ms per fence is the
hardware/driver floor on this stack. No host-side batching can
go below that for a single round-trip.

What host-side batching CAN do is cut the *number* of round-trips.

### Root cause: 8 round-trips per chain dispatch

Each chain shader call in `pymc/exmc/lib/exmc/nuts/tree.ex`
was doing:

  vulkan_upload(q)              → alloc + upload + submit_and_wait  [1]
  vulkan_upload(p)              → alloc + upload + submit_and_wait  [2]
  vulkan_upload(inv_mass)       → alloc + upload + submit_and_wait  [3]
  Nx.Vulkan.leapfrog_chain_*    → submit_and_wait                   [4]
  vulkan_to_tensor(q_chain)     → cmd_copy + submit_and_wait        [5]
  vulkan_to_tensor(p_chain)     → cmd_copy + submit_and_wait        [6]
  vulkan_to_tensor(grad_chain)  → cmd_copy + submit_and_wait        [7]
  vulkan_to_tensor(logp_chain)  → cmd_copy + submit_and_wait        [8]

8 fences × 1.27 ms = ~10.2 ms/dispatch on Linux. Matches the
H2 measurement almost exactly.

### Fix landed (this session)

Added to `nx_vulkan` (commit `af15284` on `main`):

- `nxv_buf_upload_batch` / `nxv_buf_download_batch` C primitives:
  pack N source pointers into one staging buffer + one cmd buffer
  + one submit_and_wait.
- Rust NIFs: `upload_binary_into` (in-place upload, no alloc),
  `upload_binary_into_batch2` (q+p batched), `download_binary_batch4`
  (4-buffer batched download).
- Elixir bindings in `Nx.Vulkan.Native`.

Applied to all 6 chain-shader families in `exmc/lib/exmc/nuts/tree.ex`
(commit `fbad3e1c8` on `feat/dsl-shader-codegen`):

- Persistent q/p/inv_mass GPU buffers cached in process dict,
  keyed by dimension. Reused across all chain dispatches in one
  sampler run.
- q+p uploaded in one batched call; inv_mass uploaded only when
  content actually changed (rare — only at warmup window boundaries).
- Chain output: 4 downloads → 1 batched download.

8 round-trips → 3-4 round-trips per dispatch.

### Post-fix Linux numbers (RACE_QUICK 100/100, post-trial-kill)

| Cell           | Pre-fix Vulkan | Post-fix Vulkan | Speedup | EXLA→Vulkan ratio |
|----------------|----------------|-----------------|---------|-------------------|
| Normal d=1     |       2,751 ms |       1,311 ms  |  2.10x  |              0.66 |
| Normal d=8     |       3,682 ms |       1,399 ms  |  2.63x  |          **1.22** |
| Normal d=50    |       3,637 ms |       1,698 ms  |  2.14x  |          **3.17** |
| Exponential    |       3,158 ms |       1,893 ms  |  1.67x  |              0.84 |
| StudentT df=3  |       2,578 ms |       1,342 ms  |  1.92x  |          **1.04** |
| HalfNormal     |       3,463 ms |       2,031 ms  |  1.71x  |              0.49 |
| Weibull k=2    |       2,941 ms |       1,807 ms  |  1.65x  |              0.91 |

**Average ~2× wall-time speedup. Vulkan now beats EXLA on
4 of 7 cells (was 1 of 7 pre-fix).**

### Comparison vs FreeBSD GT 750M

Approximate, scaled for iteration count:

- FreeBSD GT 750M, Normal d=1, 1000/1000:  1,651 ms (your R1 number)
- Linux RTX 3060 Ti, Normal d=1, 1000/1000 post-fix: ~13,000 ms
  (extrapolated; full 1000/1000 race is in flight as I write this
   and will be appended to bench/fair_race_results_linux.md)

Linux is now ~8-10× slower than FreeBSD. Down from 18× pre-fix.

### What this asks of you (R3)

Three things, in order of value:

#### R3.1 — Re-run R1 on FreeBSD GT 750M with the new code (high value)

Pull `nx_vulkan` main (commit `af15284`) and `pymc` branch
`feat/dsl-shader-codegen` (commit `fbad3e1c8`). Re-run the
fair race (`mix run bench/fair_race.exs`, full 1000/1000).

Expected outcome: FreeBSD numbers should improve marginally
(maybe 10-20%) because mesa's per-fence latency was already
low. The persistent-buffer fix mostly helps Linux, where the
8-fence round-trip cost was the dominant overhead.

If FreeBSD numbers don't change much: confirms that the
Linux gap was specifically the NVIDIA driver's blocking-wait
latency, not a code defect that affects both platforms.

If FreeBSD numbers DO improve substantially (>30%): the
persistent-buffer pattern was leaving performance on the table
on FreeBSD too, and the cross-platform speedup goes into the
*walkable-path* blog as the win.

#### R3.2 — Run the timing_get instrumentation on FreeBSD (high value)

Same code path, on FreeBSD. Sample any single Vulkan cell with
`Process.put(:exmc_count_dispatches, true)` and call
`Nx.Vulkan.Native.timing_get/0` after sampling completes.
Report:

- count       (number of dispatch() calls)
- submit_ns   (total time inside vkQueueSubmit across all calls)
- wait_ns     (total time inside vkWaitForFences across all calls)
- record_ns   (total time recording the cmd buffer)

Compare per-call to the Linux numbers above:
- Linux: 138us submit + 1130us wait + 19us record per submit_and_wait

If FreeBSD shows ~138us submit + ~150us wait + ~19us record,
the gap is purely the wait. mesa-radv on FreeBSD does
non-blocking or short-spinwait fence waits where NVIDIA Linux
sleeps the CPU. That's the conclusion that ships with the
*walkable-path* blog.

#### R3.3 — Optional: artificial-load contention test (skip-if-busy)

The original optional test from R2 — spin up some artificial
GPU load while running one Vulkan cell, measure contention
multiplier. Only useful if R3.1 and R3.2 leave any ambiguity
about the Linux gap being driver-specific.

### What R3 explicitly does NOT ask

- No FreeBSD-side code changes. The Linux fix is in shared code
  (`nx_vulkan/main` + `pymc/feat/dsl-shader-codegen`); pulling
  it gets you both the timing instrumentation and the
  persistent-buffer fix automatically.
- No new shaders. The 6 chain shaders you shipped are still
  the chain shaders.
- No re-vendoring of `nx_vulkan/c_src/spirit/`. The C++ changes
  to `Backend_par_vulkan.cpp` are vendored — you get them on pull.

### Cross-references

- Linux post-fix race results (in flight): `~/projects/learn_erl/pymc/exmc/bench/fair_race_results_linux.md`
- This TODO supersedes R2 (H1) which is closed.
- The H3 instrumentation is `Nx.Vulkan.Native.timing_reset/0` and `timing_get/0`,
  documented in `lib/nx_vulkan/native.ex`.

---

## R4 (2026-05-05) — GPU node Phase 0 cross-platform validation

The Phase 0 pieces of `PLAN_GPU_NODE.md` landed today on
`nx_vulkan@feat/gpu-node` and `pymc@feat/gpu-node`. Six workstreams
shipped from scaffold to deliverable in one session on the Linux
side; the cross-platform validation lives with you.

You have ssh to mac-247, so this whole matrix can run from
mac-248 — drive 247 over ssh as a second worker, report back as
one set of numbers.

### Pull both branches

```sh
# on mac-248 (and via ssh on mac-247)
cd ~/projects/learn_erl/nx_vulkan
git fetch nas && git checkout feat/gpu-node && git pull --rebase nas feat/gpu-node

cd ~/projects/learn_erl/pymc
git fetch origin && git checkout feat/gpu-node && git pull --rebase origin feat/gpu-node

cd ~/projects/learn_erl/nx_vulkan && mix deps.get && mix compile
cd ~/projects/learn_erl/pymc/exmc && mix deps.get && mix compile
```

`feat/gpu-node` is forked off `feat/dsl-shader-codegen` and
includes everything in R1/R2/R3 plus the new gpu_node infrastructure.

### The R4 matrix

For each of three platforms — your existing Linux RTX 3060 Ti R3
numbers (already in `bench/fair_race_results_linux.md`), mac-248's
FreeBSD GT 750M, mac-247's macOS GT 650M (MoltenVK→Metal) — run
each of the four measurements:

| | W2 validator | W4 warmup | W5 pipeline cache (when spike lands) | W6 chaos |
|---|---|---|---|---|
| Linux RTX 3060 Ti | green (auto-skip 6 known-failures) | done | pending | done (3 tests) |
| **mac-248 FreeBSD GT 750M** | **R4 ask 1** | **R4 ask 2** | (Phase 2) | **R4 ask 3** |
| **mac-247 macOS GT 650M** | **R4 ask 4** | **R4 ask 5** | (Phase 2) | **R4 ask 6** |

W5 is research-only on Linux right now (no spike code yet), so
nothing to test cross-platform until that lands. The other three
are testable today.

### R4 ask 1 + 4 — W2 validator on each Mac

```sh
# from each Mac:
cd ~/projects/learn_erl/pymc/exmc
EXMC_COMPILER=vulkan mix test test/exmc/gpu_node/validator_test.exs --include vulkan
```

What we want: do the same 6 chain shaders that pass on Linux also
pass on the Mac platform? The validator's auto-skip table tags 4 of
the 6 as `vulkan_known_failure` on Linux (Exponential, HalfNormal,
Weibull, Cauchy — Stage 1.5.4 chain-integrator drift, pre-existing).
Normal and StudentT pass cleanly with KS D=0.

If the same 4 fail on both Macs with similar fingerprints: the bug
is platform-agnostic, lives in shader logic.

If only some of those 4 fail on one platform: the bug is
driver/shader-compiler specific. MoltenVK → Metal transpiles SPIR-V
to MSL with its own quirks; mesa-radv on FreeBSD is yet another path.

If MORE than 4 fail (or different ones): a real platform-specific
break — escalate.

Calibration table you'll want to compare against:
`pymc/exmc/research/gpu_node/validation_calibration.md`.

### R4 ask 2 + 5 — W4 warmup characterization

```sh
cd ~/projects/learn_erl/pymc/exmc
mix run bench/warmup_characterization.exs
```

Produces:
- `bench/warmup_curves/{family}.csv` — per-window timing for
  Normal, Exponential, StudentT, HalfNormal, Weibull (Cauchy not
  in the cell list yet — known TODO).
- `../../nx_vulkan/research/gpu_node/warmup_summary.md` — overwrites
  the Linux summary if you run the script. Don't push that
  overwrite directly; instead copy your numbers into a new section
  of the same file under `## FreeBSD GT 750M (mac-248)` and
  `## macOS GT 650M (mac-247)`.

Linux R4 baseline (from yesterday's run, RTX 3060 Ti):
- Normal: cold 254 ms, warm@20, p99/p50 = 2.04
- Exponential: cold 916 ms, warm@38, p99/p50 = 1.65
- StudentT: cold 414 ms, warm@20, p99/p50 = 1.69
- HalfNormal: cold 596 ms, warm@20, p99/p50 = 1.49
- Weibull: cold 553 ms, warm@50 (never settled)

Hypothesis: mesa-radv on FreeBSD has lower per-fence latency (~150 µs
vs ~1.13 ms on NVIDIA Linux), so cold/warm ratios will be tighter
and the warm point will land earlier. MoltenVK on macOS sits over
Metal which has its own pipeline-state-object cache; Metal's first
pipeline create can be slow but subsequent ones are very fast.
Both Macs likely settle faster than Linux NVIDIA.

If a Mac shows a wildly different curve shape — e.g. cold/warm ratio
of 50× — flag it, that's likely a driver issue we need to know about.

### R4 ask 3 + 6 — W6 chaos test

```sh
cd ~/projects/learn_erl/pymc/exmc
EXMC_COMPILER=vulkan mix test test/exmc/gpu_node/bulkhead_test.exs --include vulkan
```

3 cases: timeout fires, dead server returns expected error, sampler
falls back to EXLA when GPU node times out. All 3 pass on Linux
RTX 3060 Ti.

This is mostly a portability check (does the bulkhead path crash on
a different OS?), not a driver-recovery test — Phase 1 W6 will add
the deliberately-bad-shader chaos test.

### Reporting back

Best path: a single new commit on `nx_vulkan/feat/gpu-node` named
`R4: cross-platform results — FreeBSD GT 750M + macOS GT 650M`,
appending to `research/gpu_node/warmup_summary.md` and a new
`research/gpu_node/validation_calibration_macs.md`. Push to NAS.

If anything on either Mac fails outright (build error, NIF crash,
test pathology), flag it inline in the commit message and I'll
investigate from the Linux side.

### What R4 explicitly does NOT ask

- No W1 work — the codegen substrate decision (substrate (a),
  templated GLSL) is settled.
- No W3 work — the GenServer is platform-agnostic Elixir.
- No W5 work yet — the spike hasn't landed; will be R5.
- No new shaders — the existing 6 are still the universe.

---

## R5 (2026-05-06) — Phase 1 synthesis cross-platform + Beta W2 validation

Phase 1 of `PLAN_GPU_NODE.md` shipped today. The first synthesized
chain shader (Beta(α, β) on logit-unconstrained space) renders →
compiles → dispatches end-to-end on Linux RTX 3060 Ti in ~200 ms
cold path. Now needs cross-platform validation.

### Pull

```sh
cd ~/projects/learn_erl/nx_vulkan && git fetch nas && git pull --rebase nas feat/gpu-node && mix deps.get && mix compile
cd ~/projects/learn_erl/pymc && git fetch origin && git pull --rebase origin feat/gpu-node && cd exmc && mix deps.get && mix compile
```

New on `nx_vulkan@feat/gpu-node`:
- `67fa832` — `leapfrog_chain_synth` generic NIF (raw push-data, max 128 bytes)

New on `pymc@feat/gpu-node`:
- `521550173` — `ShaderTemplate`, `ShaderSpecs.beta`, `Synthesis` (renders + glslangValidator + cache)

### R5 ask 1 + 2 — Synthesis works on FreeBSD GT 750M and GT 650M

`glslangValidator` is a build-time dependency. Verify it's installed:

```sh
which glslangValidator || pkg install vulkan-tools  # mac-248 (FreeBSD)
which glslangValidator || brew install glslang      # mac-247 (driven via ssh)
```

Quick sanity script (run on each Mac):

```sh
cd ~/projects/learn_erl/pymc/exmc
cat > /tmp/r5_synth_smoke.exs <<'EOF'
{:ok, _} = Application.ensure_all_started(:nx_vulkan)
Nx.Vulkan.Native.init()

t0 = System.monotonic_time(:millisecond)
spec = Exmc.GPUNode.ShaderSpecs.beta()
{:ok, spv_path} = Exmc.GPUNode.Synthesis.compile(spec)
t1 = System.monotonic_time(:millisecond)
IO.puts("synth+compile: #{t1 - t0}ms")

n = 1; k = 32
{:ok, q_ref} = Nx.Vulkan.upload_binary(<<0.0::little-float-32>>)
{:ok, p_ref} = Nx.Vulkan.upload_binary(<<0.5::little-float-32>>)
{:ok, m_ref} = Nx.Vulkan.upload_binary(<<1.0::little-float-32>>)
push = Exmc.GPUNode.ShaderSpecs.beta_push(n, k, 0.1, 2.0, 5.0)

t2 = System.monotonic_time(:microsecond)
{:ok, {_q, _p, _g, logp}} =
  Nx.Vulkan.Native.leapfrog_chain_synth(q_ref, p_ref, m_ref, push, k, spv_path)
t3 = System.monotonic_time(:microsecond)
IO.puts("first dispatch: #{t3 - t2}us")

{:ok, logp_bin} = Nx.Vulkan.Native.download_binary(logp, k * 4)
[first | _] = for <<v::little-float-32 <- logp_bin>>, do: Float.round(v, 4)
IO.puts("first logp: #{first}  (analytic ≈ -1.520)")
EOF
mix run /tmp/r5_synth_smoke.exs
```

Linux baseline: synth+compile 157 ms cold / 8 ms cached, first dispatch 39 ms,
first logp -1.5162.

What we want from each Mac:
1. **synth+compile time** — does `glslangValidator` complete in <200 ms?
   On FreeBSD with the system pkg version, expect similar order of magnitude.
2. **first dispatch time** — pipeline-create cost. On mesa-radv expect
   <20 ms (mesa's pipeline creation is faster than NVIDIA Linux).
3. **first logp value** — should match -1.5162 within f32 precision
   (≈ ±0.001). If it differs by more than a percent, something is
   off (push-constant layout, byte order, GLSL semantics).

### R5 ask 3 + 4 — Re-run the W2 validator post-Phase-1

Same as R4 ask 1 + 4 but on the new branch. Should still be
13 tests, 0 failures, 4 excluded (Exponential/HalfNormal/Weibull/Cauchy
known-failures). Confirms Phase 1 didn't regress anything.

```sh
cd ~/projects/learn_erl/pymc/exmc
EXMC_COMPILER=vulkan mix test test/exmc/gpu_node/validator_test.exs --include vulkan
```

### R5 ask 5 + 6 — Re-run W4 warmup characterization post-Phase-1

Same as R4 ask 2 + 5. The synthesis path doesn't touch the existing
6 hand-written shaders, so warmup curves should be unchanged. If
they ARE significantly different on either Mac, the GenServer
routing or batched-IO might be interacting differently with the
mesa/MoltenVK paths.

```sh
cd ~/projects/learn_erl/pymc/exmc
mix run bench/warmup_characterization.exs
```

(Don't overwrite `warmup_summary.md` this time — append a new section
"## R5 — Post-Phase-1 numbers" with the per-Mac tables, like the R4
table format from `r4_cross_platform_results.md`.)

### R5 ask 7 + 8 — Re-run W6 chaos tests

Same as R4 ask 3 + 6. The `:gpu_dispatch_timeout` calibration
issue from R4 (1 ms timeout too generous on FreeBSD) is still
present — that test will fail on mac-248 again. Acceptable for
now; will be fixed once Phase 1 W6 lands the proper chaos test
(deliberately bad shader instead of artificial timeout).

```sh
cd ~/projects/learn_erl/pymc/exmc
EXMC_COMPILER=vulkan mix test test/exmc/gpu_node/bulkhead_test.exs --include vulkan
```

### R5 ask 9 — NEW: extend Beta with the W2 validator

The W2 validator harness can hit a Beta model now that we have a
synthesized shader. Want this to be part of the cross-platform
matrix because if Beta passes on Linux but fails the same checks on
a Mac, that's our first cross-platform shader bug — and the
synthesized path is where future bugs will most often live.

The wiring needed:
- A Beta IR fixture in `Exmc.Builder` if not already present.
- A `Beta` distribution module in `Exmc.Dist` that the EXLA path
  can use as the reference.
- A new vulkan_meta tag `{:beta, alpha, beta}` recognized by
  `tree.ex`'s `do_dispatch` clauses.
- Test case in `validator_test.exs`:

```elixir
test "Beta(2, 5) synthesized shader matches EXLA reference" do
  {:ok, _path} = Exmc.GPUNode.Synthesis.compile(Exmc.GPUNode.ShaderSpecs.beta())
  ir = Builder.new_ir() |> Builder.rv("x", Dist.Beta, %{alpha: Nx.tensor(2.0), beta: Nx.tensor(5.0)})
  Process.put(:fused_leapfrog_meta, {:beta, 2.0, 5.0})
  assert :ok = Validator.validate(ir, {:beta, 2.0, 5.0})
end
```

This test will likely take some Linux-side wiring before it runs.
**Skip ask 9 until we ship the Beta IR + Dist + tree.ex hookup.**
For now, focus on asks 1-8 (synth smoke + replay R4 matrix).

### Reporting back

Append to `r4_cross_platform_results.md` with a `## R5` section and
push as a single new commit on `feat/gpu-node`.

If `glslangValidator` is missing on either Mac and not in the
default package repo, flag it inline — we may need to vendor a
prebuilt SPIR-V cache for distribution rather than requiring the
compiler at runtime.

---

## R6 (2026-05-06) — Phase 1 fully wired: 10-cell race + 3 new validator cases

Phase 1 of `PLAN_GPU_NODE.md` now wires synthesized chain shaders all
the way from IR → meta → synthesis → dispatch → statistical validation.
Three new distributions hit the Vulkan path: Beta, Gamma, Lognormal.

### Pull

```sh
cd ~/projects/learn_erl/nx_vulkan && git fetch nas && git pull --rebase nas feat/gpu-node && mix compile
cd ~/projects/learn_erl/pymc && git fetch origin && git pull --rebase origin feat/gpu-node && cd exmc && mix compile
```

New on `pymc@feat/gpu-node`:
- `2927bdf80` — wire Beta/Gamma/Lognormal into NUTS dispatch + W2 validator
- `0779c7e34` — add Beta/Gamma/Lognormal cells to bench/fair_race.exs
- `ace199603` — Gamma + Lognormal specs (Beta was earlier in `521550173`)

No `nx_vulkan` changes since R5 — the same `leapfrog_chain_synth` NIF
serves all three new families; only Elixir-side wiring changed.

### What's different in this race

`bench/fair_race.exs` now has **10 cells** instead of 7. The 3 new cells
exercise the runtime-synthesis path you smoke-tested in R5 — but driven
by real NUTS sampling instead of a single-dispatch fixture.

### R6 ask 1 + 2 — Re-run the 10-cell fair race

```sh
cd ~/projects/learn_erl/pymc/exmc
mix run bench/fair_race.exs    # full 1000/1000 if you have time
# OR
RACE_QUICK=1 mix run bench/fair_race.exs    # 100/100 quick mode
```

### Linux RTX 3060 Ti baseline (RACE_QUICK 100/100, post-Phase-1)

Three new rows at the bottom:

| Cell             |  d | EXLA wall (ms) | Vulkan wall (ms) | EXLA ESS/s | Vulkan ESS/s |  ratio |
|------------------|----|----------------|------------------|------------|--------------|--------|
| Normal d=1       |  1 |          873   |          1,500   |       24.2 |         14.1 |   0.58 |
| Normal d=8       |  8 |        1,960   |          1,544   |       33.7 |         42.8 | **1.27** |
| Normal d=50      | 50 |        3,623   |          1,744   |        6.3 |         23.1 | **3.67** |
| Exponential      |  1 |        1,663   |          1,951   |       25.9 |         22.1 |   0.85 |
| StudentT df=3    |  1 |        1,700   |          1,412   |       35.7 |         36.1 |   1.01 |
| HalfNormal       |  1 |        1,693   |          2,043   |       28.5 |         13.8 |   0.48 (✗ pre-existing) |
| Weibull k=2      |  1 |        1,599   |          1,835   |       27.3 |         24.2 |   0.89 |
| **Beta synth**   |  1 |        1,671   |       **43,094** |       24.7 |      **1.0** |   0.04 |
| **Gamma synth**  |  1 |        1,550   |       **14,475** |       40.2 |      **4.3** |   0.11 |
| **Lognormal synth**| 1 |       1,675   |          1,760   |       25.4 |         24.2 |   0.95 |

### IMPORTANT — Beta and Gamma look bad. They're not a driver bug.

The agreement check (mean / variance / KS test) **passes** for both
Beta and Gamma — the synthesized shaders produce statistically correct
posteriors. But the **mixing is terrible** (ESS = 1.0 and 4.3 out of
100 samples). The chain is technically valid but barely moves between
samples.

Root cause is **NUTS adaptation (step size + mass matrix + tree depth)
is tuned for Normal-shape gradients**, and the new families have
gradient profiles that adaptation hasn't been calibrated for:

- **Gamma**'s gradient `α - β·exp(q_uc)` explodes exponentially as
  q_uc grows on log-uc space. NUTS responds by setting tiny ε,
  producing depth-10 trees (~1000 leapfrogs per iter).
- **Beta**'s gradient is bounded by sigmoid, but warmup on a logit-uc
  posterior with α=2, β=3 takes a long time to settle if initial
  step size overshoots.

This is documented as Phase 1's known limitation. Performance fix
needs per-family adaptation heuristics or longer warmup — both
**Linux-side** work, not platform-specific.

What we expect to see on FreeBSD:

- **Lognormal synth**: should be in the same relative position as
  Normal (well-behaved). Probably 10× faster wall than Linux due to
  the same per-fence latency advantage measured in R3-R5.
- **Beta synth + Gamma synth**: will ALSO be slow on FreeBSD, but
  not 30× slow — the per-iter cost is dominated by the leapfrog
  step count, and FreeBSD's lower per-fence latency means each
  leapfrog is cheaper. Probably ~3-5× the Lognormal wall on FreeBSD
  vs ~25× on Linux. **Confirms the slowness is adaptation, not
  driver.**

### R6 ask 3 + 4 — Run the validator with the 3 new Phase 1 tests

```sh
cd ~/projects/learn_erl/pymc/exmc
EXMC_COMPILER=vulkan mix test test/exmc/gpu_node/validator_test.exs --include vulkan --include requires_vulkan --exclude vulkan_known_failure
```

Linux baseline: 12 of 16 tests pass. The 4 failures are the
pre-existing `:vulkan_known_failure` shaders (Exp/Cauchy/HalfNormal/
Weibull — Stage 1.5.4 chain-integrator drift). The 3 new Phase 1
tests (Beta(2,3), Gamma(2,1), Lognormal(0,1)) all pass.

What we want to confirm on each Mac:
- Same 12/16 pass rate (or better).
- The 3 Phase 1 synthesized tests pass with `:ok`.
- If any of the Phase 1 tests fail with `{:error, %{check: :ks, ...}}`
  or similar, that's a real cross-platform bug in the synthesized
  shader path — escalate.

### What R6 explicitly does NOT ask

- **Don't try to fix Beta/Gamma slowness.** Adaptation tuning is a
  Linux-side Phase 2 task.
- **No new shaders** — the catalog is Beta + Gamma + Lognormal for
  Phase 1.
- **No W4 re-run unless you want to** — warmup curves haven't
  changed since R5 (the new shaders weren't measured by W4, but
  they could be added to a Phase 2 W4 run).

### Reporting back

Append to `research/gpu_node/r4_cross_platform_results.md` with a
new `## R6 — Phase 1 wired` section, same shape as R5. Push as a
single new commit.

If Beta/Gamma's wall on FreeBSD is **<5× the Lognormal wall**, that
confirms our hypothesis (slowness is adaptation, not driver). If
it's **20×+ like on Linux**, something else is going on and we'll
investigate.

---

## R8 (2026-05-06) — W7 Stage 1 cross-platform verification

W7 Stage 1 added `precise float` qualifiers to the loop-carried
accumulators (qi, pi, p_half) and gradient intermediates in the 4
chain shaders that exhibit Linux NVIDIA fp32 drift (Exponential,
Cauchy, HalfNormal, Weibull). Compiles to SPIR-V emit
`OpDecorate ... NoContraction`, telling drivers not to fuse the
multiply-add into FMA.

### Pull

```sh
cd ~/projects/learn_erl/nx_vulkan && git fetch nas && git pull --rebase nas feat/gpu-node && mix compile
cd ~/projects/learn_erl/pymc && git fetch origin && git pull --rebase origin feat/gpu-node && cd exmc && mix compile
```

New on `nx_vulkan@feat/gpu-node`:
- `29dd09b` — vendored 4 W7 Stage 1 SPVs from `spirit@704dd2df`

If you have a sibling spirit checkout at `~/projects/learn_erl/spirit/`,
`SPIRIT_DIR=~/projects/learn_erl/spirit mix compile` will copy the
fresh SPVs over the vendored ones automatically.

### Linux RTX 3060 Ti result (post-Stage-1)

W2 validator: 13/16 (was 12/16).

| Shader | Pre-Stage-1 | Post-Stage-1 | Delta |
|---|---|---|---|
| Exponential | drift | drift | unchanged |
| Cauchy | drift (IQR 1.76 vs 8.83) | drift | unchanged |
| HalfNormal | drift | drift (mean 0.582 vs 0.896) | unchanged |
| **Weibull** | **drift (mean ~0.98 vs 0.886)** | **PASS** | FIXED |

H7.1 (FMA fusion) is the right hypothesis for Weibull. The other 3
have a different / additional cause — Stage 2 (denormal handling)
and Stage 3 (NVK driver comparison) are next.

### R8 ask 1 + 2 — Re-run W2 validator on both Macs

```sh
cd ~/projects/learn_erl/pymc/exmc
EXMC_COMPILER=vulkan mix test test/exmc/nuts/vulkan/validator_test.exs --include vulkan --include requires_vulkan
```

**Hypothesis:** mesa-radv on FreeBSD doesn't fuse multiply-add as
aggressively as NVIDIA Linux. Adding `precise` should be a **no-op**
on FreeBSD — the bytes already produced the right answer. So we
expect the FreeBSD validator to remain **16/16 on both Macs**.

What we want to confirm:
1. **No regression.** All 16 still pass on FreeBSD GT 750M and GT 650M.
2. **`precise` doesn't introduce a new failure.** If a previously-
   passing shader now drifts, our fix is wrong.

### R8 ask 3 + 4 — Spot-check Weibull wall time on FreeBSD

`precise` can disable fast paths in the shader compiler. NVIDIA's
docs explicitly warn that overuse can slow shaders by 10-30%. Most
compute kernels don't notice (they're memory-bound), but the chain
shader is arithmetic-bound by design.

Run the fair race (RACE_QUICK is enough):

```sh
RACE_QUICK=1 mix run bench/fair_race.exs
```

Compare Weibull cell wall to your R5/R6 numbers. If Weibull's
wall increases by more than ~20%, `precise` cost is significant;
we may want to scope `precise` to just qi/pi (loop-carried) and
not the per-step intermediates. If the wall is unchanged, we ship
Stage 1 as-is.

### What R8 explicitly does NOT ask

- **Don't try to fix Cauchy/Exp/HalfNormal.** Those need Stage 2
  (denormal clamping) or Stage 3 (NVK), both Linux-side
  investigations.
- **No NVK install on FreeBSD.** NVK is mesa's NVIDIA driver — only
  applicable to Linux.

### Reporting back

Append to `r4_cross_platform_results.md` with a `## R8` section, same
table format. Push as a single commit on `feat/gpu-node`.

If R8 confirms 16/16 on both Macs and Weibull wall is unchanged,
W7 Stage 1 lands as a real fix (one shader recovered) and we move
to Stage 2 for the others.

---

## R9 (2026-05-06) — W7 closure + matched-precision validator

W7 closes with three real fixes plus a clean diagnosis of what's
left:

- W7 Stage 1 (`spirit@704dd2df` + `nx_vulkan@29dd09b`): `precise float`
  on chain shader loop accumulators. **Fixed Weibull's Linux NVIDIA
  fp32 drift** — real driver-level FMA fusion bug, fix is portable.
- W7 Stage 2.5 (`pymc@83f7464cf`): matched-precision validator. Adds
  `precision: :f32 | :f64` opt to `Validator.validate/3`. Untangles
  shader-correctness from f32-vs-f64 precision-gap artifacts.
- W7 Stage 2.5 follow-up (`pymc@65cf9e486`): re-tag the 3 historical
  failures by their actual diagnosis + fix HalfNormal's transform.

### Pull

```sh
cd ~/projects/learn_erl/nx_vulkan && git fetch nas && git pull --rebase nas feat/gpu-node && mix compile
cd ~/projects/learn_erl/pymc && git fetch origin && git pull --rebase origin feat/gpu-node && cd exmc && mix compile
```

### What changed for the 4 historical "vulkan_known_failure" tests

| Shader | Pre-W7 status | New status | Tag |
|---|---|---|---|
| Weibull | red on Linux | **green on all platforms** (W7 Stage 1) | none |
| Exponential | red on Linux | **green at matched-precision** (test now uses `precision: :f32`) | none |
| HalfNormal | red on Linux | **green at matched-precision** (transform changed `:softplus` → `:log`) | none |
| Cauchy | red on Linux | green-by-skip (auto-excluded) | `:f32_precision_limited` |

The `Exmc.Dist.HalfNormal.transform/1` change from `:softplus` to
`:log` is a behavior change for non-shader users too. The
mathematical posterior is unchanged (both transforms are valid
bijections), but the unconstrained space (and therefore mass-matrix
adaptation, ESS, etc.) shifts.

### R9 ask 1 + 2 — Re-run validator + W6 + fair race on both Macs

```sh
cd ~/projects/learn_erl/pymc/exmc
EXMC_COMPILER=vulkan mix test test/exmc/nuts/vulkan/         # validator + server + bulkhead
RACE_QUICK=1 mix run bench/fair_race.exs                      # full 10-cell race
```

### Linux RTX 3060 Ti baseline (post-W7-2.5 follow-up)

```
test/exmc/nuts/vulkan/  →  22 tests, 0 failures, 1 excluded
test/exmc_test.exs      →  11 doctests, 18 tests, 0 failures
```

The 1 excluded is Cauchy (`:f32_precision_limited`). It auto-skips
under `EXMC_COMPILER=vulkan` because f32 chain shaders structurally
cannot reproduce f64 reference IQRs for fat-tailed posteriors.

### What we want to confirm on FreeBSD

Both Macs were already 16/16 on the validator under the old
f64-EXLA-vs-f32-Vulkan default. With the new tests using
`precision: :f32`, the validator becomes a stricter shader-
correctness check (smaller tolerances at matched precision).
Expectation: still 16/16 (or 22 since the suite grew slightly).
If anything goes red, that's a real shader bug we need to know
about on FreeBSD.

The HalfNormal Dist transform change is the riskier one. If your
FreeBSD samples for HalfNormal posteriors look wildly different
from before (e.g., ESS or wall_ms shifts by >2×), that's the
transform change biting.

### R9 ask 3 — Spot-check HalfNormal sampler stability on FreeBSD

If you have a HalfNormal-using model in any of your existing
benchmarks or tests, re-run it with the new transform and compare
the posterior. Should be statistically equivalent (both transforms
sample the same posterior) but mass-matrix adaptation may settle
differently.

If the existing tests don't exercise HalfNormal, just running
`mix test` (full suite) covers the basic regression check.

### Reporting back

Append to `r4_cross_platform_results.md` with a `## R9` section,
single new commit on `feat/gpu-node`. Should be the last pre-merge
mac-248 ask before this branch lands on `pymc/main`.

If R9 is clean (no regressions on either Mac), W7 is closed and
`feat/gpu-node` is ready to merge.

---

## R10 (2026-05-06) — nx_vulkan-side tests + demo portability

Phase 2 closed the architectural gap from the Exmc.GPUNode →
Nx.Vulkan.* extraction by adding standalone tests + a demo that
exercise the new modules without any exmc dependency. R10 is the
cross-platform check: the demo is the README's promise that
nx_vulkan can stand on its own — confirm it does.

### Pull

```sh
cd ~/projects/learn_erl/nx_vulkan && git fetch nas && git pull --ff-only nas main && mix compile
```

`feat/gpu-node` is now merged to `main` (`0ea8adb`). New on top:
- `168084a` — standalone tests + demo for Phase 2 GPU node API.

### R10 ask 1 + 2 — Run the new nx_vulkan-side tests on both Macs

```sh
cd ~/projects/learn_erl/nx_vulkan
mix test test/nx_vulkan/
```

Linux RTX 3060 Ti baseline: **26 tests, 0 failures, ~1.9 s wall**.

Tests cover:
- `Nx.Vulkan.Node` lifecycle + `with_node/2` (12 tests).
- `Nx.Vulkan.PipelineCache` load/persist round-trip (5 tests,
  one of which actually compiles + dispatches a Beta shader to
  produce a non-trivial cache blob).
- `Nx.Vulkan.ShaderTemplate` GLSL render (4 tests).
- `Nx.Vulkan.Synthesis` glslangValidator compile + cache hit + a
  deliberate failure case (5 tests).

Most tests are platform-agnostic (text rendering, file I/O). The
ones that touch Vulkan + glslangValidator are the new portability
check.

Hypotheses to confirm:
- All 26 pass on FreeBSD GT 750M.
- All 26 pass on FreeBSD GT 650M.
- The Synthesis cache-hit test (`< 50 ms` for a warm cache) holds
  on both Macs. If FreeBSD's filesystem has slower stat() this could
  trip; happy to relax the threshold if needed.
- The PipelineCache test that builds a real Beta shader compiles
  cleanly via `glslangValidator` (mac-248 already has it from R5).

### R10 ask 3 + 4 — Run the demo on both Macs

```sh
cd ~/projects/learn_erl/nx_vulkan
mix run examples/gpu_node_demo.exs
```

Linux baseline (warm cache):
```
synthesized Beta SPV in 5 ms (cached) / 149 ms (cold)
first dispatch via with_node: 16410 µs
logp[0]: -1.486 (analytic -1.4508, delta 0.035 ✓)
pipeline cache persisted: 12432 bytes
```

What we want from each Mac:
1. The demo runs to completion without crashing.
2. `delta after 1 leapfrog < 0.1` (the ✓ check).
3. The pipeline cache file is non-trivially sized (>0 bytes).
4. The first-dispatch wall is consistent with R5/R8 — somewhere in
   the 100-1000 µs range on FreeBSD (mesa-radv's per-fence latency
   is much lower than Linux NVIDIA's).

If the demo's first-dispatch timing on FreeBSD GT 750M is way
above the R5 baseline (e.g. > 5 ms), something regressed in the
Phase 2 plumbing. If it's around 100-300 µs as expected, R10
confirms the Phase 2 architectural split shipped cleanly across
all three platforms.

### What R10 explicitly does NOT ask

- No new shaders. The Phase 1 catalog (Beta/Gamma/Lognormal) is
  still the universe.
- No regression check on the exmc side — that's covered by R9.

### Reporting back

Append to `r4_cross_platform_results.md` with a `## R10` section,
single new commit on `main`. With R10 in the bag, the GPU node
work is fully verified and we can move to the next workstream
(Beta/Gamma adaptation tuning, or Phase 3 multi-client).
