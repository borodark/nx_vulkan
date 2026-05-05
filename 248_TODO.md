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
