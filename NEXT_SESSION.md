# Next session — state, what is blocked, what is open

**HEAD `a29246f`**, pushed to `origin` (private). Nothing pushed to `upstream`.
Working tree clean. Rewritten 2026-09-04.

**Verified on all four boxes at `a29246f`** — full suite, the property file
alone, and `sh scripts/strict_test.sh`:

    box                   suite    properties   strict
    super-io  Ampere      18.2s      1.1s       0 failures
    jetson    Tegra X1    41.7s      5.9s       0 failures
    mac-247   GT 650M     14.8s      3.6s       0 failures
    mac-248   GT 750M     10.7s      1.5s       0 failures

833 doctests, 907 tests, 0 failures everywhere; 163 excluded under strict.
Residency 755/833 (90.6%).

Note both 2012 Keplers run the suite FASTER than the modern Ampere desktop.
That is not a curiosity, it is the host-selection finding below showing up in
wall clock.

---

## The one thing to read first

**The goal is helping eXMC.** It is the only real consumer, it is **f64**, and
its cost is per-dispatch. Work that does not reduce per-dispatch cost on the f64
chain path is worth less than it looks — an earlier session spent a stretch
adding f32 transcendentals before that was said out loud, and they help nobody
today.

---

## Blocked on the exmc session

**The batched f64 chain path is built, correct on both memory architectures,
measured — and unreachable from their DEFAULT path.** Their vectorized sampler
(`sampler.ex:1256`) runs chains sequentially, `Enum.map` over chain states, so
there is never a moment when four chains want a leapfrog together. Nothing to
batch.

The route that exists: their non-vectorized path (`sampler.ex:119`) drives
chains through `Task.async_stream`, so they arrive concurrently, and their
`BatchCoordinator` was built to coalesce exactly that — inert since written
because nothing ever set `:exmc_chain_coord`.

**One fix is theirs and is written down on their side:** the coordinator's
partition key is `{phash2(meta), k, eps}`. **K is in the key**, so it refuses to
batch chains of differing depth — full batching on 16% of draws and singletons
on the other 84%, against their own histogram. Dropping `k` and padding at flush
is correct, and safe because of the prefix property pinned in `cccbd71`.

**The batching contract, so their key is right:** same shader (identical priors,
since parameters are baked), same `d`, K may differ (pad to deepest — nearly
free), inputs instance-major, `n_instances` bounded by the device workgroup
limit.

---

## What is measured, and on which box

Everything per-dispatch below is **mac-248, headless**. super-io is a desktop
compositing Firefox on the same GPU: a ~900 us noise band that does not merely
fail to resolve small effects but **manufactures large ones**. It reported the
buffer pool at 17% when the truth was 1.3%, and the fence fold at 36% when the
truth was 18.2%.

Chain dispatch cost, cumulative:

    ab2e779           365 us
    096d7bd fast OFF  238 us   8cce91c four readback fences -> one   -35%
    096d7bd fast ON   224 us   b59c4a7 small-upload fast path         -6%
    f4c00f4           210 us
    8cd19ee           172 us   fence fold, readback rides dispatch   -18.2%
    d210601           170 us   buffer pool                            -1.3%  REVERTED

**365 -> 170 us, about -53% per chain dispatch.** Both large wins are
readback-side, which the `3*K*d*8` down against `2*d*8` up asymmetry predicted.

**The buffer pool was reverted (`190bf67`).** Worth 2.2 us at concurrency one
and NOTHING at M >= 2 — both arms plateau at ~7350 dispatches/s from M=2, because
every dispatch does `submit_and_wait` on a single queue and the queue saturates
at two callers. It optimised something off the critical path in the regime the
only consumer runs.

**That ~7350/s ceiling is a fact about this path.** exmc's sampling run does
244/s — **3% of it**. Concurrency is not their lever; dispatch COUNT is, which
is what the batched path attacks.

At their operating point (d=4, K<=16, measured on 248): **intercept 91.3 us
against slope 2.5 us/step, so 86% of a dispatch is fixed cost.** Hence batching:

    4 chains serial     4 x (91.3 + 2.5*6)     = 424.7 us
    4 chains batched    91.3 + 2.5*7 (padded)  = 108.7 us

Measured at their depth histogram: **3.4-4.3x**, and batched cost is FLAT in
chain count (119 us at 2 instances, 111 at 4, 119 at 8) — the GPU runs the
workgroups concurrently, so 8 chains would be 9.6x. At d=4 a single-instance
dispatch occupies 4 of 256 threads; there is a great deal of room.

**What this does NOT touch:** their host-side tree logic, which stays per-chain.
The wall-time fraction is **unknown** — both estimates of it (1.9 ms, then
68-80%) were withdrawn by their author, the second after producing physically
impossible negative values.

---

## Open items, ranked by value to eXMC

### 1. The host-side NUTS cost — theirs

Handed over with a method: measure the split rather than subtract it; establish
whether it is per-dispatch or per-DRAW (trajectories double until a U-turn, so
tree logic runs O(2^depth) per draw while dispatches run per step); census what
the tree does with the `3*K*d*8` bytes it gets back; check for per-step
`:binary.part/3` or list conversion, which looks like GPU work until a K-sweep
separates it.

### 2. What is left of the in-NIF cost

Three bites taken: 4 fences -> 1, then 2 submissions -> 1. What remains per
chain dispatch: descriptor-set construction, the command-buffer build, three
upload buffers, four `NewBinary` allocations. **Measure before building** — the
K-sweep intercept is the instrument, and the buffer pool is what happens
otherwise.

### 3. Race 5 (MCMC) could now actually run

It never could before, because nothing could drive the chain path. The harness
used this session lives in the scratchpad only and is worth committing if Race 5
is attempted.

### 4. Remaining fallbacks

f32 is down to **5**: `erf`, `erfc`, `atan2` (both forms), `sort`. Only `sort`
is allowlisted.

* `erf`/`erfc` are not in GLSL.std.450 and need a polynomial agreeing with
  `:math.erf`. Note the f64 shader's old `erf_approx` was deleted as unreachable,
  and a series MORE accurate than BinaryBackend would DISAGREE with it.
* `atan2` needs a new binary op code across four shaders.
* The **twelve f64 transcendental forms are a deliberate decline**, not a gap:
  `Nx.sin(f64 1)` would return 0.8414708971977234 against 0.8414709848078965.
  Admitting them turned 22 of Nx's own doctests red and would invalidate the
  78-entry residency register. `exp/log/sqrt/sigmoid/tanh` keep their f64
  boundary cast as a standing decision with `grad_test` calibrated to it; new ops
  do not inherit it.

Low value to eXMC either way — it is f64.

### 5. Race 1c voids on the Jetson even on a QUIET box

Load 0.40-0.73 across 14 in-run samples, thermal control 0.0%/0.7% drift, and it
still voided at **estimator divergence 30.7% against a 30% gate**. Contended it
read 177.8%. So contention inflates it ~5.8x, but clean it still sits a few
points over — a statement about the instrument.
`scripts/staged/jetson_run_trace.sh` was staged to gather exactly this and is
still unrun. Needs re-baselining after MAXN.

**Harness gap found doing it:** a VOID run still writes
`bench_results/unified_vs_discrete_<host>.json` with no void marker. A later
reader takes it as a result.

### 6. Jetson's below-cliff `alloc_buffer` is bimodal

24 and 28 MiB read 0.109-0.122 ms; 26 and 30 read 0.150-0.342. Two populations
at one size. More interesting now that allocation is known clock-invariant.

### 7. super-io's poison flip rate is still unmeasured

Two samples, both 20/20, one showing 19/40 effectiveness with 20/20 padding —
which breaks the lockstep claim. The cross-box table is already retired.

---

## What is settled, so nobody re-litigates it

* **The elementwise shaders are not a bottleneck on Ampere.** 431 GB/s of 448.
  The "27x is now ~14x, find the rest" headline two editions ago was a **210 MHz
  reading** — the card's idle floor. There was never anything to find.
* **The 32 MiB allocation cliff is vulkano's**, per-allocation, 6/6 across boxes.
* **The BAR1 cliff is a different cliff**: 256 MiB on super-io, cumulative across
  live host-visible buffers, a whole-process budget. Do not conflate.
* **`buf_alloc` is not clock-bound; shader work is.** 1.08-1.38x idle-to-boost
  against 3.5-4.7x. Allocation is driver bookkeeping.
* **The chain NIFs push a FIXED header.** 20 bytes f32, 24 f64,
  `{k_steps, n_obs, d, _pad, eps}` (`n_instances` replaces `_pad` when batched).
  Anything past it is dropped. Family parameters belong in the shader source or
  a buffer, never a push tail.
* **The 128-byte push cap is not a width limit.** It guards bytes that never
  reach the GPU. exmc's `d <= 13` was an artifact of it; header-only packing took
  an 8-RV model from 0 dispatches to 2564, **13.1x**.
* **`d <= 256` is the real bound and is now ENFORCED** (`6d3a651`). Past it the
  chains get an undefined tail AND the logp tree reduce silently sums only the
  first 256 elements — a wrong posterior, not an error. Verified on all four
  boxes: 256 accepted, 257 refused, including on both 2012 Keplers.
* **The glslang pin is not load-bearing.** 81 of 81 shaders byte-identical at
  15.1.0, 16.2.0 AND 16.5.0. The SPIR-V generator word encodes glslang's
  GENERATOR version, not its release version. Record the version with a
  byte-comparison; do not treat a mismatch as invalidating one without checking.
* **The unified-vs-discrete question is unanswered and four designs failed.**
* **The control pair is not a pair.** mac-247 and mac-248 differ 1.39x on
  submission cost, opposite in direction to their per-dispatch GPU work.

---

## Harness invariants — every one of these cost something

### Measurement

* **Pick the host by CONTENTION, not clock observability — they are
  anti-correlated.** super-io is the only box where `clocks.sm` reads and it is a
  desktop: 57% spread against headless mac-248's **0.3%**. Anything under ~1 ms
  is unmeasurable there. Both 2012 Keplers run the suite faster than it does.
* **A noisy host does not merely fail to resolve a small effect. It manufactures
  a large one.** 1.3% measured as 17%; 18.2% measured as 36%.
* **Run a null arm on the candidate host FIRST.** A control where the change
  cannot act tells you whether the instrument can resolve anything. One reported
  4.6% of a 7.2% "effect" — the host disqualifying itself, an hour before that
  was read as such.
* **Record the GPU clock from OUTSIDE the process.** Four DVFS incidents.
  Polling `nvidia-smi` inline costs 50-100 ms of idle and drops the boost it is
  measuring.
* **Sanity-check derived figures against a physical bound.** One corrupted run
  announced itself by reporting 804 GB/s on a 448 GB/s card.
* **Amplify until the effect clears the noise floor, replicate, rotate arm
  order.** Whichever arm runs first after a binary swap eats the cold start.
* **When verifying a FIX, the broken arm must still separate.** "Fixed looks like
  pre" proves nothing if the harness cannot resolve broken from pre either.
* **Prefer a measurement to a subtraction.** A per-fence figure derived by
  subtraction missed the chain path by 3x; a split table built by differencing
  two noisy quantities produced physically impossible negatives.
* **Measure the tolerance, do not guess it.** The finite-difference h=1e-3 was
  chosen from measured error; h=1e-4 is WORSE, because the f32 boundary cast puts
  ~1e-7 on logp and a central difference divides it by 2h.
* **Warmup can look exactly like a leak.** 625 -> 902 -> 969 us across three
  replicates at a 300-dispatch warmup; it settles at 6000.

### Instruments that lie

* **`mix run` and `mix test` use DIFFERENT `_build` trees.** `mix run
  --no-compile` after a green `mix test` runs STALE code, confidently.
* **`mix compile` does not necessarily refresh `priv/native`.** A `.so` replaced
  with 25 bytes of text SURVIVED a plain rebuild — cargo found the sources
  unchanged and Rustler never re-copied. A swapped, stale or wrong-architecture
  artifact is not fixed by recompiling; touch a Rust source or clear `target/`.
  `mix deps.compile` is NOT sufficient.
* **`NXV_SKIP_NIF_BUILD` is sticky.** Rustler reads it via
  `Application.compile_env`, so the value is baked into the module and Elixir
  refuses to boot when the runtime value differs. Set it for compile AND every
  run after; to clear it, delete
  `_build/<env>/lib/nx_vulkan/ebin/Elixir.Nx.Vulkan.NativeV.beam`.
* **A check that cannot run fails OPEN.** `file` was not installed in the
  cross-build image, so an ELF-architecture bar printed `command not found` and
  was never applied while the lines around it went green.
* **A binary comprehension DROPS NaN and Infinity silently.**
  `for <<v::float-64-little <- bin>>` returns a SHORTER list, so
  `for v <- doubles(bin), do: assert v == v` **cannot fail**. Check the arity.
* **`nm -D` sees only dynamic symbols** — it reported 0 outline-atomics helpers
  where plain `nm` finds 22, and NIF functions never appear there at all. Ask the
  artifact to do the thing.
* **`pgrep -f "foo"` matches the shell running it.** Key on a pid via `/proc`,
  or use `pgrep -x`.
* **`mix` reads stdin.** Inside `ssh host 'bash -s'` with a heredoc it swallows
  the rest of the script and the missing lines look like a truncated transcript.
  Redirect with `< /dev/null` on EVERY mix call. (Documented, then repeated.)
* **Setting `Nx.default_backend/1` makes your "host reference" run on the GPU.**
  A `max_err = 0.0` that should have been ~3e-8 is what exposed it.
* **A control that fails to trigger is not evidence the instrument is broken.**
  It is evidence of nothing until you show the control CAN trigger.
* **A green strict run means "no unlisted fallback in the TESTED paths"**, never
  "no unlisted fallbacks". Sixteen sat behind a green run because nothing called
  them.

### Code and process

* **A passing test can be evidence FOR a defect.** exmc found five tests
  defending the 128-byte cap, under a describe block named for the limit they
  enforced, complete with a measured table. They were pinning a defect in place.
* **Three vacuous checks were found in one day**: a `file` bar never applied; a
  test that dispatched the single path twice and asserted it equalled itself; and
  a NaN guard that could not fail. **Every property test should have a null arm**
  — the finite-difference check has one, comparing one family's logp against
  another's grad and asserting it FAILS.
* **Read the decisions file before "fixing" a gap.** MISSION.md §3.2 lists
  broadcasting `pow` under "Decisions — recorded, not oversights"; `cf7b689`
  overturned it by accident, having found the gap by census.
* **A stale allowlist entry silently permits the regression it describes.**
* **`@moduletag` inside a `describe` tags the WHOLE module.** Use `@describetag`,
  or a future `--exclude` silently drops unrelated tests.
* **A library must not put a `File.rm_rf` target inside its consumer's
  directory.** `Synthesis.clear_cache/0` deleted exmc's shader cache on every
  `mix test` here.
* **Restore a shared box when you are done with it.** Two were left broken in one
  session: an artifact of unverifiable provenance on 248, and the Jetson
  compiled with a sticky flag so plain `mix test` died at boot.
* **Word-boundary anchor every template substitution.** `~r/\bpc\.alpha\b/`, or a
  parameter named `alpha` rewrites `pc.alpha_scale`.
* **One template, N variants.** The chain skeleton is parameterised on dtype AND
  on batched-vs-single rather than copied: a previous divergence moved the
  log-prob body above the position update and gave every distribution a one-step
  `logp` lag, blamed on the GPU for a month.
* **Interior test values are literals, not draws.** A failure on the Jetson takes
  minutes to reproduce; a seed-replay step is one nobody takes.
* **Build large test tensors with `from_binary`, not `Nx.iota`** — host-side
  construction is GPU idle time and will silently unboost the card.
* **Coordinate before adding load to a shared box.** The Jetson's failures are
  mostly `ExUnit.TimeoutError`, so on 2 cores an 85-second suite can flip
  someone else's test and the timeout is indistinguishable from a regression.

---

## Test coverage as it now stands

`test/nx_vulkan/chain_properties_test.exs` (14 tests, 1.1-5.9s across the fleet):

* **All seven guard branches** — length mismatch, `k=0`, push length at both
  boundaries, malformed push, `d=0`, `d` past the buffer, `d>256` in both dtypes,
  `n_instances=0`. All were untested; none reach GPU dispatch.
* **The prefix property in f32**, single and batched — it existed only for f64,
  and it is what exmc's ragged-depth padding depends on.
* **Shape sweep** for batched-equals-single, boundaries hit rather than sampled.
* **Determinism** — the same inputs dispatched twice must give the same bits.
* **`grad` is the derivative of `logp`**, by central difference, for all six f64
  families. Five had no numerical validation at all. Needs no second density
  implementation and the normalising constant drops out.

Plus, in `chain_f64_test.exs` / `chain_specs_test.exs`: every f64 AND f32 family
batches bit-identically to its single dispatch; instances do not bleed;
`n_instances=1` equals the single path; Normal reproduces a host leapfrog
bit-exactly.

**`leapfrog_chain_synth_batch/6` (f32) had existed since Task #154 with no
shader in this repo to drive it** — it shipped, was never exercised, and could
not have been. Same condition that let the push-block layout mismatch survive.
Closed in `f07e1f7`.

---

## Tooling

* **`.claude/skills/jetson-cross-build/`** — cross-builds the aarch64 NIF on
  super-io in ~2 minutes against ~47 native. With `NXV_SKIP_NIF_BUILD=1` it now
  applies to **every** commit, not just Rust-only ones. A full Jetson cycle
  measured at **61 seconds**.
* **`scripts/tegrastats_bars.sh`** — live bar graph with peak markers that never
  fall, for the one box with no `clocks.sm`. Still not run against the live box.
* **`buf_download_many/1`** — batched readback, exposed mainly so
  `staging_read_many` is reachable from a test.

---

## Disclosure

Per-dispatch numbers are from **headless mac-248** and are marked where they
appear. Anything measured on super-io is an upper bound: it is a desktop with
Firefox and Cinnamon on the same GPU, worth ~900 us of noise, wider than most
per-call effects in this document.

---

## Publishing (operator decision, not done)

* `~/projects/learn_erl/pymc/www.dataalienist.com` — **three** posts committed
  and not deployed: "An Absence Mistaken for a Discovery", the correction banner
  on "The Copy That Wasn't There", and "The Artifact That Survived Its Own
  Replacement".
* `mix hex.retire nx_vulkan 0.2.0`, `upstream/main` publishing, consumer pin
  bump — all still outstanding.
