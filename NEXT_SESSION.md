# Next session — state, what is blocked, what is open

**HEAD `d210601`**, pushed to `origin` (private). Nothing pushed to `upstream`.
Working tree clean. Rewritten 2026-09-01 after a long session on super-io with
the exmc session working the same problem from the consumer side.

**Verified:** 833 doctests / 882 tests / 0 failures on super-io, and the same
under `sh scripts/strict_test.sh` (163 excluded). Residency 755/833 (90.6%),
unchanged. The Jetson passed 833/876/0 at `15abc96`; it has NOT run anything
since.

---

## The one thing to read first

**The goal is helping eXMC.** It is the only real consumer, it is **f64**, and
its cost is per-dispatch. Work that does not reduce per-dispatch cost on the f64
chain path is worth less than it looks — this session spent a stretch adding f32
transcendentals before that was said out loud, and they help nobody today.

eXMC's own decomposition of a chain dispatch, on the Jetson at MAXN:

    ~1.2 ms   GPU executing
    ~1.0 ms   CPU inside the NIF call     <- ours, and this session attacked it
    ~1.9 ms   CPU in their NUTS tree logic <- theirs, and now the largest term
    ------
     4.1 ms   wall, GPU busy ~29% of a sampling run

---

## Blocked on the exmc session — do not re-derive these here

**MEASURED ON mac-248 (2026-09-02). Both real, both smaller than super-io
said.** Three arms, four rounds, order rotated each round, N=3000/sample,
6000-dispatch warmup, d=13 K=32, `.so` swap with `--no-compile`:

    f4c00f4  before the fold   209.6 210.7 210.9 209.3   median 210.2 us
    8cd19ee  fence fold        172.1 172.1 171.8 171.9   median 172.0 us
    d210601  + buffer pool     169.5 170.8 169.5 170.0   median 169.8 us

Spreads 0.1-0.6%. All three arms completely non-overlapping.

    fence fold   210.2 -> 172.0   -38.2 us   -18.2%
    buffer pool  172.0 -> 169.8    -2.2 us    -1.3%
    combined     210.2 -> 169.8   -40.4 us   -19.2%

**Both super-io figures were inflated — the fold by 2x (-36% claimed), the pool
by 13x (-17% claimed).** The pool was labelled "not significant here", which was
right, but the number was badly wrong too. A ~900 us noise band does not merely
fail to resolve a small effect; it manufactures a large one. Treat every
super-io per-dispatch delta in this document as an upper bound.

**The buffer pool is real but marginal.** 2.2 us for a global mutex, a
size-keyed cache, retained buffers and a contamination hazard needing two
dedicated tests. Kept because it is written, tested and harmless — but it should
be the first thing dropped if it ever complicates something.

Cumulative on 248, chaining the exmc session's earlier arms with these:

    ab2e779              365 us
    096d7bd fast OFF     238 us   8cce91c, four readback fences -> one   -35%
    096d7bd fast ON      224 us   b59c4a7, small-upload fast path         -6%
    f4c00f4              210 us
    8cd19ee              172 us   fence fold                            -18%
    d210601              170 us   buffer pool                            -1.3%

**365 -> 170 us, about -53% per chain dispatch.**

Note the arms could not be the ones originally proposed (096d7bd / 8cd19ee /
d210601): 096d7bd predates `13619fd`, so `ChainShaderSpecsF64` does not exist
there and the chain path is not driveable from this repo at that commit.
`f4c00f4` is the commit immediately before the fold that has it.

---

## What changed this session, in one place

**The chain path can be driven from this repo for the first time** (`15abc96`,
`13619fd`). It never could before: the NIF pushes a fixed 20/24-byte struct, so
family parameters declared inline in `ShaderTemplate` were silently dropped, and
the header disagreed besides. Parameters are now baked into the generated GLSL
as literals — the design eXMC arrived at independently, verified in their
SPIR-V. All six f64 families ported (`ChainShaderSpecsF64`); the Normal chain
reproduces a host leapfrog **bit-exact**, `max |GPU - host| = 0.0`. The six
hand-written `glsl/leapfrog_chain_*_f64.comp` and their `.spv` were deleted
(`8006a4d`) — a baked shader is parameter-specific, so there was no static
artifact to replace them with.

That is why this matters beyond tidiness: **a repo cannot benchmark a path it
cannot drive**, which is why `8cce91c` had to be measured downstream.

**Per-dispatch fixed cost, measured by sweeping K and separating the intercept**
(d=13, super-io):

    before this session   296.8 us fixed vs 278.5 us GPU at K=32  -> 51.6% overhead
    after 8cd19ee         225.5 us fixed
    after d210601         ~199 us fixed (not significant here)

**`buf_upload` had `alloc_buffer`'s heap bug** (`d7b5f08`). Presents as a cliff
on cumulative BAR1 pressure, not a constant tax: BAR1 saturates at 229 MiB and
throughput falls **5.5x at constant buffer size**. Invisible to tests — values
are bit-identical, the tensor genuinely is resident, only the heap is wrong.
`ab2e779` repaired the regression that fix introduced.

**We were deleting eXMC's shader cache** (`b024ad1`). `Synthesis.clear_cache/0`
is `File.rm_rf` and both projects used `~/.exmc/gpu_node/spv`;
`synthesis_test.exs` calls it in `setup` AND `on_exit`, so every `mix test` here
wiped it. Broke their in-flight suite twice in one 20-minute window. Both caches
moved under `~/.nx_vulkan/`, verified with sentinel files.

---

## Open items, ranked by value to eXMC

### 1. The ~1.9 ms in NUTS tree logic — theirs, and now the largest term

Handed over with a method rather than a guess: measure the split rather than
subtract it; establish whether it is per-dispatch or per-DRAW (a NUTS trajectory
doubles until a U-turn, so tree logic runs O(2^depth) per draw while dispatches
run per step); census what the tree does with the `3*K*d*8` bytes it gets back;
and check for per-step `:binary.part/3` or list conversion, which looks like GPU
work until a K-sweep separates it.

If it turns out irreducible, the GPU is starved **by design** at their model
sizes and the lever moves to batching draws rather than making dispatches
cheaper. That is a conclusion to reach from a measurement, not an assumption.

### 2. What is left of the ~1.0 ms in-NIF cost

Three bites taken (4 fences -> 1, 2 submissions -> 1, ~8 allocations pooled).
What remains per chain dispatch: descriptor-set construction, the command-buffer
build, the three upload buffers (NOT pooled — they go through
`upload_buffer_staged`'s small path, which uses `Buffer::from_iter` and would
need converting to allocate-then-write), and four `NewBinary` allocations.

Measure before building. The K-sweep intercept is the instrument.

### 3. Race 5 (MCMC) could now actually run

It never could before, because nothing could drive the chain path. That has
changed. `examples/` has no chain benchmark; the harness used this session is in
the scratchpad only and is worth committing if Race 5 is attempted.

### 4. Remaining fallbacks

f32 is down to **5**: `erf`, `erfc`, `atan2` (both forms), `sort`. Only `sort`
is allowlisted.

* `erf`/`erfc` are not in GLSL.std.450 and need a polynomial that agrees with
  `:math.erf` — note the f64 shader's old `erf_approx` was deleted as
  unreachable, and a series more accurate than BinaryBackend would DISAGREE
  with it, which is the wrong direction.
* `atan2` needs a new binary op code across four shaders.
* The **twelve f64 transcendental forms are a deliberate decline**, not a gap.
  `Nx.sin(f64 1)` would return 0.8414708971977234 against 0.8414709848078965.
  Admitting them turned 22 of Nx's own doctests red and would invalidate the
  78-entry residency register. `exp/log/sqrt/sigmoid/tanh` keep their f64
  boundary cast as a standing decision with `grad_test` tolerances calibrated to
  it; new ops do not inherit it.

Low value to eXMC either way — it is f64.

### 5. Jetson: `ab2e779` onward has not run there

It passed 833/876/0 at `15abc96`. Nine commits since. The unified path is a
no-op by construction for the upload/readback work (`record_readback` returns
empty, `record_upload` does nothing), but that is reasoning, and this session
twice had reasoning about that box need measuring.

**The Jetson switched to nvpmodel MAXN at ~21:35 on 2026-08-31** — 4 cores (was
2), 1479 MHz (was 918), GPU 921.6 MHz (was 640). **Every Jetson timing from
before that boundary is incomparable with anything after.** Pre-switch
reference: exmc suite 6054 s at 5W.

### 6. Race 1c voids on the Jetson even on a QUIET box

Re-raced at `d7b5f08` with load 0.40-0.73 across 14 in-run samples, thermal
control 0.0% compute drift / 0.7% allocation drift. Still voided: **estimator
divergence 30.7% against a 30% gate.** Contended it read 177.8%. So contention
inflates it ~5.8x, but clean it still sits a few points over.

That is a statement about the instrument, and it is what
`scripts/staged/jetson_run_trace.sh` was staged to gather. Still unrun. Note it
needs re-baselining after MAXN.

**Harness gap found doing it:** a VOID run still writes
`bench_results/unified_vs_discrete_<host>.json` with no void marker — the
verdict goes to stdout only. A later reader takes it as a result.

### 7. Jetson's below-cliff `alloc_buffer` is bimodal

24 and 28 MiB read 0.109-0.122 ms; 26 and 30 read 0.150-0.342. Two populations
at the same size. More interesting now that allocation is known to be
clock-invariant on Ampere.

### 8. super-io's poison flip rate is still unmeasured

Two samples, both 20/20, and one showing 19/40 effectiveness with 20/20 padding
— which breaks the lockstep claim. The cross-box table is already retired.

---

## What is settled, so nobody re-litigates it

* **The elementwise shaders are not a bottleneck on Ampere.** 431 GB/s of 448,
  measured by slope at verified boost. The "27x is now ~14x, find the rest"
  headline in the previous edition of this file was a **210 MHz reading** — the
  card's idle floor. There was never anything to find.
* **The 32 MiB allocation cliff is vulkano's**, per-allocation, reproduced 6/6
  across every box and commit.
* **The BAR1 cliff is a different cliff**: 256 MiB on super-io, cumulative
  across live host-visible buffers, a whole-process budget. Do not conflate.
* **`buf_alloc` is not clock-bound; shader work is.** 1.08-1.38x idle-to-boost
  against 3.5-4.7x. Allocation is driver bookkeeping and does not shrink when
  the card boosts.
* **The chain NIFs push a FIXED header.** 20 bytes f32, 24 f64,
  `{k_steps, n_obs, d, _pad, eps}`. Anything a caller puts past it is dropped.
  Family parameters belong in the shader source or in a buffer, never in a push
  tail.
* **The 128-byte push cap is not a width limit.** It guards bytes that never
  reach the GPU. eXMC's `d <= 13` bound was an artifact of it; packing
  header-only took an 8-RV model from 0 dispatches to 2564, **13.1x**. The real
  bound is the shader's `local_size_x = 256`, and it is still unenforced.
* **The unified-vs-discrete question is unanswered and four designs failed.**
  The fleet's GPUs differ 21x in throughput, which swamps the effect.
* **The control pair is not a pair.** mac-247 and mac-248 differ 1.39x on
  submission cost, in the opposite direction from their per-dispatch GPU work.
* **The poison-control rate is not a cross-box observable.** It moves with the
  commit.

---

## Harness invariants — every one of these cost something

### Measurement

* **Pick the host by CONTENTION, not clock observability, and they are
  anti-correlated.** super-io is the only box where `clocks.sm` reads and it is
  a desktop compositing Firefox on the same GPU: ~900 us noise band, 57% spread.
  Headless mac-248 resolves the same benchmark to **0.3%**. Anything under ~1 ms
  is unmeasurable on super-io.
* **Run a null arm on the candidate host FIRST.** A control where the change
  cannot act tells you whether the instrument can resolve the effect at all. The
  small-buffer A/B's null arm reproduced 4.6% of a 7.2% "effect" — the host
  disqualifying itself in advance, an hour before it was read that way.
* **Record the GPU clock for every timed quantity, from OUTSIDE the process.**
  Four DVFS incidents. `nvidia-smi` polled inline costs 50-100 ms of GPU idle
  and drops the boost it is measuring.
* **Sanity-check every derived figure against a physical bound.** A corrupted
  run announced itself by reporting 804 GB/s on a 448 GB/s card.
* **Amplify until the effect clears the noise floor, then replicate, then rotate
  arm order.** A 0.16 ms effect measured one call at a time sat inside 34%
  process-to-process variance on the SAME binary. Whichever arm runs first after
  a binary swap eats the cold start.
* **When verifying a FIX, the broken arm must still separate.** "Fixed looks
  like pre" proves nothing if the harness cannot resolve broken from pre either.
* **Prefer slopes to intercepts, and a measurement to a subtraction.** The
  ~0.16 ms per-fence figure was a subtraction and missed the chain path by 3x.
* **Warmup can look exactly like a leak.** 625 -> 902 -> 969 us across three
  replicates at a 300-dispatch warmup; it settles at 6000.
* **Check contention DURING the run, not just at the ends.** Sample load every
  10s and report the samples.

### Instruments that lie

* **`mix run` and `mix test` use DIFFERENT `_build` trees.** `mix test` compiles
  into `_build/test`; `mix run` reads `_build/dev`. `mix run --no-compile` after
  a green `mix test` runs STALE code, confidently. Run `mix compile` first.
* **A check that cannot run fails OPEN.** `file` was not installed in the
  cross-build image, so the ELF-architecture bar printed `command not found` and
  was never checked, while the lines around it went green. **A verification step
  that can be skipped must fail loudly when it is skipped.**
* **`nm -D` sees only dynamic symbols.** It reported 0 outline-atomics helpers
  where plain `nm` finds 22, and NIF functions are registered in a table rather
  than exported — so a symbol check is not a capability check. **Ask the
  artifact to do the thing.**
* **`pgrep -f "foo"` matches the shell running it.** A wait loop built that way
  never fires. Key on a pid via `/proc`, or use `pgrep -x`.
* **`mix` reads stdin.** Inside `ssh host 'bash -s'` with a heredoc it swallows
  the rest of the script, so trailing verification lines never run and their
  absence looks like a truncated transcript. Redirect with `< /dev/null`.
* **A swapped-in `.so` needs `--no-compile` AND a checksum**, before and after.
* **Set `Nx.default_backend/1` and your "host reference" is computed on the
  GPU.** A `max_err = 0.0` that should have been ~3e-8 is what exposed it.
  Compute references with `:math` outside Nx entirely.
* **A control that fails to trigger is not evidence the instrument is broken.**
  It is evidence of nothing until you show the control CAN trigger.
* **A green strict run means "no unlisted fallback in the TESTED paths"**, never
  "no unlisted fallbacks". Sixteen unallowlisted fallbacks sat behind a green
  `strict_test.sh` because the suite never called them.

### Code and process

* **A passing test can be evidence FOR a defect.** eXMC found five tests
  defending the 128-byte cap, four under a describe block named "the push block
  caps model width at 13 prior floats", complete with a measured table. They
  were pinning a defect in place.
* **Read the decisions file before "fixing" a gap.** MISSION.md §3.2 lists
  broadcasting `pow` under "Decisions — recorded, not oversights"; `cf7b689`
  overturned it by accident, having found the gap by census instead.
* **A stale allowlist entry silently permits the regression it describes.**
  Delete it when the gap closes, and narrow the condition rather than reusing a
  loose one.
* **A library must not put a `File.rm_rf` target inside its consumer's
  directory.**
* **Word-boundary anchor every template substitution.** `~r/\bpc\.alpha\b/`, not
  `String.replace`, or a parameter named `alpha` rewrites `pc.alpha_scale`.
  Codegen's unary templates had the same bug with the `r` in `sqrt`.
* **One template, two dtypes.** The chain skeleton is parameterised on scalar
  type rather than duplicated: a previous divergence moved the log-prob body
  above the position update and gave every distribution a one-step `logp` lag,
  blamed on the GPU for a month.
* **A control must re-measure the KIND of work it certifies.** A matmul cannot
  vouch for an allocation.
* **GC inside any loop that allocates.** A retained-allocation leak has appeared
  three times, once in a throwaway diagnostic written to investigate the
  previous two.
* **Build large test tensors with `from_binary`, not `Nx.iota`** — host-side
  construction is GPU idle time and will silently unboost the card.
* **Coordinate before adding load to a shared box.** The Jetson's failures are
  mostly `ExUnit.TimeoutError`, so on 2 cores an 85-second suite can flip
  someone else's test and the timeout is indistinguishable from a regression.

---

## Tooling added this session

* **`.claude/skills/jetson-cross-build/`** — cross-builds the Jetson's aarch64
  NIF in a container on super-io: **1m55s against ~47 min native**. Validated
  end to end. Applies to **Rust-only commits**: if an Elixir source changed, the
  box needs a real `mix compile` which drags Rustler along anyway and the skill
  buys nothing.
* **`scripts/tegrastats_bars.sh`** — live bar graph with peak markers that never
  fall, for the one box with no `clocks.sm`. NOT yet run against the live box.
* **`buf_download_many/1`** — batched readback, exposed mainly so
  `staging_read_many` is reachable from a test.

---

## Disclosure for every number in this document

Taken on super-io with a live desktop session on the same card — Firefox and
Cinnamon, 2.1-2.6 GiB resident, P0 throughout. **That caveat is the headline,
not a footnote**: it is worth ~900 us of noise, wider than most per-call effects
here. Read accordingly — the elementwise slopes and the DVFS finding are
large-signal and survive; the small per-dispatch deltas are directional and
belong on mac-248.

Numbers attributed to the exmc session were measured on **headless mac-248** or
on the Jetson, and are marked where they appear.

---

## Publishing (operator decision, not done)

* `~/projects/learn_erl/pymc/www.dataalienist.com` — two posts committed and
  **not deployed**: "An Absence Mistaken for a Discovery" and the correction
  banner on "The Copy That Wasn't There".
* `mix hex.retire nx_vulkan 0.2.0`, `upstream/main` publishing, consumer pin
  bump — all still outstanding.
