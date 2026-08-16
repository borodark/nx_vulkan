# Plan — after the backward-pass audit

Actionable follow-ups from
[`docs/BACKWARD_PASS_AUDIT.md`](docs/BACKWARD_PASS_AUDIT.md), ordered by
expected value. Each item states **why it is on the list** (the measurement that
motivates it), **done when** (how you know it worked), and the risk.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` deliberately
not doing.

---

## T1 — Batched command submission `[x]`

**Done and fleet-raced.** Dispatches are recorded into a pending queue and
submitted as one command buffer with one fence wait, flushed at every host
boundary and at a cap (`NXV_BATCH_MAX`, default 64; `0` restores
submit-per-dispatch and is the A/B control).

On the MNIST MLP training step: **1.71× on Ampere, 1.65× on GT 650M, 1.45× on
GT 750M**, with the loss `2.6447360515594482` in every arm on every host — two
architectures, two operating systems, bit-identical. Also 1.37× on a
fallback-free forward chain, i.e. the win scales with dispatch count, as the
mechanism predicts. Suite green on all three hosts across cap extremes, plus 10
consecutive full runs at a cap small enough to force mid-graph flushes.

**No hardware crossover**, so no `Device.class/0` gate is needed and it ships on
by default — that was not safe to assume, since the concern going in was that
holding more descriptor sets alive would show up as Kepler pool pressure.
Results: [`bench_results/BATCHED_DISPATCH.md`](bench_results/BATCHED_DISPATCH.md).

**Re-tested under concurrency, and it holds.** Those figures were all measured
with one dispatcher, while the queue producing them is a single
`OnceLock<Mutex<Vec<RecordFn>>>` static shared by every BEAM process and
`submit_and_wait` ends in a device-wide `queue.wait_idle()`. Neither costs
anything at N=1, and the deployments this backend targets are N-concurrent
(exmc runs a GenServer per instrument). Raced across N ∈ {4,8,16,32} on both
Keplers, five interleaved replicates per cell:

| N | GT 650M | GT 750M |
|---:|---:|---:|
| 4 | 1.17× | 1.01× |
| 8 | 1.22× | 1.14× |
| 16 | 1.20× | 1.06× |
| 32 | **1.33×** | 1.15× |

Batching wins at every N on both cards. **The shared bucket is not costing
measurable throughput up to 32 dispatchers**, so routing batches through
`Nx.Vulkan.Node` (`with_node/2` at the graph boundary) and owner-keyed pending
queues have **no measured motivation** and are not filed as work. A negative
result, recorded so it is not re-litigated:
[`bench_results/CONCURRENT_DISPATCH.md`](bench_results/CONCURRENT_DISPATCH.md).

Two things the race established that it was not looking for:

1. **One process under-feeds these GPUs.** Throughput roughly doubles from N=1
   to N=8 on both cards before saturating. Concurrency is worth having; the
   shared queue simply is not the obstacle to it.
2. **The GT 750M is a poor host for timing work and the GT 650M is a good one**
   — ±11–13% run-to-run spread against ±2–4%, on the same benchmark in the same
   session. This is not recorded anywhere else in the repo, and it invalidates
   single-run cells on that host: a 15% "effect" measured there is a coin flip.
   Two rounds of this race produced a confident, reproducible-looking hardware
   crossover that five replicates then erased. Race on the 650M, or replicate.

Left deliberately for T4: `matmul` and `transpose_2d` build a `ShaderModule` +
`ComputePipeline` per call instead of going through `get_or_create_pipeline`.
They join the batch, but per-call pipeline construction is per-dispatch cost
too — moved separately so it can be measured separately.

**Why.** The EXLA gap is ~20× on a dense MLP and ~29× on a conv CNN, and
[fusion does not close it](bench_results/MNIST_EXLA_RACE.md) — it regresses 24%
on dense and is neutral on conv. So the deficit is per-dispatch cost. Every op
today is its own submit + fence wait in `run_single_dispatch`.

*(Postscript, 2026-08-16: batching delivered what it promised and the gap is
structural anyway. A width sweep put EXLA on the **host CPU** 20–215× ahead of
this backend across every size tested, widening — so per-dispatch cost was a
real deficit but not the deciding one. See
[`bench_results/MODEL_SCALING.md`](bench_results/MODEL_SCALING.md) and the
reframing in `ROADMAP.md`.)*

**Do.** Record several dispatches into one command buffer and fence once.
Natural seam: `Nx.Vulkan.Compiler` already knows a whole stage schedule, so it
can batch a stage's dispatches even when it does not fuse them into one shader.
The eager path can batch opportunistically per `jit_apply`.

**Done when.** The MNIST MLP step drops materially below 14.1 ms eager on
super-io, measured with the race harness in `MNIST_EXLA_RACE.md`, losses still
bit-identical to `BinaryBackend`, fleet census unchanged at 1 fallback.

**Risk.** Medium-high. Touches synchronisation; a missed barrier is a
read-before-write race that will show up as nondeterministic wrong numbers
rather than a crash. Needs the gradient suite green on all three hosts, not
just super-io.

---

## T2 — Gate fusion by graph shape `[~]` — and first, find out why it does nothing

**The premise changed, and this item is now downstream of a mystery.**

`EXMC_PEROP_RACE.md` explained fusion's flat result by the 137 host fallbacks
in that graph: the compiler hands unsupported nodes to the Evaluator, which
hands them to the same backend callbacks. **That explanation is dead.** T11 and
T12 took the census to **0**, and fusion is still worth nothing — within noise
of per-op on **13 of 13 cells** of a width sweep, on exactly the
elementwise-heavy graph it was built for
([`bench_results/MODEL_SCALING.md`](bench_results/MODEL_SCALING.md)).

So there is no longer a story for why whole-graph compilation buys nothing
here, and that is worth more than the gating heuristic: fusion is the **only
mechanism that could close the 20–215× EXLA gap**. A heuristic that gates a
no-op is not worth building. Profile a fused dispatch against its per-op
equivalent first — is the generated shader running, is it being recompiled per
call, is the stage schedule inserting boundary copies that cost what the fusion
saved?

**Original why.** `Nx.Vulkan.Compiler` measures **0.76×** on dense-only and
**0.98×** on conv. It genuinely wins on elementwise-heavy chains. Presenting it
as a default win is not supported by the evidence.

**Do.** Follow the CSE precedent
([`bench_results/CSE_SOFTMAX_RACE.md`](bench_results/CSE_SOFTMAX_RACE.md) —
built, raced, shipped default-off). Either:
- auto-gate on a traced-graph statistic (ratio of elementwise nodes to stage
  boundaries), or
- keep it opt-in and document the shapes where it helps.

Do **not** guess the threshold — race it across the fleet, as the
`vulkan-nx-compute` skill requires.

**Done when.** A documented rule, a race table across at least Kepler + Ampere
backing it, and no graph shape where the default path is slower than eager.

**Risk.** Low. Worst case is a heuristic that is too conservative.

---

## T3 — Strict mode: `host_fallback: :raise` `[x]`

**Done.** `config :nx_vulkan, host_fallback: :allow | :warn | :raise`, read in
`host_result_recorded/3`. `:allow` is and stays the default. `:raise` raises
`Nx.Vulkan.HostFallbackError` on the first fallback not on `@allowlist` in
`lib/nx_vulkan/fallback.ex`; `Nx.Vulkan.Fallback.strict/1,2` scopes it
per-process, so a strict test cannot poison an `async: true` suite.
`sh scripts/strict_test.sh` and `.github/workflows/strict-fallback.yml` are the
ratchet.

**Allowlist** — 8 entries, one line each, no wildcards. `pow/3` (broadcast
form), `window_scatter_max/6` (overlapping windows), `sort/3`, `argsort/3`,
`triangular_solve/4`, and `transpose/3` / `reverse/3` / `broadcast/4` **gated
at rank ≥ 5** so a rank-4 fallback still raises.

**What the first strict run caught** (524 failures, one real bug class):

- **A `{:u, 8}` mask produced on the GPU can only be consumed by `select`.**
  `multiply`, `sum` and `as_type` on it all host-fall-back — there is no
  `cast_u8_to_f*` shader. `reduce_max`'s gradient is exactly this shape, so the
  **softmax backward pass leaves the GPU four times** and nobody knew. Not
  allowlisted. See T12.
- **`clip/4` was counted as a fallback it never made.** It composes GPU
  min/max and stays resident, but wrapped its result in `host_result/2`, so the
  census over-counted it and `:raise` refused it. The funnel now records only
  results that actually left the device. Two tests assert "clip stays on GPU"
  and were passing for the wrong reason.
- **`sum` / `reduce_max` / `reduce_min` were attributed to their shared
  helper.** 54 refusals said `reduce_op_host_fallback/4` and named no Nx op.
  Fixed with the explicit-attribution `host_result/3` the skill already
  prescribes.
- **`block/4` is invisible to the counter *and* to strict mode.** Everything
  nx 0.13 routes through it — `Nx.LinAlg` svd/qr/lu/cholesky/solve/eigh/
  determinant, `top_k`, `cumulative_*`, `all_close` — transfers to
  `BinaryBackend` without passing the funnel. So "LinAlg" is not an allowlist
  entry: it cannot be refused, because it is never seen. See T13.
- **The T7 table's `pow` line was wrong.** Equal-shape f32 *and* f64 `pow` run
  on the GPU; only the **broadcasting** form falls back, for either dtype,
  because `elementwise_binary_bcast_*` omits op code 4. Corrected below.
- The rank-5+ family is wider than T7 said: broadcasting **elementwise binary**
  (`add`, `multiply`, …) is also capped at rank ≤ 4, not just the remap ops.

**Corrected in the process:** the suite is green under `:raise` with two
enumerated, greppable exclusions — `:host_fallback_expected` (tests whose
subject *is* the fallback path, incl. `doctest Nx`, which is an
API-completeness suite over `{:s, 32}` and not a residency one) and
`:host_fallback_open` (the two `grad_test.exs` cases blocked on T12). Neither
is skipped by a normal `mix test`.

---

## T4 — Reduce dispatch cost structurally `[ ]`

**Why.** Same evidence as T1; these are the other two levers.

- **Pooled / persistent buffers.** Every op `buf_alloc`s its output. Long
  standing roadmap item; now has a measured motivation.
- **Better GEMM.** The register-blocked `*_rb32` shaders exist but regress on
  both Keplers, so they are benchmark-only. Revisit behind
  `Nx.Vulkan.Device.class/0` gating, which already exists for exactly this kind
  of hardware-dependent win.

**Done when.** Allocation no longer appears in a per-step profile; `matmul`
improves on Ampere without regressing Kepler (fleet-raced).

**Risk.** Medium. Buffer reuse interacts with Rustler resource lifetimes — the
current design's safety property is that a `Subbuffer` cannot outlive its
`Buffer`, and pooling must not break that.

---

## T5 — Upstream: ship `Nx.Helpers.check_grads!` in `Nx.Testing` `[ ]`

**Why.** Upstream already has a central-differences gradient checker and uses
it 33 times across a 6031-line `grad_test.exs`, but both live under `nx/test/`,
which the Hex package does not ship. Every third-party backend author therefore
reinvents gradient testing or skips it — Torchx has zero gradient tests.

**Do.** PR moving `check_grads!/4` into the already-shipped `Nx.Testing`.
Smallest useful upstream change available.

**Done when.** Merged, or declined with a reason worth recording.

**Risk.** None to this repo.

---

## T6 — Report the XLA gradient tiling bug `[ ]`

**Why.** EXLA fails to compile the gradient of **two stacked convs + stride 2 +
`channels: :first`** (forward compiles; any single relaxation compiles). It is
narrow and Axon's default layout avoids it, but it is a real upstream bug and we
have a minimal reproducer.

**Do.** File against `elixir-nx/nx` (or XLA) with the 17-variant matrix from
[`bench_results/MNIST_EXLA_RACE.md`](bench_results/MNIST_EXLA_RACE.md).

**Done when.** Filed with a reproducer someone else can run.

**Risk.** None. Note we corrected an overstated version of this claim once
already — keep the report scoped to exactly what was measured.

---

## T7 — Remaining host fallbacks `[ ]`

Corrected against the code by T3's strict run — three of these lines were
inaccurate as written.

| op | action | rationale |
|---|---|---|
| **broadcasting** `pow` (any dtype) | `[-]` **not doing** | *Corrected:* equal-shape f32 **and f64** `pow` already run on the GPU. Only the broadcasting form falls back, because `elementwise_binary_bcast_*` omits op code 4 (GLSL.std.450 has no f64 `pow`). Fixing it means boundary-casting through f32, trading real precision for a nicer table |
| overlapping pooling backward | `[-]` **not doing** | needs `GL_EXT_shader_atomic_float`, not guaranteed on the Kepler fleet; the one-thread-per-input design is what avoids atomics |
| rank-5+ index-remap ops | `[ ]` if a workload appears | mechanical: extend `transpose_nd`/`reverse_nd`/`broadcast_nd` past rank 4. *Corrected:* broadcasting **elementwise binary** is capped at rank ≤ 4 too |
| `sort` / `argsort` | `[ ]` if a workload appears | large; host path is correct |
| `Nx.LinAlg` (SVD/QR/solve/cholesky) | `[ ]` unlikely | very large; correct on host today. *Corrected:* these go through `block/4`, which is **not instrumented at all** — see T13. `triangular_solve/4` is the only one still an `Nx.Backend` callback |

**Done when.** Each line is either done or has a recorded reason it is not — the
point is that "still a fallback" is a decision, not an oversight.

---

## T8 — Unify the index-remap shader family `[ ]`

**Why.** `transpose_nd`, `reverse_nd`, `broadcast_nd` share one skeleton:
decompose the output index, map to an input index, copy. They differ only in the
mapping rule.

**Do.** At a **fourth** member, unify behind one shader with a mode selector.
Three does not justify the indirection — do not do this speculatively.

**Done when.** A fourth remap op is needed and lands as a mode rather than a
file.

**Risk.** Low, but premature unification is its own cost. Trigger-based on
purpose.

---

## T9 — Fleet race automation `[ ]`

**Why.** `scripts/race.sh` writes `bench_results/f32_race_<host>_<commit>.json`
per host, but collecting them is a manual SSH sequence, and the
`vulkan-nx-compute` skill requires every perf heuristic be fleet-validated. That
friction is why heuristics get validated on one box.

**Do.** A script that fans out to the fleet, runs the race, and gathers the
JSONs into `bench_results/`. Must fast-forward the branch explicitly — `git
checkout` alone leaves a stale local branch and silently benchmarks the wrong
commit (this happened during the audit).

**Done when.** One command produces a committed, per-host, per-commit race set.

**Risk.** Low. Guard against running while another benchmark is in flight on the
same host — contended measurements are worse than none.

---

## T10 — The Ampere over-dispersion `[x]`

**Done — and the defect was in neither the card nor this repo.**
Diagnosis: [`docs/T10_AMPERE_DISPERSION.md`](docs/T10_AMPERE_DISPERSION.md).

The fixed-input differential exonerated the shader in one step, as designed:
Ampere against a Kepler oracle, same GLSL, byte-identical inputs, K ∈ {1,2,4,32}
— `q_chain`, `p_chain`, `grad_chain` **bit-identical in all 8 cases**, one
log-density differing by 1 ulp. A 1-ulp difference cannot produce a 5.9×
variance ratio. No barrier, race, or warp-width hypothesis was needed, and the
one this plan proposed as most likely was wrong.

The actual cause is in eXMC: `multi_rv_custom_spec.ex`'s `@template` evaluates
`{{prior_logp_body_q}}` **before** the position update, so `logp_chain[k]` is
`log p(q_chain[k−1])` while `q/p/grad_chain[k]` hold post-step state.
`Tree.synth_chain_subtree/10` pairs them by index, so every NUTS leaf carries a
stale, systematically-too-high density and the multinomial over-weights the far
end of each trajectory. Mis-scaled not mis-signed, no divergences, adaptation
pushes `eps` up — the observed signature exactly. Moving the body below the
update takes Normal(0,1) variance from 8.5467 to 1.4481 against a CPU 1.4523.

**The second finding is the larger one.** `Validator.run_exla/2` selected its
reference by clearing `:exmc, :compiler` and letting `auto_detect/0` run, and
that is `EXLA -> Nx.Vulkan -> nil`. On any host without EXLA the reference arm
*was* the candidate backend. FreeBSD has no EXLA at all, so **mac-247's 16/0
was Vulkan compared against itself** and established nothing; super-io scored
8/16 because it was the only host actually running the test. The fleet's
verdict — "super-io is unfit for numerical validation, the macs are the
reference" — was precisely inverted, and it stood for three weeks over a live
defect. Fixed in exmc (`c69a6a6bf`): the reference is pinned explicitly to
`:exla`, or `:none` where EXLA is absent, and raises if it ever resolves onto
the candidate again. It now returns `{:error, %{check: :variance, ...}}` on the
model it used to pass.

**The transferable lesson:** a comparison harness must assert that its two arms
are actually two. This one had no such assertion and could not fail.

**Why (as filed).** `exmc/research/D90_BACKLOG_FIX_PLAN.md` (2026-07-22) disqualifies the
RTX 3060 Ti as a host for sampling-accuracy validation: its Vulkan leapfrog is
**systematically over-dispersed across every distribution**, deterministically
and reproducibly — Normal(0,1) variance **8.5 against EXLA's 1.45**,
Exponential mean **2.77 against 0.5**, 8/16 `validator_test.exs` failures where
mac-247 (Kepler, MoltenVK) shows 2/16 and then 16/0 after G1/G4. The fleet's
strongest GPU is currently trusted for compile-and-crash checks only, and the
reference for numerical correctness is a 2012 laptop card. That is backwards,
and it has stood unexplained for three weeks.

It is also, right now, **an accusation without a defendant.** Nobody has
established whether the bug is in this repo at all.

**Do.** Bisect by layer before touching a shader. The point of the order below
is that each step is cheap and each one can exonerate a whole layer:

1. **Is it the backend or the synthesised chain shader?** This repo's own
   suite — including `grad_test.exs`, which compares `Nx.Defn.grad` against
   `BinaryBackend` op by op — is **green on this exact card**, and
   [`bench_results/EXMC_PEROP_RACE.md`](bench_results/EXMC_PEROP_RACE.md)
   measured log_p agreeing with the host to 7 significant figures on it. Both
   are evidence for the chain shader and against the op set. Make it decisive:
   run eXMC's validator battery on the **per-op path** (`chain_meta` stripped,
   as that harness does) on super-io. If per-op is well-dispersed and chain is
   not, the bug is in `Exmc.NUTS.CustomSynth`'s generated GLSL, this repo is
   exonerated, and the work moves to the exmc side.
2. **If it is the chain shader — differential, not statistical.** Do not chase
   it through posteriors; sampling error is the fog this bug has been hiding
   in for a month. Dispatch **one** `leapfrog_chain_synth` step with a fixed
   `(q, p, eps, inv_mass)` on super-io and on mac-247 and diff the outputs
   elementwise. Kepler↔Kepler is documented bit-identical, so a Kepler is a
   trustworthy oracle. A single divergent step localises this in one run;
   1,000 sampled iterations localise nothing.
3. **Then bisect within the step.** Same fixed inputs, compare the chain
   shader's output against the same leapfrog composed from eager ops on the
   same card. That splits "the synthesised GLSL is wrong" from "a kernel it
   calls is wrong on Ampere".

**Hypotheses worth holding, in the order the evidence favours them.** An
over-dispersed chain with *no* divergences and a mis-scaled — not mis-signed —
step is the signature of a gradient that is too small or an `eps` that is too
large, not of a numerically noisy one. Warp width is the obvious architectural
difference (32 on both, but occupancy and scheduling are not), as is anything
in the shader that assumes a workgroup reduction completes without a barrier —
a missing `barrier()`/`memoryBarrierShared()` can be benign on a 2012 part
whose scheduler serialises what Ampere runs concurrently. That class of bug is
**exactly** "correct on weak hardware, wrong on strong", which is the shape of
the observation.

**Done when.** The defect is localised to a repo and a layer, with a fixed-input
reproducer that does not involve sampling. Whether it is then *fixed* is a
separate decision — but "super-io is not valid for numerical validation" stops
being a standing exclusion and becomes either a fixed bug or a documented one.

**Risk.** Medium, and mostly of the sunk-cost kind: this is a real
cross-architecture numerics investigation and step 1 may hand the whole thing
to another repo. That is a good outcome, not a wasted day — the current state
is that neither repo has ruled itself out.

---

## T11 — The gaps the eXMC race found `[x]`

**Done. Census 137 → 0**, `logp` unchanged at 7 significant figures against the
host. Suite 843 doctests / 423 tests / 0 failures, green under `NXV_BATCH_MAX=0`
and `=2`.

Three gates were refusing shapes the shaders already handled, and only one new
kernel was needed:

- `compare` (all six) and `select/4`: `>= 1` dropped. **No GLSL change** —
  `pad4/1` already pads with 1s and `pad_left/2` lifts a scalar; the loop bound
  was `0` where it wanted `max(rank, 1)`.
- `bcast_shape_ok?/3`: a third instance, found by the audit. Rank 0 reaches the
  broadcast path only when the two scalars differ in dtype, which is why
  nothing noticed — and it is where the census's 12 `multiply` came from.
- **`pad` already had a shader** (`glsl/pad.comp`, since thrust 2 — `LIMITATIONS.md`
  §2 and this item's original text were both wrong). It was refused by
  `pv.type == t.type`: `Nx.pad(t, 0.0, cfg)` hands the callback an f32/s32
  literal. The `{:s,32}` trap from §1b again.
- `put_slice`: the one genuinely new shader.

Two upstream bugs surfaced and were deliberately **not** fixed, because matching
the reference means reproducing it: `BinaryBackend.pad/4` casts the pad value
but not the tensor, returning a malformed tensor when the value is wider; and
`BinaryBackend.put_slice/5` raises on rank 0. The latter is why `put_slice`
keeps `rank >= 1` — a rank gate *with* evidence, unlike the one removed.

**T8 answered:** do not unify. The family is already seven shaders, so the
"fourth member" trigger fired long ago unnoticed, and the members differ in
*arity and bindings*, not just mapping rule — a unified shader needs the union
of bindings and dummy buffers for unused slots. The duplication worth removing
is different: port `transpose_nd`/`reverse_nd`/`broadcast_nd` to the
type-generic `ews` word-copy form `slice`/`pad`/`put_slice` already use, which
collapses 6 files to 3 and gives those ops integer-dtype support they currently
refuse.

**Why (as filed).** [`bench_results/EXMC_PEROP_RACE.md`](bench_results/EXMC_PEROP_RACE.md)
counted **137 host fallbacks in a single `value_and_grad`** of a d=8 model, and
two causes account for all of them:

- **`compare` and `select` refuse rank 0.** Both gates read
  `tuple_size(out.shape) >= 1 and tuple_size(out.shape) <= 4`
  (`vulkano_backend.ex:1087`, `:1148`). The upper bound is the documented
  rank-4 remap ceiling. The lower bound has **no shader justification** —
  elementwise arithmetic handles scalars on the same host without complaint.
  108 of the 137 are this one guard, refusing the scalar support checks
  (`x > 0` and friends) that every distribution log-prob is made of.
- **`pad` and `put_slice` have no shader at any rank.** `PointMap` unpacks the
  parameter vector with `put_slice`, once per RV. This is the one that decides
  residency: the census is a **lower bound**, so once the position vector lands
  on `BinaryBackend` everything after it computes there unrecorded.

The first is **§1b of the `vulkan-nx-compute` skill, a third time** — a gate
written against the shapes one workload produces. The audit's two instances
were a forward pass refusing its own gradient and Nx's `{:s, 32}` literals.
This one is a neural-network workload, where predicates are batched masks of
rank ≥ 1, refusing a probabilistic one. It survived a release because nothing
in the suite compares scalars on the GPU.

**Do.**
1. `>= 1` → `>= 0` in both gates, reshaping rank-0 to `{1}` for dispatch and
   back. Add `fallback_test.exs` assertions **at zero** for scalar
   `equal`/`greater`/`select`, in the "must not leave the GPU" describe block.
2. `put_slice` and `pad` shaders. Both are index-remap-family members
   (decompose the output index, map to an input index, copy) — the skeleton
   exists three times over in `transpose_nd`/`reverse_nd`/`broadcast_nd`. Per
   [`T8`](#t8--unify-the-index-remap-shader-family-), a **fourth** member is
   the stated trigger to unify behind one shader with a mode selector; two more
   arriving together makes that decision live rather than hypothetical.

**Done when.** A scalar compare/select performs zero fallbacks, `put_slice` and
`pad` dispatch natively, and the eXMC harness is re-run to a recorded census.

**Do not expect it to be a speed win for eXMC.** The same measurement found the
chain shader running **1.7× slower than `BinaryBackend`** on this host at d=8:
the model is dispatch-bound, and closing fallbacks does not change that. Build
these because they are real gaps in a general-purpose backend, and because the
scalar gate is a correctness-adjacent defect that will bite the next
probabilistic consumer. Not because a d=8 posterior will get faster.

**Risk.** Low. Both are mechanical, and both are exactly the kind of gap
[`T3`](#t3--strict-mode-host_fallback-raise-) exists to stop shipping.
## T12 — `cast_u8_to_f32/f64`: unstick the comparison mask `[x]`

**Done.** softmax backward 2 fallbacks → **0**, and the two
`:host_fallback_open` tags this item created in `grad_test.exs` are
**tags removed** — verified passing under `NXV_HOST_FALLBACK=raise`
before deleting, as this item's own "done when" required, `reduce_max`'s gradient exact on
ties (`[0.5, 0.5, 0.0]`, ⅓ each), bit-identical to `BinaryBackend` across a
60-cell (shape, axes, op) sweep. Three pieces were needed, not one: the cast
alone moved the census from `{multiply, sum}` to `{divide, sum}` — same total,
because `sum` of a u8 is typed `{:u, 32}` by Nx, so the *output* is integer and
no input cast reaches it. `reduce_axis_u8_to_u32` closed that, and
`cast_u32_to_f32/f64` closed the divide-by-tie-count left stranded after.
Pinned as still-host: middle-axis u8 `sum` (that path transposes first and
`transpose_nd` has no u8 path) and `reduce_max`/`reduce_min` on u8 (Nx keeps
their output at `{:u, 8}`, needing a byte-packed writer).

**Why (as filed).**

**Why.** Found by T3's first strict run, and it is instance nine of the §1
defect class. The compare shaders produce a `{:u, 8}` mask on-device, which was
the point — but **`select/4` is the only op that can consume it.** `multiply`,
`sum` and `as_type` on a u8 mask all host-fall-back, because `coerce_to/2` has
no u8 cast and the reduce/binary shaders have no u8 path.

`Nx.Defn.Grad` routes `reduce_max`'s gradient through exactly that mask, so
**softmax's backward pass leaves the GPU four times** — two `multiply`, two
`sum` — on a tensor-sized payload. Nothing detected it: the values are
bit-identical, and `grad_test.exs` only asserted values.

This is the same shape as audit instance #6, which `cast_s32_to_f32/f64` fixed,
and the same lesson as §1's `{:s, 32}` literals: *the dtype the gate refuses is
one the backend itself produced.*

**Do.** Add `cast_u8_to_f32.comp` / `cast_u8_to_f64.comp` on the existing cast
skeleton and wire them into `cast_spv/2`, which unblocks `as_type` directly and
`multiply` via `coerce_to/2`. `sum` over a u8 mask needs a decision separately
(cast-then-reduce, or a u8 path in `reduce_axis_*`).

**Done when.** `Nx.Vulkan.Fallback.count_total/1` is 0 for the softmax and
`reduce_max` gradients, and the two `@tag :host_fallback_open` tags in
`test/nx_vulkan/grad_test.exs` are deleted rather than moved.

**Risk.** Low. One new shader on a skeleton that exists three times already,
and its correctness test is bit-equality against `BinaryBackend`.

---

## T13 — Instrument `block/4` `[x]`

**Done.** Keyed `{:block, Nx.Block.Foo}` per struct, so a missing
`cumulative_sum` shader and `Nx.all_close` are separately decidable. Fixed on
the way in: this backend's `block/4` had its first two parameters **misnamed**
against the `Nx.Backend` contract (`block(struct, output, args, fun)`), which
the first code to read them promptly tripped over.

Two things it revealed. An `Nx.LinAlg.svd/2` records **~350** fallbacks, not
one — nx composes it from ordinary ops whose intermediates return here one at a
time. And the count depends on the process **default backend**, not the input
tensor: the same call records 1 with `BinaryBackend` as default and 350+ with
Vulkano. A census is a statement about a process.

Separately found and **not fixed**: `Nx.LinAlg.solve/2` raises `ArithmeticError`
on this backend, verified pre-existing on clean `main`.

**Why (as filed).**

**Why.** `block/4` is how nx 0.13 routes `Nx.LinAlg` (svd/qr/lu/cholesky/solve/
eigh/determinant), `top_k`, `cumulative_*` and `all_close`. It transfers every
input to `Nx.BinaryBackend` and never touches `host_result/2`, so that entire
family is invisible to `Nx.Vulkan.Fallback.count/1` **and** to strict mode. A
green strict run says nothing about it.

**Do.** Record it — but attribute per `Nx.Block.*` struct, not as one
`{:block, 4}`. A single entry would have to be allowlisted wholesale, which is
precisely the op-family wildcard `@allowlist` forbids; it would also make
`Nx.all_close` (used as an assertion helper throughout the suite) raise under
`:raise`. Needs a key shape the allowlist can express, so it is a design
question, not a one-liner.

**Done when.** A LinAlg call shows up in a census, and a `cumulative_sum`
fallback can be refused without also refusing `all_close`.

**Risk.** Low-medium. Changes census composition, so the pinned counts in
`fallback_test.exs` need re-reading rather than re-baselining.

---

## Housekeeping `[ ]`

- ~~Both Keplers are parked on `feat/conv-backward-on-gpu`~~ — done: both are
  on `main` (verified 2026-08-16 over ssh). They were fast-forwarded for the
  concurrency race and left there, which is the right place for them.
- The `f32_race_*.json` reports (7 of them) sit uncommitted in
  `bench_results/`. Either commit them as the per-host record they were
  written to be, or delete them — an uncommitted benchmark is one nobody else
  can check.
- ~~`ROADMAP.md` called "~12× ahead after batching" a measurement while
  `README.md` explicitly declined to claim a post-batching figure~~ — fixed:
  the 20× is the measurement, the 12× is arithmetic, and the ROADMAP now says
  so. Re-running the race needs a working EXLA, which this repo deliberately
  does not depend on.
- **`mix hex.retire nx_vulkan 0.2.0` is still not done** — re-checked against
  the hex.pm API on 2026-08-16, still `retirement: None`. Needs the
  maintainer's interactive Hex password, so it cannot be scripted. Needs the maintainer's interactive Hex local password:
  ```
  mix hex.retire nx_vulkan 0.2.0 deprecated --message "Backward pass ran on the host: GPU training was ~250x slower than advertised. Results were correct; use 0.3.0 for training."
  ```
  `deprecated` rather than `defect` is the accurate reason code: 0.2.0's
  results were correct, its performance was not what the README claimed.
- ~~`CHANGELOG.md` has an `## Unreleased` section; renumber to `0.3.0` at tag
  time~~ — done: `## 0.3.0 (2026-08-08)`, `mix.exs` at `0.3.0`, merged to
  `main`. The v0.2.0 entry was deliberately left unedited, since it is tagged.
  Remaining before publish: re-check the README's performance claims (last
  updated pre-batching), then `mix hex.publish`. Consider
  `mix hex.retire nx_vulkan 0.2.0 defect` pointing at 0.3.0 — 0.2.0 is correct
  but was an inference backend published as a training backend.
