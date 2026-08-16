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

Left deliberately for T4: `matmul` and `transpose_2d` build a `ShaderModule` +
`ComputePipeline` per call instead of going through `get_or_create_pipeline`.
They join the batch, but per-call pipeline construction is per-dispatch cost
too — moved separately so it can be measured separately.

**Why.** The EXLA gap is ~20× on a dense MLP and ~29× on a conv CNN, and
[fusion does not close it](bench_results/MNIST_EXLA_RACE.md) — it regresses 24%
on dense and is neutral on conv. So the deficit is per-dispatch cost. Every op
today is its own submit + fence wait in `run_single_dispatch`.

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

## T2 — Gate fusion by graph shape `[ ]`

**Why.** `Nx.Vulkan.Compiler` measures **0.76×** on dense-only and **0.98×** on
conv. It genuinely wins on elementwise-heavy chains. Presenting it as a default
win is not supported by the evidence.

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

## T3 — Strict mode: `host_fallback: :raise` `[ ]`

**Why.** `Nx.Vulkan.Fallback` makes a silent fallback *detectable if you wrote
the right assertion*. Strict mode makes it *impossible to miss*. Prior art:
PyTorch MPS requires `PYTORCH_ENABLE_MPS_FALLBACK=1` to allow CPU fallback at
all; unimplemented ops raise by default.

**Do.** Four steps, in this order:

1. **The mode itself.** `config :nx_vulkan, host_fallback: :allow | :warn |
   :raise`, read in `host_result_recorded/3` — the one funnel every fallback
   already passes through, and the one that already knows the `{fun, arity}`
   via the `__CALLER__.function` capture. `:allow` is today's behaviour and
   stays the default; a library that raises on a correct-but-slow path by
   default would be hostile to the "it always works" property that is the
   backend's main selling point.
2. **The allowlist**, as data rather than prose: a module attribute of
   `{fun, arity}` pairs each carrying the reason it is permitted, sourced from
   T7's table (`pow` f64, `window_scatter_max` overlapping, sort/argsort,
   `Nx.LinAlg` via `block/4`, rank-5+ remaps). `:raise` consults it; anything
   not listed raises with the op, the shapes, the dtypes, and a pointer to
   §1b of the `vulkan-nx-compute` skill.
3. **Scope it per-process, not globally.** The counter is already per-process
   (`Nx.Vulkan.Fallback`); strict mode must be too, or one strict test poisons
   an async suite. `Fallback.strict/2` taking a fun is the natural sibling to
   `count/1` and is what tests will actually reach for.
4. **A CI job** running the full suite under `:raise`. This is the part that
   makes it a ratchet rather than a feature.

**Watch for.** The allowlist is the whole risk surface. It must be *narrow
entries with reasons*, not a broad "these ops may fall back" — the failure
mode is an allowlist that grows silently until `:raise` means nothing, which
is how the original gates got wide in the first place. Prefer a failing test
that gets an explicit one-line exemption in the diff over a wildcard.

**Note the lower-bound property.** Strict mode inherits it: raising on the
*first* refused op is strictly better than counting, because it fires before
the tensor lands on `BinaryBackend` and takes its downstream ops with it
unrecorded. That is an argument for `:raise` over `:warn` in CI, not merely a
caveat.

**Done when.** The suite passes under `:raise` with an explicit allowlist, CI
runs it, and adding a new silent fallback fails the build rather than merely
slowing things down.

**Risk.** Low, and it is the highest-leverage item for preventing recurrence —
this whole audit exists because the bug class was invisible. [`T11`](#t11--the-gaps-the-exmc-race-found-)
is the immediate evidence that it is not a solved problem: two more instances
of the same bug class were sitting in `main` at 0.3.0, and nothing in the suite
was positioned to notice.

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

| op | action | rationale |
|---|---|---|
| `pow` f64 | `[-]` **not doing** | GLSL.std.450 has no f64 `pow`; the only fix is boundary-casting through f32, trading real precision in an f64 graph for a nicer number in a table |
| overlapping pooling backward | `[-]` **not doing** | needs `GL_EXT_shader_atomic_float`, not guaranteed on the Kepler fleet; the one-thread-per-input design is what avoids atomics |
| rank-5+ index-remap ops | `[ ]` if a workload appears | mechanical: extend `transpose_nd`/`reverse_nd`/`broadcast_nd` past rank 4 |
| `sort` / `argsort` | `[ ]` if a workload appears | large; host path is correct |
| `Nx.LinAlg` (SVD/QR/solve/cholesky) | `[ ]` unlikely | very large; correct on host today |

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

## T10 — The Ampere over-dispersion `[ ]`

**Why.** `exmc/research/D90_BACKLOG_FIX_PLAN.md` (2026-07-22) disqualifies the
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

## T11 — The gaps the eXMC race found `[ ]`

**Why.** [`bench_results/EXMC_PEROP_RACE.md`](bench_results/EXMC_PEROP_RACE.md)
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

---

## Housekeeping `[ ]`

- Both Keplers are parked on `feat/conv-backward-on-gpu`; restore mac.247 to
  `feat/168-ssbo-captures` and mac.248 to `f32-matmul-prototype` when done.
- The `f32_race_*_c622757.json` reports exist on each host and are not committed.
- ~~`ROADMAP.md` called "~12× ahead after batching" a measurement while
  `README.md` explicitly declined to claim a post-batching figure~~ — fixed:
  the 20× is the measurement, the 12× is arithmetic, and the ROADMAP now says
  so. Re-running the race needs a working EXLA, which this repo deliberately
  does not depend on.
- **`mix hex.retire nx_vulkan 0.2.0` is still not done** — hex.pm reports
  `retirement: None`. Needs the maintainer's interactive Hex local password:
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
