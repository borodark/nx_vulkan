# Plan — after the backward-pass audit

Actionable follow-ups from
[`docs/BACKWARD_PASS_AUDIT.md`](docs/BACKWARD_PASS_AUDIT.md), ordered by
expected value. Each item states **why it is on the list** (the measurement that
motivates it), **done when** (how you know it worked), and the risk.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` deliberately
not doing.

---

## T1 — Batched command submission `[~]`

**Built and measured on super-io; not yet fleet-raced.** Dispatches are
recorded into a pending queue and submitted as one command buffer with one
fence wait, flushed at every host boundary and at a cap (`NXV_BATCH_MAX`,
default 64; `0` restores submit-per-dispatch and is the A/B control).
**≈2× on the MNIST MLP training step (≈20 ms → ≈10 ms), loss bit-identical**,
and 1.37× on a fallback-free forward chain — the win scales with dispatch
count, as the mechanism predicts. Suite green across cap extremes and 10
consecutive runs at a cap small enough to force mid-graph flushes. Results and
the two benchmarking traps this walked into:
[`bench_results/BATCHED_DISPATCH.md`](bench_results/BATCHED_DISPATCH.md).

**Still open:** the Kepler race (247/248). Batching holds more descriptor sets
alive at once, and descriptor-pool pressure is the exact axis that produced the
Ampere `DeviceLost` and the 6× small-matmul regression — the default cap must
not be trusted on Kepler until raced. Also still open: `matmul` and
`transpose_2d` build a pipeline per call; they join the batch but that cost is
untouched, and moving them onto `get_or_create_pipeline` is a separate
measurable change.

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

**Do.** `config :nx_vulkan, host_fallback: :allow | :warn | :raise` read in
`host_result_recorded/3`, with a documented allowlist of ops that may fall back
(`pow` f64, sort, LinAlg, rank-5+, overlapping pooling). Add a CI job running
the suite under `:raise` with that allowlist.

**Done when.** The suite passes under `:raise` with an explicit allowlist, and
adding a new silent fallback fails CI rather than merely slowing things down.

**Risk.** Low, and it is the highest-leverage item for preventing recurrence —
this whole audit exists because the bug class was invisible.

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

## Housekeeping `[ ]`

- Both Keplers are parked on `feat/conv-backward-on-gpu`; restore mac.247 to
  `feat/168-ssbo-captures` and mac.248 to `f32-matmul-prototype` when done.
- The `f32_race_*_c622757.json` reports exist on each host and are not committed.
- `CHANGELOG.md` has an `## Unreleased` section; renumber to `0.3.0` at tag time
  (v0.2.0 is already tagged, so the released entry was deliberately not edited).
