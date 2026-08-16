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

## T12 — `cast_u8_to_f32/f64`: unstick the comparison mask `[ ]`

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

## T13 — Instrument `block/4` `[ ]`

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

- Both Keplers are parked on `feat/conv-backward-on-gpu`; restore mac.247 to
  `feat/168-ssbo-captures` and mac.248 to `f32-matmul-prototype` when done.
- The `f32_race_*_c622757.json` reports exist on each host and are not committed.
- ~~`CHANGELOG.md` has an `## Unreleased` section; renumber to `0.3.0` at tag
  time~~ — done: `## 0.3.0 (2026-08-08)`, `mix.exs` at `0.3.0`, merged to
  `main`. The v0.2.0 entry was deliberately left unedited, since it is tagged.
  Remaining before publish: re-check the README's performance claims (last
  updated pre-batching), then `mix hex.publish`. Consider
  `mix hex.retire nx_vulkan 0.2.0 defect` pointing at 0.3.0 — 0.2.0 is correct
  but was an inference backend published as a training backend.
