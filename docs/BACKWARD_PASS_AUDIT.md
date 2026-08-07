# The backward-pass audit — what we learned, and where to go next

**Scope:** the work on `feat/conv-backward-on-gpu` (`fb6221d`..`a315879`), which
began as "why is conv's gradient slow" and ended with a CNN training step going
from **20.9 s to 84 ms** and eleven host fallbacks down to one.

This document is the synthesis: the findings that generalise, the mistakes worth
not repeating, and the directions the evidence actually supports. Numbers here
are measured on the fleet (RTX 3060 Ti / GT 650M / GT 750M) unless stated.

---

## 1. The defect class: gates written against the forward pass

Every GPU fast path in this backend was guarded by a predicate describing the
shapes a *forward* pass produces. `Nx.Defn.Grad` emits shapes no human writes,
so those predicates refused the backward pass and silently routed it to the CPU.
Eight instances, found one at a time:

| # | op | rejected on | needed |
|---|---|---|---|
| 1 | conv | non-identity permutations | transpose into native layout |
| 2 | conv | mixed dtype | coerce operand |
| 3 | dot | contraction axes `[1]/[1]`, `[0]/[0]` | rotate into `(M,K)·(K,N)` |
| 4 | dot | mixed dtype | coerce operand |
| 5 | max/divide/greater | rank-0 **integer** operand | rebuild scalar at target type |
| 6 | select | **integer tensor** operand | s32→float cast shaders |
| 7 | reduce | kept axis in the *middle* | rotate kept axes to front |
| 8 | window_scatter_max | integer `init_value` | coerce operand |

**Six of eight were narrow gates, not missing capability.** The shaders could
already do the work and were being refused it. Only `reverse` and `broadcast`
needed genuinely new kernels, and both reused the index-remap skeleton written
for `transpose_nd` in the first commit.

Two sub-patterns worth naming, because they will recur:

**Nx materialises literals as `{:s, 32}`.** `max(x, 0)`, a mean's divisor,
`select`'s zeros, pooling's `init_value` — four different ops, one cause. A
four-byte constant was dragging `{32,16,14,14}` tensors to the CPU. *Any gate in
this codebase that demands an exact dtype match is a gate that will eventually be
wrong.* Coerce-then-check is the correct default.

**One generic shader unlocked three unrelated fallbacks.** `transpose_nd` was
written to fix conv permutations; it then enabled `dot`'s contraction-axis
rotation and `reduce`'s middle-axis rotation. Normalise-then-dispatch beats a
kernel per layout — the same principle as im2col reducing conv to GEMM.

---

## 2. Why nothing caught it: fallbacks are bit-identical

A host fallback transfers to `Nx.BinaryBackend`, computes, and transfers back.
It returns **exactly** the result the reference returns — because it *is* the
reference. So no assertion on values can detect that an op left the GPU. The
safety net and the blind spot are the same mechanism.

Compounding it, and verified in `deps/`:

- `Nx`'s doctests contain **zero** gradient examples, and `doctest Nx` with the
  backend as default is the community standard for validating a backend.
- The Hex package ships `lib/**` only, so `deps/nx/test` does not exist — a
  backend author cannot run Nx's own suite.
- Upstream *does* have `Nx.Helpers.check_grads!` (central differences) and a
  6031-line, 293-test `grad_test.exs`, both behind that packaging wall.
- Torchx has **zero** gradient tests; EXLA has ~4, all corner cases.

Neither reference backend has a blanket silent fallback (EXLA touches
`BinaryBackend` in 3 places; Torchx raises), so nobody upstream has this bug
class — and a shared conformance kit that only checked *values* would not have
caught it either. Detail in
[`BACKEND_VERIFICATION_GAP.md`](BACKEND_VERIFICATION_GAP.md).

**The durable fix is `Nx.Vulkan.Fallback`** — count fallbacks, assert zero. It
found instances 3–8 in minutes each, against a day of measurement and source
reading for instance 1.

### The counter's own limitation

**The count is a lower bound.** Only ops reaching *this* backend are counted;
once a fallback strands a tensor on `BinaryBackend`, Nx dispatches everything
downstream there and we never see it. This bit four times — fixing an op made
*new* fallbacks appear (`window_scatter_max`, `select`, `reduce`). A rising
count can mean the fix worked. Read composition, not totals.

---

## 3. Performance is not linear in fallback count

The most counter-intuitive result. Removing fallbacks 11 → 3 barely moved the
clock; 3 → 1 moved everything:

| | strided CNN | LeNet |
|---|---:|---:|
| before | 12 672 ms | 20 929 ms |
| after | **31 ms** | **84 ms** |

Cost is dominated by **the largest tensor that leaves the device**, not by the
number of ops that leave. Ten cheap fallbacks on `{32,10}` cost less than one on
`{32,16,14,14}`, because the host leg is pure-Elixir `BinaryBackend` and scales
with elements. Several intermediate fixes showed *no* wall-clock gain and were
still correct and necessary.

**A stopwatch-driven process would have stopped after the second fix.** The
census kept saying work was moving on-device while the clock said nothing had
changed. The census was right.

---

## 4. Fusion is not the lever we assumed

Raced against EXLA on two graph shapes
([`../bench_results/MNIST_EXLA_RACE.md`](../bench_results/MNIST_EXLA_RACE.md)):

| graph | vulkan eager | vulkan fused | EXLA | fused vs eager |
|---|---:|---:|---:|---:|
| MNIST MLP (dense-only) | 14.1 ms | 18.5 ms | 0.715 ms | **0.76×** |
| 2× strided conv CNN | 41.3 ms | 42.2 ms | 1.45 ms | 0.98× |

`Nx.Vulkan.Compiler` **regresses 24%** on the dense graph and is neutral on the
conv graph. It splits stages at `dot` boundaries, so a graph that is mostly
`dot` gives its tracing, scheduling and boundary buffers nothing to amortise
against. This is the same shape as the cross-stage CSE result
([`../bench_results/CSE_SOFTMAX_RACE.md`](../bench_results/CSE_SOFTMAX_RACE.md)):
principled, correct, and a regression on the wrong graph.

**Therefore the ~20–29× EXLA gap is dispatch overhead and GEMM quality, not
missing whole-graph compilation.** Anyone reading the timings and concluding "we
need more fusion" would build the wrong thing. That is the single most
actionable finding here.

---

## 5. Fleet observations

- **The fallback census is byte-identical on all three GPUs**, every round, for
  eleven rounds. These are code-path properties, not device-capability gates —
  so every fix benefits the whole fleet equally.
- **Absolute GPU times cluster in 25–85 ms** across cards spanning 2012–2021. At
  this model size the work is dispatch-bound, not compute-bound, which is why a
  2012 GT 650M keeps pace with a 2021 3060 Ti.
- **Speedup multipliers mostly measure the CPU you escape**, and super-io's
  `BinaryBackend` leg is consistently slower than the FreeBSD boxes'. Prefer
  absolute times in any published table.
- **f32 matmul is slower than f64** (0.45–0.61× at 512³) because f32 matmul
  defaults to an f64 accumulator. Working as designed; surprising in a table.

---

## 6. Test-methodology lessons (all learned the hard way here)

**Test inputs cleaner than production inputs pass while production fails.** The
pooling parity test built `init_value` as `Nx.tensor(0.0)` — a float. Nx passes
`{:s, 32}`. The test passed; the real path fell back. Construct inputs the way
the real caller does.

**Ties must be built, not hoped for.** `window_scatter_max` gives the gradient to
the *last* maximum in row-major order (verified against `BinaryBackend`), so the
shader needs `>=`, not `>`. With `>` it is correct on random data and wrong
wherever values repeat — and a relu's output is full of exact ties at zero.
`remainder()` data creates ties by construction; random floats never do.

**Sometimes the reference is the broken one.**
`Nx.BinaryBackend.window_scatter_max/5` round-trips f64 through f32
(`2.4715269558223154` → `2.471526861190796`). For f64 pooling gradients this
backend is now *more accurate than the thing it is tested against*. The test
asserts values are exact elements of `src` — stronger than agreement, and it
does not inherit the reference's defect.

**Tests that pin "this doesn't work yet" break when you fix things — by
design.** Four broke here (`ConvTest`, `TransposeTest`, the counter's own
meta-tests, `vulkano_backend_test`). Deliberate ones belong in
`fallback_test.exs` and are *meant* to fail. The dangerous kind is incidental
scaffolding — a test using a falling-back op as a vehicle for testing something
else. Those now point at `sort`, which has no shader and no plan for one.

**Moving ops on-device costs doctests.** `standard_deviation` and `covariance`
joined the `@rounding` bucket (863 → 851 doctests) because they now compute
natively and land 1 ULP away. Both drifts verified sub-ULP. Expect more; watch
the bucket rather than growing it silently.

---

## 7. Process lessons

**Isolation matrices are cheap; conclusions from one observation are not.**
Three claims in this session were wrong and needed correcting: a `max_pool`
"bug" in Nx that was my own channel-order error; `examples/full_bench.exs`
"missing" from a truncated listing; and — worst — a committed claim that EXLA
"failed to compile conv", which a 17-variant matrix reduced to *two stacked
convs + stride 2 + `channels: :first`, gradient only*, with Axon's default
layout unaffected. That error flattered this project at a competitor's expense,
which is exactly the direction to be most suspicious of.

**Verify extraordinary results before reporting them.** A 118× jump got a
forced-full-readback re-measurement before it was believed (it held). An earlier
race reported 635× from a model silently producing NaN; every race row now
prints its loss and is excluded from ratios if it is not a number.

---

## 8. Where to go next

Ordered by expected value, with the evidence each rests on. These are tracked as
actionable items — with a "done when" and a risk for each — in
[`PLAN_AFTER_BACKWARD_PASS.md`](../PLAN_AFTER_BACKWARD_PASS.md).

### 8.1 Cut dispatch overhead (§4 says this is the gap)

The EXLA deficit is per-dispatch cost and GEMM quality. Options, cheapest first:

- **Batch command submission.** Every op is currently its own submit + fence
  wait (`run_single_dispatch`). Recording several dispatches into one command
  buffer and fencing once would attack the dominant cost directly.
- **Persistent/pooled buffers.** Still on the roadmap; every op allocates its
  output afresh.
- **Better GEMM.** The register-blocked `*_rb32` variants exist but regress on
  Kepler and are benchmark-only. Revisit with device-class gating.

### 8.2 Make fusion earn its place, per graph shape

`Nx.Vulkan.Compiler` is a regression on dot-dominated graphs (0.76×) and neutral
on conv (0.98×). It genuinely wins on elementwise-heavy chains. Follow the CSE
precedent: **measure, then gate**. Either auto-gate on the ratio of elementwise
to boundary ops in the traced graph, or document the shapes where it helps and
stop presenting it as a default win.

### 8.3 Strict mode — make this bug class unshippable

From the research memo: PyTorch MPS makes the CPU fallback **opt-in**
(`PYTORCH_ENABLE_MPS_FALLBACK=1`); unimplemented ops raise. The analogue here is
`config :exmc, :nx_vulkan, host_fallback: :raise` plus a CI job with a
documented allowlist. That converts "detectable if you wrote the right
assertion" into "impossible to miss". The counter makes detection *possible*;
strict mode would make it *automatic*.

### 8.4 Finish the remaining fallbacks — but not all of them

| remaining | recommendation |
|---|---|
| `pow` f64 | **leave it.** GLSL.std.450 has no f64 `pow`; the only fix is boundary-casting through f32, trading real precision for a nicer table |
| overlapping pooling backward | needs float atomics (`GL_EXT_shader_atomic_float`), not guaranteed on Kepler — leave gated |
| rank-5+ index-remap ops | mechanical; extend the shaders past rank 4 if a workload appears |
| `sort`/`argsort`, LinAlg | large, and the host path is correct — only if a workload demands it |

### 8.5 Unify the index-remap shader family

`transpose_nd`, `reverse_nd`, `broadcast_nd` share one skeleton: decompose the
output index, map to an input index, copy. They differ only in the mapping rule.
A **fourth** member is the point to unify them behind one shader with a mode
selector. Three does not yet justify the indirection.

### 8.6 Upstream

- **Ship `Nx.Helpers.check_grads!` in `Nx.Testing`.** It already exists and is
  already used 33 times upstream; it is behind the packaging wall for no
  articulated reason. Smallest useful PR available.
- **A backend conformance kit** (gradient parity + an op database) is a larger
  proposal and would need buy-in; the memo sketches it. Note it would not have
  caught our bug without a residency primitive.
- **Report the XLA gradient tiling failure** — two stacked stride-2
  `channels: :first` convs — with the minimal reproducer from §7. It is a real
  upstream bug even if narrow.

### 8.7 Fleet hygiene

Race reports are written per host (`bench_results/f32_race_<host>_<commit>.json`)
but are not collected automatically. A script that fans out, runs, and gathers
them would make "validate across the fleet" a single command rather than a
manual SSH sequence — and the skill already insists every perf heuristic be
fleet-validated.

---

## 9. One-paragraph version

A backend whose fast paths were all gated on forward-pass shapes ran its entire
backward pass on the CPU, undetected, because host fallbacks return
bit-identical results and nothing in the Nx ecosystem tests a backend's
gradients. Counting fallbacks — not timing them, not checking their values —
found eight such gates in an afternoon apiece and took a CNN training step from
20.9 s to 84 ms across three GPUs spanning a decade. The remaining gap to EXLA
is roughly 20–29×, and it is dispatch overhead rather than missing fusion —
which we know because turning fusion on made it *worse*.
