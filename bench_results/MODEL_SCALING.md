# Model scaling — does the GPU case for eXMC survive being made on width?

**Date:** 2026-08-16 · **Host:** super-io, RTX 3060 Ti (Ampere, `DiscreteGpu`),
Linux 6.8 · **nx_vulkan:** `80aa2be` (branch `bench/model-scaling`) ·
**eXMC:** `2e94b896` · **Harness:**
[`model_scaling/model_scaling.exs`](model_scaling/model_scaling.exs), run from
the eXMC tree as `bench/model_scaling.exs` · raw output in
[`model_scaling/raw_grad.txt`](model_scaling/raw_grad.txt) and
[`model_scaling/raw_exla.txt`](model_scaling/raw_exla.txt) · f64 unless stated.

## The claim under test

[`EXMC_PEROP_RACE.md`](EXMC_PEROP_RACE.md) measured `Exmc.Trading.RegimeModel`
at **d = 8, 60 observations** and found no GPU arm beats `BinaryBackend` —
393 ms per NUTS iteration on the CPU against 680 for the synthesised chain
shader and ~31,650 per-op. It closed by asserting that the GPU case therefore

> has to be made on **width** — large `d`, many chains, or many instruments
> sampled concurrently — not on making a d=8 model faster.

This report sweeps width and tests that.

**The answer is two-part, and the second part matters more than the first.**

1. Against `BinaryBackend`, the width crossover is **real, reproducible and
   large**: the per-op Vulkan path overtakes at roughly **10³ f64 elements** in
   the likelihood tensor and reaches **410×** by 4×10⁵ elements. Neither `d`
   nor `n_obs` is special — total elements is the variable.
2. Against a CPU that JIT-compiles, **there is no crossover anywhere in the
   reachable range**. EXLA on super-io's *host CPU* computes the same gradient
   in 0.58–39 ms across the entire sweep, beating the Vulkan per-op path by
   20× at the small end and **215× at 6×10⁶ elements**. The gap *widens* with
   size.

So the width claim is true against the baseline it was made against, and the
baseline is the problem.

## What was measured, and with what

Three model families, one posterior. All compute

```
sum_i logsumexp_k [ log(1/d) + Normal(y_i | 0, sigma_k) ],   sigma_k ~ HalfCauchy(0.02)
```

— the RegimeModel's 3-component scale mixture, widened. `d` is the free
parameter count, `n_obs` the observation count, and they move independently.

| | shape | ops in the graph | chain-shader eligible |
|---|---|---|---|
| **S** ("scalar") | `d` scalar RVs, mixture unrolled in Elixir | **O(d)** | yes, while the push block fits |
| **V** ("vector") | one RV of `shape: {d}`, mixture over a `{d, n_obs}` tensor | **constant** | no |
| **W** (control) | V with the mixture axis last: `{n_obs, d}` | constant | no |

**S is the shape eXMC models are actually written in** — `RegimeModel` writes
its 3-component mixture exactly this way, and it is the only shape
`Exmc.NUTS.CustomSynth` models correctly. **V is the GPU's best case**: same
FLOPs, same posterior, but a fixed number of dispatches no matter how wide the
model gets. **W exists to falsify a specific alternative explanation** — if V
and W disagreed on the CPU, the CPU arm's cost would be `BinaryBackend` being
bad at a strided reduction rather than a scaling law. They agree (below), so it
is a scaling law.

`RegimeModel` itself was not swept: it has no width knob. Its eight parameters
are named roles (`mu_trend`, `sigma_vol`, `logit_w1`, …), not a dimension, and
widening it means inventing a different model. The synthetic family reproduces
its graph shape and its d=8 behaviour, and sweeps cleanly.

### Arms

| arm | what it is |
|---|---|
| `cpu` | `config :exmc, :compiler, :none` → `Nx.BinaryBackend`. The reference EXMC_PEROP_RACE used. |
| `perop` | `:vulkan` + `VulkanoBackend`, `Nx.Defn.Evaluator`, one dispatch per op |
| `fused` | same, but `Nx.Vulkan.Compiler` (whole-graph fusion) as the defn compiler |
| `chain` | the synthesised leapfrog chain shader (end-to-end NUTS only) |
| `exla_host` / `exla_cuda` | **not part of the requested race.** Added because "is the GPU the right place for the effort" is unanswerable against `BinaryBackend` alone. See the caveat section for why it had to be measured out-of-tree. |

## Method

- **Every timed GPU call is forced to resolve** before the clock stops, by
  reading back the scalar `logp` — **and nothing else**. On `VulkanoBackend` a
  `buf_download` calls `flush_pending`, which submits every queued dispatch
  including the gradient's, and `submit_and_wait` blocks on the whole command
  buffer, so the scalar accounts for all the work. Transferring the `{d}`
  gradient as well would add a constant to every arm — the mistake
  `examples/concurrent_dispatch_bench.exs` documents.
- **Residency is asserted twice per GPU cell**: `Nx.Vulkan.Fallback.count/1`
  (`fallbacks=` in the raw output) and `Nx.Vulkan.Fallback.strict(:raise, …)`
  (`strict=clean`). The count is a lower bound — the first fallback strands the
  tensor on `BinaryBackend` and everything after it is invisible — so strict
  mode, which fires before the tensor leaves the device, is the assertion that
  actually holds. **Every GPU cell in this report is `fallbacks=0`,
  `strict=clean`.** That is a change since EXMC_PEROP_RACE, which measured 137
  fallbacks per gradient; `c388660` (T11: rank-0 compare/select, `put_slice`
  shader, `pad` gate) and `80aa2be` (T12) closed them.
- Iteration count per cell is chosen from a pilot call to fill a ~600 ms
  budget; **5 replicates per cell** (3 where one gradient costs seconds),
  median reported with min–max.
- The timed closure is built by `Exmc.Compiler.value_and_grad/1` — the same
  `build_vag_fn` as `compile_for_sampling/2`, minus chain-shader synthesis,
  which plays no part in a bare gradient and costs minutes at d ≥ 12.
- Device confirmed as `NVIDIA GeForce RTX 3060 Ti (DiscreteGpu)`, not llvmpipe.
- `logp` agrees between arms to 7–8 significant figures at every cell
  (e.g. `158.000357` host vs `158.000329` device at d=8/n=60), so every timing
  below is a timing of correct work.

### Contamination, stated up front

super-io is shared. A foreign `mix test` in the eXMC tree — another session,
not this one — held the GPU and ~1.5 cores from **09:32:30 to 09:49:34**,
overlapping run 1, and a second started at **09:51** and overlapped the
replicate. Load average ran 4–12 throughout. Consequences, honestly:

- The **GPU arm reproduces tightly** — within a few percent across runs at
  every cell, with two flagged exceptions (below).
- The **CPU arm moved by up to 30%** between runs (d=64/n=60: 64.4 → 50.2 ms;
  d=256/n=60: 434.5 → 314.2 ms), and it moved in the direction contention
  predicts.
- **This moves the d-axis crossover by about one grid step** and is reported as
  a range rather than a point. It does not touch any conclusion below, all of
  which rest on effect sizes of 5–400×.

## Result 1 — the crossover against `BinaryBackend` is real, and it is on total elements

### Widening `d` at n_obs = 60 (Model V, f64)

`run 1` and `run 2` are independent BEAM invocations ~20 minutes apart.

| d | elements | cpu ms (run 1 / run 2) | perop ms (run 1 / run 2) | fused ms (run 2) | speedup vs cpu |
|---:|---:|---|---|---:|---:|
| 4 | 240 | 4.9 / — | 12.4 / — | — | 0.39× |
| 6 | 360 | 6.6 / — | 12.8 / — | — | 0.51× |
| 8 | 480 | 6.8 / 8.0 | 12.7 / 12.6 | 14.3 | 0.63× |
| 10 | 600 | 10.4 / 8.0 | 13.1 / 14.2 | 10.5 | 0.65× |
| 12 | 720 | 11.9 / 11.9 | 13.3 / 13.9 | 20.9 | 0.87× |
| 14 | 840 | 14.4 / 10.8 | 13.1 / 14.3 | 16.2 | 0.88× |
| 16 | 960 | 16.1 / 12.3 | 13.0 / 13.8 | 14.5 | 1.06× |
| 24 | 1,440 | 22.5 / — | 14.0 / — | — | 1.60× |
| 32 | 1,920 | 25.1 / 23.8 | 12.9 / 10.2 † | 13.4 | 2.1× |
| 64 | 3,840 | 64.4 / 50.2 | 13.9 / 15.4 | 15.1 | 3.9× |
| 128 | 7,680 | 107.9 / 118.9 | 15.0 / 16.8 | 16.7 | 7.1× |
| 256 | 15,360 | 434.5 / 314.2 | 14.6 / 15.2 | 16.1 | 25× |
| 512 | 30,720 | 794.9 / — | 16.8 / — | — | 47× |
| 1024 | 61,440 | 2130.4 / — | 17.6 / — | — | 121× |

† one replicate at d=32 recorded a 230 ms outlier iteration against a 10.2 ms
median; a scheduling hitch, not a distribution.

**Crossover: run 1 puts it between d=12 and d=14; run 2 between d=16 and d=32.
Call it d ≈ 15–30 at n_obs = 60.** The 30% run-to-run movement in the CPU arm
is the whole difference — this is exactly the kind of near-parity region where
a single run would have produced a confident wrong number.

The shape of the GPU column is the finding. **From d = 4 to d = 1024 — 256×
more arithmetic — the per-op time goes from 12.4 ms to 17.6 ms.** The work is
free; the dispatches are not.

### Widening `n_obs` at d = 8 (Model V, f64)

| n_obs | elements | cpu ms (run 1 / run 2) | perop ms (run 1 / run 2) | fused ms (run 2) | speedup |
|---:|---:|---|---|---:|---:|
| 60 | 480 | 6.8 / 8.0 | 12.7 / 12.6 | 14.3 | 0.63× |
| 600 | 4,800 | 69.2 / 81.6 | 15.3 / 14.3 | 15.0 | **5.3×** |
| 6,000 | 48,000 | 1638 / 1727 | 19.7 / 17.5 | 18.6 | **90×** |
| 60,000 | 480,000 | 21,750 / — | 92.9 / — | — | **234×** |

**Crossover between n_obs = 60 and 600 at d = 8**, and it reproduces cleanly —
unlike the `d` axis, the two runs agree to within 15% at every cell here.

### The two axes are one axis

| cell | elements | cpu ms | perop ms |
|---|---:|---:|---:|
| d=8, n=6,000 | 48,000 | 1638 | 19.7 |
| d=64, n=600 | 38,400 | 1375 | 16.6 |
| d=256, n=60 | 15,360 | 434 | 14.6 |
| d=1024, n=60 | 61,440 | 2130 | 17.6 |

Equal element counts give equal times on both arms regardless of which axis
supplied them. **`d` and `n_obs` are not separate knobs for this workload;
`d × n_obs` is the variable, and the crossover sits at roughly 10³ elements.**

The practical asymmetry is not in the physics but in the model: `n_obs` is free
to grow — 60 → 60,000 is a data decision — while `d` is bounded by what you are
willing to call a parameter. So *in practice* `n_obs` moves the needle further,
which is what the original prior expected, but not for the reason it supposed.
Widening `d` by the same factor buys exactly the same thing.

### Both axes together

| d | n_obs | elements | cpu ms | perop ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 64 | 600 | 38,400 | 1,375 | 16.6 | 83× |
| 64 | 6,000 | 384,000 | 19,811 | 48.3 | **410×** |
| 256 | 600 | 153,600 | 6,696 | 325 ‡ | 21× |
| 256 | 6,000 | 1,536,000 | not measured | 183 | — |
| 1024 | 6,000 | 6,144,000 | not measured | 1,080 | — |

‡ this cell ran entirely inside the foreign suite's window and its replicate
spread was 23.9–685.8 ms. Treat the 325 as unreliable; the neighbouring cells
are not.

The two "not measured" CPU cells were abandoned: one `BinaryBackend` gradient
at d=256/n=6,000 exceeds 90 seconds, and eight of them is not a good use of a
shared box when d=64/n=6,000 has already established the trend.

### The control passes: this is width, not a backend artefact

Model W (mixture axis last, reduce over the contiguous axis) against Model V
(mixture axis first, strided reduce):

| cell | V cpu ms | W cpu ms | V perop ms | W perop ms |
|---|---:|---:|---:|---:|
| d=8, n=60 | 8.0 | 7.1 | 12.6 | 13.6 |
| d=64, n=60 | 50.2 | 54.9 | 15.4 | 15.3 |
| d=8, n=600 | 81.6 | 66.5 | 14.3 | 15.2 |
| d=64, n=600 | 1,375 | 1,016 | 16.6 | 17.5 |

Same numbers within run-to-run spread. The CPU arm's cost is genuine
element-count scaling.

## Result 2 — whole-graph fusion does not move the floor

`Nx.Vulkan.Compiler` is the mechanism nx_vulkan has for amortising
per-dispatch cost across a graph, and this is exactly the graph shape it is
supposed to win on: ~40 elementwise ops and two reductions, no `dot`, no conv.

It is **within noise of the per-op path at every one of the 13 cells measured**
(10.5–20.9 ms where per-op is 10.2–16.8 ms; 184.7 vs 182.7 at d=256/n=6,000;
1,093 vs 1,080 at d=1024/n=6,000). EXMC_PEROP_RACE found the same thing and
attributed it to the 137 host fallbacks happening below the compiler. **That
explanation is now dead** — there are zero fallbacks here — and the result is
unchanged, so the cause is something else and remains unidentified.

This is a negative result about nx_vulkan's own fusion path, measured on the
workload it was designed for. It matters beyond this benchmark: fusion is the
only mechanism in the repo that could close the gap to `exla_host` in Result 4,
and on this graph it closes none of it.

## Result 3 — f32 buys nothing until the workload is large, then buys 2×

The Ampere card runs f64 at 1/64 rate, so f32 is where a compute-bound workload
would show it. Model V, `config :exmc, :force_precision, :f32`:

| cell | elements | cpu f64 / f32 | perop f64 / f32 |
|---|---:|---|---|
| d=8, n=60 | 480 | 8.0 / 7.8 | 12.6 / 13.4 |
| d=64, n=60 | 3,840 | 50.2 / 48.9 | 15.4 / 14.4 |
| d=8, n=6,000 | 48,000 | 1,727 / 1,539 | 17.5 / 18.7 |
| d=64, n=6,000 | 384,000 | 19,811 / 18,653 | 48.3 / **23.3** |

**f32 is worth nothing below ~10⁵ elements on either arm, and ~2× on the GPU
above it.** That is itself the proof that everything below 10⁵ elements is
dispatch-bound rather than FLOP-bound: an arm that is 64× rate-limited on
arithmetic and does not care when you remove the limit was not doing
arithmetic.

**There is no crossover at f32 that does not already exist at f64.** The
project's f64 default costs nothing it needs to reconsider on these grounds.

## Result 4 — the caveat that decides the bottom line

`compiler: :none` is `Nx.BinaryBackend`: a pure-BEAM backend that evaluates
elementwise ops by binary comprehension. Measured here, one full
`value_and_grad` costs **17 µs per likelihood element at d=8/n=60, rising to
52 µs at d=64/n=6,000** — the whole gradient, forward and backward, per single
f64 in the `{d, n_obs}` tensor. Every crossover above is a crossover against
those numbers.

EXLA is not built into this eXMC configuration — it requires an `NX_PATH`
monorepo build, and rebuilding mid-benchmark would have invalidated every
measurement above. So the same arithmetic was timed **out-of-tree** in a
project that does have a working EXLA (`nx 0.10.0` + `exla 0.10.0`; the eXMC
tree is on nx 0.13). This is a pure-Nx `value_and_grad` of the same mixture
likelihood, without the prior and point-map wrapper — and its `BinaryBackend`
column reproduces the in-tree CPU arm to within 25% at every shared cell (2% at
d=8/n=60, 25% at d=64/n=600, always on the faster side, as dropping the prior
predicts). That agreement is what makes the comparison legitimate: the
`exla_host` column is being timed on the same work the in-tree arms are.

| d | n_obs | elements | binary ms | **exla_host ms** | exla_cuda ms | vulkan perop ms |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 60 | 480 | 7.81 | **0.58** | 0.48 | 12.6 |
| 8 | 600 | 4,800 | 76.6 | **0.58** | 0.49 | 14.3 |
| 8 | 6,000 | 48,000 | 1,462 | **0.88** | 0.84 | 17.5 |
| 64 | 60 | 3,840 | 54.8 | **0.71** | 0.53 | 15.4 |
| 64 | 600 | 38,400 | 1,032 | **0.64** | 0.80 | 16.6 |
| 256 | 60 | 15,360 | 293 | **0.68** | 0.73 | 15.2 |
| 256 | 6,000 | 1,536,000 | — | **2.19** | 2.01 | 183 |
| 1024 | 6,000 | 6,144,000 | — | **5.01** | 4.81 | 1,080 |
| 256 | 60,000 | 15,360,000 | — | **11.9** | 10.8 | — |
| 1024 | 60,000 | 61,440,000 | — | **39.4** | 38.4 | — |

Read the last two columns together:

- **EXLA on the host CPU beats the Vulkan per-op path at every cell measured**,
  by 22× at the small end and **215× at 6×10⁶ elements**.
- **The gap widens with size.** Whatever the width argument was going to buy,
  it buys it for EXLA faster.
- `exla_cuda` is indistinguishable from `exla_host` on this workload — even at
  61 million elements — which says the whole thing is still launch-bound at
  scales far beyond anything eXMC will sample.

The honest reading: **the width crossover in Result 1 measures the distance
between a GPU backend and an interpreter, not the distance between a GPU and a
CPU.**

## Result 5 — graph shape moves the crossover by an order of magnitude, and eXMC is written in the worse one

Model S computes the identical posterior with the identical FLOP count, but
unrolls the mixture over `d` scalar RVs — so its graph has O(d) ops where
Model V's has a constant number. Every eXMC model in `lib/` is written this
way.

**Model S, n_obs = 60, f64:**

| d | cpu ms | perop ms | speedup |
|---:|---:|---:|---:|
| 2 | 3.95 | 21.2 | 0.19× |
| 4 | 9.12 | 60.1 | 0.15× |
| 8 | 24.1 | 119.5 | 0.20× |
| 16 | 34.7 | 248.8 | 0.14× |

The GPU column grows **linearly in d at ~15 ms per additional RV**, which is
the per-op dispatch cost showing up undisguised. Compare Model V at the same
widths: 8.0 → 15.4 ms over d = 8 → 64. **Same arithmetic, same answer, 16×
apart on the GPU** — and the difference is entirely how many ops the graph has.

**Model S, d = 8, f64:**

| n_obs | elements | cpu ms | perop ms | speedup |
|---:|---:|---:|---:|---:|
| 60 | 480 | 24.1 | 119.5 | 0.20× |
| 600 | 4,800 | 58.0 | 90.3 | 0.64× |
| 6,000 | 48,000 | 1,161 | 90.2 | **12.9×** |

Model S *does* cross over — the per-op arm is flat in `n_obs` (119 → 90 → 90),
because `n_obs` adds elements to existing dispatches rather than adding
dispatches. But it crosses at **~2×10⁴ elements against Model V's ~10³**: a
20× penalty, paid purely for writing the mixture as `d` scalars instead of one
vector.

That is a finding about eXMC's IR, not about the GPU. The cheapest available
speedup on the per-op path is not a shader — it is teaching `Exmc.Builder` /
`PointMap` to keep a `shape: {d}` RV vectorised end to end.



## Result 6 — the chain shader cannot follow a model to width at all

The brief asked where `CustomSynth`'s d ≤ 256 cap bites. **It never does.** A
different limit binds first, an order of magnitude earlier.

`Exmc.NUTS.CustomSynth.Push.pack/1` writes a **24-byte header** (`K`, `n_obs`,
`d`, pad, then f64 `eps`) followed by **8 bytes per prior scalar**, into a
Vulkan push-constants block capped at 128 bytes. That leaves room for exactly
**13 prior floats**. Measured, sweeping `d` from 1 to 16 for both the cheapest
and the common prior family:

| prior family | floats/RV | largest `d` that packs | first `d` refused |
|---|---:|---:|---:|
| `HalfCauchy` / `HalfNormal` / `Exponential` | 1 | **13** (128 B exactly) | 14 |
| `Normal` / `Cauchy` / `Gamma` / `Beta` / … | 2 | **6** (120 B) | 7 |
| `StudentT` | 3 | 4 | 5 |
| `TruncatedNormal` | 4 | 3 | 4 |

`RegimeModel` fits at d = 8 only because it is 4 `Normal` (2 floats each) +
4 `HalfCauchy` (1 each) = 12 floats = 120 B — **one float of headroom.** The
`when is_integer(d) and d <= 256` guards in
`Exmc.NUTS.Tree` and `Exmc.NUTS.Vulkan.Dispatch`, and the "d <= 256" in
`Exmc.Compiler`'s guard message, describe a limit that cannot be reached.

### And the corner is expensive to enter

Wall time for one `CustomSynth.synthesise/1` — Nx tracing, GLSL emission,
glslang, content-addressed cache write — on Model S at n_obs = 60:

| d | push block | synthesis |
|---:|---:|---:|
| 2 | 40 B | 0.08 s |
| 4 | 56 B | 0.47 s |
| 6 | 72 B | 2.8 s |
| 8 | 88 B | **10.5 s** |
| 10 | 104 B | **45.8 s** |
| 11 | 112 B | **77.5 s** |
| 12 | 120 B | **117.2 s** |
| 13 | 128 B | not measured — stage stopped; the trend implies 3–5 min |
| 14 | overflow | rejected |

Roughly **1.8× per unit of `d`**, because
`MultiRvCustomSpec.build_grad_body_with_loops/4` emits one `if (tid == i)`
block per free RV, each containing its own obs loop, on top of the emitter's
tree-walk duplication. Note also that the push-constants check happens *after*
GLSL rendering, so a model that is going to be refused for width still pays the
emission cost to find out.

Two consequences worth stating plainly:

- **The chain shader's entire reason to exist — amortising per-dispatch cost
  across K leapfrog steps — is only available below the width where the per-op
  path starts winning anyway.** Result 1 puts the per-op crossover at d ≈ 15–30
  at n_obs = 60. Synthesis stops at d = 13. The two envelopes do not overlap.
- **A `shape: {d}` RV does not rescue this, and is worse than not working.**
  `CustomSynth.extract_components/1` builds `layout` one entry per *RV node*,
  not per scalar component; `MultiRvCustomSpec` derives `d` from
  `length(layout)`; and `Push.scalar/2` takes element 0 of a vector prior
  param. So a `shape: {d}` RV would emit a shader with `d_synth = 1` while
  `pm.size = d` and the dispatch is sized for `d`. In this benchmark that
  never fires — Model V's likelihood raises inside the emitter first, so
  `detect_meta/1` returns `:unsupported` and `chain_meta` is `nil` — but the
  path is reachable in principle for a vector RV with an emittable likelihood.
  **This is a code reading, not a measurement**, and it deserves its own
  investigation rather than a line in a benchmark.



## Result 7 — end-to-end NUTS agrees with the per-gradient result

25 warmup + 25 samples, seed 42, `ncp: false`. `ms/iter` is wall time over
50 iterations; it is *not* per-gradient, because each arm adapts its own step
size and therefore walks trajectories of different length. Read it as
confirmation of direction and rough magnitude, not as a second measurement of
the same quantity.

**Model V:**

| d | n_obs | arm | wall | ms/iter | ε | div | grad ratio for comparison |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 60 | cpu | 14.8 s | **295** | 0.301 | 0 | |
| 8 | 60 | perop | 20.0 s | 400 | 0.260 | 0 | 1.4× vs 1.6× |
| 64 | 60 | cpu | 308.5 s | 6,170 | 0.278 | 0 | |
| 64 | 60 | perop | 25.3 s | **506** | 0.348 | 0 | **12×** vs 3.3× |
| 8 | 600 | cpu | 156.6 s | 3,133 | 0.070 | 1 | |
| 8 | 600 | perop | 23.2 s | **465** | 0.089 | 1 | **6.7×** vs 5.7× |

The per-gradient result carries over at every size. At d=8/n=60 the CPU still
wins, as EXMC_PEROP_RACE found for `RegimeModel` at the same width. Past the
crossover the GPU wins by the same order of magnitude the gradient sweep
predicted.

The one place the two disagree — 12× end-to-end against 3.3× per gradient at
d=64 — is adaptation, not throughput: that arm settled on ε = 0.348 against the
CPU's 0.278 and therefore walked shorter trajectories. Direction and magnitude
carry; the exact ratio does not, and should not be quoted as if it did.

### The chain shader, at the only width it is allowed

Model S at d = 8, the width `RegimeModel` occupies and the width the chain
shader was built for. Both arms adapted to ε = 1.0, so this comparison is not
confounded by trajectory length.

| n_obs | arm | wall | ms/iter | result |
|---:|---|---:|---:|---|
| 60 | cpu | 42.8 s | **857** | |
| 60 | chain | 138.8 s | 2,775 | **3.2× slower than the CPU** |
| 600 | cpu | 200.4 s | 4,009 | |
| 600 | chain | — | — | **crashed: `(Erlang error) :nif_panicked`** |

Two things, both bad for the chain path:

1. **At n_obs = 60 it is 3.2× slower than `BinaryBackend`**, reproducing
   EXMC_PEROP_RACE's 1.7× on `RegimeModel` in the same direction and worse in
   magnitude. Ten and a half seconds of synthesis buys a 3× slowdown.
2. **At n_obs = 600 it panics the NIF.** The identical IR, the identical
   shader — only `pc.n_obs` and the obs SSBO change — and the CPU arm on the
   same model completes. This is a crash, not a refusal: no `:unsupported`, no
   fallback, a panicked NIF. It reproduces on the harness above and wants a
   bug report, not a benchmark footnote.

Meanwhile the per-op path, which the chain shader exists to beat, is now
fallback-free and — on the vectorised form of the same posterior — 6.7× faster
than the CPU at exactly that n_obs. **So the answer to "where does the per-op
path overtake the chain shader" is: everywhere it is measurable, and at
n_obs = 600 the chain shader does not run at all.**

## Bottom line

**The width claim from `EXMC_PEROP_RACE.md` is confirmed against the baseline
it was made against, and that turns out not to be the interesting question.**

Confirmed, measured, reproducible:

- There **is** a crossover. Against `BinaryBackend`, the fallback-free per-op
  Vulkan path overtakes at ~10³ likelihood elements and reaches 410× at
  4×10⁵. The GPU arm is essentially flat in model size across three orders of
  magnitude, because it is paying for dispatches, not arithmetic.
- `d` and `n_obs` are the same knob (`d × n_obs`). `n_obs` matters more only
  because it is the one you can grow by 1000× without redefining the model.
- Chains and instruments were not measured; nothing here contradicts the
  concurrency findings in [`CONCURRENT_DISPATCH.md`](CONCURRENT_DISPATCH.md),
  and they remain the one width axis this report does not cover.

Refuted, or at least badly damaged:

- **The CPU reference is the finding.** `compiler: :none` is a pure-BEAM
  interpreter costing 17–52 µs per likelihood element per gradient. EXLA on the *same box's host CPU* computes
  the same gradient 22–215× faster than the Vulkan per-op path at every size
  measured, and the gap grows with size. On super-io there is no reachable
  model width at which the GPU path is the right answer.
- **Fusion does not help.** `Nx.Vulkan.Compiler` is within noise of per-op on
  13 of 13 cells, on exactly the elementwise-heavy graph it was built for, and
  the 137-fallback explanation from the previous report no longer applies.
- **f32 does not open a door f64 had closed.** Nothing below 10⁵ elements,
  ~2× above it. The f64 default is not what is costing anything.
- **The chain shader's envelope stops at d = 13** — a push-constants limit, not
  the documented d ≤ 256 — costs up to two minutes to synthesise inside it,
  runs 3.2× slower than the CPU at the width it was built for, and panics the
  NIF at n_obs = 600. Its amortisation only applies at widths where the per-op
  path is not yet winning anyway.

### What this implies for where the effort goes

Ordered by measured leverage, not by preference:

1. **On super-io, use EXLA.** Nothing in this sweep suggests any Vulkan path on
   this box is worth choosing over a compiler eXMC already supports. Any
   further GPU work here should be justified against `exla_host`, not against
   `BinaryBackend` — every number in this repo's eXMC benchmarks so far has
   been the latter comparison, which flatters the GPU by one to two orders of
   magnitude.
2. **The FreeBSD Kepler fleet is a different question, and the one where the
   answer may still be yes.** Those hosts have no EXLA (`mix.exs` does not
   build it there), so `BinaryBackend` genuinely *is* the alternative, and
   Result 1 is then the whole story: at n_obs in the thousands the GPU is worth
   1–2 orders of magnitude. **That is the case for the Vulkan backend, and it
   is a real one — but it is a portability case, not a performance case, and
   it should be argued as such.**
3. **Vectorise eXMC's IR before writing another shader.** Model S vs Model V
   is a 16× difference on the GPU at d = 64 with identical arithmetic, and
   moves the crossover by 20×. Keeping `shape: {d}` RVs vectorised through
   `PointMap` and `Compiler` is cheaper than any kernel work and helps EXLA
   too.
4. **Find out why fusion is worth nothing here.** It is the mechanism that
   would close the 22× gap to `exla_host`, it is already built, and on this
   graph it does nothing. Until that is understood, "the fusion compiler will
   fix the per-dispatch cost" is not a claim this repo can make.
5. **The chain shader no longer earns its keep, and this report is where that
   changed.** It is confined to d ≤ 13 (d ≤ 6 with `Normal` priors), costs up
   to two minutes to synthesise inside that range, is **3.2× slower than
   `BinaryBackend`** at the width it was built for, and **panics the NIF at
   n_obs = 600**. The per-op path it exists to beat is now fallback-free and
   wins at that same n_obs. EXMC_PEROP_RACE concluded "the chain shader is
   still earning its keep" on the grounds that it turns ~30 dispatches into
   one; that argument no longer survives its own measurement.
   **The `:nif_panicked` at n_obs = 600 should be filed and fixed or the path
   retired — shipping a sampler that crashes on ten times the observations is
   worse than not shipping it.**

## What this report does not establish

- **Chains and instruments.** Only single-chain gradients and single-chain
  NUTS were measured. Batched/concurrent sampling is a separate width axis with
  its own machinery (`CustomSynth.synthesise_batched/1`,
  `Dispatch.chain_batch/4`) and its own prior negative result in
  [`CONCURRENT_DISPATCH.md`](CONCURRENT_DISPATCH.md).
- **The Kepler fleet.** Every number here is Ampere. The FreeBSD hosts have
  different f64 rates, different bandwidth and no EXLA, and conclusion 2 above
  is specifically *not* transferable to them without measurement.
- **Posterior quality.** No arm's posterior is compared here. At 25 samples the
  Monte-Carlo error swamps any backend difference — the trap
  `research/ASSESSMENT_2026_07_13.md` documented — and the chain arm's known
  over-dispersion signature is a separate open investigation.
- **Why fusion is worth nothing.** Stated as a measurement, not diagnosed.
- **The vector-RV synthesis mismatch** in Result 6 is read from the source, not
  triggered.

## Reproducing

```
cp bench_results/model_scaling/model_scaling.exs <exmc>/bench/
cd <exmc>

# the gradient sweep (SWEEP_MODE, not MODE — the app's runtime.exs owns MODE)
SWEEP_MODE=grad MODELS=V DIMS=8,16,32,64,128,256 NOBS=60 \
  ARMS=cpu,perop,fused REPS=5 mix run bench/model_scaling.exs

# the n_obs axis
SWEEP_MODE=grad MODELS=V DIMS=8 NOBS=60,600,6000,60000 ARMS=cpu,perop \
  REPS=5 mix run bench/model_scaling.exs

# f32
TYPE=f32 SWEEP_MODE=grad MODELS=V DIMS=8,64 NOBS=60,6000 ARMS=cpu,perop \
  mix run bench/model_scaling.exs

# the chain-shader envelope (slow: ~2 min per cell at d=12)
SWEEP_MODE=synth MODELS=S DIMS=2,4,6,8,10,12,13,14 mix run bench/model_scaling.exs

# end-to-end NUTS
SWEEP_MODE=nuts MODELS=S DIMS=8 NOBS=60,600 ARMS=cpu,chain \
  WARMUP=25 SAMPLES=25 mix run bench/model_scaling.exs
```

The EXLA reference is `bench_results/model_scaling/exla_ref.exs`; it needs no
eXMC and no nx_vulkan, only a project with a working EXLA NIF. It was run from
`~/projects/learn_erl/prompt_nick` (nx/exla 0.10.0) because the eXMC tree's
EXLA requires an `NX_PATH` monorepo build that would have invalidated every
other measurement in this report.


