# Research note — Where the Emily-style named-kernel pattern wins

**Date**: May 2026
**Trigger**: A short refactor session moving eXMC's NUTS leapfrog from
direct `Nx.*` calls to `Nx.Vulkan.Fast` named kernels (Emily's
`Nx.Defn.Expr.optional` pattern). Microbenchmark settled the question
in three lines of output. This document records what we learned so we
do not relearn it.

## The measurement

One leapfrog body, RTX 3060 Ti, `d = 8`, 200 iterations, warmed:

| Path | µs / body | Ratio vs naive |
|------|-----------|----------------|
| Naive (direct `Nx.*` ops, Evaluator) | **485** | 1.00× |
| `Nx.Vulkan.Fast` (named kernels via `Expr.optional`) | 4311 | **0.11× (9× SLOWER)** |
| `Nx.Vulkan.Compiler` IR walker (one fused dispatch) | **388** | 1.25× |

The architecture we copied from Emily was nine times slower than the
unmodified path it was supposed to replace.

## Why

Each `Nx.Defn.Expr.optional` node adds an indirection layer at
evaluation time:

1. Nx evaluator decodes the optional node's args
2. Calls `function_exported?(backend, name, n)`
3. Erlang dynamic-call dispatch to the backend function
4. Backend function shape/type-validates operands
5. Backend transfers operands if needed
6. NIF call to dispatch the fused shader

We measured **~700 µs per Fast call** of overhead on top of the
GPU work. A NUTS leapfrog has ~6 elementwise ops; six × seven hundred
is forty-two hundred microseconds — exactly the figure the bench
reported.

A primitive `Nx.add(a, b)` takes ~150 µs through the backend protocol
(no optional indirection). Six primitives total ≈ 1000 µs. That
matches naive.

The IR walker dispatches the entire chain through ONE backend call
that emits ONE fused shader. It pays the dispatch overhead once. ~400
µs end to end.

## The break-even rule

The Emily pattern wins iff **GPU work per Fast call ≫ 700 µs of
indirection overhead**. Concretely:

| GPU work per Fast call | Verdict |
|------------------------|---------|
| ≥ 5 ms | Named kernels win unambiguously |
| 0.5 – 5 ms | Depends on call count and other paths |
| < 0.5 ms | Named kernels lose to primitive dispatch |

A Bumblebee LayerNorm on a 4096-element hidden state does ~3 ms of
GPU work — Emily wins. An MCMC leapfrog body on an 8-element vector
does ~10 µs — Emily loses by 9×.

## What this means for the four Probabileurs

### `eXMC` — MCMC sampler (NUTS, HMC) on small vectors

- Workload: leapfrog bodies, log-density evaluations, mass-matrix
  Welford updates. Tensor sizes: d ≤ 50 typical, d ≤ 1000 outlier.
- Per-kernel work: 1 µs to 100 µs.
- **Verdict**: Emily-style Fast kernels lose. The IR-walking
  `Nx.Vulkan.Compiler` (right-folded chain detection + 4-input fused
  dispatch + auto-detect for 3-4 arg defns) is the right path. We
  ship `Nx.Vulkan.Fast` as an explicit-opt-in API for power users
  who know their tensor sizes and want manual control, but it is
  not the primary surface.

### `StochTree-Ex` — Bayesian Additive Regression Trees

- Hot path: tree growth + mu-shrinkage + residual updates, all
  inside a Rust NIF. The Elixir layer is a thin wrapper; only ~5
  lines of `Nx.*` usage in the entire repo, all in shape coercion
  helpers.
- Per-kernel work: zero. The compute happens in the Rust NIF, not
  through `Nx.Defn`.
- **Verdict**: irrelevant. Emily's pattern presumes Nx-mediated
  tensor work. StochTree's heavy lifting is offloaded to native code
  via NIF directly. The architectural decision is already made
  upstream. No `Nx.Vulkan` work helps StochTree — it doesn't run
  through any Nx backend.
- Possible future work: if a future variant of StochTree wants
  posterior-predictive evaluation in Elixir-side defns (matrix-
  vector ops on the forest of fitted trees), THAT could route through
  `Nx.Vulkan` — but the kernels would be matmul-shaped and large
  enough that even Emily's pattern would fit. Not on the current
  roadmap.

### `SMC-Ex` — Sequential Monte Carlo (particle filters, PMCMC, SMC²)

- Hot path: parallel particle propagation via `Task.async_stream`,
  weight updates, resampling. Zero `Nx.*` calls in the library — the
  README is explicit: "Pure Elixir — zero dependencies." The
  state-transition and observation-density functions are user-
  supplied callbacks; whatever the user writes, the library
  orchestrates.
- Per-kernel work: depends entirely on the user's model. Plain
  arithmetic for compartmental epi models; could be Nx tensors for
  state-space models with vector states.
- **Verdict**: depends on the user's model. SMC-Ex itself is
  workload-agnostic. If a user supplies `transition` and
  `observation_logp` written with `Nx.Vulkan.Fast.normal_logpdf`,
  the kernel fires when the state-vector is large (≥ ~1000
  elements, e.g., a discretised PDE). For low-dim state-space models
  (epidemic SIR, finance volatility), the IR-walker is faster, same
  as eXMC.
- Useful follow-up: SMC-Ex `notebooks/` could include one example
  with a "fat" observation model (high-dim multivariate normal
  log-density) where Fast kernels demonstrably help — concrete proof
  that the pattern is workload-conditional, not an absolute design
  win.

### `sim_ex` — Discrete-event simulation engine

- Hot path: event scheduling, resource grants/releases, calendar
  advancement. No tensor work; ETS-backed accounting and Erlang
  scheduling primitives.
- **Verdict**: not applicable. Different domain.

## Where Emily's pattern actually helps in our portfolio

Listed in decreasing relevance:

1. **Bumblebee inference servings** (Qwen3, DistilBERT, ViT) — the
   canonical case Emily was designed for. Hidden states of thousands
   of elements; per-kernel work measured in milliseconds. Use it.
2. **Hypothetical: a future stochtree-ex posterior-predictive
   path that runs ENTIRELY in Nx** — large matmul over
   sample × feature × tree posteriors. Per-kernel work likely > 5 ms.
3. **Image generation / diffusion model serving** — same story as
   transformers. Big tensors per kernel.
4. **Anywhere we add a Nx-mediated workload with tensor sizes ≥ ~10K
   elements per op**.

## Where it loses (and the IR walker wins or naive primitives win)

1. **Any MCMC sampler at typical Bayesian-modeller dimensions** — d ≤
   ~100. Includes eXMC NUTS, HMC, Gibbs samplers.
2. **State-space models with low-dim states** — Kalman-style, finance
   volatility, epi compartmentals. Per-step work too small.
3. **Online algorithms with tiny per-update kernels** — sequential
   Bayesian updates, particle filtering on low-dim states.
4. **Anything where the Rust NIF already swallows the compute** —
   stochtree-ex's BART, sim_ex's calendar. The Elixir/Nx layer is
   not on the hot path; no fusion architecture helps.

## The methodological lesson

We adopted Emily's architecture on the basis of careful reading of
their source and a plausible-sounding analogy (vendor-specific fused
kernels are vendor-specific fused kernels). We did not run the
microbenchmark before refactoring eXMC's leapfrog. The benchmark, when
we eventually ran it, was three lines of code.

**Protocol going forward**: before adopting an architectural pattern
from another project, write the smallest microbenchmark that compares
the pattern's per-call cost to the existing path on representative
inputs. The break-even depends on workload, not philosophy.

The cost of measuring is always less than the cost of refactoring is
always less than the cost of a blog post you have to rewrite.

## Status of the named-kernel infrastructure in `Nx.Vulkan`

Kept (correctness was always there; only the speed claim was wrong):

- `lib/nx_vulkan/fast.ex` — six named kernels:
  `leapfrog_position`, `leapfrog_momentum_half`, `momentum_step`,
  `inv_mass_apply`, `kinetic_energy`, `normal_logpdf`. Each emits
  `Nx.Defn.Expr.optional/3` with a defn fallback that runs on any
  backend.
- Six matching backend callbacks in `Nx.Vulkan.Backend` — dispatch
  the relevant fused shader (`fused_elementwise_4in.spv`,
  `kinetic_energy.spv`, `normal_logpdf.spv`) when shapes/types fit;
  fall back via `Nx.BinaryBackend` round-trip otherwise.

Reverted:

- `Exmc.NUTS.Leapfrog.step/6` reverted from the Fast-kernel
  refactor back to direct `Nx.*` calls. The Fast version measured
  4× slower for diagonal mass at d ≤ 8.

Recommended:

- Use `Nx.Vulkan.Fast` only for explicit calls where the caller
  knows the tensor size is large enough to justify the indirection.
- For typical defn-traced eXMC code (gradient computations, mass
  matrix updates, sampler bodies), let `Nx.Vulkan.Compiler` walk
  the IR — it dispatches one fused shader per recognised chain
  with no per-op optional indirection.
- Microbenchmark new fused-kernel candidates before shipping.
  Three-line comparison of named-kernel vs naive primitives, pinned
  to representative dimensions.

## Open research questions

1. Is the 700 µs per-Fast-call overhead reducible? Some of it is
   `function_exported?` + dynamic dispatch + Rustler resource decode
   per call. Caching the resolved callback per IR node would help.
   Worth pursuing only if it pushes the break-even below 0.5 ms.
2. Is there a hybrid: IR walker that recognises chains AND, when the
   chain is long enough to amortise indirection, emits a Fast call
   instead of an inline primitive sequence? Probably overengineering.
3. For SMC-Ex specifically: would a `Nx.Vulkan.Fast.particle_step/3`
   that fuses propagate + weight + resample into one shader help?
   Only worth it if the per-particle work is fat (e.g., dense
   Kalman updates, not scalar epi compartments). Empirical question.

## Cross-references

- `~/projects/learn_erl/emily/lib/emily/fast.ex` — the reference
  implementation that prompted the experiment
- `~/projects/learn_erl/emily/lib/emily/compiler.ex` — Emily's 141-
  line compiler that "explicitly skipped IR-level fusion" because
  the win was below threshold for transformers
- `bench/leapfrog_bench.exs` — the microbenchmark that produced the
  numbers in this note
- `lib/nx_vulkan/fast.ex` — the kept named-kernel module
- `lib/nx_vulkan/compiler.ex` — the IR walker that turned out to be
  the right answer for our workload size class
- `~/projects/learn_erl/pymc/www.dataalienist.com/blog-emily-taught-us.html`
  — the public-facing version of this finding
