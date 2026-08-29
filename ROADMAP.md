# nx_vulkan roadmap

What the backward-pass audit established and what it says to build next:
[`docs/BACKWARD_PASS_AUDIT.md`](https://github.com/borodark/nx_vulkan/blob/main/docs/BACKWARD_PASS_AUDIT.md).
Its conclusions as a working TODO — each item with the measurement that
motivates it and a "done when":
[`PLAN_AFTER_BACKWARD_PASS.md`](https://github.com/borodark/nx_vulkan/blob/main/PLAN_AFTER_BACKWARD_PASS.md).

Moved from README on 2026-07-13 so the main page stays focused on
what already works. Milestones and forward-looking items live here.

Plan history: [`PLAN_GPU_NODE.md`](https://github.com/borodark/nx_vulkan/blob/main/PLAN_GPU_NODE.md) (Phase 1–2 era)
and [`docs/VULKANO_BACKEND_ROADMAP.md`](docs/VULKANO_BACKEND_ROADMAP.md)
(Phase 3+). Per-workstream notes in
[`research/gpu_node/`](https://github.com/borodark/nx_vulkan/tree/main/research/gpu_node).

> **The goal of this project is reach, not speed.** EXLA on a host CPU beats
> this backend by 20-215x on the same gradient at every size measured
> ([`bench_results/MODEL_SCALING.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/MODEL_SCALING.md)), so
> performance parity is not the bar and is not pursued. The value is running
> Nx correctly on hardware nothing else serves — NVIDIA on FreeBSD, decade-old
> Keplers, AMD/Intel, anything with a Vulkan driver. Sections below that
> discuss closing a performance gap predate this and are kept as history;
> where they conflict with this paragraph, this paragraph is newer. The
> project's internal `MISSION.md` carries the longer argument; it is not
> published, so this paragraph is the public statement of it.

## Status snapshot

**Fusion compiler shipped** (August 2026): on top of the eager backend
(roadmap stages 1–8) the `Nx.Defn` fusion compiler landed — whole-graph
fusion with a multi-stage split at dot/conv/reduce/transpose boundaries,
f32 and f64. Main branch is stable across Linux + Ampere (RTX 3060 Ti)
and FreeBSD + Kepler (GT 650M, GT 750M): **833 doctests, 871 tests, 0 failures** on the fleet. The vulkano-only architecture (C++ spirit
backend dropped) merged 2026-07-13.

| Feature | Status |
|---|---|
| Vulkano buffer lifecycle | ✓ |
| Native compute op set via specialised SPVs | ✓ |
| Native **f32 and f64** shader paths (elementwise/matmul/conv/reduce/transpose) | ✓ |
| Pipeline cache (correctness + perf) | ✓ |
| Cross-host validation (Linux + 2× FreeBSD) | ✓ |
| Axon training step end-to-end | ✓ |
| eXMC regime log_p (f64) byte-identical | ✓ |
| Autograd via `Nx.Defn.grad` | ✓ |
| Scholar linear regression (coefs match to 2e-6) | ✓ |
| Cross-Kepler bit-determinism (GT 650M ≡ GT 750M) | ✓ |
| Ampere `primary_buffer_count=128` cmd-buffer fix | ✓ |
| Batched command submission (one submit + fence per batch) | ✓ |
| Persistent buffer pool | open — see T4 |
| f64 matmul (`matmul_f64.spv`) | ✓ |
| Scholar native linalg shaders (SVD/QR/cholesky/solve) | open — unscheduled |
| Polynomial f64 log/exp (behind config) — exmc side | ✓ (default: f32-cast) |
| Custom `Nx.Defn` compiler (whole-graph fusion) | ✓ |
| Native f32 compute (elementwise/matmul/conv/reduce/transpose) | ✓ |
| Conv (im2col + GEMM) / FFT | ✓ |
| sort / scatter | demand-driven, not scheduled |

## Open items

**Op coverage — the long tail.** Convolutions and FFTs now have native
GPU shaders (conv = im2col + GEMM, in f32 and f64; conv is also a fusion
boundary). Still on host-fallback: sort, scatter,
`Nx.LinAlg.solve`/`qr`/`svd`, complex types, sparse ops — they work
today but are slow.

On effort, distinguishing two things the old "6–12 months to feature
parity with EXLA" estimate ran together:

- **Coverage parity** — every op having *a* native shader. The
  "50–100 LOC of vulkano apiece" figure holds for the mechanical ones
  (elementwise-shaped, index-remap), and the backward-pass work is
  evidence for it: six of the eight ops recovered there needed no new
  kernel at all, only a wider gate. Plausible at months of work.
- **Performance parity** — being *competitive* per op. Not the same
  problem and not the same timescale: EXLA was **measured ~20× ahead** on a
  dense MLP *before* batched dispatch landed, and the remaining lever is GEMM
  quality. A post-batching figure of ~12× circulates by dividing that 20× by
  the 1.71× batching gain — it is arithmetic, not a measurement. The race has
  not been re-run (it needs a working EXLA, which this repo deliberately does
  not depend on), so no post-batching number is claimed. See the README's
  [vs EXLA section](README.md#vs-exla-august-2026-pre-batching).

Coverage parity is a matter of grinding through a list. Performance
parity is not, and no date is claimed for it here.

**And it may not be the goal.** A width sweep in August 2026
([`bench_results/MODEL_SCALING.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/MODEL_SCALING.md))
measured EXLA on the *host CPU* beating this backend by 20–215× on the same
gradient across every size tested, with the gap widening — on the same machine,
without touching a GPU. Against `BinaryBackend` this backend wins decisively
above ~10³ elements; against a compiler it does not win anywhere reachable.

That reframes the project rather than diminishing it. The value is **reach**:
the FreeBSD Keplers cannot run EXLA at all, and there `BinaryBackend` is the
real alternative. Performance parity with EXLA is not the bar this has to
clear, and pursuing it on a CUDA-capable box is chasing a race that is already
lost to a better-resourced compiler. The bar is being *correct and available*
where nothing else is.

Which ops actually get built is **demand-driven** — the standing
position in `PLAN_AFTER_BACKWARD_PASS.md` T7 is that a remaining
fallback is a recorded decision, not an oversight, and each is picked up
"if a workload appears". Correctness never depends on it: the host path
is right, just slow.

**Custom `Nx.Defn` compiler.** Done — `Nx.Vulkan.Compiler` (thrust 3).
Eager execution still runs through `Nx.Defn.Evaluator` (one op per
dispatch); passing `compiler: Nx.Vulkan.Compiler` to `Nx.Defn.jit`
instead traces the whole graph and compiles it to a stage schedule:
elementwise chains fuse to one generated shader, an elementwise chain
feeding a reduction fuses to one parallel tree-reduce, and graphs with
`dot`/`conv`/`reduce`/`transpose` boundaries split into on-device stages
(`reshape`/`squeeze` are zero-copy views; tuples multi-output). f32 and
f64. Whole dense/CNN layers, classifier heads, softmax, layernorm and
`x @ Wᵀ` fuse with no interpreter fallback. Remaining perf-heuristic
work (cross-stage CSE) was raced and left default-off. See the README's
[fusion compiler section](README.md#the-nxdefn-fusion-compiler-thrust-3).

**Shipped ≠ a win on every graph.** Measured against eager on the Axon
MNIST model, the compiler runs at **0.76× on a dense-only MLP** (a 24%
regression) and 0.98× on a conv CNN — correct in both cases, just
slower. It splits stages at `dot` boundaries, so a graph that is almost
all `dot` has nothing to amortise tracing and boundary buffers against.
Fusion is opt-in, so this costs nothing by default, but it means
"whole-graph compilation" is not a blanket improvement here and the
README should not be read as claiming so. Gating it on a traced-graph
statistic is T2 in
[`PLAN_AFTER_BACKWARD_PASS.md`](https://github.com/borodark/nx_vulkan/blob/main/PLAN_AFTER_BACKWARD_PASS.md).

**Batched command submission.** Done — dispatches are recorded and
submitted as one command buffer with one fence wait instead of a
submit-and-block per op. **1.45–1.71× on a training step, raced on all
three fleet hosts with no hardware crossover**, losses bit-identical.
`NXV_BATCH_MAX=0` restores the old behaviour.
[`bench_results/BATCHED_DISPATCH.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/BATCHED_DISPATCH.md).

Worth recording that the evidence for this predated the work by three
months: `PLAN_GPU_NODE.md`'s H3 measured **1.13 ms fence wait against
138 µs submit** per `submit_and_wait` back in May — the wait dominating
by 8× is exactly what batching amortises. The finding sat in a plan
document until the EXLA race independently pointed at per-dispatch cost.

**Persistent buffer pool.** Still open — per-call allocation through
vulkano's `StandardMemoryAllocator`. The old note here claimed it "costs
a millisecond per dispatch"; that figure is **unverified** and predates
batched submission, which changed the per-dispatch picture, so treat it
as a hypothesis to measure rather than a number to quote. Tracked as T4
in [`PLAN_AFTER_BACKWARD_PASS.md`](https://github.com/borodark/nx_vulkan/blob/main/PLAN_AFTER_BACKWARD_PASS.md),
whose "done when" is that allocation stops appearing in a per-step profile.

**f64 matmul.** Done — `matmul_f64.spv` ships and rank-2 matmul runs
natively in f64. The backend now dtype-dispatches **native f32** as
well (matmul, conv, elementwise, reduce, transpose), with f64 the
default accumulator policy; f32 is no longer merely cast. General
`Nx.dot` axis configs outside rank-2×rank-2 still host-fall-back.

**Scholar — linalg fast paths.** Linear regression (normal equation
+ SVD) now smoke-tests cleanly via a host-fallback `block/4`
callback that routes `Nx.Block.LinAlg.SVD`/`QR`/`solve`/`cholesky`
through `BinaryBackend`. Coefficients match to 2e-6. Native SVD/QR
shaders would speed things up but aren't blocking correctness.

The old "2–4 weeks to add the most-used ones natively" estimate is
**withdrawn as unsupported**. It is the one number here not backed by a
measurement or a comparable piece of finished work, and dense GPU
SVD/QR is a materially harder problem than the shaders this project has
shipped so far — iterative, convergence-sensitive, and awkward to make
bit-reproducible across the fleet, which is a documented property here.
No estimate is offered until someone prototypes one.

## Two-backend history (why both live here)

The spirit backend (`Nx.Vulkan.Backend`) reached production first
— chain-shader synthesis, runtime SPV compilation, content-
addressed disk cache, and a long-lived `Nx.Vulkan.Node` GenServer.
Then a use-after-free in the C++ FFI layer crashed the live trader
three minutes after every restart. The failure surfaced as
`Nx.Vulkan.Native.byte_size` raising `:badarg` on a stale `VkBuf*`
pointer — a classic FFI ownership leak the C++ type system cannot
detect.

The vulkano backend (`Nx.Vulkan.VulkanoBackend`) grew from a spike
that proved the migration was mechanical: same SPV bytes in,
byte-identical chain tensors out, perf within ten percent on the
bench target. It replaced spirit for the production path.

The spirit Elixir backend (`Nx.Vulkan.Backend`) and its `Nx.Vulkan.Fuse`
macro were **dropped** (commit `bb94217`) once vulkano covered the
production path; vulkano is now the only Elixir-facing backend. The
`native/nx_vulkan_native` C++ crate directory is vestigial. Full story:
[*The Backend That Didn't Need to Know*](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html).

## Architecture

```
   ┌─────────────────────────────────────────────────────────┐
   │  Nx layer                                                │
   │  • Nx.Vulkan.VulkanoBackend        (eager backend)       │
   │  • Nx.Vulkan.Compiler              (Nx.Defn fusion JIT)  │
   └──────────────────────────┬──────────────────────────────┘
                              │
   ┌──────────────────────────▼──────────────────────────────┐
   │  Nx.Vulkan.NativeV  (Rustler crate nx_vulkan_vulkano)    │
   │  • Arc<Buffer> resources   • pipeline cache              │
   │  • specialisation          • generic dispatch (JIT SPVs) │
   └──────────────────────────┬──────────────────────────────┘
                              ▼
              ┌─────────────────────────┐
              │  Vulkan driver (loader) │
              └─────────────────────────┘
                              │
              ┌───────────────▼─────────────────────────────┐
              │  priv/shaders/*.spv (f32 + f64 variants)     │
              │  • elementwise binary/unary  • reduce_axis   │
              │  • matmul (tiled)  • conv (im2col + GEMM)     │
              │  • transpose  • select / compare / cast      │
              │  • synthesised leapfrog chain shaders        │
              │  priv/shader_cache/gen_*.spv (JIT-generated) │
              └──────────────────────────────────────────────┘
```

The SPV catalog under `priv/shaders/` backs the eager path; the fusion
compiler generates and caches shaders under `priv/shader_cache/`.
The synthesis pipeline that produces new chain shaders at runtime
(`Nx.Vulkan.Synthesis`, `Nx.Vulkan.ShaderTemplate`,
`Nx.Vulkan.ChainShaderSpecs`) lives in the Elixir layer and is
backend-agnostic.

Old spirit-era infrastructure that survives unchanged:

- **`Nx.Vulkan.Node`** — long-lived named GenServer that owns the
  `vkPipelineCache` blob and serialises dispatch via `with_node/2`.
  Used by the legacy backend; the new backend doesn't require it
  but cooperates with it.
- **`Nx.Vulkan.PipelineCache`** — disk-persistent `vkPipelineCache`
  with UUID validation. Survives BEAM restarts.
- **Runtime chain shader synthesis** — render a `FamilySpec`, hand
  to `Synthesis.compile/1`, get a content-addressed SPV path back.
  ~150 ms cold, 5 ms cache hit. Both backends consume the output.
