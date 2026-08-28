# Changelog

## Unreleased

### BREAKING — `Nx.Vulkan.Fast` removed

The module is deleted. It was the **fused-kernel dispatch seam**, and both
halves of what made it a seam are gone.

Each function emitted an `Nx.Defn.Expr` optional/3 IR node naming a backend
callback, with a `defn` fallback: a backend implementing
`fast_leapfrog_position/4` dispatched ONE fused shader, everything else ran the
composed primitives. **Nx 0.12 removed `Expr.optional/3`** (`3a77d9e`), so the
callbacks went and the functions became their own fallbacks — `leapfrog_position/3`
is `Nx.add(q, Nx.multiply(eps, p))`, a two-op composition behind a name that no
longer selects anything.

**And the fused path was f32-only.** Every callback gated on `all_f32?/1`, so a
non-f32 operand fell through to the fallback by design. This backend then went
f64-first (`bb94217` dropped the f32 shaders), which means that even with
`Expr.optional` intact, the f64 leapfrog this project actually runs would have
taken the fallback every time. `Nx.Vulkan.fused_chain_4/…`, the dispatch target,
no longer exists either.

So: no mechanism, no target, and the wrong dtype. Six functions with zero callers
in this repo or its consumer.

**If you called it**, replace with the composition each function documented —
they are one or two Nx ops apiece, and since `3a77d9e` that is literally all they
were:

| removed | equivalent |
|---|---|
| `leapfrog_position(q, eps, p)` | `Nx.add(q, Nx.multiply(eps, p))` |
| `leapfrog_momentum_half(p, half_eps, grad)` | `Nx.add(p, Nx.multiply(half_eps, grad))` |
| `momentum_step(p, eps, grad)` | `Nx.add(p, Nx.multiply(eps, grad))` |
| `inv_mass_apply(p, inv_mass)` | `Nx.multiply(p, inv_mass)` |
| `kinetic_energy(p, inv_mass)` | `0.5 * Nx.sum(p² · inv_mass)` |
| `normal_logpdf(x, mu, sigma)` | `-0.5·z² - log(σ) - 0.5·log(2π)`, `z = (x-mu)/σ` |

**One thing to carry across if you reimplement `normal_logpdf`.** The
`-0.5·log(2π)` constant must be materialised at the computation's type.
`Nx.tensor/1` defaults to `{:f, 32}`, so a bare Elixir float silently degrades an
f64 chain: the module shipped with exactly that defect for four months, giving an
f64 RESULT TYPE carrying an f32-precision VALUE, off by 1.6e-8. It was invisible
because it was correct in the f32 world the module was written for, and nothing
re-examined it when the project became f64-first.

That error is a CONSTANT, so it cancels exactly in a log-RATIO — Metropolis
acceptance was unaffected. It accumulates in a summed absolute log-likelihood:
N × 1.56e-8, which is 0.016 at N = 1e6, enough to matter for model comparison.


### Fixed — scalars were refused by three capability gates

`compare`, `select` and the broadcasting binary path all gated the GPU on
`tuple_size(out.shape) >= 1`. Nothing in the shaders needed it: rank 0 now
dispatches as rank 1 of shape `{1}`. The consequence was that
`Nx.greater(scalar, scalar)`, `Nx.equal(scalar, scalar)`,
`Nx.select(scalar, scalar, scalar)` and `Nx.multiply(f64_scalar, f32_scalar)`
computed on the host, while the same ops on tensors did not — 108 of the 137
host fallbacks in one `value_and_grad` of a small probabilistic model
(`bench_results/EXMC_PEROP_RACE.md`), since a distribution's log-prob is mostly
scalar support checks.

Same bug class as the 0.3.0 backward-pass release: a gate written against the
shapes one workload happens to produce.

### Added — `put_slice` runs on the GPU

`glsl/put_slice.comp`, an index-remap overlay: one thread per output element,
reading the slice inside the window and the tensor outside it. Any 4/8-byte
dtype, rank 1–4, integer or scalar-tensor start indices, starts clamped exactly
as `Nx.BinaryBackend` clamps them. Rank > 4, sub-word dtypes and rank-0 tensors
still host-fall-back.

### Fixed — `pad` refused a mistyped pad value

`pad` has had a shader since 0.2.0, but its gate required the pad value to
already carry the tensor's dtype, and `Nx.pad(t, 0.0, cfg)` hands the callback
an f32 (or s32) literal. The pad value is now cast to the output type instead.

### Fixed — `pad`'s host tail returned wrong values for integer dtypes

`Nx.pad(Nx.iota({4}, type: :u8), 0, [{1, 1, 0}])` returned garbage: the host
tail re-entered `Nx.pad/3` with a *tensor* pad value, which merges types more
strictly than the number Nx was originally given, and the differently-typed
result was then filed under the callback's output type. Pre-existing; found by
the parity sweep for the above.

## 0.3.0 (2026-08-08)

The backward-pass release. 0.2.0 shipped conv, the fusion compiler and native
f32 — but a CNN *training* step still ran mostly on the CPU, and nothing in the
suite could see it.

Being precise about what that means, because "broken" would overstate it:
**0.2.0 always computed correct results.** The host fallback *is* the
`Nx.BinaryBackend` reference, bit for bit, and forward/inference genuinely ran
on the GPU. What was wrong is that it was an *inference* backend published as a
*training* backend. A LeNet training step took 20.9 s; it takes 84 ms here.
Anyone following the README's "autograd for free" headline got ~250× less than
advertised, and no assertion on values could have revealed it — which is why
this release ships the tooling that can (`Nx.Vulkan.Fallback`) alongside the
fix.

This release also stops paying one command-buffer submit and one fence wait per
op, worth a further 1.45–1.71× on a training step across the fleet.

If you are on 0.2.0 and training on the GPU, upgrade.

### Fixed — the backward pass

A host fallback returns a bit-identical result, because the fallback *is* the
`Nx.BinaryBackend` reference every test compares against. So no assertion on
values can detect that an op silently left the GPU, and one had: conv's entire
backward pass ran in pure Elixir. Nx.Defn.Grad (hidden, so not linked) emits
ops nobody writes by
hand, and every GPU fast path had been gated on the shapes a *forward* pass
produces.

Eight ops moved back on-device. Seven were narrow gates refusing work the
existing shaders could already do; only `reverse` and `broadcast` needed new
kernels.

- **conv** — accepts non-identity input/kernel/output permutations (the
  gradient swaps the first two axes) by rotating into the native layout
  on-device, and coerces a mismatched operand dtype instead of refusing it.
- **dot** — accepts any rank-2 contraction orientation. `y = x·W` contracts
  `[1]/[0]` and always hit the shader; its gradients arrive as `[1]/[1]` and
  `[0]/[0]`, so **every dense layer** was paying two host matmuls per step.
- **reduce** — accepts a kept axis in the middle (`sum(axes: [0,2,3])`, the
  conv bias gradient) by rotating the kept axes to the front.
- **max/min pooling**, both directions. Forward is one thread per output;
  backward is one thread per *input*, which is what avoids float atomics and
  is why it requires non-overlapping windows. Ties go to the **last** maximum
  in row-major order, verified against `BinaryBackend` — `>` instead of `>=`
  is correct on random data and wrong on every relu-zero tie.
- **reverse** and **broadcast** — new index-remap shaders (rank ≤ 4). Both had
  no shader at all; `broadcast` produced the `{:s, 32}` zeros that in turn made
  `select` fall back.
- **integer literals** — `coerce_to/2` now converts rank-0 integer constants
  and, via new `cast_s32_to_f32/f64` shaders, integer tensors of any shape. Nx
  materialises literals as `{:s, 32}`: relu's `max(x, 0)`, a mean's divisor,
  `select`'s zeros and pooling's `init_value` were each dragging a whole tensor
  to the CPU behind a four-byte constant.

A CNN training step now performs exactly **one** host fallback: `pow` in f64,
which GLSL.std.450 does not provide and which should stay a fallback rather
than silently boundary-cast through f32.

### Added — verification that can see this class of bug

- **`Nx.Vulkan.Fallback`** — counts host fallbacks per process, attributing
  each to the callback at compile time. `assert Fallback.count_total(fun) == 0`
  turns an invisible performance cliff into a test failure. Off by default.
- **`test/nx_vulkan/grad_test.exs`** — gradient parity against
  `BinaryBackend`. The suite previously had **no** gradient tests at all, which
  is how the conv regression survived; "autograd for free" was the headline
  claim of the README and nothing exercised it.
- **`docs/BACKEND_VERIFICATION_GAP.md`** — what the Nx ecosystem does and does
  not verify about a third-party backend. `doctest Nx` contains zero gradient
  examples, `deps/nx` ships no `test/`, and upstream's own
  `Nx.Helpers.check_grads!` is behind the packaging wall.

### Performance

One `value_and_grad` step, batch 32, vs `Nx.BinaryBackend`, losses
bit-identical:

| model | RTX 3060 Ti | GT 650M (2012) | GT 750M (2013) |
|---|---|---|---|
| conv→conv→dense | 31.0 ms (436×) | 35.1 ms (477×) | 25.4 ms (440×) |
| LeNet-style | 84.1 ms (363×) | 77.6 ms (434×) | 64.3 ms (334×) |

The LeNet step was **20.9 s** before this work. Absolute GPU times cluster in
25–85 ms across cards spanning 2012–2021 — at this size the work is
dispatch-bound, not compute-bound.

### Changed — one submit per batch of dispatches, not one per op

Acting on "dispatch-bound" above. Every op used to build its own command
buffer, submit it, and block on `queue.wait_idle()`. Dispatches are now
recorded into a pending queue and submitted as **one command buffer with one
fence wait**, flushed automatically at every host boundary (`buf_download`,
`buf_upload_into`, `concat_buffers`, `fft`) and at a size cap. Correctness
never depends on flushing by hand.

This is safe rather than a synchronisation minefield for two reasons:
vulkano's `AutoCommandBufferBuilder` tracks resource usage while recording and
inserts the pipeline barriers between commands in `build()`, so a
read-after-write between two batched dispatches is synchronised; and the only
way a value reaches the host is a download, which flushes first.

One `value_and_grad` step of an MNIST MLP at batch 32, submit-per-dispatch vs
batched, losses bit-identical:

| host | GPU | before | after | |
|---|---|---:|---:|---|
| super-io | RTX 3060 Ti (Ampere) | 16.446 ms | 9.627 ms | **1.71×** |
| mac.247 | GT 650M (Kepler, 2012) | 14.583 ms | 8.829 ms | **1.65×** |
| 248 | GT 750M (Kepler, 2013) | 13.301 ms | 9.147 ms | **1.45×** |

The loss is identical in every arm on every host — two architectures, two
operating systems. Batching changes *when* work is submitted, never what is
computed. **No hardware crossover**, unlike register blocking (Ampere-only) and
the many-slot fused reduce (Kepler-only), so this needs no
`Nx.Vulkan.Device.class/0` gate and is on by default.
[`bench_results/BATCHED_DISPATCH.md`](bench_results/BATCHED_DISPATCH.md).

### Added — dispatch batching controls

- **`Nx.Vulkan.NativeV.flush/0`** — submits any recorded-but-unsubmitted
  dispatches and waits. Never required for correctness; it matters for
  *measurement*, since timing a loop with no readback now times the recording
  rather than the work.
- **`NXV_BATCH_MAX`** — dispatches to record before forcing a submit (default
  64). `0` restores submit-per-dispatch, and is the A/B control for any
  measurement of this change as well as the escape hatch if a driver dislikes
  long command buffers. The cap measured flat from 32 to 256 on all three
  fleet hosts, so 64 is a safe default rather than a tuned constant.

### Measured — where the remaining gap is, and where it is not

Raced against EXLA (CUDA) on the Axon MNIST model, one training step, losses
bit-identical to `BinaryBackend`
([`bench_results/MNIST_EXLA_RACE.md`](bench_results/MNIST_EXLA_RACE.md)):

- **The fusion compiler *regresses* on a matmul-dominated graph** — 0.76× on
  the dense MLP, 0.98× on the CNN. `Nx.Vulkan.Compiler` splits stages at `dot`
  boundaries, so a graph that is almost all `dot`s has nothing for tracing and
  boundary buffers to amortise against. Reading the eager-vs-fused rows and
  concluding "we need more fusion" would build the wrong thing; that is what
  motivated batching instead.
- **Correction to an earlier claim.** A previous version of that file said EXLA
  "failed to compile" conv. That was wrong, and wrong in the direction that
  flattered this project — it was written from one failing run without
  isolating the cause. EXLA compiles and trains convolutional models normally.
  A 17-variant matrix narrows the real failure to **two stacked convs + stride
  2 + `channels: :first`, in the gradient only**; any single relaxation
  compiles, and Axon's default layout (`:last`) avoids it entirely.

### Notes

- `standard_deviation/2` and `covariance/3` joined the doctest `@rounding`
  bucket: both used to host-fall-back and match exactly, and now run natively
  1 ULP away. Excepting a function drops all of its doctests (863 → 851), so
  the bucket is worth watching rather than growing silently.
- Nx.BinaryBackend's `window_scatter_max` round-trips f64 through f32. For f64
  pooling gradients this backend is now *more* accurate than the reference it
  is tested against, so the pooling test asserts values are exact elements of
  the source rather than agreement with the host.
- `docs/BACKWARD_PASS_AUDIT.md` records what the audit established and how the
  bug class stayed invisible; `PLAN_AFTER_BACKWARD_PASS.md` carries the
  remaining work, each item with the measurement that motivates it.
- `test/nx_vulkan/node_test.exs` did check-then-act (`whereis` then `stop`) on a
  globally named GenServer that every test `start_link`s from the test process,
  so the node was already dying from its link when `on_exit` ran. A latent
  flake, unrelated to any GPU work; now tolerates the pid dying in that window.

## 0.2.0 (2026-08-02)

The fusion compiler release. First release since 0.1.0.

### Added

- **`Nx.Defn` fusion compiler (`Nx.Vulkan.Compiler`).** An
  `Nx.Defn.Compiler` that traces a `defn` to a stage schedule running
  on-device with GPU-resident intermediates and no interpreter fallback:
  elementwise fusion (one shader, one dispatch), parallel fused
  reductions (workgroup-per-slot tree reduce), and a multi-stage split at
  `dot` / `conv` / `reduce` / `transpose` boundaries with `reshape` /
  `squeeze` as zero-copy view boundaries and tuple/multi-output support.
  Whole dense/CNN layers, classifier heads, softmax, layernorm, and
  `x @ Wᵀ` fuse. Both f32 and f64. Cross-stage CSE exists but is
  default-off (raced across the fleet, never wins; opt-in `NXV_CSE=1`).
  Use: `Nx.Defn.jit(&fun/2, compiler: Nx.Vulkan.Compiler)`.
- **Native f32 compute.** The hot ops (elementwise, matmul, conv, reduce,
  transpose) now dtype-dispatch native **f32** shaders alongside f64,
  instead of casting f32 to f64. f64 remains the default accumulator
  policy (safe); f32 wins on bandwidth-bound ops and is available where a
  workload opts into it. This supersedes the "f64-only compute" note
  below for those ops.
- **conv** (im2col + GEMM) and **transpose** as native GPU ops, in f32
  and f64.

### Changed

- **f64-only compute** *(later superseded — see "Native f32 compute"
  under Added above; native f32 shaders were re-added for the hot ops).*
  Native shaders — elementwise binary/unary,
  reductions, rank-2 matmul (`matmul_f64.spv`), and the leapfrog chain
  synth — now run in **f64**; f32 inputs are accepted and cast to f64.
  This supersedes 0.1.0's "elementwise f32 + f64 / matmul f32-only"
  coverage: matmul is now native f64 (was host-fallback for f64), and
  the eXMC precision contract is `:f64` (EMLX, the f32-only backend, was
  dropped). Consumer GPUs are slower at f64, but correctness took
  priority over the f32 speed path.
- **Support Nx 0.13.** The `:nx` version constraint is now `{:nx, "~> 0.13"}`
  (was `~> 0.10 or ~> 0.11 or ~> 0.12`). `VulkanoBackend` and the f64
  chain-shader synthesis path run unchanged against nx 0.13's backend API.
  Required for consumers on nx 0.13 — e.g. eXMC's vulkan milestone, which
  references this repo from `mix` and could not resolve against the prior
  sub-0.13 constraint.

## 0.1.0 (2026-05-20)

First Hex release.

### Added — `Nx.Vulkan.VulkanoBackend` (pure-Rust path)

A new `Nx.Backend` implementation built on the [vulkano](https://github.com/vulkano-rs/vulkano)
Rust wrapper around Vulkan compute. Sibling to the existing
`Nx.Vulkan.Backend` (C++ spirit-backed); they share the SPV
catalog under `priv/shaders/` and the chain-shader synthesis
pipeline.

**Why a second backend.** A use-after-free in the C++ FFI layer
crashed the live trader three minutes after every restart —
`Nx.Vulkan.Native.byte_size` raising `:badarg` on a stale
`VkBuf*` pointer that had outlived its referent. Vulkano's
`Arc<Buffer>` ownership makes that bug class structurally
impossible: a `Subbuffer<u8>` cannot outlive its parent at the
Rust type level.

**What it ships.**

- Buffer lifecycle: `buf_upload`, `buf_alloc`, `buf_download`,
  `buf_byte_size`, `buf_upload_into`. Each wraps a vulkano
  `Subbuffer<[u8]>` in a Rustler resource; the BEAM GC's drop
  triggers vulkano's `vkDestroyBuffer + vkFreeMemory` chain.
- Compute ops (24 native through specialised SPVs):
  - **Elementwise binary** (f32 + f64): add, subtract, multiply,
    divide, pow, max, min.
  - **Elementwise unary** (f32 + f64): exp, log, sqrt, abs,
    negate, sigmoid, tanh, floor, ceil, sign.
  - **Reductions** (f32 + f64): sum, reduce_max, reduce_min;
    all-axes, leading-axis, trailing-axis.
  - **Shape / movement**: reshape (zero-copy), squeeze
    (zero-copy), 2D transpose.
  - **Matmul**: rank-2 × rank-2, f32 only.
- Host-fallback callbacks (correctness first; perf-native
  shaders pending): slice, as_type, comparison ops (equal,
  not_equal, less, less_equal, greater, greater_equal), select,
  all, any, dot (non-standard axis configs), `block/4`
  (routes `Nx.Block.LinAlg.SVD/QR/Cholesky/solve` through
  `BinaryBackend`).
- Pipeline cache keyed by `(spv_path, op_code)`. First call
  builds the layout + pipeline; subsequent calls reuse them.
  Required for long-running workloads (without it, vulkano's
  `StandardDescriptorSetAllocator` creates a fresh
  `DescriptorPool` per unique layout identity, eventually
  exhausting driver limits on FreeBSD).

**Validated workloads.**

- **Axon training step**: Dense → sigmoid → Dense + MSE +
  `Nx.Defn.value_and_grad`. Forward loss matches `BinaryBackend`
  byte-identical; gradient sum agrees to 1e-8. 100-step SGD
  trajectory matches at every step within 2e-6; final loss
  agrees to 4e-7 with both backends converging by 350×.
- **eXMC regime model log-posterior**: 8 free RVs, softmax-mixture
  custom likelihood over 200 observations. Matches `BinaryBackend`
  to 1e-7 at f64 precision. Roughly 2× faster than the C++ path
  on the bench target (GT 650M, FreeBSD 15.0).
- **Scholar linear regression** (normal equation + SVD):
  coefficients match `BinaryBackend` to 2e-6 on synthetic
  regression. SVD via host-fallback `block/4`.

**Autograd.** No backward callbacks were written. `Nx.Defn.grad`
is a graph transformation that expresses backward ops in terms
of forward ops — forward op coverage is therefore gradient
coverage when running through `Nx.Defn.Evaluator`. Validated
end-to-end via the Axon training step.

### Added — Mission II chain-shader synthesis

`Exmc.NUTS.CustomSynth`-style runtime synthesis of multi-RV
HMC/NUTS chain shaders. Take a multi-RV IR with a Custom
likelihood, trace via `Nx.Defn`, emit GLSL, compile to SPIR-V,
content-address cache, dispatch. Validated on the regime model
(8 RVs + 200-obs softmax-mixture) on GT 650M at 60 ms per K=32
leapfrog dispatch — 8.3× under the 500 ms/sample budget.

### Existing — `Nx.Vulkan.Backend` (C++ spirit path)

The legacy backend stays in this release. It runs the chain-shader
synthesis pipeline and the Mission II dispatch. The stale-handle
bug class that motivated the migration is still present; the
recommended path forward is `VulkanoBackend` for general Nx
work plus the spirit-backed chain dispatch (or vulkano's
chain-shader dispatch via `Nx.Vulkan.NativeV.leapfrog_chain_synth`)
for HMC.

### Build notes

- Rust 1.85 pinned via `rust-toolchain.toml`. See the comment in
  that file for the upstream rustler reason.
- Vulkan SDK + `glslangValidator` required:
  - Linux: `apt install libvulkan-dev vulkan-tools glslang-tools`
  - FreeBSD: `pkg install vulkan-loader vulkan-headers vulkan-tools glslang shaderc`
- vulkano 0.34 builds in ~30s on Linux, ~3:18 on FreeBSD 15.0.

### What's missing (the honest queue)

- Persistent buffer pool (per-call allocation works but costs
  a millisecond per dispatch).
- f64 matmul shader (regime model's `Nx.dot` falls back to host).
- Native linalg shaders (SVD, QR, Cholesky, solve) — Scholar
  currently routes these through host.
- Custom `Nx.Defn` compiler — today we run through
  `Nx.Defn.Evaluator` op-by-op; whole-graph optimisation is
  EXLA-style work.
- Convolutions, FFTs, sort, scatter — the long tail of Nx ops.
- R4 live-trader cutover — the production trader has not been
  switched to `VulkanoBackend` yet.

### Links

- Blog: [The Backend That Didn't Need to Know](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html)
- Roadmap: [`docs/VULKANO_BACKEND_ROADMAP.md`](docs/VULKANO_BACKEND_ROADMAP.md)
- 10-minute intro: [`livebooks/intro_10min.livemd`](livebooks/intro_10min.livemd)
- Examples: [`examples/axon_training_loop.exs`](examples/axon_training_loop.exs),
  [`examples/full_bench.exs`](examples/full_bench.exs)
