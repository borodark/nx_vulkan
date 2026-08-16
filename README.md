# Nx.Vulkan

A GPU tensor backend for [Nx](https://github.com/elixir-nx/nx) that runs on **anything with a Vulkan driver** — including FreeBSD, where CUDA and Metal don't exist.

```
✓ Linux + NVIDIA RTX 3060 Ti      (proprietary driver)
✓ FreeBSD + NVIDIA GT 750M        (NVIDIA legacy driver)
✓ FreeBSD + NVIDIA GT 650M        (NVIDIA legacy driver)
```

**Why this exists →** [`WHY.md`](WHY.md) — the f64 conviction, autograd-for-free, reach over peak FLOPS, and one-GPU-to-a-fleet.

## Goals

- **Cover the FreeBSD gap.** Nx's existing GPU backends (EXLA on
  Linux+CUDA, EMLX on macOS+Metal) don't run on FreeBSD. If you
  have NVIDIA on FreeBSD, this is the only path.
- **Portable Vulkan across vendors.** Same SPV blobs run on NVIDIA,
  AMD RADV, Intel, and via MoltenVK on macOS. The Rust vulkano crate
  keeps the driver surface pure Rust — no C++ FFI ownership traps.
- **Enough coverage that Nx models Just Work.** A native op set plus a
  host-fallback long tail let real workloads (Axon training, eXMC
  NUTS sampling, Scholar linear regression) run today without
  op-by-op porting.
- **Deterministic across hardware.** Cross-Kepler runs of the same
  IR produce byte-identical posterior chains. Reproducible science
  across GPU generations is a documented feature.

## What works today

- **A native compute op set** — elementwise binary/unary, reductions
  (sum / max / min over ANY axis set, including a kept axis in the middle),
  reshape / squeeze, transpose (any permutation, rank <= 4), matmul (any
  rank-2 contraction orientation), conv (im2col + GEMM, any layout
  permutation), FFT, select / compare / cast / broadcast / reverse, and
  max/min pooling in both directions.
  **Native f32 and f64** — the hot ops (elementwise, matmul, conv,
  reduce, transpose) dtype-dispatch native f32 shaders as well as f64;
  f64 is the default accumulator policy, f32 wins on bandwidth-bound ops.
- **Whole-graph fusion** via `Nx.Vulkan.Compiler`, an `Nx.Defn.Compiler`
  — see [The `Nx.Defn` fusion compiler](#the-nxdefn-fusion-compiler-thrust-3).
- **Batched command submission** — dispatches are recorded and submitted as one
  command buffer with one fence wait instead of a submit-and-block per op,
  worth **1.45–1.71×** on a training step across the fleet. Flushed
  automatically at every host boundary, so correctness never depends on
  managing it. `NXV_BATCH_MAX=0` restores submit-per-dispatch; `flush/0` exists
  for benchmarks that need to time the work rather than the recording of it.
- **Host fallback for the long tail** — sort/argsort, SVD/QR/solve/cholesky
  from `Nx.LinAlg` (via `Nx.Block.LinAlg`), rank-5+ shapes, and broadcasting
  `pow` (GLSL.std.450 has no f64 `pow`, so the broadcast shader omits it).
  Slow but correct.
- **A fallback counter** (`Nx.Vulkan.Fallback`) — a host fallback returns a
  bit-identical result, so no assertion on values can detect one. `count/1`
  makes it countable, and the suite asserts *zero* fallbacks for ops that must
  stay on-device. A full CNN training step now performs exactly **one**:
  broadcasting `pow`.
- **Strict mode** (`config :nx_vulkan, host_fallback: :allow | :warn | :raise`)
  — turns "detectable if you wrote the right assertion" into "impossible to
  miss". `:allow` is the default and always will be; `:raise` refuses any
  fallback not on a documented, one-line-per-entry allowlist. Scope it to a
  block with `Nx.Vulkan.Fallback.strict/1,2`, which is per-process and so safe
  in an `async: true` suite. `sh scripts/strict_test.sh` runs the whole suite
  that way. See [Strict mode](#strict-mode).
- **`Nx.Defn.grad` autograd**, for free — see [The autograd insight](#the-autograd-insight).
- **Axon training step** end-to-end, gradient sum agrees to 1e-8
  vs `BinaryBackend` reference.
- **eXMC NUTS regime log_p** at f64: byte-identical to CPU
  reference.
- **Scholar linear regression**: coefficients match `BinaryBackend`
  to 2e-6 (SVD via host fallback).
- **Long-running stability** — 7000+ chain-shader dispatches on
  Ampere without crash after the 2026-07 `primary_buffer_count=128`
  fix; unbounded on Kepler.
- **Pipeline cache** on disk, UUID-validated, survives BEAM restarts.

Roadmap and future work: [`ROADMAP.md`](ROADMAP.md).

## The `Nx.Defn` fusion compiler (thrust 3)

EXLA's structural edge over an eager backend is whole-graph compilation:
it fuses a chain of ops into one kernel instead of dispatching each
separately. `Nx.Vulkan.Compiler` is an `Nx.Defn.Compiler` that does the
same for the cases it supports — the closest this project comes to
closing that gap.

```elixir
Nx.Defn.jit(&my_fun/2, compiler: Nx.Vulkan.Compiler).(a, b)
```

It traces a `defn` to an expression DAG and compiles it to a **stage
schedule** that runs on-device with GPU-resident intermediates and no
fallback to the interpreter:

- **Elementwise fusion** — a same-shape f32/f64 chain becomes one
  generated GLSL shader, one dispatch (3.62× over eager per-op on a
  10-op chain). Broadcasting, CSE-within-a-shader, and scalar constants
  are baked in.
- **Parallel fused reductions** — an elementwise chain feeding
  `sum`/`product`/`max`/`min`/`mean` fuses into one workgroup-per-slot
  tree reduce (f64 accumulator, matches `BinaryBackend`).
- **Multi-stage split at boundaries** — a graph with a `dot`, `conv`,
  `reduce`, or `transpose` splits into stages: each boundary is a stage,
  each maximal elementwise region between boundaries is one shader whose
  inputs may be earlier stages' buffers. `reshape`/`squeeze` are
  zero-copy view boundaries (no dispatch); `transpose` moves data.
- **Multi-output** — a `defn` returning a tuple compiles to one shared
  schedule; subexpressions common to several outputs are computed once.
- **f32 and f64** — the codegen is dtype-parameterised (f64 gated on the
  device advertising `shader_float64`; f64 transcendentals host-fall-back
  rather than silently losing precision).

Whole layers fuse end-to-end: `relu(x @ W + b)`, `relu(conv(x, k) + b)`,
a CNN classifier head (`conv → flatten → dense`), softmax and layernorm
reduction patterns, and transposed-weight layers (`x @ Wᵀ`). Anything
unsupported falls back to `Nx.Defn.Evaluator`, so results are always
correct — worst case is "no fusion, same as eager."

## Standing

The fusion compiler is the goal this effort set out to reach: a credible
**#2 compute backend** for `elixir-nx`, with EXLA's whole-graph
compilation now present in the one place a Vulkan backend can offer it —
on any GPU with a driver, CUDA or not.

- **Correctness first.** Every fused result is checked exact against
  `Nx.BinaryBackend`. The suite — **843 doctests, 456 tests, 0 failures**
  — is green on both a 2012 Kepler (GT 650M, FreeBSD) and a 2021 Ampere
  (RTX 3060 Ti, Linux), with the f64 fused path active on both. Gradient
  parity and host-fallback counts are asserted, not assumed: `Nx.Defn.grad`
  is compared against `BinaryBackend` op by op, and a CNN training step is
  asserted to leave the GPU exactly once.
- **Fusion's win is structural — and it has a floor.** It removes dispatches
  and intermediate buffers and keeps the interpreter out of the loop; it does
  not make kernels faster. So the gain grows with the elementwise work around
  a boundary — and **below some amount of it, fusion is a net loss**. Measured
  on a dense-only MLP, where the graph is almost all `dot` and there is
  nothing to amortise tracing and boundary buffers against,
  `Nx.Vulkan.Compiler` runs at **0.76× of eager — a 24% regression** — and at
  0.98× on a conv CNN. Both are bit-identical to the host; they are correct
  and slower. Fusion is opt-in (`compiler: Nx.Vulkan.Compiler`), so this is a
  choice to make per graph shape rather than a default you inherit.
- **Every heuristic is fleet-validated, never assumed.** Win/loss
  crossovers are hardware-specific, so they are measured across the
  fleet (Kepler + Ampere), not the local box. The many-slot reduce is
  device-class-gated because it wins on weak GPUs and regresses on
  strong ones. Batched submission was raced the same way and wins on
  **all three** hosts with no crossover, which is why it needs no gate and
  ships on by default — the outcome, not the assumption. Cross-stage CSE was
  built, raced, and found to **never
  win on either device class** (recompute is cheaper than the dispatch
  it takes to avoid) — so it ships **default-off**, opt-in via
  `NXV_CSE=1`. See
  [`bench_results/CSE_SOFTMAX_RACE.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/CSE_SOFTMAX_RACE.md)
  and the write-up,
  [*Compute It Twice: When CSE Lost the Race*](https://www.dataalienist.com/blog-compute-it-twice.html).

Building on a compute kernel of your own? See the
[`vulkan-nx-compute`](https://github.com/borodark/nx_vulkan/tree/main/.claude/skills/vulkan-nx-compute) skill for the
shader → NIF → Nx playbook and the hard-won parity/dispatch gotchas.

## Position vs EXLA and EMLX

| | EXLA | EMLX | Nx.Vulkan.VulkanoBackend |
|---|---|---|---|
| **Backing API** | Google XLA | Apple MLX (Metal) | Khronos Vulkan via vulkano (Rust) |
| **Maturity** | Years; production | Released 2024 | Released 2026 |
| **Linux + NVIDIA CUDA** | ✓ canonical | ✗ | ✓ via Vulkan |
| **macOS + Apple Silicon** | ✗ | ✓ canonical | ✓ via MoltenVK |
| **FreeBSD + NVIDIA** | ✗ | ✗ | **✓ only path** |
| **Windows / WSL2** | partial via TF | ✗ | ✓ (Vulkan ships on Windows) |
| **Op coverage** | full Nx surface (~200) | full Nx surface | native core (elementwise, matmul, conv, reduce, pooling, layout ops), rest via host fallback |
| **`Nx.Defn` fusion compiler** | ✓ XLA whole-graph | ✓ MLX | **✓ multi-stage split** (elementwise/reduce/dot/conv/transpose, f32+f64) |
| **`Nx.Defn.grad` (autograd)** | full | full | **✓ free** (graph transformation) |
| **fp64 compute** | full | none (Metal limit) | ✓ native f32 **and** f64 (binary/unary/reduce/matmul/conv/transpose) |

### The autograd insight

`Nx.Defn.grad` is a graph transformation that runs at compile time
on the `Nx.Defn.Expr` AST. For every forward op in the graph, it
inserts the corresponding backward op expressed in terms of *more
forward ops*. The backend never sees a "backward op" — it just keeps
executing forward primitives. Forward op coverage IS gradient
coverage when running through `Nx.Defn.Evaluator`.

That means **VulkanoBackend supports gradients for any function
expressible in its native ops + host-fallback long tail**. No
backward callbacks were written. Validated by running a complete
Axon training step (Dense → sigmoid → Dense → MSE →
`Nx.Defn.value_and_grad`) on `Nx.Vulkan.VulkanoBackend`, with
gradient sum agreeing to 1e-8 against the `BinaryBackend` reference.

## Benchmarks

> **Read the baseline before the multipliers.** Most tables here compare against
> `Nx.BinaryBackend`, which is a pure-Elixir interpreter. That is the right
> reference for *correctness* — it is the host fallback, so it is what every
> result must match bit-for-bit — and it is the honest reference on the FreeBSD
> Keplers, where EXLA is not built and the interpreter genuinely is the
> alternative. It is **not** a performance reference on a machine that can run
> EXLA.
>
> Measured across a width sweep on the RTX 3060 Ti: **EXLA on the host CPU
> computes the same gradient 20× faster than this backend at small model sizes
> and 215× faster at 6×10⁶ elements**, with the gap widening. There is no
> reachable model width on that box where this backend is the faster choice
> ([`bench_results/MODEL_SCALING.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/MODEL_SCALING.md)).
>
> So a "436×" below means *436× a tree-walking interpreter*, not 436× a
> competent CPU. **This project's case is reach, not speed** — see
> [Position vs EXLA and EMLX](#position-vs-exla-and-emlx). Where CUDA exists,
> use EXLA.

### Batched dispatch (August 2026)

One `value_and_grad` step of an MNIST MLP at batch 32, submit-per-dispatch
(`NXV_BATCH_MAX=0`) vs batched. Same graph, same commit, arms differing only in
the environment variable:

| host | GPU | before | after | |
|---|---|---:|---:|---|
| super-io | RTX 3060 Ti (Ampere, 2021) | 16.4 ms | **9.6 ms** | 1.71× |
| mac-247 | GT 650M (Kepler, 2012) | 14.6 ms | **8.8 ms** | 1.65× |
| mac-248 | GT 750M (Kepler, 2013) | 13.3 ms | **9.1 ms** | 1.45× |

The loss is identical in every arm on every host — two architectures, two
operating systems. Batching changes *when* work is submitted, never what is
computed. Reproduce with
[`examples/mnist_mlp_step_bench.exs`](examples/mnist_mlp_step_bench.exs);
method and the full cap sweep in
[`bench_results/BATCHED_DISPATCH.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/BATCHED_DISPATCH.md).

> **The three tables below predate batched dispatch** and are therefore
> pessimistic by roughly the factors above. They have not been re-run because
> their harnesses were scratch projects rather than committed examples — so
> they are left at their measured values rather than adjusted by arithmetic.

### CNN training (August 2026, pre-batching)

One `value_and_grad` step, batch 32, versus `Nx.BinaryBackend`. Losses are
bit-identical to the host on every row.

| model | super-io (RTX 3060 Ti) | mac-247 (GT 650M, 2012) | mac-248 (GT 750M, 2013) |
|---|---|---|---|
| conv→conv→dense, strided | **31.0 ms** (436×) | 35.1 ms (477×) | **25.4 ms** (440×) |
| LeNet-style, max-pooled | **84.1 ms** (363×) | 77.6 ms (434×) | **64.3 ms** (334×) |
| inference, batch 256 | 17.5 ms (1107×) | 71.1 ms (274×) | 31.8 ms (407×) |

The LeNet step was **20.9 s** before the backward pass stopped falling back to
the host — the same measurement, same box. Read the absolute times rather than
the multipliers: a speedup here mostly measures how slow pure-Elixir
`BinaryBackend` is, which varies by host CPU. The absolute GPU times cluster in
25–85 ms across three cards spanning 2012–2021, because at this model size the
work is dispatch-bound rather than compute-bound — which is why a 2012 laptop
GPU is competitive with a 2021 desktop one.

### vs EXLA (August 2026, pre-batching)

The [Axon MNIST guide](https://axon.hexdocs.pm/mnist.html) model, one training
step at batch 32 on the RTX 3060 Ti — a dense-only MLP, which is the shape most
favourable to EXLA and least favourable here:

| backend | ms | vs BinaryBackend |
|---|---:|---:|
| Vulkan, eager | 14.1 | 485× |
| Vulkan, `Nx.Vulkan.Compiler` | 18.5 | 370× |
| EXLA (CUDA) | 0.715 | 9581× |

**EXLA is ~20× ahead, and fusion does not close it — it costs 24%** (0.76×
fused vs eager). On a graph that is almost all `dot`, there is no elementwise
work for fusion to amortise against, so the deficit is dispatch overhead and
GEMM quality rather than missing whole-graph compilation.

That diagnosis is what batched submission acts on, and it is why this race is
labelled pre-batching: the eager row above has since improved ~1.7× on this
box. The race has **not** been re-run — it needs a working EXLA, which this
repo deliberately does not depend on — so no combined figure is claimed here.
The remaining lever the measurement points at is GEMM quality.

On a 2×strided-conv CNN the gap is similar — 41.3 ms eager vs 1.45 ms, with
fusion neutral at 0.98× — so EXLA leads on both graph shapes tested. The
qualitative difference is availability: EXLA cannot be installed on the two
FreeBSD Keplers at all, where Vulkan runs the same CNN in 64–78 ms. Full
numbers, an XLA gradient-compile failure narrowed to one specific conv
configuration, and what it took to get EXLA running:
[`bench_results/MNIST_EXLA_RACE.md`](https://github.com/borodark/nx_vulkan/blob/main/bench_results/MNIST_EXLA_RACE.md).

### f32 vs f64 per op

`sh scripts/race.sh` — f32 speedup over f64, same shapes, all on-GPU:

| op | super-io | mac-247 | mac-248 |
|---|---|---|---|
| matmul 512³ | 0.61× | 0.47× | 0.45× |
| conv 16→32ch | 1.7× | 1.21× | 3.48× |
| elementwise add 1M | 2.38× | 4.3× | **7.01×** |
| sum 1M | 1.97× | 1.9× | 1.9× |

**f32 matmul is slower than f64, and that is the accumulator policy working as
designed**: f32 matmul defaults to `matmul_f32_f64acc.spv`, paying an f32→f64
conversion on top of the same f64 MAC rate. f32 wins where it is meant to —
bandwidth-bound elementwise and reductions. Switch with
`Nx.Vulkan.VulkanoBackend.put_f32_matmul_accumulator(:f32)` if you want speed
over the f64-accumulated reference.

### Square matmul (May 2026)

Milliseconds per dispatch, median of 50–200 iterations:

| size | bin (super-io) | bin (mac-247) | vulkano (super-io) | vulkano (mac-247) |
|---|---|---|---|---|
| 16×16 | 2.76 | 2.51 | 1.18 | **1.06** |
| 64×64 | 130.76 | 158.45 | 7.07 | 7.92 |
| 256×256 | 20,097 | 13,891 | 149.19 | **136.10** |
| 1024×1024 | n/a (hours) | n/a (hours) | 2,323 | 2,843 |

Full bench: [`examples/full_bench.exs`](examples/full_bench.exs).

## Quickstart

### As a backend in your project

```elixir
# mix.exs
def deps do
  [
    {:nx, "~> 0.13"},
    {:nx_vulkan, "~> 0.3"}
  ]
end
```

Or track `main` directly:

```elixir
{:nx_vulkan, git: "https://github.com/borodark/nx_vulkan"}
```

> On 0.2.0 and training on the GPU? Upgrade. 0.2.0 computed correct results,
> but its backward pass ran largely on the host — a LeNet step took 20.9 s
> there and 84 ms here. See the [CHANGELOG](CHANGELOG.md).

```elixir
# Build a tensor, transfer to GPU, do work
x_bin = Nx.tensor([1.0, 2.0, 3.0, 4.0], type: :f32)
x_vk  = Nx.backend_transfer(x_bin, Nx.Vulkan.VulkanoBackend)

y_vk  = Nx.sigmoid(x_vk)
y_bin = Nx.backend_transfer(y_vk, Nx.BinaryBackend)
IO.inspect(Nx.to_list(y_bin))
# [0.7310585975646973, 0.8807970881462097, 0.9525741338729858, 0.9820137619972229]
```

### Strict mode

Every op this backend cannot run natively transfers to `Nx.BinaryBackend`,
computes, and transfers back. The result is **bit-identical** — the fallback
*is* the reference implementation — so no assertion on values can ever tell
"ran on the GPU" from "silently didn't." That is not hypothetical: conv's
entire backward pass ran on the CPU for the whole life of the conv shaders,
with a green suite and green doctests, until somebody counted.

`Nx.Vulkan.Fallback.count/1` makes a fallback detectable. Strict mode makes it
impossible to miss:

```elixir
config :nx_vulkan, host_fallback: :allow   # default — correct, just slower
config :nx_vulkan, host_fallback: :warn    # log every non-allowlisted fallback
config :nx_vulkan, host_fallback: :raise   # raise Nx.Vulkan.HostFallbackError
```

`:allow` is the default and is meant to stay the default. A library that
raised on a correct-but-slow path would give up the property that makes this
backend usable at all: *every* `Nx` op works on it.

Scope it to a block instead of the whole VM. This is per-process, so one strict
test cannot poison an `async: true` suite:

```elixir
Nx.Vulkan.Fallback.strict(fn -> Nx.Defn.jit_apply(train_step, params) end)
Nx.Vulkan.Fallback.strict(:warn, fn -> ... end)
Nx.Vulkan.Fallback.strict(:allow, fn -> Nx.LinAlg.svd(a) end)   # escape hatch
```

**Why raise rather than count.** The counter is documented as a *lower bound*:
once a fallback strands a tensor on `Nx.BinaryBackend`, Nx dispatches
everything downstream there without this backend ever seeing it, so a census
shows the visible edge of a cascade. `:raise` fires on the **first** refused
op, before the tensor has left the device — so a strict failure names the
cause, not the symptom.

**The allowlist is the whole risk surface.** `@allowlist` in
`lib/nx_vulkan/fallback.ex` is one line per exemption, each naming a single
`{fun, arity}` and the reason it is permitted. There are no wildcards and no
"this op family may fall back": `{:transpose, 3}` is not exempt — only
`{:transpose, 3}` *at rank ≥ 5* is, because the rank-4 case has a shader and a
rank-4 transpose leaving the GPU is a bug. An allowlist that grows loosely is
how `:raise` comes to mean nothing, which is the same way the original gates
got too wide.

Run the whole suite that way:

```sh
sh scripts/strict_test.sh
```

Two tag-based exclusions, both enumerated in the tree and both printed by the
CI job: `:host_fallback_expected` for tests whose *subject* is the fallback
path, and `:host_fallback_open` for real fallbacks that are tracked and open.
Neither is skipped by a normal `mix test`.

Nx's own doctests run under the ratchet too, minus the ones named in
`test/nx_doctest_register.exs` — 524 of 843 that still leave the GPU for at
least one op, each line carrying its reason. That makes the share of upstream
Nx this backend actually computes on-device a number you can print:

```sh
sh scripts/doctest_residency.sh
#=> doctest Nx residency: 319 / 843 (37.8%) run with host fallbacks refused
```

It fails if a doctest not in the register falls back (a regression) *or* if one
in the register stops falling back (a stale entry holding the number down), so
the rate moves only when someone edits the register deliberately.

### Try the Axon training example

```sh
git clone https://github.com/borodark/nx_vulkan
cd nx_vulkan
mix deps.get && mix compile
elixir examples/axon_training_loop.exs
```

Runs a 100-step Dense(4→32, tanh)→Dense(1) regression with manual
SGD. Compares loss trajectories on `BinaryBackend` vs
`VulkanoBackend`. PASS verdict on both Linux + FreeBSD.

### Try the full bench

```sh
mix run examples/full_bench.exs
```

Per-op + end-to-end + robustness across every backend Nx can find.
Auto-detects EXLA availability. Runs in ~10 minutes on RTX 3060 Ti,
~15 on GT 650M.

## Building

### Prerequisites

- Erlang/OTP 26+, Elixir 1.17+
- Rust 1.78+
- C++ compiler (only needed for the legacy spirit backend; vulkano
  is pure Rust)
- Vulkan SDK + `glslangValidator`:
  - Debian/Ubuntu: `apt install libvulkan-dev vulkan-tools glslang-tools`
  - FreeBSD: `pkg install vulkan-loader vulkan-headers vulkan-tools glslang shaderc`

### Build

```sh
mix deps.get
mix compile
```

Vulkano compiles in ~30s on Linux, ~3:18 on FreeBSD 15.0 (mostly
dependency compilation). The spirit/C++ path compiles in parallel.

### Rust toolchain pin

`rust-toolchain.toml` pins rustc to 1.85. The reason is in the
file's comment; bump when upstream rustler emits a corrected
`rustler-sys` signature.

## Blog series

- [*Compute It Twice: When CSE Lost the Race*](http://www.dataalienist.com/blog-compute-it-twice.html) — the `Nx.Defn` fusion compiler; why cross-stage CSE never won the fleet race
- [*The Backend That Didn't Need to Know*](http://www.dataalienist.com/blog-backend-didnt-need-to-know.html) — the C++→vulkano migration; descriptor pool debugging; autograd was free
- [*The GPU That Doesn't Need CUDA*](http://www.dataalienist.com/blog-vulkan-on-freebsd.html) — the FreeBSD Vulkan story (spirit-era)
- [*A Walkable Path Under the Mountain*](http://www.dataalienist.com/blog-walkable-path.html) — eXMC + zed integration

## Sibling: zed

[`zed`](https://github.com/borodark/zed) is the declarative ZFS + Elixir deploy tool that
orchestrates BEAM nodes. `nx_vulkan` is consumed *inside* deployed
BEAM nodes — not as a zed dependency. See `specs/nx-vulkan-execution.md`
in the zed repo for the integration story.

## License

Apache 2.0. Same as Spirit and Nx.
