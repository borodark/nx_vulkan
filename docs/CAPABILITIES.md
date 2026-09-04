# What Nx.Vulkan can do

**Scope:** the op surface, the fusion compiler, and why gradients came for free.
Benchmarks live in [`BENCHMARKS.md`](BENCHMARKS.md); how it compares to EXLA and
EMLX is in [`STANDING.md`](STANDING.md).

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
  that way. See [`STRICT_MODE.md`](STRICT_MODE.md).
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

Roadmap and future work: [`ROADMAP.md`](../ROADMAP.md).

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
  device advertising `shader_float64`). f64 transcendentals are excluded from
  FUSION, but they do not reach the host: the evaluator dispatches them to the
  eager f64 shader, which computes them at **f32 precision**. See
  [`LIMITATIONS.md`](../LIMITATIONS.md) §1 — this is a real limit and the sentence
  here previously claimed the opposite.

Whole layers fuse end-to-end: `relu(x @ W + b)`, `relu(conv(x, k) + b)`,
a CNN classifier head (`conv → flatten → dense`), softmax and layernorm
reduction patterns, and transposed-weight layers (`x @ Wᵀ`). Anything
unsupported falls back to `Nx.Defn.Evaluator`, so results are always
correct — worst case is "no fusion, same as eager."

## The autograd insight

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
