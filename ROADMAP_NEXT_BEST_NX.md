# Roadmap — VulkanoBackend as the next-best Nx compute backend

**Goal:** make `Nx.Vulkan.VulkanoBackend` the credible #2 compute backend for
`elixir-nx/nx` after EXLA. Written 2026-07-30 (branch `f32-matmul-prototype`).

## The positioning — portability is the moat

EXLA is fastest but needs XLA (CUDA / ROCm / TPU) and a supported Linux/Mac. It
does not run on FreeBSD, on AMD/Intel GPUs without ROCm, on older cards, or on
Apple via a simple path. **VulkanoBackend runs anywhere Vulkan does** —
NVIDIA (Kepler→Ampere verified), and by design AMD/Intel/Apple(MoltenVK) across
Linux/FreeBSD/Windows. So the pitch is: *full Nx parity + GPU acceleration for
the hot kernels, on the hardware/OS where EXLA can't go, at competitive-enough
speed.* "Next best" = **best where the best isn't available.**

## Where we are (done)

- Full `Nx.Backend` parity (all 115 callbacks; verified vs BinaryBackend).
- Native f64 **and** f32 GPU shaders for the hot kernels: matmul, conv
  (im2col+GEMM), fft/ifft, axis reductions, elementwise (unary/binary), 2-D
  transpose — all 16×16 tiled, dtype-dispatched.
- f32 accumulator policy → 1.8–3.0× on the compute-bound fast path.
- 3-GPU fleet CI-by-hand (GT 650M / GT 750M / RTX 3060 Ti) + labelled benchmark
  reports.

## How Nx validates a backend (researched 2026-07-30)

Not a separate certification — you run **Nx's own test surface with your backend
set as default**. The canonical pattern (Torchx/EXLA/EMLX):

- `test_helper.exs`: `Nx.default_backend(YourBackend)` + `ExUnit.start(exclude:
  …)` with device-conditional excludes.
- **`doctest Nx` + `doctest Nx.LinAlg`** — Nx's own documented examples are the
  conformance suite, run with an `:except` list bucketed into: float rounding /
  `inspect` diffs, inherently-unsupported ops (`population_count`,
  function-based `map`/`reduce`/`window_reduce`), and irrelevant
  (`default_backend`).
- Mirrored hand-written suites: `nx_test`, `nx_linalg_test`, `defn_test`,
  `nx_block_test`, `complex_test`, `random_test`, `device_test`.
- Assertions via `Nx.Testing.assert_all_close` / `assert_equal`.
- `backend_documentation_test` for the backend-doc convention.

So "validated backend" = **passes Nx's doctest suite as default backend** (minus
a documented `:except`), plus the mirrored suites. Our current tests are
hand-rolled parity-vs-BinaryBackend — we have never run Nx's actual conformance
suite. That is thrust 0.

## The thrusts (prioritised)

### 0. Run Nx's conformance suite (validation foundation) — LANDED
`test/nx_vulkan/nx_doctest_test.exs` runs `doctest Nx` with VulkanoBackend as
default. **839 / 954 pass**; 115 excepted, bucketed: `@rounding` (native-shader
last-ULP inspect diffs), `@unsupported` (complex, f8/f16), `@backlog` (real bugs,
below). It immediately found + we fixed two real bugs hand-rolled tests missed:
slice with dynamic tensor indices, and composed fallbacks leaking the default
backend (`with_binary_backend/1`). Verified across the fleet (247/248/249).

**Remaining thrust-0 backlog (real bugs, tracked):**
- `encode_scalar/2` missing dtype clauses (f16 etc.) → breaks `reflect`,
  `concatenate` under those dtypes. Also `{:bf,16}` currently encodes as IEEE f16
  (wrong format) — latent.
- f8/f16 tensors inspect as `<unreadable>` (to_binary/inspect dtype gap).
- `deserialize` round-trip of unsupported dtypes; residual `slice` /
  `window_scatter_*` edge cases.
- Still TODO: `doctest Nx.LinAlg` + mirror torchx's `nx_test`/`nx_linalg_test`/
  `nx_block_test`/`defn_test`.

### 1. Measure the gap to EXLA — harness ready, EXLA blocked on 249
`examples/backend_baseline.exs` races BinaryBackend / VulkanoBackend / EXLA
(EXLA optional; picked up when the project depends on it) on matmul/conv/tanh/
sum/mlp-fwd, with correctness checked vs BinaryBackend. **Interim (GT 650M vs
pure-Elixir Binary):** matmul 428×, conv 82×, mlp 106×, tanh 2.4× (exact); sum
0.93× (dispatch-bound). **EXLA three-way blocked:** on super-io the built
`libexla.so` (xla-0.10.0 / exla-0.13.0, CUDA 12) fails to `dlopen` at runtime
("EXLA.NIF is not available") — a library-path/CUDA env fix on that box (cf. the
`_nx-exla-fix` checkout). Once EXLA loads, run the harness in an exla-enabled
project for the real head-to-head. **Do NOT add exla to nx_vulkan's committed
mix.exs** — it would break the CUDA-less Kepler boxes' `mix compile`.

### 1b. (original) Measure the gap to EXLA
Stand up EXLA-CUDA on super-io (249, Linux + RTX 3060 Ti) and benchmark
VulkanoBackend vs EXLA vs BinaryBackend on representative Nx + DL workloads
(matmul/conv sweeps, a small MLP/CNN forward+grad, a softmax/layernorm chain).
Establishes *how far from best* and prioritises everything below. Demonstrable,
leverages the fleet + race infra. **Lead candidate.**

### 2. Kill the host-fallback round-trips — in progress
**Done:**
- broadcasting elementwise binary (bias-add / relu-via-max / softmax-sub /
  scaling) — `elementwise_binary_bcast_{f32,f64}` + `apply_binary_broadcast`.
- `clip` composed from GPU broadcast min/max.
- `as_type` f32<->f64 via cast shaders (`cast` NIF).
- `select` (masking / where / relu-grad value) — `select_{f32,f64}` +
  `apply_select`, 3-way broadcast, u8 `pred` read as u32 (enabled
  `robust_buffer_access`).
- comparison ops `equal/ne/lt/le/gt/ge` -> u8 — `compare_{f32,f64}` +
  `apply_compare`, results packed into u32 words (no 8-bit-storage needed).

**Result:** a full f32 mlp + softmax **forward** and the relu-grad **mask chain**
(`x > 0` -> `select`) now run entirely on the GPU (`nn_gpu_coverage_test`,
`compare_test`). The f32/f64 numeric surface — matmul, conv, elementwise,
broadcast, reductions, transpose, clip, cast, select, compare — is on-device.

**Remaining (harder):** `gather`/`take` (indexing/embeddings), on-device
`pad`/`slice`, `argmax`/`argmin`, and the mixed-dtype scalar broadcast (f64
tensor + f32 scalar literal — currently host-falls-back on the type mismatch).
Original notes:


Profile the DL examples; the ops that bounce to host (broadcast binary, gather/
scatter, pad, slice, sort, `where`/select) each cost a GPU↔host copy and dominate
end-to-end time. Wire GPU dispatch (or at least keep-on-device) for the top few —
broadcast elementwise (the unwired `elementwise_binary_broadcast` shader), a
native gather, on-device slice/pad. Directly closes the EXLA gap on real graphs.

### 3. `Nx.Defn` compiler with fusion (the marquee)
EXLA's moat. Build a real `Nx.Defn.Compiler` that walks the defn IR, fuses
elementwise chains into a single dispatch (revive the dropped Fuse work
properly), and avoids materialising intermediates. Multi-week; the single biggest
perf lever for graphs. Depends on #2's on-device data path.

**Increment 1 — DONE** (`Nx.Vulkan.Compiler` + `Nx.Vulkan.Codegen`,
`dispatch_generated` NIF). Traces the defn, and for a same-shape f32 elementwise
chain (single output) JIT-generates one GLSL shader for the whole chain,
compiles it once (cached by hash in `priv/shader_cache/`), dispatches once.
Everything else falls through to `Nx.Defn.Evaluator` (always correct).
Measured **3.62x** over eager per-op on a 10-op chain (GT 650M, n=1e6). Use:
`Nx.Defn.jit(&fun/2, compiler: Nx.Vulkan.Compiler)`. `NXV_FUSE_DEBUG=1` logs the
path per defn.

**Increment 2 — parallel fused elementwise→reduce: DONE.** `Codegen.emit_fused_reduce`
emits a **workgroup-per-slot shared-memory tree reduce** (256 threads cooperate
per output slot, f64 accumulator for sum), `dispatch_generated_reduce` launches
one workgroup per slot. Enabled by default for the winning regime — a full
reduction or contiguous last-axis reduce (`inner_stride == 1`) with few output
slots (`reduce_beneficial?/3`). It beats even eager, whose own `reduce_axis` is
one-thread-per-slot serial: **full `sum` 256² 9.9x, 1024² 27x, fused chain+reduce
8.5x** over eager (GT 650M), exact to BinaryBackend. That takes Vulkano's `sum`
256² from ~5.6x behind EXLA to ~1.4x. Many-slot / non-contiguous / short-axis
reductions fall back to eager (already parallel) — no regressions. The first
serial attempt regressed 0.3–0.6x everywhere; the parallel version is the fix.

**Increment 2b — many-slot fused reduce (grid-stride): DONE, but opt-in only.**
The workgroup-per-slot tree reduce now grid-strides over output slots (one launch
handles any slot count, past the 65535 workgroup limit). It wins ~4.4x on the
weak Kepler eager path — but FLEET VALIDATION on the RTX 3060 Ti showed it
**regresses ~0.44x on Ampere**: a strong GPU's one-thread-per-slot eager reduce
is already well-fed by thousands of slots, so the fused kernel only adds
overhead. Hardware-dependent → NOT a default. `reduce_beneficial?/3` keeps only
the FEW-slot regime (full/small-output reductions) on by default — that wins on
both Kepler (8-27x) and Ampere (2.8-6.7x). The many-slot path is available via
`NXV_FUSE_REDUCE=1` for weak-GPU deployments. Lesson: validate perf heuristics
across the fleet, not just the local box — the win/loss crossover is HW-specific.

**Increment 2c — device-class auto-enable: DONE.** `Nx.Vulkan.Device.class/0`
labels the active GPU `:weak | :strong` (heuristic over the Vulkan device
name+type: software/integrated/virtual and the older low-end discrete NVIDIA
GeForce GT line are `:weak`; GTX/RTX and unknown discrete are `:strong`, cached
in `persistent_term`; `NXV_GPU_CLASS=weak|strong` overrides). The compiler now
auto-enables the many-slot fused reduce ONLY on `:weak` GPUs, where it wins —
verified: the GT 650M classifies `:weak` and fuses `{2048,256}` by default, a
strong GPU falls back. Best of both: Kepler gets the 4.4x, Ampere avoids the
0.44x regression, no env flag needed.

**Increment 2d — CSE + `mean`/`product` fusion: DONE.** (i) `Codegen.emit_dag/2`
linearises the elementwise DAG into SSA temporaries (post-order, id-deduped topo
sort), so a fan-out node is computed once instead of re-inlined — naive inlining
was exponential (8 chained squarings → 255 multiplies; 12 → 4095, enough to choke
glslangValidator; now 8 / 12 temps). Applies to both the elementwise and
fused-reduce bodies. (ii) `product` is fused as a reduce (f64 mul accumulator);
`mean` (which lowers to `divide(sum(...), n)`) is fused as a `sum` with a `/n`
post-scale baked into the shader — plain `divide` still routes through the
elementwise path. All correct vs BinaryBackend.

**Increment 2e — broadcasting in the fused kernel: DONE.** Nx.Defn carries
mismatched-shape operands directly (no `:broadcast` node), and in a valid
elementwise tree every node broadcasts to the root shape — so only the PARAMETER
loads need to be broadcast-aware. `Codegen.emit_loads` loads each input at its
NumPy-broadcast source index computed from `i` (or the reduce index `idx`) with
the compile-time-constant shapes baked into the GLSL; full-shape inputs still
load linearly. `fusable?` relaxed via `broadcasts_to?/2`. Covers scalar-tensor
scale, row `{n}` / column `{m,1}` vectors, and n-d broadcast, in both the
elementwise and reduce paths. Common NN pattern `relu(x*scale{n} + bias{n})` now
fuses to ONE dispatch: 1.47x over eager on the GT 650M, exact.

**Increment 2f — multi-stage split (dot boundaries): DONE.** A graph containing
a `dot` isn't one fusable region, so it fell back to the Evaluator. The compiler
now splits it into a stage schedule (`try_multistage` + `plan_node` + `run_plan`):
each 2D-f32 `dot` is a matmul stage, and each maximal elementwise region is ONE
generated shader (`Codegen.emit_region`) whose leaf inputs may be earlier stages'
GPU buffers — intermediates stay on-device. Whole NN layers fuse: `relu(x@W+b)`
→ matmul + one fused `max(dot+b,0)` stage; a 2-layer MLP → 4 stages. All correct
vs BinaryBackend. Speedup over the (already-on-GPU) eager path is modest on
matmul-dominated graphs (~1.1x — the matmul is the bottleneck; the win is the
saved elementwise dispatches/intermediates + no Evaluator fallback); it grows
with heavier elementwise epilogues. Codegen was generalised so region leaves can
be `{:param, pidx}` or `{:stage, node_id}` buffers.

Next increments — **all DONE (2026-08, merged to `main`)**: (a) conv as a
boundary op ✓; (b) reduce as a boundary (`mean(x)` materialised so `x - mean(x)`
layernorm / softmax patterns fuse) ✓; (c) tuple/multi-output ✓; (d) f64 fusion ✓;
plus reshape/squeeze view boundaries and transpose as a data-movement boundary.
Cross-stage CSE was built and raced — it never wins on either device class
(recompute beats the dispatch it saves), so it ships default-off (`NXV_CSE=1`).
See [`bench_results/CSE_SOFTMAX_RACE.md`](bench_results/CSE_SOFTMAX_RACE.md).

### 4. Package, document, position
Hex release, README with the portability pitch + a support matrix (OS × GPU
vendor × verified), install docs, the fleet benchmark numbers, and a
"why VulkanoBackend" page. Adoption is a real deliverable, not an afterthought.

## Execution notes

- Fleet over SSH (key auth, user `io`): 247 (GT 650M, `doas kldload nvidia`),
  248 (GT 750M), 249/super-io (RTX 3060 Ti, Linux). On 249 the working checkout's
  local-server remote is **`o`** (`git@localhost:...`), not `origin` (GitHub) —
  pull/push there with `o`. See memory `gpu-fleet-and-f32`.
- Everything stays verified vs BinaryBackend; correctness is non-negotiable for a
  backend people trust.
