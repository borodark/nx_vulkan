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
**Done:** broadcasting elementwise binary (bias-add / relu-via-max / softmax-sub
/ scaling) on the GPU (new `elementwise_binary_bcast_{f32,f64}` +
`apply_binary_broadcast` NIF); `clip` composed from GPU broadcast min/max;
`as_type` f32<->f64 via cast shaders (`cast` NIF). **Result: an entire f32
mlp + softmax forward now stays on the GPU with zero host round-trips**
(`nn_gpu_coverage_test.exs`); the whole f32/f64 numeric surface (matmul, conv,
elementwise, broadcast, reductions, transpose, clip, cast) is on-device.

**Remaining (harder):** comparison ops (u8 output — needs 8-bit storage or a
pack step), `select`/`where` (3-input broadcast; relu-grad / masking), `gather`,
on-device `pad`/`slice`, and the mixed-dtype scalar broadcast (f64 tensor + f32
scalar). Original notes:


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
