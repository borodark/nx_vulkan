# v0.2.0 — The fusion compiler

The first release since 0.1.0, and the one the project was built toward: a
**whole-graph fusion compiler** for Nx on Vulkan — EXLA's structural edge,
now on any GPU with a driver, CUDA or not.

## ✨ Highlights

- **`Nx.Vulkan.Compiler` — an `Nx.Defn.Compiler`.** Traces a `defn` to a stage
  schedule that runs on-device with GPU-resident intermediates and no
  interpreter fallback. Elementwise chains fuse to one generated shader; an
  elementwise chain feeding a reduction fuses to one parallel tree-reduce; graphs
  with `dot`/`conv`/`reduce`/`transpose` boundaries split into on-device stages.

  ```elixir
  Nx.Defn.jit(&my_fun/2, compiler: Nx.Vulkan.Compiler).(a, b)
  ```

- **Native f32 *and* f64.** The hot ops (elementwise, matmul, conv, reduce,
  transpose) dtype-dispatch native f32 shaders alongside f64 — f32 is no longer
  merely cast. f64 stays the default accumulator policy (correctness first);
  f32 wins on bandwidth-bound work.

- **New native ops:** `conv` (im2col + GEMM), `fft`/`ifft`, and `transpose` — in
  both f32 and f64.

## What fuses, end-to-end on the GPU

| graph | compiles to |
|---|---|
| `relu(x @ W + b)` | matmul stage + fused `max(dot + b, 0)` |
| `relu(conv(x, k) + b)` | conv stage + fused epilogue |
| `conv → flatten → dense` | CNN classifier head, one schedule |
| `x - mean(x)`, softmax, layernorm | reduce boundaries + fused regions |
| `x @ Wᵀ` | transpose stage + matmul stage |
| `{mean, variance}` | one shared schedule, computed once |

`reshape`/`squeeze` are zero-copy view boundaries (no dispatch). Anything
unsupported falls back to `Nx.Defn.Evaluator`, so results are always correct —
worst case is "no fusion, same as eager."

## 🔬 Correctness & the fleet

Every fused result is checked **exact against `Nx.BinaryBackend`**. The suite —
**863 doctests, 361 tests, 0 failures** — is green on three GPUs across two OSes:
a 2012 GT 650M (Kepler, FreeBSD), a GT 750M (FreeBSD), and a 2021 RTX 3060 Ti
(Ampere, Linux), with the f64 fused path active on all of them.

Every perf heuristic is **fleet-validated, never assumed** — win/loss crossovers
are hardware-specific. The many-slot reduce is device-class-gated (wins on weak
GPUs, regresses on strong). Cross-stage CSE was built, raced, and found to **never
win on either device class** — on a GPU, recompute is cheaper than the dispatch it
takes to avoid — so it ships **default-off** (`NXV_CSE=1` to opt in). The story:
[*Compute It Twice: When CSE Lost the Race*](https://www.dataalienist.com/blog-compute-it-twice.html).

## ⚙️ Notes

- **`nx ~> 0.13`** (was `~> 0.10 or ~> 0.11 or ~> 0.12`).
- The C++ **spirit** Elixir backend (`Nx.Vulkan.Backend`) and the old
  `Nx.Vulkan.Fuse` macro were **removed** — `Nx.Vulkan.VulkanoBackend` is the
  only backend, and the fusion compiler supersedes the prototype's constraints.
- `Nx.Defn.grad` autograd still works for free (graph transformation, not a
  backward-op backend).
- Runs anywhere Vulkan does — Linux/NVIDIA, macOS via MoltenVK, Windows, and the
  **only** GPU path for FreeBSD + NVIDIA.

Full details in [`CHANGELOG.md`](CHANGELOG.md).
