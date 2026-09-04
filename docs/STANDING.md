# Standing — what this backend is, and is not

**Scope:** an honest position statement. Where this backend wins, where it
loses, and the discipline behind every claim in it. Numbers are in
[`BENCHMARKS.md`](BENCHMARKS.md).

## The position

The fusion compiler is the goal this effort set out to reach: a credible
**#2 compute backend** for `elixir-nx`, with EXLA's whole-graph
compilation now present in the one place a Vulkan backend can offer it —
on any GPU with a driver, CUDA or not.

- **Correctness first.** Every fused result is checked exact against
  `Nx.BinaryBackend`. The suite — **833 doctests, 903 tests, 0 failures**
  — is green on two 2012-era Keplers (GT 650M / GT 750M, FreeBSD), a 2021
  Ampere (RTX 3060 Ti, Linux) and a Tegra X1 Jetson Nano (unified memory,
  Ubuntu), with the f64 fused path active on all four. Gradient
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
