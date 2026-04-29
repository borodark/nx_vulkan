# Kernel fusion for Nx.Vulkan — research forward

**Status:** research note, April 2026.
**Predecessor:** [EXMC_PORT_PLAN.md](EXMC_PORT_PLAN.md) flagged "no
kernel fusion" as the structural perf gap vs EXLA. This doc says
how to close it without making fusion the whole effort.
**Why it matters:** exmc's NUTS leapfrog runs ~30 elementwise +
reduction ops per integration step. EXLA fuses these into 1–2
CUDA kernels via XLA HLO; Nx.Vulkan today dispatches 30 times.
Per-dispatch overhead is ~70 µs on the 3060 Ti, so 30 dispatches
= ~2 ms host-side. EXLA's fused kernel runs in ~50 µs. **40× wall
clock difference on the leapfrog hot path** is the gap fusion
closes.

---

## The architectural insight

Nx.Defn already gives us **an IR**. The compile-time IR is a list
of operators with their shapes and dependencies — exactly the
shape XLA's HLO consumes. EXLA reads that IR and produces XLA
HLO. We can read the same IR and produce fused GLSL.

Fusion is therefore a **compile-time concern**, not a runtime
one. We don't have to add it to the dispatch path. We add it to
the `Nx.Defn.Compiler` protocol (the same protocol EXLA
implements). When the user writes `defn`, our compiler walks the
IR and emits a smaller number of shader dispatches than the
naïve op-by-op interpretation would.

This means fusion has a clean architectural seam — it's a
separate compiler module that can land independently of the rest
of the backend, and progressive complexity (none → patterns →
full JIT) is the right way to ship it.

---

## Three known paths, ranked by effort

### Path A — pattern catalog (~1 person-month)

Hand-write fused shaders for the **10 most common patterns**
exmc produces. Match those patterns at IR-walk time; for matched
spans, dispatch the fused shader; for unmatched spans, fall back
to op-by-op.

**Patterns that matter** (from inspection of exmc's defn
functions and from XLA's known fusion classes):

  1. **n-way elementwise chain** — e.g. `c = exp(a + b * 2.0)`.
     Generic shader template parametrised by op tree depth and op
     ids. One fused dispatch instead of three.
  2. **map-then-sum** — `sum(f(x))` where `f` is elementwise. Two
     dispatches → one (workgroup-local accumulation in registers).
  3. **map-then-mean** — same as above + final divide on host.
  4. **two-input elementwise then unary** — `relu(a + b)`,
     `sigmoid(matmul + bias)`. Common in MLP/CNN forward.
  5. **broadcast-then-elementwise** — `(a - mean) / std` over
     a batch axis. Saves one materialisation.
  6. **scale-add (axpy)** — `out = a + scale * b`. Saves one alloc.
  7. **logsumexp** — `log(sum(exp(x - max(x))))`. Three reductions
     + map → fused into two dispatches.
  8. **softmax** — `exp / sum(exp)`. Same as logsumexp's tail.
  9. **layer-norm** — mean + variance + rescale. Five-op chain
     fused into two dispatches.
  10. **matmul + bias + activation** — `relu(A · B + b)`. Common
      in inference.

These ten patterns cover the vast majority of what a probabilistic
sampler or an MLP forward pass actually does. EXLA's auto-clustering
in its first version did roughly this; later versions added the
general JIT.

**Implementation shape**:

```elixir
defmodule Nx.Vulkan.DefnCompiler do
  @behaviour Nx.Defn.Compiler

  def __compile__(_key, vars, fun, opts) do
    expr = fun.(Nx.Defn.Composite.traverse(vars, &Expr.parameter/1))
    # expr is the Nx.Defn IR — a tree of Expr.t() nodes
    fused_plan = Nx.Vulkan.Fuse.plan(expr)
    # fused_plan is a list of either:
    #   {:fused, shader_id, [input_refs]} or
    #   {:single, op, [args]}
    fn args -> Nx.Vulkan.Runtime.run(fused_plan, args) end
  end
end
```

`Nx.Vulkan.Fuse` is a pattern matcher over the IR; for each
recognized pattern it picks a pre-compiled shader. For unrecognized
spans it falls back to op-by-op.

**Effort breakdown** (~4 weeks single dev):
  - Week 1: implement `Nx.Vulkan.Fuse` walker + simple n-way
    elementwise pattern. Prove the design on `c = exp(a + b)`.
  - Week 2: add map-then-sum, map-then-mean, scale-add, matmul+bias.
  - Week 3: write the parametric GLSL generator for n-way
    elementwise chains. This is the only "novel" shader; the rest
    are hand-written one-offs.
  - Week 4: integrate with `Exmc.JIT`, run exmc benchmarks,
    measure the gap.

**Expected outcome**: closes 60–80% of the EXLA fusion gap on
exmc's hot path. Leapfrog dispatches drop from ~30 to ~5–8.
Wall-clock improvement on the 3060 Ti: ~3–5× over op-by-op.

### Path B — JIT GLSL generator (~3 person-months)

Walk the full Nx.Defn IR, emit GLSL source per fused fragment,
hand to a runtime GLSL→SPIR-V compiler, cache the resulting
`VkShaderModule`.

**Toolchain options**, ranked by tradeoff:

  1. **shaderc** (Google's library) — links into the C++ shim,
     wraps glslang, returns SPIR-V binary. Most mature, biggest
     dep (~5MB). What MoltenVK and many engines use.
  2. **rspirv** (Rust crate) — programmatic SPIR-V construction.
     We'd emit SPIR-V directly without GLSL as an intermediate.
     Lower-level, less proven for compute kernels.
  3. **rust-gpu** — Rust source → SPIR-V via custom rustc backend.
     Most ergonomic if Nx.Defn IR can be transpiled to Rust;
     least mature.
  4. **Naga** (Mozilla / wgpu) — translator between SPIR-V, WGSL,
     GLSL. Could be the codegen backend.
  5. **glslang-as-subprocess** — invoke `glslangValidator` per
     compile. Slowest (process spawn) but trivially correct;
     useful as the v0.2 starter.

**My lean for Path B**: shaderc, linked into the C++ shim. The
GLSL we'd emit is the same shape as our hand-written shaders;
we'd write a small templating engine in Rust that walks the IR
and produces GLSL strings.

**Why it's three months**: the IR walker has to handle every
shape Nx.Defn produces (broadcasts, transposes, slices,
reductions over multiple axes, etc.); the GLSL generator has to
emit correct workgroup sizes and bindings; shader caching has
to use a robust hash key (op tree shape + types, not strings).
Most of the time is making sure the JIT produces *correct* code,
not fast code.

**Expected outcome**: closes the remaining 20–40% gap that Path A
left. Leapfrog dispatches drop to 1–2 per integration step,
matching EXLA. Wall-clock improvement over Path A: another
~2–3×.

### Path C — operator-fusion compiler (~6 person-months, the
"real" version)

Path B + dataflow optimisation passes. Common subexpression
elimination, dead code elim, loop reordering, schedule choice
between fusion-as-one-kernel vs kernels-with-shared-buffers.

This is what XLA's full optimiser does. It's the path that gets
to production-grade ML compiler quality. We won't be on this in
the next year, and we shouldn't be — paths A + B cover the
trader's needs at >70% of EXLA performance, which is the bar to
beat.

---

## Prior art worth reading

  1. **Kompute** (KomputeProject/kompute) — high-level Vulkan
     compute wrapper. No fusion, but the dispatch primitives are
     similar to ours; useful for command-buffer patterns.
  2. **GGML** (ggerganov/ggml) — Vulkan backend. Hand-written
     fused shaders for transformer hot paths
     (rms_norm + matmul + activation). Path A precedent.
  3. **MLX** (Apple) — has a similar IR walker that emits Metal
     Shading Language. Architecture lessons even if the toolchain
     differs.
  4. **wgpu / WebGPU** — has Naga as its translator. Good
     reference for Rust-side shader codegen.
  5. **Triton** (OpenAI) — Python DSL → CUDA via LLVM IR.
     Different scope (writing kernels by hand, not fusing graph
     ops) but the JIT compilation pattern is reusable.
  6. **TVM / Relay** — full graph compiler with Vulkan backend
     since 2019. The most thorough academic precedent.
  7. **XLA's auto-clustering** (the first version of XLA that
     fused only specific patterns) — Path A precedent.
  8. **The Nx.Defn.Compiler protocol** itself — the contract is
     small (4 callbacks); EXLA's ~2000-line implementation shows
     what a serious implementation looks like.

---

## What the port plan actually changes

[EXMC_PORT_PLAN.md](EXMC_PORT_PLAN.md) sets up Phase 4 as
"benchmark + acceptance." That phase becomes the trigger for
choosing A vs B.

**Updated phasing**:

| Stage | Backed by | When |
|---|---|---|
| Port forward path (no fusion) | Phase 1 of port plan | weeks 1–4 |
| Run exmc tests | Phase 3 of port plan | week 5 |
| **Measure the fusion gap** on exmc benchmark | Phase 4 of port plan | week 6 |
| **If gap ≥ 5×: start Path A pattern catalog** | new phase | weeks 7–10 |
| Re-measure | | week 11 |
| **If gap ≥ 2× after Path A: start Path B JIT** | new phase | weeks 12–24 |

The decision points are gated on measured numbers, not
speculation. Path A is one person-month and closes most of the
gap; Path B is the long pole only if we need it.

**Critically**: nothing about the port plan blocks on fusion. The
port lands the forward path, the trader runs (slower than EXLA),
the benchmark reveals how slow, the fusion work then closes
exactly the gap that was measured. Fusion isn't the whole effort
because it doesn't have to start until the rest is in.

---

## What this DOES NOT cover

- **Compile-time vs runtime fusion.** Some fusions only become
  apparent with knowledge of input shapes; others can be
  determined statically. Path A is purely static; Path B can
  optionally specialize on shape. Path C is where shape-driven
  re-optimisation matters.
- **Cross-defn fusion.** Each `defn` function is a separate IR;
  we don't fuse across function boundaries. EXLA does in some
  cases. Out of scope until v0.3.
- **Mixed-precision fusion.** f32 forward + f64 accumulator
  fusion is a subtle topic. Defer to after Path B.
- **Memory hierarchy modeling.** Choosing when to fuse vs when
  to leave intermediate buffers in shared memory is the v1.0
  problem.

---

## What we get if Path A lands

Returning to the EXMC_PORT_PLAN.md acceptance criterion
("Nx.Vulkan within 2× of EXLA-CUDA on the leapfrog hot path"):

  - **Without fusion**: 5–10× slower than EXLA on small models,
    1.5–2× slower on large models (where compute dominates
    dispatch overhead). Acceptance criterion fails on small
    models.
  - **With Path A (10-pattern catalog)**: 1.5–2.5× slower than
    EXLA on small, ~1.2× on large. Acceptance criterion likely
    passes.
  - **With Path B (full JIT)**: parity-ish with EXLA. Bonus, not
    required.

**The honest answer to "is fusion the whole effort?": no, it's
an optional 1–2 month iteration that lands after the port,
gated on measured numbers, with a clear yes/no decision point.**

---

## Cross-references

- [EXMC_PORT_PLAN.md](EXMC_PORT_PLAN.md) — the migration plan
  this complements
- [PLAN.md](PLAN.md) — Nx.Vulkan v0.1 roadmap
- Nx.Defn.Compiler docs:
  <https://hexdocs.pm/nx/Nx.Defn.Compiler.html>
- EXLA source (especially `lib/exla/defn.ex`):
  <https://github.com/elixir-nx/nx/tree/main/exla>
- ggml Vulkan backend (Path A precedent):
  <https://github.com/ggerganov/ggml/tree/master/src/ggml-vulkan.cpp>
- shaderc (Path B leading toolchain):
  <https://github.com/google/shaderc>
