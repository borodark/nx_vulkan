# W1 — Shader Synthesis Substrate

**Question:** What's the right substrate for synthesizing a chain shader from a runtime spec like `{:beta, alpha, beta}`?

**Budget:** synthesize + compile + load + validate a new chain shader in **<1000 ms end-to-end**. Acceptable if the steady-state dispatch performance matches a hand-written shader.

## Three candidates

| Substrate | How it works | Pros | Cons |
|---|---|---|---|
| (a) Parameterized GLSL templates | Existing 6 shaders are already templated by push constants. New families = new GLSL files with text-substitution holes for the family-specific math | Familiar; reuses existing shader skeleton; easiest to validate | Per-family hand work for the gradient. No automation. Limited to families that fit one template skeleton |
| (b) Elixir IR → GLSL transpiler | Walk an `Exmc.IR` distribution definition, emit GLSL with the leapfrog skeleton + family-specific dlogp/dq | One-time code. Any new family that has an IR representation gets a shader for free | Symbolic differentiation in Elixir. GLSL emission. ~2-3 weeks of code. Can break in subtle ways |
| (c) Direct SPIR-V via rspirv | Skip GLSL entirely. Emit SPIR-V instructions directly using rspirv (Rust crate) | No glslc dependency; tighter control; sub-50 ms compile | SPIR-V is verbose; no human-readable artifact for debugging; rspirv has a learning curve |

## Protocol

1. Pick **one** target distribution (Beta(2, 5) — known closed-form posterior, easy to validate).
2. Build a minimal prototype of each substrate that produces a chain shader for that target.
3. Measure:
   - Substrate code line count (host-side codegen logic only)
   - Synthesis time (template render → SPIR-V binary in memory)
   - Compile/load time (SPIR-V → vkPipeline)
   - Validation pass/fail against EXLA reference (1000 draws, KS test, mean/var within 3σ)
4. Pick winner for Phase 1.

## Expected outcome

Likely (a) wins for Phase 1 — it's the lowest-risk path that gets us to "synthesized Beta shader works end-to-end" within the 1-second budget. (b) and (c) are documented as future work.

The harder question (b) addresses — *can a non-shader-author add a distribution at runtime?* — is the long-term goal but probably needs a quarter of focused work, not a sprint.

## Notes / log

### Existing shader anatomy

Sources live at `/home/io/projects/learn_erl/spirit/shaders/leapfrog_chain_*.comp` (the `.spv` artifacts at `/home/io/projects/learn_erl/nx_vulkan/priv/shaders/` are the build output). Six families, line counts:

| Shader | LoC | Push-constant params | Constraint transform |
|---|---|---|---|
| `leapfrog_chain_normal.comp` | 89 | `n, K, eps, mu, sigma` | none (real line) |
| `leapfrog_chain_exponential.comp` | 80 | `n, K, eps, lambda` | log-transform (q_uc = log q) |
| `leapfrog_chain_halfnormal.comp` | 85 | `n, K, eps, sigma, log_const` | log-transform |
| `leapfrog_chain_studentt.comp` | 89 | `n, K, eps, mu, sigma, nu, logp_const` | none |
| `leapfrog_chain_cauchy.comp` | 81 | `n, K, eps, loc, scale, log_pi_scale` | none |
| `leapfrog_chain_weibull.comp` | 80 | `n, K, eps, k, lambda, logp_const` | log-transform |

**Shared skeleton (~65 lines per shader, ~80% identical):**

- `#version 450` + `local_size_x = 256` + 256-element `shared float partial[]`
- Push-constant block (always `n, K, eps`, family-specific scalars after)
- Seven SSBOs at fixed bindings 0-6: `q_init, p_init, inv_mass` (in) + `q_chain, p_chain, grad_chain, logp_chain` (out, K-strided)
- Per-thread state load (`qi`, `pi`, `mi`) with `in_bounds` mask
- For-K loop with strict layout: half-step momentum at q → full-step position → half-step at q_new → write per-dim chains → workgroup reduce of per-element logp contribution → thread-0 write logp + final barrier
- Tree-reduction stencil (`for s = 128; s > 0; s /= 2`)

**Family-specific portion (~3-8 lines):**

- Two scalar expressions: `dlogp/dq(qi)` and the per-element `logp_contrib(qi)`
- An optional handful of precomputed constants at function entry (e.g., `inv_var = 1/(sigma*sigma)`, `grad_coeff = -(nu+1)/(nu*sigma^2)`)
- An optional host-side `logp_const` push-constant that absorbs all q-independent terms (this is how Cauchy, StudentT, HalfNormal, Weibull keep the GPU code tight)

The Weibull file even factors the family-specific math into two GLSL functions (`weibull_grad`, `weibull_logp_contrib`) — i.e. the codebase is *already* using a manual macro pattern. Substrate (a) just makes that pattern programmatic.

A diff between any two shader pairs (Normal vs Exponential, Cauchy vs StudentT) is dominated by the math expressions plus the push-constant block. The control flow, barriers, reduction stencil, buffer bindings, and writes are byte-identical across families.

### Substrate survey (related work)

**TVM** (Apache, C++/Python). Tensor-expression DSL → schedule → code-gen to CUDA/OpenCL/LLVM/Vulkan. Compile times: seconds to minutes for AutoTVM tuning, ~100-500 ms per kernel for templated lowering. User-facing API is Python with `te.compute(...)` + `s = te.create_schedule(...)`. *Heavyweight* — adds a full IR layer, schedule abstractions, target codegen backends. Useful only if we expect dozens of distinct shader shapes.

**Triton** (OpenAI, Python; Rust port `triton-rs` is early). JIT-compiles Python-syntax kernels to PTX via an MLIR pipeline. Compile times: 200-2000 ms first call, then cached. Strength is autotuning block sizes; weakness is CUDA-only (no Vulkan/SPIR-V backend in mainline). Not directly applicable.

**PyTensor codegen** (PyMC's underlying graph compiler, formerly Theano/Aesara). Symbolic graph of `Op`s → Python or C/CUDA source → cc/nvcc → import. Compile times: 1-3 s per fresh graph, cached on disk. *Closest analog to substrate (b)*: PyMC takes a user model, walks the graph, derives gradients via reverse-mode AD over the IR, and emits source. The compile-time pain (PyTensor's notorious "compiling C code..." pause) is exactly what we'd inherit.

**Halide** (MIT/Adobe, C++). Algorithm + schedule split, JIT to LLVM. Compile times: 50-500 ms. Vulkan/SPIR-V target exists but is less mature than CUDA/Metal. Same heavyweight critique as TVM.

**naga** (Mozilla, Rust). Multi-format shader translator (WGSL/GLSL/SPIR-V/MSL/HLSL ↔). It's a *translator*, not a generator: you build a `naga::Module` programmatically (an IR of statements/expressions/types), and naga emits the chosen output format. Compile times: <10 ms for translation. *Useful as a backend* for substrate (b) or (c) — gives us SPIR-V emission without writing rspirv by hand, plus a free GLSL/WGSL pretty-print for debugging. Maintained and shipped in `wgpu`.

**rspirv** (Rust, Khronos-aligned). Direct SPIR-V op-code builder. You emit `OpLoad`, `OpFMul`, `OpStore` etc. by hand. Compile time: microseconds (it's just byte-emission). Strength: zero compile dependency at runtime (no `glslc`). Weakness: *very* low-level — you manage SSA ids, type ids, decorations, capability declarations. Effectively writing assembler.

**wgpu shader builder / cranelift / Rust GPU** (`rust-gpu` from EmbarkStudios). Compile Rust → SPIR-V. Heavy build-time tooling, not designed for runtime synthesis.

Single takeaway: nobody in this genre runs a "synthesize compute kernel in <1 s" loop without paying *some* compile cost. The cheapest paths are (rspirv: 1 ms emit + 0 compile) or (template GLSL: ~1 ms render + 50-200 ms `glslc`). Everything else is ≥500 ms.

### Substrate critiques

#### (a) Parameterized GLSL templates

*Implementation effort to "Beta(α, β) chain shader works end-to-end":* ~2 days. Define a template (`leapfrog_chain.comp.eex` or a `String.replace/3` skeleton) with three holes: push-constant block, `dlogp_expr(qi)`, `logp_contrib_expr(qi)`. Add a Beta entry to a per-family table containing the constraint transform (`logit` for q ∈ (0,1)) and the two GLSL expressions hand-derived once. Render → write to tempfile → `glslc` → load. The 6 existing shaders become reference checks: re-emitting Normal from the template should produce a SPIR-V module that round-trips identical fair-race numbers.

*Risk of silently-wrong gradients:* moderate. Each new family still costs a hand-derivation; a typo in `dlogp/dq` for Beta passes glslc and may even pass smoke tests but break inference. Mitigation lives entirely in W2 (statistical validation harness).

*Debuggability:* excellent. The intermediate artifact is human-readable GLSL text in a tempfile or cache directory. `glslc` errors point to source lines.

*Performance ceiling:* identical to hand-written. Same skeleton, same compiler.

#### (b) Elixir IR → GLSL transpiler

*Implementation effort to Beta end-to-end:* ~3 weeks. Three sub-pieces: (i) a `dlogp/dq` symbolic differentiator over `Exmc.IR` distribution definitions (PyMC has this, we don't yet), (ii) an Elixir → GLSL expression printer (operator precedence, `pow` vs `**`, `log1p` mapping), (iii) the constraint-transform table (log, logit, identity, softplus). The simple cases (Exp, HalfNormal) are easy; Beta needs `log_beta(α, β) = lgamma(α) + lgamma(β) - lgamma(α+β)`, and GLSL has no `lgamma`. We'd need a polynomial approximation in the shader prelude or move the constant to a host-side push-constant (which is what every existing shader already does). The differentiator must handle `lgamma'(x) = digamma(x)`, which propagates a digamma approximation into shader code.

*Risk of silently-wrong gradients:* high. The whole point is automation, but a bug in the symbolic differentiator silently miscompiles every distribution. A single sign error in the digamma series gives plausibly-shaped but biased posteriors.

*Debuggability:* moderate. The emitted GLSL is still readable, but mapping a glslc error back to an IR node requires source-mapping infrastructure we'd have to build.

*Performance ceiling:* identical to hand-written *if* the printer emits the same idioms. A naïve printer (always `pow(x, 2.0)` instead of `x*x`, no constant folding) leaves 5-15% on the table. Fixable with a peephole pass.

#### (c) Direct SPIR-V via rspirv (or naga IR)

*Implementation effort to Beta end-to-end:* ~3-4 weeks. Hand-built SPIR-V module: declare types (`OpTypeFloat 32`, `OpTypeStruct` for push-constant block, `OpTypeRuntimeArray` for SSBOs), entry point, decorations (`Binding`, `DescriptorSet`, `Offset`), workgroup size, function body with explicit SSA. The leapfrog skeleton is ~400 SPIR-V instructions; a builder helper that stamps it out is feasible. Expression emission for Beta's gradient is straightforward op-by-op (`OpFSub`, `OpFDiv`). The barrier sequence and `OpControlBarrier`/`OpMemoryBarrier` semantics are tricky to get right.

*Risk of silently-wrong gradients:* same as (a) or (b) depending on whether we hand-derive or auto-derive — substrate-orthogonal. *Additional risk* unique to (c): a malformed SPIR-V module (wrong storage class, wrong decoration, missing capability) can crash the driver instead of failing validation, or worse, silently corrupt memory. The Vulkan validation layer catches most but not all.

*Debuggability:* poor. SPIR-V disassembly via `spirv-dis` is readable but verbose (~500 lines for a chain shader). Mapping driver errors back to the rspirv builder call site is a manual exercise. naga at least emits readable WGSL/GLSL we can inspect.

*Performance ceiling:* potentially slightly above hand-written GLSL (we control the exact ops, no glslc surprises) but in practice glslc + driver compiler optimizes shader IR aggressively, so this is theoretical. The real win is *latency*: skip glslc's 50-200 ms entirely.

### Recommendation

**Phase 1: substrate (a), parameterized GLSL templates.** The 1-second end-to-end budget makes this the only choice that we already know hits the target on first attempt:

- Template render: ~1 ms (Elixir string interpolation)
- `glslc` compile: 50-200 ms (measured on existing shaders)
- `vkCreateShaderModule` + `vkCreateComputePipelines`: ~50-150 ms cold
- W2 statistical validation gate (10k draws): ~300-500 ms

Total: ~400-850 ms. Comfortably under 1 s, with margin for the validation harness to stretch.

The shader-anatomy survey is decisive: 80% of each existing shader is byte-identical skeleton, 20% is two scalar expressions plus a push-constant block. Substrate (a) automates the 80% and asks the human to write the 20% once per family — exactly where the actual physics knowledge has to live anyway. The auto-differentiation in substrate (b) buys us nothing for Beta/Gamma/Lognormal/von Mises (all have closed-form `dlogp/dq` in any stats reference) and adds a class of silent-correctness bugs that the validation harness would have to be 10x stronger to catch.

Substrate (c) is the right answer if and when we hit the `glslc`-latency wall — but until measurement says we are wall-bound, the human-readable GLSL artifact is worth the 100 ms.

**Phase 2 (defer):** revisit (b) once we have ≥10 families and a regression suite that gives genuine confidence. Revisit (c) only if synthesis latency becomes user-visible (it won't at <200 ms).

**Surprises during the survey:**

1. The existing Weibull shader (`/home/io/projects/learn_erl/spirit/shaders/leapfrog_chain_weibull.comp`, lines 37-45) already factors the family math into helper GLSL functions (`weibull_grad`, `weibull_logp_contrib`). The codebase's own author was reaching for the macro pattern manually — substrate (a) is just formalizing convention.
2. naga (the wgpu translator) is more useful than expected: it gives us an IR-level back-end that could serve *both* substrate (b) and substrate (c) via the same module, with a free GLSL/WGSL pretty-print fallback. Worth a footnote when (b) is reconsidered.
3. The `logp_const` push-constant trick already in use across StudentT/Cauchy/HalfNormal/Weibull (precomputing q-independent terms host-side) is exactly the discipline we'd want to enforce in a templated synthesizer — keeps the GPU code tight and the template shape uniform across families.
