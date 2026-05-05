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

(empty — populate as evidence lands)
