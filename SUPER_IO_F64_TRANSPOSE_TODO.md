# super-io brief: f64 fusion + transpose-as-a-boundary

Two thrust-3 increments to implement on the fusion compiler
(`lib/nx_vulkan/compiler.ex` + `lib/nx_vulkan/codegen.ex`). Both extend the
multi-stage split. Read the `vulkan-nx-compute` skill
(`.claude/skills/vulkan-nx-compute/`) first — it documents the shader/NIF/Nx
data flow and the std430 / accumulator / dispatch conventions referenced below.

Branch: `f32-matmul-prototype`. Baseline before this work: `b4608a3`
(863 doctests, 328 tests, 0 failures on the real GT 750M/650M).

**Working rules (same as every thrust-3 increment):**
- Correctness is checked against `Nx.BinaryBackend` for every case, exact to
  dtype eps. Add tests to `test/nx_vulkan/compiler_test.exs`.
- Never add `exla` to the committed `mix.exs`.
- Confirm the REAL GPU is active before trusting anything:
  `Nx.Vulkan.NativeV.device_name()` must not say llvmpipe.
- Fleet-validate any perf heuristic (247 Kepler / 248 Kepler / 249 Ampere) — do
  NOT default-on a win/loss that flips across GPUs; gate via
  `Nx.Vulkan.Device.class/0` or an env flag, as with the many-slot reduce and the
  new `NXV_CSE`.
- Anything not covered must throw `:unschedulable` and fall back whole-graph to
  the Evaluator (still correct). Document GPU-covered vs host-fallback.

---

## Increment A — f64 fusion

Today the JIT codegen and the multi-stage planner are **f32-only**: `Codegen`
hardcodes `float` SSBOs and arithmetic, and the compiler gates on `{:f, 32}`
(`try_multistage(%T{type: {:f, 32}})`, `dot_2d_f32!`, `conv_schedulable!`,
the reduce/region f32 checks). Widen the whole pipeline to also fuse `{:f, 64}`.
The f64 matmul/conv/transpose shaders and the f64 reduce accumulator already
exist — this is mostly parameterising the codegen on element type + fixing
buffer byte-sizes.

### A1. Codegen: parameterise on element type
In `lib/nx_vulkan/codegen.ex`, thread an element type (`{:f, 32} | {:f, 64}`)
into `emit_region/2`, `emit_reduce_region/4`, `emit_elementwise/1`,
`emit_fused_reduce/3` (take it from the root/inner tensor's `.type`). Replace the
hardcoded GLSL `float` with a helper `glsl_scalar(type)` → `"float" | "double"`,
and:
- `input_decls/1` and the `Out` buffer decl: `buffer A { <scalar> a[]; }`.
- For any f64 shader emit `#extension GL_ARB_gpu_shader_fp64 : require` at the top
  (the reduce shader already does; the elementwise `emit_region` does not).
- `helper_functions/0` (erf/expm1 approximations) and the unary op templates:
  GLSL.std.450 has **no f64 transcendentals**. For f64, either boundary-cast
  (`double(exp(float(x)))`) — acceptable, document the precision cost — or, if any
  op in the tree is a transcendental and the tree is f64, have `fusable?`/
  `fusable_op?` reject it so it host-falls-back. Simplest correct first cut:
  reject f64 trees containing exp/log/pow/tanh/sigmoid/erf/… (add an
  `f64_safe_op?` check), fuse only f64 arithmetic/min/max/compare. Widen later.
- The reduce accumulator is already `double`; for an f64 store it's a plain
  `double`, for f32 it stays `float(acc)`.

### A2. Compiler: accept f64 + size buffers by dtype
In `lib/nx_vulkan/compiler.ex`:
- `try_fuse` / `try_multistage`: relax `{:f, 32}` guards to
  `t in [{:f, 32}, {:f, 64}]`. Keep the single-tensor and composite paths.
- **Buffer byte-sizes**: every `NativeV.buf_alloc(n * 4)` in `run_fused`,
  `run_fused_reduce`, `run_plan`, `run_plan_multi`, and `exec_stage({:fused,..})`
  / `{:reduce,..}` is f32-hardcoded. Introduce `ebytes(type)` (4 or 8) and size
  every alloc `n * ebytes`. The `%VulkanoBackend{type: ...}` you stamp on the
  result already carries the dtype — thread it through so allocs match.
- `dot`/`conv` stages: `dot_2d_f32!` and `conv_schedulable!` currently force
  f32. Add f64: pick the matmul/conv SPV by dtype (`matmul_f64.spv`,
  `conv_im2col_f64.spv` + `conv_gemm_f64.spv` — see
  `VulkanoBackend.conv_plan/5`, which already selects f64 shaders for
  `{:f, 64}`; reuse it). The matmul stage needs an f64 variant of `@matmul_spv`
  and `NativeV.matmul` output sized `m*n*8`.
- `emit_reduce_region` / `emit_region` calls: pass the node dtype.

### A3. dispatch NIFs
`dispatch_generated` / `dispatch_generated_reduce` are element-agnostic (they bind
byte buffers and push `n` = element COUNT, not bytes) — no Rust change needed as
long as the shader declares `double` and Elixir allocs `n*8`. Verify: an f64
fused add of two `{N}` tensors returns bit-exact vs BinaryBackend.

### A4. Tests
Mirror the f32 fusion tests at f64: an elementwise chain, a reduce
(`sum`/`mean`), a 2D dot, and a multi-stage `relu(x@W+b)` — all `type: :f64`,
exact vs BinaryBackend. Plus: an f64 tree containing a transcendental
host-falls-back (until A1 widens it). Confirm f32 paths are byte-identical to
before (no regression).

### A5. Gotchas
- The device must advertise `shader_float64` (the NIF enables it only then — see
  `ctx()` in `lib.rs`). On a GPU without it, f64 fusion must fall back. Guard on a
  device capability (add a `Nx.Vulkan.Device.f64?/0` if needed) or catch the
  compile failure and `:unschedulable`.
- f64 matmul is much slower than f32 on the consumer NVIDIA cards (1/32 rate) —
  correctness first; this is not a perf play, it's coverage.

---

## Increment B — transpose as a boundary

Unlike reshape (a zero-copy view), transpose **moves data** (permutes axes), so it
needs a real dispatch: a transpose stage that produces a new buffer. The eager
backend already does 2D transpose on the GPU — reuse it.

### B1. Reuse the existing 2D transpose shader
`VulkanoBackend.transpose/3` handles `tuple_size == 2 and axes == [1, 0]` via
`NativeV.transpose_2d(out_ref, a_ref, m, n, spv)` with
`priv/shaders/transpose_f32.spv` / `transpose_f64.spv` (selected by
`transpose_spv(type)`); higher-rank / other perms host-fall-back. The multi-stage
transpose stage should cover exactly the same envelope (2D, `[1,0]`, f32/f64) and
throw `:unschedulable` otherwise.

### B2. Planner clause + executor
In `lib/nx_vulkan/compiler.ex`:
- The Expr node is `%Expr{op: :transpose, args: [tensor, axes]}` (axes fully
  resolved). Add a `plan_new` clause:
  ```elixir
  defp plan_new(%T{data: %Expr{op: :transpose, args: [inp, axes]}} = node, state) do
    unless tuple_size(inp.shape) == 2 and axes == [1, 0] and
             node.type in [{:f, 32}, {:f, 64}], do: throw(:unschedulable)
    {in_ref, state} = plan_node(inp, state)
    {m, n} = {elem(inp.shape, 0), elem(inp.shape, 1)}
    {sid, state} = new_sid(state)
    spv = transpose_spv(node.type)          # add this helper, mirror VulkanoBackend
    state = add_stage(state, {:transpose, sid, in_ref, m, n, ebytes(node.type), spv})
    {{:stage, sid}, memoize(state, node, {:stage, sid})}
  end
  ```
- Add `:transpose` to `has_boundary?` so a graph containing one enters
  multi-stage.
- Executor:
  ```elixir
  defp exec_stage({:transpose, sid, in_ref, m, n, eb, spv}, values, params) do
    {a, values} = resolve(in_ref, values, params)
    {:ok, out} = NativeV.buf_alloc(m * n * eb)
    :ok = NativeV.transpose_2d(out, a, m, n, spv)
    Map.put(values, {:stage, sid}, out)
  end
  ```
- `transpose_spv/1`: point at the module-attr SPV paths, same as
  `VulkanoBackend.transpose_spv/1` (`Path.expand("../../priv/shaders/transpose_f32.spv", __DIR__)` etc.).

### B3. Why this matters
It lets `x @ W^T` (a `dot` with a transposed weight — the standard dense-layer
form in many models) fuse: transpose stage → matmul stage, instead of falling
back. Also `(A @ B)^T` and transpose feeding an elementwise region.

### B4. Tests
- Bare `Nx.transpose(x)` (2D) → single transpose stage, correct, on GPU.
- `Nx.dot(x, Nx.transpose(w))` → transpose stage + matmul stage.
- `relu(Nx.transpose(x @ w))` → matmul + transpose + fused relu.
- A 3D/`>2`-axis transpose or a non-`[1,0]` perm → `:unschedulable`, whole-graph
  fallback, still correct.
- Both f32 and (once Increment A lands) f64.

### B5. Interaction with reshape
transpose is NOT a view — do not alias its buffer like reshape/squeeze. A
`reshape(transpose(x))` chain is fine: transpose materialises a contiguous
buffer, and the reshape aliases THAT (reshape over a contiguous transpose output
is a valid view).

---

## Order & validation
Do Increment B first (small, self-contained, reuses a shipped shader), then
Increment A (broader). Commit each separately with a `thrust 3:` message and the
suite green on the real GPU. Then race a representative graph (e.g. a dense layer
`relu(x @ Wᵀ + b)` at f32, and an f64 matmul) across the fleet and record numbers
in `bench_results/` — same as prior increments. Ping back with the commits + the
suite summary + any fleet perf notes.
