# mac-248 — Codegen branch end-to-end gate (post-merge of chain shaders)

All prior tasks complete and merged to main:

- ✅ X3 sign fix (5 leapfrog shaders)
- ✅ Y queue (Student-t, Cauchy, HalfNormal Phase 2 chains)
- ✅ Z1 Philox audit + Z2 canonical Random123 replacement
- ✅ X1 f64 chain + X2 reduce_full_f64
- ✅ FreeBSD bring-up validated on mac-248
- ✅ Hex-prep merged onto main (`6a4540d` after the 718d80a repair)
- ✅ Stage 1.5.4 variance bias resolved (`var = 1.03` vs ref `1.01`)

**Current TODO is the gate for `feat/vulkan-codegen`** —
your fused-reduce codegen work at `9a9e3ad` is great on its own
tests (154/0) but doesn't yet cover the cases the eXMC NUTS
sampler exercises. End-to-end benchmark on Linux side surfaced
three concrete gaps. Closing them and re-running the same
benchmark is the merge gate.

---

## Codegen gate — three concrete asks

The codegen branch passes its own unit tests but breaks when an
eXMC NUTS sampler runs under `EXMC_COMPILER=vulkan`. Tested on
Linux dev box (RTX 3060 Ti) on 2026-05-04 against
`feat/vulkan-codegen` at `9a9e3ad`. Trivial model `x ~ N(0, 1)`,
50 warmup + 200 samples, seed 42. Both the unfused-codegen path
AND the chain-shader-meta-set path die with the same error in
`Nx.Vulkan.Codegen.analyze/2`.

### G1 — Tuple-output handling in `Codegen.analyze/2`

**Symptom**:

```
** (FunctionClauseError) no function clause matching in
   Nx.Vulkan.Codegen.analyze/2

The following arguments were given:
  # 1
  {#Nx.Tensor<f32 …>,    # logp expression
   #Nx.Tensor<f32[1] …>}  # grad expression
  # 2
  [{0, #Reference<…>, {1}, {:f, 32}}, …]
```

**Root cause**: `Nx.Defn.value_and_grad/2` returns a 2-tuple of
expressions `{loss_expr, grad_expr}`. Currently `analyze/2` only
has a clause for a single `Nx.Tensor` (with embedded
`Nx.Defn.Expr`). It needs clauses for tuples and lists of
expressions, recursing into each element and emitting one shader
per output.

ADVI's ELBO is also tuple-shaped. Generalising now avoids a
second round when ADVI hits this.

**Acceptance**: this snippet stops raising:

```elixir
Application.put_env(:exmc, :compiler, :vulkan)
alias Exmc.{Builder, Dist.Normal, NUTS.Sampler}
ir = Builder.new_ir() |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
{trace, _} = Sampler.sample(ir, %{}, num_warmup: 50, num_samples: 200, seed: 42)
```

### G2 — Graceful fall-through on unhandled `Expr` shapes

**Symptom**: same error path as G1. Even after G1 lands, the
inner expressions contain ops `Codegen.analyze/2` doesn't enumerate
yet — visible in the failing trace:

```
b = slice a, [0], [1], [1]      f32[1]
c = reshape b                   f32
d = multiply c, c               f32
e = pad e, 0.0, [{0, 0, 0}]     f32[1]
g = put_slice 0.0, [0], f       f32[1]
```

So `slice`, `reshape`, `pad`, `put_slice` are all hit by NUTS
sampling. `reshape` is trivially supported (no-op for codegen).
`slice` and `put_slice` need shader-side index math; either
implement or fall back. `pad` likely needs a fall-back unless
zero-pad is special-cased.

**The principle**: when codegen can't handle an Expr shape, it
should fall through to the existing IR walker (which dispatches
each Nx primitive separately) — slower, but **correct**. Today,
unhandled shapes raise, which breaks the entire defn JIT.

**Acceptance**: any Expr that the IR walker handles (i.e., any
arbitrary Nx.Defn body) compiles cleanly under codegen — codegen
fuses what it can, dispatches what it can't to the IR walker per
op. No FunctionClauseError reaches the caller.

### G3 — Specialization precedence (codegen vs hand-tuned shaders)

**Symptom**: when both codegen and a hand-tuned shader path
apply (e.g., the user opts in via
`Application.put_env(:exmc, :fused_leapfrog_normal_meta, {0.0, 1.0})`
which routes the eXMC speculative path through
`Nx.Vulkan.leapfrog_chain_normal/7`), codegen still intercepts
at the `Nx.Defn.Compiler` level and dies before the chain-shader
dispatch is reached.

**Root cause**: codegen is wired into `Nx.Vulkan.Compiler.__compile__/4`
(or `__jit__/5`) at the defn-traced level. It claims every defn
JIT call. The chain-shader path lives one level UP in the eXMC
speculative-precomputation code (which calls
`Nx.Vulkan.leapfrog_chain_normal/7` directly, bypassing defn
JIT). But the rest of the model (logp + grad) still goes
through defn → codegen.

**The fix**: codegen should detect when an Expr is "trivially
specialised" (e.g., the entire body is a single Fast named-kernel
call via `Nx.Defn.Expr.optional/3`) and delegate to that
path instead of generating a fused shader. This was the same
issue Emily's compiler had to solve.

**Acceptance**:
- With `fused_leapfrog_normal_meta` set, the chain-shader code
  path runs cleanly under EXMC_COMPILER=vulkan (matches today's
  behaviour on `main`)
- Without the meta, codegen handles the model on its own (per
  G1 + G2)
- Per-step bench at K=32: codegen-only ≤ 100 µs (chain shader
  hits 49.7 µs; codegen with all the right fusions in place
  should be in the same order of magnitude)

---

## Re-run protocol when G1/G2/G3 land

```sh
# On the Linux dev box (super-io):
cd ~/projects/learn_erl/nx_vulkan
git pull nas feat/vulkan-codegen
SPIRIT_DIR= mix compile --force
mix test                                          # gate: 154/0+

cd ~/projects/learn_erl/pymc/exmc
mix deps.compile nx_vulkan --force
mix compile --force

# Test 1: codegen only (no chain meta)
EXMC_COMPILER=vulkan mix run -e '
Application.put_env(:exmc, :compiler, :vulkan)
alias Exmc.{Builder, Dist.Normal, NUTS.Sampler}
ir = Builder.new_ir() |> Builder.rv("x", Normal, %{mu: Nx.tensor(0.0), sigma: Nx.tensor(1.0)})
t0 = System.monotonic_time(:millisecond)
{trace, _} = Sampler.sample(ir, %{}, num_warmup: 50, num_samples: 200, seed: 42)
t1 = System.monotonic_time(:millisecond)
xs = trace["x"] |> Nx.to_flat_list()
mean = Enum.sum(xs) / length(xs)
var  = Enum.sum(Enum.map(xs, fn x -> (x - mean) * (x - mean) end)) / length(xs)
IO.puts("CODEGEN ONLY: wall=#{t1-t0}ms, mean=#{Float.round(mean,4)}, var=#{Float.round(var,4)}")
'

# Test 2: chain meta still works (regression check for G3)
EXMC_COMPILER=vulkan mix run -e '
Application.put_env(:exmc, :compiler, :vulkan)
Application.put_env(:exmc, :fused_leapfrog_normal_meta, {0.0, 1.0})
# ... same body ...
'

# Reference numbers (from 2026-05-03 measurement on RTX 3060 Ti):
#   EXLA:                wall=2435 ms, mean=-0.222, var=1.3552
#   Vulkan unfused (IR walker only): never completes in 16 min
#   Vulkan fused chain:  wall=21000 ms, mean=-0.117, var=1.034
```

**Merge gate**: both tests complete in < 60 seconds with `var
∈ [0.7, 1.5]` and `mean ∈ [-0.5, 0.5]` (small-sample MCMC
noise). If yes, ready to merge to main + bump version + maybe
hex publish 0.2.0.

---

## What this gate is NOT

- Not asking codegen to beat the hand-tuned chain shader. The
  chain shader's K=32 batching is fundamentally different from
  what codegen does (single defn JIT). Codegen's win is on the
  **non-NUTS workloads** and the **long tail of small fused
  subgraphs** that don't justify a hand-written shader.
- Not asking for full Nx.Defn op coverage. The fall-through
  pattern (G2) means coverage gaps degrade gracefully.
- Not asking for ADVI/SMC integration testing yet. Once NUTS
  works, ADVI is the natural next benchmark; SMC is workload-
  specific.

## Cross-reference

- `~/projects/learn_erl/nx_vulkan/feat/vulkan-codegen` — branch
  with current codegen work
- `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md` —
  Phase 1.5 done; Phase 2 done; this is Phase 3 in spirit
  (general defn → GLSL JIT)
- `~/projects/learn_erl/nx_vulkan/RESEARCH_FAST_KERNELS.md` —
  research note on per-dispatch cost. Codegen amortizes
  per-dispatch cost across the entire fused subgraph; chain
  shader amortizes across K consecutive sampler iterations.
  They're complementary.
- The benchmark snippets above are from the 2026-05-04
  Linux-side run that surfaced G1+G2+G3.
