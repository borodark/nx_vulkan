# mac-248 — Y4: leapfrog_chain_weibull.spv (closes the last `:vulkan_known_failure`)

## Recent state — what's already done

All prior shader tasks merged:

- ✅ X3 sign fix (5 leapfrog shaders)
- ✅ Y queue (Student-t, Cauchy, HalfNormal Phase 2 chains)
- ✅ Z1 Philox audit + Z2 canonical Random123 replacement
- ✅ X1 f64 chain + X2 reduce_full_f64 (.spv vendored; Linux-side
  wiring of reduce_full_f64 is a follow-up over there, not your task)
- ✅ FreeBSD bring-up validated on mac-248
- ✅ Hex-prep merged onto main (`6a4540d` after the 718d80a repair)
- ✅ Stage 1.5.4 variance bias resolved (var = 1.03 vs ref 1.01)
- ✅ eXMC dispatch generalised to all 5 chain shaders via tagged
  meta on pymc/main `807a1db3f`
- ✅ End-to-end smoke test of the four new families on
  EXMC_COMPILER=vulkan (Linux RTX 3060 Ti, 2026-05-04). Result:

  | Family | EXLA m / v | Vulkan m / v | Verdict |
  |--------|------------|--------------|---------|
  | Normal      | -0.222 / 1.355 | -0.222 / 1.355 | ✓ identical |
  | Exponential |  0.491 / 0.286 |  0.549 / 0.285 | ✓ matches |
  | StudentT    |  0.576 / 29.26 |  0.549 / 29.77 | ✓ matches |
  | Cauchy      |  1.584 / 68.7  |  2.230 / 1182  | (Cauchy has no finite variance — vacuous) |
  | HalfNormal  |  0.951 / 0.574 |  0.593 / 0.139 | ✓ within tolerance |

  Your shaders work end-to-end through eXMC's NUTS speculative path.

The codegen branch (`feat/vulkan-codegen` at `2cd9f19`) is on a
separate gate and is **not mac-248 work** — its remaining open
item (G4: op coverage in `Nx.Vulkan.Codegen.analyze/2`) is
Elixir compiler engineering. Stays off your queue.

---

## Y4 — `leapfrog_chain_weibull.spv` (the active task)

The one remaining `:vulkan_known_failure` test in the eXMC suite
is **`test/weibull_test.exs:98` "Weibull RV compiles and samples
via NUTS"** — flagged in `pymc/exmc/docs/VULKAN_KNOWN_ISSUES.md`
issue #2. Same template as your Phase 2 shaders. Exponential is
the closest sibling: also unconstrained-via-log-transform.

### Distribution and parameterization

eXMC's `Exmc.Dist.Weibull` parameterises Weibull(k, λ) with
shape `k > 0`, scale `λ > 0`. NUTS samples the unconstrained
parameter `q_uc = log(q)`, `q ∈ (0, ∞)`. Working on the
unconstrained line, the log-density (with log-Jacobian) is:

```
log p(q_uc | k, λ) = log(k) − k·log(λ) + k·q_uc − (exp(q_uc)/λ)^k
```

(the `+q_uc` Jacobian for the log transform combines with
Weibull's pdf's `(k-1)·q_uc` to give `k·q_uc`.)

Gradient (closed form — no autodiff in the shader):

```
∇logp(q_uc) = k · (1 − (exp(q_uc)/λ)^k)
```

Per-element logp contribution (constants pulled out):

```
contrib = k·q_uc − (exp(q_uc)/λ)^k
```

The constant `n · (log(k) − k·log(λ))` is precomputed by the
host and passed in as `logp_const`. **No `lgamma` in the
shader** — the log-gamma terms become host-side constants.

### Shader spec

```glsl
#version 450

// Fused chain of K NUTS leapfrog steps for a Weibull(k, lambda)
// log-density model on the unconstrained line.
// Closed-form gradient; no autodiff. Sign convention is post-X3:
//   p_half = p + 0.5 * eps * grad_q   (PLUS, not MINUS)

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint  n;
    uint  K;
    float eps;
    float k;            // Weibull shape parameter
    float lambda;       // Weibull scale parameter
    float logp_const;   // n * (log(k) - k * log(lambda))
} pc;

layout (std430, binding = 0) readonly  buffer In_q     { float q_init[]; };
layout (std430, binding = 1) readonly  buffer In_p     { float p_init[]; };
layout (std430, binding = 2) readonly  buffer In_mass  { float inv_mass[]; };
layout (std430, binding = 3) writeonly buffer Out_q    { float q_chain[]; };
layout (std430, binding = 4) writeonly buffer Out_p    { float p_chain[]; };
layout (std430, binding = 5) writeonly buffer Out_grad { float grad_chain[]; };
layout (std430, binding = 6) writeonly buffer Out_logp { float logp_chain[]; };

shared float partial[256];

float weibull_grad(float q_uc) {
    float ratio = exp(q_uc) / pc.lambda;
    return pc.k * (1.0 - pow(ratio, pc.k));
}

float weibull_logp_contrib(float q_uc) {
    float ratio = exp(q_uc) / pc.lambda;
    return pc.k * q_uc - pow(ratio, pc.k);
}

void main() {
    uint i   = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationIndex;
    bool in_bounds = (i < pc.n);

    float qi = in_bounds ? q_init[i] : 0.0;
    float pi = in_bounds ? p_init[i] : 0.0;
    float mi = in_bounds ? inv_mass[i] : 0.0;

    for (uint k = 0; k < pc.K; k++) {
        float grad_q = in_bounds ? weibull_grad(qi) : 0.0;
        float p_half = pi + 0.5 * pc.eps * grad_q;
        qi = qi + pc.eps * mi * p_half;
        float grad_qn = in_bounds ? weibull_grad(qi) : 0.0;
        pi = p_half + 0.5 * pc.eps * grad_qn;

        if (in_bounds) {
            q_chain[k * pc.n + i]    = qi;
            p_chain[k * pc.n + i]    = pi;
            grad_chain[k * pc.n + i] = grad_qn;
        }

        partial[tid] = in_bounds ? weibull_logp_contrib(qi) : 0.0;
        barrier();
        for (uint s = 128u; s > 0u; s /= 2u) {
            if (tid < s) partial[tid] += partial[tid + s];
            barrier();
        }
        if (tid == 0u) {
            logp_chain[k] = partial[0] + pc.logp_const;
        }
        barrier();
    }
}
```

Push constants total: 24 bytes (well under Vulkan's 128-byte floor).

### Hand-check at K=1, n=4

With `k=2.0, λ=1.0, eps=0.1, q_init=[0,0,0,0], p_init=[0.5,...],
inv_mass=[1,...]`:

- `grad(0) = 2·(1 − (e^0/1)^2) = 2·(1 − 1) = 0`
- `p_half = 0.5 + 0.05·0 = 0.5`
- `qn = 0 + 0.1·1·0.5 = 0.05`
- `grad(0.05) = 2·(1 − (e^0.05)^2) = 2·(1 − 1.10517) = -0.21034`
- `pn = 0.5 + 0.05·(-0.21034) = 0.48948`
- `contrib(0.05) = 2·0.05 − (e^0.05)^2 = 0.1 − 1.10517 = -1.00517`
- sum over 4 elements: -4.02068
- `logp_const = 4·(log(2) − 2·log(1)) = 4·0.6931 = 2.7726`
- `logp_chain[0] = -4.02068 + 2.7726 ≈ -1.248`

Sanity: q_chain[0,0]≈0.050, p_chain[0,0]≈0.4895,
grad_chain[0,0]≈-0.2103, logp_chain[0]≈-1.248. Confirm in your
commit message before pushing.

### Compile + push

```sh
cd ~/spirit
git pull origin feat/fused-leapfrog-chain-normal
glslangValidator -V shaders/leapfrog_chain_weibull.comp \
                 -o shaders/leapfrog_chain_weibull.spv
git add shaders/leapfrog_chain_weibull.{comp,spv}
git commit -m "shaders: leapfrog_chain_weibull — fused Weibull chain (sanity: q[0,0]≈0.05, logp[0]≈-1.248 at K=1, k=2, λ=1)"
git push origin feat/fused-leapfrog-chain-normal
```

### After your push, Linux side will

1. Vendor `leapfrog_chain_weibull.spv` into `nx_vulkan/priv/shaders/`.
2. Add C++ shim entry, Rust NIF, Native stub, Elixir wrapper —
   same template as `leapfrog_chain_exponential` (closest sibling).
3. Add a `{:weibull, k, lambda, logp_const}` clause to
   `Exmc.NUTS.Tree.do_dispatch/10` in pymc/exmc — multi-clause
   defp pattern matching, no case/if, same convention as the
   other 5 chain dispatch clauses.
4. Re-run `test/weibull_test.exs:98` under
   `EXMC_COMPILER=vulkan`. When it passes, untag from
   `:vulkan_known_failure`. Update
   `exmc/docs/VULKAN_KNOWN_ISSUES.md` to mark issue #2 closed.

This closes the last `:vulkan_known_failure` in the eXMC test
suite. The chain shader story becomes complete for the
distributions eXMC currently ships.

---

## What's NOT in this task

- **Multi-WG variants** of Phase 2 chain shaders (Y5 placeholder)
  — useful when models have d > 256 but no current eXMC test
  exercises that. Defer until a real model needs it.
- **Codegen op coverage** (G4 from prior TODO) — Elixir-side
  work, not your strength. Stays on `feat/vulkan-codegen` for
  whoever picks up codegen performance work.
- **`reduce_full_f64.spv` wiring** — `.spv` already vendored on
  main; Linux side adds the C++ shim + NIF + Elixir wrapper.
  Not your task.

## Cross-reference

- `~/projects/learn_erl/pymc/exmc/lib/exmc/dist/weibull.ex` —
  reference Weibull math (the gradient + logp formulas the
  shader reimplements)
- `~/projects/learn_erl/pymc/exmc/test/weibull_test.exs:98` —
  the test that flips green
- `~/projects/learn_erl/pymc/exmc/docs/VULKAN_KNOWN_ISSUES.md`
  issue #2 — the doc this closes
- `~/projects/learn_erl/spirit/shaders/leapfrog_chain_exponential.comp`
  — closest sibling shader; copy and adapt
- `~/projects/learn_erl/nx_vulkan/PLAN_FUSED_LEAPFROG.md` —
  Phase 2 scope; Weibull was always the last family on the list
