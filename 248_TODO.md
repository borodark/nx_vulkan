# mac-248 — 4-input fused elementwise shader

**Goal**: extend fused chains to 4 input buffers so real NUTS leapfrog
bodies fuse into one dispatch. Closes the structural reason the 17
timeout failures persist.

**Why 4**: covers leapfrog-body cardinality (3-4 unique input tensors).
On the diminishing-returns curve this is the sweet spot — bigger jumps
to 6/8 are straightforward but should be data-driven.

## Design

Same chain semantics as `fused_elementwise.comp`: register `r` starts
from `a[i]`, ops apply left-to-right. The change: binary ops pick
their second operand from one of three buffers (b, c, d) via a
per-op `buf_idx`.

The compile-time auto-fusion (Linux side) will arrange args so the
chain folds correctly. E.g., `q + eps * p` compiles to:

```
inputs:  a=eps, b=p, c=q
chain:   [multiply(buf=1), add(buf=2)]
result:  r = eps; r *= p; r += q  →  q + eps*p ✓
```

That's 1 dispatch instead of 2-3. For `p + 0.5 * eps * grad`:

```
inputs:  a=eps, b=grad, c=p, d=0.5_const
chain:   [multiply(buf=1), multiply(buf=3), add(buf=2)]
result:  r = eps * grad * 0.5 + p
```

Single dispatch.

## Layout note

Mac-248 uses the flat layout: `~/spirit/`, `~/nx_vulkan/`.

## Steps

```
cd ~/spirit
git pull
git checkout -b feat/fused-4in-shader
```

### New file: `shaders/fused_elementwise_4in.comp`

```glsl
#version 450

// Fused n-way elementwise chain — up to 8 ops, 4 input buffers.
//
// Same chain model as fused_elementwise.comp: register r starts from
// a[i], ops apply left-to-right. Difference: each binary op picks its
// second operand from b/c/d via per-op `buf_idx`.
//
// Op codes (unchanged):
//   Binary 0..6:    add/multiply/subtract/divide/pow/max/min
//   Unary  100..114: exp/log/sqrt/abs/negate/sigmoid/tanh/relu/
//                    ceil/floor/sign/reciprocal/square/erf/expm1
//   255: nop (chain terminator)
//
// buf_idx values for binary ops: 1=b, 2=c, 3=d. Ignored for unary.

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint n;
    uint n_ops;
    uint ops[8];
    uint buf_idx[8];
} pc;

layout (std430, binding = 0) readonly  buffer InputA { float a[]; };
layout (std430, binding = 1) readonly  buffer InputB { float b[]; };
layout (std430, binding = 2) readonly  buffer InputC { float c[]; };
layout (std430, binding = 3) readonly  buffer InputD { float d[]; };
layout (std430, binding = 4) writeonly buffer Output { float out_buf[]; };

// erf and expm1 helpers — copy verbatim from elementwise_unary.comp.
// (If you keep them in a separate file you can #include with
// glslangValidator's -I flag; otherwise just paste.)
float erf_approx(float x) {
    // Abramowitz & Stegun 7.1.26 — error ≤ 1.5e-7
    float a1 =  0.254829592;
    float a2 = -0.284496736;
    float a3 =  1.421413741;
    float a4 = -1.453152027;
    float a5 =  1.061405429;
    float p  =  0.3275911;
    float s  = sign(x);
    float ax = abs(x);
    float t  = 1.0 / (1.0 + p * ax);
    float y  = 1.0 - (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t) * exp(-ax * ax);
    return s * y;
}

float expm1_approx(float x) {
    if (abs(x) < 0.5) {
        float x2 = x * x;
        return x + x2 * 0.5 + x2 * x * (1.0 / 6.0)
             + x2 * x2 * (1.0 / 24.0) + x2 * x2 * x * (1.0 / 120.0);
    } else {
        return exp(x) - 1.0;
    }
}

float read_y(uint i, uint idx) {
    if (idx == 2u) return c[i];
    if (idx == 3u) return d[i];
    return b[i];
}

float apply_unary(float r, uint op) {
    switch (int(op - 100u)) {
        case 0:  return exp(r);
        case 1:  return log(r);
        case 2:  return sqrt(r);
        case 3:  return abs(r);
        case 4:  return -r;
        case 5:  return 1.0 / (1.0 + exp(-r));
        case 6:  return tanh(r);
        case 7:  return max(r, 0.0);
        case 8:  return ceil(r);
        case 9:  return floor(r);
        case 10: return sign(r);
        case 11: return 1.0 / r;
        case 12: return r * r;
        case 13: return erf_approx(r);
        case 14: return expm1_approx(r);
        default: return r;
    }
}

float apply_binary(float r, float y, uint op) {
    switch (int(op)) {
        case 0: return r + y;
        case 1: return r * y;
        case 2: return r - y;
        case 3: return r / y;
        case 4: return pow(r, y);
        case 5: return max(r, y);
        case 6: return min(r, y);
        default: return r;
    }
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;

    float r = a[i];

    for (uint s = 0u; s < pc.n_ops; s++) {
        uint op = pc.ops[s];
        if (op == 255u) break;

        if (op >= 100u) {
            r = apply_unary(r, op);
        } else {
            float y = read_y(i, pc.buf_idx[s]);
            r = apply_binary(r, y, op);
        }
    }

    out_buf[i] = r;
}
```

### Compile + push

```
glslangValidator -V shaders/fused_elementwise_4in.comp \
                 -o shaders/fused_elementwise_4in.spv
git add shaders/fused_elementwise_4in.comp shaders/fused_elementwise_4in.spv
git commit -m "shaders: fused_elementwise_4in (4 input buffers, 8 ops)"
git push origin feat/fused-4in-shader
```

### Sanity check (optional)

If you want to verify before pushing — dispatch a chain that
exercises buf_idx switching:

```
inputs:  a = [1.0, 1.0]   (q)
         b = [2.0, 2.0]   (eps)
         c = [3.0, 3.0]   (p)
         d = [0.5, 0.5]   (half_const)
chain:   [multiply(buf=2), multiply(buf=3), add(buf=0...)]
                    ↑ p          ↑ 0.5
expected: eps * p * 0.5 + (start register a) — but our shader starts r=a (=q=1)
          so result = q * p * 0.5 + ... no wait, the chain starts at a,
          so r = q; r = r * p (idx=2); r = r * 0.5 (idx=3); ... r += eps?
          That gives q * p * 0.5 + eps = 1*3*0.5 + 2 = 3.5

Or for the canonical leapfrog body q + eps*p:
inputs:  a = eps,  b = p,  c = q
chain:   [multiply(buf=1), add(buf=2)]
         r = eps; r *= p; r += q  →  3*2 + 1 = 7 (with values eps=3, p=2, q=1)
```

Skip the sanity check if no quick dispatcher — Linux side will validate
via `mix test` once the .spv lands.

## Push size note

Push struct: `{uint n; uint n_ops; uint ops[8]; uint buf_idx[8]}` =
4 + 4 + 32 + 32 = **72 bytes**. Linux side will need to bump the
pipeline cache push_size from 56 → 72 in `nx_vulkan_shim.cpp`. Vulkan
ignores any unused push range bytes for shaders that declare less.

## After your push

Linux side will:

1. Merge `feat/fused-4in-shader` to `feature/vulkan-backend` in spirit.
2. Bump push_size 56 → 72 in `c_src/nx_vulkan_shim.cpp` for the
   pipeline cache.
3. Add `nxv_fused_chain_4` C shim (5 buffers: a, b, c, d, out; 72-byte
   push).
4. Add Rust NIF `fused_chain_4(a, b, c, d, ops, buf_idx, spv_path)`.
5. Add `Nx.Vulkan.fused_chain_4/4` API helper.
6. Extend `Nx.Vulkan.Compiler` to detect 3-4 arg defns and emit the
   4-input chain when the IR matches:
   - 3-arg `fn a, b, c -> ... end`: c stays unused (passes c as d, or skip)
   - 4-arg `fn a, b, c, d -> ... end`: full coverage
   - Right-fold + commutative-arrange to produce a valid (op, buf_idx)
     sequence
7. Add tests: leapfrog-body shape `(q, p, eps) -> q + eps * p`,
   half-step `(q, p, eps, grad) -> q + 0.5 * eps * grad`.
8. Re-run exmc Vulkan suite (v12). Expected: timeouts drop from
   ~17 to ≤ 5 once leapfrog bodies fuse.

## What this DOES NOT do

- **Doesn't help defns with > 4 unique inputs** — log-prob composites
  with 5+ inputs still split. If after this lands the timeout count
  is still meaningful, we'll go to 6 or 8.
- **Doesn't reduce per-dispatch overhead** — that's Iter 4 fence
  reuse work. The two changes compose: 4-input shader cuts dispatch
  count, fence reuse cuts per-dispatch cost.
- **Doesn't lift the chain-start-at-a constraint**. Patterns where
  the chain has to start mid-graph (e.g., `(a + b) * (c + d)`) still
  split.

## Cross-reference

- `PERSISTENT_BUFFERS_PLAN.md` — Iter 4 fence reuse (parallel work)
- `LIMITATIONS.md` §3 (fused chain) — the 2-input limit this lifts
- `bench/leapfrog_bench.exs` — re-run after wiring; expect 4-arg case
  to drop from "falls through" to "fused" with measurable speedup
