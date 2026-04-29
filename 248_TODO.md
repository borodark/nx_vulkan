# mac-248 — Quick TODO: extend fused_elementwise.comp with erf + expm1

**Why**: `Nx.Vulkan.fused_chain/3` is wired on the Linux side and tests
pass for ops 0..12 (binary + most unary). Cases 13 (erf) and 14 (expm1)
have op codes assigned but `apply_unary` in `fused_elementwise.comp`
only switches on cases 0..12 — `case 13` and `case 14` fall to the
default and pass the register through unchanged. Adding two case arms
+ pulling in the helper functions from `elementwise_unary.comp` closes
this gap so erf/expm1 can participate in fused chains.

**Scope**: 2 case arms + 2 helper function copies (or `#include`-style
re-paste). 5 minutes. Recompile, push.

## Layout note

Mac-248 uses the flat layout: `~/spirit/` and `~/nx_vulkan/`.

## Steps

```
cd ~/spirit
git pull
git checkout -b feat/fused-erf-expm1
```

Edit `shaders/fused_elementwise.comp`. Above `void main()`, add the two
helpers (same as in `elementwise_unary.comp` — copy verbatim):

```glsl
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
        return x
             + x2 * 0.5
             + x2 * x  * (1.0 / 6.0)
             + x2 * x2 * (1.0 / 24.0)
             + x2 * x2 * x * (1.0 / 120.0);
    } else {
        return exp(x) - 1.0;
    }
}
```

In `apply_unary`, extend the switch with two new cases (right after
`case 12: return r * r;`):

```glsl
        case 12: return r * r;                   // square
        case 13: return erf_approx(r);           // erf
        case 14: return expm1_approx(r);         // expm1
        default: return r;
```

Compile and push:

```
glslangValidator -V shaders/fused_elementwise.comp -o shaders/fused_elementwise.spv
git add shaders/fused_elementwise.comp shaders/fused_elementwise.spv
git commit -m "fused_elementwise: extend apply_unary with cases 13=erf, 14=expm1"
git push origin feat/fused-erf-expm1
```

## Sanity check (optional)

Dispatch a chain `[:multiply, :add, :erf]` with `a=[1.0]`, `b=[0.0]` —
shorthand for erf(a*0 + 0) = erf(0) = 0. With `b=[1.0]` and chain
`[:add, :erf]`: erf(a+1) for a=[0.0] → erf(1.0) = 0.8427.

Or skip — Linux side will validate via `mix test` once the .spv lands.

## After your push

Linux side will:
1. Merge `feat/fused-erf-expm1` to `feature/vulkan-backend` in spirit.
2. Update the moduledoc on `Nx.Vulkan.fused_chain/3` to drop the
   "erf/expm1 pass through unchanged" caveat.
3. Add tests verifying erf/expm1 in mid-chain.
4. Ping you to `cd ~/nx_vulkan && git pull && mix test` for parity.
