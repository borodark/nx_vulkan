# mac-248 — Phase 1.7 shader work

**Target:** add `erf` (op 13) and `expm1` (op 14) to `elementwise_unary.comp`,
compile, push.

## Steps

```
cd ~/projects/learn_erl/spirit
git pull
git checkout -b feat/phase-1.7-shaders
```

Edit `shaders/elementwise_unary.comp`. Add both helpers near the top of the file
(above `void main()`):

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
        // Taylor: x + x^2/2 + x^3/6 + x^4/24 + x^5/120
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

In the op switch, add both new arms:

```glsl
else if (pc.op == 13u) v = erf_approx(in_v);
else if (pc.op == 14u) v = expm1_approx(in_v);
```

Compile and push:

```
glslangValidator -V shaders/elementwise_unary.comp -o shaders/elementwise_unary.spv
git add shaders/elementwise_unary.comp shaders/elementwise_unary.spv
git commit -m "elementwise_unary: op 13 = erf, op 14 = expm1 (Phase 1.7)"
git push origin feat/phase-1.7-shaders
```

## Sanity check (optional)

- op 13 with `[-2.0, -1.0, 0.0, 1.0, 2.0]` → `[-0.9953, -0.8427, 0.0, 0.8427, 0.9953]` (±1.5e-7)
- op 14 with `[-1.0, -0.1, 0.0, 0.1, 1.0]`  → `[-0.6321, -0.0952, 0.0, 0.1052, 1.7183]`

Skip if no quick dispatcher — Linux side will validate via `mix test` once the
`.spv` lands.

## After your push

Linux side will:
1. Merge `feat/phase-1.7-shaders` to `main` in `spirit`.
2. Bump op cap in `nx_vulkan/native/nx_vulkan_native/src/lib.rs` from 12 → 14.
3. Wire `erf`/`expm1` through `Nx.Vulkan` (`@ops_unary` map) and
   `Nx.Vulkan.Backend` (callbacks).
4. Add tests; commit Phase 1.7 nx_vulkan side.
5. Ping you to `git pull` nx_vulkan and re-run `mix test` for three-host parity.
