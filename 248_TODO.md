# mac-248 — Day 6 (Week 2 step 2c): f64 elementwise shaders

**Goal**: f64 variants of the elementwise binary + unary shaders so the
mass-matrix accumulator path (and any defn that explicitly requests
:f64) runs on GPU instead of falling back to BinaryBackend round-trip.
Closes the precision side of the 8 ArithmeticError failures from
your 622/29 baseline.

**Why now**: Day 5 (step size clamp + stable inverse-softplus) is in.
That kills overflow at the *adaptation* layer. Day 6 lets the
*compute* layer carry f64 when callers ask for it — currently any f64
operand forces host fallback (~50µs/op vs ~5µs on GPU).

**Scope for this TODO**: 2 shaders only. Reduce_axis_f64 and
matmul_f64 land later if the elementwise pair doesn't move the needle
enough.

## Layout note

Mac-248 uses the flat layout: `~/spirit/`, `~/nx_vulkan/`. Paths below
assume that.

## Steps

```
cd ~/spirit
git pull
git checkout -b feat/elementwise-f64-shaders
```

### New file: `shaders/elementwise_binary_f64.comp`

Copy your existing `shaders/elementwise_binary.comp` and apply these
changes:

1. Add the f64 extension at the top (right after `#version 450`):

```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require
```

2. Replace every `float` with `double` in the buffer + `r` declarations.
3. Replace `1.0` literals in the compare ops with `1.0LF` (double
   literal) — the lit float→double conversion can produce a warning
   but is functionally fine; the explicit form is cleaner.

The body stays identical — same op switch, same op codes 0..9.

### New file: `shaders/elementwise_unary_f64.comp`

Same recipe applied to `elementwise_unary.comp`. The erf and expm1
helpers (op 13/14) need their float arithmetic doubled too:

```glsl
double erf_approx(double x) {
    double a1 =  0.254829592LF;   // ... etc
    // (same A&S 7.1.26 formula, all literals as double)
}

double expm1_approx(double x) {
    if (abs(x) < 0.5LF) {
        // Taylor — all coefficients as double
    } else {
        return exp(x) - 1.0LF;
    }
}
```

GLSL's built-in `exp`, `log`, `sqrt`, `tanh`, `pow`, etc. all have
double overloads when the f64 extension is enabled, so the body
otherwise just inherits.

### Compile + push

```
glslangValidator -V shaders/elementwise_binary_f64.comp \
                 -o shaders/elementwise_binary_f64.spv
glslangValidator -V shaders/elementwise_unary_f64.comp \
                 -o shaders/elementwise_unary_f64.spv

git add shaders/elementwise_binary_f64.comp shaders/elementwise_binary_f64.spv \
        shaders/elementwise_unary_f64.comp shaders/elementwise_unary_f64.spv

git commit -m "shaders: f64 elementwise binary + unary (Day 6 / step 2c)"
git push origin feat/elementwise-f64-shaders
```

### Verify on FreeBSD GT 750M

Optional but useful — the GT 750M reports `f64=yes` (per spirit init);
this confirms the .spv actually loads:

```
cd ~/spirit
# Quick sanity: any matmul-bench-style call that loads the new .spv
# and writes/reads two doubles. Or just trust the Linux side to
# validate via mix test once the wiring lands.
```

## After your push

Linux side will:

1. Merge `feat/elementwise-f64-shaders` to `feature/vulkan-backend` in
   spirit.
2. Generalize the C++ shim's `nxv_apply_binary` / `nxv_apply_unary`
   into f32/f64 variants (or add `nxv_apply_binary_f64`/_unary_f64 —
   simplest approach since binding sizes differ: 8 bytes/element).
3. Rust NIF: dispatch by element type; `Nx.Vulkan.Native.apply_binary`
   takes a precision atom or the caller passes the right .spv path.
4. Backend `do_binary`/unary: when type is `{:f, 64}` and operands
   match shapes, dispatch the f64 shader instead of host fallback.
   Mixed-type still host-falls-back; this is the all-f64 fast path.
5. Add tests with f64 arithmetic that previously round-tripped:
   small-shift addition, tanh saturation, pow with f64 base/exp.
6. Ping you to `cd ~/nx_vulkan && git pull && mix test` — expect ≥
   124/0, with f64 round-trip tests no longer slow.

## Estimated impact

| Metric | Before Day 6 | After Day 6 |
|---|---|---|
| Per-op cost on f64 tensor | ~50 µs (host round-trip) | ~5–10 µs (GPU) |
| Mass matrix Welford steady-state | Slow — N round-trips per window | One reduce_axis call per window (still f32 — Day 6 stretch goal is f64 reduce, see below) |
| ArithmeticError remaining after Day 5 | 8 (per the 622/29 breakdown) | **target 0–3** |

If 2-4 ArithmeticError still show after Day 6 + Day 5:

- Profile which test surfaces them — the remaining cases likely need
  Day 7 (logsumexp shader) or a model-side fix.
- Day 6 stretch: copy `reduce_axis.comp` → `reduce_axis_f64.comp` and
  push the same way. Linux wires it the same as the elementwise pair.

## Cross-reference

- `PATH_TO_FULL_PASS.md` step 2c
- `LIMITATIONS.md` §1 (compute precision) — this TODO closes most of it
- spirit `core/include/engine/Backend_par_vulkan.hpp` —
  `g_vk_ctx.has_float64` is already detected at init; no spirit
  backend changes needed.
