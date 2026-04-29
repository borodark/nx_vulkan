# mac-247 — Optional speedup shader: `cast.comp` (f32↔f64)

**Goal**: replace the Phase 1.8 host-materialize cast for `as_type` between
f32 and f64. The host path downloads → decodes in Erlang → re-encodes →
uploads. A 1-op compute shader does the same conversion in-kernel without
the round-trip. Useful for the mass-matrix accumulator hot path in exmc.

**Scope**: f32→f64 and f64→f32 only. Integer ↔ float casts stay
host-materialized (rare, edge cases for index types).

## Layout note

Mac-247 uses the flat layout: `~/spirit/` and `~/nx_vulkan/` (no
`projects/learn_erl/` prefix). Paths below assume that.

## Steps

```
cd ~/spirit
git pull
git checkout -b feat/cast-shader
```

### New file: `shaders/cast.comp`

```glsl
#version 450

// 1-D cast between f32 and f64. Op spec constant chooses direction:
//   op == 0u : f32 → f64  (bind 0 = float[]  ; bind 1 = double[])
//   op == 1u : f64 → f32  (bind 0 = double[] ; bind 1 = float[])
//
// Vulkan std430 happily holds float and double in the same SSBO slot
// type-punned via separate bindings; we use one shader and switch the
// bound buffer pair per op. The dispatch is 1-D over n elements.

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint n;
    uint op;     // 0 = f32→f64, 1 = f64→f32
} pc;

layout (std430, binding = 0) readonly  buffer In32  { float  in32[];  };
layout (std430, binding = 0) readonly  buffer In64  { double in64[];  };
layout (std430, binding = 1) writeonly buffer Out32 { float  out32[]; };
layout (std430, binding = 1) writeonly buffer Out64 { double out64[]; };

// NOTE: GLSL alias-warning on duplicate binding numbers — pick the
// non-aliased version in the .spv compile. Cleaner: split into two
// shaders cast_f32_to_f64.comp and cast_f64_to_f32.comp. If aliasing
// trips glslangValidator, fall back to two files.

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.n) return;

    if (pc.op == 0u) {
        out64[i] = double(in32[i]);
    } else {
        out32[i] = float(in64[i]);
    }
}
```

**If glslangValidator rejects the duplicated `binding = 0` aliasing**,
split into two files:
- `cast_f32_to_f64.comp` (single binding pair: float in, double out)
- `cast_f64_to_f32.comp` (single binding pair: double in, float out)

Either layout works; the two-file split is the safer default. Pick whichever
compiles cleanly.

### Compile

If single shader:
```
glslangValidator -V shaders/cast.comp -o shaders/cast.spv
```

If two-file split:
```
glslangValidator -V shaders/cast_f32_to_f64.comp -o shaders/cast_f32_to_f64.spv
glslangValidator -V shaders/cast_f64_to_f32.comp -o shaders/cast_f64_to_f32.spv
```

### f64 capability

The `f64=yes` device flag at init is what lets the compiled .spv run.
Both Macs report `f64=yes` (verified in earlier test runs). If a future
device reports `f64=no`, the host fallback path stays intact.

### Commit + push

```
git add shaders/cast*.comp shaders/cast*.spv
git commit -m "shaders: cast f32↔f64 (compute-side as_type for Phase 1.8)"
git push origin feat/cast-shader
```

## After your push

Linux side will:
1. Merge `feat/cast-shader` to `feature/vulkan-backend` in `spirit`.
2. Add `nxv_cast` to the C++ shim and a `cast` Rust NIF.
3. Wire `Nx.Vulkan.Backend.as_type/2` to dispatch f32↔f64 to the shader
   path (other types stay on the host fallback).
4. Add tests + benchmark vs the host path; merge.
5. Ping you to `cd ~/nx_vulkan && git pull && mix test` for parity.
