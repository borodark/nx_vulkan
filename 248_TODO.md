# mac-248 — Optional speedup shader: `reduce_axis.comp`

**Goal**: replace the Phase 1.4 host-materialize partial-axis reduction
with a real GPU shader. The current path downloads → walks output coords
in Erlang → uploads. For exmc's per-window Welford updates (sum/max along
a batch axis with thousands of elements per kept slot) the host loop is
the bottleneck.

**Scope**: per-axis `sum`, `reduce_max`, `reduce_min` over a single
contiguous axis. Multi-axis reduce can stay host-materialized for v0.1
(or compose: dispatch the shader N times for N reduced axes — second
dispatch onward operates on the previous output).

## Layout note

Mac-248 uses the flat layout: `~/spirit/` and `~/nx_vulkan/` (no
`projects/learn_erl/` prefix). Paths below assume that.

## Steps

```
cd ~/spirit
git pull
git checkout -b feat/reduce-axis-shader
```

### New file: `shaders/reduce_axis.comp`

The trick is the layout: any partial-axis reduction can be flattened
into a 3-D iteration `(outer, reduce, inner)` where:
- `outer` = product of dim sizes before the reduced axis
- `reduce` = size of the reduced axis itself
- `inner` = product of dim sizes after the reduced axis

Output is `outer × inner` row-major. Each invocation handles one
`(outer, inner)` slot and folds across `reduce`.

```glsl
#version 450

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint outer;        // product of dims before reduced axis
    uint reduce_size;  // size of the reduced axis
    uint inner;        // product of dims after reduced axis
    uint op;           // 0 = sum, 1 = max, 2 = min
} pc;

layout (std430, binding = 0) readonly  buffer In  { float a[]; };
layout (std430, binding = 1) writeonly buffer Out { float c[]; };

void main() {
    uint slot = gl_GlobalInvocationID.x;
    uint n_slots = pc.outer * pc.inner;
    if (slot >= n_slots) return;

    uint o = slot / pc.inner;
    uint i = slot % pc.inner;
    uint stride_outer = pc.reduce_size * pc.inner;
    uint base = o * stride_outer + i;

    float acc;
    if (pc.op == 0u) acc = 0.0;
    else if (pc.op == 1u) acc = -1.0/0.0;   // -inf
    else                  acc =  1.0/0.0;   // +inf

    for (uint k = 0u; k < pc.reduce_size; ++k) {
        float v = a[base + k * pc.inner];
        if (pc.op == 0u)      acc += v;
        else if (pc.op == 1u) acc = max(acc, v);
        else                  acc = min(acc, v);
    }

    c[slot] = acc;
}
```

**Optimization note (optional, v0.2)**: for very large `reduce_size`
(>1024), tree-reduce within a workgroup using shared memory. For exmc's
typical `reduce_size` of 100..2000 the linear-scan is fine and avoids
shared-memory complexity. Skip the tree variant for now; ship the simple
linear loop.

### Compile + push

```
glslangValidator -V shaders/reduce_axis.comp -o shaders/reduce_axis.spv
git add shaders/reduce_axis.comp shaders/reduce_axis.spv
git commit -m "shaders: reduce_axis — per-axis sum/max/min (Phase 1.4 GPU path)"
git push origin feat/reduce-axis-shader
```

### Sanity check (optional)

Dispatch with `outer=2, reduce_size=3, inner=2, op=0` on input
`[1,2, 3,4, 5,6, 7,8, 9,10, 11,12]` (a 2×3×2 row-major tensor reducing
axis 1):
- slot (0,0): a[0]+a[2]+a[4] = 1+3+5 = 9
- slot (0,1): a[1]+a[3]+a[5] = 2+4+6 = 12
- slot (1,0): a[6]+a[8]+a[10] = 7+9+11 = 27
- slot (1,1): a[7]+a[9]+a[11] = 8+10+12 = 30

Expected output: `[9, 12, 27, 30]`.

## After your push

Linux side will:
1. Merge `feat/reduce-axis-shader` to `feature/vulkan-backend` in `spirit`.
2. Add `nxv_reduce_axis(in, out, outer, reduce_size, inner, op)` to the
   C++ shim + a `reduce_axis` Rust NIF.
3. Add `Nx.Vulkan.reduce_axis/4` API + rewire
   `Nx.Vulkan.Backend.do_reduce/4` to:
   - compute `outer`/`reduce`/`inner` from the IR layout,
   - dispatch to the shader for single-axis reduce,
   - fold N times for multi-axis reduce, OR fall back to host for the
     multi-axis case if the layout permutation is awkward.
4. Add tests + benchmark vs the host path; merge.
5. Ping you to `cd ~/nx_vulkan && git pull && mix test` for parity.
