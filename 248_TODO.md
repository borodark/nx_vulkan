# mac-248 — Quick TODO: extend `elementwise_binary_broadcast.comp` with compare ops

**Why**: spirit already ships
`shaders/elementwise_binary_broadcast.comp` with arithmetic ops 0..6
(add/multiply/subtract/divide/pow/max/min). We're about to wire it
into `Nx.Vulkan.Backend.do_binary` on the Linux side, replacing the
host-fallback path for shape-mismatched elementwise ops (the highest-
impact gap in `LIMITATIONS.md` §2).

For parity with the non-broadcast `elementwise_binary.spv` (which
covers ops 0..9), the broadcast variant needs the same three compare
ops added: 7=equal, 8=less, 9=greater. Without them, `Nx.equal(a, b)`
on mismatched shapes still hits host fallback, which defeats half the
point of wiring this shader.

**Scope**: 3 case arms in the switch. 5 minutes. Recompile, push.

## Layout note

Mac-248 uses the flat layout: `~/spirit/`.

## Steps

```
cd ~/spirit
git pull
git checkout -b feat/broadcast-compare-ops
```

Edit `shaders/elementwise_binary_broadcast.comp`. The switch in
`main()` currently ends at case 6:

```glsl
        case 6: r = min(x, y); break;
        default: r = 0.0; break;
```

Add three more cases right before `default`:

```glsl
        case 6: r = min(x, y); break;
        case 7: r = (x == y) ? 1.0 : 0.0; break;   // equal
        case 8: r = (x  < y) ? 1.0 : 0.0; break;   // less
        case 9: r = (x  > y) ? 1.0 : 0.0; break;   // greater
        default: r = 0.0; break;
```

(Same 1.0/0.0 result convention as `elementwise_binary.comp`.)

Compile and push:

```
glslangValidator -V shaders/elementwise_binary_broadcast.comp \
                 -o shaders/elementwise_binary_broadcast.spv
git add shaders/elementwise_binary_broadcast.comp \
        shaders/elementwise_binary_broadcast.spv
git commit -m "elementwise_binary_broadcast: add compare ops 7/8/9 for parity"
git push origin feat/broadcast-compare-ops
```

## Sanity check (optional)

Dispatch op 7 (equal) with `a=[1.0]` shape `{1}` broadcast against
`b=[1.0, 2.0, 3.0]` shape `{3}`. Expected output `[1.0, 0.0, 0.0]`.
Skip if no quick dispatcher — Linux side validates after wiring.

## After your push

Linux side will:

1. Merge `feat/broadcast-compare-ops` to `feature/vulkan-backend` in
   spirit.
2. **Bump push_size 40 → 56** in `nx_vulkan_shim.cpp`. The broadcast
   shader's push constant block is `{n, ndim, out_shape[4],
   a_strides[4], b_strides[4]} = 8 + 16 + 16 + 16 = 56 bytes`. Vulkan
   ignores any push range bytes the shader doesn't read — bumping
   doesn't break existing pipelines.
3. Add `nxv_apply_binary_broadcast(out, a, b, op, ndim, out_shape[4],
   a_strides[4], b_strides[4], spv_path)` to the C++ shim.
4. Add Rust NIF that takes `out_shape` and `a_strides`/`b_strides` as
   `Vec<u32>` (length-4, padded), builds the push constant struct.
5. Add `Nx.Vulkan.Native.apply_binary_broadcast/...` stub.
6. Add `Nx.Vulkan.<op>_broadcast/3` for each arithmetic op + the
   three new compare ops (or one polymorphic helper that takes an
   op atom).
7. Rewire `Nx.Vulkan.Backend.do_binary`: when shapes differ, compute
   the broadcast strides via `Nx.Shape.broadcast/3` semantics
   (stride=0 on a broadcast axis, native stride otherwise), dispatch
   the broadcast shader instead of host fallback.
8. Tests:
   - scalar `{1}` × vector `{4}` → vector
   - vector `{4}` × matrix `{2, 4}` → matrix (broadcast along axis 0)
   - broadcast in last dim: `{2, 1}` × `{2, 4}`
   - 4-D broadcast (max ndim the shader supports)
   - shape-mismatch for unsupported broadcast (shape that doesn't
     broadcast, or ndim > 4) → falls back to host
9. Ping you to `cd ~/nx_vulkan && git pull && mix test`.

## Why this matters

The post-Phase 3 failure breakdown showed `size_mismatch` was the
**single largest remaining bucket** at 41 failures. Most of those are
broadcast cases (e.g., `Nx.add(matrix, vector)` for per-row offsets).
Wiring this shader closes that whole bucket. Combined with mac-248's
fused chain work, the full exmc test suite should drop from "hangs at
60min" to a tractable wall time — the precondition for an honest
Phase 4 benchmark vs EXLA-CUDA.

Estimated impact on full-suite pass rate: targeted-subset 91.9% →
full-suite ~85% after this lands (the remaining gaps are the long tail
of unimplemented backend callbacks documented in `LIMITATIONS.md` §2).
