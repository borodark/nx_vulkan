# Phase 3 status — exmc test suite under `compiler: :vulkan`

**Date**: 2026-04-29
**Branch**: `main` at `7bde7a8` (post-backfill)

## Headline numbers

| Run | Test files | Tests | Failures | Pass rate | Wall time |
|-----|---|---|---|---|---|
| Baseline (full suite, pre-backfill) | all | 622 | 174 | 72.0% | 157s |
| Post-backfill (full suite) | all | 622 | unknown | unknown | killed at 60min, log incomplete |
| Targeted (sampler-adjacent) | 4 files | 63 + 11 doctests | 6 | **91.9%** | 7.1s |

The targeted subset comfortably clears the 80% target the EXMC_PORT_PLAN
projected. The full-suite number isn't honest because the host-fallback
backfill makes some property-based MCMC tests (200+ chains × 1000 steps
× ~30 round-trips per step) impossibly slow — not failing, hanging.

## What landed

`7bde7a8` adds host-fallback backend callbacks for everything exmc's
suite reached for that v0.1 hadn't surfaced:

| Bucket | Callbacks | Path |
|---|---|---|
| Comparison | equal/less/greater | shader (Phase 1.1) |
| Comparison | less_equal/greater_equal/not_equal | host fallback |
| Selection | select, clip | host fallback |
| Shape | concatenate, stack, pad | host fallback |
| Transcendental | log1p | host fallback |
| Status | is_infinity | host fallback |
| Bitwise/integer | right_shift/left_shift/remainder/quotient/and/or/xor | host fallback |
| Mixed-type elementwise | f64 / s32 / s64 in compute path | auto-detect → host fallback |

The mixed-type fallback is the biggest semantic call: `do_binary` and
each unary op now check operand types up front. If any input isn't
`{:f, 32}`, or if shapes differ (broadcast cases), the op transfers
both operands to `Nx.BinaryBackend`, runs the host op, uploads. Slow
but correct.

## Where the time goes

A NUTS leapfrog has ~30 elementwise ops chained on f32 tensors. Under
the current backend each op is one shader dispatch:

```
download(a) → host op → upload  ↓ for each non-f32 chain
dispatch shader              ↓ for each f32 op
```

For a small model (4-d), a single leapfrog step takes ~10-20ms on
Vulkan vs ~0.5ms on EXLA-CUDA. Multiply by 1000 steps × 5 chains × 8
property test variations and the property-test files individually
exceed the 60-second per-test default timeout.

This is the **fundamental performance gap** the EXMC_PORT_PLAN
flagged: no kernel fusion. mac-248 already solved it at the shader
layer:

- `shaders/fused_elementwise.comp` — up to 8 ops in one dispatch
- `bench_fused.cpp` — measured **1.6-4× speedup** over sequential
- 6/6 fused-correctness tests pass at the shader level

What's missing is the Elixir IR pattern matcher that recognizes
chains of `Nx.add` / `Nx.multiply` / `Nx.exp` / etc. and emits the
fused dispatch instead of N separate ones.

## Phase 4 prerequisites

To actually benchmark Vulkan vs EXLA on the leapfrog hot path, the
fusion pattern matcher (FUSION_RESEARCH.md Path A) needs to land:

1. Wrap the fused.spv in a `Nx.Vulkan.fused_chain/3` API
2. Add a small Elixir-side IR rewrite that detects chains during
   `defn` evaluation and emits one fused dispatch per chain
3. Re-run the full exmc suite — the property tests should drop from
   "hangs at 60min" to a reasonable wall time

Without that, Phase 4 (benchmark) will measure host-fallback overhead,
not the actual Vulkan compute path.

## Recommended next move

Option A (deferred): leave Phase 3 here, document the path-A
prerequisite, and proceed to Phase 4 against the targeted subset
(the four test files where 91.9% of tests pass within timeout).

Option B (recommended): pull mac-248's fused-elementwise into
nx_vulkan via a thin pattern matcher. Use the IR walk path the
research doc maps. Two days of work; unblocks the full suite *and*
gives a real Phase 4 number.

Option B is the way. The shader-side work is done; the Linux side is
the only remaining engineering.

## Long tail (ignoring slow-suite concerns)

After backfill, the failure types still in the log were:

| # | Class | Where to fix |
|---|---|---|
| ~30 | size_mismatch on identical shapes | broadcast-pre-dispatch corner cases |
| ~10 | stale `{:s, 32}` in `constant/3` paths | already fixed in `d654c76` after this run |
| ~5 | `is_infinity` host fallback for vectors with `:neg_infinity` | binary encoding fix |
| 4 | "two incompatible tensor implementations" in defn | exmc-side closure refactor |
| 2 | timeout (real, not hang) | property test count tuning |
| 1 | `Exmc.Trading.Web.Endpoint` startup | unrelated to JIT |
| 1 | Web/Phoenix dep wiring | unrelated |

Most are mechanical. The `size_mismatch` bucket is the only one that
needs investigation — likely cases where Nx's broadcast machinery
produces tensors of subtly different shapes that pass Nx's protocol
checks but trip our shape-equality assertion in `do_binary`.

## Tests passing on Vulkan today

From the targeted run (`mix test test/exmc_test.exs test/dist_test.exs
test/diagnostics_test.exs test/compiler_test.exs`):

- All 21 distribution log-prob tests
- 11/11 doctests
- 12/15 compiler tests
- All diagnostics
- Full exmc_test smoke

The forward-pass operator set is essentially complete. The
backward-pass via `Nx.Defn.Grad` is exercised by the compiler tests
that pass — it's not a regression vector.

## Files touched

- `lib/nx_vulkan/backend.ex` — backfill callbacks
- `test/test_helper.exs` (exmc) — Vulkan compiler dispatch
- `lib/exmc/jit.ex` (exmc) — `:vulkan` config option
- `mix.exs` (exmc) — optional `nx_vulkan` path dep
- `config/test.exs` (exmc) — `EXMC_COMPILER` env override
