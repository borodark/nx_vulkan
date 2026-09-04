# Limitations and Loose Ends

> ⚠️ **PARTIALLY SUPERSEDED (2026-08-02).** This document was written at
> commit `bccdb92` (post-Phase 3) and predates three major changes now on
> `main`, so several sections below are **out of date**:
> - **Fusion is real.** The `Nx.Vulkan.Compiler` (`Nx.Defn.Compiler`)
>   ships whole-graph fusion — elementwise chains, fused reductions, and a
>   multi-stage split at dot/conv/reduce/transpose boundaries. The old
>   §3–§5 below describe the *removed* `Nx.Vulkan.Fuse` macro /
>   `fused_chain` prototype and its "No fusion" / "two inputs only" /
>   "f64 only" / "no reductions" constraints — **all superseded**. See the
>   [README fusion section](README.md#the-nxdefn-fusion-compiler-thrust-3).
> - **Native f32 exists.** The "compute is f64, f32 cast to f64" claims are
>   stale — the hot ops dtype-dispatch native f32 shaders.
> - **conv is implemented** (native im2col + GEMM, f32/f64) and is a fusion
>   boundary; `fft`/`ifft` are native too.
> - The `Nx.Vulkan.Backend` (spirit C++) Elixir backend was **dropped**;
>   references to it below are historical.
> - Test counts below (e.g. "112/0") are stale: current suite is
>   **833 doctests, 903 tests, 0 failures** on super-io, both Keplers and the Jetson.
>
> The genuinely-still-true limitations (host-fallback long tail: sort,
> scatter, native linalg; batched/non-2D dot; etc.) remain accurate.

**Status**: post-Phase 3 / Path A.2 (commit `bccdb92`) — see banner above
**Audience**: future contributors, anyone benchmarking, anyone wondering
why a particular operator is slow.

This document enumerates the shortcuts taken, the operators that are
unimplemented, the cases where we know the path is slow, and the
follow-ups required to close each gap. Read this before drawing
conclusions from a Vulkan benchmark number.

---

## 0. Nx.Backend parity — current status (2026-07-28, nx 0.13)

> **This section is authoritative for parity; the older per-op table in §2 is
> historical** (it predates the host-fallback batch and lists ops as
> "not implemented" that are now implemented). See `docs/PARITY_STATUS.md` for
> the full regenerated gap and bucket classification.

`VulkanoBackend` implements **every** `Nx.Backend` callback nx 0.13 declares —
`MapSet.difference(callbacks, impl)` is empty. Ops are either native f64 Vulkan
shaders (hot kernels) or correct host fallbacks through `BinaryBackend`
(incidental ops). Verified against a `BinaryBackend` reference in f64 by
`test/nx_vulkan/parity_fallback_test.exs`.

### Permanent skips (never implemented — fall back to EXLA / BinaryBackend)

| Op | Why it is skipped |
|---|---|
| `from_pointer/5` (callback) | FFI handle — no computation to accelerate; `BinaryBackend` itself raises. Delegated to `Nx.BinaryBackend.from_pointer`. |
| `to_pointer/2` (callback) | FFI handle — nothing to compute; transfers to `BinaryBackend` and delegates. |
| `phase` (not a callback) | `Nx.phase` composes from primitives; it only means anything for complex inputs, and the shader ISA is f64-**real** with no complex type. Nothing to implement or accelerate. |

### Native GPU shaders (Phase 2, landed) — with correct host-fallback tails

`conv/4`, `fft/3`, `ifft/3` now run **real f64 Vulkan compute shaders** on the
GPU for the common case, and host-fall-back (still correct) for the rest:

| Op | On GPU | Host fallback (correct) |
|---|---|---|
| `fft` / `ifft` | last axis, power-of-two length == axis size, real-f64/complex-f64 input → c128 | other axes, padded/sliced/non-pow2 length, f32/int → c64, `fft2` |
| `conv` | spatial rank 1–3, feature/batch groups == 1, identity permutations, f64 | groups > 1 (incl. depthwise), non-identity permutations, non-f64, rank > 3 |

Follow-on GPU work (all currently correct via fallback): native 2-D `fft2`,
mixed-radix (non-power-of-two) FFT, grouped/depthwise conv, and channels-last
(permuted) conv. `phase` stays a permanent skip (complex-only).

---

## 1. Compute precision

> **Updated twice — read this, not the paragraph below it.** This section
> has been superseded in *both* directions and the intermediate wording
> survived longer than it was true:
>
> 1. Phase 3 said "f32-only, f64 deferred". Superseded.
> 2. The f64 migration then made it **f64-only**, casting f32 up. That is
>    the claim the paragraph below was written for, and it is **also
>    superseded**.
>
> **What is true now (0.2.0 onward):** the hot ops — elementwise
> binary/unary, matmul, conv, reduce, transpose — **dtype-dispatch native
> f32 shaders as well as f64**. f32 is no longer cast up. f64 remains the
> default *accumulator* policy for matmul/conv (correctness first;
> consumer GPUs are slow at f64), switchable with
> `Nx.Vulkan.VulkanoBackend.put_f32_matmul_accumulator(:f32)`.

### f64 TRANSCENDENTALS COMPUTE AT f32 PRECISION

The most important thing in this section, and until 2026-08-28 it was recorded
only in a shader comment.

`Nx.exp`, `Nx.log`, `Nx.tanh` and `Nx.sigmoid` on an `{:f, 64}` tensor return a
GPU-resident `{:f, 64}` result whose VALUE carries only f32 accuracy. Measured
against `:math` at x = 1.5, relative error:

| op | relative error | f64 would give |
|---|---|---|
| `Nx.log` | 1.18e-7 | ~1e-16 |
| `Nx.exp` | 2.10e-8 | ~1e-16 |
| `Nx.tanh` | 1.56e-8 | ~1e-16 |
| `Nx.sigmoid` | 5.22e-9 | ~1e-16 |

**Eight orders of magnitude**, with no signal in the type. `Nx.type/1` says
`{:f, 64}`, the tensor is on the device, and nothing raises.

The cause is in `glsl/elementwise_unary_f64.comp`, stated plainly there:

> GLSL builtins (exp, log, tanh) lack f64 overloads. Cast through float for
> transcendentals — precision limited to f32 for these ops, but arithmetic
> stays in full f64.

`float64_t log_f64(float64_t x) { return float64_t(log(float(x))); }`. SPIR-V's
`GLSL.std.450` genuinely has no 64-bit `Exp`/`Log`/`Tanh`, so the alternatives
are a hand-written double-precision polynomial per op (a project, not a task) or
a host fallback (a round trip per call). The cast is a deliberate third choice.

**`sigmoid` is a different case from the other three, and the distinction is
testable.** For `log`, `exp` and `tanh` the f64 result is EXACTLY the f32 result
widened — compute the op at `{:f, 32}`, convert to f64, and it compares equal,
for every value tried on three architectures. `sigmoid` is not: the shader is

    case 5: y = 1.0LF / (1.0LF + exp_f64(-x));

so the add and the reciprocal are genuine f64 and only the `exp` inside casts
through `float`. It INHERITS f32 accuracy through `exp` rather than being an f32
value widened, and that is why its relative error (5.2e-9) is smaller than
`exp`'s own (2.1e-8) — the sigmoid derivative attenuates it.

Worth stating because "everything here is just f32 widened" is the natural
reading of the paragraph above, and it is wrong for one of the four. It was
written that way here until the fleet checked each op separately instead of
generalising from `log`.

**What is NOT true, and was claimed in README.md until 2026-08-28:** that these
"host-fall-back rather than silently losing precision". They do neither of those
things. `Nx.Vulkan.Codegen`'s `@f64_unsafe_ops` excludes them from FUSION — that
part is real — but the evaluator then dispatches them EAGERLY to the shader
above, so the fused and eager paths return the identical f32-precision value.
Verified: `Nx.log(f64 2.0)` gives `0.6931471824645996` through both paths, where
Erlang gives `0.6931471805599453`.

**The clearest way to see it: the same call on another backend.** Measured on
the Jetson, the one box in this project's fleet carrying both backends. Same op,
same dtype, x = 1.5, f64:

| op | this backend | EXLA (XLA-CPU) |
|---|---|---|
| `Nx.log` | 1.18e-7 | **0.0** — bit-identical to `:math.log/1` |
| `Nx.exp` | 2.10e-8 | **0.0** |
| `Nx.tanh` | 1.56e-8 | 1.23e-16 (1 ulp) |
| `Nx.sigmoid` | 5.22e-9 | **0.0** |

`Nx.type/1` reports `{:f, 64}` in every cell. Nine orders of magnitude, same
call, same dtype, only the backend differs.

**The figures above are hardware-independent.** Verified on three
architectures — Ampere (RTX 3060 Ti), Kepler (GT 650M and GT 750M) and Maxwell
(Tegra X1) — agreeing not merely to the printed precision but bit-for-bit on the
raw doubles. That follows from the mechanism: the loss is imposed by the shader
casting to `float`, and every driver's f32 `exp`/`log`/`tanh` lands on the same
value.

**Consequences worth knowing.** Anything built on these inherits the loss —
a Normal log-density's `log(sigma)` term is f32-accurate however carefully the
rest is written. The error is not a constant, so unlike a fixed offset it does
NOT cancel in a log-ratio. If you need true f64 transcendentals, compute them on
`Nx.BinaryBackend` and transfer.

The ops with no f64 form at all — `sin`, `cos`, `atan2`, `erf`, `log1p`,
`expm1`, `cbrt`, `rsqrt` and the rest — are a different case: those genuinely do
host-fall-back, and are the 40 doctests `NEXT.md` §1.3 files as decided.

---

**What's true**: compute shaders exist in **both f32 and f64**, selected
by tensor dtype. Storage round-trips any numeric type (f32, f64, s8..s64,
u8..u64). `Exmc.JIT.precision()` returns `:f64` for the Vulkan path (EMLX,
the f32-only backend, was dropped). Both `*_f32.spv` and `*_f64.spv`
inventories ship in `priv/shaders/`.

---

## 2. Host-fallback operators

The following backend callbacks **always** host-materialize. Each is
correct but pays the GPU↔host round-trip on every call.

> **The rows marked ✅ below are done** and were re-verified by counting
> fallbacks, not by reading this file. It said `pad/4` had "no shader" for a
> year after `pad.comp` shipped, which is part of why `pad` kept
> host-falling-back on a mistyped literal with nobody noticing (T11). Trust
> `test/nx_vulkan/fallback_test.exs` over this table.

| Callback | Why host | Proper fix |
|---|---|---|
| `concatenate/3` | ✅ done both ways — `axis == 0` is a buffer append needing no shader (word-copyable types only), `axis > 0` uses `concat_nd.spv` (`c9b1a31`). **The sub-word restriction is dtype-wide, not axis-0-specific** — `concat_nd` carries the same gate, so s8/u8/s16/u16/f16/bf16 fall back on every axis | sub-word dtypes, all axes |
| `stack/3` | ✅ done — reshapes and delegates to `concatenate/3`; resident at any axis for word-copyable types. Reaches `stack_host_fallback` only when coercion fails | sub-word dtypes, inherited from `concatenate/3` and attributed to it |
| `pad/4` | ✅ done — `glsl/pad.comp`, 4/8-byte dtypes, rank ≤ 4, edge/interior/negative configs | — |
| `slice/5` | ✅ done — `glsl/slice.comp`, static starts, 4/8-byte dtypes, rank ≤ 4 | — |
| `put_slice/4` | ✅ done — `glsl/put_slice.comp` overlay, 4/8-byte dtypes, rank 1-4 (rank 0 raises in `BinaryBackend`, so it stays host) | — |
| `gather/4` | ✅ done — `glsl/gather.comp` for the leading-prefix / default-axes case | other axis sets |
| `indexed_put/5` | ✅ done — `scatter.spv`, and `scatter_ordered.spv` when duplicate indices must resolve last-wins (`NXV_SCATTER_ORDERED=0` opts out). Index rank 2, k ≥ 1, target rank 1–4, 4/8-byte dtypes, 4-byte index type; non-prefix axes rotate | rank 5+; 1/2-byte dtypes |
| `indexed_add/5` | ✅ done — same path, op code ≠ 0. Narrower than `indexed_put` by one condition: **4-byte integer dtypes only**, since the accumulate is an integer atomic | float and 8-byte accumulate (needs float atomics) |
| `iota/3` | ✅ on-device (no shader needed) | — |
| `eye/2` | tiny | trivial; not bandwidth-bound |
| `broadcast/4` | ✅ done — `glsl/broadcast_nd.comp`, rank ≤ 4, any 4/8-byte dtype (W1) | rank 5+, 1/2-byte dtypes |
| `transpose/3` (rank ≥ 3) | ✅ done — `glsl/transpose_nd.comp`, rank ≤ 4, any 4/8-byte dtype (W1) | rank 5+, 1/2-byte dtypes |
| `select/4` | ✅ done — `glsl/select_f{32,64}.comp`, rank 0-4 (rank 0 since T11) | rank 5+, non-f32/f64 |
| `clip/4` | ✅ done — composes `Nx.min(Nx.max(t, lo), hi)` from the broadcast max/min shaders and never leaves the device. Resident on u8, s8, u16, s16, u32, s32, f32, f64. **Mixed-type bounds do NOT fall back** — Nx promotes them before the callback | `{:s,64}`, where the composed max/min have no kernel |
| `log1p/2` | no shader | extend `elementwise_unary` op 15 |
| `is_infinity/2`, `is_nan/2` | ✅ done — `@predicate_unary_ops` codes 10 and 9 (W5), f32/f64 | non-float dtypes |
| `right_shift/3`, `left_shift/3`, `remainder/3`, `quotient/3` | ✅ done — `@binary_ops` codes 12, 13, 8, 7 into `elementwise_binary_{s32,u32}.spv`. Integer-only by Nx contract, and `binary_spv/2` refuses to pair them with a float shader (whose `default:` arm would return a silent 0.0) | 64-bit ints host. (Narrow ints s8/u8/s16/u16 are already resident, widening through s32 with the dtype preserved — done, not pending) |
| `bitwise_and/3`, `bitwise_or/3`, `bitwise_xor/3` | ✅ done — `@binary_ops` codes 9, 10, 11, same integer shaders and the same refusal | 64-bit ints host; narrow ints already resident via s32 |
| `less_equal/3`, `greater_equal/3`, `not_equal/3` | ✅ done — `glsl/compare_f{32,64}.comp` spec constant, rank 0-4 (rank 0 since T11) | rank 5+, non-f32/f64 |
| Per-axis reduction over **multiple** axes | ✅ done for `sum`/`product`/`reduce_max`/`reduce_min`, **contiguous or not**, rank ≤ 4: `reduce_via_transpose/5` rotates the kept axes to the front, which turns any axis set into a trailing-suffix reduce (this is what the conv bias gradient `sum(axes: [0,2,3])` needs). `all`/`any` do NOT have this branch — `do_allany/6` gates on `classify_reduce_axes/2` alone | `{:s,64}`/`{:u,64}`; rank 5+; `all`/`any` on non-contiguous sets |
| Linear algebra: `determinant`, `solve`, `cholesky`, `triangular_solve` | host BinaryBackend | LU/Cholesky shader (only wins at d ≥ 256, irrelevant for MCMC) |
| `sort/3`, `argsort/3` | not implemented | `bitonic_sort.comp` |
| `atan2/3` | host — the only member of `@host_fallback_binary_ops`. A genuine two-argument transcendental, and GLSL.std.450 has no f64 form | — |
| `argmax/3`, `argmin/3` | ✅ done — `argreduce_{f32,f64,s32,u32}.spv`, contiguous axis run, 4-byte integer out type; `tie_break: :high` is a spec-constant offset | s8/s16/u8/u16/s64/u64/f16/bf16 inputs. (**Not** non-contiguous axis sets: `Nx.argmax/2` takes `:axis` singular and raises on `:axes`, so that condition is unreachable through the public API — it was listed as future work that cannot be reached) |
| `all/3`, `any/3` | ✅ done — `allany_{f32,f64,s32,u8}.spv` (u32 reuses the s32 kernel — the test is against zero, so signedness cannot reach the answer), contiguous axis run, `{:u,8}` out. The **one** call site needing `buf_alloc_zeroed/1`: the shader `atomicOr`s one thread per slot | s8/s16/u16/s64/u64/f16/bf16 inputs; non-contiguous axis sets — **`do_allany/6` lacks the transpose-rotation branch that `do_reduce/5` has**, which is the whole difference |
| `product/3` | ✅ done — `reduce_axis_*.spv` op code 3, contiguous axis run; type pairs f32→f32, f64→f64, s32→s32, u32→u32, u8→u32, s8→s32, s16→s32, u16→u32. **Non-contiguous axis sets work** — `do_reduce/5`'s second arm rotates kept-axes-first via an on-GPU transpose (`reduce_via_transpose/5`) | `{:s,64}`/`{:u,64}`; rank 5+ |
| `conv/4` | ✅ done — im2col + GEMM, f32/f64, both directions | grouped conv, rank 5+ |
| `window_*/{4,6}` | ✅ done for non-overlapping windows, both directions | overlapping backward (needs float atomics) |
| `lu/3`, `qr/3` | not implemented | host fallback acceptable for MCMC sizes |

**The broadcast shader is wired.** This paragraph used to say it was the
highest-impact missing piece — that the backend fell back to host whenever
`a.shape != b.shape`. It dispatches: `gpu_bcast_binary/5` at line 538 into
`elementwise_binary_bcast_{f32,f64,s32,u32}.spv`.

---

## 3. Fusion (Path A) limits

`Nx.Vulkan` fused_chain/3 and the `Nx.Vulkan.Fuse.fuse/1` macro share
the same constraints, inherited from `fused_elementwise.spv`:

- **Two input buffers only.** Op chain operates on `a` (running
  register) and `b` (second operand for binary steps). A third tensor
  `c` cannot participate. `Nx.add(Nx.multiply(a, b), c)` doesn't fuse.
- **f64.** Same as the rest of compute (see §1).
- **Same shape only.** No broadcast within a chain. `a` and `b` must
  match.
- **Up to 8 ops per dispatch.** Longer chains must be split (the user
  can manually compose two `fused_chain` calls; the macro doesn't yet).
- **No reductions in chain.** A chain that ends with `Nx.sum/2` doesn't
  fuse — sum is not in the fused shader's switch. Workaround: fuse the
  elementwise prefix, then dispatch `reduce_axis` separately.
- **No scalar literals.** `Nx.add(a, 1.0)` doesn't fuse — the macro
  expects `b` as a real tensor variable. Workaround: pre-build `b` as a
  constant tensor.
- **erf/expm1 in chains require spirit `161296d1` or later.** Earlier
  fused.spv had op codes 113/114 assigned but the switch fell through.

---

## 4. The `Nx.Vulkan.Fuse` macro shortcuts

The macro is a v1 demonstration of Path A.2; the proper auto-detector
(v2) is a real `Nx.Defn.Compiler`.

| Limit | Why | Workaround |
|---|---|---|
| 2-arg functions only | macro signature is `fuse(fn a, b -> ... end)` | wrap n-arg fns with explicit `fused_chain` |
| Linear chain only | macro walks one nested-call path; no branching | split the function |
| `b` must literally be the second arg of every binary op | macro doesn't reorder | rewrite the body to canonical form |
| Output is `{:ok, ref}`, not an `%Nx.Tensor{}` | doesn't roundtrip cleanly with non-fused code | use within a Vulkan-only flow |
| **No autograd integration** | Fuse output isn't a `Nx.Defn.Expr` node | use Nx.Defn.Grad against the unfused version |
| **Doesn't fuse inside `defn`** | macro operates on plain Elixir AST, not defn IR | manual `fuse` on defn body, or wait for v2 |

**v2 plan**: implement `Nx.Vulkan.Compiler` that satisfies the
`Nx.Defn.Compiler` behaviour. It walks the defn IR (a tree of
`%Nx.Defn.Expr{}` nodes via `Nx.Defn.Tree`/`Nx.Defn.Composite`),
identifies chains, replaces them with synthetic `:fused_chain` nodes,
then evaluates. Multi-day work; the Evaluator source is ~500 lines and
the IR walking has its own cache/refcount system.

---

## 5. `Nx.Defn` integration shortcuts

`Nx.Vulkan.jit/2` uses `Nx.Defn.Evaluator` rather than a custom
compiler. Three consequences:

1. **No fusion** — every Nx call inside a defn is one shader dispatch.
   Fixed for explicit chains by Path A.2 v1; auto-detect waits for v2.
2. **Mutates global state** — `jit/2` calls
   `Nx.global_default_backend(Nx.Vulkan.Backend)` if not already set.
   In mixed-backend test suites, callers must save/restore. The
   nx_vulkan test suite does this; user code should follow the same
   pattern.
3. **No graph caching** — Evaluator re-walks the IR per call. EXLA's
   `__compile__` caches an HLO module keyed by the function and arg
   shapes; we don't. For repeated calls with identical shapes (every
   MCMC step), this is the same setup cost on every dispatch.

---

## 6. Exmc test suite under `:vulkan`

Two honest numbers:

- **Targeted subset** (`exmc_test`, `dist_test`, `diagnostics_test`,
  `compiler_test` = 4 files, 63 tests + 11 doctests): **91.9% pass**,
  7.1s wall time. Above the 80% target the port plan projected.
- **Full suite**: hangs past 60min in property-test files. Not
  failing — actively executing through the slow path. A NUTS leapfrog
  has ~30 elementwise ops; under host fallback that's ~30 round-trips
  per step × thousands of steps × hundreds of property variations.

**Phase 4 (benchmark vs EXLA-CUDA) cannot run honestly until either:**
- The fused chain auto-detect (Path A.2 v2) lands and exmc's
  leapfrog dispatches one fused shader per chain, or
- The remaining missing-shader gaps (broadcast, slice, gather, etc.)
  are wired so the host-fallback rate drops to near zero.

Without one of those, a benchmark would measure round-trip overhead,
not actual GPU compute.

---

## 7. Concurrency

`SUBMIT_LOCK: Mutex<()>` in `lib.rs` serializes **every** Vulkan
submit globally. This is the conservative correctness guarantee
established in adversarial round 2 (zero DEVICE_LOST under 100 BEAM
procs hammering the queue).

**Cost**: with N concurrent NIF calls, only one is dispatching at a
time; the others wait. For a 4-core MCMC run with 67 instruments, the
queue depth can be 67 jobs deep; each one takes its turn.

**Proper fix (PERSISTENT_BUFFERS_PLAN.md)**: pre-record command
buffers per pipeline + multiple submit queues. Spirit's backend was
designed for this; the hookup hasn't been done. Estimated 1.5× to 4×
improvement in throughput-bound workloads.

---

## 8. Test coverage gaps

- **No fuzz/property tests for `Fuse` macro**. The AST walker has
  edge cases (binary ops where `b` is the first arg, chains broken
  by intermediate vars, etc.) that aren't exercised.
- **No stress benchmark for `fused_chain`**. mac-248's
  `bench_fused.cpp` measured 1.6–4× speedup at the C++ level but the
  Elixir wrapper hasn't been benchmarked end-to-end.
- **No mixed-backend tests**. We don't test scenarios where
  Vulkan and EXLA tensors coexist in the same defn (would surface
  the "two incompatible tensor implementations" error class that
  appeared in the Phase 3 long tail).
- **No FreeBSD test for the `Fuse` macro path**. The macro is pure
  Elixir + AST, so it should work, but cross-host parity isn't
  verified for it.

---

## 9. Build / dev friction

- **`build.rs` shader copy** triggers on `cargo:rerun-if-changed=<dir>`
  for the spirit shaders directory. New shaders appearing trigger a
  recopy *after* a rust source touch. If you add a shader and rebuild
  without touching anything Rust, the .spv may not propagate. Manual
  `cp` is the workaround.
- **Pipeline cache** in `nx_vulkan_shim.cpp` is process-global. There's
  no way to clear it short of `nxv_destroy()` which tears down the
  whole context. For dev workflows that hot-reload Elixir without
  restarting the BEAM, stale pipelines (e.g., after a shader update)
  require a BEAM restart.
- **`Nx.global_default_backend`** mutation in `Nx.Vulkan.jit/2`
  bleeds across processes. Callers in tests must save/restore. We
  documented this; we didn't fix it. A clean fix would be a per-call
  backend override option that Nx may not currently support.

---

## 10. Out of scope (deferred to v0.2 or later)

- **GPU passthrough into FreeBSD jails.** Bare-metal FreeBSD is the v1
  target. zed roadmap tracks the jail-GPU work separately.
- **Multi-GPU.** Picks device 0. Spirit has the API; we don't expose
  device selection from Elixir yet.
- **fp16 / bf16.** No mixed-precision compute. f16 would halve memory
  for the trader's per-instrument inference but is not on the critical
  path.
- **Forward-mode autograd.** `Nx.Defn` supports it; we'd inherit it via
  the same path as reverse-mode, but no consumer has asked.
- **Dynamic shape support.** Buffer sizes are bound at upload time. A
  real `Nx.Defn.Compiler` with shape polymorphism is v0.3+ work.
- **Symbolic differentiation of fused chains.** Each fused chain is
  opaque to Nx.Defn.Grad. The full IR-rewrite compiler (Path A.2 v2)
  would need a backward-pass plan for fused nodes.

---

## What we have, despite all of the above

- **Three-host parity**: Linux RTX 3060 Ti, FreeBSD GT 750M, FreeBSD
  GT 650M all run 112/0 tests on `main`. Same shaders, same Elixir
  code, three GPU generations and two operating systems.
- **Phase 1 complete**: every callback the EXMC sampler reaches for is
  implemented. Forward pass and Nx.Defn.Grad backward pass work.
- **Path A demonstrated**: 1.6–4× shader-level speedup measured by
  mac-248; user-facing API and macro both shipping.
- **Honest about the gaps**: this document.

---

## Suggested next-up priorities

In rough order of leverage:

1. **Wire `elementwise_binary_broadcast.spv`** into `do_binary`. Closes
   the broadcast-driven host-fallback bucket. Spirit already has the
   shader; the C shim, NIF, and `do_binary` dispatch arm are the only
   missing pieces. Half a day of work.
2. **Path A.2 v2** — proper `Nx.Defn.Compiler` with chain detection
   inside any defn block. Multi-day work but unblocks the full exmc
   suite under `:vulkan`. The IR walk pattern is well-mapped in the
   Evaluator source; this is engineering, not research.
3. **Pre-recorded command buffers** for hot-path shaders (matmul,
   reduce, fused chain). PERSISTENT_BUFFERS_PLAN.md scopes the work.
   1.5–4× throughput win on the trading-style concurrent workload.
4. **Phase 4 benchmark** once 1 or 2 lands — `quick_bench.exs`
   head-to-head: EXLA-CUDA vs `:vulkan` on Linux RTX 3060 Ti, then
   `:vulkan` on FreeBSD GT 750M. The first cross-platform GPU number
   for exmc.
