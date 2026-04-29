# exmc → Nx.Vulkan port plan

**Status:** plan, April 2026.
**Predecessor:** Nx.Vulkan v0.0.7 + adversarial round 2 done
(see [PLAN.md](PLAN.md) and the
[*What the Mutex Saved*](https://www.dataalienist.com/blog-what-the-mutex-saved.html)
blog post). 7 shaders, 42 tests, no DEVICE_LOST under
load. The wrapper survives.
**Goal:** run `exmc` (Igor's Bayesian/MCMC trading framework, currently
~250 tests on EXLA-CUDA) on FreeBSD via Vulkan, where EXLA does not
build. The migration is the consumer-driven requirements doc for
Nx.Vulkan v0.1 — every gap surfaced here is a Vulkan iteration.

---

## What exmc actually uses

A grep over `lib/exmc/` enumerates the Nx surface exmc touches:

```
abs add as_type axis_size backend_copy broadcast clip concatenate
divide dot equal erf exp expm eye from_binary indexed_put iota less
log logsumexp max mean min multiply negate pow put_slice rank
reduce_max reshape rsqrt select shape sigmoid size slice
slice_along_axis sqrt squeeze stack standard_deviation subtract sum
t take_diagonal tanh tensor to_binary to_flat_list to_number
transpose type
```

Plus a handful of compiler-level features:

  - `Nx.Defn.Compiler.value_and_grad/2` — autograd for log-prob gradients
  - `defn` / `defnp` — graph compilation
  - `Nx.LinAlg.determinant/1`, `Nx.LinAlg.solve/2` — for dense mass matrix
  - Per-axis reductions (mass matrix Welford updates)
  - f32 default; f64 for the mass matrix accumulator path

51 distinct `Nx.*` calls. Roughly 30 lines of `defn` per hot kernel
(leapfrog, log_prob, gradient steps).

---

## Coverage matrix: Nx.Vulkan v0.0.7 vs exmc requirements

| Bucket | Nx.Vulkan today | exmc needs | Gap |
|---|---|---|---|
| Elementwise binary | add/mul/sub/div/pow/max/min ✅ | Same + `equal`, `less` (comparisons) | comparisons |
| Elementwise unary | exp/log/sqrt/abs/neg/sigmoid/tanh/relu/ceil/floor/sign/reciprocal/square ✅ | Same + `erf`, `expm` (expm1?), `rsqrt` | erf, expm1, rsqrt |
| Reductions | sum/min/max/mean (all-axis) ✅ | sum/mean/max + per-axis variants + `standard_deviation`, `logsumexp` | **per-axis reductions** (the big one) |
| Linear algebra | matmul (naive + tiled) ✅ | dot, t (transpose), `take_diagonal`, `Nx.LinAlg.{determinant, solve}` | **transpose, det, solve** |
| Indexing | none | `slice`, `slice_along_axis`, `put_slice`, `indexed_put`, `squeeze`, `take_diagonal` | **all of indexing** |
| Reshape | none | `reshape`, `broadcast`, `concatenate`, `stack` | **all of reshape** |
| Construction | `from_binary` ✅ | `tensor`, `iota`, `eye` | iota, eye |
| Comparison | none | `equal`, `less`, `clip`, `select` | **all of comparison** |
| Type | f32 only | f32 + f64 (mass matrix), `as_type` | **f64 + casts** |
| Autograd | **none** | `Nx.Defn.Compiler.value_and_grad/2` | **the killer** |
| Random | uniform + normal ✅ | uniform + normal | ✅ |

**Honest tally:** Nx.Vulkan covers ~30% of exmc's Nx surface. The
other 70% is mostly mechanical (per-axis reductions, indexing,
reshape, comparisons), but autograd is structural.

---

## The autograd question

exmc's NUTS sampler computes log-probability gradients on every
leapfrog step. That gradient is what the symplectic integrator
follows; without it, no MCMC. The gradient computation is currently
done via `Nx.Defn.Compiler.value_and_grad/2`, which EXLA implements
through XLA's reverse-mode AD pass.

Nx ships its own backend-agnostic AD transform — `Nx.Defn.Grad` —
that operates on the IR Nx.Defn produces, not on the backend's
native ops. **The good news:** if we implement enough of the
forward-pass operators on Nx.Vulkan, `Nx.Defn.Grad` runs over the
same IR and produces backward-pass IR that the same backend can
execute. We don't need to write a Vulkan-specific autograd; we
need to make sure the backward pass's operator set is also covered.

**The constraint that emerges:** every op exmc uses in its forward
pass must also work in its `grad`-flipped backward form. Some
backward forms add operators the forward doesn't:

  - `Nx.exp(x)` forward → `Nx.exp(x) * grad_y` backward (no new ops)
  - `Nx.sum(x)` forward → `Nx.broadcast(grad_y, shape_x)` backward (needs broadcast)
  - `Nx.dot(A, B)` forward → `dot(grad_y, B.T)` and `dot(A.T, grad_y)` backward (needs transpose)
  - `Nx.slice(x, ...)` forward → `Nx.put_slice(zeros, grad_y, ...)` backward (needs put_slice + iota for zeros)

So implementing transpose + put_slice + broadcast is necessary
to close autograd, even if exmc never directly calls them.

---

## Phased migration

### Phase 1 — surface gap-filling (Nx.Vulkan v0.1)

Implement the missing Nx.Backend callbacks in dependency order
(later phases depend on these). Each is a separate iteration; most
are mechanical.

| Priority | Iteration | Ops | Why first |
|---|---|---|---|
| 1.1 | comparisons | `equal`, `less`, `greater`, `select`, `clip` | Used in `clip` for HMC step-size, in `clamp` for stability |
| 1.2 | reshape | `reshape`, `broadcast`, `transpose` | Backward pass for sum/dot needs these |
| 1.3 | slicing | `slice`, `slice_along_axis`, `put_slice`, `squeeze` | NCP unpack; gradient slicing |
| 1.4 | per-axis reductions | `sum/2`, `reduce_max/2`, `mean/2` with `axes:` opt | Mass matrix Welford; per-feature ops |
| 1.5 | construction | `iota`, `eye`, `tensor/2` proper | `eye` for mass matrix init; `iota` for index ops |
| 1.6 | indexing | `take_diagonal`, `indexed_put` | Mass matrix diagonal ops |
| 1.7 | transcendentals | `erf`, `expm1` | Probit / log-Normal CDFs |
| 1.8 | f64 + casts | `as_type`, dtype dispatch | Mass matrix accumulator precision |
| 1.9 | dense linalg | `determinant`, `solve` | Dense mass matrix path; non-trivial |

Effort: ~1.5–2 person-months. Most of these are small (one
shader + one backend callback + tests), but f64 (1.8) and dense
linalg (1.9) are days each.

Each iteration can land independently. exmc-on-Vulkan stays
broken until the **autograd-prerequisite set** (1.2, 1.3, 1.4) is
in. The other gaps are progressive; exmc gracefully falls back
to BinaryBackend for missing ops via Nx's auto-conversion.

### Phase 2 — exmc.JIT integration (parallel to phase 1)

Add `Nx.Vulkan` as a fourth backend in `Exmc.JIT`. The module
already abstracts over `EXLA / EMLX / BinaryBackend`; the new
case is one config flag.

```elixir
# config/runtime.exs
config :exmc, :compiler, :vulkan   # was :exla, :emlx, :binary_backend

# Exmc.JIT module already has:
def jit(fun, args) do
  case compiler() do
    :exla -> EXLA.jit(fun).(args)
    :emlx -> EMLX.jit(fun).(args)
    :binary_backend -> apply(fun, args)
    :vulkan -> Nx.Vulkan.jit(fun).(args)   # NEW
  end
end
```

`Nx.Vulkan.jit/1` is a small helper that ensures the default
backend is set, evaluates the defn, and falls back per-op when
needed. **No code change in exmc's defn functions** — just the
config flag.

### Phase 3 — incremental coverage (the long tail)

Run `mix test` in exmc with `compiler: :vulkan`. Each test that
fails because Nx.Vulkan doesn't yet support an op points to the
next iteration. Sort by frequency: ops that fail many tests get
fixed first.

The expected order (informed by exmc's current usage patterns):

  - **Sampler core:** leapfrog + log_prob + gradient — covered by phase 1
  - **Distribution log-probs:** Normal, Lognormal, HalfCauchy, etc. — mostly elementwise + erf
  - **Initialisation:** transformer pass + bijector chain — slice + reshape
  - **Mass matrix:** Welford + Cholesky — per-axis reductions + linalg
  - **WAIC/LOO model comparison:** logsumexp + per-axis reductions

Estimated drop-out point: **~80% of exmc tests pass on
Nx.Vulkan after phase 1 + 2 land.** The 20% that don't are mostly
high-precision linalg (LU/Cholesky) where the f64 path isn't
critical for live trading but matters for correctness tests.

### Phase 4 — performance comparison + acceptance

Run the `benchmark/quick_bench.exs` suite (5 seeds × 3 model
sizes) on three configurations:

  | Config | Hardware | Backend |
  |---|---|---|
  | A | Linux, RTX 3060 Ti | EXLA on CUDA |
  | B | Linux, RTX 3060 Ti | Nx.Vulkan |
  | C | FreeBSD, Mac Pro | Nx.Vulkan |

Acceptance: **B should be within 2× of A** on the hot path
(leapfrog steps/sec). C is the new substrate; expect 1–2× of A on
the same hardware (FreeBSD's NVIDIA driver path adds some host
overhead).

If B is much slower than A, the gap is one of:
  - Per-call NIF overhead (cured by persistent pipeline cache,
    already done; verify via profile)
  - Per-leapfrog command-buffer recording (cured by
    pre-compiled command buffers, optimisation iter v0.2)
  - Nx.Defn graph fragmentation (each Nx call = one shader
    dispatch; XLA fuses many ops into one kernel — Nx.Vulkan
    can't fuse; this is the **fundamental performance gap** vs
    EXLA)

---

## Concrete iteration order

Six weeks, single developer, sequential:

| Week | Phase | Deliverable |
|---|---|---|
| 1 | 1.1 + 1.2 | comparisons + reshape/broadcast/transpose |
| 2 | 1.3 + 1.4 | slicing + per-axis reductions |
| 3 | 1.5 + 1.6 + 1.7 | construction + indexing + erf/expm1 |
| 4 | 1.8 + 2 | f64 + Exmc.JIT integration |
| 5 | 1.9 + 3 | dense linalg + run exmc test suite, log fails |
| 6 | 3 + 4 | close remaining failures + benchmark |

**Critical path:** weeks 1–4. By end of week 4, exmc's
non-mass-matrix tests should run on Nx.Vulkan (with the f64 mass
matrix path falling back to BinaryBackend or running f32 with
known precision loss). End of week 6 = full coverage + a
benchmark number.

---

## Out of scope (deferred)

- **GPU passthrough into FreeBSD jails.** The
  [zed demo plan](https://github.com/borodark/zed/blob/main/specs/demo-cluster-plan.md)
  flagged this: jails can't pass through NVIDIA GPUs cleanly.
  exmc-on-Vulkan-on-FreeBSD-on-bare-metal is the v1 deploy
  target. exmc-on-Vulkan-in-a-FreeBSD-jail awaits the
  jail-GPU-passthrough work.
- **Kernel fusion.** XLA fuses a chain of Nx ops into one CUDA
  kernel; we can't yet. Per-op dispatch through Vulkan adds
  ~10–50µs each. exmc's leapfrog has ~30 ops per step; the
  fusion gap is potentially 0.3–1.5ms per leapfrog. Address
  in v0.2 with a custom `Nx.Defn.Compiler` that emits combined
  shaders for known patterns.
- **Multi-GPU.** Nx.Vulkan picks device 0. Multi-device
  selection is a one-day iteration when the demand surfaces.
- **Mixed precision (fp16, bf16).** Nx.Vulkan is f32 + f64
  (after 1.8). fp16 would be useful for the trader's per-instrument
  inference (256MB of f32 → 128MB of f16) but isn't on the
  critical path.

---

## What we get

If phase 1–4 lands as planned, exmc gains:

1. **A FreeBSD GPU runtime.** The first one. The trader's existing
   architecture — per-instrument GenServer + GPU scheduler — stays
   intact; only the JIT backend changes.
2. **A second performance reference for the project.** EXLA-on-CUDA
   has been the only GPU baseline; Nx.Vulkan-on-NVIDIA gives us a
   second number to compare against, validating that
   exmc's perf characteristics are hardware-agnostic and not
   accidentally CUDA-pinned.
3. **Portability dividend.** The same exmc binary runs on Linux
   NVIDIA, FreeBSD NVIDIA, AMD via Mesa RADV, Intel iGPU via ANV,
   Apple Silicon via MoltenVK. Each platform was previously a
   separate engineering question; under Nx.Vulkan they are one.

---

## What we don't get

- **EXLA-class fusion.** The 30-op leapfrog dispatches 30 times.
  XLA dispatches once. This is ~5–10× perf gap on small models;
  smaller on large models (where compute > dispatch overhead).
- **Forward-mode autograd.** Nx.Defn supports it; Nx.Vulkan would
  inherit it via the same path as reverse-mode. Not currently
  needed by exmc.

---

## Cross-references

- [PLAN.md](PLAN.md) — Nx.Vulkan v0.1 master roadmap
- [PERSISTENT_BUFFERS_PLAN.md](PERSISTENT_BUFFERS_PLAN.md) — the
  persistent-pipeline + buffer-pool optimization that mitigates
  the dispatch-overhead gap
- [SHADERS_PLAN.md](SHADERS_PLAN.md) — shader inventory, gates
  some of phase 1 here (per-axis reductions need a new shader,
  not just a backend callback)
- exmc repo: `~/projects/learn_erl/pymc/exmc/`
- Nx documentation: <https://hexdocs.pm/nx/Nx.Backend.html>
- Nx.Defn.Grad source: <https://github.com/elixir-nx/nx/blob/main/nx/lib/nx/defn/grad.ex>
