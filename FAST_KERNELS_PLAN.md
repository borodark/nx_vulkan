# Nx.Vulkan.Fast — explicit named-kernel approach

**Status**: plan, May 2026.
**Trigger**: research into Emily's `Emily.Fast` pattern (in
`~/projects/learn_erl/emily/lib/emily/fast.ex`) revealed a cleaner
architecture for fused dispatch than IR-level auto-detection.

## Approach change in one paragraph

We've been building `Nx.Vulkan.Compiler` to walk Nx IR and detect
fusable patterns automatically (left-folded, right-folded, 1-arg,
2-arg, 3-arg, 4-arg, …). It works but doesn't scale: each new pattern
is more compiler code, false negatives are silent, and the matched
shapes drift from real exmc usage. Emily takes the opposite approach:
provide named functions like `Emily.Fast.layer_norm` that emit
`Nx.Defn.Expr.optional/3` IR nodes. At eval time the evaluator
dispatches the named callback if the active backend has it, otherwise
runs the defn fallback. **No IR walking. No false negatives. Trivial
to add new kernels.** Emily explicitly skipped IR-level fusion
("measured the fusion win at <1.2× on transformer-shaped workloads —
below the threshold that justified the integration cost") and ship
explicit Fast kernels instead.

## What we ship

`Nx.Vulkan.Fast` module mirroring the `Emily.Fast` shape:

```elixir
def leapfrog_position(q, eps, p) do
  Nx.Defn.Expr.optional(
    :fast_leapfrog_position,
    [q, eps, p, []],
    &leapfrog_position_fallback/4
  )
end

defp leapfrog_position_fallback(q, eps, p, _opts) do
  Nx.add(q, Nx.multiply(eps, p))
end
```

`Nx.Vulkan.Backend` exports `fast_leapfrog_position/4` that calls
`Nx.Vulkan.fused_chain_4` with `[multiply: 1, add: 2]`. The fallback
runs on EXLA, BinaryBackend, EMLX, anywhere — same Nx primitives,
correct everywhere.

## Initial kernel set (priority order by hotness in NUTS)

| # | Kernel | Defn fallback | Fused dispatch |
|---|---|---|---|
| 1 | `leapfrog_position(q, eps, p)` | `q + eps * p` | `fused_chain_4 [multiply:1, add:2]` |
| 2 | `leapfrog_momentum_half(p, half_eps, grad)` | `p + half_eps * grad` | `fused_chain_4 [multiply:1, add:2]` |
| 3 | `momentum_step(p, eps, grad)` | `p + eps * grad` | `fused_chain_4 [multiply:1, add:2]` |
| 4 | `inv_mass_apply(p, inv_mass)` | `p * inv_mass` | trivial — keep just for naming clarity |
| 5 | `kinetic_energy(p, inv_mass)` | `0.5 * sum(p² * inv_mass)` | needs reduce in chain — needs new shader |
| 6 | `normal_logpdf(x, mu, sigma)` | full Gaussian formula | needs new shader (mac-248) |

Shipping 1–4 first (all map to existing `fused_chain_4`); 5–6 are
follow-ups that potentially need new shaders.

## Work split

### Linux (this session)

1. Branch `feat/fast-kernels` ✓
2. Create `lib/nx_vulkan/fast.ex` with kernels 1–4, each as
   `Nx.Defn.Expr.optional/3` + a defn-style fallback.
3. Add backend callbacks `fast_leapfrog_position/4`,
   `fast_leapfrog_momentum_half/4`, `fast_momentum_step/4`,
   `fast_inv_mass_apply/3` to `Nx.Vulkan.Backend`. Each dispatches
   `Nx.Vulkan.fused_chain_4`.
4. Tests: end-to-end correctness + verify the optional path fires
   under `Nx.Vulkan.Backend` and the fallback fires elsewhere.
5. Document the approach in `LIMITATIONS.md` §3 update.

### Mac-248 (parallel branch)

1. Prototype `kinetic_energy.spv` — a single shader that does
   `r = p[i]² * inv_mass[i]` then reduces via shared-memory tree.
   Skeleton in PERSISTENT_BUFFERS_PLAN style, output is a single
   scalar f32. Skip if too speculative.
2. Prototype `normal_logpdf.spv` — `-0.5 * ((x-mu)/sigma)² - log(sigma) - 0.5*log(2π)`
   in one shader. Output same shape as x. Useful for the LOG_PROB hot
   path in distribution code.
3. Optional: review the Linux side's Fast wiring once it lands and
   suggest additions.

### Exmc (follow-up, separate session)

1. Audit NUTS leapfrog code (`lib/exmc/nuts/leapfrog.ex`).
2. Replace the leapfrog body with calls to `Nx.Vulkan.Fast.*`.
   Currently exmc writes the body inline as Nx ops; switching to
   named kernels is mechanical.
3. Run exmc Vulkan suite (v15+) and measure timeout count.

## Recap blog plan

Title (working): "**What Emily Taught Us**" — or "**The IR-Walker
We Stopped Building**" — Hitchens-flavored architectural retro on
why we walked away from the auto-detect compiler.

Outline:

1. **The 17-timeout problem.** Set the scene. NUTS sampling under
   :vulkan, the dispatch-overhead floor, the bench numbers.
2. **The IR-walk solution.** Show the work: right-folded chain
   detector, 4-input shader, auto-detect for 3-4 arg defns. Code
   weight: 600+ lines in `compiler.ex` plus shader machinery. Real
   wins on canonical patterns; false negatives on everything that
   didn't fit.
3. **Reading Emily.** The aside that reframes. "Emily decided not
   to do IR-level fusion. The fusion win was below the threshold
   that justified the integration cost. They ship `Emily.Fast`
   instead — named kernels, optional dispatch, defn fallback."
4. **The refactor.** What the new code looks like (~50 lines of
   `Nx.Vulkan.Fast`). What got deleted (TBD after measurement).
   What stays: `fused_chain_4`, the buffer pool, the compiled
   shaders. The compiler walker shrinks back to "delegate to
   Evaluator with opt validation."
5. **Results.** Timeout count, dispatch count per leapfrog,
   wall-time on the exmc Vulkan suite. Compare to v11/v12/v13/v14
   numbers we already have.
6. **The lesson.** Compiler-driven auto-detection is the wrong
   abstraction for vendor-specific fused kernels. The right place
   for caller intent is at the call site. Nx's `optional` Expr is
   how the framework already supports this — we built around it
   instead of through it.

Style: blog template per `blog-nuts-statem.html`. Voice: technical
authority, the well-placed aside about Emily's choice, the
self-deprecating moment about the 600 lines we stopped maintaining.

Length: 2500-3500 words magazine-feature.

## Acceptance

- `Nx.Vulkan.Fast` module exists with kernels 1–4
- Backend callbacks dispatch correctly to `fused_chain_4`
- Fallback runs on `Nx.BinaryBackend` (cross-backend correctness)
- nx_vulkan tests still ≥ 137; new tests cover Fast kernels both
  paths
- exmc audit + refactor ready for separate session
- Blog draft committed under `~/projects/learn_erl/pymc/www.dataalienist.com/`

## Cross-references

- `~/projects/learn_erl/emily/lib/emily/fast.ex` — reference impl
- `~/projects/learn_erl/emily/lib/emily/compiler.ex` — 141-line
  compiler that just delegates
- `LIMITATIONS.md` §3 — fused chain limits this approach lifts
