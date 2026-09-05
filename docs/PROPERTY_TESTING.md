# Property testing on the chain path — what it does, and why it is shaped this way

**Scope:** `test/nx_vulkan/chain_properties_test.exs`, added 2026-09-04 at
`a29246f`, and the reasoning that produced it. 14 tests; 1.1 s on the RTX
3060 Ti, 5.9 s on the Jetson, 1.5–3.6 s on the Keplers.

This is not a general argument for property testing. It is a record of which
properties were chosen for *this* code, which were rejected, and what each one
is actually able to detect — because on a GPU backend the dangerous failure is
almost never an exception.

---

## 1. The failure class this code actually has

A wrong answer here does not crash. It is a number.

- A shader reading `alpha` out of `eps`'s bytes produces a plausible float.
- A `logp` tree reduce that sums 256 of 300 elements produces a plausible float.
- A batched dispatch that drops the `inst` offset produces four copies of one
  chain, all individually plausible.
- A padded instance whose extra steps corrupted its prefix produces a plausible
  trajectory.

None of these raise. None fail a size check. Several would pass a tolerance
comparison against a host reference, because they are wrong by a *modelling*
amount rather than a *numerical* one.

That is why almost every assertion in this file is **bit-identity between two
paths that must agree**, rather than a value compared to a computed expectation.
It is the only bar that a plausible-looking wrong answer cannot clear.

---

## 2. Why bit-identity, and between what

The chain path has three pairs that must agree exactly, by construction:

| A | B | why they must be identical |
|---|---|---|
| batched instance *i* | that chain dispatched alone | the batched shader IS the single shader with indices offset by `inst` |
| first *n* steps of a K=7 dispatch | a K=n dispatch | each step writes from state depending only on earlier steps |
| `n_instances = 1` batched | the single-instance path | one workgroup either way |

None of these needs a model of what a leapfrog *should* produce. They are
internal consistencies, so they hold on every device regardless of vendor
fast-math, and they hold in f32 and f64 alike. That is what makes them safe to
assert at bit level across an RTX 3060 Ti, two 2012 Keplers and a Tegra X1.

**The prefix property is load-bearing downstream.** eXMC batches chains of
unequal NUTS depth by dispatching all of them at the deepest K and slicing each
caller back to what it asked for. That is only sound if a longer dispatch's
prefix is exactly a shorter dispatch's output. It is, and it is now pinned, so a
future change to the step loop cannot silently break their flush.

---

## 3. The guard branches: a guard never observed to fire

Seven refusal paths exist in the Rust NIFs. All were untested until this file.

| guard | returns |
|---|---|
| `q_init.len() != p_init.len()` | `:size_mismatch` |
| `k == 0` | `:bad_input` |
| `push.len() == 0` or `> 128` | `:bad_input` |
| push-block parse failure | `:bad_input` |
| `d == 0` or `d*elem > q_init.len()` | `:size_mismatch` |
| `d > 256` | `:bad_input` |
| `n_instances == 0` | `:bad_input` |

All fire before `ctx()` — no GPU work, no allocation — so they are the cheapest
tests in the repo and run unconditionally on every box.

They are here because **a guard that has never been observed to fire is
indistinguishable from a guard that cannot**. That is not hypothetical in this
tree: `d > 256` was unenforced for months while the chains silently produced an
undefined tail and a `logp` summed over only the first 256 elements. It was
harmless purely by accident — `d` sat near 13 because of a push-budget artifact
— and the accident evaporated when the downstream consumer removed that budget.

Boundaries are tested on both sides. `push.len()` is checked at 0 and at 129,
not merely "too long". `d` is checked at 256 (accepted) and 257 (refused), in
f32 as well as f64.

---

## 4. Shapes: boundaries hit, interiors fixed

```
d  : 1, 2, 7, 19, 41, 256
K  : 1, 2, 4, 7, 11
ni : 1, 2, 3, 4
```

Swept pairwise, not Cartesian — each axis at its boundaries with the others
held mid-range. Fifteen shapes per family instead of 120, and every boundary value
still appears in every assertion.

**The interior values are literals in the source, not draws.** This is
deliberate and it is the one place this file departs from conventional
property-testing practice:

- A failure on the Jetson takes minutes to reproduce. A seed-replay step between
  "CI is red" and "I can run it" is a step nobody takes at two in the morning.
- The rest of this suite asserts exact bytes everywhere. Live generation would
  make this file the only one whose failures are not directly reproducible.
- The value of generation here is *coverage of the space*, which a fixed
  well-chosen set delivers, not *adversarial search*, which needs a shrinker and
  a corpus this code does not have.

`stream_data` is not a dependency and was not added. The cost/benefit did not
justify a new dependency for fifteen shapes.

**Family is not swept here.** `chain_f64_test.exs` already sweeps all six
families at one fixed shape. This file sweeps shape at three representatives:

- `normal_f64` — cheapest, and the only family whose shader contains no
  boundary-cast transcendental at all
- `weibull_f64` — the only family with GLSL `helpers`, functions emitted before
  `main()`, the highest-risk construct under `inst`-offset indexing
- `beta` (f32) — dtype coverage, and cache-warm from the other test files

Repeating the family axis would multiply dispatches without covering anything
new.

---

## 5. Determinism

The same inputs dispatched twice must produce the same bits, single and batched.

Nothing in the suite did this before. Every example test dispatched each input
set exactly once, and a single dispatch cannot see:

- a buffer reused from a pool without being fully written
- a reduction whose workgroup ordering varies between launches
- anything that reads uninitialised memory only sometimes

Two dispatches, one comparison. It is the cheapest property here and it covers a
class nothing else touches.

---

## 6. `grad` is the derivative of `logp` — the substitute for host references

Five of the six f64 families had **no numerical validation whatsoever**. Only
"compiles, dispatches, no NaN". Normal was the exception, with a bit-exact host
leapfrog reference.

The obvious fix — write host references for the other five — is a correctness
liability, not a solution:

- It means re-deriving five probability densities and their gradients in Elixir.
- Student-t's normalising constant needs `lgamma`, which Erlang's `:math` does
  not have. Hand-rolling Lanczos or Stirling to test a shader is a second
  implementation that can itself be wrong, and in the same direction.
- Those five shaders boundary-cast through f32 (`double(exp(float(x)))`,
  GLSL.std.450 having no f64 transcendentals), so any host comparison is really
  asserting "Elixir's `:math.exp` agrees with this vendor's fast-math within a
  fudge factor" — a device-dependent claim across four different drivers.

**So the shader is checked against itself.** If `grad_block` is the derivative of
`logp_block`, a central difference of `logp` must reproduce `grad`:

```
(logp(q + h·eᵢ) − logp(q − h·eᵢ)) / 2h  ≈  grad_chain[0][i]
```

Setup: `K = 1`, `p₀ = 0`, `mass = 1`, `eps = 1e-8`. The leapfrog position update
is then `0.5·eps²·grad ≈ 1e-16`, far below any finite-difference step, so
`logp_chain[0]` is effectively `logp(q₀)` despite the shader's
evaluate-after-update output contract.

Two properties fall out of this that a host reference would not have given:

1. **No second density implementation.** Nothing about the model is restated.
2. **The normalising constant drops out.** Constants differentiate to zero, so
   `logp_const` — the part this library explicitly documents as the caller's
   responsibility, and the part needing `lgamma` — is not under test and does not
   need to be.

### The step size is measured, not guessed

Max `|fd − grad|` at d=4, on the RTX 3060 Ti:

| family | h = 1e-3 | h = 1e-4 |
|---|---|---|
| `normal_f64` | 7.3e-14 | 1.6e-12 |
| `cauchy_f64` | 4.7e-05 | 5.2e-04 |
| `exponential_f64` | 9.0e-05 | 9.7e-04 |
| `halfnormal_f64` | 2.7e-05 | 3.1e-04 |
| `studentt_f64` | 1.4e-04 | 4.1e-04 |
| `weibull_f64` | 1.7e-04 | 2.2e-03 |

**Smaller h is worse.** The f32 boundary cast puts ~1e-7 of noise on `logp`, and
a central difference divides that by `2h`, so halving h doubles the noise term
while cutting the O(h²) truncation term fourfold from a base that is already
negligible. The optimum is dominated by noise, not truncation.

`normal_f64` sits at 1e-13 rather than 1e-5 because its shader has no boundary
cast anywhere — every constant is precomputed on the host in full f64. That is
independent confirmation of why it is the only family for which a bit-exact host
comparison is even coherent.

Tolerance is `max(5e-3·|grad|, 2e-3)` — roughly 30× the worst observed error.
Wide enough to absorb vendor fast-math differences across the fleet (confirmed:
it holds on Kepler and Tegra too), tight enough that a sign error or a wrong
coefficient, both O(1) relative, cannot hide.

---

## 7. Every property test needs a null arm

The finite-difference test is followed by one that compares **one family's
`logp` against another family's `grad`** and asserts the comparison **fails**.

This is not decoration. Three vacuous checks were found in a single day in this
project:

1. A cross-build verification invoked `file` to confirm an ELF architecture.
   `file` was not installed. The line printed `command not found`, the lines
   around it succeeded, and the architecture was never checked.
2. A batching test dispatched the single-instance path twice and asserted the
   results were equal — true of any implementation whatsoever.
3. A NaN guard written as `for v <- doubles(bin), do: assert v == v` could not
   fail. See §8.

All three looked exactly like passing tests. "Grad matches finite differences"
is evidence only if a mismatch would have been detected, and the only way to
know that is to arrange a mismatch and watch it fail.

---

## 8. The arity-guarded decode

`for <<v::float-64-little <- bin>>` does **not** raise on a NaN or Infinity bit
pattern. It skips the segment:

```
<<0,0,0,0,0,0,240,127>>  (+inf)  ->  []
<<1,0,0,0,0,0,240,127>>  (NaN)   ->  []
<<0,0,0,0,0,0,248,63>>   (1.5)   ->  [1.5]
```

So the natural NaN guard — decode, then assert every value equals itself —
cannot fail. The NaN removes itself from the list before the assertion sees it,
and the surviving finite values all pass.

(A direct `<<v::float-64-little>> = bin` match *does* raise `MatchError`. The
comprehension form, which is what one naturally writes for a buffer, silently
truncates instead.)

`doubles/1` and `floats/1` therefore assert that they decoded
`byte_size(bin) / element_size` values. Every caller inherits the guard, and a
NaN anywhere in a chain output now fails the test that decodes it.

---

## 9. What was deliberately not tested, and why

**Host references for cauchy / exponential / halfnormal / studentt / weibull.**
See §6. Replaced by finite-difference self-consistency.

**The real `maxComputeWorkGroupCount[0]` boundary.** Device-dependent — the
Keplers may report Vulkan's guaranteed 65535 floor while the 3060 Ti reports far
more, verified by running 70000 instances there with no clamping. Testing the
true boundary means hardcoding per-device knowledge that will drift, or querying
the same value the guard reads (tautological), or allocating enough to cross a
real ceiling, which risks OOM on the Jetson's unified memory. The guard is
correct by construction; the boundary is not worth asserting.

**Cross-dtype numerical parity for the same family.** Impossible without new
production code: `ChainShaderSpecs` (f32: beta, gamma, lognormal) and
`ChainShaderSpecsF64` (normal, cauchy, exponential, halfnormal, studentt,
weibull) share **zero family names**.

**Extreme-value fuzzing of q / p / mass.** Several families work in unconstrained
log space where a large `q_uc` legitimately overflows the f32 boundary cast in
`exp()`. That is a documented limitation, not a bug, so fuzzing into it produces
noise rather than signal.

It is now *characterised* rather than merely asserted. Measured on the RTX 3060
Ti at d=2, K=1 — the first |q| at which `logp` or `grad` goes non-finite:

| family | threshold | why |
|---|---|---|
| `normal_f64` | none to 700 | no boundary cast anywhere |
| `cauchy_f64` | none to 700 | `log(1 + z²)` grows too slowly to reach it |
| `studentt_f64` | none to 700 | same |
| `exponential_f64` | **88.72** | `exp(float(q))`, and ln(f32_max) = 88.7228 |
| `halfnormal_f64` | **44.36** | `exp(float(2q))`, so half of it |
| `weibull_f64` | **44.36** | same |

The two numbers are exactly `ln(f32_max)` and `ln(f32_max)/2`. Nothing about
this is device-specific — it is the IEEE f32 range meeting a cast the shader
performs deliberately, because GLSL.std.450 has no f64 transcendentals.

**What this means for a sampler.** A NUTS chain that drives a scale parameter
toward zero can reach these magnitudes in unconstrained space *during warmup*
while every fixed-point test stays finite, which is why the finite-difference
property in §6 cannot see it: it perturbs around a well-conditioned point by
construction. A failure that requires the sampler to get somewhere first is
outside what any of these properties test, and `chain_boundary_test.exs` pins
the thresholds rather than pretending otherwise.

**Instances-do-not-bleed as a swept axis.** Strictly subsumed by per-instance
bit-identity: if instance *i* equals its own single-dispatch reference for every
*i*, and the inputs differ per instance, bleed is already excluded. The single
example test is kept for documentation value.

---

## 10. Adding a family or a property

**A new f64 family:** add it to `ChainShaderSpecsF64.all/0` and it is picked up
automatically by the all-family batching test and by the finite-difference test.
Nothing else needs editing. If it needs GLSL helper functions, look at
`weibull_f64` — helpers are emitted before `main()` and must take their operand
as a parameter to stay index-blind under batching.

**A new property:** write the null arm first. If you cannot construct an input
for which the property fails, you do not yet know what the property detects.

**Runtime:** the whole file costs ~1.1 s on the 3060 Ti and ~5.9 s on the
Jetson, which is the budget to respect. If a new property pushes the
Jetson materially past ~6 s, tag it `:slow` and wire the exclusion into
`test/test_helper.exs`; the P3 tier (full 9-family × 10-shape grid, wide
`n_instances` stress) was scoped for that treatment and has not been needed yet.
