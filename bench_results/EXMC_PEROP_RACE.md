# The eXMC per-op Vulkan path, re-measured on 0.3.0

**Date:** 2026-08-15 · **Host:** super-io, RTX 3060 Ti (Ampere), Linux
**Harness:** [`exmc/bench/perop_vulkan_race.exs`](https://github.com/borodark/exmc)
**Model:** `Exmc.Trading.RegimeModel`, 3-regime mixture, **d = 8** free
parameters, 60 observations, f64 throughout.

## Why re-run it

`exmc/lib/exmc/nuts/tree.ex:1139` justifies the whole chain-shader synthesis
path with the claim that a per-op leapfrog dispatches

> through `multi_step_fn` → `Nx.Defn.Evaluator` → `VulkanoBackend` per-op →
> `BinaryBackend` host fallback, **producing zero GPU work**.

That was written in May 2026. Since then 0.3.0 moved the backward pass onto the
GPU ([`docs/BACKWARD_PASS_AUDIT.md`](../docs/BACKWARD_PASS_AUDIT.md)) and
batched command submission landed
([`BATCHED_DISPATCH.md`](BATCHED_DISPATCH.md)). A leapfrog step *is* a gradient
of log_p, so the one change most likely to invalidate that comment had never
been tested against it.

## Result: the comment still holds, for reasons that are now specific

### One `value_and_grad` of log_p

| arm | ms / gradient | host fallbacks |
|---|---:|---:|
| `BinaryBackend` (CPU reference) | **6.0** | — |
| `VulkanoBackend`, per-op (`Nx.Defn.Evaluator`) | 54.4 | **137** |
| `VulkanoBackend` + `Nx.Vulkan.Compiler` (fusion) | 58.3 | **137** |

`logp = -113.1858527033` on both GPU arms against `-113.1858456325` on the
host — agreement to 7 significant figures, i.e. the numbers are right and the
path is simply slow.

Fusion changes nothing, and the identical fallback count says why: the refusals
happen at the **backend** callback, below the compiler. `Nx.Vulkan.Compiler`
hands unsupported nodes to the Evaluator, which hands them to the same
`VulkanoBackend` callbacks, which take the same host path. Whole-graph
compilation cannot fuse across an op that is not on the device to begin with.

### End-to-end NUTS, 25 warmup + 25 samples

| arm | wall | ms / iteration |
|---|---:|---:|
| CPU (`compiler: :none`) | 19,661 ms | **393** |
| synthesised chain shader | 33,994 ms | 680 |
| per-op Vulkan | — | ~31,650 (from a 2+2 run) |

**At d = 8, no GPU path beats the CPU on this box.** The chain shader — the
fastest GPU arm, the one built specifically to avoid per-dispatch cost — is
1.7× *slower* than `BinaryBackend`. This is not a defect; it is what a
dispatch-bound workload looks like. Eight free parameters and sixty
observations is a few hundred bytes of arithmetic per leapfrog step. There is
nothing for 4,864 CUDA cores to do that the round trip does not already cost
more than.

The GPU case for MCMC therefore has to be made on **width** — large `d`, many
chains, or many instruments sampled concurrently — not on making a d=8 model
faster. That is a claim this benchmark can now test rather than assume.

## The 137: a census, and a familiar diagnosis

```
%{{:equal, 3} => 54, {:select, 4} => 54, {:multiply, 3} => 12,
  {:pad, 4} => 8, {:put_slice, 4} => 8, {:add, 3} => 1}
```

Probing `VulkanoBackend` directly (scratch script, same host) isolates two
causes:

| probe | fallbacks |
|---|---:|
| `greater(scalar, scalar)` | **1** |
| `greater(vector, scalar)` | 0 |
| `equal(scalar, scalar)` | **1** |
| `select(scalar, scalar, scalar)` | **1** |
| `select(vector_pred, vector, vector)` | 0 |
| `multiply` / `add` / `log` on scalars | 0 |
| `sum(vector) -> scalar` | 0 |
| `pad(vector)` | **1** |
| `put_slice(vector)` | **1** |
| `slice(vector)` | 0 |

### Cause 1 — rank-0 is excluded by the compare/select gate

`vulkano_backend.ex:1087` (compare) and `:1148` (select) both guard with

```elixir
tuple_size(out.shape) >= 1 and tuple_size(out.shape) <= 4
```

The upper bound is the documented rank-4 ceiling of the index-remap family.
**The lower bound refuses scalars**, and nothing in the shader requires that —
elementwise arithmetic on the same host handles rank 0 without complaint.

This is the **same bug class the backward-pass audit named**: a capability gate
written against the shapes one workload happens to produce. There it was a
forward pass refusing its own gradient. Here it is a neural-network workload —
where predicates are batched masks of rank ≥ 1 — refusing a probabilistic
one, whose distribution log-probs are full of scalar support checks
(`x > 0` for HalfCauchy, and so on). 108 of the 137 are this one guard.

### Cause 2 — `pad` and `put_slice` have no shader at any rank

Both are listed in [`LIMITATIONS.md`](../LIMITATIONS.md) §2 as
never-implemented. `PointMap` packs and unpacks the flat parameter vector with
`put_slice`, once per RV — hence exactly 8, hence `d` of them per gradient.

This one is worse than its count. **The fallback census is a lower bound**:
once `put_slice` transfers the position vector to `BinaryBackend`, every
downstream op computes there without being recorded. The 12 `multiply` and 1
`add` are the visible edge of that; the true residency of this graph after the
first unpack is close to zero. The May comment's "producing zero GPU work" is
still approximately correct, and `put_slice` is why.

## What this says to build

Ordered by the measurement:

1. **Drop `>= 1` to `>= 0` in the compare and select gates**, reshaping rank-0
   to `{1}` for dispatch. Removes 108 of 137. Mechanical, and it wants a
   `fallback_test.exs` assertion at zero for scalar compare/select so it cannot
   silently return.
2. **`put_slice` and `pad` shaders.** Strided-copy overlay and a padded copy;
   both are in the index-remap family whose skeleton already exists three times
   over. This is the one that actually decides residency.
3. Only then re-run this harness. Steps 1–2 are worth doing on their own merits
   — they are real gaps in a general-purpose backend — but they will not make a
   d=8 model faster than a CPU, and no one should expect them to.

The honest conclusion for eXMC is unchanged by any of it: **the chain shader is
still earning its keep**, and the reason to keep investing in it is that it
turns ~30 dispatches per leapfrog step into one. What has changed is that the
per-op path's failure is now itemised rather than asserted.

## Method notes

- Every timed GPU call is forced to resolve (`backend_copy` + `to_flat_list`)
  before the clock stops. Batched submission makes "time the work, not the
  recording of it" a real distinction — see `BATCHED_DISPATCH.md`.
- The per-op arm is produced by stripping `chain_meta` to `nil` in the compiled
  tuple, which is the same switch `Tree.dispatch_subtree_hot/11` reads. The
  Plan B′ guard is bypassed with
  `config :exmc, :allow_vulkan_perop_sampling, true`.
- Pipeline and shader caches are warmed with one untimed call per arm.
- **Posterior agreement is not reported here.** At 25 samples the Monte-Carlo
  error on a heavy-tailed HalfCauchy posterior swamps any backend difference,
  which is the trap `research/ASSESSMENT_2026_07_13.md` already documented once.
  The chain arm's over-dispersion signature on this host is a separate open
  investigation and needs its own design, not a by-product of a timing run.
