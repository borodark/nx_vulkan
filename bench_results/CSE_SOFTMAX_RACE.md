# Cross-stage CSE race — softmax (full, last-axis)

Cross-stage CSE (commit `b4608a3`) materialises a boundary-crossing shared
subexpression once (the softmax numerator `n = exp(x - max(x))`, used by both the
`sum(n)` reduce and the final `divide`) instead of re-inlining it into each
consumer. Tradeoff: an extra dispatch + buffer vs the saved recompute.

`examples/cse_softmax_bench.exs` compares, per shape: eager | fused CSE-on |
fused CSE-off (`NXV_CSE=0`, the pre-CSE re-inline path) | **on/off ratio**
(>1 => CSE helps; <1 => regresses). All errors 0.0 (bit-exact vs BinaryBackend).

## GT 650M (Kepler, mac.247) — CSE NEVER WINS

| shape        | eager (ms) | CSE-on (ms) | CSE-off (ms) | on/off |
|--------------|-----------:|------------:|-------------:|-------:|
| {64,64}      | 0.626 | 0.788 | 0.628 | **0.80** |
| {64,256}     | 1.108 | 0.833 | 0.693 | **0.83** |
| {64,1024}    | 2.642 | 1.130 | 0.963 | **0.85** |
| {256,64}     | 0.724 | 0.955 | 0.832 | **0.87** |
| {256,256}    | 1.480 | 1.338 | 1.079 | **0.81** |
| {256,1024}   | 4.109 | 2.754 | 2.362 | **0.86** |
| {1024,64}    | 1.307 | 1.414 | 1.388 | 0.98 |
| {1024,256}   | 2.934 | 3.264 | 3.256 | 1.00 |
| {1024,1024}  | 10.697 | 11.017 | 10.910 | 0.99 |

Read: hoisting the numerator into its own stage costs more (extra dispatch +
buffer) than the recompute it saves — a regression of ~0.8x on small/medium
tensors, converging to neutral (~1.0x) only when the tensors are large enough
that dispatch overhead is amortised. It is **never a net win on Kepler**.

## RTX 3060 Ti (Ampere, super-io/249) — CSE NEVER WINS EITHER

Two runs; on/off ratio (representative — small/medium regress, large neutral):

| shape        | on/off run1 | on/off run2 |
|--------------|------------:|------------:|
| {64,64}      | 0.80 | 0.96 |
| {64,256}     | 0.73 | 0.77 |
| {64,1024}    | 0.85 | 0.87 |
| {256,64}     | 0.77 | 0.72 |
| {256,256}    | 0.87 | 0.84 |
| {256,1024}   | 0.83 | 0.84 |
| {1024,64}    | 0.93 | 1.13 |
| {1024,256}   | 0.97 | 1.01 |
| {1024,1024}  | 0.97 | 0.99 |

Read: same shape as Kepler and worse in the mid-range (down to **0.72x**). The
only >1 reading anywhere is {1024,64} run2 at 1.13x, contradicted by 0.93x in
run1 — noise, not a win. On this compute-rich discrete GPU the recompute CSE-off
does is essentially free, while hoisting costs an extra dispatch + a global-memory
round-trip that dominates. No size threshold where hoisting starts to pay off.

Note: CSE-on's *fusion* speedup vs eager is still real (e.g. {256,1024} ~1.7x) —
it's the isolated hoisting decision (on vs off) that never wins.

## Decision — DEFAULT-OFF, opt-in via `NXV_CSE=1`

Both device classes measured (weak Kepler + strong Ampere) show cross-stage CSE
ranging from harmful (~0.72x) to neutral (~1.0x), with **no class where it pays
off**. Unlike the many-slot fused reduce — which genuinely helps weak GPUs and is
correctly weak-gated — there is no evidence to justify device-class gating here.
Shipped **default-off** (commit after `aa15d5c`); `NXV_CSE=1` opts the hoisting in
for the rare graph with a genuinely expensive boundary-crossing shared subexpr
(cheap softmax arithmetic isn't it). Revisit only if such a workload shows a
repeatable >1 region.

The always-beneficial multi-output **memo-reuse** path (reusing a stage buffer
already materialised for another tuple output — no extra dispatch) is independent
of `NXV_CSE` and stays on; e.g. `{n, sum(n)}` still shares `n`.
