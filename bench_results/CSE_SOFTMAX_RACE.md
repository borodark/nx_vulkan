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

## RTX 3060 Ti (Ampere, super-io/249) — PENDING
(race in flight; fills in whether hoisting ever pays off on a strong GPU, which
decides the default: device-class-gated vs default-off.)

## Decision
TBD after Ampere. Given Kepler = never-win, the likely outcome is default-OFF on
weak GPUs. The always-beneficial multi-output memo-reuse path (reusing a stage
buffer already materialised for another tuple output — no extra dispatch) is
unaffected by `NXV_CSE` and stays on.
