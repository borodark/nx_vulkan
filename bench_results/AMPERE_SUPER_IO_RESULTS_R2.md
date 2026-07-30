# Ampere race results, round 2 — super-io (RTX 3060 Ti), tiled kernels

Round-2 run of `RACE_TODO_SUPER_IO.md` on **super-io (192.168.0.249)**,
2026-07-29. Device `{:ok, "NVIDIA GeForce RTX 3060 Ti", "DiscreteGpu"}` (GA104,
consumer Ampere, f64 ~1/32 rate). Commit `d47afcd`. `mix test`: **174 tests, 0
failures**. Round 1 (untiled) is `AMPERE_SUPER_IO_RESULTS.md` at `84f3f11`.

## Verdict

**The 1024³ cliff is gone, and `:f32acc` is now size-stable and *improving* with
size: 2.18× (512) → 2.47× (1024) → 2.97× (2048).** Conv `:f32acc` lands
2.46–3.06× on the two real layers. Tiling the *f32* GEMM is a clear win on
Ampere.

Two round-2 findings that qualify the asks:

- **Tiling the f64 matmul is ~1.15× at 512³ and nothing at 1024³ on Ampere** —
  not the 1.35–1.4× seen on the GT 650M. Measured by direct shader A/B, below.
- **This card is jittery enough to have faked a result.** f64 512³ spans
  3.10–5.32 ms across identical interleaved rounds (1.7×). Single-shot numbers
  here are not trustworthy; everything below is a median of many rounds after a
  200-iteration soak to steady-state clocks.

## 1. Matmul sweep — `Nx.dot`, 512+ interleaved, accuracy included

`rel_err` = max |f32[:f32acc] − f64| / max |f64| over the full output.

```
n      f64 ms   :f64acc ms (x)     :f32acc ms (x)   rel_err     GFLOP/s(:f32acc)
--------------------------------------------------------------------------------
128     1.081     1.078 (1.0x)      0.874 (1.24x)   0.00000129     4.8
256     1.572     1.682 (0.93x)     1.561 (1.01x)   0.00000216    21.5
512     5.343     7.860 (0.68x)     2.446 (2.18x)   0.00000281   109.7
1024   22.576    32.233 (0.70x)     9.157 (2.47x)   0.00000323   234.5
2048   97.332   209.344 (0.46x)    32.820 (2.97x)   0.00000547   523.5
```

Against round 1 (`:f32acc` via `Nx.dot`), the cliff fix is unambiguous:

```
n       round 1 (untiled)   round 2 (tiled)
128     0.697 (1.40x)       0.874 (1.24x)
256     1.328 (1.09x)       1.561 (1.01x)
512     1.735 (1.86x)       2.446 (2.18x)
1024   13.130 (1.09x)       9.157 (2.47x)   <-- cliff gone
2048   —                   32.820 (2.97x)
```

Absolute ms are not comparable across the two rounds (see the jitter note — the
f64 baseline moved more than tiling did); the **ratios** are the signal, and each
ratio is measured against a f64 baseline taken in the same process.

Small n (128/256) is dispatch-overhead-bound, not compute-bound: 4.8 GFLOP/s at
n=128 vs 523 at n=2048. `:f32acc` can't win what the GEMM isn't spending time in.
It still never *loses*.

Accuracy is textbook f32 GEMM: ~1–5e-6 max relative error, growing as ~√K
(f32 eps 1.2e-7, K=2048 → ≈5.4e-6 expected, 5.47e-6 measured).

## 2. Step-4 policy check at 512³ — long soak, 20 warm + 50 timed, 5 rounds

```
medians:        f64=5.223   :f64acc=4.890   :f32acc=1.938
median ratios:  :f64acc=1.07x            :f32acc=2.70x
spreads:        f64 3.097–5.323ms        :f32acc 1.707–3.774ms
```

`:f32acc` at **2.70×** on medians is the most trustworthy single number in this
report. Note `:f64acc` came out at ~1.07× here against 0.68× in the sweep — that
pair is inside the noise band on this card, so treat "`:f64acc` ≈ f64, ±40%" as
the honest statement and don't quote a precise `:f64acc` ratio at 512³.

## 3. Conv — `Nx.conv`, tiled conv-GEMM, real layers

```
layer                       f64 ms   :f64acc (x)     :f32acc (x)    rel_err
---------------------------------------------------------------------------
{8,32,28,28}·{64,32,3,3}     16.470   13.134 (1.25x)   5.382 (3.06x)  9.8e-7
{4,64,16,16}·{128,64,3,3}     5.919    2.979 (1.99x)   2.409 (2.46x)  4.9e-6
```

Conv differs from matmul in one useful way: **`:f64acc` is also a win here**
(1.25×, 1.99×), because conv's f32 path additionally gets an f32 im2col, which is
pure bandwidth-bound data movement and unaffected by the accumulator. So conv f32
is worth taking on this card even at the conservative default.

`scripts/race.sh` at `d47afcd` (its matmul/conv f32 rows are `:f64acc`, now
recorded in the JSON — thanks):

```
op                        f64 ms     f32 ms   speedup      round-1 f64
matmul 128x128x128          0.767      1.114    0.69x       0.836
matmul 256x256x256          1.404      1.679    0.84x       1.577
matmul 512x512x512          2.905      4.768    0.61x       3.203
conv 8->16ch 24sq           0.513      0.504    1.02x       0.454
conv 16->32ch 32sq           1.19      0.828    1.44x       1.895
elementwise add 1M          3.927      1.543    2.55x       4.308
elementwise tanh 1M         2.832      1.536    1.84x       3.270
sum 1M (full)             131.889     66.753    1.98x     133.077
sum 1024x1024 axis0         0.545      1.109    0.49x       0.548
```

## 4. Did tiling speed up the f64 matmul on Ampere? Barely.

`race.sh` suggested ~1.10× (3.203 → 2.905 at 512³), but that is one sample per
commit and the jitter band is wider than the effect. Proper A/B: swap
`priv/shaders/matmul_f64.spv` between the tiled build and `84f3f11`'s untiled
shader, everything else identical, same process shape, 7 rounds each, medians:

```
n      tiled f64        untiled f64      tiled/untiled
512    3.575 (min 3.456)  4.106 (min 3.573)   1.15x faster
1024  13.289 (min 12.476) 12.572 (min 11.959) 0.95x — slightly SLOWER
```

So: **~1.15× at 512³, a wash-to-slight-regression at 1024³.** Compare the GT
650M's 1.35–1.4×. That asymmetry is expected — Kepler has a small, weak cache
hierarchy that explicit shared-memory staging rescues, whereas Ampere's L1/L2
were already feeding the naive f64 kernel most of what tiling would have staged.
The tiled f64 shader is still worth keeping (it's f64-exact and helps at 512),
but the Kepler f64 speedup does not transfer, and it should not be quoted as a
cross-device win.

## 5. The decisions this reopens

### Device-aware default: no. Flip the default outright, or leave it.

The *performance* case is now made — 2.2–3.0× on matmul, 2.5–3.1× on conv,
size-stable, never a loss, ~5e-6 worst-case relative error. Round 1's objection
(a 1.09× win at 1024 isn't worth a semantics change) no longer applies.

But **device-aware is the wrong shape for the fix.** A default that depends on a
runtime device probe means identical code produces different numerics on
super-io than on mac-247 — irreproducible across hosts, and the failure mode is
a results diff nobody can attribute. For a numerical library that is worse than
either uniform choice.

If the 2.2–3.0× is wanted by default, the defensible move is to **flip the
default to `:f32` uniformly** and keep `:f64` as the opt-in. That matches what
every other f32 GEMM does — cuBLAS `SGEMM`, EXLA, PyTorch all accumulate f32 in
f32 — so `:f64` accumulation is the surprising behaviour here, not the reverse. A
caller who built an f32 tensor has already accepted f32 precision; paying 2–3×
to accumulate more precisely than the data warrants is a courtesy few asked for.

**Recommendation: flip the default to `:f32` uniformly, document the ~√K·eps
error growth, keep `:f64` opt-in for accumulation-sensitive work. Do not wire a
device-aware default.** If that feels too aggressive for the branch, the status
quo (`:f64` default, `:f32` opt-in) is fine and cheap — but pick one of those
two, not the per-device one.

### Register blocking: yes, clearly worth chasing.

At n=2048, `:f32acc` hits **523 GFLOP/s against ~16 TFLOP/s of f32 peak on this
card — about 3%.** A 16×16 tile with one output element per thread is
shared-memory-bound; the standard next step (8×8 outputs per thread, 128×128
macro-tile, so each shared-memory load feeds 8 FMAs instead of 1) typically buys
3–5× on exactly this kernel shape. That is a far larger prize than anything left
in the accumulator question, and it needs no semantics change at all.

Also worth fixing first, since it caps every small-matrix result here: **n=128
runs at 4.8 GFLOP/s**, i.e. per-dispatch overhead dominates below ~256. If
`Nx.dot` submits a command buffer and waits per call, batching or reusing
submissions would lift the entire small-matmul range regardless of accumulator.

## Method note for whoever runs round 3

This card needs the soak-and-median treatment: idle SM clock is 210 MHz against
a 2100 MHz max, so short bursts measure clock ramp. Every number above used ≥20
warm-up iterations after a 200-iteration soak, ≥15 timed iterations, tensors
allocated once, configs interleaved within each round, and medians across 5–7
rounds. No other process held the GPU (`nvidia-smi --query-compute-apps` empty,
60 °C at idle). Even so, expect ±40% on the f64 path — quote ratios from the same
process, never absolute ms across runs.
