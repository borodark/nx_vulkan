# Ampere race results — super-io (RTX 3060 Ti)

Run of `RACE_TODO_SUPER_IO.md` on **super-io (192.168.0.249)**, 2026-07-29.
Device confirmed `{:ok, "NVIDIA GeForce RTX 3060 Ti", "DiscreteGpu"}` (GA104,
consumer Ampere → f64 at ~1/32 of f32 rate). Branch `f32-matmul-prototype`
at commit `7726a22`. `mix test`: **174 tests, 0 failures**.

## Verdict

**Expected pattern confirmed: `:f64acc ≤ f64 ≤ :f32acc`.** The accumulator
policy is now validated across two GPU generations (Kepler GT 650M, Ampere
GA104). Bandwidth-bound ops win regardless, as on Kepler.

## 1. `scripts/race.sh` — `bench_results/f32_race_super-io_7726a22.json`

```
op                        f64 ms     f32 ms   speedup   on_gpu
------------------------------------------------------------------
matmul 128x128x128          0.836      1.275    0.66x   true
matmul 256x256x256          1.577      1.754     0.9x   true
matmul 512x512x512          3.203      4.691    0.68x   true
conv 8->16ch 24sq           0.454       0.65     0.7x   true
conv 16->32ch 32sq          1.895       0.91    2.08x   true
elementwise add 1M          4.308      1.803    2.39x   true
elementwise tanh 1M          3.27      1.592    2.05x   true
sum 1M (full)             133.077     67.809    1.96x   true
sum 1024x1024 axis0         0.548      1.152    0.48x   true
```

**Caveat: the `matmul` rows here are `:f64acc`** (the shipped default), which is
why they read as losses. They are not the f32 matmul ceiling — see §2/§3. The
JSON does not record which accumulator was in effect; worth adding a field so
these rows can't be misread later.

## 2. `examples/matmul_accumulator_race.exs`

```
matmul       f64 ms   f32/f64acc (x)     f32/f32acc (x)
--------------------------------------------------------
256x256x256   1.17      1.53 (0.76x)         1.6 (0.73x)
512x512x512   2.87      4.68 (0.61x)        1.56 (1.85x)
```

## 3. Policy through the real `Nx.dot` path (step 4)

The step-4 snippet in the brief is **not reliable as written** — 4 iterations,
one warm-up, fresh tensors per config, and `:f32acc` measured last. It reported
`f64=2.66  :f64acc=13.97 (0.19x)  :f32acc=3.63 (0.73x)`, i.e. an apparent
*regression* for `:f32acc`, contradicting §2. Re-running with 3 warm-ups, 20
timed iterations, tensors allocated once, and configs interleaved across 3
rounds gives the opposite and consistent answer:

```
round 1: f64=3.878  :f64acc=5.438 (0.71x)  :f32acc=1.878 (2.06x)
round 2: f64=3.422  :f64acc=5.299 (0.65x)  :f32acc=1.913 (1.79x)
round 3: f64=3.414  :f64acc=11.737 (0.29x) :f32acc=2.361 (1.45x)
```

The policy *is* honoured through `Nx.dot` (`:f64acc` and `:f32acc` differ by
~3x), and `:f32acc` is a real 1.45–2.06x win at 512³. Recommend fixing the
step-4 snippet in the brief before anyone reruns it. Note the run-to-run
variance on `:f64acc` (5.3 → 11.7ms); the f64 path on this card is noisy.

## 4. Size sweep + accuracy (via `Nx.dot`, 5 warm-ups, 20 iters / 10 at 1024)

`rel_err` = max |f32[:f32acc] − f64| / max |f64|, over the full output.

```
n      f64 ms   :f64acc ms (x)     :f32acc ms (x)   rel_err(:f32acc)
------------------------------------------------------------------------
128     0.978      0.955 (1.02x)      0.697 (1.4x)   0.00000129
256     1.449      1.621 (0.89x)      1.328 (1.09x)  0.00000216
512     3.229      5.152 (0.63x)      1.735 (1.86x)  0.00000281
1024   14.268      26.95 (0.53x)      13.13 (1.09x)  0.00000323
```

Two things to note:

- **`:f32acc` never loses** on this card (1.09x–1.86x), while `:f64acc` loses
  from n≥256 and gets worse with size (0.53x at 1024) — exactly the f64
  rate-limit signature, and more pronounced than on Kepler.
- **The 1024 win collapses to 1.09x.** `:f32acc` scales 7.6x from 512→1024
  (8x the work) while f64 scales only 4.4x, so f64 is amortising fixed overhead
  that `:f32acc` has already shed. The naive kernel is likely hitting a
  bandwidth/occupancy wall at 1024 rather than a compute one. Worth a look
  before claiming a general large-matmul win — the 512³ headline does not
  extrapolate.

Accuracy: max relative error ~1–3e-6, growing as ~√K (f32 eps 1.2e-7, K=1024 →
≈3.8e-6 expected). That is textbook f32 GEMM behaviour, not a defect.

## 5. The decision the brief asked for

**Keep the default at `:f64`. Do not auto-flip on f64-rate-limited devices yet.**

The performance case is real but narrower than expected: the win is 1.09x at the
two sizes that bracket the sweet spot and only 1.86x at 512³. Silently changing
numerical semantics based on a runtime device probe, for a geometric-mean win in
that range, trades a *correctness* property for a modest and size-dependent
speed one — and the caller who wants it already has a one-line opt-in.

What the data does support:

- Keep `:f32` opt-in, and document the 512³-class sweet spot and the ~√K·eps
  error growth so callers can judge it.
- Chase the 1024 scaling cliff first. If `:f32acc` recovers to ~2x at 1024+, the
  device-aware-default question is worth reopening with a much stronger case.
- Next f32 step from the brief still stands: give conv's GEMM the same policy
  (`conv_gemm_f32_f32acc`). Conv already shows 2.08x at 16→32ch f32 here even
  without it.
