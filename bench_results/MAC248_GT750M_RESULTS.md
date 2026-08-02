# mac.248 race results — GT 750M (second Kepler)

Driven over SSH from mac-247 on **mac.248 (192.168.0.248)**, 2026-07-30, to
split the effort and get a second-Kepler data point. Device
`{:ok, "NVIDIA GeForce GT 750M", "DiscreteGpu"}` (Kepler, f64 rate-limited).
Branch `f32-matmul-prototype` at commit `0a22805`.

## Verdict

Corroborates the GT 650M (mac-247): on Kepler, **f32 wins big on bandwidth-bound
ops and the tiled `:f32acc` GEMM; register blocking regresses the fast path.**
Two Kepler cards now agree, so the reverted default (plain 16×16 tiling) is right
for Kepler. Register blocking remains a modern-GPU (Ampere) play — see
`AMPERE_SUPER_IO_RESULTS_R2.md`.

## 1. Full family race (`f32_race_free-macpro-nvidia_0a22805.json`, default :f64acc)

```
op                        f64 ms     f32 ms   speedup   on_gpu
matmul 128x128x128          0.736      1.498    0.49x   true   (:f64acc default)
matmul 256x256x256          2.213      4.175    0.53x   true   (:f64acc default)
matmul 512x512x512         13.05      29.33     0.44x   true   (:f64acc default)
conv 8->16ch 24sq           0.376      0.431    0.87x   true
conv 16->32ch 32sq          6.263      1.433    4.37x   true
elementwise add 1M          7.408      1.534    4.83x   true
elementwise tanh 1M         2.003      1.007    1.99x   true
sum 1M (full)             253.414    132.985    1.91x   true
sum 1024x1024 axis0         1.373      0.732    1.88x   true
```

(matmul rows are the `:f64acc` default, so they read as losses — see §2 for the
fast `:f32acc` path.)

## 2. Accumulator race (3-way, `matmul_accumulator_race.exs`)

```
matmul       f64 ms   f32/f64acc (x)     f32/f32acc (x)
256x256x256   2.23      4.11 (0.54x)        2.75 (0.81x)
512x512x512  13.02     29.33 (0.44x)        7.31 (1.78x)
```

`:f32acc` is the win (1.78× at 512³); `:f64acc` is f64-ALU-bound (0.44×) — same
shape as the GT 650M.

## 3. Register blocking vs tiling (`matmul_rb_race.exs`) — the second-Kepler check

```
variant       size    tiled ms   rb32 ms   rb/tiled
f64           512³     13.24     17.99    0.74x
f64          1024³    101.2     121.08    0.84x
f32/f64acc    512³     30.49     25.70    1.19x
f32/f64acc   1024³    229.24    184.20    1.24x
f32/f32acc    512³      7.27      9.70    0.75x
f32/f32acc   1024³     54.17     58.50    0.93x
```

**RB regresses the paths that matter** (`f32/f32acc` 0.75–0.93×, `f64`
0.74–0.84×) on the GT 750M, confirming the GT 650M finding on a second Kepler
card. The one exception: RB **helps `f32/f64acc` (1.19–1.24×)** — that variant is
f64-ALU-bound, so the higher arithmetic intensity of register blocking pays off
there — but `f32/f64acc` isn't the speed path (if you want speed you pick
`:f32acc`, which RB slows). Net: keep plain tiling as the Kepler default.

## Split-effort takeaway

Kepler (GT 650M + GT 750M) and Ampere (RTX 3060 Ti) now bracket the design:
tiling is the right universal default; the `:f32` accumulator is a real
1.8–3.0× win on the fast path everywhere; register blocking helps Ampere
(~3% of f32 peak, headroom) but not Kepler — so RB should stay a device-aware /
opt-in kernel, not the default.
