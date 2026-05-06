# W4 — Warmup Curve Summary

Method: per-window cumulative timing via `Nx.Vulkan.Native.timing_get/0`.
Each window = 5 samples (~14-30 dispatches depending on tree depth).
`Warm @ window` = first window where p99 of the previous 50-window slice ≤ 1.5 × p50.

| Family | Cold window (µs) | Warm p50 (µs) | Warm p99 (µs) | Warm @ window | Total dispatches |
|---|---|---|---|---|---|
| Normal | 26248 | 15530 | 25571 | 20 | 2771 |
| Exponential | 37109 | 27171 | 37697 | 20 | 5034 |
| StudentT | 28971 | 25524 | 35188 | 20 | 6981 |
| HalfNormal | 56530 | 45175 | 62081 | 20 | 8781 |
| Weibull | 33331 | 28933 | 50941 | 20 | 5769 |

Per-family CSVs in `bench/warmup_curves/{family}.csv`.
