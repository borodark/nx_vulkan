# W4 — Warmup Curve Summary

Method: per-window cumulative timing via `Nx.Vulkan.Native.timing_get/0`.
Each window = 5 samples (~14-30 dispatches depending on tree depth).
`Warm @ window` = first window where p99 of the previous 50-window slice ≤ 1.5 × p50.

| Family | Cold window (µs) | Warm p50 (µs) | Warm p99 (µs) | Warm @ window | Total dispatches |
|---|---|---|---|---|---|
| Normal | 20494 | 10938 | 20941 | 50 | 2771 |
| Exponential | 30831 | 27288 | 37845 | 20 | 5034 |
| StudentT | 29138 | 26078 | 36587 | 20 | 6981 |
| HalfNormal | 45791 | 45279 | 58196 | 20 | 8781 |
| Weibull | 23391 | 28099 | 40736 | 20 | 5769 |

Per-family CSVs in `bench/warmup_curves/{family}.csv`.
