# W4 — Warmup Curve Summary

Method: per-window cumulative timing via `Nx.Vulkan.Native.timing_get/0`.
Each window = 5 samples (~14-30 dispatches depending on tree depth).
`Warm @ window` = first window where p99 of the previous 50-window slice ≤ 1.5 × p50.

| Family | Cold window (µs) | Warm p50 (µs) | Warm p99 (µs) | Warm @ window | Total dispatches |
|---|---|---|---|---|---|
| Normal | 254242 | 284723 | 581756 | 20 | 2771 |
| Exponential | 916281 | 569223 | 941586 | 38 | 5034 |
| StudentT | 414523 | 326177 | 552703 | 20 | 6981 |
| HalfNormal | 596099 | 684116 | 1017005 | 20 | 8781 |
| Weibull | 553461 | 495897 | 889225 | 50 | 5769 |

Per-family CSVs in `bench/warmup_curves/{family}.csv`.
