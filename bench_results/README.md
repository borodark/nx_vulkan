# VulkanoBackend bench results — git-tracked

Each CSV is the output of `examples/vulkano_ops_bench.exs` on one host
on one date. Columns:

```
op_class, op_name, shape, n_reps, vulkano_us_median, vulkano_us_p95,
binary_us_median, binary_us_p95, speedup
```

`speedup = binary_us_median / vulkano_us_median`. >1 = vulkano wins.

## Files

| File | Host | GPU | OS |
|---|---|---|---|
| `super-io_2026-05-22.csv` | super-io | NVIDIA RTX 3060 Ti (Ampere) | Linux 6.8 |
| `free-macpro-nvidia_2026-05-22.csv` | mac-248 | NVIDIA GT 750M (Kepler) | FreeBSD 15.0 |

## Latest (post-Tier-1) summary medians

| class | super-io n wins / total | super-io median | mac-248 n wins / total | mac-248 median |
|---|---|---|---|---|
| binary | 48/49 | 56.17× | 48/49 | 68.48× |
| unary | 54/70 | 13.75× | 60/70 | 16.23× |
| linalg | 5/5 | 88.04× | 5/5 | 54.69× |
| reduction | 15/21 | 1.38× | 2/21 | 0.45× |
| movement | 3/5 | 1.00× | 2/4 | 7.59× |
| storage | 2/11 | 0.85× | 2/11 | 0.93× |
| compare | 0/42 | 0.59× | 7/42 | 0.93× |
| host | 4/32 | 0.74× | 12/32 | 0.99× |
| sampler-host | 0/45 | 0.12× | 1/45 | 0.22× |

## Reading the table

- Compute-bound classes (binary, unary, linalg) — solid vulkano wins
  on both hosts. RTX 3060 Ti pulls further ahead at larger sizes
  (compute-bound regime).
- Reductions — super-io wins by ~1.4×; mac-248 doesn't have the
  memory bandwidth to make them worthwhile (median 0.45×).
- Sampler-host (pad/put_slice/indexed_put/...) — always loses to
  BinaryBackend in the op-only bench because the host-fallback
  round-trip dominates. Tier 1 of SHAPE_C_PLAN.md still moves
  the consumer-bench needle by ~1.25-1.3× (when `to_flat_list` reads
  the result) — that win is captured in `examples/vulkano_consumer_bench.exs`,
  not this op-only bench.
- mac-248 (FreeBSD + GT 750M) beats super-io at *small* sizes for
  dispatch-bound work — the FreeBSD-NVIDIA driver path has lower
  per-dispatch overhead. Inverts at ≥256k for compute-bound ops.

## Caveats

These post-Tier-1 numbers were taken while the exmc `mix test` sweep
was also running on super-io. That CPU/memory contention adds noise
to host-fallback rows (the ones that hit BinaryBackend). mac-248 ran
alone with no contention; its numbers are cleaner. For tight
comparisons across runs, run the bench on a quiescent host.

## Consumer-aware bench (post-Tier-2)

`examples/vulkano_consumer_bench.exs` measures host-fallback ops
the way a real caller experiences them — including the read-back the
caller will eventually need, and (in the worst case) the upload-back
the old host-fallback path imposed. Output CSVs use a different schema:

```
op_name, size, A_bin_us, B_bin_rd_us, C_vulk_us,
D_vk_rd_us, E_vk_up_rd_us, D_over_B, D_over_E
```

| col | what it counts |
|---|---|
| `A_bin_us`      | BinaryBackend, no transfers |
| `B_bin_rd_us`   | BinaryBackend + read-back (consumer model) |
| `C_vulk_us`     | VulkanoBackend op-only |
| `D_vk_rd_us`    | VulkanoBackend + read-back the consumer pays |
| `E_vk_up_rd_us` | VulkanoBackend + upload-back-after-host-fallback (worst case) |
| `D_over_B`      | `D/B` &mdash; the GPU-vs-host comparison that matters |
| `D_over_E`      | `D/E` &mdash; how much Tier 1 saved by not uploading back |

| File | Host | GPU | OS |
|---|---|---|---|
| `super-io_consumer-aware_2026-05-24.csv` | super-io | NVIDIA RTX 3060 Ti | Linux 6.8 |
| `mac247-gt650m_consumer-aware_2026-05-24.csv` | mac-247 | NVIDIA GT 650M | FreeBSD 15.0 |

Tier 2 step 1 (commit `672d32f`) replaced the host-fallback
`concatenate` with a native `vkCmdCopyBuffer` path. Concatenate's
`D_over_B` now stays below 1.0 across the size range on both hosts.
The remaining host-fallback rows (pad/put_slice/indexed_put/broadcast/gather/take)
still benefit from the Tier 1 contract: `D_over_E > 1` on every row
means the Tier 0 upload-back was always wasted.
