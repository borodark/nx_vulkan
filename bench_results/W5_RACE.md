# W5 kernels vs the fallbacks they replaced

`mix run examples/w5_kernels_race.exs` — median of 5 replicates, both arms
ending with the answer on the host. Reports here are at `8d9d23e`.

**Read the `box_was_busy` field before quoting any of this.** The harness samples
the load average around every race for a reason recorded below.

## The trustworthy run: mac-248 (GT 750M), idle at 0.62 throughout

No regressions. Every family W5 added is faster than the host path it replaced:

| op | gpu ms | host ms | speedup |
|---|---:|---:|---:|
| `add` s32 n=262144 | 0.84 | 79.43 | **94.5x** |
| `add` s32 n=1048576 | 3.57 | 429.16 | **120.2x** |
| `greater` s32 n=262144 | 2.72 | 81.06 | **29.8x** |
| `select` s32 n=262144 | 3.43 | 78.09 | **22.8x** |
| `sum` s32 512x512 axis 1 | 4.92 | 62.27 | **12.7x** |
| `sum` s32 1024x1024 axis 1 | 2.88 | 268.58 | **93.1x** |
| `window_sum` f32 512x512 {3,3} | 2.88 | 1779.01 | **618.9x** |
| `window_sum` f32 padded `:same` | 3.45 | 1705.94 | **494.3x** |
| `indexed_put` s32 k=8192 | 1.96 | 17.39 | **8.9x** |
| `indexed_add` s32 k=8192 | 1.54 | 19.45 | **12.6x** |
| `argmax` s32 512x512 axis 1 | 3.71 | 70.70 | **19.1x** |
| `all` s32 512x512 axis 1 | 3.50 | 61.49 | **17.6x** |
| `dot` s32 256x256 | 6.10 | 5944.11 | **975.2x** |
| `dot` s32 512x512 | 8.56 | 53550.93 | **6259x** |

The smallest win is `indexed_put` at 8.9x and the largest is integer `dot`, where
the host arm takes **53 seconds** for a 512x512 matmul that the shader does in
under 9 ms. That gap is why the census counted `dot/7` as W5's biggest item even
though the shader closed only four doctests.

## What the race found that no test could

`all`/`any` was **3.2x slower than its own siblings** at the same shape:

| | before (`1e2daab`) | after (`635faf8`) |
|---|---:|---:|
| `sum` s32 512x512 | 3.94 ms | 4.92 ms |
| `argmax` s32 512x512 | 3.97 ms | 3.71 ms |
| **`all` s32 512x512** | **12.54 ms** | **3.50 ms** |

Same NIF, same shape, same dispatch as `sum` and `argmax`. The first version of
`allany_*.comp` gave each thread FOUR output slots so it could build a packed u32
word locally, and the commit that shipped it said reusing `reduce_axis/7` was
"worth the idle threads". It was not — four slots per thread is 4x the serial
work *and* a quarter of the parallelism. Rewritten to one thread per slot with
`atomicOr` (core GLSL 4.30; the output buffer arrives zeroed from `buf_alloc`, so
only true bits are written).

The shader was correct, the doctests were green, and residency was unchanged at
670/833. **Nothing in the test suite would ever have reported this.** Only the
race did — which is the same lesson `docs/BACKWARD_PASS_AUDIT.md` records, turned
on W5's own work.

## Why the load sampling exists

Two false findings, both from busy boxes, both plausible-looking:

* **mac-247 reported four regressions** — `greater` 0.43x, `select` 0.28x, `sum`
  0.4x — while `argmax` and `all`, which use the *same* NIF at the *same* shape,
  came in at 0.9 ms and 2.4 ms. No hardware story explains that; an eXMC build
  sharing the box does. Re-run: 40.7x, 99.3x, 97.6x.
* **super-io reported `add` at n=1048576 taking 245 ms** against 2.66 ms at a
  quarter the size — exactly the shape of a memory cliff. Without the competing
  `mix test` it is 8.74 ms.

Neither was real, both were quotable, and one of them was already written into a
draft as a finding. The harness now labels a slow row on a loaded box as
"RE-RUN IDLE" rather than REGRESSION, flags any op whose replicates spread more
than 50%, and sets `box_was_busy` in the JSON so a stale report cannot be quoted
later as if it were clean.

## Still owed

mac-247 has not produced a fully idle run — it started at 0.66 and rose past 1.5
mid-race both times, because that box also hosts eXMC dependency builds. Its
numbers corroborate mac-248 on every op and confirm the `all` fix (4.08 ms
against `argmax`'s 4.41 ms, where it was 2.39 against 0.90 before), but the
report is marked busy and should be re-run in a genuinely quiet window before
being cited on its own.
