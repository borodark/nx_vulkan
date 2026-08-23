# W5 kernels vs the fallbacks they replaced

`mix run examples/w5_kernels_race.exs` — median of 5 replicates, both arms
ending with the answer on the host. Reports here are at `8d9d23e`.

**Read the `box_was_busy` field before quoting any of this.** The harness samples
the load average around every race for a reason recorded below.

## Both Keplers, idle, no regressions

mac-248 (GT 750M) at baseline 0.62 and mac-247 (GT 650M) at baseline 0.32. Every
family W5 added is faster than the host path it replaced, on both:

| op | 248 gpu ms | 248 speedup | 247 gpu ms | 247 speedup |
|---|---:|---:|---:|---:|
| `add` s32 n=262144 | 0.84 | 94.5x | 0.95 | 129.3x |
| `add` s32 n=1048576 | 3.57 | 120.2x | 4.73 | 195.1x |
| `greater` s32 n=262144 | 2.72 | 29.8x | 2.66 | 44.5x |
| `select` s32 n=262144 | 3.43 | 22.8x | 3.58 | 33.9x |
| `sum` s32 512x512 axis 1 | 4.92 | 12.7x | 1.28 | 87.5x |
| `sum` s32 1024x1024 axis 1 | 2.88 | 93.1x | 3.27 | 148.9x |
| `window_sum` f32 512x512 {3,3} | 2.88 | 618.9x | 3.60 | 742.0x |
| `window_sum` f32 padded `:same` | 3.45 | 494.3x | 15.65 | 226.0x |
| `indexed_put` s32 k=8192 | 1.96 | 8.9x | 0.73 | 34.5x |
| `indexed_add` s32 k=8192 | 1.54 | 12.6x | 0.77 | 34.4x |
| `argmax` s32 512x512 axis 1 | 3.71 | 19.1x | 0.75 | 156.3x |
| `all` s32 512x512 axis 1 | 3.50 | 17.6x | 0.67 | 150.3x |
| `dot` s32 256x256 | 6.10 | 975.2x | 6.86 | 1470.5x |
| `dot` s32 512x512 | 8.56 | 6259x | 29.26 | 3114x |

The smallest win is `indexed_put` at 8.9x and the largest is integer `dot`, where
the host arm takes **53 seconds** (91 s on the 650M) for a 512x512 matmul the
shader does in under 30 ms. That gap is why the census counted `dot/7` as W5's
biggest item even though the shader closed only four doctests.

**One hardware-specific cost worth knowing.** Padding a window reduction is
nearly free on the GT 750M (3.45 ms against 2.88 unpadded, 1.2x) and expensive
on the GT 650M (15.65 against 3.60, **4.3x**) — spread was 2.8% on that row, so
it is not noise. The `if (!inside) continue` branch and the per-element bounds
arithmetic cost real time on the older part. Still 226x faster than the host, so
this is a note rather than a problem, but do not assume the padded and unpadded
paths cost the same.

## The follow-on items — stack, gather, bitcast, tensordot

Raced at `797ef57` (GT 650M, baseline 0.02) and `1e934c5` (GT 750M, baseline
0.43), both clean. Four of the five are GATE WIDENINGS on kernels that already
existed, so the question is not "is the shader fast" but whether the
NORMALISATION each inserts costs less than the host round trip it replaced.

| op | 650M gpu | 650M | 750M gpu | 750M |
|---|---:|---:|---:|---:|
| `stack` 2×512×512 axis 0 | 3.06 ms | 6.9x | 2.62 ms | 2.6x |
| `stack` 2×512×512 axis 1 | 4.31 ms | 3.0x | 4.80 ms | 1.9x |
| `take` axis 0 (prefix) | 0.80 ms | 8.7x | 0.75 ms | 5.4x |
| `take` axis 1 (**rotated**) | 2.13 ms | 73.5x | 2.17 ms | 49.7x |
| `bitcast` n=1048576 | 20.09 ms | 0.99x | 13.73 ms | 1.03x |
| `dot` r3 contract axis 1 | 3.14 ms | 167.9x | 1.55 ms | 233.4x |
| `dot` **batched** b=8 32×32 | 0.98 ms | 70.9x | 0.21 ms | 230.8x |
| `dot` **batched** b=64 16×16 | 0.33 ms | 491.4x | 0.22 ms | 224.6x |

**The rotation pays for itself many times over.** `take` at axis 1 costs ~2.7x
more GPU time than at axis 0 — that is the transpose, and it is real — but the
HOST arm costs 22x more at axis 1 than at axis 0, so the widened gate wins 73x
where refusing would have won nothing. Pricing a normalisation against the
unrotated GPU path rather than against the host is the mistake to avoid here.

**The batched kernel's third dispatch dimension behaves.** b=64 at 16×16 is the
adversarial case — many tiny matrices, where a z-dimension occupancy problem
would show — and it is the FASTEST row in the table on the 650M at 0.33 ms.

**`stack` is the weakest win in the whole suite** at 1.9–3.0x, and that is
expected: it is a copy either way and `BinaryBackend`'s concatenate is not
especially slow. It is still positive on both boxes and it keeps the result
resident, which this harness does not count.

**`bitcast` at ~1.0x is CORRECT, not a regression.** It is a relabel — same
buffer, new type — so there is nothing for a kernel to be fast at, and both arms
of this harness are then purely the transfer they both pay. The harness flagged
it REGRESSION on the first run; that was the harness lying about its own blind
spot, in the same direction as the load-check bug below, and it is now labelled
for what it is. The real win is residency, which the header says plainly is not
counted here.

## What the race found that no test could

`all`/`any` was **3.2x slower than its own siblings** at the same shape:

| GT 750M | before (`1e2daab`) | after (`635faf8`) |
|---|---:|---:|
| `sum` s32 512x512 | 3.94 ms | 4.92 ms |
| `argmax` s32 512x512 | 3.97 ms | 3.71 ms |
| **`all` s32 512x512** | **12.54 ms** | **3.50 ms** |

Confirmed independently on the GT 650M, where the effect is larger: `all` went
from 2.39 ms against `argmax`'s 0.90 (2.7x slower) to **0.67 against 0.75** —
now the fastest of the three.

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

## A third false finding, this one from the harness itself

The first load check judged contamination on MID-RUN load, and it flagged a
genuinely idle mac-247 run — baseline 0.00 / 0.08 / 0.45, only `epmd` on the
box — as contaminated. The cause was this benchmark: its host arm is a
single-threaded Elixir loop over `Nx.BinaryBackend` and drives the 1-minute
average past 1.5 by itself.

That is the more dangerous of the two failures. Admitting a bad run is
recoverable, because someone re-runs it. Discarding a good one costs the
measurement *and* teaches the reader to ignore the banner, which is how a
warning stops working. The verdict now comes from a single sample taken before
the script does anything (`baseline_load` in the JSON); per-row load is still
recorded for spotting a mid-run spike, but decides nothing.
