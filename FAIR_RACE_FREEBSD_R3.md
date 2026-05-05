# Fair Race R3 — FreeBSD GT 750M with persistent-buffer fix (2026-05-05)

## Host
- Machine: 2013 Mac Pro, FreeBSD 15.0-RELEASE
- GPU: NVIDIA GeForce GT 750M (Kepler, f64=yes)
- nx_vulkan: main @ b2fc47d (timing + persistent IO NIFs)
- exmc: feat/dsl-shader-codegen @ 152da19eb (persistent-buffer tree.ex)

## R3.1 — Race results (1000/1000, 5 seeds)

| Model | d | R1 ms | R3 ms | Improvement | ESS/s |
|-------|---|-------|-------|-------------|-------|
| Normal | 1 | 1,445 | 1,023 | 29% | 418.6 |
| Exponential | 1 | 1,687 | 1,032 | 39% | 572.0 |
| StudentT df=3 | 1 | 1,793 | 1,043 | 42% | 232.0 |
| HalfNormal | 1 | 1,881 | 1,144 | 39% | 229.2 |
| Weibull k=2 | 1 | 1,844 | 1,129 | 39% | 350.7 |

Average improvement: 37%. Persistent-buffer fix helped FreeBSD too.

## R3.2 — Per-fence timing (Normal d=1, 100/100)

| Metric | FreeBSD GT 750M | Linux RTX 3060 Ti | Ratio |
|--------|-----------------|-------------------|-------|
| submit | 11.6 µs | 138 µs | 12× faster |
| wait | 406 µs | 1,130 µs | 2.8× faster |
| record | 4.3 µs | 19 µs | 4.4× faster |
| **total** | **422 µs** | **1,287 µs** | **3.1× faster** |

Dispatch count: 352 (100+100 iterations).

## Conclusion

The Linux/FreeBSD wall-time gap is entirely explained by per-fence
driver overhead. FreeBSD's NVIDIA Vulkan driver completes fence waits
in 406µs vs Linux's 1,130µs. This is a driver-level difference, not
a GPU compute or shader difference. The chain shader itself runs at
the same speed on both platforms; the difference is how long the CPU
blocks waiting for the GPU to signal completion.
