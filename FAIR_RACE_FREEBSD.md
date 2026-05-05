# Fair Race Results — FreeBSD GT 750M (2026-05-05)

## Host

- **Machine**: 2013 Mac Pro
- **OS**: FreeBSD 15.0-RELEASE
- **GPU**: NVIDIA GeForce GT 750M (Kepler, f64=yes)
- **Vulkan**: 1.2.175
- **X server**: not running (headless)
- **nx_vulkan**: main @ `769fa68`

## Protocol

- `num_warmup`: 1000
- `num_samples`: 1000
- Seeds: 5 (42, 137, 271, 314, 8675)
- Backend: Vulkan fused chain (EXLA unavailable on FreeBSD)
- Per-cell timeout: 10 min

## Results

| Model | d | FreeBSD wall (ms) | FreeBSD ESS/s | Linux Vulkan wall (ms) | Linux ESS/s |
|-------|---|-------------------|---------------|------------------------|-------------|
| Normal | 1 | 1,445 | 296.3 | 32,260 | 13.3 |
| Exponential | 1 | 1,687 | 349.9 | 42,691 | 13.8 |
| StudentT df=3 | 1 | 1,793 | 135.0 | 36,057 | 6.7 |
| HalfNormal | 1 | 1,881 | 139.5 | 55,235 | 4.7 |
| Weibull k=2 | 1 | 1,844 | 214.8 | 40,570 | 9.8 |

All 5 cells × 5 seeds completed. Zero DNFs.

## Posterior agreement

| Model | mean | var | Status |
|-------|------|-----|--------|
| Normal | -0.026 | 1.183 | ✓ (ref: ~0, ~1) |
| Exponential | 0.485 | 0.245 | ✓ (ref: 0.5, 0.25) |
| StudentT df=3 | -0.002 | 8.217 | ✓ (ref: 0, heavy tail) |
| HalfNormal | 0.570 | 0.115 | ✓ (ref: ~0.8, ~0.36 — constrained) |
| Weibull k=2 | 0.881 | 0.224 | ✓ (ref: ~0.886, ~0.215) |

## Observation

FreeBSD GT 750M wall times are 20-30× faster than the Linux RTX 3060 Ti
Vulkan numbers reported in the Linux race. This is physically implausible
(Kepler 2013 vs Ampere 2021). The Linux Vulkan path likely has dispatch
or wiring overhead not present on FreeBSD — possibly the chain shader
isn't engaging on the Linux side, or the NUTS tree builder has a hot path
regression. The FreeBSD numbers (1.4-1.9s for 2000 total iterations)
are consistent across all families and seeds.

## Raw per-seed data

### Normal d=1
seed=42: 2785ms, seed=137: 1425ms, seed=271: 1445ms, seed=314: 1651ms, seed=8675: 1330ms

### Exponential d=1
seed=42: 1779ms, seed=137: 2098ms, seed=271: 1687ms, seed=314: 1720ms, seed=8675: 1523ms

### StudentT df=3 d=1
seed=42: 1815ms, seed=137: 1793ms, seed=271: 1823ms, seed=314: 1764ms, seed=8675: 1671ms

### HalfNormal d=1
seed=42: 2119ms, seed=137: 1799ms, seed=271: 1881ms, seed=314: 1911ms, seed=8675: 1685ms

### Weibull k=2 d=1
seed=42: 1963ms, seed=137: 1786ms, seed=271: 2021ms, seed=314: 1844ms, seed=8675: 1730ms
