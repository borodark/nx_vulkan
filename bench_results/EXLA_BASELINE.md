# EXLA baseline — VulkanoBackend vs EXLA vs BinaryBackend (thrust 1)

super-io (249), NVIDIA **RTX 3060 Ti** (Ampere), 2026-07-30. First real
head-to-head, via `examples/backend_baseline.exs` (eager / per-op Nx, f32).

```
workload                 binary ms   vulkano ms    exla ms   vs EXLA        correctness
matmul 256x256           23462.12      20.71       35.04     Vulkano 1.7x   vulk exact, exla<8e-3
conv 2x8x16x16 k16         313.48       4.33        2.36     EXLA 1.8x      vulk exact, exla<4e-6
tanh 100k                   50.80      21.59       20.98     parity         both exact
sum 256x256                  9.89       3.74        0.67     EXLA 5.6x      vulk exact, exla<2e-5
mlp fwd 64x128->128->10    417.97       3.68        2.13     EXLA 1.7x      vulk exact, exla<3e-2
```

## Re-run after thrust-2 (2026-07-31, @ af7292d) — NN forward fully on-GPU

Same RTX 3060 Ti, after broadcast/select/compare/cast/clip landed (forward +
relu-grad mask on-device):

```
workload                 binary ms   vulkano ms    exla ms   vs EXLA
matmul 256x256           23570.19      20.64       37.67     Vulkano 1.8x
conv 2x8x16x16 k16         302.42       3.97        2.23     EXLA 1.8x
tanh 100k                   53.73      21.67       21.05     ~parity
sum 256x256                 10.06       3.82        0.68     EXLA 5.6x
mlp fwd 64x128->128->10    419.36       3.22        2.23     EXLA 1.45x
```

**mlp fwd closed from ~1.7x to ~1.45x** vs EXLA (Vulkano 3.68 -> 3.22 ms) purely
from keeping the forward on the GPU — no more host round-trips for bias/relu/
softmax. Still ~130x over pure-Elixir Binary. The remaining EXLA edge is the
reduction (`sum` 5.6x) and its compiled/fused kernels — thrust 3.

## Thrust-3 parallel fused reduce (2026-07-31, @ 7ef2c8a) — RTX 3060 Ti

The fusion compiler's parallel workgroup-per-slot tree reduce, validated on the
Ampere box (device: RTX 3060 Ti). Fused = `Nx.Defn.jit(_, compiler:
Nx.Vulkan.Compiler)`; exact to BinaryBackend (err 0.0).

```
workload                       eager ms   fused ms   speedup
sum(a*b) 256x256                 12.13      2.18       5.55x
sum(a*b) 1024x1024              103.67     15.82       6.55x
sum(tanh(a*b+a)) 512x512         45.79     15.82       2.89x
sum axes:[1] 2048x256 (many)      3.89      3.49       1.11x  (eager fallback)
```

Speedups are smaller than the GT 650M's (9.9x / 27x / 8.5x) because Ampere's
eager path is relatively much faster — the fused kernel is the same win, the
baseline it beats is higher. Elementwise-fusion bench on the same box: the 10-op
n=1e6 chain is 31.35 -> 23.30 ms (1.35x, vs 3.62x on Kepler — same reason).

### Many-slot fused reduce is hardware-dependent (fleet finding)

The grid-stride many-slot wide-reduce regime (slots >= 2048, reduce >= 256) that
wins ~4.4x on the GT 650M **regresses on the 3060 Ti**: `sum axes:[1]` 4096x256
measured **0.44x** (eager 6.53 -> fused 14.76 ms) and 16384x256 **0.81x** — exact,
but slower. Ampere's one-thread-per-slot eager reduce is already well-fed by
thousands of slots, leaving the fused kernel no headroom. So the many-slot regime
is **opt-in only** (`NXV_FUSE_REDUCE=1`); the default keeps just the few-slot
regime, which wins on both GPUs (full `sum` 256x256: 2.78x, 1024²: 6.72x,
`sum(tanh(a*b+a))` 512²: 6.07x on the 3060 Ti). Lesson: perf heuristics must be
fleet-validated — the win/loss crossover is HW-specific.

### Vulkano vs EXLA head-to-head on `Nx.sum` (f32, RTX 3060 Ti)

```
shape        Vulkano eager   Vulkano fused   EXLA JIT
256x256        7.89 ms         1.05 ms        0.708 ms
1024x1024    117.28 ms         8.26 ms        0.475 ms
```

The fused reduce takes `sum 256x256` from ~11x behind EXLA (eager) to **~1.5x**
(1.05 vs 0.708 ms) — most of the gap closed. EXLA's compiled reduction still
leads, more so at 1024x1024; note EXLA's 1024² (0.475) being *faster* than its
own 256² (0.708) means those absolute figures are XLA launch-overhead-bound, not
throughput-bound, so read them as "EXLA still ahead on tiny reductions," not a
17x throughput deficit. Correctness: Vulkano fused == eager exactly; EXLA within
f32 tolerance. (OTP differs between the pure-nx_vulkan run (kerl 26) and the
EXLA/bench249 run (kerl 27) per each harness's config.)

## Read

**VulkanoBackend is in EXLA's league for eager execution** — same order of
magnitude across the board, *faster* than EXLA on matmul at this size (per-op
dispatch dominates and Vulkano's is lean), ~1.7–1.8x behind on conv / mlp, 5.6x
behind only on small reductions (EXLA's reduction kernels + no per-op wait), and
at parity on elementwise. Both are 100–1000x over pure-Elixir BinaryBackend.
Vulkano matches BinaryBackend exactly; EXLA differs within f32 tolerance
(tf32 / different reduction order) — expected, not a defect.

**Caveat — this is eager, not compiled.** EXLA's real advantage is `Nx.Defn`
whole-graph compilation + fusion, which this per-op harness does not exercise.
On a fused defn graph EXLA pulls ahead; closing that is thrust 3. For the eager
path that most Nx code and interactive use hits, VulkanoBackend is competitive —
and it runs on hardware/OSes EXLA can't (the moat).

## Reproducing EXLA on super-io (the setup fix)

The hex `exla` fails to build from source here (an XLA-FFI header incompat —
`OutputBuffer()` no matching ctor; this is what the `_nx-exla-fix` checkout
addresses). Use the **working `~/projects/learn_erl/nx/exla` checkout** (its
cached `libexla.so`, xla-0.10.0/exla-0.13.0), and fix the NIF `dlopen`:

1. The NIF needs CUDA libs from the python nvidia wheels, not on the default
   path. Add them: `LD_LIBRARY_PATH=$(printf %s: ~/.local/lib/python3.12/site-packages/nvidia/*/lib)$LD_LIBRARY_PATH`.
2. Version skew: it wants `nvshmem_transport_ibrc.so.3` but the wheel ships
   `.so.4`. Symlink it (the IB transport is unused on this non-IB box):
   `ln -sf nvshmem_transport_ibrc.so.4 .../nvidia/nvshmem/lib/nvshmem_transport_ibrc.so.3`.
3. `XLA_TARGET=cuda12`, and `~/.cargo/bin` on PATH for the nx_vulkan Rust NIF.

The three-way runs from `~/bench249` (a project with path deps on the monorepo
`nx`, the `nx/exla` checkout, and `nx_vulkan`); see `~/bench249/run.sh`.
