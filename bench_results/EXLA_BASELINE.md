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
