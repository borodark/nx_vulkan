# MNIST race — EXLA vs VulkanoBackend (eager and fused)

The [Axon MNIST guide](https://axon.hexdocs.pm/mnist.html) model, real MNIST,
one `value_and_grad` training step at batch 32, on super-io (RTX 3060 Ti,
Linux, CUDA 12.6, driver 580.173.02). Commit `4827c82`.

```elixir
Axon.input("input", shape: {nil, 1, 28, 28})
|> Axon.flatten()
|> Axon.dense(128, activation: :relu)
|> Axon.dense(10, activation: :softmax)
```

Best of three after a warm-up (EXLA compiles on first call). Losses are
reported so a NaN or a diverged run cannot hide inside a timing.

| backend / compiler | ms | loss |
|---|---:|---|
| BinaryBackend / `Nx.Defn.Evaluator` | 6850.375 | 2.3268656730651855 |
| **Vulkan / `Nx.Defn.Evaluator` (eager)** | **14.140** | 2.3268656730651855 |
| **Vulkan / `Nx.Vulkan.Compiler` (fused)** | **18.509** | 2.3268656730651855 |
| **EXLA / `EXLA` (CUDA)** | **0.715** | 2.326899528503418 |

| ratio | |
|---|---:|
| vulkan eager vs BinaryBackend | 484.5× |
| vulkan fused vs BinaryBackend | 370.1× |
| exla vs BinaryBackend | 9580.9× |
| **fused vs eager (vulkan)** | **0.76×** |
| exla vs vulkan eager | 19.78× |
| **exla vs vulkan fused** | **25.89×** |

Both Vulkan paths are **bit-identical** to `BinaryBackend`. EXLA differs in the
6th decimal — its fusion reassociates the arithmetic, which is expected.

## Finding 1 — fusion REGRESSES on this graph (0.76×)

`Nx.Vulkan.Compiler` is **24% slower than eager dispatch** here, and correct
while doing it (identical loss). This was not the expected result: the race was
run to see how much of the EXLA gap fusion closes, and it widens it, from
19.78× to 25.89×.

The reason is structural and consistent with the compiler's design.
`Nx.Vulkan.Compiler` splits stages at `dot` boundaries, and
`flatten → dense → relu → dense → softmax` is almost entirely `dot`s with thin
elementwise work between them. There is nothing for the tracing, stage
scheduling, and boundary buffers to amortise against. The README's claim that
fusion's win "grows with the elementwise work around the boundary" has a floor,
and this measures it: **below 1.0 when that work approaches zero.**

This is the same shape as the cross-stage CSE result
([`CSE_SOFTMAX_RACE.md`](CSE_SOFTMAX_RACE.md)) — an optimisation that is
principled, correct, and a regression on the wrong graph. The lesson repeats:
**measure per graph shape; do not assume an optimisation is free.**

## Finding 2 — the EXLA gap is not a fusion gap

Since fusion makes it worse, the ~20× deficit on a matmul-dominated graph is
**dispatch overhead and GEMM kernel quality**, not missing whole-graph
compilation. At 0.715 ms EXLA is near the floor for a model this small; both
GPU figures are dominated by per-dispatch cost, which is exactly what XLA's
fusion removes and what eager dispatch pays repeatedly.

Anyone reading the first two rows and concluding "we need more fusion" would
build the wrong thing. The work that would close this gap is fewer, larger
dispatches and a better GEMM — not a broader fusion pass.

## Finding 3 — the comparison does not exist for conv

The same race on a strided-conv CNN could not be run: **EXLA failed to compile
it**, with

```
Failed to analyze the computation (Failed to compute symbolic tile for
 (d0, d1, d2, d3) -> (d2, (d1 * 28 + d0) floordiv 49, ...)
```

identically for a hand-written `Nx.conv` graph and for Axon's `Axon.conv`, so
it is not an artifact of how the model was written. The environment was fully
working at that point — NIF loaded, CUDA and nvshmem resolved. Vulkan ran the
same CNN in **29.6 ms** on this host, and on the two FreeBSD Keplers where EXLA
cannot be installed at all.

## Setup cost, recorded because it is part of the comparison

Getting EXLA to run on this box took four interventions:

1. **EXLA 0.13.0 will not build from source** under gcc 13.3 — its own
   `c_src/exla/custom_calls/runtime_callback_bridge.h` needs a default
   constructor for `OutputBuffer`. `CXX=g++-12` does not help (nvcc selects its
   own host compiler).
2. The prebuilt 0.13.0 from the exmc release tree loads but wants
   `libnvshmem_host.so.3` — Ubuntu 24.04 ships `.so.6`. Resolved from a pip
   venv, as recorded in that project's own build notes.
3. Then it wants `libnvrtc-builtins.so.12.9`; the box has CUDA 12.6. Resolved
   from a second venv (`~/xla-cuda`).
4. Axon is absent from the tree that has a working EXLA, so the race runs from
   a scratch project with EXLA borrowed at runtime via `ERL_LIBS`.

`nx_vulkan` compiled with `mix compile` on all three fleet hosts, two of them
FreeBSD. That asymmetry is not a benchmark result, but it is the project's
premise showing up as a measurement rather than a claim.

## Reproducing

The race is a scratch harness, not a committed example — it needs a working
EXLA, which this repo deliberately does not depend on. Sketch:

```sh
# a scratch mix project with {:axon, "~> 0.7"}, {:req, "~> 0.5"},
# {:nx_vulkan, path: "..."} — EXLA comes in at run time, unbuilt:
NVLIBS=$(ls -d ~/xla-cuda/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')
LD_LIBRARY_PATH="${NVLIBS}~/nvshmem-venv/.../nvshmem/lib:$LD_LIBRARY_PATH" \
ERL_LIBS="arena/_build/dev/lib:/path/to/tree/with/built/exla/_build/dev/lib" \
XLA_TARGET=cuda12 elixir mnist_race.exs
```

Compare `Nx.Defn.jit_apply(step, args, compiler: C)` for
`C ∈ {Nx.Defn.Evaluator, Nx.Vulkan.Compiler, EXLA}` with the backend moved to
match, and assert the loss is not NaN before believing any timing — an earlier
version of this race reported a 635× figure from a model that was producing
NaN.
