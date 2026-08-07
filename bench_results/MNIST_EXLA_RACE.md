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

## Finding 3 — CORRECTED: EXLA runs conv fine; one narrow graph fails

**An earlier version of this file said the conv comparison "does not exist"
because EXLA "failed to compile it". That was wrong, and wrong in the direction
that flatters this project.** It was written from a single failing run without
isolating the cause. Isolating it (17 variants) shows EXLA compiles and trains
convolutional models normally.

The failure needs **three conditions at once**, and is in the GRADIENT only:

| variant | forward | gradient |
|---|---|---|
| 2 conv, stride 2, `channels: :first` | OK | **FAIL** |
| 2 conv, stride 1, `channels: :first` | OK | OK |
| 2 conv, stride 2, `channels: :last` | OK | OK |
| 2 conv, stride 1, `channels: :last` | OK | OK |
| 1 conv, any stride, either layout | OK | OK |

So: **two stacked convs + stride 2 + `channels: :first`, backward pass.** Any
one relaxation compiles. The loss function is irrelevant — it fails identically
with softmax+cross-entropy, plain cross-entropy, and `Nx.sum`.

`channels: :last` is **Axon's default**, so a user following Axon's own guides
would not hit this. The original race used `:first` because MNIST is naturally
NCHW and because this project's notebook documents `:first` as required here —
a configuration choice from *this* side meeting an XLA edge, not an EXLA
weakness at conv.

Worth noting the symmetry, since it is this project's own recent history:
**both backends' conv problems were in the gradient, not the forward pass.**
nx_vulkan's GPU gate rejected the permuted convs `Nx.Defn.Grad` emits (fixed in
`fb6221d`); XLA's symbolic tiler cannot tile one particular gradient conv shape.
Autodiff generates graphs that neither forward path anticipated.

## The conv comparison, which does exist

Same harness, 2× strided conv → flatten → dense softmax, `channels: :last`,
cross-entropy, batch 32:

| backend / compiler | ms | loss |
|---|---:|---|
| BinaryBackend / `Nx.Defn.Evaluator` | 14658.032 | 2.470658779144287 |
| Vulkan / `Nx.Defn.Evaluator` (eager) | 41.325 | 2.470658779144287 |
| Vulkan / `Nx.Vulkan.Compiler` (fused) | 42.222 | 2.470658779144287 |
| EXLA / `EXLA` (CUDA) | 1.448 | 2.4706473350524902 |

| ratio | |
|---|---:|
| vulkan eager vs BinaryBackend | 354.7× |
| fused vs eager | 0.98× |
| exla vs vulkan eager | 28.54× |
| exla vs vulkan fused | 29.16× |

The conv graph does not rescue fusion either — 0.98×, neutral rather than the
0.76× regression on the MLP, but still not a win. And EXLA's lead is slightly
*wider* here (28.5×) than on the dense-only MLP (19.8×), which contradicts the
intuition that conv-heavy graphs would favour a hand-written im2col+GEMM.

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
