# Batched command submission — T1

`PLAN_AFTER_BACKWARD_PASS.md` T1. Every op used to be its own command buffer,
its own `vkQueueSubmit`, and its own `queue.wait_idle()`. Dispatches are now
recorded into a pending queue and submitted as one command buffer with one
fence wait, flushed at every host boundary (`buf_download`, `buf_upload_into`,
`concat_buffers`, `fft`) and at a cap (`NXV_BATCH_MAX`, default 64).

Why this and not more fusion: `MNIST_EXLA_RACE.md` measured the eager step at
14.140 ms against EXLA's 0.715 ms **and** measured whole-graph fusion making it
*worse* (0.76×). An optimisation that removes work from the shaders cannot
explain a gap that fusing the shaders widens, so the deficit is per-dispatch
cost.

Commit `1dbba7f`, raced across the whole fleet — SHA verified on each host
after `git merge --ff-only`, because `git checkout` alone leaves a
pre-existing local branch stale and silently benchmarks the wrong code.

| host | GPU | arch | OS |
|---|---|---|---|
| super-io | RTX 3060 Ti | Ampere | Linux |
| mac.247 | GeForce GT 650M | Kepler | FreeBSD 15.0 |
| free-macpro-nvidia (248) | GeForce GT 750M | Kepler | FreeBSD 15.0 |

## Result — MNIST MLP training step, all three hosts

`examples/mnist_mlp_step_bench.exs` — the Axon MNIST MLP written out in plain
Nx (flatten → dense 128 + relu → dense 10 + softmax → categorical
cross-entropy), one `Nx.Defn.value_and_grad` step at batch 32, under
`Nx.Defn.Evaluator`. Each arm is a separate `mix run` (the env var is read once
per OS process); each cell is best-of-5 blocks of 20 steps, two runs per arm.

| `NXV_BATCH_MAX` | super-io (Ampere) | mac.247 (GT 650M) | 248 (GT 750M) |
|---|---|---|---|
| 0 — **control, submit per dispatch** | 16.446, 20.828 | 14.583, 16.547 | 13.473, 13.301 |
| 32 | 9.503, 9.439 | 8.827, 8.930 | 9.116, 9.222 |
| 64 (default) | 9.627, 12.921 | 8.829, 9.057 | 9.147, 9.275 |
| 256 | 10.104, 9.972 | 12.107, 8.895 | 9.338, 9.330 |

Speedup at the default cap, control min ÷ batched min:

| host | control (ms) | batched (ms) | |
|---|---:|---:|---|
| super-io (Ampere) | 16.446 | 9.627 | **1.71×** |
| mac.247 (Kepler GT 650M) | 14.583 | 8.829 | **1.65×** |
| 248 (Kepler GT 750M) | 13.301 | 9.147 | **1.45×** |

**The loss is `2.6447360515594482` in every cell of that table** — every arm,
every cap, all three hosts, two architectures, two operating systems. Batching
changes when work is submitted, never what is computed.

**No hardware crossover.** This is the outcome that could not be assumed:
register blocking wins on Ampere and regresses on both Keplers, and the
many-slot fused reduce wins 4.4× on Kepler and regresses to 0.44× on Ampere,
which is why `Nx.Vulkan.Device.class/0` exists. Batching wins on all three, so
it needs no device-class gate and ships on by default.

The cap is flat from 32 to 256 on every host, so the default of 64 is not a
tuned value and does not need to be. Larger caps hold more descriptor sets
alive at once, which is pool churn rather than a hard limit (vulkano allocates
additional pools and recycles them through a reserve) — the concern going in
was that Kepler would show this as a regression at 256, and it does not.

Run-to-run noise is several ms, so single-sample A/B would not have been
trustworthy: an early sweep read `NXV_BATCH_MAX=1` (one dispatch per submit —
the control's work through the batched code path) as 3.7 ms faster than the
control, which is noise, not a finding. The two outliers above (super-io 12.921
at cap 64, mac.247 12.107 at cap 256) are that same noise; the control and
batched ranges still do not overlap on any host.

## Result — dispatch-bound graph, no host fallbacks

The step above still takes 3 host fallbacks, so part of its clock is CPU round
trips. A forward-only chain with a `sum` tail instead of `mean` is fallback-free
(`%{}`), so it isolates dispatch cost — best-of-5 blocks of 200 iterations:

| `NXV_BATCH_MAX` | best (ms) | all (ms) |
|---|---:|---|
| 0 — control | 2.4697 | 3.7714, 2.9817, 2.8864, 2.9681, 2.4697 |
| 64 | 1.8017 | 2.3613, 2.2944, 2.3135, 2.0657, 1.8017 |

1.37×, identical output (`0.24374596774578094`). Smaller than the training step
because the forward pass is only ~7 dispatches — batching saves 6 submits.
The gradient graph has many more, which is where the 2× comes from. **The win
scales with dispatch count**, which is the shape you would predict if the
mechanism is real.

## Correctness

`AutoCommandBufferBuilder` tracks resource usage while recording and inserts
the pipeline barriers between commands in `build()` (vulkano-0.34.2
`command_buffer/auto/builder.rs:272`), so read-after-write between two batched
dispatches is synchronised without hand-rolled barriers.

The builder cannot be parked in a static between NIF calls —
`StandardCommandBufferAllocator` deliberately does not implement `Send` for its
builder (`command_buffer/allocator.rs:568`), because a command buffer may not
migrate threads mid-recording and consecutive NIFs land on whichever dirty
scheduler is free. So the pending queue holds *closures* (`Send`, since
`BufferContents: Send + Sync + 'static`) replayed into a builder created on the
flushing thread.

A missed barrier shows up as nondeterministic wrong numbers rather than a
crash, so the suite was run repeatedly rather than once:

- super-io: full suite × 3 fixed seeds, default cap — green
- super-io: full suite at `NXV_BATCH_MAX` ∈ {0, 1, 4, 256} — green
- super-io: full suite × 10 consecutive runs at `NXV_BATCH_MAX=4` (a cap small
  enough to force a flush mid-graph on nearly every op) — green
- **mac.247 and 248: full suite at `NXV_BATCH_MAX` ∈ {0, 4, 256} — green**
  (851 doctests, 415 tests, 0 failures on each host at each cap)
- fallback census unchanged (the suite pins known fallbacks at exact counts)

That covers the plan's requirement that the gradient suite be green on all
three hosts, not just super-io.

One flake surfaced during this work and was **not** caused by batching:
`node_test.exs` did `Process.whereis` then `GenServer.stop` on a globally named
node that every test `start_link`s from the test process, so the node is already
dying from its link when `on_exit` runs in a different process. Check-then-act
on a dying pid; it exits `:noproc`. The failing test (`status/1`) never
dispatches anything. Fixed by tolerating the race.

## Not done

- **The 14.140 ms figure is not directly comparable.** That row came from the
  real Axon model in a scratch project with EXLA available; this harness is the
  same architecture rewritten in plain Nx and takes 3 host fallbacks where the
  fleet census records 1. The claim here is the A/B ratio on one graph, not that
  the recorded 14.140 ms number moved.
- **`matmul` and `transpose_2d` still build a `ShaderModule` + `ComputePipeline`
  per call** rather than going through `get_or_create_pipeline`. They join the
  batch, but per-call pipeline construction is per-dispatch cost too. Moving
  them onto the cache is a separate change so it can be measured separately.

## Two traps this benchmark walked into, recorded because they are cheap to repeat

1. **No default backend.** The first version never called
   `Nx.global_default_backend/1`, so every tensor `defn` materialises
   internally — grad constants, the scalar in `Nx.max(h, 0.0)` — landed on
   `Nx.BinaryBackend` and dragged the graph to the CPU. It measured 6.8 s per
   step, which is precisely the `BinaryBackend` row of the race table.
2. **Fallback contagion is invisible.** The second version built its inputs with
   `Nx.sin`, which is not one of the supported unary op codes and host-falls-back.
   Its result lands on `BinaryBackend`, and then *everything downstream computes
   there without being recorded*, because the counter only sees ops that reach
   this backend. A 32×784×128 matmul underneath it took **1039 ms on the CPU
   while `Nx.Vulkan.Fallback.count` reported `%{}`**. The same matmul on the GPU
   is 1.41 ms.

   This is the skill's "the count is a lower bound" warning with a price tag.
   The benchmark now asserts `%VulkanoBackend{}` residency on every input and on
   every gradient before it will report a timing, alongside the existing
   not-a-NaN guard.
