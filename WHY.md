# Why Nx.Vulkan exists

The short version: **a Bayesian posterior should run on the GPU you already
own — not the GPU a vendor decided you should have bought.** Everything below
is the long version.

For the reference tables (op coverage, position vs EXLA/EMLX, autograd), see
the [README](README.md). This document is the *why*.

## Two backends, and the hardware neither serves

Nx has two GPU backends, and between them they draw a hard border:

- **EXLA** (Google XLA) — the canonical, production path. It needs **CUDA** (or
  a TPU). NVIDIA on Linux, and nothing else.
- **EMLX** (Apple MLX) — Apple Silicon, through Metal. f32 only.

Stand anywhere outside that border and Nx has no GPU for you:

- **FreeBSD with an NVIDIA card** — no CUDA toolchain, no XLA. EXLA can't build.
- **A 2013 Kepler** (GT 650M / 750M) — too old for the CUDA versions modern XLA
  targets.
- **AMD or Intel GPUs** — not CUDA, not Metal.
- **Windows without the CUDA stack** — partial at best.

In every one of those cases the machine has a perfectly capable GPU sitting
idle, and the framework's answer is "run on the CPU." That answer is what this
project refuses to accept.

## Why Vulkan

Vulkan is the one compute API that spans the whole border. It is a Khronos
open standard, and it runs on:

- NVIDIA, AMD, and Intel GPUs,
- Linux, **FreeBSD**, and Windows drivers,
- Apple Silicon and Intel Macs, through **MoltenVK** (Vulkan → Metal).

One backend, written once against Vulkan compute, reaches all of them. There is
no CUDA dependency to install, no vendor toolchain to match, no platform matrix
to satisfy. If the machine has a Vulkan driver — and essentially every machine
made in the last decade does — it can run tensor compute.

The substrate is [Spirit](https://github.com/borodark/spirit)'s Vulkan shaders
wrapped as an Elixir-side `Nx.Backend` through a Rust NIF (`vulkano`). The
write-up of the first working proof is [*The GPU That Doesn't Need
CUDA*](http://www.dataalienist.com/blog-vulkan-on-freebsd.html).

## The f64 conviction

There is an unfashionable decision at the core of this backend:
**VulkanoBackend computes in f64 by default** — elementwise, reductions,
matmul, and the fused leapfrog chain synth all run in double precision.

This is unfashionable because consumer GPUs charge a steep tax for f64 (a
Kepler runs f64 at roughly 1/24 of its f32 rate), and the entire ML industry
has spent a decade racing *toward* lower precision. For deep learning, that's
right.

For a **Bayesian posterior**, it is wrong. A posterior computed in f32 drifts
in the tails — and the tails are exactly where the interesting inference lives:
tail probabilities, rare-event mass, the behaviour of a mixture far from its
modes. There was, three tags back, a commit literally named *"f32 is bad for
business."* It was right, and the backend was built around it. We would rather
be slow and correct than fast and quietly wrong in the tail. Where a GPU can't
do f64 at all (Apple Metal), we simply don't pretend it can.

## Autograd came for free

Building a new backend sounds like it should mean writing forward *and*
backward passes for every operation. It didn't, because of how Nx is layered:
`Nx.Defn.grad` is a **graph transformation** on the `Nx.Defn.Expr` AST. It
rewrites each forward op into backward ops expressed in terms of *more forward
ops*. The backend never sees a "gradient op" — it just keeps executing forward
primitives.

So **forward-op coverage *is* gradient coverage.** We implemented 24 native
ops (with a host fallback for the long tail) and got automatic differentiation
for anything expressible in them, with no backward callbacks written. For a
NUTS sampler — which is nothing but gradients of a log-density, over and over —
that is the difference between a weekend and a year.

## Reach, not peak FLOPS

Let's be honest about what this backend is and isn't.

It is **not** a bid to beat EXLA. On an RTX 3060 Ti with CUDA available, XLA is
a mature, deeply optimised compiler and will win most head-to-head throughput
races. If you have CUDA and only ever CUDA, use EXLA.

What VulkanoBackend offers is **reach** — inference on hardware that otherwise
has none. The thesis of the project, stated plainly: a 2013 Mac Pro running
FreeBSD, with a thirteen-year-old GT 750M, samples a real hierarchical
posterior — and returns the same numbers a modern Linux workstation does. That
machine "shouldn't" be able to do Bayesian inference on its GPU. It does. The
verification is [posteriordb, 33 models, matched against Stan reference
draws](http://www.dataalienist.com/blog-two-backends-one-posterior.html); the
FreeBSD proof is [*A Posterior on Any
GPU*](http://www.dataalienist.com/blog-a-posterior-on-any-gpu.html). Where the
model doesn't fit the fast fused path, it takes the slower per-op path and
still comes back correct — see [`LIMITATIONS.md`](LIMITATIONS.md).

Reach is worth more than peak FLOPS when the alternative is *nothing at all*.

## From one GPU to a fleet

The any-GPU thesis has a natural conclusion. If a posterior can run on *any*
Vulkan device, then it can run on *several at once* — a coordinator hands each
sampling job to whichever GPU in a cluster is free, and a fleet of otherwise-idle
machines becomes one inference engine. Heterogeneous by design: an Ampere card
and two 2013 Keplers, each correctness-verified, sharing the work. Two Keplers
already cut a four-posterior batch from 184s to 96s — near-linear. The oldest
hardware in the building, put back to work.

That only becomes possible *because* the backend runs everywhere. Portability
wasn't a nice-to-have; it was the precondition for everything after it.

## The one-liner

CUDA made GPU compute fast and made it a moat. Vulkan is the same silicon
without the moat. Nx.Vulkan exists so that a probabilistic program in Elixir
can compute its posterior on the GPU you have — Ampere or Kepler, Linux or
FreeBSD or a Mac — not the one a vendor decided you should have bought.
