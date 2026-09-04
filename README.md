# Nx.Vulkan

A GPU tensor backend for [Nx](https://github.com/elixir-nx/nx), built on Vulkan.
It runs wherever a Vulkan driver runs — Linux, FreeBSD, Windows, macOS via
MoltenVK, ARM boards — which includes hardware CUDA has dropped and platforms
Metal never reached.

```elixir
# mix.exs
{:nx, "~> 0.13"},
{:nx_vulkan, "~> 0.3"}
```

```elixir
Nx.default_backend(Nx.Vulkan.VulkanoBackend)

Nx.sigmoid(Nx.tensor([1.0, 2.0, 3.0, 4.0]))
#=> #Nx.Tensor<f32[4] [0.7310586, 0.8807971, 0.95257413, 0.98201376]>
```

Native **f32 and f64** compute, whole-graph fusion, and working `Nx.Defn.grad` —
for which no backward pass was ever written. Autograd is a graph transformation
that runs above the backend, so forward op coverage *is* gradient coverage.

A LeNet training step took **20 929 ms** on 0.2.0 and takes **84 ms** now — same
box, same graph, bit-identical loss. The difference was eight GPU fast paths
gated on shapes only a *forward* pass produces.

## The fleet

Every correctness claim here is green on all four of these. Every performance
heuristic is raced on all four before it ships.

| host | GPU | year | OS | arch |
|---|---|---|---|---|
| super-io | RTX 3060 Ti (Ampere) | 2021 | Linux | x86_64 |
| mac-248 | GT 750M (Kepler) | 2013 | FreeBSD | x86_64 |
| mac-247 | GT 650M (Kepler) | 2012 | FreeBSD | x86_64 |
| jake-desktop | Tegra X1, Jetson Nano | 2015 | Ubuntu | aarch64 |

**833 doctests, 903 tests, 0 failures** on every one of them. Two CPU
architectures, three operating systems, four GPU generations spanning
2012–2021, one set of
SPIR-V binaries — and, where it is asserted, the same posterior bit for bit.

Two of those four boxes were abandoned by their vendor: CUDA 13 retired Kepler
long ago, and the Jetson's CUDA support stopped at 10.2. Their Vulkan drivers
did not stop. The Jetson is also the fleet's only **unified-memory** board,
which makes it a natural control arm — a code path that exists because there is
a PCIe bus is a no-op there, and more than one optimisation has been disproved
by winning *least* on it. See [`docs/FLEET.md`](docs/FLEET.md).

## Where to go next

**Start here**
- [`WHY.md`](WHY.md) — why this exists: the f64 conviction, reach over peak FLOPS, one GPU to a fleet
- [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) — the op surface, the fusion compiler, how autograd came free
- [`docs/BUILDING.md`](docs/BUILDING.md) — install, prerequisites, the two examples worth running first

**Before you trust a number**
- [`docs/STANDING.md`](docs/STANDING.md) — an honest position vs EXLA and EMLX, including where fusion *loses*
- [`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) — every measured figure with its method and its caveats
- [`docs/FLEET.md`](docs/FLEET.md) — the hardware, and the optimisations a single box would have got wrong

**Method**
- [`docs/STRICT_MODE.md`](docs/STRICT_MODE.md) — a host fallback returns a bit-identical result, so no assertion on values can detect one. This is how they are detected anyway.
- [`docs/PROPERTY_TESTING.md`](docs/PROPERTY_TESTING.md) — what each property can actually detect, and the tests that could not fail
- [`docs/BACKWARD_PASS_AUDIT.md`](docs/BACKWARD_PASS_AUDIT.md) — the 20 929 ms defect class, and how to avoid writing it again
- [`.claude/skills/vulkan-nx-compute`](https://github.com/borodark/nx_vulkan/tree/main/.claude/skills/vulkan-nx-compute) — the shader → NIF → Nx playbook

**Ahead**
- [`ROADMAP.md`](ROADMAP.md) · [`CHANGELOG.md`](CHANGELOG.md)

## Writing

- [*The Test That Could Not Fail*](http://www.dataalienist.com/blog-test-that-could-not-fail.html) — a NaN guard incapable of detecting a NaN, and the seven guards nobody had watched fire
- [*Compute It Twice: When CSE Lost the Race*](http://www.dataalienist.com/blog-compute-it-twice.html) — the fusion compiler, and why the textbook optimisation lost
- [*The Alibi of a Correct Answer*](http://www.dataalienist.com/blog-alibi-of-a-correct-answer.html) — the backward pass that ran on the CPU for months behind a green suite
- [*The GPU That Doesn't Need CUDA*](http://www.dataalienist.com/blog-vulkan-on-freebsd.html) — the FreeBSD Vulkan story

## Sibling: zed

[`zed`](https://github.com/borodark/zed) is the declarative ZFS + Elixir deploy
tool that orchestrates BEAM nodes. `nx_vulkan` is consumed *inside* deployed
nodes, not as a zed dependency.

## License

Apache 2.0. Same as Nx.
