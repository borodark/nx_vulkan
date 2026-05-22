# Three Shapes of Vulkano Speedup

*What 309 benchmarks across two machines tell you about a Vulkan
backend — and why "GPU is faster" is usually the wrong question.*

---

## The setup

VulkanoBackend is the pure-Rust Nx backend that nx_vulkan added in
the Mission II work. It implements every Nx callback it can on the
GPU via Vulkan compute shaders, with a host-fallback path through
BinaryBackend for anything not yet ported. Once it landed, the obvious
question was: how much faster, exactly, and *where*?

The bench script `examples/vulkano_ops_bench.exs` is the answer.
Every callback the backend implements, swept across ~10 shapes,
compared against BinaryBackend at the same shape. Written to
`bench_results/<hostname>_<date>.csv` so cross-machine diffs are a
`csvkit join` away.

Run on two machines:

| Host | GPU | OS | PCIe |
|------|-----|----|------|
| `super-io` | NVIDIA RTX 3060 Ti (Ampere, 8GB, 4864 cores) | Linux 6.8 | gen4 |
| `free-macpro-nvidia` (mac-248) | NVIDIA GT 750M (Kepler, 2GB, 384 cores) | FreeBSD 15.0 | gen2 |

On paper, super-io is about 12× the compute of mac-248. The bench
result is more interesting than that.

## Three curves

Plot vulkano-speedup-vs-BinaryBackend against input size for any op
and you get one of three shapes:

```
Shape A: compute-bound        Shape B: dispatch-bound        Shape C: host-fallback
   speedup                       speedup                        speedup
      |                             |                             |
 1000 |     ____________       7x  |       /\                1.0 +-----------------
      |    /                       |      /  \                   |
  100 |  /                       1 +-/----+--+----              0.1
      |/                           |/     |                      |\______________
   1 -+---------------    BinBack -+------+--+---                +-----------------
      | size                       |  256k|512k                  |  size
```

**Shape A** (matmul, big elementwise binary/unary): scales nearly
linearly with the work. Speedup runs from 1.4× at small sizes to
~10,000× at `dot {256×256}` on super-io. The GPU was made for this.

**Shape B** (reductions): speedup is a hump. At small sizes the GPU
loses — dispatch overhead dwarfs work. The hump peaks around 256k
elements (7.14× for `reduce_max` on super-io), then *falls back
toward parity* at 1M as the BinaryBackend cache effects play out and
the GPU saturates memory bandwidth, not arithmetic.

**Shape C** (pad, put_slice, indexed_put, indexed_add, broadcast,
concatenate, gather, take): speedup is always under 1. These are
host-fallback callbacks. The "compute" runs on BinaryBackend either
way; vulkano just pays for the round trip — `backend_transfer →
BinaryBackend.<op> → from_binary`. The transfer is the entire cost,
and it grows with the tensor.

## The hardware twist

What makes the bench interesting isn't the headline matmul number.
It's that **mac-248 — a thirteen-year-old GT 750M on FreeBSD — beats
super-io on dispatch-bound ops up to roughly 16k elements**:

| op | shape | super-io (Linux, RTX 3060 Ti) | mac-248 (FreeBSD, GT 750M) |
|----|-------|------|---------|
| `add {1024}` | binary | 102 µs | **69 µs** |
| `multiply {1024}` | binary | 139 µs | **74 µs** |
| `iota {256}` | storage | 237 µs | 354 µs (close) |
| `dot {16×16}` | matmul | 966 µs | **410 µs** |

The 3060 Ti has ~12× the raw compute. The work fits in its L1 cache
ten times over. And yet for a 1024-element vector add it loses by a
factor of 1.5.

The cause is the per-dispatch tax: `vkCreateBuffer` + descriptor set
binding + command buffer record + queue submit + fence wait. On
Linux with the modern driver and a Wayland desktop alongside, that
sequence eats hundreds of microseconds. On FreeBSD with no display
server competing for the queue and a simpler driver path, it's
faster — by enough to outweigh the GPU compute disadvantage when the
compute is tiny. The same effect was noted on mac-247 last year in
the matmul characterisation. The bench reproduces it across the full
op set.

## What the dispatch tax buys you

The cost of *any* GPU op on the FreeBSD path is roughly
constant — a few tens of microseconds per dispatch, regardless of
how much work follows. Once the work crosses that threshold, you're
back on the compute curve where the 3060 Ti dominates:

| `add` size | super-io | mac-248 | ratio (mac-248 / super-io) |
|------------|----------|---------|----------------------------|
| {1024} | 102 µs | 69 µs | 0.68× |
| {16384} | 154 µs | 119 µs | 0.77× |
| {65536} | 260 µs | 164 µs | 0.63× |
| {262144} | 748 µs | 1152 µs | **1.54×** |
| {1048576} | 2246 µs | (timed-out at 0.97×) | 0.46× |

The crossover happens between 65k and 256k for binary ops. Below
it, you're dispatch-bound and mac-248 wins on driver overhead.
Above it, you're compute-bound and super-io wins on raw FLOPs.

## What this means for routing

The sampler-host class is the loud finding. *Every* op there is
~5-50× slower on vulkano than on BinaryBackend. Including
`indexed_put` and `put_slice` — the two ops the NUTS leapfrog
loop calls dozens of times per step.

In the production eXMC trial those tensors live on VulkanoBackend
because we made it the global default at boot. Every leapfrog step
pays the host-fallback round trip. The total impact is muted —
those ops are tiny relative to the synth-shader dispatch that
dominates wall time — but it's *worth knowing*. A future routing
policy could keep small-shape state tensors on BinaryBackend even
when the default backend is vulkano, and reserve the GPU storage
class for the buffers that actually feed shaders.

Concatenate is the bench's most embarrassing row, by the way.
0.02× at 65k. BinaryBackend's concatenate is essentially a
`<<a::binary, b::binary>>` — free. Vulkano's host-fallback path
downloads two GPU tensors, concats them on the host, uploads the
result. The GPU never touches the data. The only thing vulkano
contributes is the round trip.

## The takeaway

Three curves, two machines, 309 measurements. The numbers don't
say "use the GPU." They say *use the right backend at the right
size*, and pay attention to which side of the dispatch tax your
workload actually lives on.

The conventional reading of "use the GPU for everything" is the
sales pitch. The bench is the truth: there is a regime — small,
host-routed, latency-sensitive — where a 2013 Mac Pro on FreeBSD
will out-dispatch a 2021 RTX 3060 Ti on Linux. Filing that under
"things I now have data for."

---

*Raw data: `nx_vulkan/bench_results/super-io_2026-05-22.csv`,
`nx_vulkan/bench_results/free-macpro-nvidia_2026-05-22.csv`.
Script: `examples/vulkano_ops_bench.exs`. Run it yourself.*
