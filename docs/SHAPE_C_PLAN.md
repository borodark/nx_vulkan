# Plan: Closing the Shape C Gap

Shape C is the host-fallback class of VulkanoBackend ops:
`pad`, `put_slice`, `indexed_put`, `indexed_add`, `broadcast`,
`concatenate`, `gather`, `take`. Across the bench, all show
speedups < 1 — vulkano consistently loses to BinaryBackend, by
factors ranging from 0.02× (concatenate) to 0.91× (gather large).

This is the next optimisation frontier. The plan has three tiers,
ordered by effort and reward.

## Why they lose

The current impl pattern for every Shape C op is:

```elixir
def op(out, tensor, ...) do
  t_bin = Nx.backend_transfer(tensor, Nx.BinaryBackend)  # 1. download from GPU
  result = Nx.op(t_bin, ...)                             # 2. compute on host
  from_binary(out, Nx.to_binary(result), [])             # 3. upload back to GPU
end
```

The "work" runs on BinaryBackend either way — vulkano contributes
nothing to the compute. What it *adds* is two round trips (download
+ upload). For ops whose actual work is microseconds, the round
trip dominates.

Worst case is `concatenate`. BinaryBackend's `concatenate/3` is
essentially `<<a::binary, b::binary>>` — a binary append at the
BEAM level, free. Vulkano's host-fallback downloads two GPU
tensors, concats them on the host, uploads the result. At 65k
elements, the round trip is 1384 µs vs BinaryBackend's 30 µs —
a 46× slowdown for an op the GPU never touches.

## Tier 1 — Skip the upload-back (small, contained, ~1 day)

Easiest win: change the host-fallback callbacks to return a
**BinaryBackend** tensor instead of uploading the result back to
vulkano. The result is already on the host — there is no reason
to re-upload unless a subsequent op needs it on GPU.

```elixir
def op(out, tensor, ...) do
  t_bin = Nx.backend_transfer(tensor, Nx.BinaryBackend)
  result = Nx.op(t_bin, ...)
  # result.data is %Nx.BinaryBackend{state: bin} — leave it.
  result
end
```

### Effect

- `concatenate`, `broadcast`, `pad`, `gather`, `take` on large
  tensors: cuts wall time roughly in half (just the download
  round trip remains, the upload-back is gone).
- `indexed_put`/`indexed_add` on small sampler shapes: similar,
  ~2× faster.
- If the consumer needs the result on GPU again, Nx will
  auto-transfer on first GPU op — same cost as today, just
  deferred and amortised across whatever happens next.

### Risks

- Nx allows mixed-backend tensors flowing through a pipeline, so
  the contract holds. But code that *assumes* a particular tensor
  is on VulkanoBackend (via `match?(%VulkanoBackend{}, t.data)`)
  could break. None known in the current codebase, but worth a
  `mix test` sweep.
- Some Nx ops dispatch based on the first operand's backend; a
  BinaryBackend result followed by a Vulkano operand may route
  through BinaryBackend's impl. Usually fine (BinaryBackend is
  the universal fallback), occasionally a perf regression for
  ops that *are* GPU-fast.

### Quick measurement gate before shipping

Re-run the bench with the patched callbacks. Expected:
- concatenate: 0.02× → ~0.3-0.5× (still loses, but 10-20× better)
- pad/broadcast: 0.5× → close to 1.0×
- gather/take: largely unchanged (download dominates)

If the gate doesn't show 5×+ improvement on concatenate, revert
and reconsider.

### What Tier 1 actually measured

Run on super-io (Linux + RTX 3060 Ti) and mac-248 (FreeBSD + GT
750M). Two findings, one expected, one not:

**Op-only bench (`vulkano_ops_bench.exs`)**: no improvement.
concatenate, pad, broadcast, etc. all show the same speedups as
pre-Tier-1. Reason: the op-only bench discards every iteration's
result. Tier 1 trades NIF-resource cleanup (Rust Drop, fast) for
BEAM GC of large host binaries (slower). On a tight loop those
costs cancel.

**Consumer bench (`vulkano_consumer_bench.exs`)**: clear win.
Measures `op + Nx.to_flat_list(result)` — the realistic flow
where the host actually reads the result (as the eXMC trial's
`signal_params` does on stored trace tensors). Headline D/E
speedups (Tier 1 active vs pre-Tier-1 simulated):

| op | median D/E |
|----|------------|
| broadcast | 1.42-2.46× |
| put_slice | 1.13-1.57× |
| indexed_put | 1.38-1.65× |
| concatenate | 0.99-1.30× |
| take | 1.02-1.35× |
| pad | 1.07-1.22× |
| gather | 0.85-1.16× (sometimes neutral) |

Median ~1.25-1.3× saved wall time when the consumer reads the
result. The Tier 1 win is real, just measured by the wrong bench
initially. Filed as a lesson: bench what the *consumer* does, not
just what the op does.

## Tier 2 — Native SPV for the bandwidth-bound four (~1 week)

Once Tier 1 lands, the remaining bottleneck for these four ops is
the **download** itself. Replace it with GPU-native shaders that
operate directly on the input buffers:

| op | shader | complexity |
|----|--------|------------|
| `broadcast` | Fill: write a single value to all positions. | trivial |
| `pad` | Two passes: copy interior, fill edges. | low |
| `concatenate` | Two `vkCmdCopyBuffer` calls + a destination buffer. | low (no shader at all — pure command-buffer plumbing) |
| `put_slice` | `vkCmdCopyBuffer` with offsets. | low |

These are bandwidth-bound, not compute-bound. They should land at
~1-3× over BinaryBackend on the 3060 Ti for ≥16k elements, and
roughly parity on mac-248 (where the dispatch tax still eats
small ops).

### Why not start here?

Native shaders cost a week of work per op, including the bench
sweep to confirm crossovers. Tier 1 buys 70% of the value for
1 day of work. Do Tier 1 first, measure, decide whether Tier 2
on the remaining hot ops is worth the time.

## Tier 3 — Index-handling shaders for the gather family

`gather`, `take`, `indexed_put`, `indexed_add` need real
GPU-native impls because their work is index-driven scatter or
gather. These are the highest-effort, lowest-yield ops in the
group — by the time the inputs are large enough for the GPU to
win on bandwidth, the index list itself is large and the upload
of the index buffer competes with the work.

Defer indefinitely. If a real workload surfaces (e.g., a sampler
or model with thousand-row indexed updates), revisit. The eXMC
NUTS leapfrog uses `indexed_put` only at tiny scales
({200, 8}-ish) where even an optimal GPU impl would not beat
BinaryBackend.

## Routing-policy follow-up (orthogonal but related)

A separate fix that *avoids* the Shape C problem entirely for
the NUTS hot path: keep small-shape NUTS scratch tensors on
BinaryBackend even when the global default is VulkanoBackend.

In `Exmc.JIT.backend/0`, add a "size hint" branch:

```elixir
def backend_for(shape, type) do
  n = shape |> Tuple.to_list() |> Enum.reduce(1, &*/2)
  if n < 16_384 do
    Nx.BinaryBackend
  else
    backend()  # the configured default
  end
end
```

Then in `RegimeModel.build/2` and similar state-creating
callsites, use `Nx.tensor(..., backend: Exmc.JIT.backend_for(shape, type))`
instead of relying on the global default. Small priors,
intermediate scratch, and the trace itself stay on the host.

This belongs in `exmc`, not `nx_vulkan` — file it there.

## Suggested order

1. Tier 1 (one afternoon). Ship, re-run the bench, commit the
   updated CSVs. Confirm concatenate gains 10×+.
2. Routing policy in eXMC. Half-day. Independent of nx_vulkan.
3. Tier 2 on `concatenate` first (purely command-buffer
   plumbing, no GLSL needed). Half-day, then bench.
4. Tier 2 on `broadcast`, `pad`, `put_slice`. One day each.
5. Stop. Tier 3 is not currently justified.

Total runway: roughly a week if everything works first try.
The bench will tell us if it didn't.
