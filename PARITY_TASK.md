# PARITY_TASK — close the Nx.Backend gap in VulkanoBackend

**For:** the Claude instance running on **mac-247** (FreeBSD, GT 650M, Vulkan).
**Branch:** `parity-tier1` (you are on it). Commit here; push to `nas` so the
super-io box (249) can watch progress.
**Brief written by:** Claude on super-io (249), 2026-07-27, from `d7ab05a`.

---

## Objective

Bring `Nx.Vulkan.VulkanoBackend` to full `Nx.Backend` parity **except** an
intentional skip set, each op **verified against `BinaryBackend` on this host's
real Vulkan**. Goal: downstream libraries (Scholar, Axon) stop crashing on
missing callbacks. eXMC itself uses none of these, so the trader is not at risk.

## Read this first — the old analysis is STALE

`docs/NX_PARITY_RESEARCH.md` and `docs/nx_parity_gap.csv` are from **2026-05-25**
and are out of date — a lot has been implemented since. **Do not work from that
CSV.** Regenerate the real gap in Step 0.

The host-fallback machinery **already exists** in
`lib/nx_vulkan/vulkano_backend.ex` — reuse it, don't reinvent:
`host_result/2`, `ensure_on_backend/1`, `binary_op_host_fallback/4`, the
`for op <- [:all, :any]` reduction loop, and `block/4` (routes SVD/QR/LU through
`BinaryBackend`). These are your templates.

## Step 0 — Baseline (commit before changing any op)

1. `git checkout parity-tier1`
2. Regenerate the **real** missing set (path-independent, run in `iex -S mix`):
   ```elixir
   impl = Nx.Vulkan.VulkanoBackend.__info__(:functions) |> Enum.map(&elem(&1, 0)) |> MapSet.new()
   cbs  = Nx.Backend.behaviour_info(:callbacks)          |> Enum.map(&elem(&1, 0)) |> MapSet.new()
   MapSet.difference(cbs, impl) |> Enum.sort()   # <-- the actual worklist
   ```
   Write it to `docs/PARITY_STATUS.md` (fresh) with today's date.
3. Snapshot the suite: `mix compile 2>&1 | grep -i warning` (note the dead
   `all/3`, `any/3`, `to_batched/3` clauses) and `mix test`. Record pass/fail.
4. Commit: `parity: baseline — regenerated gap + test/warning snapshot`.

## The iteration loop — one callback (or one family) per commit

Work **easy → hard**: reduction → cumulative → shape → logic → sort → window →
linalg. For each callback still missing after Step 0:

1. Implement via the established host-fallback template:
   ```elixir
   @impl true
   def <cb>(out, tensor, opts) do
     bin = Nx.backend_transfer(ensure_on_backend(tensor), Nx.BinaryBackend)
     host_result(out, apply(Nx, :<cb>, [bin, opts]))
   end
   ```
   - **Multi-tensor ops** (`all_close`, `take_along_axis`, `solve`, …): transfer
     each tensor argument to `BinaryBackend` first.
   - **linalg** (`cholesky`, `determinant`, `eigh`, `lu`, `qr`, `svd`, `solve`,
     `triangular_solve`): route through `BinaryBackend` / the existing `block/4`
     path rather than writing GPU shaders.
2. Add a focused test in `test/` comparing VulkanoBackend vs a `BinaryBackend`
   reference for representative shapes, **in f64** (this backend is f64-first).
   Assert with `Nx.all_close/3`.
3. `mix test` — the new test plus the full suite must be green **on this host's
   Vulkan** (that's the whole point — this is the correctness reference).
4. Commit: `parity: <cb> host-fallback (verified vs BinaryBackend)`, then
   `git push nas parity-tier1`.

## Cleanup (do this early)

Remove the **dead duplicate** `all/3`, `any/3`, `to_batched/3` clauses that the
compiler flags as *"cannot match because a previous clause always matches,"* and
fix the `to_batched` unused-variable warning. Land at zero warnings for these.

## Skip set — do NOT implement

`fft`, `fft2`, `ifft`, `conv`, `phase`, `from_pointer`, `to_pointer`. These are
out of scope (spectral / CNN / FFI / complex). Document them in `LIMITATIONS.md`
as "falls back to EXLA / BinaryBackend."

## Definition of done

- The Step-0 regeneration shows **only the skip set** remaining.
- `mix test` **green on FreeBSD Vulkan** (this host).
- The parity ops compile **warning-free**.
- `LIMITATIONS.md` lists the intentional skips.

## Guardrails

- Stay on `parity-tier1`; commit per callback/family; push to `nas` periodically.
- Confine changes to `lib/nx_vulkan/vulkano_backend.ex`, `test/`, and the parity
  docs. **Do not touch** the native shader ops or the fused leapfrog chain path.
- Always run the real `mix test` on this host — never a mock. FreeBSD + Vulkan
  is the reference the whole project is validated against.
