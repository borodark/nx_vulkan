# poison_control.exs — does the uninitialised allocator ever leak residue?
#
#   mix run scripts/poison_control.exs           # fast path
#   mix run scripts/poison_control.exs --sweep   # + the size-class evidence table
#
# `alloc_buffer` hands back UNINITIALISED memory (`Buffer::new_slice`, 084b937).
# That is safe only if every shader writes every byte it is given. Four shaders
# do not — `allany_{f32,f64,s32,u8}` atomicOr one thread per slot — and those use
# `buf_alloc_zeroed/1`. This script is the standing check on that claim.
#
# WHY THIS FILE EXISTS AT ALL, AND WHY IT CAN FAIL WITHOUT FINDING A BUG.
#
# A clean run proves nothing on its own. Freshly mapped pages are OS-zeroed, so
# a shader that wrongly depends on zeroed memory passes cold and only fails once
# blocks recycle. The result is only worth reading if the memory was DIRTY when
# it was handed out. So the control must first prove it can poison — and this is
# where the project has now been caught twice:
#
#   * The Keplers' first scheme built blocks with `buf_upload`. Those land in a
#     different suballocator pool: 0/24 dirty.
#   * The Jetson's scheme used the right calls but at 6 x 32 MiB: 0/40 dirty.
#     32 MiB is exactly the cliff where vulkano stops suballocating and issues a
#     dedicated `vkAllocateMemory`. That memory goes back to the DRIVER on free
#     and returns as fresh zeroed pages, never re-entering the small-buffer pool.
#     It is the one size class that cannot poison — and it sat in NEXT.md for a
#     week as corroboration (corrected in c8e4332).
#
# Both schemes reported "clean". Both were vacuous. So:
#
#   ** IF THE POOL CANNOT BE DIRTIED, THIS SCRIPT EXITS NON-ZERO. **
#
# A control that cannot detect the bug must not report its absence. That is the
# single rule this file exists to enforce; everything below is mechanism.
#
# Poisoning requires BOTH:
#   1. `buf_alloc` + `buf_upload_into` (not `buf_upload` — wrong pool), and
#   2. a block size strictly BELOW the 32 MiB dedicated-allocation cliff.

alias Nx.Vulkan.VulkanoBackend, as: VB
alias Nx.Vulkan.NativeV

sweep? = "--sweep" in System.argv()

# Known-good below the cliff. If this ever stops dirtying the pool the script
# sweeps for a replacement rather than silently reporting a vacuous pass.
default_scheme = {"16 x 8 MiB", 16, 8 * 1024 * 1024}

candidates = [
  {"6 x 32 MiB  (AT the cliff — the known-vacuous one)", 6, 32 * 1024 * 1024},
  default_scheme,
  {"64 x 1 MiB", 64, 1024 * 1024},
  {"512 x 64 KiB", 512, 64 * 1024},
  {"2048 x 4 KiB", 2048, 4 * 1024}
]

poison = fn count, size ->
  refs =
    for _ <- 1..count do
      {:ok, r} = NativeV.buf_alloc(size)
      :ok = NativeV.buf_upload_into(r, :binary.copy(<<0xFF>>, size))
      r
    end

  :ok = NativeV.flush()
  _ = length(refs)
  :ok
end

# How many of `probes` fresh allocations come back holding a non-zero byte?
dirty_frac = fn probes, size ->
  Enum.reduce(1..probes, 0, fn _, acc ->
    {:ok, r} = NativeV.buf_alloc(size)
    {:ok, bin} = NativeV.buf_download(r)
    if bin |> :binary.bin_to_list() |> Enum.any?(&(&1 != 0)), do: acc + 1, else: acc
  end)
end

# Score a scheme out of 40: 20 probes at 64 B and 20 at 4 KiB, re-poisoning
# between the two so the second probe set is not reading the first's leavings.
score = fn count, size ->
  poison.(count, size)
  :erlang.garbage_collect()
  poison.(count, size)
  :erlang.garbage_collect()
  d64 = dirty_frac.(20, 64)
  :erlang.garbage_collect()
  poison.(count, size)
  :erlang.garbage_collect()
  d4k = dirty_frac.(20, 4096)
  d64 + d4k
end

IO.puts("\n===== CONTROL: can this box's allocator be poisoned at all? =====\n")

{label, count, size, dirty} =
  if sweep? do
    IO.puts("  scheme                                              dirty/40")

    results =
      for {l, c, s} <- candidates do
        d = score.(c, s)
        IO.puts("  #{String.pad_trailing(l, 50)} #{String.pad_leading("#{d}/40", 8)}")
        {l, c, s, d}
      end

    IO.puts("")
    Enum.max_by(results, fn {_, _, _, d} -> d end)
  else
    {l, c, s} = default_scheme
    d = score.(c, s)
    IO.puts("  #{String.pad_trailing(l, 50)} #{String.pad_leading("#{d}/40", 8)}")

    if d > 0 do
      {l, c, s, d}
    else
      IO.puts("\n  default scheme failed to dirty the pool — sweeping for one that does\n")
      IO.puts("  scheme                                              dirty/40")

      results =
        for {l2, c2, s2} <- candidates do
          d2 = score.(c2, s2)
          IO.puts("  #{String.pad_trailing(l2, 50)} #{String.pad_leading("#{d2}/40", 8)}")
          {l2, c2, s2, d2}
        end

      IO.puts("")
      Enum.max_by(results, fn {_, _, _, dd} -> dd end)
    end
  end

# Effectiveness is a SAMPLE, not a property of the box. The Jetson scored the
# same 16 x 8 MiB scheme 40/40 in one run and 20/40 in the next, and its sweep
# then picked a different winner than its fast path did. So a single 0 is not
# proof the pool cannot be dirtied — and a false vacuous-FAIL is its own wrong
# answer, condemning a box that poisons perfectly well. Re-sample every
# candidate before declaring anything.
{label, count, size, dirty} =
  if dirty > 0 do
    {label, count, size, dirty}
  else
    IO.puts("\n  0/40 on the first sample — re-sampling before declaring vacuous\n")

    Enum.reduce_while(1..3, {label, count, size, 0}, fn round, _acc ->
      best =
        candidates
        |> Enum.map(fn {l, c, sz} ->
          d = score.(c, sz)
          IO.puts("  round #{round}: #{String.pad_trailing(l, 50)} #{d}/40")
          {l, c, sz, d}
        end)
        |> Enum.max_by(fn {_, _, _, d} -> d end)

      if elem(best, 3) > 0, do: {:halt, best}, else: {:cont, best}
    end)
  end

if dirty == 0 do
  IO.puts("""

  !! NO SCHEME DIRTIED THE POOL ON THIS BOX.

     Every freed buffer came back zeroed across FOUR independent samples of
     every candidate scheme, so a shader that depends on zeroed memory would
     pass here regardless. This run can neither confirm nor deny
     the uninitialised-allocator claim — it is VACUOUS, and a vacuous control
     must not be recorded as a clean result.

     Find a scheme that dirties before trusting any allocator conclusion here.

  POISON CONTROL: FAIL (vacuous — pool could not be dirtied)
  """)

  System.halt(1)
end

IO.puts("  using: #{label} — #{dirty}/40 dirty (EFFECTIVE)\n")
churn = fn -> poison.(count, size) end

# ---------------------------------------------------------------------------

IO.puts("===== PADDING: which writers hand back a buffer larger than logical? =====\n")

# The scheme score above proves the pool is dirty at 64 B and 4 KiB. The buffers
# inspected BELOW are 4 B and 8 B, and nothing so far shows an allocation of THAT
# size comes back dirty. Reading "padding is all zero" off probes whose sizes were
# never shown to be poisonable is the same generalisation this file exists to
# refuse, so measure it at the sizes actually used rather than leaning on 64 B.
_ = churn.()
:erlang.garbage_collect()
pad_dirty = dirty_frac.(10, 4) + dirty_frac.(10, 8)

IO.puts("  poisonability at the inspected sizes (4 B, 8 B): #{pad_dirty}/20")

if pad_dirty == 0 do
  IO.puts("  -> allocations at these sizes never came back dirty on this box.")
  IO.puts("     A zero-padding result below is therefore UNPROVEN, not clean.\n")
else
  IO.puts("")
end

probe = fn label, t ->
  case t.data do
    %VB{ref: ref} ->
      {:ok, raw} = NativeV.buf_download(ref)
      {_, bits} = t.type
      logical = Nx.size(t) * div(bits, 8)
      pad = :binary.bin_to_list(binary_part(raw, logical, byte_size(raw) - logical))
      dirty? = Enum.any?(pad, &(&1 != 0))

      IO.puts(
        "  #{String.pad_trailing(label, 42)} buf=#{String.pad_leading("#{byte_size(raw)}", 4)}B " <>
          "logical=#{String.pad_leading("#{logical}", 4)}B" <>
          if(dirty?, do: "  <-- DIRTY PADDING", else: "")
      )

      {byte_size(raw) > logical, dirty?}

    other ->
      IO.puts("  #{String.pad_trailing(label, 42)} not resident (#{inspect(other.__struct__)})")
      {false, false}
  end
end

pad_probes =
  for n <- [1, 2, 3, 5, 7] do
    _ = churn.()
    :erlang.garbage_collect()
    t = Nx.tensor(for(i <- 1..n, do: for(j <- 1..3, do: rem(i * j, 2))), backend: VB)
    a = probe.("all(#{n}x3, axes:[1])   [allany writer]", Nx.all(t, axes: [1]))
    b = probe.("any(#{n}x3, axes:[1])   [allany writer]", Nx.any(t, axes: [1]))

    c =
      probe.(
        "greater(#{n})           [compare writer]",
        Nx.greater(
          Nx.iota({n}, backend: VB),
          Nx.tensor(for(i <- 1..n, do: rem(i * 7, 5)), backend: VB)
        )
      )

    [a, b, c]
  end
  |> List.flatten()

padded = Enum.count(pad_probes, fn {p, _} -> p end)
dirty_pad = Enum.count(pad_probes, fn {_, d} -> d end)

IO.puts(
  "\n  padded buffers: #{padded}/#{length(pad_probes)}   with non-zero padding: #{dirty_pad}\n"
)

# ---------------------------------------------------------------------------

IO.puts("===== CONCAT UNDER POISONING =====")
IO.puts("(operands from the PADDED writers are the case that was actually wrong;")
IO.puts(" Nx.tensor-built operands are exact-sized and cannot exercise it)\n")

padded_bad =
  for n <- [1, 2, 3, 5, 6, 7, 9, 13, 17, 31, 33], reduce: [] do
    acc ->
      _ = churn.()
      :erlang.garbage_collect()

      build = fn b ->
        t = Nx.tensor(for(i <- 1..n, do: for(j <- 1..3, do: rem(i * j, 2))), backend: b)
        u = Nx.tensor(for(i <- 1..n, do: for(j <- 1..3, do: rem(i + j, 2))), backend: b)
        Nx.concatenate([Nx.all(t, axes: [1]), Nx.any(t, axes: [1]), Nx.all(u, axes: [1])])
      end

      g = Nx.to_flat_list(build.(VB))
      r = Nx.to_flat_list(build.(Nx.BinaryBackend))
      if g == r, do: acc, else: [{n, g, r} | acc]
  end

IO.puts("  padded-writer operands:  #{length(padded_bad)} mismatches")
Enum.each(padded_bad, fn m -> IO.puts("    #{inspect(m)}") end)

typed_bad =
  for type <- [
        {:u, 8},
        {:s, 8},
        {:u, 16},
        {:s, 16},
        {:f, 32},
        {:f, 64},
        {:s, 32},
        {:u, 32},
        {:s, 64}
      ],
      n <- [1, 2, 3, 5, 7, 9, 13, 17, 31, 33],
      reduce: [] do
    acc ->
      _ = churn.()
      :erlang.garbage_collect()

      operands = fn b ->
        [
          Nx.tensor(Enum.map(1..n, &rem(&1 * 7, 100)), type: type, backend: b),
          Nx.tensor(Enum.map(1..n, &rem(&1 * 13, 100)), type: type, backend: b),
          Nx.tensor(Enum.map(1..n, &rem(&1 * 29, 100)), type: type, backend: b)
        ]
      end

      vb_operands = operands.(VB)
      got = Nx.concatenate(vb_operands)
      g = Nx.to_flat_list(got)
      r = Nx.to_flat_list(Nx.concatenate(operands.(Nx.BinaryBackend)))

      # Word-copyable types must ALSO stay resident: a silent fallback would
      # return the right answer off the GPU and hide the path under test.
      #
      # But demand that only when the OPERANDS are themselves resident, and
      # measure it rather than assuming it. `concatenate` declines when
      # `all_vulkano?/1` is false, which is correct behaviour, not a defect —
      # and on a box missing an integer kernel (`Nx.add` on {:s,64} has none)
      # an operand can arrive on the host without anyone intending it. Both
      # Keplers and the Ampere box each independently wrote this assertion the
      # assuming way first and each got a false failure out of it.
      word? = type in [{:f, 32}, {:f, 64}, {:s, 32}, {:u, 32}, {:s, 64}]
      operands_resident? = Enum.all?(vb_operands, &match?(%VB{}, &1.data))

      cond do
        g != r ->
          [{:value, type, n, g, r} | acc]

        word? and operands_resident? and not match?(%VB{}, got.data) ->
          [{:residency, type, n} | acc]

        true ->
          acc
      end
  end

IO.puts("  typed concat:            #{length(typed_bad)} problems")
Enum.each(typed_bad, fn m -> IO.puts("    #{inspect(m)}") end)

# ---------------------------------------------------------------------------

IO.puts("\n===== SUMMARY =====")
IO.puts("  scheme:                #{label}")

IO.puts(
  "  effectiveness:         #{dirty}/40 dirty probes" <>
    if(dirty < 40, do: " (a sample — this varies run to run)", else: "")
)

IO.puts("  padded buffers seen:   #{padded} (non-zero padding: #{dirty_pad})")
IO.puts("  padding-size probes:   #{pad_dirty}/20 dirty at 4 B / 8 B")
IO.puts("  padded-writer concat:  #{length(padded_bad)} mismatches")
IO.puts("  typed concat:          #{length(typed_bad)} problems")

ok? = padded_bad == [] and typed_bad == [] and dirty_pad == 0

cond do
  not ok? ->
    IO.puts("\nPOISON CONTROL: FAIL")
    System.halt(1)

  # Everything came back clean, but the padding leg could not have come back any
  # other way. Say so rather than banking it: the concat results still stand on
  # the 40/40 scheme, the padding result does not stand on anything.
  pad_dirty == 0 ->
    IO.puts(
      "\nPOISON CONTROL: PASS with the padding leg UNPROVEN " <>
        "(scheme=#{label}, #{dirty}/40 dirty, but 0/20 at 4 B / 8 B)"
    )

  true ->
    IO.puts(
      "\nPOISON CONTROL: PASS (scheme=#{label}, #{dirty}/40 dirty, " <>
        "#{pad_dirty}/20 at padding sizes)"
    )
end
