defmodule Nx.Vulkan.OneOffsTest do
  @moduledoc """
  The last three refusing doctests, and the three unrelated gates behind them.

  `NEXT.md` §1.3 listed these as "three unexamined one-offs" and said to probe
  before scoping. Probing was the whole job: each turned out to be a gate that
  was narrower than the kernel behind it, and none needed a shader.

    * **`dot/7` at rank 5.** The `rank <= 4` cap exists because `transpose_nd`
      addresses at most four dims — but `dot_flatten/3` transposes ONLY when the
      permutation is not already the identity. An operand arriving in
      contraction order needs no rotation, so the cap was refusing a capability
      it does not use.
    * **`indexed_put/5` over a non-prefix axis subset.** `gather/4` already
      rotates for this; scatter did not. Scatter additionally needs the rotation
      BACK, because its result has the TARGET's shape.
    * **`slice/5` with a dynamic (tensor) start.** Four bytes, read back —
      against a host round trip for the entire source tensor.

  Values are checked against `Nx.BinaryBackend` and residency is asserted
  separately, since a fallback returns a bit-identical answer.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp check(build) do
    got = build.(VulkanoBackend)
    assert match?(%VulkanoBackend{}, got.data), "expected the result to stay on the GPU"
    ref = build.(Nx.BinaryBackend)
    assert Nx.type(got) == Nx.type(ref)
    assert Nx.shape(got) == Nx.shape(ref)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  describe "dot beyond rank 4, when no rotation is needed" do
    test "rank 5 against rank 1 — Nx's own doctest" do
      check(fn b ->
        Nx.dot(
          Nx.tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]], backend: b),
          Nx.tensor([2.0, 2.0], backend: b)
        )
      end)
    end

    test "rank 6 too — the cap was never about rank, only about transposing" do
      check(fn b ->
        Nx.dot(Nx.iota({1, 2, 1, 2, 2, 3}, type: {:f, 32}, backend: b),
               Nx.iota({3}, type: {:f, 32}, backend: b))
      end)
    end

    test "an integer rank-5 contraction" do
      check(fn b ->
        Nx.dot(Nx.iota({2, 1, 2, 2, 3}, backend: b), Nx.iota({3}, backend: b))
      end)
    end

    # TAGGED because the fallback IS the subject. `strict_test.sh` excludes only
    # this tag, so a test that deliberately provokes a refusal has to opt out or
    # it turns the strict run red.
    #
    # I wrote this file's first cut with the tag on the WRONG test — on the
    # duplicate-index pin below, which does not fall back at all — and strict
    # went red on exactly the rule I had documented that morning. `mix test`
    # could not see it, which is the entire point of that rule.
    @tag :host_fallback_expected
    test "a rank-5 operand that DOES need rotating still falls back" do
      # Contracting axis 0 of a rank-5 puts a non-identity permutation in front
      # of transpose_nd, which cannot address five dims. This must fall back
      # rather than transpose wrongly.
      a = Nx.iota({2, 3, 4, 5, 6}, type: {:f, 32}, backend: VulkanoBackend)
      b = Nx.iota({2}, type: {:f, 32}, backend: VulkanoBackend)
      assert Nx.Vulkan.Fallback.count_total(fn -> Nx.dot(a, [0], [], b, [0], []) end) > 0

      got = Nx.dot(a, [0], [], b, [0], [])
      ref = Nx.dot(Nx.backend_transfer(a, Nx.BinaryBackend), [0], [],
                   Nx.backend_transfer(b, Nx.BinaryBackend), [0], [])
      assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
    end
  end

  describe "indexed_put / indexed_add over a non-prefix axis subset" do
    test "axes: [0, 2] on a rank-3 target — Nx's own doctest" do
      check(fn b ->
        Nx.indexed_put(
          Nx.iota({1, 2, 3}, backend: b),
          Nx.tensor([[0, 0], [0, 2]], backend: b),
          Nx.tensor([[0, 30], [20, 50]], backend: b),
          axes: [0, 2]
        )
      end)
    end

    test "every axis subset of a rank-3 target, both ops" do
      # The rotation has to put the indexed axes at the front IN THE ORDER
      # GIVEN, because index column j addresses axes[j]. A permutation that
      # sorted them would pass the prefix case and quietly transpose the
      # columns.
      #
      # The two index rows must be DISTINCT. Writing both to the same slot is a
      # race on the GPU and last-wins on the host — a real divergence, but not
      # this test's subject, and it is pinned separately below. The first cut of
      # this test used all-zero indices and failed on every subset INCLUDING the
      # prefix ones, which is what made it obvious the rotation was not the
      # cause.
      shape = {2, 3, 4}
      dims = Tuple.to_list(shape)

      for axes <- [[0], [1], [2], [0, 1], [0, 2], [1, 2]] do
        # row 0 at the origin, row 1 at the far corner of the indexed axes
        row0 = Enum.map(axes, fn _ -> 0 end)
        row1 = Enum.map(axes, &(elem(shape, &1) - 1))
        idx = Nx.tensor([row0, row1])

        free = Enum.reject(0..2, &(&1 in axes))
        upd_shape = List.to_tuple([2 | Enum.map(free, &Enum.at(dims, &1))])

        for op <- [:indexed_put, :indexed_add] do
          check(fn b ->
            apply(Nx, op, [
              Nx.iota(shape, backend: b),
              Nx.backend_transfer(idx, b),
              Nx.add(Nx.iota(upd_shape, backend: b), 100),
              [axes: axes]
            ])
          end)
        end
      end
    end

    test "the rotation is inverted, not merely applied" do
      # Scattering in permuted space and forgetting to rotate back returns a
      # tensor of the right SHAPE with its axes swapped — which for a symmetric
      # shape would still pass a shape assertion. {1, 2, 3} is deliberately not
      # symmetric.
      got =
        Nx.indexed_put(
          Nx.iota({1, 2, 3}, backend: VulkanoBackend),
          Nx.tensor([[0, 0], [0, 2]], backend: VulkanoBackend),
          Nx.tensor([[0, 30], [20, 50]], backend: VulkanoBackend),
          axes: [0, 2]
        )

      assert Nx.to_flat_list(got) == [0, 1, 20, 30, 4, 50]
    end
  end

  describe "DUPLICATE indices — HARDWARE-DEPENDENT, and that is the finding" do
    # NOT introduced by the axis rotation, and not closed by it. Found because
    # the first cut of the subset test above used all-zero indices, which made
    # every subset fail — including the prefix ones that never touch the new
    # code.
    #
    # `indexed_put` writes without ordering, so two index rows naming the same
    # slot RACE. `Nx.BinaryBackend` applies updates in order and the LAST one
    # wins. This backend keeps whichever invocation wrote last in hardware, and
    # **the fleet gives two different stable answers**:
    #
    #     BinaryBackend         [30, 0, 0]  last update  (defined)
    #     RTX 3060 Ti (Ampere)  [10, 0, 0]  FIRST row    (stable, 5/5)
    #     GT 650M  (Kepler)     [30, 0, 0]  LAST row     (stable, 10/10)
    #     GT 750M  (Kepler)     [30, 0, 0]  LAST row     (stable, 10/10)
    #     Tegra X1 (Maxwell)    [30, 0, 0]  LAST row     (stable, 25/25)
    #
    # Two stable answers that disagree across boxes in one fleet, and **the
    # majority agrees with the host** — three of four boxes return the correct
    # answer by accident. Ampere is the outlier. That is worse than a uniform
    # divergence: the bug is invisible on most of the fleet, and a developer on
    # a Kepler or the Jetson would have no reason to suspect it.
    #
    # Nothing in Vulkan orders two writes to the same address from different
    # invocations, so all of these are permitted.
    #
    # THIS TEST USED TO ASSERT `refute gpu == host`, which was Ampere's answer
    # written down as if it were the rule. It went red on both Keplers, where
    # the race resolves the other way and the GPU coincidentally AGREES. The
    # pin now asserts only what is true everywhere — that the result is one of
    # the updates and is stable within a box — and records WHICH answer each box
    # gives in the table above rather than in an assertion.
    #
    # Fixing the underlying race means a two-pass scatter: atomicMax the winning
    # ROW INDEX into an output-sized scratch buffer, then a second dispatch to
    # write that row's value. Two extra dispatches and an allocation on EVERY
    # indexed_put, to be correct on an input no doctest exercises. That is a
    # cost/benefit call for the operator, so it is measured and pinned rather
    # than quietly fixed or quietly ignored.
    test "indexed_put with duplicate indices returns SOME update, stably" do
      idx = Nx.tensor([[0], [0], [0]])
      upd = Nx.tensor([10, 20, 30])

      run = fn ->
        Nx.to_flat_list(
          Nx.indexed_put(
            Nx.tensor([0, 0, 0], backend: VulkanoBackend),
            Nx.backend_transfer(idx, VulkanoBackend),
            Nx.backend_transfer(upd, VulkanoBackend)
          )
        )
      end

      host = Nx.indexed_put(Nx.tensor([0, 0, 0], backend: Nx.BinaryBackend), idx, upd)
      assert Nx.to_flat_list(host) == [30, 0, 0], "BinaryBackend applies updates in order"

      got = run.()

      # It must be one of the updates — never a stale target value, never
      # garbage, and never a partial write.
      assert hd(got) in [10, 20, 30],
             "expected one of the updates, got #{inspect(got)}"

      assert tl(got) == [0, 0], "untouched slots must be untouched"

      # Stable WITHIN a box. Both observed behaviours are stable; an unstable
      # one would be a different and worse problem, so it is worth separating.
      assert Enum.uniq(for(_ <- 1..5, do: run.())) == [got],
             "the duplicate-index result varied between runs on this box"
    end

    test "indexed_add with duplicate indices is CORRECT, and that is not luck" do
      # Addition is commutative and the shader uses atomics, so ordering cannot
      # change the answer. This is the contrast that shows the problem above is
      # specifically about WRITE ordering and not about scatter generally — and
      # it holds on every box in the fleet.
      idx = Nx.tensor([[0], [0], [0]])
      upd = Nx.tensor([10, 20, 30])

      gpu =
        Nx.indexed_add(
          Nx.tensor([0, 0, 0], backend: VulkanoBackend),
          Nx.backend_transfer(idx, VulkanoBackend),
          Nx.backend_transfer(upd, VulkanoBackend)
        )

      assert Nx.to_flat_list(gpu) == [60, 0, 0]
    end
  end

  describe "slice with a dynamic (tensor) start" do
    test "a rank-0 tensor start — Nx's own doctest" do
      check(fn b ->
        Nx.slice_along_axis(Nx.iota({2, 5}, backend: b), Nx.tensor(0, backend: b), 1, axis: 0)
      end)
    end

    test "CLAMPING — a dynamic start out of range does not raise, it clamps" do
      # This is what makes it more than a readback. Nx clamps a dynamic start
      # into [0, dim - len]; the shader has no such logic and would index off
      # the end of the buffer, where robust_buffer_access returns ZEROS. A
      # silently wrong answer, which is this backend's worst failure mode.
      for {starts, lengths} <- [
            {[0, 0], [1, 3]},
            {[1, 3], [1, 3]},
            {[5, 9], [1, 3]},
            {[-3, -1], [1, 3]},
            {[1, 1], [1, 3]}
          ] do
        check(fn b ->
          Nx.slice(
            Nx.iota({2, 5}, backend: b),
            Enum.map(starts, &Nx.tensor(&1, backend: b)),
            lengths
          )
        end)
      end
    end

    test "dynamic starts combine with strides" do
      check(fn b ->
        Nx.slice(
          Nx.iota({2, 5}, backend: b),
          [Nx.tensor(1, backend: b), Nx.tensor(1, backend: b)],
          [1, 3],
          strides: [1, 2]
        )
      end)
    end

    test "mixed static and dynamic starts in one call" do
      check(fn b ->
        Nx.slice(Nx.iota({2, 5}, backend: b), [0, Nx.tensor(2, backend: b)], [2, 2])
      end)
    end

    test "f64 and integer sources, since the path is dtype-generic" do
      for type <- [{:f, 64}, {:s, 32}, {:u, 32}] do
        check(fn b ->
          Nx.slice(Nx.iota({3, 4}, type: type, backend: b), [Nx.tensor(1, backend: b), 1], [2, 2])
        end)
      end
    end
  end
end
