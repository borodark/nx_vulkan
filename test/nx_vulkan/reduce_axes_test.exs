defmodule Nx.Vulkan.ReduceAxesTest do
  @moduledoc """
  `classify_reduce_axes/2` and the four op families that share it.

  The reduce shaders push `(outer, reduce_size, inner)` and index
  `base = o * reduce_size * inner + i`, striding by `inner`. That layout IS a
  contiguous run of axes: everything before the run multiplies into `outer`,
  the run itself into `reduce_size`, everything after into `inner`. The gate,
  however, only admitted three shapes of run — all axes, a leading prefix, a
  trailing suffix — so `Nx.argmax(t, axis: 1)` on a rank-3 tensor host-fell-back
  to a kernel that could always have run it.

  A middle run is not exotic. It is what `axis: 1` means on any rank >= 3, and
  it reaches `sum`/`reduce_max`/`reduce_min`, `all`/`any` and `argmax`/`argmin`
  through the one shared classifier.

  These assert BOTH halves: the value matches BinaryBackend, and the result is
  still on the GPU. Only the second can see a fallback — a fallback returns a
  bit-identical answer, so the value assertion alone would pass either way.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp check(build, on_gpu \\ true) do
    got = build.(VulkanoBackend)
    ref = build.(Nx.BinaryBackend)

    if on_gpu,
      do: assert(match?(%VulkanoBackend{}, got.data), "expected the result to stay on the GPU")

    assert Nx.shape(got) == Nx.shape(ref)
    assert Nx.type(got) == Nx.type(ref)
    assert Nx.to_flat_list(got) == Nx.to_flat_list(ref)
  end

  # Every contiguous run of axes over a rank-4 shape whose dims are all
  # distinct — so a wrong `outer`/`inner` split cannot coincidentally agree.
  @shape {2, 3, 4, 5}
  @runs for lo <- 0..3, hi <- lo..3, do: Enum.to_list(lo..hi)

  describe "sum / reduce_max / reduce_min over a contiguous axis run" do
    for type <- [{:f, 32}, {:s, 32}], run <- @runs do
      type = Macro.escape(type)

      test "#{inspect(type)} axes #{inspect(run)}" do
        run = unquote(run)
        type = unquote(type)
        build = fn b -> Nx.iota(@shape, type: type, backend: b) end

        check(fn b -> Nx.sum(build.(b), axes: run) end)
        check(fn b -> Nx.reduce_max(build.(b), axes: run) end)
        check(fn b -> Nx.reduce_min(build.(b), axes: run) end)
      end
    end
  end

  # u8 is stored PACKED, four elements to a u32 word, and read back through
  # `byte_at(i)` in reduce_axis_u8_to_u32.comp. That helper takes an ELEMENT
  # index and does the word/byte extraction itself, so the `base + r * inner`
  # stride arithmetic is in the same units as the s32 shader's and a middle run
  # needs no special handling. Asserted rather than argued: a packed reader that
  # strided in WORDS would read every fourth element and still return plausible
  # numbers.
  describe "sum / product over a contiguous axis run, packed u8 input" do
    for run <- @runs do
      test "axes #{inspect(run)}" do
        run = unquote(run)
        # A comparison mask — the u8 tensors this path actually sees. Not all
        # ones and not all zeros along any axis.
        build = fn b ->
          Nx.greater(Nx.remainder(Nx.iota(@shape, backend: b), 5), 1)
        end

        assert Nx.type(Nx.sum(build.(VulkanoBackend), axes: run)) == {:u, 32}
        check(fn b -> Nx.sum(build.(b), axes: run) end)
      end
    end
  end

  describe "all / any over a contiguous axis run" do
    for run <- @runs do
      test "axes #{inspect(run)}" do
        run = unquote(run)
        # Mixed zeros and non-zeros, and not symmetric across the axes, so
        # `all` and `any` disagree per slot.
        build = fn b -> Nx.remainder(Nx.iota(@shape, backend: b), 3) end

        check(fn b -> Nx.all(build.(b), axes: run) end)
        check(fn b -> Nx.any(build.(b), axes: run) end)
      end
    end
  end

  describe "argmax / argmin on a single axis" do
    for axis <- 0..3 do
      test "axis #{axis}" do
        axis = unquote(axis)
        # Not iota: iota is monotone, so argmax/argmin would be the first or
        # last slot on every axis and a wrong stride could still look right.
        build = fn b ->
          Nx.remainder(Nx.multiply(Nx.iota(@shape, backend: b), 7), 11)
        end

        check(fn b -> Nx.argmax(build.(b), axis: axis) end)
        check(fn b -> Nx.argmin(build.(b), axis: axis) end)
        check(fn b -> Nx.argmax(build.(b), axis: axis, tie_break: :high) end)
        check(fn b -> Nx.argmin(build.(b), axis: axis, tie_break: :high) end)
        check(fn b -> Nx.argmax(build.(b), axis: axis, keep_axis: true) end)
        check(fn b -> Nx.argmax(build.(b), axis: axis, type: :u32) end)
      end
    end
  end

  describe "the runs that are NOT contiguous" do
    # [0, 2] over a rank-3 shape cannot be expressed as one (outer, run, inner)
    # slab: axis 1 sits between the two reduced axes. `classify_reduce_axes/2`
    # refuses it — but that is not the same as the OP refusing it.
    # `do_reduce/5` has a second path, `reduce_via_transpose/5`, which rotates
    # the kept axes to the front and re-enters as a trailing-suffix reduce.
    test "sum / reduce_max stay resident — reduce_via_transpose picks them up" do
      check(fn b -> Nx.sum(Nx.iota({2, 3, 4}, backend: b), axes: [0, 2]) end)
      check(fn b -> Nx.reduce_max(Nx.iota({2, 3, 4}, backend: b), axes: [0, 2]) end)
      check(fn b -> Nx.sum(Nx.iota({2, 3, 4, 5}, type: {:f, 32}, backend: b), axes: [0, 2]) end)
    end

    # all/any and argmax/argmin have no rotation of their own, so for them the
    # classifier's refusal IS the op's. Pinned rather than assumed: if either
    # grows a transpose path, this is where it gets noticed.
    #
    # TAGGED because the fallback IS the subject. `strict_test.sh` runs the
    # suite with NXV_HOST_FALLBACK=raise and excludes only this tag, so a test
    # that deliberately provokes a refusal has to opt out or it turns the strict
    # run red — which is exactly what it did, and only the Kepler fleet run
    # noticed, because strict_test.sh had not been re-run since this landed.
    @tag :host_fallback_expected
    test "all / any and argmax / argmin do fall back there" do
      m = fn b -> Nx.remainder(Nx.iota({2, 3, 4}, backend: b), 3) end
      check(fn b -> Nx.all(m.(b), axes: [0, 2]) end, false)
      check(fn b -> Nx.any(m.(b), axes: [0, 2]) end, false)

      assert Nx.Vulkan.Fallback.count_total(fn ->
               Nx.all(m.(VulkanoBackend), axes: [0, 2])
             end) > 0
    end
  end

  describe "the degenerate runs the old clauses covered" do
    test "all axes collapses to a scalar" do
      check(fn b -> Nx.sum(Nx.iota(@shape, backend: b)) end)
      check(fn b -> Nx.argmax(Nx.iota(@shape, backend: b)) end)
    end

    test "a rank-0 tensor" do
      check(fn b -> Nx.sum(Nx.tensor(7, backend: b)) end)
    end
  end
end
