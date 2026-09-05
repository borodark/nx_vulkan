defmodule Nx.Vulkan.StrictFallbackTest do
  @moduledoc """
  Tests strict mode itself: the three modes, the allowlist, and the scoping.

  `Nx.Vulkan.Fallback.count/1` makes a silent fallback *detectable if you wrote
  the right assertion*. Strict mode makes it impossible to miss. These tests are
  the ones that keep it honest — a strict mode that silently stopped refusing
  would look exactly like a backend with no fallbacks left.
  """

  use ExUnit.Case, async: true

  import ExUnit.CaptureLog

  alias Nx.Vulkan.Fallback
  alias Nx.Vulkan.HostFallbackError
  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp t(shape, seed \\ 1) do
    size = Tuple.product(shape)
    data = for i <- 1..size, do: :math.sin(seed * 0.7 + i * 0.41) + 1.5
    Nx.tensor(data, type: {:f, 64}, backend: VulkanoBackend) |> Nx.reshape(shape)
  end

  # A fallback that is NOT on the allowlist. `quotient` has no shader and no
  # exemption; picked because it is unlikely ever to get one.
  defp refused_fallback do
    a = Nx.tensor([6, 8, 10], type: {:s, 64}, backend: VulkanoBackend)
    b = Nx.tensor([2, 2, 5], type: {:s, 64}, backend: VulkanoBackend)
    Nx.quotient(a, b)
  end

  # A fallback that IS on the allowlist: sort has no shader and no plan for one.
  defp allowed_fallback, do: Nx.sort(t({8}), axis: 0)

  describe ":allow — the default, and it must stay the default" do
    test "an unconfigured application falls back to :allow" do
      # The library's selling point is that every Nx op works. A default that
      # raised on a correct-but-slow path would destroy that. Read the key the
      # way mode/0 does, from an app that has never set it.
      assert Application.get_env(:no_such_app, :host_fallback, :allow) == :allow
      assert Fallback.strict(:allow, fn -> Fallback.mode() end) == :allow
    end

    test "computes the fallback and returns the right answer" do
      got = Fallback.strict(:allow, fn -> refused_fallback() end)
      assert Nx.to_flat_list(got) == [3, 4, 2]
    end

    test "still counts" do
      assert Fallback.strict(:allow, fn ->
               Fallback.count_total(fn -> refused_fallback() end)
             end) == 1
    end
  end

  describe ":warn" do
    test "logs the refused op and returns the same answer" do
      log =
        capture_log(fn ->
          got = Fallback.strict(:warn, fn -> refused_fallback() end)
          assert Nx.to_flat_list(got) == [3, 4, 2]
        end)

      assert log =~ "host fallback reported: quotient/3"
      assert log =~ "{3}"
    end

    test "says nothing about an allowlisted op" do
      log = capture_log(fn -> Fallback.strict(:warn, fn -> allowed_fallback() end) end)
      refute log =~ "host fallback"
    end
  end

  describe ":raise" do
    test "raises on a fallback that is not on the allowlist" do
      assert_raise HostFallbackError, ~r/host fallback refused: quotient\/3/, fn ->
        Fallback.strict(fn -> refused_fallback() end)
      end
    end

    test "the error carries the op, the shape and the dtype" do
      err =
        assert_raise HostFallbackError, fn ->
          Fallback.strict(fn -> refused_fallback() end)
        end

      assert err.op == {:quotient, 3}
      assert err.shape == {3}
      assert err.type == {:s, 64}
    end

    test "the message points at the defect class, not just the symptom" do
      err =
        assert_raise HostFallbackError, fn -> Fallback.strict(fn -> refused_fallback() end) end

      # §1b of the skill is where the "gate written against the forward pass"
      # story lives. A strict-mode failure with no pointer to it just looks
      # like an annoying assertion.
      assert err.message =~ "vulkan-nx-compute/SKILL.md"
      assert err.message =~ "BACKWARD_PASS_AUDIT.md"
      assert err.message =~ "@allowlist"
    end

    test "does NOT raise on an allowlisted op" do
      got = Fallback.strict(fn -> allowed_fallback() end)
      assert Nx.size(got) == 8
    end

    test "does not raise on a native op" do
      assert Fallback.strict(fn -> Nx.add(t({4, 4}), t({4, 4}, 2)) end) |> Nx.size() == 16
    end

    test "fires on the FIRST refused op, so it attributes the cause not the cascade" do
      # The counter is a lower bound: once a tensor lands on BinaryBackend,
      # everything after it computes there unrecorded. Raising happens before
      # the tensor leaves, so the op named is the one that started it.
      err =
        assert_raise HostFallbackError, fn ->
          Fallback.strict(fn ->
            refused_fallback()
            |> Nx.sort()
            |> Nx.sum()
          end)
        end

      assert err.op == {:quotient, 3}
    end
  end

  describe "the allowlist" do
    test "every entry names one {fun, arity} or one Nx.Block struct, a condition, and a reason" do
      float_meta = Nx.tensor([[1.0, 2.0]], type: {:f, 64}, backend: Nx.BinaryBackend)

      for {op, condition, reason} <- Fallback.allowlist() do
        # Exactly two shapes are legal, and both name ONE thing:
        #   {fun, arity}          — one backend callback
        #   {:block, Nx.Block.X}  — one block kind (T13)
        # The second exists because block/4 dispatches an entire API family
        # through a single callback, so {:block, 4} would exempt Nx.LinAlg,
        # top_k, cumulative_* and all_close together. That is the wildcard the
        # next assertion forbids by name.
        assert match?({name, arity} when is_atom(name) and is_integer(arity), op) or
                 match?({:block, mod} when is_atom(mod), op),
               "allowlist entry #{inspect(op)} is neither a {fun, arity} pair nor a " <>
                 "{:block, Nx.Block.X} kind — no wildcards"

        refute match?({:block, arity} when is_integer(arity), op),
               "{:block, arity} exempts every Nx.Block struct at once — the op-family " <>
                 "wildcard this list exists to forbid. Name the struct."

        # Three conditions are legal, and the extra two exist because a reason
        # has to apply to the case it excuses. A fourth, `:float_output`, was
        # added when `{:pow, 3}` was found excusing INTEGER pow with an argument
        # about GLSL.std.450 lacking an f64 `pow` — true, and irrelevant to s32.
        # `{:dtype, t}` then superseded it for that same entry, because once f32
        # broadcasting pow moved onto the GPU `:float_output` would have gone on
        # excusing an f32 fallback that had become a bug. That left it carried by
        # no entry, hence reachable by no test, and it was deleted on 2026-09-05.
        assert condition == :always or
                 match?({:rank_at_least, n} when is_integer(n), condition) or
                 match?({:dtype, {k, b}} when is_atom(k) and is_integer(b), condition)

        assert is_binary(reason) and byte_size(reason) > 40,
               "allowlist entry #{inspect(op)} has no real reason: #{inspect(reason)}"

        # The assertion above checks the condition against a list written HERE,
        # which drifts. This one checks it against the code that consumes it: a
        # condition with no `condition_met?/2` clause compiles cleanly and
        # raises FunctionClauseError the first time an op carrying it is
        # checked — under `:raise`, on a refused op, in somebody else's suite.
        assert is_boolean(Fallback.allowed?(op, float_meta))
        assert is_boolean(Fallback.allowed?(op, nil))
      end
    end

    test "no duplicate {op, condition} pairs — a second entry can never be reached" do
      keys = for {op, condition, _reason} <- Fallback.allowlist(), do: {op, condition}
      dupes = keys -- Enum.uniq(keys)

      assert dupes == [],
             "duplicate allowlist entries are dead lines nobody will notice: #{inspect(dupes)}"
    end

    test "a block kind is exempt alone — its neighbours in the family still raise" do
      # T13's whole point: block/4 is one callback for a large API surface, so
      # attribution must be per Nx.Block struct. all_close is this suite's own
      # assertion helper and must not raise; a missing cumulative_sum shader is
      # a genuine gap and must.
      assert Fallback.allowed?({:block, Nx.Block.AllClose})
      assert Fallback.allowed?({:block, Nx.Block.LinAlg.SVD})

      refute Fallback.allowed?({:block, Nx.Block.CumulativeSum})
      refute Fallback.allowed?({:block, Nx.Block.TopK})
      refute Fallback.allowed?({:block, Nx.Block.Take})
      refute Fallback.allowed?({:block, Nx.Block.LogicalNot})
    end

    test "entries are unique — an op is exempt for exactly one stated reason" do
      ops = Enum.map(Fallback.allowlist(), fn {op, _c, _r} -> op end)
      assert ops == Enum.uniq(ops)
    end

    test "a rank-gated entry does not exempt the ranks the shader handles" do
      # This is the whole design: {:transpose, 3} is not exempt, only
      # {:transpose, 3} at rank >= 5 is. A rank-4 transpose falling back is
      # the bug that started the audit, and must still raise.
      refute Fallback.allowed?({:transpose, 3}, t({2, 3, 4, 5}))
      assert Fallback.allowed?({:transpose, 3}, t({2, 2, 2, 2, 2}))
      refute Fallback.allowed?({:transpose, 3}, nil)
    end

    test "rank-5 transpose is permitted; rank-4 transpose is not" do
      # rank 5 is past what transpose_nd handles, and is allowlisted.
      Fallback.strict(fn -> Nx.transpose(t({2, 2, 2, 2, 2}), axes: [4, 3, 2, 1, 0]) end)
    end

    test "an op with no entry at all is refused" do
      refute Fallback.allowed?({:quotient, 3}, t({3}))
    end
  end

  describe "per-process scoping" do
    test "a strict scope does not leak to another process" do
      # The reason this matters: one strict test must not poison an async
      # suite. `Fallback` is per-process, and strict mode has to be too.
      parent = self()
      app_default = Application.get_env(:nx_vulkan, :host_fallback, :allow)

      Fallback.strict(:raise, fn ->
        task =
          Task.async(fn ->
            send(parent, {:mode, Fallback.mode()})
            Fallback.strict(:allow, fn -> refused_fallback() end)
            :ok
          end)

        assert Task.await(task) == :ok
      end)

      # The task saw the application default, NOT the parent's :raise scope.
      assert_receive {:mode, ^app_default}
    end

    test "restores the previous mode afterwards" do
      before = Fallback.mode()
      Fallback.strict(fn -> :ok end)
      assert Fallback.mode() == before
    end

    test "restores the previous mode even when the body raises" do
      before = Fallback.mode()
      assert_raise HostFallbackError, fn -> Fallback.strict(fn -> refused_fallback() end) end
      assert Fallback.mode() == before
    end

    test "nests, inner wins, outer resumes" do
      Fallback.strict(:raise, fn ->
        assert Fallback.mode() == :raise
        Fallback.strict(:allow, fn -> assert Fallback.mode() == :allow end)
        assert Fallback.mode() == :raise
      end)
    end

    test ":allow inside :raise is the documented escape hatch" do
      Fallback.strict(fn ->
        got = Fallback.strict(:allow, fn -> refused_fallback() end)
        assert Nx.to_flat_list(got) == [3, 4, 2]
      end)
    end

    test "composes with count/1" do
      {_r, counts} =
        Fallback.count(fn ->
          Fallback.strict(:allow, fn -> refused_fallback() end)
        end)

      assert counts == %{{:quotient, 3} => 1}
    end
  end

  describe "clip/4 — the false positive strict mode found" do
    test "an op that composes GPU primitives is not recorded as a fallback" do
      # clip/4 computes min(max(t, lo), hi) with the broadcast shaders and stays
      # resident, but wrapped its result in host_result/2 and so was counted —
      # and, under :raise, refused — for a round trip it never made.
      v = Nx.subtract(Nx.iota({4, 5}, type: {:f, 32}, backend: VulkanoBackend), 8.0)

      got = Fallback.strict(fn -> Nx.clip(v, 0.0, 5.0) end)
      assert match?(%VulkanoBackend{}, got.data)
      assert Fallback.count_total(fn -> Nx.clip(v, 0.0, 5.0) end) == 0
    end
  end

  describe "reduce attribution" do
    test "a host reduce names the reduction, not the shared helper" do
      # sum/reduce_max/reduce_min share reduce_op_host_fallback/4; the
      # __CALLER__.function capture blamed the helper, so a refusal said
      # `reduce_op_host_fallback/4` and named no Nx op at all.
      #
      # THE SUBJECT HAS NOW MOVED TWICE, and the second time is the lesson.
      #
      # It first reduced a u8 mask with `sum`, which reduce_axis_u8_to_u32 made
      # native. It was then rewritten to `reduce_max` on the same mask, "which
      # would need a byte-packed writer rather than a word one" — and
      # cast_s32_to_narrow.comp is now that writer, so it broke again for
      # exactly the reason its own comment predicted.
      #
      # A test ABOUT ATTRIBUTION should not be anchored to a gap someone is
      # trying to close. {:s, 64} is anchored to a decided one: the reduce
      # shaders accumulate in 32 bits and Int64 is a device capability this
      # backend does not require (MISSION.md 3.2). If that ever changes, this
      # test should be re-pointed rather than "fixed" — the assertion it exists
      # to make is that the refusal names `reduce_max/3` and not the shared
      # helper `reduce_op_host_fallback/4`.
      wide = Nx.tensor([[1, 2], [3, 4]], type: {:s, 64}, backend: VulkanoBackend)

      {_r, counts} =
        Fallback.count(fn ->
          Fallback.strict(:allow, fn -> Nx.reduce_max(wide, axes: [1]) end)
        end)

      assert counts == %{{:reduce_max, 3} => 1},
             "expected the reduction to name itself, got #{inspect(counts)}"
    end
  end
end
