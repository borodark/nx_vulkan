defmodule Nx.Vulkan.VulkanoBackend.HostFallbackTest do
  @moduledoc """
  Targeted regression tests for the host-fallback callbacks in
  VulkanoBackend — the ops that download to BinaryBackend, run the
  compute there, and (per Tier 1 of SHAPE_C_PLAN.md) return the
  result on BinaryBackend rather than uploading back to vulkano.

  These tests cover:
  - Correctness: result matches direct BinaryBackend computation
  - Tier 1 contract: result tensors stay on BinaryBackend
  - take/4 regression: opts must flow through verbatim, not get
    rewrapped as `axis: opts` (that bug surfaced in the bench)
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  @moduletag :vulkan_live

  # The subject of this module IS the host-fallback path — it asserts the Tier 1
  # contract that a fallback's result stays on BinaryBackend. Excluded from the
  # strict run (scripts/strict_test.sh), which asserts fallbacks do not happen.
  @moduletag :host_fallback_expected

  defp v(t), do: Nx.backend_transfer(t, VulkanoBackend)

  defp f32(list_or_int, shape \\ nil) when not is_nil(list_or_int) do
    base = Nx.tensor(list_or_int, type: :f32, backend: Nx.BinaryBackend)
    if shape, do: Nx.reshape(base, shape), else: base
  end

  describe "Tier 1 — host-fallback results stay on BinaryBackend" do
    test "concatenate axis-0 all-vulkano: result stays on VulkanoBackend (Tier 2 fast path)" do
      # Tier 2 step 1: outer-axis concat of vulkano-resident inputs
      # routes to concat_buffers NIF (vkCmdCopyBuffer), result stays
      # GPU-resident. Older Tier 1 behaviour was BinaryBackend; that
      # path still applies to mixed-backend or non-outer-axis concats
      # (see "concatenate mixed-backend falls back" below).
      a = v(f32([1.0, 2.0, 3.0]))
      b = v(f32([4.0, 5.0, 6.0]))
      r = Nx.concatenate([a, b])
      assert r.data.__struct__ == VulkanoBackend
      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "concatenate mixed-backend falls back to BinaryBackend host path" do
      a = v(f32([1.0, 2.0, 3.0]))
      b = f32([4.0, 5.0, 6.0])
      r = Nx.concatenate([a, b])
      assert r.data.__struct__ == Nx.BinaryBackend
      assert Nx.to_flat_list(r) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    # pad now runs a GPU type-generic copy shader (thrust 2) for static config +
    # 4/8-byte dtypes — stays on VulkanoBackend.
    test "pad (f32) stays on VulkanoBackend and is correct" do
      a = v(f32([1.0, 2.0, 3.0]))
      pv = v(Nx.tensor(0.0, type: :f32, backend: Nx.BinaryBackend))
      r = Nx.pad(a, pv, [{1, 1, 0}])
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [0.0, 1.0, 2.0, 3.0, 0.0]
    end

    # broadcast got an index-remap shader (broadcast_nd) and now stays on the
    # GPU, like pad above. It used to strand its result on BinaryBackend, which
    # is what made select/4 fall back on a relu gradient's zeros.
    test "broadcast (f32) stays on VulkanoBackend and is correct" do
      a = v(Nx.tensor(7.0, type: :f32, backend: Nx.BinaryBackend))
      r = Nx.broadcast(a, {4})
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [7.0, 7.0, 7.0, 7.0]
    end

    # slice now runs a GPU strided-copy shader (thrust 2) for static starts +
    # 4/8-byte dtypes — stays on VulkanoBackend.
    test "slice (static start, f32) stays on VulkanoBackend and is correct" do
      a = v(f32([1.0, 2.0, 3.0, 4.0, 5.0]))
      r = Nx.slice(a, [1], [3])
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [2.0, 3.0, 4.0]
    end

    # put_slice got an index-remap overlay shader (T11) and now stays on the
    # GPU, like pad and slice above. This test used to pin it to BinaryBackend;
    # the pin failing is the intended signal that the op was promoted.
    test "put_slice (f32) stays on VulkanoBackend and is correct" do
      target = v(f32([0.0, 0.0, 0.0, 0.0]))
      slice = v(f32([7.0, 8.0]))
      r = Nx.put_slice(target, [1], slice)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [0.0, 7.0, 8.0, 0.0]
    end

    # indexed_put got `glsl/scatter.comp` and now stays on the GPU, like
    # put_slice above. This test used to pin it to BinaryBackend; the pin
    # failing is the intended signal that the op was promoted.
    #
    # Note that indexed_ADD below did NOT move, and that is the interesting
    # half: the two ops share a shader and differ only in whether duplicate
    # indices must accumulate. `indexed_put` documents its race, so a plain
    # write is the specified behaviour at any dtype; `indexed_add` needs an
    # atomic, and an f32 one needs GL_EXT_shader_atomic_float, which the Kepler
    # fleet does not guarantee. Integer indexed_add does run on the GPU.
    test "indexed_put (f32) stays on VulkanoBackend and is correct" do
      target = v(f32([0.0, 0.0, 0.0, 0.0]))
      idx = Nx.tensor([[0], [2]], type: :s64, backend: Nx.BinaryBackend)
      upd = Nx.tensor([1.0, 3.0], type: :f32, backend: Nx.BinaryBackend)
      r = Nx.indexed_put(target, idx, upd)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [1.0, 0.0, 3.0, 0.0]
    end

    test "indexed_add on INTEGERS stays on VulkanoBackend, unlike the f32 case" do
      target = Nx.tensor([10, 10, 10], type: :s32, backend: VulkanoBackend)
      idx = Nx.tensor([[0], [2], [0]], type: :s32, backend: Nx.BinaryBackend)
      upd = Nx.tensor([1, 3, 5], type: :s32, backend: Nx.BinaryBackend)
      r = Nx.indexed_add(target, idx, upd)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      # Duplicate index 0 accumulates: 10 + 1 + 5. That is the atomic working.
      assert Nx.to_flat_list(r) == [16, 10, 13]
    end

    test "indexed_add result is on BinaryBackend" do
      target = v(f32([10.0, 10.0, 10.0]))
      idx = Nx.tensor([[0], [2]], type: :s64, backend: Nx.BinaryBackend)
      upd = Nx.tensor([1.0, 3.0], type: :f32, backend: Nx.BinaryBackend)
      r = Nx.indexed_add(target, idx, upd)
      assert r.data.__struct__ == Nx.BinaryBackend
      assert Nx.to_flat_list(r) == [11.0, 10.0, 13.0]
    end

    # gather now runs a GPU type-generic copy shader (thrust 2) for the common
    # leading-prefix / default-all-axes case — stays on VulkanoBackend.
    test "gather (default axes, f32) stays on VulkanoBackend and is correct" do
      a = v(f32([10.0, 20.0, 30.0, 40.0]))
      idx = v(Nx.tensor([[0], [2], [3]], type: :s64, backend: Nx.BinaryBackend))
      r = Nx.gather(a, idx)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [10.0, 30.0, 40.0]
    end

    # select now runs a GPU broadcast shader (thrust 2) for u8 pred + f32/f64
    # branches — stays on VulkanoBackend instead of host round-tripping.
    test "select (u8 pred, f32) stays on VulkanoBackend and is correct" do
      pred = v(Nx.tensor([1, 0, 1], type: {:u, 8}, backend: Nx.BinaryBackend))
      on_t = v(f32([1.0, 2.0, 3.0]))
      on_f = v(f32([10.0, 20.0, 30.0]))
      r = Nx.select(pred, on_t, on_f)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [1.0, 20.0, 3.0]
    end

    test "as_type cast result is on BinaryBackend" do
      a = v(f32([1.5, 2.7, 3.9]))
      r = Nx.as_type(a, :s32)
      assert r.data.__struct__ == Nx.BinaryBackend
      assert Nx.to_flat_list(r) == [1, 2, 3]
    end

    # argmax/argmin got glsl/argreduce_*.comp and now stay on the GPU. These
    # four used to pin them to BinaryBackend; the pins failing is the intended
    # signal that the ops were promoted.
    test "argmax result is on VulkanoBackend" do
      a = v(f32([1.0, 5.0, 2.0, 4.0]))
      r = Nx.argmax(a)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_number(r) == 1
    end

    test "argmax along axis (2D source)" do
      a = v(f32([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]]))
      r = Nx.argmax(a, axis: 1)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [1, 0]
    end

    test "argmin result is on VulkanoBackend" do
      a = v(f32([1.0, 5.0, 2.0, 4.0]))
      r = Nx.argmin(a)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_number(r) == 0
    end

    test "argmin along axis (2D source)" do
      a = v(f32([[1.0, 5.0, 2.0], [4.0, 0.0, 3.0]]))
      r = Nx.argmin(a, axis: 1)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [0, 1]
    end

    # clip now composes from GPU broadcast min/max (thrust 2) — it stays on
    # VulkanoBackend for same-type f32/f64 rather than host round-tripping.
    test "clip (f32) stays on VulkanoBackend and is correct" do
      a = v(f32([-1.0, 0.5, 1.5, 3.0]))
      lo = v(Nx.tensor(0.0, type: :f32, backend: Nx.BinaryBackend))
      hi = v(Nx.tensor(2.0, type: :f32, backend: Nx.BinaryBackend))
      r = Nx.clip(a, lo, hi)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [0.0, 0.5, 1.5, 2.0]
    end

    test "clip with mixed-backend bounds (bounds transferred to GPU)" do
      a = v(f32([-5.0, 0.0, 5.0]))
      lo = Nx.tensor(-1.0, type: :f32, backend: Nx.BinaryBackend)
      hi = Nx.tensor(1.0, type: :f32, backend: Nx.BinaryBackend)
      r = Nx.clip(a, lo, hi)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.to_flat_list(r) == [-1.0, 0.0, 1.0]
    end
  end

  describe "take/4 — opts keyword forwarded verbatim" do
    # Regression: the host-fallback signature `def take(out, t, idx, axis)`
    # bound the FULL opts keyword as a bare `axis`, then re-wrapped it as
    # `axis: axis`, producing `Nx.take(t, idx, axis: [axis: 0])` which
    # Nx rejects with "given axis ([axis: 0]) invalid for shape with rank 2".

    # W4: no longer a host fallback. `Nx.Block.Take` is routed on-device, and
    # at axis 0 the body's `gather/4` meets its GPU path (indexed axes are a
    # leading prefix), so the result never leaves the device. The BinaryBackend
    # indices are transferred up rather than dragging the operand down.
    test "take along axis 0 (2D source) stays on the GPU" do
      a = v(f32([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
      idx = Nx.tensor([0, 2], type: :s64, backend: Nx.BinaryBackend)
      r = Nx.take(a, idx, axis: 0)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.Vulkan.Fallback.count_total(fn -> Nx.take(a, idx, axis: 0) end) == 0
      assert Nx.to_flat_list(r) == [1.0, 2.0, 5.0, 6.0]
      assert Nx.shape(r) == {2, 2}
    end

    # Axis 1 now stays on the GPU too. The note above says axis 0 worked because
    # "indexed axes are a leading prefix" — gather no longer REQUIRES that, it
    # rotates the source with a transpose first, the way dot_orient/6 does for
    # matmul. This pin failing is the intended signal that the gate widened.
    test "take along axis 1 (2D source) stays on the GPU — the axes are rotated" do
      a = v(f32([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
      idx = Nx.tensor([0, 2], type: :s64, backend: Nx.BinaryBackend)
      r = Nx.take(a, idx, axis: 1)
      assert r.data.__struct__ == Nx.Vulkan.VulkanoBackend
      assert Nx.Vulkan.Fallback.count_total(fn -> Nx.take(a, idx, axis: 1) end) == 0
      assert Nx.to_flat_list(r) == [1.0, 3.0, 4.0, 6.0]
      assert Nx.shape(r) == {2, 2}
    end

    test "take from rank-1 source" do
      a = v(f32([10.0, 20.0, 30.0, 40.0]))
      idx = Nx.tensor([3, 1], type: :s64, backend: Nx.BinaryBackend)
      r = Nx.take(a, idx)
      assert Nx.to_flat_list(r) == [40.0, 20.0]
    end
  end

  describe "round-trip identity through host-fallback ops" do
    test "concatenate matches direct BinaryBackend result" do
      a_bin = f32([1.0, 2.0, 3.0, 4.0])
      b_bin = f32([5.0, 6.0, 7.0, 8.0])
      expected = Nx.concatenate([a_bin, b_bin])

      a_vk = v(a_bin)
      b_vk = v(b_bin)
      actual = Nx.concatenate([a_vk, b_vk])

      assert Nx.to_flat_list(actual) == Nx.to_flat_list(expected)
    end

    test "indexed_put matches direct BinaryBackend result" do
      target_bin = f32([0.0, 0.0, 0.0, 0.0, 0.0])
      idx_bin = Nx.tensor([[1], [3]], type: :s64, backend: Nx.BinaryBackend)
      upd_bin = Nx.tensor([5.0, 7.0], type: :f32, backend: Nx.BinaryBackend)
      expected = Nx.indexed_put(target_bin, idx_bin, upd_bin)

      target_vk = v(target_bin)
      actual = Nx.indexed_put(target_vk, idx_bin, upd_bin)

      assert Nx.to_flat_list(actual) == Nx.to_flat_list(expected)
    end
  end
end
