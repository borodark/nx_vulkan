defmodule Nx.Vulkan.BufferBatchTest do
  @moduledoc """
  `buf_download_many/1` — the batched readback behind `staging_read_many`.

  This exists because the batching was written for the three leapfrog chain
  NIFs, which read back four buffers per dispatch and paid four submit-and-fence
  pairs to do it, and **nothing in this repo can drive those NIFs**: their
  `parse_push_block` reads `d` at byte offset 8 while the synthesised shader's
  push block has `eps` there. So the chain path could not verify its own
  batching, and "the logic reads correctly" is not evidence — the ordering is
  the whole risk, and a transposition would be invisible to a size check.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.NativeV

  @moduletag :gpu

  defp f32(list), do: for(v <- list, into: <<>>, do: <<v::float-32-little>>)

  test "returns every buffer, in the order given" do
    # distinct CONTENT and distinct LENGTH, so both a reorder and an
    # off-by-one in the staging loop are visible.
    a = f32([1.0, 2.0, 3.0, 4.0])
    b = f32([10.0, 20.0])
    c = f32([100.0, 200.0, 300.0])

    {:ok, ra} = NativeV.buf_upload(a)
    {:ok, rb} = NativeV.buf_upload(b)
    {:ok, rc} = NativeV.buf_upload(c)

    assert {:ok, [got_a, got_b, got_c]} = NativeV.buf_download_many([ra, rb, rc])

    assert got_a == a
    assert got_b == b
    assert got_c == c
  end

  test "agrees with buf_download one at a time" do
    bins = for n <- 1..5, do: f32(Enum.map(1..(n * 3), &(&1 * 1.5)))
    refs = for bin <- bins, do: (fn -> {:ok, r} = NativeV.buf_upload(bin); r end).()

    {:ok, batched} = NativeV.buf_download_many(refs)
    singly = for r <- refs, do: (fn -> {:ok, d} = NativeV.buf_download(r); d end).()

    assert batched == singly
    assert batched == bins
  end

  test "a reversed ref list comes back reversed, not reordered by luck" do
    a = f32([1.0])
    b = f32([2.0, 2.0])
    c = f32([3.0, 3.0, 3.0])
    {:ok, ra} = NativeV.buf_upload(a)
    {:ok, rb} = NativeV.buf_upload(b)
    {:ok, rc} = NativeV.buf_upload(c)

    assert {:ok, [^c, ^b, ^a]} = NativeV.buf_download_many([rc, rb, ra])
  end

  test "single-element and empty lists" do
    {:ok, r} = NativeV.buf_upload(f32([7.0]))
    assert {:ok, [one]} = NativeV.buf_download_many([r])
    assert one == f32([7.0])
    assert {:ok, []} = NativeV.buf_download_many([])
  end

  test "sees writes a shader made, not a stale staging copy" do
    n = 1024
    spv = Path.expand("priv/shaders/elementwise_binary_f32.spv", File.cwd!())
    ones = for _ <- 1..n, into: <<>>, do: <<1.5::float-32-little>>
    {:ok, a} = NativeV.buf_upload(ones)
    {:ok, out1} = NativeV.buf_alloc(n * 4)
    {:ok, out2} = NativeV.buf_alloc(n * 4)

    # op 1 = multiply
    :ok = NativeV.apply_binary(out1, a, a, n, 1, spv)
    :ok = NativeV.apply_binary(out2, out1, a, n, 1, spv)

    # no explicit flush: buf_download_many must drain the pending batch itself
    assert {:ok, [d1, d2]} = NativeV.buf_download_many([out1, out2])

    assert <<first1::float-32-little, _::binary>> = d1
    assert <<first2::float-32-little, _::binary>> = d2
    assert_in_delta first1, 2.25, 1.0e-6
    assert_in_delta first2, 3.375, 1.0e-6
  end
end
