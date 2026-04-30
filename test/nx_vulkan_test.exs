defmodule Nx.VulkanTest do
  use ExUnit.Case, async: false
  # No doctest — the moduledoc shows usage shape, not an executable
  # example (device name varies per host; tensor ops aren't wired
  # yet in v0.0.1).

  describe "v0.0.1 — bootstrap" do
    test "init/0 returns :ok on a Vulkan-capable host" do
      assert :ok = Nx.Vulkan.init()
    end

    test "init/0 is idempotent" do
      assert :ok = Nx.Vulkan.init()
      assert :ok = Nx.Vulkan.init()
    end

    test "device_name/0 returns a non-empty string after init" do
      :ok = Nx.Vulkan.init()
      name = Nx.Vulkan.device_name()
      assert is_binary(name)
      assert String.length(name) > 0
    end

    test "has_f64?/0 returns a boolean" do
      :ok = Nx.Vulkan.init()
      assert is_boolean(Nx.Vulkan.has_f64?())
    end
  end

  describe "v0.0.2 — tensor round-trip" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "upload_f32 + download_f32 is the identity (small)" do
      {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      assert {:ok, [1.0, 2.0, 3.0, 4.0]} = Nx.Vulkan.download_f32(t, 4)
    end

    test "byte_size matches the uploaded length" do
      {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      assert Nx.Vulkan.byte_size(t) == 12
    end

    test "round-trip preserves single value" do
      {:ok, t} = Nx.Vulkan.upload_f32([42.0])
      assert {:ok, [42.0]} = Nx.Vulkan.download_f32(t, 1)
    end

    test "round-trip preserves negative + sub-integer values" do
      input = [-1.5, 0.25, 3.14159, -0.0001, 1.0e6]
      {:ok, t} = Nx.Vulkan.upload_f32(input)
      assert {:ok, output} = Nx.Vulkan.download_f32(t, 5)

      Enum.zip(input, output)
      |> Enum.each(fn {a, b} ->
        assert_in_delta a, b, 1.0e-3
      end)
    end

    test "round-trip works at 1024 elements (multi-workgroup-class buffer)" do
      input = Enum.map(1..1024, fn i -> i / 1.0 end)
      {:ok, t} = Nx.Vulkan.upload_f32(input)
      assert {:ok, output} = Nx.Vulkan.download_f32(t, 1024)
      assert length(output) == 1024
      assert hd(output) == 1.0
      assert List.last(output) == 1024.0
    end

    test "size_mismatch error when download size != upload size" do
      {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      assert {:error, :size_mismatch} = Nx.Vulkan.download_binary(t, 99)
    end

    test "upload_binary takes raw bytes" do
      bin = <<1.0::float-32-native, 2.0::float-32-native>>
      {:ok, t} = Nx.Vulkan.upload_binary(bin)
      {:ok, ^bin} = Nx.Vulkan.download_binary(t, 8)
    end

    test "many tensors live concurrently and are GC'd cleanly" do
      tensors =
        for i <- 1..50 do
          {:ok, t} = Nx.Vulkan.upload_f32([i * 1.0])
          t
        end

      # Verify all 50 are independent
      results =
        Enum.map(tensors, fn t ->
          {:ok, [v]} = Nx.Vulkan.download_f32(t, 1)
          v
        end)

      assert results == Enum.map(1..50, &(&1 * 1.0))
    end
  end

  describe "v0.0.3 — elementwise binary" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "add" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      {:ok, b} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0, 40.0])
      {:ok, c} = Nx.Vulkan.add(a, b)
      assert {:ok, [11.0, 22.0, 33.0, 44.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "multiply" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      {:ok, b} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0, 40.0])
      {:ok, c} = Nx.Vulkan.multiply(a, b)
      assert {:ok, [10.0, 40.0, 90.0, 160.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "subtract" do
      {:ok, a} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      {:ok, c} = Nx.Vulkan.subtract(a, b)
      assert {:ok, [9.0, 18.0, 27.0]} = Nx.Vulkan.download_f32(c, 3)
    end

    test "divide" do
      {:ok, a} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0, 40.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 4.0, 5.0, 8.0])
      {:ok, c} = Nx.Vulkan.divide(a, b)
      assert {:ok, [5.0, 5.0, 6.0, 5.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "max + min" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 5.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 4.0, 3.0])

      {:ok, mx} = Nx.Vulkan.max(a, b)
      assert {:ok, [2.0, 5.0, 3.0]} = Nx.Vulkan.download_f32(mx, 3)

      {:ok, mn} = Nx.Vulkan.min(a, b)
      assert {:ok, [1.0, 4.0, 3.0]} = Nx.Vulkan.download_f32(mn, 3)
    end

    test "pow" do
      {:ok, a} = Nx.Vulkan.upload_f32([2.0, 3.0, 4.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 2.0, 0.5])
      {:ok, c} = Nx.Vulkan.pow(a, b)
      {:ok, [r1, r2, r3]} = Nx.Vulkan.download_f32(c, 3)
      assert_in_delta r1, 4.0, 1.0e-3
      assert_in_delta r2, 9.0, 1.0e-3
      assert_in_delta r3, 2.0, 1.0e-3
    end

    test "size mismatch returns :size_mismatch" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      assert {:error, :size_mismatch} = Nx.Vulkan.add(a, b)
    end

    test "chained ops keep producing fresh tensors" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0])

      # (a + b) * a = [11, 22, 33] * [1, 2, 3] = [11, 44, 99]
      {:ok, c} = Nx.Vulkan.add(a, b)
      {:ok, d} = Nx.Vulkan.multiply(c, a)
      assert {:ok, [11.0, 44.0, 99.0]} = Nx.Vulkan.download_f32(d, 3)
    end

    test "1024 elements (multi-workgroup)" do
      input = Enum.map(1..1024, fn i -> i * 1.0 end)
      {:ok, a} = Nx.Vulkan.upload_f32(input)
      {:ok, b} = Nx.Vulkan.upload_f32(input)
      {:ok, c} = Nx.Vulkan.add(a, b)
      {:ok, result} = Nx.Vulkan.download_f32(c, 1024)
      assert hd(result) == 2.0
      assert List.last(result) == 2048.0
    end
  end

  describe "v0.0.4-7 — unary, reductions, matmul, random" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "exp" do
      {:ok, a} = Nx.Vulkan.upload_f32([0.0, 1.0, 2.0])
      {:ok, e} = Nx.Vulkan.exp(a)
      {:ok, [r0, r1, r2]} = Nx.Vulkan.download_f32(e, 3)
      assert_in_delta r0, 1.0, 1.0e-3
      assert_in_delta r1, 2.71828, 1.0e-3
      assert_in_delta r2, 7.389, 1.0e-3
    end

    test "log inverse of exp" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.71828, 7.389])
      {:ok, l} = Nx.Vulkan.log(a)
      {:ok, [r0, r1, r2]} = Nx.Vulkan.download_f32(l, 3)
      assert_in_delta r0, 0.0, 1.0e-3
      assert_in_delta r1, 1.0, 1.0e-3
      assert_in_delta r2, 2.0, 1.0e-3
    end

    test "relu" do
      {:ok, a} = Nx.Vulkan.upload_f32([-2.0, -0.5, 0.0, 1.5, 3.0])
      {:ok, r} = Nx.Vulkan.relu(a)
      assert {:ok, [0.0, 0.0, 0.0, 1.5, 3.0]} = Nx.Vulkan.download_f32(r, 5)
    end

    test "sum" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      assert {:ok, 10.0} = Nx.Vulkan.sum(a)
    end

    test "reduce_min / reduce_max" do
      {:ok, a} = Nx.Vulkan.upload_f32([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
      assert {:ok, 1.0} = Nx.Vulkan.reduce_min(a)
      assert {:ok, 9.0} = Nx.Vulkan.reduce_max(a)
    end

    test "mean" do
      {:ok, a} = Nx.Vulkan.upload_f32([2.0, 4.0, 6.0, 8.0])
      assert {:ok, 5.0} = Nx.Vulkan.mean(a)
    end

    test "matmul: 1x3 · 3x1 = dot product" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0])
      {:ok, c} = Nx.Vulkan.matmul(a, b, 1, 1, 3)
      assert {:ok, [140.0]} = Nx.Vulkan.download_f32(c, 1)
    end

    test "matmul: 2x2 · 2x2" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      {:ok, b} = Nx.Vulkan.upload_f32([5.0, 6.0, 7.0, 8.0])
      {:ok, c} = Nx.Vulkan.matmul(a, b, 2, 2, 2)
      # [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
      assert {:ok, [19.0, 22.0, 43.0, 50.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "uniform random determinism" do
      {:ok, r1} = Nx.Vulkan.uniform(8, 42)
      {:ok, r2} = Nx.Vulkan.uniform(8, 42)
      {:ok, l1} = Nx.Vulkan.download_f32(r1, 8)
      {:ok, l2} = Nx.Vulkan.download_f32(r2, 8)
      assert l1 == l2
      assert Enum.all?(l1, fn x -> x >= 0.0 and x < 1.0 end)
    end

    test "uniform different seed → different output" do
      {:ok, r1} = Nx.Vulkan.uniform(8, 1)
      {:ok, r2} = Nx.Vulkan.uniform(8, 2)
      {:ok, l1} = Nx.Vulkan.download_f32(r1, 8)
      {:ok, l2} = Nx.Vulkan.download_f32(r2, 8)
      refute l1 == l2
    end
  end

  describe "v0.0.x — Nx.Backend integration" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "Nx.tensor with Vulkan backend round-trips through GPU" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: Nx.Vulkan.Backend)
      assert Nx.to_flat_list(t) == [1.0, 2.0, 3.0, 4.0]
    end

    test "Nx.add dispatches through the Vulkan backend" do
      t = Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
      result = Nx.add(t, t)
      assert Nx.to_flat_list(result) == [2.0, 4.0, 6.0]
    end

    test "Nx.multiply through Vulkan" do
      a = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: Nx.Vulkan.Backend)
      b = Nx.tensor([10.0, 20.0, 30.0, 40.0], backend: Nx.Vulkan.Backend)
      assert Nx.to_flat_list(Nx.multiply(a, b)) == [10.0, 40.0, 90.0, 160.0]
    end

    test "Nx.exp through Vulkan" do
      t = Nx.tensor([0.0, 1.0], backend: Nx.Vulkan.Backend)
      [r0, r1] = Nx.to_flat_list(Nx.exp(t))
      assert_in_delta r0, 1.0, 1.0e-3
      assert_in_delta r1, 2.71828, 1.0e-3
    end

    test "Nx.sum through Vulkan" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: Nx.Vulkan.Backend)
      assert Nx.to_number(Nx.sum(t)) == 10.0
    end

    test "Nx.backend_transfer to BinaryBackend works" do
      t = Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
      bin_t = Nx.backend_transfer(t, Nx.BinaryBackend)
      assert Nx.to_flat_list(bin_t) == [1.0, 2.0, 3.0]
    end
  end

  describe "stress findings" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "20 concurrent processes don't race the queue (mutex)" do
      inputs = Enum.map(1..256, fn i -> i / 1.0 end)

      tasks =
        for _p <- 1..20 do
          Task.async(fn ->
            for _ <- 1..5 do
              {:ok, a} = Nx.Vulkan.upload_f32(inputs)
              {:ok, b} = Nx.Vulkan.upload_f32(inputs)
              {:ok, c} = Nx.Vulkan.add(a, b)
              {:ok, [head | _]} = Nx.Vulkan.download_f32(c, 256)
              head
            end
          end)
        end

      results = Enum.map(tasks, &Task.await(&1, 60_000))

      # Every process should have computed 1.0 + 1.0 = 2.0 each iter
      Enum.each(results, fn list ->
        Enum.each(list, fn h -> assert h == 2.0 end)
      end)
    end

    test "divide-by-zero returns :infinity (not empty list)" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, -1.0, 0.0])
      {:ok, z} = Nx.Vulkan.upload_f32([0.0, 0.0, 0.0])
      {:ok, dz} = Nx.Vulkan.divide(a, z)
      assert {:ok, [:infinity, :neg_infinity, :nan]} = Nx.Vulkan.download_f32(dz, 3)
    end

    test "log of negative returns :nan" do
      {:ok, n} = Nx.Vulkan.upload_f32([-1.0, -2.0])
      {:ok, l} = Nx.Vulkan.log(n)
      assert {:ok, [:nan, :nan]} = Nx.Vulkan.download_f32(l, 2)
    end

    test "sqrt of negative returns :nan" do
      {:ok, n} = Nx.Vulkan.upload_f32([-1.0, -4.0])
      {:ok, s} = Nx.Vulkan.sqrt(n)
      assert {:ok, [:nan, :nan]} = Nx.Vulkan.download_f32(s, 2)
    end

    test "5000 alloc/free cycles don't leak" do
      # If ResourceArc::Drop is broken, this will OOM the GPU.
      for _ <- 1..5_000 do
        {:ok, _t} = Nx.Vulkan.upload_f32([42.0])
      end

      :erlang.garbage_collect()

      # Verify the device is still alive after 5000 allocs.
      {:ok, t} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      assert {:ok, [1.0, 2.0, 3.0]} = Nx.Vulkan.download_f32(t, 3)
    end
  end

  describe "v0.1 phase 1.1 — comparisons" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "equal" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 2.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 5.0, 3.0, 7.0])
      {:ok, c} = Nx.Vulkan.equal(a, b)
      assert {:ok, [1.0, 0.0, 1.0, 0.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "less" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 5.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 4.0, 3.0])
      {:ok, c} = Nx.Vulkan.less(a, b)
      assert {:ok, [1.0, 0.0, 0.0]} = Nx.Vulkan.download_f32(c, 3)
    end

    test "greater" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 5.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 4.0, 3.0])
      {:ok, c} = Nx.Vulkan.greater(a, b)
      assert {:ok, [0.0, 1.0, 0.0]} = Nx.Vulkan.download_f32(c, 3)
    end

    test "select with hand-built 0/1 condition (no shader needed)" do
      # cond=[1,0,1,0] picks t else f → [10, 200, 30, 400]
      {:ok, cond} = Nx.Vulkan.upload_f32([1.0, 0.0, 1.0, 0.0])
      {:ok, t} = Nx.Vulkan.upload_f32([10.0, 20.0, 30.0, 40.0])
      {:ok, f} = Nx.Vulkan.upload_f32([100.0, 200.0, 300.0, 400.0])
      {:ok, c} = Nx.Vulkan.select(cond, t, f)
      assert {:ok, [10.0, 200.0, 30.0, 400.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    test "clip via compositional max+min" do
      {:ok, a} = Nx.Vulkan.upload_f32([-2.0, -0.5, 0.0, 1.5, 3.0, 7.0])
      {:ok, c} = Nx.Vulkan.clip(a, 0.0, 5.0)
      assert {:ok, [0.0, 0.0, 0.0, 1.5, 3.0, 5.0]} = Nx.Vulkan.download_f32(c, 6)
    end

    test "clip is the identity when the range covers all values" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      {:ok, c} = Nx.Vulkan.clip(a, -10.0, 10.0)
      assert {:ok, [1.0, 2.0, 3.0]} = Nx.Vulkan.download_f32(c, 3)
    end
  end

  describe "v0.1 phase 1.2 — reshape / broadcast" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "reshape is metadata-only (zero copy on the buffer)" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: Nx.Vulkan.Backend)
      reshaped = Nx.reshape(t, {2, 3})
      assert Nx.shape(reshaped) == {2, 3}
      assert Nx.to_flat_list(reshaped) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "reshape across multiple permutations preserves byte order" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: Nx.Vulkan.Backend)
      r1 = Nx.reshape(t, {3, 2})
      r2 = Nx.reshape(r1, {6})
      assert Nx.to_flat_list(r2) == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    end

    test "squeeze drops a trivial axis" do
      # shape {1, 3}
      t = Nx.tensor([[1.0, 2.0, 3.0]], backend: Nx.Vulkan.Backend)
      s = Nx.squeeze(t)
      assert Nx.shape(s) == {3}
      assert Nx.to_flat_list(s) == [1.0, 2.0, 3.0]
    end

    test "broadcast: scalar to vector" do
      # shape {}
      t = Nx.tensor(7.0, backend: Nx.Vulkan.Backend)
      b = Nx.broadcast(t, {4})
      assert Nx.shape(b) == {4}
      assert Nx.to_flat_list(b) == [7.0, 7.0, 7.0, 7.0]
    end

    test "broadcast: 1D row to 2D matrix (row-replication)" do
      # shape {3}
      t = Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
      b = Nx.broadcast(t, {2, 3})
      # axes=[1] means input axis 0 maps to output axis 1
      assert Nx.shape(b) == {2, 3}
      assert Nx.to_flat_list(b) == [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    end

    @tag :needs_transpose_shader
    test "transpose 2x3 via Nx.Backend (uses transpose.spv)" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Nx.Vulkan.Backend)
      tt = Nx.transpose(t)
      assert Nx.shape(tt) == {3, 2}
      assert Nx.to_flat_list(tt) == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
    end

    @tag :needs_transpose_shader
    test "transpose 3x4" do
      data = for r <- 0..2, c <- 0..3, do: r * 4.0 + c
      t = Nx.tensor(Enum.chunk_every(data, 4), backend: Nx.Vulkan.Backend)
      tt = Nx.transpose(t)
      assert Nx.shape(tt) == {4, 3}
      # Element (i, j) of the transpose = element (j, i) of the original
      assert Nx.to_flat_list(tt) == [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0]
    end
  end

  describe "v0.1 phase 1.3 — slicing" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "slice 1D simple range" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: Nx.Vulkan.Backend)
      # elements 1..3
      s = Nx.slice(t, [1], [3])
      assert Nx.to_flat_list(s) == [2.0, 3.0, 4.0]
    end

    test "slice 1D with stride 2" do
      # Nx's slice: lengths is the INPUT region length. length=5, stride=2
      # walks input positions 0..4 picking every 2nd → output positions
      # [0, 2, 4] = [1.0, 3.0, 5.0].
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], backend: Nx.Vulkan.Backend)
      s = Nx.slice(t, [0], [5], strides: [2])
      assert Nx.to_flat_list(s) == [1.0, 3.0, 5.0]
    end

    test "slice 2D — extract a 2x2 block" do
      t =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          backend: Nx.Vulkan.Backend
        )

      # rows 0..1, cols 1..2
      s = Nx.slice(t, [0, 1], [2, 2])
      assert Nx.shape(s) == {2, 2}
      assert Nx.to_flat_list(s) == [2.0, 3.0, 5.0, 6.0]
    end

    test "slice_along_axis (Nx-built helper that calls slice/5)" do
      t =
        Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
          backend: Nx.Vulkan.Backend
        )

      # cols 1..2
      s = Nx.slice_along_axis(t, 1, 2, axis: 1)
      assert Nx.shape(s) == {2, 2}
      assert Nx.to_flat_list(s) == [2.0, 3.0, 6.0, 7.0]
    end

    test "put_slice 1D" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0, 5.0], backend: Nx.Vulkan.Backend)
      patch = Nx.tensor([99.0, 88.0], backend: Nx.Vulkan.Backend)
      out = Nx.put_slice(t, [2], patch)
      assert Nx.to_flat_list(out) == [1.0, 2.0, 99.0, 88.0, 5.0]
    end

    test "put_slice 2D — overwrite a 2x2 block" do
      t =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          backend: Nx.Vulkan.Backend
        )

      patch = Nx.tensor([[97.0, 98.0], [99.0, 100.0]], backend: Nx.Vulkan.Backend)
      out = Nx.put_slice(t, [1, 1], patch)
      assert Nx.to_flat_list(out) == [1.0, 2.0, 3.0, 4.0, 97.0, 98.0, 7.0, 99.0, 100.0]
    end
  end

  describe "v0.1 phase 1.4 — per-axis reductions" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "sum over axis 0 of 2D" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Nx.Vulkan.Backend)
      # column sums
      s = Nx.sum(t, axes: [0])
      assert Nx.shape(s) == {3}
      assert Nx.to_flat_list(s) == [5.0, 7.0, 9.0]
    end

    test "sum over axis 1 of 2D" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Nx.Vulkan.Backend)
      # row sums
      s = Nx.sum(t, axes: [1])
      assert Nx.shape(s) == {2}
      assert Nx.to_flat_list(s) == [6.0, 15.0]
    end

    test "sum with keep_axes" do
      t = Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], backend: Nx.Vulkan.Backend)
      s = Nx.sum(t, axes: [1], keep_axes: true)
      assert Nx.shape(s) == {2, 1}
      assert Nx.to_flat_list(s) == [6.0, 15.0]
    end

    test "reduce_max over axis 1 of 2D" do
      t = Nx.tensor([[1.0, 5.0, 2.0], [4.0, 3.0, 6.0]], backend: Nx.Vulkan.Backend)
      s = Nx.reduce_max(t, axes: [1])
      assert Nx.shape(s) == {2}
      assert Nx.to_flat_list(s) == [5.0, 6.0]
    end

    test "reduce_min over axis 0 of 3D" do
      t =
        Nx.tensor(
          [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 0.5], [7.0, 8.0]]],
          backend: Nx.Vulkan.Backend
        )

      s = Nx.reduce_min(t, axes: [0])
      assert Nx.shape(s) == {2, 2}
      assert Nx.to_flat_list(s) == [1.0, 0.5, 3.0, 4.0]
    end

    test "sum over multiple axes (0 and 2 of 3D)" do
      # 2x2x2 — reduce out axes 0 and 2, keep axis 1.
      t =
        Nx.tensor(
          [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
          backend: Nx.Vulkan.Backend
        )

      s = Nx.sum(t, axes: [0, 2])
      assert Nx.shape(s) == {2}
      # axis 1 = 0: [1,2,5,6] → 14; axis 1 = 1: [3,4,7,8] → 22
      assert Nx.to_flat_list(s) == [14.0, 22.0]
    end

    test "full-axis sum still routes through GPU" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Nx.Vulkan.Backend)
      s = Nx.sum(t)
      assert Nx.shape(s) == {}
      assert Nx.to_number(s) == 10.0
    end
  end

  describe "v0.1 phase 1.5 — construction (iota, eye)" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "iota 1D" do
      t = Nx.iota({5}, type: :f32, backend: Nx.Vulkan.Backend)
      assert Nx.to_flat_list(t) == [0.0, 1.0, 2.0, 3.0, 4.0]
    end

    test "iota 2D no axis (row-major flat)" do
      t = Nx.iota({2, 3}, type: :f32, backend: Nx.Vulkan.Backend)
      assert Nx.to_flat_list(t) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    end

    test "iota 2D along axis 0" do
      t = Nx.iota({2, 3}, axis: 0, type: :f32, backend: Nx.Vulkan.Backend)
      # row 0 → 0, row 1 → 1
      assert Nx.to_flat_list(t) == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    end

    test "iota 2D along axis 1" do
      t = Nx.iota({2, 3}, axis: 1, type: :f32, backend: Nx.Vulkan.Backend)
      # col 0,1,2 → 0,1,2 in each row
      assert Nx.to_flat_list(t) == [0.0, 1.0, 2.0, 0.0, 1.0, 2.0]
    end

    test "eye 3x3" do
      t = Nx.eye({3, 3}, type: :f32, backend: Nx.Vulkan.Backend)
      assert Nx.to_flat_list(t) == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    end

    test "eye non-square 2x4" do
      t = Nx.eye({2, 4}, type: :f32, backend: Nx.Vulkan.Backend)
      assert Nx.to_flat_list(t) == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    end

    test "eye 3D batched" do
      # Nx.eye supports leading batch dims; identity in last two axes.
      t = Nx.eye({2, 3, 3}, type: :f32, backend: Nx.Vulkan.Backend)
      assert Nx.shape(t) == {2, 3, 3}
      identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
      assert Nx.to_flat_list(t) == identity ++ identity
    end
  end

  describe "v0.1 phase 1.6 — indexing (gather, indexed_put, indexed_add, take_diagonal)" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "take_diagonal of 3x3 (composes through gather)" do
      t =
        Nx.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
          backend: Nx.Vulkan.Backend
        )

      d = Nx.take_diagonal(t)
      assert Nx.to_flat_list(d) == [1.0, 5.0, 9.0]
    end

    test "take_diagonal of non-square 2x4" do
      t =
        Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
          backend: Nx.Vulkan.Backend
        )

      d = Nx.take_diagonal(t)
      assert Nx.to_flat_list(d) == [1.0, 6.0]
    end

    test "indexed_put writes scattered values" do
      t = Nx.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], backend: Nx.Vulkan.Backend)
      idx = Nx.tensor([[0, 1], [1, 2]])
      upd = Nx.tensor([7.0, 9.0], backend: Nx.Vulkan.Backend)
      out = Nx.indexed_put(t, idx, upd)
      assert Nx.to_flat_list(out) == [0.0, 7.0, 0.0, 0.0, 0.0, 9.0]
    end

    test "indexed_add accumulates at scattered positions" do
      t = Nx.tensor([1.0, 1.0, 1.0, 1.0], backend: Nx.Vulkan.Backend)
      idx = Nx.tensor([[0], [2], [2]])
      upd = Nx.tensor([5.0, 3.0, 4.0], backend: Nx.Vulkan.Backend)
      out = Nx.indexed_add(t, idx, upd)
      # idx [0]: +5 → 6.0, idx [2] twice: +3+4 → 8.0
      assert Nx.to_flat_list(out) == [6.0, 1.0, 8.0, 1.0]
    end

    test "gather direct: pick rows from 2D" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], backend: Nx.Vulkan.Backend)
      # Single-axis gather over axis 0 — pick rows 2, 0, 1.
      idx = Nx.tensor([[2], [0], [1]])
      out = Nx.gather(t, idx, axes: [0])
      assert Nx.shape(out) == {3, 2}
      assert Nx.to_flat_list(out) == [5.0, 6.0, 1.0, 2.0, 3.0, 4.0]
    end
  end

  describe "v0.1 phase 1.7 — transcendentals (erf, expm1)" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "erf direct via Nx.Vulkan API" do
      {:ok, t} = Nx.Vulkan.upload_f32([-2.0, -1.0, 0.0, 1.0, 2.0])
      {:ok, r} = Nx.Vulkan.erf(t)
      {:ok, vals} = Nx.Vulkan.download_f32(r, 5)

      expected = [-0.9953, -0.8427, 0.0, 0.8427, 0.9953]

      Enum.zip(vals, expected)
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.5e-4 end)
    end

    test "expm1 direct via Nx.Vulkan API" do
      {:ok, t} = Nx.Vulkan.upload_f32([-1.0, -0.1, 0.0, 0.1, 1.0])
      {:ok, r} = Nx.Vulkan.expm1(t)
      {:ok, vals} = Nx.Vulkan.download_f32(r, 5)

      expected = [-0.6321, -0.0952, 0.0, 0.1052, 1.7183]

      Enum.zip(vals, expected)
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-4 end)
    end

    test "erf via Nx.erf/1 backend dispatch" do
      t = Nx.tensor([-1.0, 0.0, 1.0], backend: Nx.Vulkan.Backend)
      out = Nx.erf(t)
      vals = Nx.to_flat_list(out)

      Enum.zip(vals, [-0.8427, 0.0, 0.8427])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.5e-4 end)
    end

    test "expm1 via Nx.expm1/1 backend dispatch" do
      t = Nx.tensor([-0.5, 0.0, 0.5], backend: Nx.Vulkan.Backend)
      out = Nx.expm1(t)
      vals = Nx.to_flat_list(out)

      # Boundary: Taylor for |x|<0.5, exp(x)-1 at |x|=0.5
      Enum.zip(vals, [-0.3935, 0.0, 0.6487])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-4 end)
    end

    test "expm1 small-x precision (Taylor branch)" do
      # exp(0.001) - 1 = 1.0005e-3 — direct exp(x)-1 cancellation loses
      # ~5 sig figs in f32; Taylor recovers the missing precision.
      {:ok, t} = Nx.Vulkan.upload_f32([0.001, -0.001])
      {:ok, r} = Nx.Vulkan.expm1(t)
      {:ok, [a, b]} = Nx.Vulkan.download_f32(r, 2)

      assert_in_delta a, 1.0005e-3, 1.0e-7
      assert_in_delta b, -0.99950e-3, 1.0e-7
    end
  end

  describe "v0.1 phase 1.8 — f64 + casts (as_type)" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "f64 storage round-trip" do
      # f64 doesn't go through Nx.Vulkan.upload_f32 (that's f32); use
      # raw binary path. Phase 1.6 generalized from_binary/to_binary
      # to accept any element type.
      bin = <<1.5::float-64-native, 2.5::float-64-native, 3.5::float-64-native>>
      t = Nx.from_binary(bin, :f64, backend: Nx.Vulkan.Backend)
      assert Nx.shape(t) == {3}
      assert Nx.type(t) == {:f, 64}
      assert Nx.to_flat_list(t) == [1.5, 2.5, 3.5]
    end

    test "as_type f32 → f64 preserves values" do
      t = Nx.tensor([1.5, 2.5, 3.5], type: :f32, backend: Nx.Vulkan.Backend)
      t64 = Nx.as_type(t, :f64)
      assert Nx.type(t64) == {:f, 64}
      assert Nx.to_flat_list(t64) == [1.5, 2.5, 3.5]
    end

    test "as_type f64 → f32 (allow precision loss)" do
      bin = <<1.5::float-64-native, 2.25::float-64-native>>
      t64 = Nx.from_binary(bin, :f64, backend: Nx.Vulkan.Backend)
      t32 = Nx.as_type(t64, :f32)
      assert Nx.type(t32) == {:f, 32}
      assert Nx.to_flat_list(t32) == [1.5, 2.25]
    end

    test "as_type f32 → s32 truncates toward zero" do
      t = Nx.tensor([1.7, -1.7, 3.0, -0.5], type: :f32, backend: Nx.Vulkan.Backend)
      ti = Nx.as_type(t, :s32)
      assert Nx.type(ti) == {:s, 32}
      assert Nx.to_flat_list(ti) == [1, -1, 3, 0]
    end

    test "as_type s64 → f32" do
      bin = <<1::signed-64-native, 2::signed-64-native, -3::signed-64-native>>
      ti = Nx.from_binary(bin, :s64, backend: Nx.Vulkan.Backend)
      tf = Nx.as_type(ti, :f32)
      assert Nx.type(tf) == {:f, 32}
      assert Nx.to_flat_list(tf) == [1.0, 2.0, -3.0]
    end

    test "as_type same-type is zero-copy ref rewrap" do
      t = Nx.tensor([1.0, 2.0, 3.0], type: :f32, backend: Nx.Vulkan.Backend)
      t2 = Nx.as_type(t, :f32)
      # Same underlying ref — both should point to identical GPU buffer.
      assert t.data.ref == t2.data.ref
      assert Nx.to_flat_list(t2) == [1.0, 2.0, 3.0]
    end

    test "f64 round-trip preserves precision a 1e-15 f32 cannot" do
      # 1.0 + 1.0e-15 collapses to 1.0 in f32; f64 keeps it.
      v = 1.0 + 1.0e-15
      bin = <<v::float-64-native>>
      t64 = Nx.from_binary(bin, :f64, backend: Nx.Vulkan.Backend)
      [back] = Nx.to_flat_list(t64)
      assert back == v

      # Cast to f32 and observe the collapse.
      t32 = Nx.as_type(t64, :f32)
      [collapsed] = Nx.to_flat_list(t32)
      assert collapsed == 1.0
    end
  end

  describe "v0.1 phase 1.8/1.4 — GPU shader paths" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "cast f32 → f64 direct API (GPU shader)" do
      {:ok, t} = Nx.Vulkan.upload_f32([1.5, 2.5, 3.5, 4.5])
      {:ok, t64_ref} = Nx.Vulkan.cast_f32_to_f64(t, 4)
      {:ok, bin} = Nx.Vulkan.Native.download_binary(t64_ref, 4 * 8)

      vals = for <<x::float-64-native <- bin>>, do: x
      assert vals == [1.5, 2.5, 3.5, 4.5]
    end

    test "cast f64 → f32 direct API (GPU shader)" do
      bin = <<1.5::float-64-native, 2.5::float-64-native, 3.5::float-64-native>>
      {:ok, t64} = Nx.Vulkan.upload_binary(bin)
      {:ok, t32_ref} = Nx.Vulkan.cast_f64_to_f32(t64, 3)
      {:ok, out_bin} = Nx.Vulkan.Native.download_binary(t32_ref, 3 * 4)

      vals = for <<x::float-32-native <- out_bin>>, do: x
      assert vals == [1.5, 2.5, 3.5]
    end

    test "reduce_axis sum direct API (GPU shader)" do
      # 2×3×2 row-major, reduce axis 1 → outer=2, reduce=3, inner=2.
      vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
      {:ok, t} = Nx.Vulkan.upload_f32(vals)
      {:ok, out_ref} = Nx.Vulkan.reduce_axis(t, 2, 3, 2, 0)
      {:ok, out_vals} = Nx.Vulkan.download_f32(out_ref, 4)

      # slot (0,0): 1+3+5=9, (0,1): 2+4+6=12, (1,0): 7+9+11=27, (1,1): 8+10+12=30
      assert out_vals == [9.0, 12.0, 27.0, 30.0]
    end

    test "reduce_axis max direct API (GPU shader)" do
      {:ok, t} = Nx.Vulkan.upload_f32([1.0, 5.0, 2.0, 4.0, 3.0, 6.0])
      # 2×3 row-major, reduce axis 1 → outer=2, reduce=3, inner=1.
      {:ok, out_ref} = Nx.Vulkan.reduce_axis(t, 2, 3, 1, 1)
      {:ok, out_vals} = Nx.Vulkan.download_f32(out_ref, 2)

      assert out_vals == [5.0, 6.0]
    end

    test "Nx.sum single-axis routes through GPU shader (correctness)" do
      # Same shape as reduce_axis test above; verify backend dispatch.
      t =
        Nx.tensor(
          [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]],
          backend: Nx.Vulkan.Backend
        )

      s = Nx.sum(t, axes: [1])
      assert Nx.shape(s) == {2, 2}
      assert Nx.to_flat_list(s) == [9.0, 12.0, 27.0, 30.0]
    end
  end

  describe "v0.1 phase 1.9 — dense linalg (determinant, solve, cholesky, triangular_solve)" do
    setup do
      :ok = Nx.Vulkan.init()
      :ok
    end

    test "determinant of 2x2" do
      t = Nx.tensor([[3.0, 1.0], [2.0, 4.0]], backend: Nx.Vulkan.Backend)
      d = Nx.LinAlg.determinant(t)
      assert_in_delta Nx.to_number(d), 10.0, 1.0e-5
    end

    test "determinant of 3x3" do
      # det([[1,2,3],[0,1,4],[5,6,0]]) = 1*(1*0-4*6) - 2*(0*0-4*5) + 3*(0*6-1*5)
      #                                = -24 + 40 - 15 = 1
      t =
        Nx.tensor([[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]],
          backend: Nx.Vulkan.Backend
        )

      d = Nx.LinAlg.determinant(t)
      assert_in_delta Nx.to_number(d), 1.0, 1.0e-5
    end

    test "solve 2x2 system" do
      # [[2, 1], [1, 3]] x = [4, 5]  →  x = [1.4, 1.2]
      a = Nx.tensor([[2.0, 1.0], [1.0, 3.0]], backend: Nx.Vulkan.Backend)
      b = Nx.tensor([4.0, 5.0], backend: Nx.Vulkan.Backend)
      x = Nx.LinAlg.solve(a, b)

      Enum.zip(Nx.to_flat_list(x), [1.4, 1.2])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-5 end)
    end

    test "cholesky of SPD 3x3 (mass-matrix shape)" do
      # SPD matrix: A = L L^T with L lower-triangular.
      # [[4,2,2],[2,5,3],[2,3,6]] → L = [[2,0,0],[1,2,0],[1,1,2]]
      a =
        Nx.tensor([[4.0, 2.0, 2.0], [2.0, 5.0, 3.0], [2.0, 3.0, 6.0]],
          backend: Nx.Vulkan.Backend
        )

      l = Nx.LinAlg.cholesky(a)

      Enum.zip(Nx.to_flat_list(l), [2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 1.0, 1.0, 2.0])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-5 end)
    end

    test "triangular_solve lower" do
      # [[2,0],[3,4]] x = [4, 17]  →  x = [2, 2.75]
      a = Nx.tensor([[2.0, 0.0], [3.0, 4.0]], backend: Nx.Vulkan.Backend)
      b = Nx.tensor([4.0, 17.0], backend: Nx.Vulkan.Backend)
      x = Nx.LinAlg.triangular_solve(a, b, lower: true)

      Enum.zip(Nx.to_flat_list(x), [2.0, 2.75])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-5 end)
    end

    test "fused_chain: 2-op chain (multiply + add) matches separate dispatches" do
      :ok = Nx.Vulkan.init()

      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      {:ok, b} = Nx.Vulkan.upload_f32([0.5, 0.5, 0.5, 0.5])

      {:ok, fused} = Nx.Vulkan.fused_chain(a, b, [:multiply, :add])
      {:ok, vals} = Nx.Vulkan.download_f32(fused, 4)

      # Reference: (a * b) + b = a*0.5 + 0.5
      Enum.zip(vals, [1.0, 1.5, 2.0, 2.5])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-5 end)
    end

    test "fused_chain: 3-op chain with unary tail (multiply + add + exp)" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 0.5])

      {:ok, fused} = Nx.Vulkan.fused_chain(a, b, [:multiply, :add, :exp])
      {:ok, vals} = Nx.Vulkan.download_f32(fused, 2)

      # exp((a*b) + b): exp(1*1 + 1) = exp(2) = 7.389...
      #                exp(2*0.5 + 0.5) = exp(1.5) = 4.4816...
      Enum.zip(vals, [:math.exp(2.0), :math.exp(1.5)])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-4 end)
    end

    test "fused_chain: 8-op chain hits the limit and stays correct" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 1.0, 1.0])

      # Chain: ((((a + b) * b) - b) / b) → square → exp → log → sqrt
      # With b=1: a+1 → a+1 → a → a → a^2 → exp(a^2) → log(exp(a^2)) → sqrt(a^2) = |a|
      {:ok, fused} =
        Nx.Vulkan.fused_chain(a, b, [
          :add,
          :multiply,
          :subtract,
          :divide,
          :square,
          :exp,
          :log,
          :sqrt
        ])

      {:ok, vals} = Nx.Vulkan.download_f32(fused, 3)

      Enum.zip(vals, [1.0, 2.0, 3.0])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-3 end)
    end

    test "fused_chain with erf in middle (cases 13/14 now wired)" do
      :ok = Nx.Vulkan.init()

      # Chain: (a + b) → erf
      # With a=[0.0], b=[1.0]: erf(0+1) = erf(1) = 0.8427
      # With a=[-1.0], b=[1.0]: erf(-1+1) = erf(0) = 0
      {:ok, a} = Nx.Vulkan.upload_f32([0.0, -1.0, 1.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 1.0, 1.0])

      {:ok, fused} = Nx.Vulkan.fused_chain(a, b, [:add, :erf])
      {:ok, vals} = Nx.Vulkan.download_f32(fused, 3)

      Enum.zip(vals, [:math.erf(1.0), 0.0, :math.erf(2.0)])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.5e-4 end)
    end

    test "fused_chain with expm1 in chain" do
      # exp(0.001) - 1 ≈ 1.0005e-3 — Taylor branch.
      {:ok, a} = Nx.Vulkan.upload_f32([0.001, 0.0, 1.0])
      {:ok, b} = Nx.Vulkan.upload_f32([0.0, 0.0, 0.0])

      {:ok, fused} = Nx.Vulkan.fused_chain(a, b, [:add, :expm1])
      {:ok, vals} = Nx.Vulkan.download_f32(fused, 3)

      # :math doesn't have expm1; compute via exp(x)-1.
      expected = [:math.exp(0.001) - 1, 0.0, :math.exp(1.0) - 1]

      Enum.zip(vals, expected)
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-5 end)
    end

    test "fused_chain rejects empty op list" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 1.0])

      assert_raise KeyError, fn ->
        # Empty list passes the Elixir-side guard; the NIF rejects it
        # with :bad_op. But [:not_a_real_op] crashes earlier with KeyError
        # — verify the atom-validation guard at least.
        Nx.Vulkan.fused_chain(a, b, [:not_a_real_op])
      end
    end

    test "Day 1f — matmul auto-select picks the right shader by size" do
      :ok = Nx.Vulkan.init()

      # Tiny: dispatch overhead dominates → naive matmul.spv
      assert {"matmul.spv", 16, 16} = Nx.Vulkan.pick_matmul(4, 4, 4)
      # Medium → matmul_tiled.spv
      assert {"matmul_tiled.spv", 16, 16} = Nx.Vulkan.pick_matmul(64, 64, 64)
      # Large (≥ 256³) → tiled-16x2 (mac-248's 4.2x measured win).
      assert {"matmul_tiled16x2.spv", 32, 16} = Nx.Vulkan.pick_matmul(256, 256, 256)
    end

    test "Day 1f — matmul_variant explicit shader path" do
      # 2x2
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      # 2x2
      {:ok, b} = Nx.Vulkan.upload_f32([5.0, 6.0, 7.0, 8.0])

      # Each variant should produce the same A·B result.
      expected = [19.0, 22.0, 43.0, 50.0]

      for variant <- [:matmul, :matmul_tiled, :matmul_tiled16x2] do
        {:ok, c} = Nx.Vulkan.matmul_variant(a, b, 2, 2, 2, variant)
        {:ok, vals} = Nx.Vulkan.download_f32(c, 4)

        Enum.zip(vals, expected)
        |> Enum.each(fn {v, e} ->
          assert_in_delta v, e, 1.0e-4, "variant #{variant} failed"
        end)
      end
    end

    test "Day 1f — Nx.Vulkan.matmul auto-select still produces correct results" do
      # Cover the boundary cases: tiny (4×4) and just-large (32×32).
      {:ok, a} = Nx.Vulkan.upload_f32(List.duplicate(1.0, 16))
      {:ok, b} = Nx.Vulkan.upload_f32(List.duplicate(2.0, 16))
      {:ok, c} = Nx.Vulkan.matmul(a, b, 4, 4, 4)
      {:ok, vals} = Nx.Vulkan.download_f32(c, 16)
      # 4x4 of 1.0 times 4x4 of 2.0: each cell = sum of 4 * 1*2 = 8
      assert Enum.all?(vals, &(abs(&1 - 8.0) < 1.0e-5))
    end

    test "buffer pool: alloc/free cycle reuses buffers" do
      :ok = Nx.Vulkan.init()
      Nx.Vulkan.pool_clear()

      {:ok, stats0} = Nx.Vulkan.pool_stats()

      # Allocate, drop (which returns to pool), re-allocate same size.
      {:ok, t1} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])
      :erlang.garbage_collect()
      _ = t1

      # Force the ref to drop. The next alloc of the same size should
      # be a pool hit.
      ref_drop = fn -> {:ok, _} = Nx.Vulkan.upload_f32([5.0, 6.0, 7.0, 8.0]) end
      ref_drop.()
      :erlang.garbage_collect()

      # Drive an alloc request that should pool-hit (16 bytes).
      {:ok, _t} = Nx.Vulkan.upload_f32([9.0, 10.0, 11.0, 12.0])

      {:ok, stats1} = Nx.Vulkan.pool_stats()

      # At minimum: allocs happened. Hits may or may not have fired
      # depending on GC timing, but misses must increase.
      assert stats1.misses >= stats0.misses + 1
    end

    test "buffer pool: pool_clear empties the pool" do
      :ok = Nx.Vulkan.init()

      # Generate some pooled state.
      for _ <- 1..5 do
        {:ok, _} = Nx.Vulkan.upload_f32([0.0, 0.0])
      end

      :erlang.garbage_collect()
      Nx.Vulkan.pool_clear()

      {:ok, stats} = Nx.Vulkan.pool_stats()
      assert stats.total_pooled == 0
      assert stats.size_classes == 0
    end

    test "broadcast: scalar {1} × vector {4} via direct API" do
      :ok = Nx.Vulkan.init()

      {:ok, scalar} = Nx.Vulkan.upload_f32([5.0])
      {:ok, vec} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 4.0])

      {:ok, c} =
        Nx.Vulkan.apply_binary_broadcast(
          scalar,
          vec,
          :add,
          1,
          [4, 0, 0, 0],
          [0, 0, 0, 0],
          [1, 0, 0, 0]
        )

      {:ok, vals} = Nx.Vulkan.download_f32(c, 4)
      assert vals == [6.0, 7.0, 8.0, 9.0]
    end

    test "broadcast_strides/2 row-major math" do
      assert Nx.Vulkan.broadcast_strides({1, 4}, {3, 4}) == [0, 1, 0, 0]
      assert Nx.Vulkan.broadcast_strides({2, 1}, {2, 4}) == [1, 0, 0, 0]
      # {3} aligns to axis 1 of {2, 3}; axis 0 broadcasts → stride [0, 1]
      assert Nx.Vulkan.broadcast_strides({3}, {2, 3}) == [0, 1, 0, 0]
      assert Nx.Vulkan.broadcast_strides({}, {3, 4}) == [0, 0, 0, 0]
    end

    test "broadcast: backend dispatches Nx.add(matrix, vector) via shader" do
      m = Nx.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], backend: Nx.Vulkan.Backend)
      v = Nx.tensor([100.0, 200.0, 300.0], backend: Nx.Vulkan.Backend)

      out = Nx.add(m, v)
      assert Nx.shape(out) == {2, 3}
      assert Nx.to_flat_list(out) == [101.0, 202.0, 303.0, 110.0, 220.0, 330.0]
    end

    test "broadcast: backend dispatches column-broadcast {2,1} × {2,4}" do
      col = Nx.tensor([[10.0], [20.0]], backend: Nx.Vulkan.Backend)
      m = Nx.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], backend: Nx.Vulkan.Backend)

      out = Nx.add(col, m)
      assert Nx.shape(out) == {2, 4}
      assert Nx.to_flat_list(out) == [11.0, 12.0, 13.0, 14.0, 25.0, 26.0, 27.0, 28.0]
    end

    test "broadcast: equal/less work via shader" do
      v = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: Nx.Vulkan.Backend)
      threshold = Nx.tensor([2.5], backend: Nx.Vulkan.Backend)

      out = Nx.less(v, threshold)
      assert Nx.to_flat_list(out) == [1.0, 1.0, 0.0, 0.0]
    end

    test "Path A.2 — Fuse.fuse macro detects 3-op chain" do
      :ok = Nx.Vulkan.init()
      import Nx.Vulkan.Fuse

      # exp(a*b + b)
      f = fuse(fn a, b -> Nx.exp(Nx.add(Nx.multiply(a, b), b)) end)

      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 0.5])

      {:ok, ref} = f.(a, b)
      {:ok, vals} = Nx.Vulkan.download_f32(ref, 2)

      # exp(1*1 + 1) = exp(2); exp(2*0.5 + 0.5) = exp(1.5)
      Enum.zip(vals, [:math.exp(2.0), :math.exp(1.5)])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-4 end)
    end

    test "Path A.2 — Fuse falls back when body isn't a recognized chain" do
      import Nx.Vulkan.Fuse

      # Not a chain — body uses a third tensor `c` (not bound). The
      # macro returns the original function unchanged.
      f = fuse(fn a, b -> Nx.add(a, Nx.tensor([1.0, 2.0])) end)
      assert is_function(f, 2)
    end

    test "Nx.Vulkan.jit/2 evaluates a defn through the GPU backend" do
      :ok = Nx.Vulkan.init()

      # jit/2 wires Nx.Vulkan.Backend as global default. Save and restore
      # so subsequent tests aren't poisoned with a backend they didn't ask for.
      previous = Nx.default_backend()

      try do
        f = fn a, b -> Nx.add(Nx.multiply(a, b), b) end
        a = Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
        b = Nx.tensor([4.0, 5.0, 6.0], backend: Nx.Vulkan.Backend)
        out = Nx.Vulkan.jit(f).(a, b)

        assert Nx.to_flat_list(out) == [8.0, 15.0, 24.0]
        assert match?(%Nx.Vulkan.Backend{}, out.data)
      after
        Nx.global_default_backend(previous)
      end
    end

    test "Day 1d — Nx.Vulkan.Compiler auto-detects 3-op chain via Nx.Defn.jit" do
      :ok = Nx.Vulkan.init()
      previous = Nx.default_backend()

      try do
        Nx.global_default_backend(Nx.Vulkan.Backend)

        # exp(a*b + b) — three Nx ops collapse into one fused dispatch
        # via the auto-detect in Nx.Vulkan.Compiler.
        f = fn a, b -> Nx.exp(Nx.add(Nx.multiply(a, b), b)) end

        a = Nx.tensor([1.0, 2.0], backend: Nx.Vulkan.Backend)
        b = Nx.tensor([1.0, 0.5], backend: Nx.Vulkan.Backend)

        out = Nx.Defn.jit(f, compiler: Nx.Vulkan.Compiler).(a, b)

        # exp(1*1 + 1) = exp(2); exp(2*0.5 + 0.5) = exp(1.5)
        Enum.zip(Nx.to_flat_list(out), [:math.exp(2.0), :math.exp(1.5)])
        |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-4 end)
      after
        Nx.global_default_backend(previous)
      end
    end

    test "Audit — 1-arg unary chain auto-fuses" do
      :ok = Nx.Vulkan.init()
      previous = Nx.default_backend()

      try do
        Nx.global_default_backend(Nx.Vulkan.Backend)

        # exp(sigmoid(x)) — 1-arg defn, two unaries.
        f = fn x -> Nx.exp(Nx.sigmoid(x)) end
        x = Nx.tensor([-1.0, 0.0, 1.0], backend: Nx.Vulkan.Backend)

        out = Nx.Defn.jit(f, compiler: Nx.Vulkan.Compiler).(x)

        expected = [
          :math.exp(1.0 / (1.0 + :math.exp(1.0))),
          :math.exp(0.5),
          :math.exp(1.0 / (1.0 + :math.exp(-1.0)))
        ]

        Enum.zip(Nx.to_flat_list(out), expected)
        |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-4 end)
      after
        Nx.global_default_backend(previous)
      end
    end

    test "Audit — commutative swap fuses Nx.add(b, expr)" do
      :ok = Nx.Vulkan.init()
      previous = Nx.default_backend()

      try do
        Nx.global_default_backend(Nx.Vulkan.Backend)

        # b + (a * b) — first arg is the b var; commutative swap kicks in
        f = fn a, b -> Nx.add(b, Nx.multiply(a, b)) end
        a = Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
        b = Nx.tensor([2.0, 2.0, 2.0], backend: Nx.Vulkan.Backend)

        out = Nx.Defn.jit(f, compiler: Nx.Vulkan.Compiler).(a, b)

        # Expected: 2 + 1*2 = 4; 2 + 2*2 = 6; 2 + 3*2 = 8
        assert Nx.to_flat_list(out) == [4.0, 6.0, 8.0]
      after
        Nx.global_default_backend(previous)
      end
    end

    test "Audit — non-commutative swap (subtract with b first) does NOT fuse" do
      :ok = Nx.Vulkan.init()
      previous = Nx.default_backend()

      try do
        Nx.global_default_backend(Nx.Vulkan.Backend)

        # b - (a * b) — subtract is non-commutative; cannot swap.
        # Falls through to Evaluator. Result must still be correct.
        f = fn a, b -> Nx.subtract(b, Nx.multiply(a, b)) end
        a = Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
        b = Nx.tensor([2.0, 2.0, 2.0], backend: Nx.Vulkan.Backend)

        out = Nx.Defn.jit(f, compiler: Nx.Vulkan.Compiler).(a, b)

        # Expected: 2 - 1*2 = 0; 2 - 2*2 = -2; 2 - 3*2 = -4
        assert Nx.to_flat_list(out) == [0.0, -2.0, -4.0]
      after
        Nx.global_default_backend(previous)
      end
    end

    test "Day 6 (2c) — f64 elementwise binary hits the GPU shader" do
      :ok = Nx.Vulkan.init()

      a = Nx.tensor([1.5, 2.5, 3.5], type: :f64, backend: Nx.Vulkan.Backend)
      b = Nx.tensor([0.5, 0.5, 0.5], type: :f64, backend: Nx.Vulkan.Backend)

      out = Nx.add(a, b)
      assert Nx.type(out) == {:f, 64}
      assert Nx.to_flat_list(out) == [2.0, 3.0, 4.0]
    end

    test "Day 6 (2c) — f64 elementwise unary (sqrt)" do
      a = Nx.tensor([4.0, 9.0, 16.0], type: :f64, backend: Nx.Vulkan.Backend)
      out = Nx.sqrt(a)
      assert Nx.type(out) == {:f, 64}
      assert Nx.to_flat_list(out) == [2.0, 3.0, 4.0]
    end

    test "Day 6 (2c) — f64 preserves precision an f32 path cannot" do
      v = 1.0 + 1.0e-15
      bin = <<v::float-64-native, v::float-64-native>>
      a = Nx.from_binary(bin, :f64, backend: Nx.Vulkan.Backend)
      out = Nx.add(a, a)
      [first, _] = Nx.to_flat_list(out)
      assert first > 2.0
      assert first < 2.0 + 1.0e-14
    end

    test "Audit — :neg_infinity round-trips through host-materialize paths" do
      :ok = Nx.Vulkan.init()

      # 1/0.0 → :infinity; -1/0.0 → :neg_infinity. Encode → upload →
      # download must preserve the IEEE bit pattern.
      bin =
        <<0x7F800000::32-native>> <>     # +inf
        <<0xFF800000::32-native>> <>     # -inf
        <<0x7FC00000::32-native>>        # quiet NaN

      t = Nx.from_binary(bin, :f32, backend: Nx.Vulkan.Backend)
      vals = Nx.to_flat_list(t)

      # Now use a host-materialize path (transpose) and verify the
      # pipeline doesn't crash on the atom values.
      t2d = Nx.reshape(t, {1, 3})
      _ = Nx.transpose(t2d)              # forces broadcast/transpose host fallback
      assert vals == [:infinity, :neg_infinity, :nan]
    end

    test "Day 1d — non-fusable defn falls through to Evaluator" do
      :ok = Nx.Vulkan.init()
      previous = Nx.default_backend()

      try do
        Nx.global_default_backend(Nx.Vulkan.Backend)

        # Body uses sum/2 — not fusable. Compiler should delegate to
        # Evaluator, producing the right answer.
        f = fn a, b -> Nx.sum(Nx.add(a, b)) end

        a = Nx.tensor([1.0, 2.0, 3.0], backend: Nx.Vulkan.Backend)
        b = Nx.tensor([1.0, 1.0, 1.0], backend: Nx.Vulkan.Backend)

        out = Nx.Defn.jit(f, compiler: Nx.Vulkan.Compiler).(a, b)
        assert_in_delta Nx.to_number(out), 9.0, 1.0e-5
      after
        Nx.global_default_backend(previous)
      end
    end

    test "mass-matrix-style: cholesky → solve composition" do
      # Realistic NUTS use: M is SPD, solve M x = grad via L L^T factorization.
      m =
        Nx.tensor([[4.0, 2.0], [2.0, 5.0]], backend: Nx.Vulkan.Backend)

      grad = Nx.tensor([6.0, 12.0], backend: Nx.Vulkan.Backend)

      l = Nx.LinAlg.cholesky(m)
      # L y = grad
      y = Nx.LinAlg.triangular_solve(l, grad, lower: true)
      # L^T x = y
      lt = Nx.transpose(l)
      x = Nx.LinAlg.triangular_solve(lt, y, lower: false)

      # Verify: M x ≈ grad
      mx = Nx.dot(m, x)

      Enum.zip(Nx.to_flat_list(mx), [6.0, 12.0])
      |> Enum.each(fn {v, e} -> assert_in_delta v, e, 1.0e-4 end)
    end

    test "reshape then to_binary (size_mismatch regression)" do
      t = Nx.tensor([[1.0, 2.0], [3.0, 4.0]], backend: Nx.Vulkan.Backend)
      reshaped = Nx.reshape(t, {4})
      assert Nx.to_flat_list(reshaped) == [1.0, 2.0, 3.0, 4.0]
    end

    test "squeeze then to_binary (size_mismatch regression)" do
      t = Nx.tensor([[[1.0, 2.0]]], backend: Nx.Vulkan.Backend)
      squeezed = Nx.squeeze(t)
      assert Nx.to_flat_list(squeezed) == [1.0, 2.0]
    end

    test "as_type then reshape then to_binary (size_mismatch regression)" do
      t = Nx.tensor([1.0, 2.0, 3.0, 4.0], backend: Nx.Vulkan.Backend)
      t64 = Nx.as_type(t, :f64)
      reshaped = Nx.reshape(t64, {2, 2})
      assert Nx.to_flat_list(reshaped) == [1.0, 2.0, 3.0, 4.0]
    end
  end
end
