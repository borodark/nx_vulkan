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

    @tag :needs_compare_shader
    test "equal" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 2.0, 3.0, 2.0])
      {:ok, b} = Nx.Vulkan.upload_f32([1.0, 5.0, 3.0, 7.0])
      {:ok, c} = Nx.Vulkan.equal(a, b)
      assert {:ok, [1.0, 0.0, 1.0, 0.0]} = Nx.Vulkan.download_f32(c, 4)
    end

    @tag :needs_compare_shader
    test "less" do
      {:ok, a} = Nx.Vulkan.upload_f32([1.0, 5.0, 3.0])
      {:ok, b} = Nx.Vulkan.upload_f32([2.0, 4.0, 3.0])
      {:ok, c} = Nx.Vulkan.less(a, b)
      assert {:ok, [1.0, 0.0, 0.0]} = Nx.Vulkan.download_f32(c, 3)
    end

    @tag :needs_compare_shader
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
end
