defmodule Nx.Vulkan.CompilerTest do
  @moduledoc """
  Thrust 3 — the `Nx.Defn.Compiler` fusion compiler. A fusable same-shape f32
  elementwise chain JIT-compiles to ONE generated GLSL shader dispatched in a
  single GPU call; anything unsupported (reductions, tuple outputs, dot/conv,
  non-f32) falls through to `Nx.Defn.Evaluator` and stays correct.

  Every fused result is checked against the eager BinaryBackend computation.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.{VulkanoBackend, Codegen}

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  defp jit(fun), do: Nx.Defn.jit(fun, compiler: Nx.Vulkan.Compiler)

  defp bin(list), do: Nx.tensor(list, type: :f32, backend: Nx.BinaryBackend)

  defp close?(got, ref) do
    Nx.to_flat_list(got)
    |> Enum.zip(Nx.to_flat_list(ref))
    |> Enum.all?(fn {x, y} -> abs(x - y) <= 1.0e-5 end)
  end

  describe "fused elementwise chains run on the GPU in one dispatch" do
    setup do
      %{a: bin([1.0, 2.0, 3.0, 4.0]), b: bin([0.5, 1.5, 2.5, 3.5])}
    end

    test "tanh(a*b + a)", %{a: a, b: b} do
      got = jit(fn x, y -> Nx.tanh(Nx.add(Nx.multiply(x, y), x)) end).(a, b)
      ref = Nx.tanh(Nx.add(Nx.multiply(a, b), a))
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "a * 2.0 + b (scalar constant folded into the shader)", %{a: a, b: b} do
      got = jit(fn x, y -> Nx.add(Nx.multiply(x, 2.0), y) end).(a, b)
      ref = Nx.add(Nx.multiply(a, 2.0), b)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "relu via max(a - b, 0)", %{a: a, b: b} do
      got = jit(fn x, y -> Nx.max(Nx.subtract(x, y), 0.0) end).(a, b)
      ref = Nx.max(Nx.subtract(a, b), 0.0)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "sigmoid(a) * b", %{a: a, b: b} do
      got = jit(fn x, y -> Nx.multiply(Nx.sigmoid(x), y) end).(a, b)
      ref = Nx.multiply(Nx.sigmoid(a), b)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "single-argument deep unary chain sqrt(exp(negate(a)))", %{a: a} do
      got = jit(fn x -> Nx.sqrt(Nx.exp(Nx.negate(x))) end).(a)
      ref = Nx.sqrt(Nx.exp(Nx.negate(a)))
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, ref)
    end

    test "2D same-shape chain", %{} do
      a = Nx.reshape(bin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), {2, 3})
      b = Nx.reshape(bin([6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), {2, 3})
      got = jit(fn x, y -> Nx.multiply(Nx.add(x, y), Nx.subtract(x, y)) end).(a, b)
      ref = Nx.multiply(Nx.add(a, b), Nx.subtract(a, b))
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.shape(got) == {2, 3}
      assert close?(got, ref)
    end

    test "plain divide (not a mean) still fuses as an elementwise op", %{a: a, b: b} do
      # divide/2 whose LHS is not a reduce must route through the elementwise
      # path, not the mean special-case.
      got = jit(fn x, y -> Nx.divide(x, Nx.add(y, 1.0)) end).(a, b)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.divide(a, Nx.add(b, 1.0)))
    end
  end

  describe "broadcasting inputs fuse (NumPy-style, baked-in shapes)" do
    setup do
      x = Nx.iota({4, 3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.add(1.0)
      %{x: x}
    end

    test "row-vector bias: x{4,3} + b{3}", %{x: x} do
      b = bin([10.0, 20.0, 30.0])
      got = jit(fn a, v -> Nx.add(a, v) end).(x, b)
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.shape(got) == {4, 3}
      assert close?(got, Nx.add(x, b))
    end

    test "column-vector: x{4,3} + c{4,1}", %{x: x} do
      c = Nx.reshape(bin([1.0, 2.0, 3.0, 4.0]), {4, 1})
      got = jit(fn a, v -> Nx.add(a, v) end).(x, c)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.add(x, c))
    end

    test "scalar-tensor scale: x * s{}", %{x: x} do
      s = Nx.tensor(2.5, type: :f32, backend: Nx.BinaryBackend)
      got = jit(fn a, v -> Nx.multiply(a, v) end).(x, s)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.multiply(x, s))
    end

    test "chained broadcast: tanh(x*c{4,1} + b{3})", %{x: x} do
      c = Nx.reshape(bin([1.0, 2.0, 3.0, 4.0]), {4, 1})
      b = bin([0.1, 0.2, 0.3])
      got = jit(fn a, cc, bb -> Nx.tanh(Nx.add(Nx.multiply(a, cc), bb)) end).(x, c, b)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.tanh(Nx.add(Nx.multiply(x, c), b)))
    end

    test "3D broadcast: x{2,3,4} + v{4}" do
      x = Nx.iota({2, 3, 4}, type: :f32, backend: Nx.BinaryBackend)
      v = bin([1.0, 2.0, 3.0, 4.0])
      got = jit(fn a, vv -> Nx.add(a, vv) end).(x, v)
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.shape(got) == {2, 3, 4}
      assert close?(got, Nx.add(x, v))
    end
  end

  # With BinaryBackend inputs the fused path lands on VulkanoBackend while the
  # eager fallback stays on BinaryBackend, so the result backend distinguishes
  # which path ran.
  defp biota(shape, scale \\ 1.0),
    do: Nx.iota(shape, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(scale)

  describe "reductions fuse on the GPU by default in the few-slot regime (wins on Kepler + Ampere)" do
    test "few-slot full reduction with a large axis fuses" do
      v = biota({256})
      got = jit(fn x -> Nx.sum(x) end).(v)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.sum(v))
    end

    test "small-output single-axis reduction (slots <= 256) fuses" do
      # {8, 256} axes:[1] -> 8 slots, contiguous reduce of 256 -> few-slot win
      m = biota({8, 256}, 1.0e-4)
      got = jit(fn x -> Nx.sum(x, axes: [1]) end).(m)
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.shape(got) == {8}
      assert close?(got, Nx.sum(Nx.backend_transfer(m, VulkanoBackend), axes: [1]))
    end
  end

  describe "reductions fall back to the eager path outside the winning regimes (still correct)" do
    test "short-axis full reduction falls back" do
      a = bin([1.0, 2.0, 3.0, 4.0])
      b = bin([1.0, 1.0, 1.0, 1.0])
      got = jit(fn x, y -> Nx.sum(Nx.multiply(x, y)) end).(a, b)
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.to_number(got) == 10.0
    end

    test "short single-axis sum falls back" do
      m = Nx.reshape(bin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), {2, 3})
      got = jit(fn x -> Nx.sum(x, axes: [1]) end).(m)
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.to_flat_list(got) == [6.0, 15.0]
    end

    test "narrow-reduce many-slot falls back (reduce_size < 256)" do
      m = biota({4096, 8})
      got = jit(fn x -> Nx.sum(x, axes: [1]) end).(m)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.sum(m, axes: [1]))
    end

    test "mid slot-count falls back (256 < slots < 2048)" do
      m = biota({1024, 512}, 1.0e-4)
      got = jit(fn x -> Nx.sum(x, axes: [1]) end).(m)
      refute match?(%VulkanoBackend{}, got.data)
    end

    test "many-slot wide reduce falls back on STRONG GPUs (regresses there)" do
      # Wins ~4.4x on Kepler but regresses ~0.44x on Ampere; on a strong GPU it
      # must fall back regardless of the test host's actual device.
      System.put_env("NXV_GPU_CLASS", "strong")
      on_exit(fn -> System.delete_env("NXV_GPU_CLASS") end)
      m = biota({2048, 256}, 1.0e-4)
      got = jit(fn x -> Nx.sum(x, axes: [1]) end).(m)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.sum(m, axes: [1]))
    end
  end

  describe "many-slot wide reduce auto-enables on weak GPUs (device-class check)" do
    setup do
      System.put_env("NXV_GPU_CLASS", "weak")
      on_exit(fn -> System.delete_env("NXV_GPU_CLASS") end)
      :ok
    end

    test "fuses on a weak GPU and is correct" do
      m = biota({2048, 256}, 1.0e-4)
      got = jit(fn x -> Nx.sum(x, axes: [1]) end).(m)
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.shape(got) == {2048}
      assert close?(got, Nx.sum(Nx.backend_transfer(m, VulkanoBackend), axes: [1]))
    end

    test "narrow reduce still falls back even on a weak GPU (reduce < 256)" do
      m = biota({4096, 8})
      got = jit(fn x -> Nx.sum(x, axes: [1]) end).(m)
      refute match?(%VulkanoBackend{}, got.data)
    end
  end

  describe "NXV_FUSE_REDUCE=1 opt-in — fused reduce runs on the GPU, still correct" do
    setup do
      System.put_env("NXV_FUSE_REDUCE", "1")
      on_exit(fn -> System.delete_env("NXV_FUSE_REDUCE") end)
      :ok
    end

    test "full sum of a product" do
      a = bin([1.0, 2.0, 3.0, 4.0])
      b = bin([0.5, 1.5, 2.5, 3.5])
      got = jit(fn x, y -> Nx.sum(Nx.multiply(x, y)) end).(a, b)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.sum(Nx.multiply(a, b)))
    end

    test "reduce_max of a product" do
      a = bin([1.0, 2.0, 3.0, 4.0])
      b = bin([0.5, 1.5, 2.5, 3.5])
      got = jit(fn x, y -> Nx.reduce_max(Nx.multiply(x, y)) end).(a, b)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.reduce_max(Nx.multiply(a, b)))
    end

    test "product of a chain fuses and is correct" do
      a = bin([1.0, 2.0, 3.0, 4.0])
      got = jit(fn x -> Nx.product(Nx.add(x, 1.0)) end).(a)
      assert match?(%VulkanoBackend{}, got.data)
      # (1+1)(2+1)(3+1)(4+1) = 2*3*4*5 = 120
      assert close?(got, Nx.tensor(120.0))
    end

    test "mean fuses as sum with a /n post-scale" do
      a = bin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
      full = jit(fn x -> Nx.mean(x) end).(a)
      assert match?(%VulkanoBackend{}, full.data)
      assert close?(full, Nx.tensor(3.5))

      m = Nx.reshape(bin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), {2, 3})
      rows = jit(fn x -> Nx.mean(x, axes: [1]) end).(m)
      assert match?(%VulkanoBackend{}, rows.data)
      assert close?(rows, Nx.tensor([2.0, 5.0]))
    end

    test "mean of an elementwise chain fuses (single dispatch)" do
      a = bin([1.0, 2.0, 3.0, 4.0])
      got = jit(fn x -> Nx.mean(Nx.multiply(x, 2.0)) end).(a)
      assert match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.tensor(5.0))
    end

    test "contiguous last-axis sum (inner==1) fuses; non-last axis falls back" do
      m = Nx.reshape(bin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), {2, 3})
      # axes:[1] -> inner_stride 1 -> parallel tree reduce on the GPU
      g1 = jit(fn x -> Nx.sum(Nx.multiply(x, 2.0), axes: [1]) end).(m)
      assert match?(%VulkanoBackend{}, g1.data)
      assert Nx.to_flat_list(g1) == [12.0, 30.0]

      # axes:[0] -> inner_stride 3 (uncoalesced) -> falls back, still correct
      g0 = jit(fn x -> Nx.sum(x, axes: [0]) end).(m)
      refute match?(%VulkanoBackend{}, g0.data)
      assert Nx.to_flat_list(g0) == [5.0, 7.0, 9.0]
    end

    test "multi-axis reduction still falls back" do
      m = Nx.reshape(bin([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), {2, 2, 2})
      got = jit(fn x -> Nx.sum(x, axes: [0, 2]) end).(m)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.sum(m, axes: [0, 2]))
    end

    test "many-slot reduce with grid-stride (> 65535 slots) fuses and is correct" do
      # 100k output slots exceeds maxComputeWorkGroupCount[0] (65535); the
      # shader grid-strides over slots. Opt-in only (regresses on Ampere).
      #
      # THE REDUCE WIDTH IS DELIBERATELY SMALL — do not grow it back. Only the
      # SLOT COUNT is under test here, and under `NXV_FUSE_REDUCE=1`
      # `reduce_beneficial?/3` returns true on the `force == "1"` clause before
      # it ever looks at `reduce_size`, so the width cannot change what this
      # exercises.
      #
      # It was {100_000, 128} — 12.8M elements — and `biota/2` builds them on the
      # HOST via `Nx.BinaryBackend.iota/3`, which materialises the range as a
      # list before it becomes a binary. Measured cost of that one line:
      #
      #     width 128  12.8M elements   +3069 MB      <- was
      #     width   8   800k elements    +218 MB
      #     width   4   400k elements     +99 MB      <- is
      #     width   2   200k elements     +48 MB
      #
      # 3 GB on a 4 GB box killed the whole BEAM on the Jetson
      # (`eheap_alloc: Cannot allocate 1098556536 bytes of memory`), taking the
      # entire suite with it — not a timeout, a VM death, so nothing after it
      # ran either. It had been failing there as a TIMEOUT before
      # `strict_test.sh` gained a default `--timeout`; raising the timeout only
      # let it live long enough to exhaust memory instead.
      #
      # The slot count is untouched at 100_000, which is the only thing the
      # test is about.
      m = biota({100_000, 4}, 1.0e-6)
      got = jit(fn x -> Nx.sum(x, axes: [1]) end).(m)
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.shape(got) == {100_000}
      assert close?(got, Nx.sum(Nx.backend_transfer(m, VulkanoBackend), axes: [1]))
    end
  end

  describe "unsupported graphs fall back to the Evaluator (still correct)" do

    test "tuple output falls back" do
      a = bin([1.0, 2.0])
      got = jit(fn x -> {Nx.negate(x), Nx.exp(x)} end).(a)
      {n, e} = got
      assert close?(n, Nx.negate(a))
      assert close?(e, Nx.exp(a))
    end

    test "integer (non-f32) chain falls back" do
      a = Nx.tensor([1, 2, 3], type: :s32, backend: Nx.BinaryBackend)
      got = jit(fn x -> Nx.add(x, x) end).(a)
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.to_flat_list(got) == [2, 4, 6]
    end
  end

  describe "multi-stage split (dot boundaries) — whole NN layers fuse on-GPU" do
    setup do
      x = Nx.iota({8, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      w = Nx.iota({4, 3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      b = bin([0.1, 0.2, 0.3])
      %{x: x, w: w, b: b}
    end

    defp ms_close?(got, ref) do
      match?(%VulkanoBackend{}, got.data) and Nx.shape(got) == Nx.shape(ref) and
        Nx.to_flat_list(got)
        |> Enum.zip(Nx.to_flat_list(ref))
        |> Enum.all?(fn {a, c} -> abs(a - c) <= 1.0e-3 end)
    end

    test "bare dot x @ W runs as a single matmul stage", %{x: x, w: w} do
      got = jit(fn a, ww -> Nx.dot(a, ww) end).(x, w)
      assert ms_close?(got, Nx.dot(x, w))
    end

    test "relu(x @ W + b): matmul stage + fused epilogue", %{x: x, w: w, b: b} do
      got = jit(fn a, ww, bb -> Nx.max(Nx.add(Nx.dot(a, ww), bb), 0.0) end).(x, w, b)
      assert ms_close?(got, Nx.max(Nx.add(Nx.dot(x, w), b), 0.0))
    end

    test "tanh(x @ W + b)", %{x: x, w: w, b: b} do
      got = jit(fn a, ww, bb -> Nx.tanh(Nx.add(Nx.dot(a, ww), bb)) end).(x, w, b)
      assert ms_close?(got, Nx.tanh(Nx.add(Nx.dot(x, w), b)))
    end

    test "fused input to a dot: dot(relu(x), W)", %{x: x, w: w} do
      got = jit(fn a, ww -> Nx.dot(Nx.max(a, 0.0), ww) end).(x, w)
      assert ms_close?(got, Nx.dot(Nx.max(x, 0.0), w))
    end

    test "2-layer MLP fuses to 4 stages", %{x: x, w: w, b: b} do
      w2 = Nx.iota({3, 2}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.03)
      b2 = bin([0.5, 0.6])

      fun = fn a, ww, bb, w22, b22 ->
        Nx.sigmoid(Nx.add(Nx.dot(Nx.tanh(Nx.add(Nx.dot(a, ww), bb)), w22), b22))
      end

      got = jit(fun).(x, w, b, w2, b2)
      assert ms_close?(got, fun.(x, w, b, w2, b2))
      assert Nx.shape(got) == {8, 2}
    end

    test "batched (non-2D) dot falls back but stays correct" do
      a = Nx.iota({2, 3, 4}, type: :f32, backend: Nx.BinaryBackend)
      b = Nx.iota({2, 4, 5}, type: :f32, backend: Nx.BinaryBackend)
      got = jit(fn x, y -> Nx.dot(x, [2], [0], y, [1], [0]) end).(a, b)
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.to_flat_list(got) == Nx.to_flat_list(Nx.dot(a, [2], [0], b, [1], [0]))
    end
  end

  describe "multi-stage split (conv boundaries) — CNN layers fuse on-GPU" do
    setup do
      # input {n=1, cin=2, 5, 5}, kernel {cout=3, cin=2, 3, 3} -> out {1, 3, 3, 3}
      x = Nx.iota({1, 2, 5, 5}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      k = Nx.iota({3, 2, 3, 3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      # bias {1, cout, 1, 1} broadcasts over the conv output
      b = Nx.iota({1, 3, 1, 1}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.1)
      %{x: x, k: k, b: b}
    end

    test "bare conv(x, k) runs as a single im2col+GEMM stage", %{x: x, k: k} do
      got = jit(fn a, kk -> Nx.conv(a, kk) end).(x, k)
      assert ms_close?(got, Nx.conv(x, k))
      assert Nx.shape(got) == {1, 3, 3, 3}
    end

    test "relu(conv(x, k) + b): conv stage + fused epilogue", %{x: x, k: k, b: b} do
      fun = fn a, kk, bb -> Nx.max(Nx.add(Nx.conv(a, kk), bb), 0.0) end
      got = jit(fun).(x, k, b)
      assert ms_close?(got, fun.(x, k, b))
    end

    test "fused input to a conv: conv(relu(x), k)", %{x: x, k: k} do
      fun = fn a, kk -> Nx.conv(Nx.max(a, 0.0), kk) end
      got = jit(fun).(x, k)
      assert ms_close?(got, fun.(x, k))
    end

    test "conv honours strides + padding", %{x: x, k: k} do
      fun = fn a, kk -> Nx.conv(a, kk, strides: [2, 2], padding: [{1, 1}, {1, 1}]) end
      got = jit(fun).(x, k)
      assert ms_close?(got, fun.(x, k))
    end

    test "grouped conv (feature_group_size > 1) falls back but stays correct" do
      x = Nx.iota({1, 4, 5, 5}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      k = Nx.iota({2, 2, 3, 3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      fun = fn a, kk -> Nx.conv(a, kk, feature_group_size: 2) end
      got = jit(fun).(x, k)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, fun.(x, k))
    end
  end

  describe "reshape / squeeze are no-op view boundaries (buffer alias, no dispatch)" do
    setup do
      # conv {1,2,5,5} * {4,2,3,3} -> {1,4,3,3} = 36 elems; flatten -> {1,36}
      x = Nx.iota({1, 2, 5, 5}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      k = Nx.iota({4, 2, 3, 3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      w = Nx.iota({36, 3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.005)
      b = bin([0.1, 0.2, 0.3])
      %{x: x, k: k, w: w, b: b}
    end

    test "CNN classifier head fuses end-to-end: relu(flatten(conv) @ W + b)",
         %{x: x, k: k, w: w, b: b} do
      fun = fn a, kk, ww, bb ->
        flat = Nx.reshape(Nx.conv(a, kk), {1, 36})
        Nx.max(Nx.add(Nx.dot(flat, ww), bb), 0.0)
      end

      got = jit(fun).(x, k, w, b)
      assert ms_close?(got, fun.(x, k, w, b))
      assert Nx.shape(got) == {1, 3}
    end

    test "reshape of a matmul output as the root", %{} do
      a = Nx.iota({2, 6}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      w = Nx.iota({6, 6}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      fun = fn x, ww -> Nx.reshape(Nx.dot(x, ww), {3, 4}) end
      got = jit(fun).(a, w)
      assert ms_close?(got, fun.(a, w))
      assert Nx.shape(got) == {3, 4}
    end

    test "reshape feeding an elementwise epilogue after a dot", %{} do
      a = Nx.iota({2, 6}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      w = Nx.iota({6, 6}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      fun = fn x, ww -> Nx.tanh(Nx.reshape(Nx.dot(x, ww), {4, 3})) end
      got = jit(fun).(a, w)
      assert ms_close?(got, fun.(a, w))
    end

    test "squeeze after a conv, then dense", %{x: x, k: k} do
      # conv -> {1,4,3,3}; squeeze axis 0 -> {4,3,3}; flatten -> {4,9}
      w = Nx.iota({9, 2}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)

      fun = fn a, kk, ww ->
        s = Nx.squeeze(Nx.conv(a, kk), axes: [0])
        Nx.dot(Nx.reshape(s, {4, 9}), ww)
      end

      got = jit(fun).(x, k, w)
      assert ms_close?(got, fun.(x, k, w))
      assert Nx.shape(got) == {4, 2}
    end
  end

  describe "f64 fusion — the generated shaders are dtype-parameterised" do
    defp b64(list), do: Nx.tensor(list, type: :f64, backend: Nx.BinaryBackend)

    # For f64 the reference is `Nx.Defn.Evaluator` — the fallback this compiler
    # defers to — not an eagerly-computed expression. The two genuinely differ
    # when a bare Elixir float literal is involved: eager Nx types it f32 and
    # then promotes (carrying an f32-rounded constant), while inside a traced
    # graph the constant is already f64. Fusion must equal the fallback.
    defp evaluated(fun, args), do: apply(Nx.Defn.jit(fun, compiler: Nx.Defn.Evaluator), args)

    defp close64?(got, ref, tol \\ 1.0e-12) do
      match?(%VulkanoBackend{}, got.data) and Nx.shape(got) == Nx.shape(ref) and
        Nx.type(got) == {:f, 64} and
        Nx.to_flat_list(got)
        |> Enum.zip(Nx.to_flat_list(ref))
        |> Enum.all?(fn {a, c} -> abs(a - c) <= tol * max(1.0, abs(c)) end)
    end

    setup do
      x = Nx.iota({8, 4}, type: :f64, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      w = Nx.iota({4, 3}, type: :f64, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      %{a: b64([1.0, 2.0, 3.0, 4.0]), b: b64([0.5, 1.5, 2.5, 3.5]), x: x, w: w}
    end

    test "elementwise chain fuses at f64", %{a: a, b: b} do
      fun = fn p, q -> Nx.add(Nx.multiply(p, q), Nx.multiply(p, 2.0)) end
      got = jit(fun).(a, b)
      assert close64?(got, evaluated(fun, [a, b]))
    end

    test "f64 arithmetic keeps double precision (not an f32 round trip)", %{a: a} do
      # 0.1 is not representable in binary; at f32 it rounds to
      # 0.100000001490116..., so an f32 round trip anywhere shows up at ~1e-9.
      fun = fn p -> Nx.add(p, 0.1) end
      got = jit(fun).(a)
      assert close64?(got, evaluated(fun, [a]))
      assert hd(Nx.to_flat_list(got)) == 1.1
    end

    test "broadcast operands fuse at f64", %{x: x} do
      row = b64([0.1, 0.2, 0.3, 0.4])
      fun = fn p, r -> Nx.multiply(Nx.add(p, r), 2.0) end
      got = jit(fun).(x, row)
      assert close64?(got, evaluated(fun, [x, row]))
    end

    test "full sum reduces at f64 in one fused dispatch" do
      t = Nx.iota({256}, type: :f64, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      fun = fn p -> Nx.sum(Nx.multiply(p, p)) end
      got = jit(fun).(t)
      assert close64?(got, evaluated(fun, [t]))
    end

    test "mean fuses at f64 (sum with a baked /n)" do
      t = Nx.iota({4, 64}, type: :f64, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      fun = fn p -> Nx.mean(p, axes: [1]) end
      got = jit(fun).(t)
      assert close64?(got, evaluated(fun, [t]))
      assert Nx.shape(got) == {4}
    end

    test "reduce_max fuses at f64 (element-typed accumulator)" do
      t = Nx.iota({4, 64}, type: :f64, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      fun = fn p -> Nx.reduce_max(Nx.multiply(p, -1.0), axes: [1]) end
      got = jit(fun).(t)
      assert close64?(got, evaluated(fun, [t]))
    end

    test "2-D dot runs the f64 GEMM as a stage", %{x: x, w: w} do
      fun = fn p, q -> Nx.dot(p, q) end
      got = jit(fun).(x, w)
      assert close64?(got, evaluated(fun, [x, w]))
      assert Nx.shape(got) == {8, 3}
    end

    test "relu(x @ W + b) fuses end-to-end at f64", %{x: x, w: w} do
      bias = b64([0.1, 0.2, 0.3])
      fun = fn p, q, r -> Nx.max(Nx.add(Nx.dot(p, q), r), 0.0) end
      got = jit(fun).(x, w, bias)
      assert close64?(got, evaluated(fun, [x, w, bias]))
    end

    test "x @ Wᵀ fuses at f64 (transpose stage picks the f64 shader)", %{x: x} do
      w = Nx.iota({3, 4}, type: :f64, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      fun = fn p, q -> Nx.dot(p, Nx.transpose(q)) end
      got = jit(fun).(x, w)
      assert close64?(got, evaluated(fun, [x, w]))
      assert Nx.shape(got) == {8, 3}
    end

    test "conv fuses at f64" do
      i = Nx.iota({1, 2, 5, 5}, type: :f64, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      k = Nx.iota({3, 2, 3, 3}, type: :f64, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      fun = fn p, q -> Nx.max(Nx.conv(p, q), 0.0) end
      got = jit(fun).(i, k)
      assert close64?(got, evaluated(fun, [i, k]))
      assert Nx.shape(got) == {1, 3, 3, 3}
    end

    # GLSL.std.450 has no f64 exp/log/pow/tanh — those trees must host-fall-back
    # rather than silently boundary-cast through f32.
    for {name, fun} <- [
          exp: &Nx.exp/1,
          log: &Nx.log/1,
          tanh: &Nx.tanh/1,
          sigmoid: &Nx.sigmoid/1,
          erf: &Nx.erf/1
        ] do
      test "f64 #{name} falls back to the Evaluator (no f64 transcendental in SPIR-V)", %{a: a} do
        f = unquote(Macro.escape(fun))
        got = jit(fn p -> f.(Nx.multiply(p, 2.0)) end).(a)
        refute match?(%VulkanoBackend{}, got.data)
        assert close?(got, f.(Nx.multiply(a, 2.0)))
      end
    end

    test "f64 pow falls back but stays correct", %{a: a, b: b} do
      got = jit(fn p, q -> Nx.pow(p, q) end).(a, b)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.pow(a, b))
    end

    test "an f64 transcendental inside a dot graph drops the whole graph", %{x: x, w: w} do
      fun = fn p, q -> Nx.tanh(Nx.dot(p, q)) end
      got = jit(fun).(x, w)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, evaluated(fun, [x, w]))
    end

    test "f64 sqrt/abs/sign DO fuse (GLSL.std.450 covers 64-bit for these)", %{a: a} do
      fun = fn p -> Nx.multiply(Nx.sqrt(Nx.abs(p)), Nx.sign(p)) end
      got = jit(fun).(a)
      assert close64?(got, evaluated(fun, [a]))
    end

    test "mixed f32/f64 graph is not fused (would need a cast stage)", %{w: w} do
      x32 = Nx.iota({8, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      fun = fn p, q -> Nx.dot(p, q) end
      got = jit(fun).(x32, w)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, Nx.dot(x32, w))
    end

    # A3: the dispatch NIFs bind byte buffers and push element COUNTS, so they
    # are element-agnostic — an f64 add is bit-exact, not merely close, provided
    # Elixir allocated n*8 and the shader declared `double`. A wrong ebytes or a
    # `float` decl shows up here immediately.
    test "f64 fused add is BIT-EXACT vs BinaryBackend", %{a: a, b: b} do
      got = jit(fn p, q -> Nx.add(p, q) end).(a, b)
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.to_flat_list(got) == Nx.to_flat_list(Nx.add(a, b))
    end

    test "f64 buffers are 8 bytes/element (a 4-byte alloc would truncate)" do
      t = Nx.tensor(Enum.map(1..64, &(&1 * 1.0)), type: :f64, backend: Nx.BinaryBackend)
      got = jit(fn p -> Nx.multiply(p, p) end).(t)
      assert match?(%VulkanoBackend{}, got.data)
      assert Nx.to_flat_list(got) == Nx.to_flat_list(Nx.multiply(t, t))
    end

    test "Device.f64?/0 agrees with the NIF capability bit" do
      assert {:ok, bit} = Nx.Vulkan.NativeV.device_supports_f64()
      assert is_boolean(bit)
      assert Nx.Vulkan.Device.f64?() == bit
    end

    test "NXV_F64=0 sends f64 graphs to the Evaluator (still correct)", %{x: x, w: w} do
      System.put_env("NXV_F64", "0")

      try do
        fun = fn p, q -> Nx.max(Nx.dot(p, q), 0.0) end
        got = jit(fun).(x, w)
        refute match?(%VulkanoBackend{}, got.data)
        assert close?(got, evaluated(fun, [x, w]))

        ew = jit(fn p -> Nx.multiply(p, 2.0) end).(x)
        refute match?(%VulkanoBackend{}, ew.data)
      after
        System.delete_env("NXV_F64")
      end
    end

    test "NXV_F64=0 leaves f32 fusion untouched", %{} do
      System.put_env("NXV_F64", "0")

      try do
        a32 = bin([1.0, 2.0, 3.0, 4.0])
        got = jit(fn p -> Nx.multiply(p, 2.0) end).(a32)
        assert match?(%VulkanoBackend{}, got.data)
        assert close?(got, Nx.multiply(a32, 2.0))
      after
        System.delete_env("NXV_F64")
      end
    end

    test "integer graphs still fall back" do
      i = Nx.iota({8}, type: :s32, backend: Nx.BinaryBackend)
      got = jit(fn p -> Nx.add(p, 1) end).(i)
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.to_flat_list(got) == Nx.to_flat_list(Nx.add(i, 1))
    end
  end

  describe "transpose is a real stage boundary (moves data, unlike reshape)" do
    setup do
      x = Nx.iota({8, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      w = Nx.iota({3, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      %{x: x, w: w}
    end

    test "bare 2-D transpose runs as a single transpose stage", %{x: x} do
      got = jit(fn a -> Nx.transpose(a) end).(x)
      assert ms_close?(got, Nx.transpose(x))
      assert Nx.shape(got) == {4, 8}
    end

    test "x @ Wᵀ — the standard dense form — fuses as transpose + matmul", %{x: x, w: w} do
      fun = fn a, ww -> Nx.dot(a, Nx.transpose(ww)) end
      got = jit(fun).(x, w)
      assert ms_close?(got, fun.(x, w))
      assert Nx.shape(got) == {8, 3}
    end

    test "relu(transpose(x @ W)): matmul + transpose + fused epilogue", %{x: x} do
      w = Nx.iota({4, 3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      fun = fn a, ww -> Nx.max(Nx.transpose(Nx.dot(a, ww)), 0.0) end
      got = jit(fun).(x, w)
      assert ms_close?(got, fun.(x, w))
      assert Nx.shape(got) == {3, 8}
    end

    test "transpose feeding an elementwise region", %{x: x} do
      fun = fn a -> Nx.tanh(Nx.multiply(Nx.transpose(a), 2.0)) end
      got = jit(fun).(x)
      assert ms_close?(got, fun.(x))
    end

    # B5: transpose materialises a contiguous buffer, so a reshape over it is a
    # valid view and may alias it — unlike a reshape over a non-contiguous view.
    test "reshape over a transpose aliases the transposed buffer", %{x: x} do
      fun = fn a -> Nx.reshape(Nx.transpose(a), {2, 16}) end
      got = jit(fun).(x)
      assert ms_close?(got, fun.(x))
      assert Nx.shape(got) == {2, 16}
    end

    test "transposed weight reused by two consumers computes the transpose once", %{x: x, w: w} do
      # wᵀ feeds a dot AND an elementwise region feeding a second dot; it is
      # materialised once (memoised) and both consumers read that buffer.
      fun = fn a, ww ->
        t = Nx.transpose(ww)
        Nx.add(Nx.dot(a, t), Nx.dot(a, Nx.multiply(t, 2.0)))
      end

      got = jit(fun).(x, w)
      assert ms_close?(got, fun.(x, w))
    end

    test "higher-rank transpose falls back to the Evaluator (still correct)" do
      a = Nx.iota({2, 3, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      got = jit(fn t -> Nx.transpose(t, axes: [0, 2, 1]) end).(a)
      refute match?(%VulkanoBackend{}, got.data)
      assert Nx.to_flat_list(got) == Nx.to_flat_list(Nx.transpose(a, axes: [0, 2, 1]))
    end

    test "non-[1,0] 3-D permutation feeding a dot falls back (still correct)" do
      a = Nx.iota({2, 3, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      w = Nx.iota({2, 5}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      fun = fn t, ww -> Nx.dot(Nx.transpose(t, axes: [2, 1, 0]), [2], [], ww, [0], []) end
      got = jit(fun).(a, w)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, fun.(a, w))
    end
  end

  describe "multi-stage split (reduce boundaries) — layernorm/softmax patterns fuse" do
    setup do
      # {4, 64}: reduce over axis 1 -> outer=4, rsize=64, inner=1, slots=4 (in the
      # beneficial few-slot regime: slots <= 256, rsize >= 64).
      x = Nx.iota({4, 64}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      %{x: x}
    end

    test "mean-centering x - mean(x, axis) (layernorm numerator) fuses", %{x: x} do
      fun = fn a -> Nx.subtract(a, Nx.mean(a, axes: [1], keep_axes: true)) end
      got = jit(fun).(x)
      assert ms_close?(got, fun.(x))
      assert Nx.shape(got) == {4, 64}
    end

    test "full-reduction centering x - mean(x) (scalar mean) fuses", %{x: x} do
      fun = fn a -> Nx.subtract(a, Nx.mean(a)) end
      got = jit(fun).(x)
      assert ms_close?(got, fun.(x))
    end

    test "row normalize x / sum(x, axis) fuses", %{x: x} do
      fun = fn a -> Nx.divide(a, Nx.sum(a, axes: [1], keep_axes: true)) end
      got = jit(fun).(x)
      assert ms_close?(got, fun.(x))
    end

    test "softmax numerator exp(x - max(x, axis)) fuses (reduce_max boundary)", %{x: x} do
      fun = fn a -> Nx.exp(Nx.subtract(a, Nx.reduce_max(a, axes: [1], keep_axes: true))) end
      got = jit(fun).(x)
      assert ms_close?(got, fun.(x))
    end

    test "two reduce stages: sum(x, axis) + reduce_max(x, axis)", %{x: x} do
      fun = fn a ->
        Nx.add(
          Nx.sum(a, axes: [1], keep_axes: true),
          Nx.reduce_max(a, axes: [1], keep_axes: true)
        )
      end

      got = jit(fun).(x)
      assert ms_close?(got, fun.(x))
      assert Nx.shape(got) == {4, 1}
    end

    test "non-beneficial reduce (short axis) falls back whole-graph but stays correct" do
      # rsize = 8 < @min_reduce_few (64) -> reduce_beneficial? false -> :unschedulable
      x = Nx.iota({4, 8}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      fun = fn a -> Nx.subtract(a, Nx.mean(a, axes: [1], keep_axes: true)) end
      got = jit(fun).(x)
      refute match?(%VulkanoBackend{}, got.data)
      assert close?(got, fun.(x))
    end
  end

  describe "multi-output (tuple) graphs fuse into one shared stage schedule" do
    setup do
      x = Nx.iota({8, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      w = Nx.iota({4, 3}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      %{x: x, w: w}
    end

    test "two outputs share a matmul: {x@W, relu(x@W)}", %{x: x, w: w} do
      fun = fn a, ww ->
        y = Nx.dot(a, ww)
        {y, Nx.max(y, 0.0)}
      end

      {g1, g2} = jit(fun).(x, w)
      {r1, r2} = fun.(x, w)
      assert ms_close?(g1, r1)
      assert ms_close?(g2, r2)
    end

    test "layernorm stats {mean, variance} share the sum stage" do
      x = Nx.iota({4, 64}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)

      fun = fn a ->
        m = Nx.mean(a, axes: [1], keep_axes: true)
        d = Nx.subtract(a, m)
        v = Nx.mean(Nx.multiply(d, d), axes: [1], keep_axes: true)
        {m, v}
      end

      {gm, gv} = jit(fun).(x)
      {rm, rv} = fun.(x)
      assert ms_close?(gm, rm)
      assert ms_close?(gv, rv)
    end

    test "softmax stats {exp(x-max), sum(exp(x-max))}" do
      x = Nx.iota({4, 64}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)

      fun = fn a ->
        n = Nx.exp(Nx.subtract(a, Nx.reduce_max(a, axes: [1], keep_axes: true)))
        {n, Nx.sum(n, axes: [1], keep_axes: true)}
      end

      {gn, gs} = jit(fun).(x)
      {rn, rs} = fun.(x)
      assert ms_close?(gn, rn)
      assert ms_close?(gs, rs)
    end

    test "output that is a passthrough parameter: {x, relu(x@W)}", %{x: x, w: w} do
      fun = fn a, ww -> {a, Nx.max(Nx.dot(a, ww), 0.0)} end
      {g1, g2} = jit(fun).(x, w)
      {r1, r2} = fun.(x, w)
      assert ms_close?(g1, r1)
      assert ms_close?(g2, r2)
    end

    test "a non-schedulable leaf drops the whole tuple to the Evaluator (still correct)" do
      a = Nx.iota({2, 3, 4}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)
      b = Nx.iota({2, 4, 5}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.02)
      # a batched dot is unschedulable -> the tuple falls back whole
      fun = fn x, y -> {Nx.dot(x, [2], [0], y, [1], [0]), Nx.add(x, 1.0)} end
      {g1, g2} = jit(fun).(a, b)
      refute match?(%VulkanoBackend{}, g1.data)
      {r1, r2} = fun.(a, b)
      assert Nx.to_flat_list(g1) == Nx.to_flat_list(r1)
      assert close?(g2, r2)
    end
  end

  describe "cross-stage CSE — a subexpression shared across a stage boundary is computed once" do
    test "full single-output softmax x/sum(exp(x-max)) fuses and is correct" do
      x = Nx.iota({4, 64}, type: :f32, backend: Nx.BinaryBackend) |> Nx.multiply(0.01)

      softmax = fn a ->
        n = Nx.exp(Nx.subtract(a, Nx.reduce_max(a, axes: [1], keep_axes: true)))
        Nx.divide(n, Nx.sum(n, axes: [1], keep_axes: true))
      end

      got = jit(softmax).(x)
      assert match?(%VulkanoBackend{}, got.data)
      assert ms_close?(got, softmax.(x))
      # rows are a probability distribution
      rowsums = Nx.sum(got, axes: [1]) |> Nx.to_flat_list()
      assert Enum.all?(rowsums, &(abs(&1 - 1.0) <= 1.0e-4))
    end

    test "the shared numerator is hoisted (referenced by the divide and the sum)" do
      alias Nx.Defn.Expr
      x = Expr.parameter(Nx.template({4, 64}, :f32), :root, 0)
      n = Nx.exp(Nx.subtract(x, Nx.reduce_max(x, axes: [1], keep_axes: true)))
      result = Nx.divide(n, Nx.sum(n, axes: [1], keep_axes: true))

      shared = Nx.Vulkan.Compiler.__hoist_ids__(result)
      assert MapSet.member?(shared, n.data.id)
    end

    test "a node shared only among elementwise parents is NOT hoisted (intra-region CSE handles it)" do
      alias Nx.Defn.Expr
      x = Expr.parameter(Nx.template({64}, :f32), :root, 0)
      f = Nx.multiply(x, 2.0)
      # f is used by two elementwise parents (add, subtract) that fuse into one region
      result = Nx.add(Nx.add(f, 1.0), Nx.subtract(f, 1.0))

      shared = Nx.Vulkan.Compiler.__hoist_ids__(result)
      refute MapSet.member?(shared, f.data.id)
    end
  end

  describe "Codegen unit" do
    test "fusable?/1 accepts an f32 elementwise tree, rejects a reduction" do
      alias Nx.Defn.Expr
      p0 = Expr.parameter(Nx.template({4}, :f32), :root, 0)
      p1 = Expr.parameter(Nx.template({4}, :f32), :root, 1)
      assert Codegen.fusable?(Nx.tanh(Nx.add(Nx.multiply(p0, p1), p0)))
      refute Codegen.fusable?(Nx.sum(Nx.multiply(p0, p1)))
    end

    test "emit_elementwise assigns bindings in ascending parameter order" do
      alias Nx.Defn.Expr
      p0 = Expr.parameter(Nx.template({4}, :f32), :root, 0)
      p1 = Expr.parameter(Nx.template({4}, :f32), :root, 1)
      # Use p1 before p0 in the source order; binding order must still be [0, 1].
      {glsl, meta} = Codegen.emit_elementwise(Nx.subtract(p1, p0))
      assert meta.param_order == [0, 1]
      assert meta.n_inputs == 2
      # CSE: the interior node is emitted as a temp, then referenced as the root.
      assert glsl =~ "float t0 = (v1 - v0);"
      assert glsl =~ "out_buf[i] = t0;"
    end

    test "broadcast param loads at its mapped index; full-shape param loads at i" do
      alias Nx.Defn.Expr
      x = Expr.parameter(Nx.template({4, 3}, :f32), :root, 0)
      bias = Expr.parameter(Nx.template({3}, :f32), :root, 1)
      {glsl, _} = Codegen.emit_elementwise(Nx.add(x, bias))
      # full-shape input loads linearly; the {3} row-vector loads at i % 3
      assert glsl =~ "float v0 = buf0[i];"
      assert glsl =~ "((i / 1u) % 3u)"
      assert glsl =~ "float v1 = buf1["
    end

    test "CSE — a fan-out node is computed once, not re-inlined (no exponential blowup)" do
      alias Nx.Defn.Expr
      p0 = Expr.parameter(Nx.template({4}, :f32), :root, 0)
      # 8 chained squarings of the same node: naive inlining -> 255 multiplies,
      # CSE -> 8 temps / 8 multiplies.
      expr = Enum.reduce(1..8, p0, fn _, t -> Nx.multiply(t, t) end)
      {glsl, _} = Codegen.emit_elementwise(expr)
      # count `tN = (... * ...)` temp assignments (exclude helper-fn bodies)
      squaring_temps =
        Regex.scan(~r/float t\d+ = \(t?v?\d+ \* t?v?\d+\);/, glsl) |> length()

      assert squaring_temps == 8
    end
  end
end
