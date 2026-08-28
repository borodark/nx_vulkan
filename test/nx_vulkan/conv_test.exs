defmodule Nx.Vulkan.ConvTest do
  @moduledoc """
  Native f64 convolution on the GPU (im2col + GEMM), verified against a
  BinaryBackend reference. Covers correctness across strides, padding,
  input/kernel dilation, multi-channel, batch and spatial ranks 1/2/3, asserts
  the covered cases dispatch on the GPU — including non-identity permutations,
  which are transposed into the native layout on-device — and checks that the
  still-unsupported cases (feature groups, spatial rank > 3) fall back but stay
  correct.
  """

  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  # Deterministic non-trivial data, reshaped to the requested shape, on `backend`.
  defp gen(shape, backend, seed) do
    size = Tuple.product(shape)
    data = for i <- 1..size, do: :math.sin(seed * 0.3 + i * 0.37)
    Nx.tensor(data, type: {:f, 64}, backend: backend) |> Nx.reshape(shape)
  end

  defp run(ishape, kshape, opts) do
    iv = gen(ishape, VulkanoBackend, 1)
    kv = gen(kshape, VulkanoBackend, 2)
    ib = gen(ishape, Nx.BinaryBackend, 1)
    kb = gen(kshape, Nx.BinaryBackend, 2)
    {Nx.conv(iv, kv, opts), Nx.conv(ib, kb, opts)}
  end

  defp max_abs_diff(a, b) do
    assert Nx.shape(a) == Nx.shape(b)

    Nx.subtract(Nx.backend_copy(a, Nx.BinaryBackend), b)
    |> Nx.abs()
    |> Nx.reduce_max()
    |> Nx.to_number()
  end

  describe "GPU path — on-device and matches BinaryBackend (f64)" do
    for {label, ishape, kshape, opts} <- [
          {"2d basic", {1, 1, 5, 5}, {1, 1, 3, 3}, []},
          {"2d stride 2", {1, 1, 7, 7}, {1, 1, 3, 3}, [strides: 2]},
          {"2d padding :same", {1, 1, 5, 5}, {1, 1, 3, 3}, [padding: :same]},
          {"2d padding general", {1, 1, 5, 5}, {1, 1, 3, 3}, [padding: [{1, 2}, {2, 1}]]},
          {"2d multichannel", {1, 3, 6, 6}, {4, 3, 3, 3}, []},
          {"2d batched", {2, 2, 5, 5}, {3, 2, 2, 2}, []},
          {"2d kernel_dilation 2", {1, 1, 7, 7}, {1, 1, 3, 3}, [kernel_dilation: 2]},
          {"2d input_dilation 2", {1, 1, 4, 4}, {1, 1, 3, 3}, [input_dilation: 2]},
          {"2d stride+pad+dilation", {1, 2, 8, 8}, {3, 2, 3, 3},
           [strides: 2, padding: [{1, 1}, {1, 1}], kernel_dilation: 2]},
          {"1d basic", {1, 1, 8}, {1, 1, 3}, []},
          {"1d multichannel stride 2", {2, 3, 10}, {5, 3, 4}, [strides: 2]},
          {"3d basic", {1, 1, 4, 4, 4}, {1, 1, 2, 2, 2}, []}
        ] do
      test "#{label}" do
        {got, ref} =
          run(unquote(Macro.escape(ishape)), unquote(Macro.escape(kshape)), unquote(opts))

        assert match?(%VulkanoBackend{}, got.data), "expected on-GPU dispatch"
        assert Nx.type(got) == {:f, 64}
        assert max_abs_diff(got, ref) < 1.0e-10
      end
    end
  end

  describe "host fallback — correct for unsupported cases" do
    @tag :host_fallback_expected
    test "feature groups fall back but stay correct" do
      {got, ref} = run({1, 4, 5, 5}, {4, 2, 3, 3}, feature_group_size: 2)
      refute match?(%VulkanoBackend{}, got.data)
      assert max_abs_diff(got, ref) < 1.0e-10
    end

    test "non-identity input permutation stays on the GPU and is correct" do
      # channels-last input {N, H, W, C}. The native shaders only run the
      # canonical layout, so the backend transposes into it on-device rather
      # than host-falling-back — see permuted_gpu_conv/4. This is what keeps
      # conv's backward pass (whose permutations always swap the first two
      # axes) on the GPU.
      {got, ref} = run({1, 5, 5, 3}, {4, 3, 3, 3}, input_permutation: [0, 3, 1, 2])
      assert match?(%VulkanoBackend{}, got.data)
      assert max_abs_diff(got, ref) < 1.0e-10
    end

    test "grad-shaped permutation (first two axes swapped) stays on the GPU" do
      {got, ref} =
        run({3, 2, 5, 5}, {3, 4, 3, 3},
          input_permutation: [1, 0, 2, 3],
          kernel_permutation: [1, 0, 2, 3],
          output_permutation: [1, 0, 2, 3]
        )

      assert match?(%VulkanoBackend{}, got.data)
      assert max_abs_diff(got, ref) < 1.0e-10
    end

    @tag :host_fallback_expected
    test "rank>3 (spatial rank 4) falls back but stays correct" do
      # 4 spatial dims -> sr=4 > 3, host fallback
      i = gen({1, 1, 3, 3, 3, 3}, VulkanoBackend, 1)
      k = gen({2, 1, 2, 2, 2, 2}, VulkanoBackend, 2)
      got = Nx.conv(i, k)
      refute match?(%VulkanoBackend{}, got.data)
      ib = Nx.backend_copy(i, Nx.BinaryBackend)
      kb = Nx.backend_copy(k, Nx.BinaryBackend)
      assert max_abs_diff(got, Nx.conv(ib, kb)) < 1.0e-10
    end
  end

  describe "f32 conv runs on the GPU and matches BinaryBackend" do
    setup do
      prev = VulkanoBackend.f32_matmul_accumulator()
      on_exit(fn -> VulkanoBackend.put_f32_matmul_accumulator(prev) end)
      :ok
    end

    test "2d multichannel f32, both accumulator policies on GPU + correct" do
      iv =
        Nx.tensor(for(i <- 1..75, do: i * 0.1), type: {:f, 32}, backend: VulkanoBackend)
        |> Nx.reshape({1, 3, 5, 5})

      kv =
        Nx.tensor(for(i <- 1..54, do: i * 0.05), type: {:f, 32}, backend: VulkanoBackend)
        |> Nx.reshape({2, 3, 3, 3})

      ib = Nx.backend_copy(iv, Nx.BinaryBackend)
      kb = Nx.backend_copy(kv, Nx.BinaryBackend)
      ref = Nx.conv(ib, kb)

      for policy <- [:f64, :f32] do
        VulkanoBackend.put_f32_matmul_accumulator(policy)
        got = Nx.conv(iv, kv)
        assert match?(%VulkanoBackend{}, got.data), "#{policy} should stay on GPU"
        assert Nx.type(got) == {:f, 32}
        assert max_abs_diff(got, ref) < 1.0e-4
      end
    end
  end
end
