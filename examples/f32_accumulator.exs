# Why the f32 matmul uses an f64 accumulator.
#
# Compares two f32 matmul shaders against an f64 ground truth:
#   * matmul_f32_naive     — accumulates the dot product in f32
#   * matmul_f32_f64acc    — accumulates in f64 (what the backend ships)
#
#   mix run examples/f32_accumulator.exs

Application.ensure_all_started(:nx_vulkan)
alias Nx.Vulkan.VulkanoBackend
alias Nx.Vulkan.NativeV

naive_spv = Path.expand("priv/shaders/matmul_f32_naive.spv")
f64acc_spv = Path.expand("priv/shaders/matmul_f32_f64acc.spv")

# run an f32 matmul with a chosen shader, return the result as a flat f64 list
mm32 = fn av, bv, m, k, n, spv ->
  %{data: %{ref: aref}} = av
  %{data: %{ref: bref}} = bv
  {:ok, oref} = NativeV.buf_alloc(m * n * 4)
  :ok = NativeV.matmul(oref, aref, bref, m, n, k, spv)
  {:ok, bin} = NativeV.buf_download(oref)
  for <<x::float-32-little <- binary_part(bin, 0, m * n * 4)>>, do: x * 1.0
end

maxerr = fn got_list, truth_tensor ->
  truth = Nx.to_flat_list(truth_tensor)
  Enum.zip(got_list, truth) |> Enum.map(fn {a, b} -> abs(a - b) end) |> Enum.max()
end

IO.puts("\n== A. Ill-conditioned dot product (large value swamps small terms) ==")
IO.puts("   row = [1e9, 1, 1, ..., 1, -1e9]; true value is the count of 1s = K-2.")
# A large value sits in the accumulator; each +1.0 is < half an ulp of 1e9 in
# f32 (ulp(1e9) ~ 64) and is silently dropped, so naive f32 collapses to 0.
for k <- [64, 256, 1024] do
  a = for i <- 0..(k - 1) do
    cond do
      i == 0 -> 1.0e9
      i == k - 1 -> -1.0e9
      true -> 1.0
    end
  end
  b = List.duplicate(1.0, k)

  av = Nx.tensor(a, type: {:f, 32}, backend: VulkanoBackend) |> Nx.reshape({1, k})
  bv = Nx.tensor(b, type: {:f, 32}, backend: VulkanoBackend) |> Nx.reshape({k, 1})
  truth = Nx.dot(Nx.tensor(a, type: {:f, 64}, backend: Nx.BinaryBackend) |> Nx.reshape({1, k}),
                 Nx.tensor(b, type: {:f, 64}, backend: Nx.BinaryBackend) |> Nx.reshape({k, 1}))

  naive = mm32.(av, bv, 1, k, 1, naive_spv)
  f64acc = mm32.(av, bv, 1, k, 1, f64acc_spv)
  IO.puts("   K=#{String.pad_leading(to_string(k), 5)}  true=#{k - 2}  naive_f32=#{Float.round(hd(naive), 1)} (err #{Float.round(maxerr.(naive, truth), 1)})  f64acc=#{Float.round(hd(f64acc), 1)} (err #{maxerr.(f64acc, truth)})")
end

IO.puts("\n== B. Well-conditioned random matmul (realistic DL case), growing K ==")
IO.puts("   32x K . K x 32, entries ~U[-1,1]; max abs error vs f64 truth.")
for k <- [64, 512, 4096] do
  al = for i <- 1..(32 * k), do: :math.sin(i * 0.137)
  bl = for i <- 1..(k * 32), do: :math.cos(i * 0.091)
  av = Nx.tensor(al, type: {:f, 32}, backend: VulkanoBackend) |> Nx.reshape({32, k})
  bv = Nx.tensor(bl, type: {:f, 32}, backend: VulkanoBackend) |> Nx.reshape({k, 32})
  truth = Nx.dot(Nx.tensor(al, type: {:f, 64}, backend: Nx.BinaryBackend) |> Nx.reshape({32, k}),
                 Nx.tensor(bl, type: {:f, 64}, backend: Nx.BinaryBackend) |> Nx.reshape({k, 32}))
  naive = mm32.(av, bv, 32, k, 32, naive_spv)
  f64acc = mm32.(av, bv, 32, k, 32, f64acc_spv)
  IO.puts("   K=#{String.pad_leading(to_string(k), 5)}  naive_f32 err=#{Float.round(maxerr.(naive, truth), 7)}   f64acc err=#{Float.round(maxerr.(f64acc, truth), 7)}")
end

IO.puts("")
