defmodule Nx.Vulkan do
  @moduledoc """
  Nx tensor backend on Vulkan compute (vulkano/Rust, f64-only).

  All compute dispatches go through `Nx.Vulkan.VulkanoBackend` which
  owns tensors as Rustler `ResourceArc<VulkanoTensor>` handles. The
  former C++/spirit backend has been removed.

  ## Usage

      Nx.global_default_backend(Nx.Vulkan.VulkanoBackend)

      t = Nx.tensor([1.0, 2.0, 3.0])
      Nx.sum(t)

  ## JIT

      f = fn x -> Nx.add(x, x) end
      Nx.Defn.jit(f, compiler: Nx.Defn.Evaluator).(Nx.tensor([1.0, 2.0]))
  """

  @doc """
  Returns the priv/shaders path for a given shader filename.
  """
  def shader_path(name) do
    :nx_vulkan
    |> :code.priv_dir()
    |> Path.join("shaders")
    |> Path.join(name)
  end

  @doc """
  JIT-compile a function so each op dispatches through VulkanoBackend.

  Sets `Nx.Vulkan.VulkanoBackend` as the global default backend if it
  isn't already, so tensors created inside the function land on the GPU.

      iex> f = fn x -> Nx.add(x, x) end
      iex> Nx.Vulkan.jit(f).(Nx.tensor([1.0, 2.0]))
      #Nx.Tensor<f64[2] [2.0, 4.0]>
  """
  def jit(fun, opts \\ []) do
    ensure_default_backend!()
    Nx.Defn.jit(fun, [{:compiler, Nx.Defn.Evaluator} | opts])
  end

  defp ensure_default_backend! do
    case Nx.default_backend() do
      {Nx.Vulkan.VulkanoBackend, _} ->
        :ok

      _ ->
        Nx.global_default_backend(Nx.Vulkan.VulkanoBackend)
        :ok
    end
  end
end
