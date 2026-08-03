defmodule Nx.Vulkan.Node do
  @moduledoc """
  Long-lived per-machine GPU node. A named `GenServer` that serializes
  GPU work and owns the watchdog/timeout layer. Clients submit work via
  `with_node/2` (or the lower-level `exec/2`); the node runs it one call
  at a time and reports timeouts/dead-server/etc. as error tuples.

  Serialization is the point: concurrent processes dispatching to one
  device is how you get descriptor-pool exhaustion and interleaved
  push-constant state. Funnelling through a single process makes GPU
  access a queue.

  It no longer owns a pipeline cache — that belonged to the removed
  C++/spirit backend, and vulkano manages its own caching internally.
  `Nx.Vulkan.PipelineCache` survives as a no-op stub, and `init/1`
  still calls it for callers that pass `load_pipeline_cache: false`.

  This is the GPU-only generic core. MCMC / NUTS / sampler-specific
  dispatch logic lives in `Exmc.NUTS.Vulkan.Dispatch` (or any other
  client) and calls into this node via `with_node/2`.

  ## Lifecycle

  Start under your application's supervisor:

      children = [
        {Nx.Vulkan.Node, []}
      ]

  Or for ad-hoc use (REPL, benchmarks):

      {:ok, _pid} = Nx.Vulkan.Node.start_link()

  ## Generic dispatch

  `with_node/2` runs an arbitrary 0-arity function inside the GenServer
  process, so every caller's GPU work is serialized against every other
  caller's. The function may stash per-shader buffer state in the process
  dictionary, which persists for the life of the node.

      result = Nx.Vulkan.Node.with_node(fn ->
        # any GPU work — uses Nx.Vulkan.NativeV NIFs directly,
        # serialized through this GenServer process.
        Nx.Vulkan.NativeV.leapfrog_chain_synth_f64(q, p, m, push, k, spv)
      end)

  Returns the function's return value, or `{:error, reason}` on
  watchdog timeout / dead node.

  ## Watchdog

  Reads `Application.get_env(:nx_vulkan, :node_timeout_ms, :infinity)`.
  On timeout the calling process gets `{:error, :node_timeout}` and is
  free to fall back to a CPU/EXLA path. The GenServer process itself
  remains blocked on the in-flight NIF call until that returns — by
  design; cancelling Vulkan dispatches mid-flight is unsafe.
  """

  use GenServer

  @name __MODULE__

  ## API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: opts[:name] || @name)
  end

  @doc "Whether the named node is alive."
  def alive?(name \\ @name) do
    case Process.whereis(name) do
      nil -> false
      pid -> Process.alive?(pid)
    end
  end

  @doc """
  Run a 0-arity function inside the node's GenServer process. Used for
  any GPU work that needs to share the pipeline cache and buffer state
  with other callers.

  Returns the function's return value, or:
    * `{:error, :node_timeout}` if the call exceeded `:nx_vulkan/:node_timeout_ms`
    * `{:error, :node_dead}` if no node is registered under `name`
  """
  def with_node(fun, name \\ @name) when is_function(fun, 0) do
    timeout = Application.get_env(:nx_vulkan, :node_timeout_ms, :infinity)

    try do
      GenServer.call(name, {:exec, fun}, timeout)
    catch
      :exit, {:timeout, _} -> {:error, :node_timeout}
      :exit, {:noproc, _} -> {:error, :node_dead}
    end
  end

  @doc """
  Quick status read — uptime + total exec count.
  """
  def status(name \\ @name), do: GenServer.call(name, :status)

  ## GenServer callbacks

  @impl true
  def init(opts) do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)

    if Keyword.get(opts, :load_pipeline_cache, true) do
      _ = Nx.Vulkan.PipelineCache.load()
    end

    {:ok,
     %{
       started_at: System.monotonic_time(:millisecond),
       exec_count: 0
     }}
  end

  @impl true
  def handle_call({:exec, fun}, _from, state) do
    result = fun.()
    {:reply, result, %{state | exec_count: state.exec_count + 1}}
  end

  @impl true
  def handle_call(:status, _from, state) do
    uptime_ms = System.monotonic_time(:millisecond) - state.started_at

    {:reply,
     %{
       uptime_ms: uptime_ms,
       exec_count: state.exec_count
     }, state}
  end

  @impl true
  def terminate(_reason, _state) do
    # Persist the pipeline cache on graceful shutdown so the next
    # process gets the warm cache.
    _ = Nx.Vulkan.PipelineCache.persist()
    :ok
  end
end
