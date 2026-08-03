defmodule Nx.Vulkan.Fallback do
  @moduledoc """
  Counts host fallbacks so a silent one becomes a test failure.

  Every op this backend cannot run natively transfers to `Nx.BinaryBackend`,
  computes, and transfers back. That keeps results correct — the fallback *is*
  the reference implementation — which is exactly why no assertion on values can
  ever detect that an op left the GPU. A fallback is bit-identical to the GPU
  path by construction, so correctness tests are structurally blind to it, and a
  performance cliff shows up as nothing at all.

  That blindness is not hypothetical. Conv's backward pass ran entirely on the
  CPU for the whole life of the conv shaders: `Nx.Defn.Grad` emits convolutions
  with the first two axes swapped, those failed the identity-permutation gate,
  and every gradient conv fell back. The suite stayed green, the doctests stayed
  green, and a CNN training step took 30 seconds.

  This module makes the invisible thing countable:

      {result, counts} = Nx.Vulkan.Fallback.count(fn -> Nx.Defn.grad(...) end)
      assert counts == %{}

  ## Cost

  Recording is per-process and off by default. When off, the instrumentation is
  a single `Process.get/2` on a path that is already doing a device→host copy —
  unmeasurable. Attribution (which op fell back) reads the current stacktrace,
  which only happens while recording.

  ## Scope

  Instrumented at `host_result/2` in `Nx.Vulkan.VulkanoBackend`, the common exit
  point of every fallback path. Counting is process-local, so a fallback that
  happens inside another process (e.g. work funnelled through
  `Nx.Vulkan.Node`) is not counted by the caller's `count/1`.
  """

  @key :nx_vulkan_fallback_counts

  @doc """
  Run `fun` with fallback recording enabled, returning `{result, counts}` where
  `counts` maps `{function, arity}` of the backend callback that fell back to
  the number of times it did.

  Nests safely: an inner `count/1` sees only its own fallbacks, and the outer
  one resumes with its tally intact.
  """
  @spec count((-> result)) :: {result, %{{atom(), arity()} => pos_integer()}} when result: term()
  def count(fun) when is_function(fun, 0) do
    previous = Process.get(@key)
    Process.put(@key, %{})

    try do
      result = fun.()
      {result, Process.get(@key, %{})}
    after
      if previous, do: Process.put(@key, previous), else: Process.delete(@key)
    end
  end

  @doc """
  Total number of host fallbacks `fun` performs, discarding its result.

  The assertion you usually want: `assert Nx.Vulkan.Fallback.count_total(fun) == 0`.
  """
  @spec count_total((-> term())) :: non_neg_integer()
  def count_total(fun) when is_function(fun, 0) do
    {_result, counts} = count(fun)
    counts |> Map.values() |> Enum.sum()
  end

  @doc """
  Record one host fallback, attributed to `op` (a `{function, arity}` pair).

  A no-op unless the calling process is inside `count/1`, which is the normal
  state. `op` is supplied by the caller at **compile time** rather than derived
  from a stacktrace: the backend calls its fallback wrapper in tail position, so
  TCO has already discarded the frame that would name the callback.
  """
  @spec note({atom(), arity()} | atom()) :: :ok
  def note(op) do
    case Process.get(@key) do
      nil -> :ok
      counts -> Process.put(@key, Map.update(counts, op, 1, &(&1 + 1)))
    end

    :ok
  end

  @doc "Whether the calling process is currently recording."
  @spec recording?() :: boolean()
  def recording?, do: Process.get(@key) != nil
end
