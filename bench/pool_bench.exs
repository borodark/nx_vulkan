#!/usr/bin/env elixir

# Pool win benchmark — Day 1 (PATH_TO_FULL_PASS step 1a).
#
# Measures `Nx.Vulkan.add` per-op steady-state cost with the buffer
# pool engaged (default) vs forcibly disabled (pool_clear between ops).
# The delta is the alloc-overhead the pool eliminates.
#
# Run via:
#   mix run bench/pool_bench.exs

:ok = Nx.Vulkan.init()

defmodule PoolBench do
  def warm_up(n_elems, iters) do
    {:ok, a} = Nx.Vulkan.upload_f32(List.duplicate(1.0, n_elems))
    {:ok, b} = Nx.Vulkan.upload_f32(List.duplicate(2.0, n_elems))
    for _ <- 1..iters, do: Nx.Vulkan.add(a, b)
    :ok
  end

  def measure(label, n_elems, iters, clear_between?) do
    {:ok, a} = Nx.Vulkan.upload_f32(List.duplicate(1.0, n_elems))
    {:ok, b} = Nx.Vulkan.upload_f32(List.duplicate(2.0, n_elems))

    Nx.Vulkan.pool_clear()
    {:ok, before_stats} = Nx.Vulkan.pool_stats()

    # Realistic shape: many short "chains" of ops, with GC between
    # chains. Each chain runs a few ops in sequence (intermediates
    # accumulate), then we drop the result and let GC reclaim. This
    # matches how a NUTS sampler structures its leapfrog steps —
    # within a step, intermediates pile up; between steps, they drop.
    chain_len = 5
    n_chains = div(iters, chain_len)

    {us, _} =
      :timer.tc(fn ->
        for _ <- 1..n_chains do
          # Build a chain: c1 = a+b, c2 = c1+b, c3 = c2+b, ...
          _final =
            Enum.reduce(1..chain_len, a, fn _, acc ->
              {:ok, c} = Nx.Vulkan.add(acc, b)
              c
            end)

          # End of chain — drop the result, GC, give pool a chance.
          :erlang.garbage_collect()
          if clear_between?, do: Nx.Vulkan.pool_clear()
        end
      end)

    :erlang.garbage_collect()
    {:ok, after_stats} = Nx.Vulkan.pool_stats()

    per_op_us = us / iters
    delta_misses = after_stats.misses - before_stats.misses
    delta_hits = after_stats.hits - before_stats.hits
    hit_rate =
      if delta_hits + delta_misses == 0,
        do: 0.0,
        else: delta_hits / (delta_hits + delta_misses)

    IO.puts("""
    [#{label}] N=#{n_elems}, iters=#{iters}
      total wall:  #{Float.round(us / 1000, 2)} ms
      per op:      #{Float.round(per_op_us, 2)} µs
      pool hits:   #{delta_hits}
      pool misses: #{delta_misses}
      hit rate:    #{Float.round(hit_rate * 100, 1)}%
    """)

    per_op_us
  end
end

# Two warmups to compile pipelines and prime any first-call costs.
PoolBench.warm_up(1024, 32)
PoolBench.warm_up(1024, 32)

IO.puts("=" |> String.duplicate(60))
IO.puts("Buffer pool benchmark — Nx.Vulkan.add steady-state per-op cost")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

for n <- [1024, 16_384, 262_144, 1_048_576] do
  pooled = PoolBench.measure("pooled", n, 200, false)
  unpooled = PoolBench.measure("unpooled", n, 200, true)
  ratio = unpooled / pooled
  IO.puts(">>> N=#{n}: pooled #{Float.round(pooled, 2)} µs, " <>
          "unpooled #{Float.round(unpooled, 2)} µs, " <>
          "ratio #{Float.round(ratio, 2)}x")
  IO.puts("")
end

Nx.Vulkan.pool_clear()
IO.puts("[pool cleared]")
