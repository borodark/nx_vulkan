#!/usr/bin/env elixir

# Diagnostic microbenchmark — single fake-leapfrog step under Vulkan.
#
# Counts how many ops in a NUTS-style leapfrog ACTUALLY auto-fuse via
# Nx.Vulkan.Compiler vs how many fall through to per-op dispatches.
# Tells us whether the timeout class in the exmc suite is:
#
#   (a) "auto-fusion gap" — most ops fall through to per-op dispatch,
#       so the leapfrog dispatches 30 shaders × 1000 steps = 30k
#       dispatches/chain. Fix: extend the chain detector.
#
#   (b) "fundamental dispatch overhead" — most ops fuse correctly but
#       per-fused-dispatch cost is still high enough that thousands
#       of steps add up. Fix: PERSISTENT_BUFFERS_PLAN iter 2–4
#       (staging buffer, transfer batching, pre-recorded command
#       buffers).
#
# Run via:
#   mix run bench/leapfrog_bench.exs

:ok = Nx.Vulkan.init()
Nx.global_default_backend(Nx.Vulkan.Backend)

defmodule LeapfrogBench do
  @moduledoc """
  A simulated leapfrog body. Real NUTS does:

      1. p_half = p + 0.5 * eps * grad(q)
      2. q_new  = q + eps * p_half / mass
      3. grad_new = ∇log_prob(q_new)
      4. p_new  = p_half + 0.5 * eps * grad_new

  Step 3 is the problem child (it's the user's defn). Steps 1, 2, 4
  are simple elementwise chains. We simulate steps 1+2 here as a
  proxy for "how well does the auto-fusion compiler catch
  leapfrog-style chains".
  """

  # Two-arg variant — fits the auto-fusion 2-input shader exactly.
  def step1_simple(q, p) do
    Nx.add(q, Nx.multiply(p, p))
  end

  # Three-arg in Nx terms: p + 0.5 * eps * grad. eps is a constant
  # tensor passed as an arg. Real leapfrog has this shape; the
  # 2-arg compiler should fall through.
  def step1_three_arg(q, p, eps) do
    Nx.add(q, Nx.multiply(p, eps))
  end

  # Realistic leapfrog body: the chain has 4 elementwise ops, all
  # binary, all on (q, p) — fits 2-arg pattern.
  def step1_real(q, p) do
    half_p = Nx.multiply(p, p)
    q_step = Nx.multiply(half_p, p)
    Nx.add(q, q_step)
  end

  def time_call(label, fun, args, iters) do
    # Warm up
    for _ <- 1..10, do: apply(fun, args)
    :erlang.garbage_collect()

    {us, _} =
      :timer.tc(fn ->
        for _ <- 1..iters, do: apply(fun, args)
      end)

    per_op = us / iters
    IO.puts("[#{label}] #{Float.round(per_op, 1)} µs/call (#{iters} iters)")
    per_op
  end
end

n = 64
q = Nx.tensor(List.duplicate(1.0, n))
p = Nx.tensor(List.duplicate(0.5, n))
eps = Nx.tensor(List.duplicate(0.01, n))

IO.puts("=" |> String.duplicate(60))
IO.puts("Leapfrog auto-fusion diagnostic")
IO.puts("=" |> String.duplicate(60))
IO.puts("")

# Measure per-call cost when going through the auto-fusion compiler.
two_arg_jit = Nx.Defn.jit(&LeapfrogBench.step1_simple/2, compiler: Nx.Vulkan.Compiler)
three_arg_jit = Nx.Defn.jit(&LeapfrogBench.step1_three_arg/3, compiler: Nx.Vulkan.Compiler)
real_jit = Nx.Defn.jit(&LeapfrogBench.step1_real/2, compiler: Nx.Vulkan.Compiler)

# Same paths through Evaluator (no fusion attempt).
two_arg_eval = Nx.Defn.jit(&LeapfrogBench.step1_simple/2, compiler: Nx.Defn.Evaluator)
three_arg_eval = Nx.Defn.jit(&LeapfrogBench.step1_three_arg/3, compiler: Nx.Defn.Evaluator)
real_eval = Nx.Defn.jit(&LeapfrogBench.step1_real/2, compiler: Nx.Defn.Evaluator)

iters = 200

IO.puts("--- 2-arg simple body — should auto-fuse ---")
fused_simple = LeapfrogBench.time_call("vulkan-fused", two_arg_jit, [q, p], iters)
eval_simple = LeapfrogBench.time_call("evaluator", two_arg_eval, [q, p], iters)
ratio_simple = eval_simple / fused_simple
IO.puts(">>> fusion saves #{Float.round(ratio_simple, 2)}x")
IO.puts("")

IO.puts("--- 3-arg body — does NOT fuse (compiler bails) ---")
fused_3 = LeapfrogBench.time_call("vulkan-fused", three_arg_jit, [q, p, eps], iters)
eval_3 = LeapfrogBench.time_call("evaluator", three_arg_eval, [q, p, eps], iters)
ratio_3 = eval_3 / fused_3
IO.puts(">>> fusion saves #{Float.round(ratio_3, 2)}x (expected ~1.0 — falls through)")
IO.puts("")

IO.puts("--- Real-shape body (q + p*p*p) — should fuse ---")
fused_real = LeapfrogBench.time_call("vulkan-fused", real_jit, [q, p], iters)
eval_real = LeapfrogBench.time_call("evaluator", real_eval, [q, p], iters)
ratio_real = eval_real / fused_real
IO.puts(">>> fusion saves #{Float.round(ratio_real, 2)}x")
IO.puts("")

# Also a baseline for raw single-op cost — just to see the floor.
{:ok, a_ref} = Nx.Vulkan.upload_f32(List.duplicate(1.0, n))
{:ok, b_ref} = Nx.Vulkan.upload_f32(List.duplicate(0.5, n))

raw_fn = fn -> Nx.Vulkan.add(a_ref, b_ref) end
{us, _} = :timer.tc(fn -> for _ <- 1..iters, do: raw_fn.() end)
IO.puts("[raw single-op add] #{Float.round(us / iters, 1)} µs/call")
IO.puts("")

IO.puts("=" |> String.duplicate(60))
IO.puts("Conclusions to draw from the numbers:")
IO.puts("=" |> String.duplicate(60))
IO.puts("")
IO.puts("1. If 2-arg fused × ratio ≥ 2x: auto-fusion is doing real work.")
IO.puts("   If ratio ~ 1x: the chain detector isn't firing — investigate.")
IO.puts("")
IO.puts("2. 3-arg ratio expected ~1x: confirms current limitation —")
IO.puts("   real exmc leapfrogs (which pass eps as an arg) fall through.")
IO.puts("   Lifting this to fused dispatch is the highest-leverage fix.")
IO.puts("")
IO.puts("3. fused_simple per-call cost vs raw_op cost: shows JIT overhead")
IO.puts("   from compile/trace per call. If much higher, persistent buffers")
IO.puts("   or pre-recorded command buffers help.")
