# Nx.Vulkan.Node end-to-end demo.
#
# Boots the long-lived GPU node, synthesizes a Beta(2, 5) chain
# shader from a runtime spec, dispatches it under the node's
# serialized GenServer, and reports timings + the produced
# trajectory's first logp value (verified against the analytic
# expected value).
#
# Demonstrates the full Phase 2 contract:
#   1. Nx.Vulkan.PipelineCache.load    — restore vkPipelineCache from disk
#   2. Nx.Vulkan.Node.start_link       — boot the GenServer
#   3. Nx.Vulkan.ChainShaderSpecs.beta — fetch a family spec
#   4. Nx.Vulkan.Synthesis.compile     — render GLSL + glslangValidator
#   5. Nx.Vulkan.Node.with_node        — dispatch inside the node
#   6. Nx.Vulkan.PipelineCache.persist — write the cache back to disk
#
# Run from the nx_vulkan repo root:
#   mix run examples/gpu_node_demo.exs

defmodule GPUNodeDemo do
  @alpha 2.0
  @beta 5.0
  @n 1
  @k 32
  @eps 0.05

  def run do
    IO.puts("\n=== Nx.Vulkan.Node demo ===\n")

    # 1. Boot.
    Nx.Vulkan.Native.init()
    {:ok, _pid} = Nx.Vulkan.Node.start_link()

    IO.puts("device:  #{Nx.Vulkan.device_name()}")
    IO.puts("f64?     #{Nx.Vulkan.has_f64?()}")
    IO.puts("uuid:    #{Nx.Vulkan.PipelineCache.device_uuid_hex()}")

    # 2. Synthesize a Beta(α, β) chain shader at runtime.
    spec = Nx.Vulkan.ChainShaderSpecs.beta()
    t_synth = System.monotonic_time(:millisecond)
    {:ok, spv_path} = Nx.Vulkan.Synthesis.compile(spec)
    synth_ms = System.monotonic_time(:millisecond) - t_synth

    IO.puts("\nsynthesized Beta SPV in #{synth_ms} ms")
    IO.puts("  → #{spv_path}")
    IO.puts("  → #{File.stat!(spv_path).size} bytes")

    # 3. Build push-constants blob. The caller computes lgamma-derived
    # constants (nx_vulkan stays out of the math library business).
    logp_const = log_beta_neg(@alpha, @beta)
    push = Nx.Vulkan.ChainShaderSpecs.beta_push(@n, @k, @eps, @alpha, @beta, logp_const)

    # 4. Upload the initial state to GPU buffers.
    {:ok, q_ref} = Nx.Vulkan.upload_binary(<<0.0::little-float-32>>)
    {:ok, p_ref} = Nx.Vulkan.upload_binary(<<0.5::little-float-32>>)
    {:ok, m_ref} = Nx.Vulkan.upload_binary(<<1.0::little-float-32>>)

    # 5. Dispatch INSIDE Nx.Vulkan.Node. This is the API that any
    # client (MCMC sampler, distributed worker, etc.) uses to share
    # the pipeline cache and watchdog protection.
    t_dispatch = System.monotonic_time(:microsecond)

    {:ok, {_q_chain, _p_chain, _grad_chain, logp_chain}} =
      Nx.Vulkan.Node.with_node(fn ->
        Nx.Vulkan.Native.leapfrog_chain_synth(q_ref, p_ref, m_ref, push, @k, spv_path)
      end)

    dispatch_us = System.monotonic_time(:microsecond) - t_dispatch
    IO.puts("\nfirst dispatch via with_node: #{dispatch_us} µs")

    # 6. Read the trajectory.
    {:ok, logp_bin} = Nx.Vulkan.Native.download_binary(logp_chain, @k * 4)
    [first_logp | _] = for <<v::little-float-32 <- logp_bin>>, do: v

    expected = analytic_logp_at_q_uc_zero(@alpha, @beta, @n)
    delta = abs(first_logp - expected)

    # logp[0] is recorded AFTER the first leapfrog step, so the chain
    # has moved slightly from q_uc=0. A delta of ~0.05 is expected for
    # a single half-step + position-step + half-step at this ε.
    IO.puts("\nlogp[0]:   #{Float.round(first_logp, 4)}")
    IO.puts("logp at q_uc=0 (analytic): #{Float.round(expected, 4)}")
    IO.puts("delta after 1 leapfrog:    #{Float.round(delta, 5)} #{if delta < 0.1, do: "✓", else: "✗"}")

    # 7. Status check.
    status = Nx.Vulkan.Node.status()
    IO.puts("\nnode uptime:       #{status.uptime_ms} ms")
    IO.puts("with_node calls:   #{status.exec_count}")

    # 8. Persist the pipeline cache for the next run.
    :ok = Nx.Vulkan.PipelineCache.persist()
    IO.puts("\npipeline cache persisted to disk:")
    IO.puts("  → #{Nx.Vulkan.PipelineCache.default_path()}")
    IO.puts("  → #{File.stat!(Nx.Vulkan.PipelineCache.default_path()).size} bytes")

    IO.puts("\ndone.\n")
  end

  # log p(q_uc=0 | Beta(α, β)) on logit-uc:
  #   q = sigmoid(0) = 0.5
  #   = α·log(0.5) + β·log(0.5) - log B(α, β)
  defp analytic_logp_at_q_uc_zero(alpha, beta, n) do
    log_half = :math.log(0.5)
    n * (alpha * log_half + beta * log_half - log_beta_2_5())
  end

  defp log_beta_neg(_alpha, _beta), do: -log_beta_2_5()

  # Hardcoded log_beta(2, 5) so the demo doesn't depend on lgamma
  # (Erlang/OTP's :math lacks it; nx_vulkan stays out of the math
  # library business — see ChainShaderSpecs.beta_push/6 docs).
  #
  # log_beta(2, 5) = lgamma(2) + lgamma(5) - lgamma(7)
  #               = log(1!) + log(4!) - log(6!)
  #               = 0 + log(24) - log(720) = -log(30) ≈ -3.4012
  defp log_beta_2_5, do: -:math.log(30.0)
end

GPUNodeDemo.run()
