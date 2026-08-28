defmodule Nx.Vulkan.Synthesis do
  @moduledoc """
  Phase 1 — end-to-end shader synthesis driver.

  Render template → write GLSL to disk → compile to SPIR-V via
  `glslangValidator` → cache by content hash → return SPV path
  ready for `Nx.Vulkan.NativeV.leapfrog_chain_synth_f64/6`.

  ## Cache layout

      ~/.exmc/gpu_node/spv/{spec_hash}.spv

  where `spec_hash = :crypto.hash(:sha256, glsl_source) |> Base.encode16(case: :lower)`.

  Re-synthesizing the same spec is a hash-match → instant cache hit
  (no disk write, no glslangValidator call).

  ## Usage

      iex> spec = Nx.Vulkan.ChainShaderSpecs.beta()
      iex> {:ok, spv_path} = Nx.Vulkan.Synthesis.compile(spec)
      iex> push = Nx.Vulkan.ChainShaderSpecs.beta_push(1, 32, 0.05, 2.0, 5.0)
      iex> Nx.Vulkan.NativeV.leapfrog_chain_synth_f64(q_ref, p_ref, inv_mass_ref,
      ...>                                       push, 32, spv_path)
      {:ok, {q_chain, p_chain, grad_chain, logp_chain}}
  """

  alias Nx.Vulkan.ShaderTemplate

  @cache_dir Path.expand("~/.exmc/gpu_node/spv")

  @doc """
  Compile a `%FamilySpec{}` to SPIR-V on disk.

  Returns `{:ok, spv_path}` on success or `{:error, reason}` on
  `glslangValidator` failure.
  """
  def compile(%ShaderTemplate.FamilySpec{} = spec) do
    glsl = ShaderTemplate.render(spec)
    hash = :crypto.hash(:sha256, glsl) |> Base.encode16(case: :lower)
    spv_path = Path.join(@cache_dir, "#{hash}.spv")

    if File.exists?(spv_path) do
      {:ok, spv_path}
    else
      File.mkdir_p!(@cache_dir)
      compile_fresh(glsl, spv_path)
    end
  end

  defp compile_fresh(glsl, spv_path) do
    glsl_tmp = spv_path <> ".comp"
    File.write!(glsl_tmp, glsl)

    case System.cmd("glslangValidator", ["-V", glsl_tmp, "-o", spv_path], stderr_to_stdout: true) do
      {_output, 0} ->
        # `_ =` on purpose: the temp .comp is best-effort cleanup and a failed
        # unlink must not fail a successful compile. Explicit so that
        # :unmatched_returns stays on for the returns that DO matter.
        _ = File.rm(glsl_tmp)
        {:ok, spv_path}

      {output, code} ->
        _ = File.rm(glsl_tmp)
        {:error, %{exit: code, stderr: output, glsl_path: glsl_tmp}}
    end
  end

  @doc """
  Render + compile + return both spv_path AND the rendered GLSL source.
  Useful for debugging — when something goes wrong post-compile (e.g. a
  GPU dispatch failure), you have the source string at hand.
  """
  def compile_with_source(%ShaderTemplate.FamilySpec{} = spec) do
    glsl = ShaderTemplate.render(spec)

    case compile(spec) do
      {:ok, spv_path} -> {:ok, spv_path, glsl}
      {:error, reason} -> {:error, Map.put(reason, :glsl, glsl)}
    end
  end

  @doc """
  Clear the SPV cache directory. Useful for tests.
  """
  def clear_cache, do: File.rm_rf(@cache_dir)
end
