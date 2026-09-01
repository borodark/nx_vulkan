defmodule Nx.Vulkan.Synthesis do
  @moduledoc """
  Phase 1 — end-to-end shader synthesis driver.

  Render template → write GLSL to disk → compile to SPIR-V via
  `glslangValidator` → cache by content hash → return SPV path
  ready for `Nx.Vulkan.NativeV.leapfrog_chain_synth_f64/6`.

  ## Cache layout

      ~/.nx_vulkan/spv/{spec_hash}.spv

  where `spec_hash = :crypto.hash(:sha256, glsl_source) |> Base.encode16(case: :lower)`.

  Re-synthesizing the same spec is a hash-match → instant cache hit
  (no disk write, no glslangValidator call).

  ## Usage

  A SKETCH, not a runnable example — `q_ref`, `p_ref` and `inv_mass_ref` are
  device buffers the caller already holds, and the last line binds rather than
  asserts. It was written as an `iex>` block, which made it look executable; it
  never was, and the first attempt to run this module's doctests failed to
  compile on five undefined variables. Fenced as plain code so it stops
  claiming to be something it is not.

  ```elixir
  spec = Nx.Vulkan.ChainShaderSpecs.beta()
  {:ok, spv_path} = Nx.Vulkan.Synthesis.compile(spec)
  push = Nx.Vulkan.ChainShaderSpecs.beta_push(1, 32, 0.05, 2.0, 5.0)

  {:ok, {q_chain, p_chain, grad_chain, logp_chain}} =
    Nx.Vulkan.NativeV.leapfrog_chain_synth_f64(
      q_ref, p_ref, inv_mass_ref, push, 32, spv_path
    )
  ```
  """

  alias Nx.Vulkan.ShaderTemplate

  # Under ~/.nx_vulkan, NOT ~/.exmc/gpu_node/spv, since 2026-09-01.
  #
  # This library and its downstream consumer eXMC both synthesise shaders and
  # both used that one directory — `Exmc.NUTS.CustomSynth.Compile` still does,
  # and its own comment said "same cache directory as Nx.Vulkan.Synthesis". So
  # `clear_cache/0` below, which is `File.rm_rf`, deleted THEIR cache too.
  #
  # It is not hypothetical and it is not symmetric in who notices.
  # `synthesis_test.exs` calls `clear_cache/0` in both `setup` and `on_exit`, so
  # every `mix test` in this repo wiped the directory — twice within half an
  # hour during one exmc suite run on super-io, which failed with
  #
  #     {:error, :dispatch_failed, "read spv: No such file or directory"}
  #
  # three frames deep in an unrelated test, with nothing in the message
  # suggesting another process had deleted the file. It passes in isolation
  # every time, so only a concurrent run can see it — which is exactly when
  # nobody is looking for it.
  #
  # Sharing bought nothing: the hashes are over different source text, so there
  # were never cross-project cache hits to lose.
  @cache_dir Path.expand("~/.nx_vulkan/spv")

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
