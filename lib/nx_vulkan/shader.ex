defmodule Nx.Vulkan.Shader do
  @moduledoc """
  The one way to turn GLSL into a `.spv` this backend will dispatch.

  ## The boundary this draws

  `nx_vulkan` owns **the device**: invoking `glslangValidator`, the version it
  is pinned to, what counts as a valid SPIR-V module, cache policy, and what
  happens when a compile produces something a driver will refuse. A consumer
  owns **the model** — what GLSL text expresses which computation.

  So "compile this GLSL and give me something dispatchable" belongs here. Every
  constraint that makes it fail is a SPIR-V or driver constraint; the
  reproducibility claims this project makes are relative to one pinned glslang;
  and this library is what panics when the result is malformed.

  ## Why it exists

  It did not, and that had a cost. There were three glslang call sites across
  two repositories with three cache policies:

      Nx.Vulkan.Synthesis.compile/1   %FamilySpec{} -> ~/.nx_vulkan/spv
      Nx.Vulkan.Codegen.compile_cached/1  GLSL -> priv_dir(:nx_vulkan)/shader_cache
      eXMC's own compile_glsl/1       GLSL -> ~/.exmc/gpu_node/spv

  A consumer generating its own GLSL could use neither of ours. `Synthesis`
  wants a `%FamilySpec{}`, a structured spec rather than source text. `Codegen`
  takes source text but cached into **this package's own `priv` directory** —
  a dependency's install dir, shared by every application using the library and
  replaced on redeploy. So eXMC wrote a third path, and when SPIR-V validation
  was added here it did not cover them: their shaders never passed through it.

  That was an API gap, not their oversight. This module closes it.

  ## Self-healing cache

  The cache is validated on **hit**, not only on write. A corrupt `.spv`
  written before validation existed — or by any other tool sharing the
  directory — is detected, deleted and recompiled rather than returned. This is
  the difference between a fix and a fix plus an instruction to go and delete
  something by hand, and the class of corruption involved (see
  `Nx.Vulkan.Spirv`) is one that a content-addressed cache would otherwise
  serve forever.

  ## Usage

      Nx.Vulkan.Shader.compile(glsl)
      Nx.Vulkan.Shader.compile(glsl, cache_dir: Path.expand("~/.myapp/spv"))
      Nx.Vulkan.Shader.compile(glsl, cache_dir: dir, key: my_content_hash)

  Returns `{:ok, spv_path}`, or `{:error, reason}` where reason is one of:

      %{exit: integer, stderr: String.t(), glsl_path: Path.t()}
      %{invalid_spirv: String.t(), glsl_path: Path.t()}

  On failure the `.comp` is left on disk at `glsl_path` so the source can be
  inspected; on success it is removed.
  """

  @default_cache_dir Path.expand("~/.nx_vulkan/spv")

  @doc """
  Compile `glsl` to a cached, validated SPIR-V module.

  ## Options

    * `:cache_dir` — where to write. Defaults to `~/.nx_vulkan/spv`. Pass your
      own so a consumer's shaders are not mixed with this library's.
    * `:key` — cache key, without extension. Defaults to the SHA-256 of the
      GLSL, which is what makes the cache content-addressed. Override only if
      you already have a content hash and want to avoid hashing twice.
  """
  @spec compile(String.t(), keyword()) :: {:ok, Path.t()} | {:error, map()}
  def compile(glsl, opts \\ []) when is_binary(glsl) do
    cache_dir = Keyword.get(opts, :cache_dir, @default_cache_dir)

    key =
      Keyword.get_lazy(opts, :key, fn ->
        :crypto.hash(:sha256, glsl) |> Base.encode16(case: :lower)
      end)

    spv_path = Path.join(cache_dir, "#{key}.spv")

    if File.exists?(spv_path) and valid?(spv_path) do
      {:ok, spv_path}
    else
      File.mkdir_p!(cache_dir)
      compile_fresh(glsl, spv_path)
    end
  end

  @doc "The default cache directory, exposed so a caller can clear or inspect it."
  def default_cache_dir, do: @default_cache_dir

  # A cache hit is only a hit if what is cached is a SPIR-V module. Anything
  # else is deleted here so the recompile below has a clean path to write.
  defp valid?(spv_path) do
    case Nx.Vulkan.Spirv.validate_file(spv_path) do
      :ok ->
        true

      {:error, _why} ->
        _ = File.rm(spv_path)
        false
    end
  end

  defp compile_fresh(glsl, spv_path) do
    glsl_path = spv_path <> ".comp"
    File.write!(glsl_path, glsl)

    case System.cmd("glslangValidator", ["-V", glsl_path, "-o", spv_path],
           stderr_to_stdout: true
         ) do
      {_output, 0} ->
        # Exit 0 is NOT proof of a valid module: an instruction needing more
        # than 65535 words wraps SPIR-V's 16-bit word count and glslang still
        # succeeds. Validating before returning is what keeps the corrupt
        # artifact out of a content-addressed cache that would reuse it forever,
        # and out of vulkano's parser, which asserts rather than erroring.
        case Nx.Vulkan.Spirv.validate_file(spv_path) do
          :ok ->
            # `_ =` on purpose: cleanup is best-effort and a failed unlink must
            # not fail a successful compile. Explicit so :unmatched_returns stays
            # on for the returns that DO matter.
            _ = File.rm(glsl_path)
            {:ok, spv_path}

          {:error, why} ->
            _ = File.rm(spv_path)
            {:error, %{invalid_spirv: why, glsl_path: glsl_path}}
        end

      {output, code} ->
        {:error, %{exit: code, stderr: output, glsl_path: glsl_path}}
    end
  end
end
