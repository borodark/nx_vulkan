defmodule Nx.Vulkan.ShaderTest do
  @moduledoc """
  The public GLSL -> validated SPIR-V entry point, and the boundary it draws.

  These are the guarantees a CONSUMER generating its own GLSL depends on, so
  they are asserted rather than described: a chosen cache directory is honoured,
  a corrupt cache entry repairs itself instead of being served forever, and a
  compile that fails leaves the source on disk to look at.
  """
  use ExUnit.Case, async: true

  alias Nx.Vulkan.Shader

  @glsl """
  #version 450
  layout(local_size_x = 64) in;
  layout(std430, binding = 0) readonly buffer A { float a[]; };
  layout(std430, binding = 1) writeonly buffer O { float o[]; };
  layout(push_constant) uniform Push { uint n; } p;
  void main() {
      uint i = gl_GlobalInvocationID.x;
      if (i >= p.n) return;
      o[i] = a[i] * 3.0;
  }
  """

  setup do
    dir = Path.join(System.tmp_dir!(), "nxv_shader_test_#{System.unique_integer([:positive])}")
    on_exit(fn -> File.rm_rf(dir) end)
    {:ok, dir: dir}
  end

  describe "compiling into a caller's own directory" do
    test "writes where it is told and returns a valid module", %{dir: dir} do
      assert {:ok, spv} = Shader.compile(@glsl, cache_dir: dir)
      assert Path.dirname(spv) == dir
      assert Nx.Vulkan.Spirv.validate_file(spv) == :ok
      # The .comp is cleaned up on success; only the artifact remains.
      assert Path.wildcard(Path.join(dir, "*.comp")) == []
    end

    test "the default cache is not the package's priv directory", %{dir: _} do
      # The bug this module was written to fix: caching into a dependency's own
      # install dir, shared across applications and replaced on redeploy.
      refute Shader.default_cache_dir() =~ "priv"
      refute Shader.default_cache_dir() =~ :code.priv_dir(:nx_vulkan) |> to_string()
    end

    test "content-addressed — same GLSL twice is one artifact", %{dir: dir} do
      assert {:ok, a} = Shader.compile(@glsl, cache_dir: dir)
      assert {:ok, b} = Shader.compile(@glsl, cache_dir: dir)
      assert a == b
      assert length(Path.wildcard(Path.join(dir, "*.spv"))) == 1
    end

    test "an explicit key is honoured", %{dir: dir} do
      assert {:ok, spv} = Shader.compile(@glsl, cache_dir: dir, key: "my_own_name")
      assert Path.basename(spv) == "my_own_name.spv"
    end
  end

  describe "the cache repairs itself" do
    test "a corrupt entry is deleted and recompiled, not served", %{dir: dir} do
      assert {:ok, spv} = Shader.compile(@glsl, cache_dir: dir)
      good = File.read!(spv)

      # Exactly the corruption glslang produces on a word-count wrap: a valid
      # header followed by an instruction declaring zero words. vulkano asserts
      # on this, so serving it from cache is how one bad compile poisons every
      # later run.
      File.write!(spv, <<0x07230203::little-32, 0x00010600::little-32, 8::little-32,
                         1::little-32, 0::little-32, 0::little-32>>)
      refute Nx.Vulkan.Spirv.validate_file(spv) == :ok

      assert {:ok, ^spv} = Shader.compile(@glsl, cache_dir: dir)
      assert Nx.Vulkan.Spirv.validate_file(spv) == :ok
      assert File.read!(spv) == good, "recompiled artifact should match the original"
    end

    test "the null arm — a VALID cache entry is served untouched", %{dir: dir} do
      assert {:ok, spv} = Shader.compile(@glsl, cache_dir: dir)
      File.touch!(spv, {{2001, 1, 1}, {0, 0, 0}})
      before = File.stat!(spv, time: :posix).mtime

      assert {:ok, ^spv} = Shader.compile(@glsl, cache_dir: dir)

      assert File.stat!(spv, time: :posix).mtime == before,
             "a valid entry was recompiled — the hit path is not actually a hit"
    end
  end

  describe "failure" do
    test "bad GLSL reports glslang's own message and KEEPS the source", %{dir: dir} do
      assert {:error, %{exit: code, stderr: out, glsl_path: path}} =
               Shader.compile("#version 450\nthis is not glsl\n", cache_dir: dir)

      assert code != 0
      assert is_binary(out) and out != ""
      assert File.exists?(path), "the .comp must survive a failure or there is nothing to debug"
      assert File.read!(path) =~ "not glsl"
    end

    test "no .spv is left behind by a failed compile", %{dir: dir} do
      assert {:error, _} = Shader.compile("#version 450\nnope\n", cache_dir: dir)
      assert Path.wildcard(Path.join(dir, "*.spv")) == []
    end
  end
end
