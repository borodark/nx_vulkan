defmodule Nx.Vulkan.OwnDoctestTest do
  @moduledoc """
  This project's OWN documented examples.

  Until now the suite ran `doctest Nx` — all 833 of Nx's examples — and not a
  single one of its own. `lib/` carries ten `iex>` examples across three
  modules, and none had ever been executed. Documented examples that nothing
  runs are the cheapest possible source of wrong documentation: they look
  authoritative and rot silently.

  `Nx.Vulkan.Fast`'s doctest lived here too, until that module was deleted —
  see the CHANGELOG. `Nx.Vulkan.jit/2`'s example is deliberately NOT here. It calls
  `Nx.global_default_backend/1`, which mutates process-global state for every
  other test in the suite — a doctest with that side effect is worse than no
  doctest. It is covered by `compiler_test.exs` instead, which owns that
  concern.
  """
  use ExUnit.Case, async: false

  alias Nx.Vulkan.VulkanoBackend

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  describe "Nx.Vulkan — the top-level module, which had 0% coverage" do
    test "shader_path/1 resolves into the shipped priv/shaders" do
      path = Nx.Vulkan.shader_path("scatter.spv")
      assert String.ends_with?(path, "/priv/shaders/scatter.spv")
      assert File.exists?(path), "shader_path/1 pointed at a file that is not there"
    end

    test "jit/2 runs on the GPU, and restores the caller's backend" do
      # `jit/2` calls `Nx.global_default_backend/1`, which is why its @doc
      # example is not a doctest here: process-global mutation inside a shared
      # suite would leak into every other test. Covered explicitly instead, with
      # the previous backend put back in an `after`.
      previous = Nx.default_backend()

      try do
        got = Nx.Vulkan.jit(fn x -> Nx.add(x, x) end).(Nx.tensor([1.0, 2.0]))
        assert Nx.to_flat_list(got) == [2.0, 4.0]
        assert match?(%VulkanoBackend{}, got.data), "jit/2 should dispatch to the GPU"
      after
        Nx.global_default_backend(previous)
      end

      assert Nx.default_backend() == previous
    end
  end
end
