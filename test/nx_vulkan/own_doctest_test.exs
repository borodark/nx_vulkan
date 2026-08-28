defmodule Nx.Vulkan.OwnDoctestTest do
  @moduledoc """
  This project's OWN documented examples.

  Until now the suite ran `doctest Nx` — all 833 of Nx's examples — and not a
  single one of its own. `lib/` carries ten `iex>` examples across three
  modules, and none had ever been executed. Documented examples that nothing
  runs are the cheapest possible source of wrong documentation: they look
  authoritative and rot silently.

  `Nx.Vulkan.jit/2`'s example is deliberately NOT here. It calls
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

  doctest Nx.Vulkan.Fast

  # Nx.Vulkan.Synthesis's "Usage" block was an `iex>` example that had NEVER
  # compiled — it referenced five undefined variables (q_ref, p_ref,
  # inv_mass_ref, q_chain, p_chain) because it was an illustrative sketch, not a
  # runnable example. Adding `doctest Nx.Vulkan.Synthesis` here is what
  # discovered that. It is now a fenced ```elixir block, which is what a sketch
  # should have been, so there is no doctest left to run.

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

  describe "Nx.Vulkan.Fast — the cross-backend guarantee, tested" do
    # The moduledoc claims: "Cross-backend correctness is guaranteed (EXLA,
    # BinaryBackend, VulkanoBackend)." That claim had ZERO tests — the module
    # measured 0.00% coverage and nothing in lib/, test/, bench/ or examples/
    # called it. Six documented public functions, one doctest between them,
    # exercising one function.
    #
    # These check the half of the guarantee this repo can check: VulkanoBackend
    # against BinaryBackend. EXLA is absent on every box but the Jetson, so the
    # third leg stays unverified here and the claim should be read accordingly.
    alias Nx.Vulkan.Fast

    defp close?(a, b) do
      # f32 tolerance: these run on the GPU at whatever precision the default
      # backend gives, and `normal_logpdf` puts a log and a divide in the chain.
      Nx.to_number(Nx.all_close(a, b, rtol: 1.0e-5, atol: 1.0e-8)) == 1
    end

    defp both(build) do
      got = build.(VulkanoBackend)
      ref = build.(Nx.BinaryBackend)

      assert close?(got, ref),
             "#{inspect(Nx.to_flat_list(got))} vs #{inspect(Nx.to_flat_list(ref))}"

      got
    end

    test "leapfrog_position / momentum_half / momentum_step" do
      v = fn l, b -> Nx.tensor(l, type: {:f, 64}, backend: b) end

      both(fn b ->
        Fast.leapfrog_position(
          v.([1.0, 2.0, -3.0], b),
          v.([0.5, 0.5, 0.5], b),
          v.([2.0, 4.0, 6.0], b)
        )
      end)

      both(fn b ->
        Fast.leapfrog_momentum_half(v.([1.0, 2.0], b), v.([0.25, 0.25], b), v.([4.0, -8.0], b))
      end)

      both(fn b ->
        Fast.momentum_step(v.([1.0, 2.0], b), v.([0.5, 0.5], b), v.([4.0, -8.0], b))
      end)
    end

    test "inv_mass_apply and kinetic_energy" do
      v = fn l, b -> Nx.tensor(l, type: {:f, 64}, backend: b) end

      both(fn b -> Fast.inv_mass_apply(v.([1.0, -2.0, 3.0], b), v.([0.5, 2.0, 0.25], b)) end)

      # Reduces to a scalar: 0.5 * sum(p^2 * inv_mass).
      # 0.5 * (1*0.5 + 4*2 + 9*0.25) = 0.5 * 10.75 = 5.375
      ke = both(fn b -> Fast.kinetic_energy(v.([1.0, -2.0, 3.0], b), v.([0.5, 2.0, 0.25], b)) end)
      assert Nx.shape(ke) == {}
      assert_in_delta Nx.to_number(ke), 5.375, 1.0e-9
    end

    test "normal_logpdf against the closed form" do
      v = fn l, b -> Nx.tensor(l, type: {:f, 64}, backend: b) end

      build = fn b ->
        Fast.normal_logpdf(
          v.([0.0, 1.0, -2.0], b),
          v.([0.0, 0.0, 0.0], b),
          v.([1.0, 1.0, 2.0], b)
        )
      end

      # `both/1` already asserts GPU vs BinaryBackend. The closed form below is
      # checked against the HOST value, not the GPU one, and the two questions
      # are separated on purpose:
      #
      #   * is the FORMULA right?              -> host vs closed form, 1.0e-9
      #   * is the GPU close enough to it?     -> both/1, rtol 1.0e-5
      #
      # Conflating them fails for the wrong reason. GLSL.std.450 has no f64
      # transcendentals, so this backend's f64 `log` is its own polynomial and
      # differs from `:math.log/1` by ~2e-9 at sigma = 2. That is a legitimate
      # implementation difference — the same species as the Kepler `sqrt` 3-ULP
      # finding — and asserting the GPU against exact math at 1e-9 would pin a
      # hardware property this test has no business pinning.
      _gpu = both(build)
      host = build.(Nx.BinaryBackend)

      # Independent of both backends: -0.5*z^2 - log(sigma) - 0.5*log(2pi).
      expected =
        for {x, mu, sigma} <- [{0.0, 0.0, 1.0}, {1.0, 0.0, 1.0}, {-2.0, 0.0, 2.0}] do
          z = (x - mu) / sigma
          -0.5 * z * z - :math.log(sigma) - 0.5 * :math.log(2 * :math.pi())
        end

      for {a, e} <- Enum.zip(Nx.to_flat_list(host), expected) do
        assert_in_delta a, e, 1.0e-9
      end
    end
  end
end
