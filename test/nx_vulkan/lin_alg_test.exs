defmodule Nx.Vulkan.LinAlgTest do
  @moduledoc """
  W3 regression suite. `Nx.LinAlg` has no GPU path here — every op routes
  through `block/4` to `Nx.BinaryBackend` — so these tests are not about
  residency. They are about the two ways a host-routed op still managed to be
  wrong on this backend, both of which shipped:

    1. `encode_scalar/2` raised `ArithmeticError` on the non-finite float
       ATOMS (`:infinity`, `:neg_infinity`, `:nan`) that `Nx.Constants` returns
       and that nx's LU pivot search uses.
    2. `block/4` transferred its args to `BinaryBackend` but left the process
       *default* backend alone, so the defn body — where the evaluator
       materialises every constant and intermediate — still ran here. LU
       returned a wrong matrix for the identity, and `solve/2` then declared a
       non-singular matrix singular.

  (2) was invisible for as long as (1) existed, because (1) crashed first. That
  is the argument for testing the values and not just the absence of a raise.
  """
  use ExUnit.Case, async: false

  setup do
    Nx.default_backend(Nx.Vulkan.VulkanoBackend)
    :ok
  end

  setup_all do
    {:ok, _} = Application.ensure_all_started(:nx_vulkan)
    :ok
  end

  # A well-conditioned symmetric positive-definite matrix, so cholesky and eigh
  # are defined on it too and one fixture serves the whole family.
  @spd [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]

  defp pair(list, opts \\ []) do
    host = Nx.tensor(list, [{:backend, Nx.BinaryBackend} | opts])
    {Nx.backend_transfer(host, Nx.Vulkan.VulkanoBackend), host}
  end

  # Run `fun` with BinaryBackend as the process default. Needed for the
  # reference side: nx routes a BinaryBackend-input LinAlg call through
  # Nx.BinaryBackend.block/4, which this backend cannot intercept, and the defn
  # body would otherwise pick up whatever default is installed.
  defp on_host(fun) do
    prev = Nx.default_backend(Nx.BinaryBackend)

    try do
      fun.()
    after
      Nx.default_backend(prev)
    end
  end

  defp assert_matches(gpu, host, tol \\ 1.0e-6) do
    g = gpu |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_flat_list()
    h = Nx.to_flat_list(host)

    assert length(g) == length(h)

    for {x, y} <- Enum.zip(g, h) do
      assert abs(x - y) <= tol, "expected #{inspect(g)} to match #{inspect(h)}"
    end
  end

  describe "non-finite float constants (W3 bug 1)" do
    # These are what Nx.Constants returns. Every numeric clause of
    # encode_scalar/2 raises on them: `s / 1.0` and `trunc(s)` alike.
    for {atom, hex} <- [
          {:infinity,
           %{{:f, 16} => "007C", {:f, 32} => "0000807F", {:f, 64} => "000000000000F07F"}},
          {:neg_infinity,
           %{{:f, 16} => "00FC", {:f, 32} => "000080FF", {:f, 64} => "000000000000F0FF"}},
          {:nan, %{{:f, 16} => "007E", {:f, 32} => "0000C07F", {:f, 64} => "000000000000F87F"}}
        ],
        type <- [{:f, 16}, {:f, 32}, {:f, 64}] do
      @atom atom
      @type_ type
      @hex hex[type]

      test "#{inspect(atom)} at #{inspect(type)} encodes to the IEEE-754 pattern" do
        t = Nx.tensor(@atom, type: @type_)
        assert %Nx.Vulkan.VulkanoBackend{} = t.data

        bin = t |> Nx.backend_transfer(Nx.BinaryBackend) |> Nx.to_binary()

        # Byte-identical to the reference, not merely "also infinite" — a
        # payload difference in a NaN is a real difference.
        assert Base.encode16(bin) == @hex
        assert bin == Nx.to_binary(Nx.tensor(@atom, type: @type_, backend: Nx.BinaryBackend))
      end
    end

    test "a non-finite atom at an integer dtype defers to BinaryBackend" do
      # No encoding exists here, so the reference decides — including if it
      # refuses. What must NOT happen is an ArithmeticError out of this backend.
      host_result =
        try do
          {:ok, Nx.to_binary(Nx.tensor(:infinity, type: {:s, 32}, backend: Nx.BinaryBackend))}
        rescue
          e -> {:raised, e.__struct__}
        end

      gpu_result =
        try do
          {:ok,
           Nx.tensor(:infinity, type: {:s, 32})
           |> Nx.backend_transfer(Nx.BinaryBackend)
           |> Nx.to_binary()}
        rescue
          e -> {:raised, e.__struct__}
        end

      assert gpu_result == host_result
    end
  end

  describe "Nx.LinAlg through block/4 (W3 bug 2)" do
    test "solve/2 on the identity — the original reproducer" do
      # `Nx.LinAlg.solve(Nx.eye(2), [1.0, 2.0])` raised ArithmeticError, and
      # after that was fixed raised "can't solve for singular matrix" on a
      # matrix that is the definition of non-singular.
      x = Nx.LinAlg.solve(Nx.eye(2), Nx.tensor([1.0, 2.0]))
      assert_matches(x, Nx.tensor([1.0, 2.0], backend: Nx.BinaryBackend))
    end

    test "lu/1 of the identity is the identity" do
      {p, l, u} = Nx.LinAlg.lu(Nx.eye(2))

      # The failing values were P = [[0,1],[1,0]], L = [[1,0],[1,1]], U = zeros.
      assert_matches(p, Nx.tensor([[1, 0], [0, 1]], backend: Nx.BinaryBackend))
      assert_matches(l, Nx.tensor([[1.0, 0.0], [0.0, 1.0]], backend: Nx.BinaryBackend))
      assert_matches(u, Nx.tensor([[1.0, 0.0], [0.0, 1.0]], backend: Nx.BinaryBackend))
    end

    # `Nx.dot(a, x)` here is matrix·vector, which has no GPU path — the fast
    # path wants rank-2 × rank-2 (MISSION §3.3.4, W8). Incidental to what this
    # test asserts, but real, so it is tagged rather than worked around.
    @tag :host_fallback_expected
    test "solve/2 reconstructs: a . solve(a, b) == b" do
      {a_g, _a_h} = pair(@spd)
      {b_g, _b_h} = pair([1.0, 2.0, 3.0])

      x = Nx.LinAlg.solve(a_g, b_g)

      assert_matches(
        Nx.dot(a_g, x),
        Nx.tensor([1.0, 2.0, 3.0], backend: Nx.BinaryBackend),
        1.0e-5
      )
    end

    test "solve/2 matches BinaryBackend" do
      {a_g, a_h} = pair(@spd)
      {b_g, b_h} = pair([1.0, 2.0, 3.0])

      assert_matches(
        Nx.LinAlg.solve(a_g, b_g),
        on_host(fn -> Nx.LinAlg.solve(a_h, b_h) end),
        1.0e-5
      )
    end

    test "determinant/1 matches BinaryBackend" do
      {a_g, a_h} = pair(@spd)

      assert_matches(
        Nx.LinAlg.determinant(a_g),
        on_host(fn -> Nx.LinAlg.determinant(a_h) end),
        1.0e-5
      )
    end

    # `Nx.LinAlg.invert/1` is NOT an `Nx.Block.*` — it composes at the Nx level
    # from solve and an identity, so with_binary_backend/1 never sees it and its
    # intermediates land here one at a time. `indexed_put/5` is the one that
    # refuses. Pre-existing and unrelated to W3; recorded here because this is
    # the test that walks into it.
    @tag :host_fallback_expected
    test "invert/1 matches BinaryBackend" do
      {a_g, a_h} = pair(@spd)
      assert_matches(Nx.LinAlg.invert(a_g), on_host(fn -> Nx.LinAlg.invert(a_h) end), 1.0e-5)
    end

    test "cholesky/1 matches BinaryBackend" do
      {a_g, a_h} = pair(@spd)
      assert_matches(Nx.LinAlg.cholesky(a_g), on_host(fn -> Nx.LinAlg.cholesky(a_h) end), 1.0e-5)
    end

    test "lu/1 matches BinaryBackend on a general matrix" do
      {a_g, a_h} = pair(@spd)
      {p_g, l_g, u_g} = Nx.LinAlg.lu(a_g)
      {p_h, l_h, u_h} = on_host(fn -> Nx.LinAlg.lu(a_h) end)

      assert_matches(p_g, p_h)
      assert_matches(l_g, l_h, 1.0e-5)
      assert_matches(u_g, u_h, 1.0e-5)
    end

    test "qr/1 matches BinaryBackend" do
      {a_g, a_h} = pair(@spd)
      {q_g, r_g} = Nx.LinAlg.qr(a_g)
      {q_h, r_h} = on_host(fn -> Nx.LinAlg.qr(a_h) end)

      assert_matches(q_g, q_h, 1.0e-5)
      assert_matches(r_g, r_h, 1.0e-5)
    end

    test "the default backend is restored after a block/4 call" do
      # with_binary_backend/1 swaps the process default for the duration. If it
      # ever stopped restoring, every subsequent tensor in the process would
      # silently land on BinaryBackend — the exact invisible-residency failure
      # this repo exists to prevent.
      assert Nx.default_backend() == {Nx.Vulkan.VulkanoBackend, []}
      _ = Nx.LinAlg.solve(Nx.eye(2), Nx.tensor([1.0, 2.0]))
      assert Nx.default_backend() == {Nx.Vulkan.VulkanoBackend, []}

      assert %Nx.Vulkan.VulkanoBackend{} = Nx.tensor([1.0, 2.0]).data
    end
  end
end
