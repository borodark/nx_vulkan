defmodule Nx.Vulkan.HostFallbackError do
  @moduledoc """
  Raised when a host fallback happens under `host_fallback: :raise` and the op
  is not on `Nx.Vulkan.Fallback`'s allowlist.

  See `Nx.Vulkan.Fallback` for the mode, the allowlist, and how to scope strict
  mode to a single block.
  """
  defexception [:message, :op, :shape, :type]
end

defmodule Nx.Vulkan.Fallback do
  @moduledoc """
  Counts host fallbacks so a silent one becomes a test failure.

  Every op this backend cannot run natively transfers to `Nx.BinaryBackend`,
  computes, and transfers back. That keeps results correct — the fallback *is*
  the reference implementation — which is exactly why no assertion on values can
  ever detect that an op left the GPU. A fallback is bit-identical to the GPU
  path by construction, so correctness tests are structurally blind to it, and a
  performance cliff shows up as nothing at all.

  That blindness is not hypothetical. Conv's backward pass ran entirely on the
  CPU for the whole life of the conv shaders: Nx.Defn.Grad (hidden, so not
  linked) emits convolutions with the first two axes swapped, those failed the
  identity-permutation gate, and every gradient conv fell back. The suite stayed green, the doctests stayed
  green, and a CNN training step took 30 seconds.

  This module makes the invisible thing countable:

      {result, counts} = Nx.Vulkan.Fallback.count(fn -> Nx.Defn.grad(...) end)
      assert counts == %{}

  ## Cost

  Recording is per-process and off by default. When off, the instrumentation is
  a single `Process.get/2` on a path that is already doing a device→host copy —
  unmeasurable. Attribution (which op fell back) reads the current stacktrace,
  which only happens while recording.

  ## Scope

  Instrumented at `host_result/2` in `Nx.Vulkan.VulkanoBackend`, the common exit
  point of every fallback path. Counting is process-local, so a fallback that
  happens inside another process (e.g. work funnelled through
  `Nx.Vulkan.Node`) is not counted by the caller's `count/1`.

  ## The count is a lower bound

  Only ops that reach *this* backend can be counted. Once a fallback puts a
  tensor on `Nx.BinaryBackend`, Nx dispatches every subsequent op on that tensor
  straight to `Nx.BinaryBackend` — this module's callbacks are never invoked, so
  the downstream host work is invisible here.

  That is not theoretical. A LeNet training step reported no `window_max` at all
  while `dot/7` was still falling back: the dot dropped the pooling gradient
  onto the host, and everything after it ran there unseen. Fixing `dot` kept the
  tensor resident, and `window_scatter_max/6` promptly appeared in the census.

  So a count going *up* after a fix can mean the fix worked and exposed
  something that was already happening. Read the composition, not just the
  total.

  ## Strict mode

      config :nx_vulkan, host_fallback: :allow | :warn | :raise

  `:allow` is the default and is today's behaviour: a fallback is correct, just
  slower, and "it always works" is this backend's main property. `:warn` logs
  each refused fallback and returns the same (correct) answer. `:raise` raises
  `Nx.Vulkan.HostFallbackError` on the first fallback that is not on the
  allowlist below.

  Scope it to a block instead of the whole VM — the sibling of `count/1`, and
  what a test should reach for:

      Nx.Vulkan.Fallback.strict(fn -> Nx.Defn.jit_apply(step, args) end)
      Nx.Vulkan.Fallback.strict(:warn, fn -> ... end)

  Strict mode is **per-process**, like counting, so one strict test cannot
  poison an `async: true` suite. The application config sets the default for
  processes that have not scoped it.

  ### Why raising beats counting

  The count above is a *lower bound*: once a fallback strands a tensor on
  `Nx.BinaryBackend`, everything downstream computes there without reaching
  this backend at all, so the census shows the visible edge of a cascade rather
  than its cause. `:raise` fires at the **first** refused fallback — before the
  tensor has left the device — so a strict run names the op that actually
  started it. That is the whole reason to prefer it to an assertion on
  `count_total/1`.

  ### The allowlist

  `@allowlist` is the entire risk surface of strict mode. An allowlist that
  grows loosely makes `:raise` mean nothing, which is the same failure that
  produced the narrow forward-pass gates in the first place
  (`docs/BACKWARD_PASS_AUDIT.md` §1). So:

    * every entry is one line, names one `{fun, arity}`, and carries the reason
      it is permitted — a new exemption is legible in a diff;
    * there are **no wildcards and no op families** — `{:transpose, 3}` is not
      exempt, only `{:transpose, 3}` *at rank ≥ 5* is, because a rank-4
      transpose falling back is a bug and must still raise;
    * "it made CI red" is not a reason. Widen the gate instead.

  ### What strict mode cannot see

  Everything reaches the funnel as of T13. `block/4` — the callback nx 0.13
  routes `Nx.LinAlg` (svd/qr/lu/cholesky/solve/eigh/determinant), `top_k`,
  `cumulative_*`, `take` and `all_close` through — used to transfer to
  `Nx.BinaryBackend` without passing through it, so that whole family was
  invisible to `count/1` and to `:raise` alike and a green strict run said
  nothing about any of it. It is now recorded as `{:block, Nx.Block.Foo}`,
  attributed per struct so that a missing `cumulative_sum` shader and
  `Nx.all_close` are separately decidable.

  Worth knowing what that revealed: a single `Nx.LinAlg.svd/2` records around
  350 fallbacks, because nx composes it from ordinary ops whose intermediates
  land back on this backend one at a time — 51 `pow`, 38 `dot`, 35
  `concatenate`, plus nested `Cholesky`, `Eigh` and 17 `Take` blocks. Scholar's
  linear regression goes through it. "One host fallback" was never the shape of
  this cost.
  """

  require Logger

  @key :nx_vulkan_fallback_counts
  @strict_key :nx_vulkan_fallback_strict
  @modes [:allow, :warn, :raise]

  @typedoc """
  When an allowlist entry applies. Every form here must have a matching
  `condition_met?/2` clause — one that does not raises `FunctionClauseError` at
  the first refused op, under `:raise`, in somebody else's suite.

  This spec listed only the first two forms for a day after `{:dtype, _}` was
  added; `mix dialyzer` with `:specdiffs` is what noticed, and
  `test/nx_vulkan/fallback_test.exs` now asserts it without needing that flag.
  """
  @type condition ::
          :always
          | {:rank_at_least, pos_integer()}
          | {:dtype, Nx.Type.t()}
          | :float_output

  @doc """
  Run `fun` with fallback recording enabled, returning `{result, counts}` where
  `counts` maps `{function, arity}` of the backend callback that fell back to
  the number of times it did.

  Nests safely: an inner `count/1` sees only its own fallbacks, and the outer
  one resumes with its tally intact.
  """
  @spec count((-> result)) :: {result, %{{atom(), arity()} => pos_integer()}} when result: term()
  def count(fun) when is_function(fun, 0) do
    previous = Process.get(@key)
    Process.put(@key, %{})

    try do
      result = fun.()
      {result, Process.get(@key, %{})}
    after
      if previous, do: Process.put(@key, previous), else: Process.delete(@key)
    end
  end

  @doc """
  Total number of host fallbacks `fun` performs, discarding its result.

  The assertion you usually want: `assert Nx.Vulkan.Fallback.count_total(fun) == 0`.
  """
  @spec count_total((-> term())) :: non_neg_integer()
  def count_total(fun) when is_function(fun, 0) do
    {_result, counts} = count(fun)
    counts |> Map.values() |> Enum.sum()
  end

  @doc """
  Record one host fallback, attributed to `op` (a `{function, arity}` pair).

  `meta` is the op's output template (an `Nx.Tensor`, or `nil` when the
  callback has no single one); it is only read when strict mode has something
  to report, so `:allow` pays nothing for it.

  Counting is a no-op unless the calling process is inside `count/1`, which is
  the normal state. `op` is supplied by the caller at **compile time** rather
  than derived from a stacktrace: the backend calls its fallback wrapper in
  tail position, so TCO has already discarded the frame that would name the
  callback.
  """
  @spec note({atom(), arity() | module()} | atom()) :: :ok
  @spec note({atom(), arity() | module()} | atom(), Nx.Tensor.t() | nil) :: :ok
  def note(op, meta \\ nil) do
    case mode() do
      :allow -> :ok
      strict -> enforce(strict, op, meta)
    end

    case Process.get(@key) do
      nil -> :ok
      counts -> Process.put(@key, Map.update(counts, op, 1, &(&1 + 1)))
    end

    :ok
  end

  @doc "Whether the calling process is currently recording."
  @spec recording?() :: boolean()
  def recording?, do: Process.get(@key) != nil

  # ------------------------------------------------------------ strict mode

  @doc """
  Run `fun` with host fallbacks strict in the calling process only.

  `strict/1` uses `:raise`. Restores the previous mode afterwards, so it nests
  the same way `count/1` does, and never affects another process.

      Nx.Vulkan.Fallback.strict(fn -> Nx.Defn.jit_apply(step, args) end)
      Nx.Vulkan.Fallback.strict(:warn, fn -> maybe_slow_thing() end)
      Nx.Vulkan.Fallback.strict(:allow, fn -> known_host_op() end)
  """
  @spec strict((-> result)) :: result when result: term()
  def strict(fun) when is_function(fun, 0), do: strict(:raise, fun)

  @spec strict(:allow | :warn | :raise, (-> result)) :: result when result: term()
  def strict(mode, fun) when mode in @modes and is_function(fun, 0) do
    previous = Process.get(@strict_key)
    Process.put(@strict_key, mode)

    try do
      fun.()
    after
      if previous, do: Process.put(@strict_key, previous), else: Process.delete(@strict_key)
    end
  end

  @doc """
  The mode in effect for the calling process.

  A `strict/2` scope wins; otherwise `config :nx_vulkan, host_fallback:`;
  otherwise `:allow`.

  The application lookup is an ETS read, and it happens only on a path that is
  by construction about to copy a tensor off the device — a GPU-resident op
  never reaches it. There is no per-op cost on the fast path.
  """
  @spec mode() :: :allow | :warn | :raise
  def mode do
    case Process.get(@strict_key) do
      nil -> Application.get_env(:nx_vulkan, :host_fallback, :allow)
      mode -> mode
    end
  end

  # Ops permitted to fall back. Each line: {fun, arity}, a condition, and the
  # reason. Adding one is a deliberate, reviewable act — read the "allowlist"
  # section of the moduledoc before you do.
  #
  # Conditions:
  #   :always              — this callback has no GPU path at all
  #   {:rank_at_least, n}  — permitted only from rank n up; below that it is a bug
  #   {:dtype, type}       — permitted only at exactly that output dtype. Use
  #                          this rather than :float_output when a gap is
  #                          dtype-specific: :float_output would also excuse
  #                          the dtypes that DO run on the GPU, so a real
  #                          regression there would pass silently.
  #   :float_output        — permitted only when the OUTPUT is a float type; an
  #                          integer result means the reason does not apply
  @allowlist [
    # Was `:float_output` until 2026-09-01, covering f32 and f64 together. f32
    # broadcasting pow now runs on the GPU — GLSL.std.450 has a native f32
    # `Pow`, so the old entry excluded it on an f64 limitation and cost
    # precision nothing — and narrowing to `{:dtype, {:f, 64}}` means an f32
    # regression here would now RAISE instead of being quietly excused.
    #
    # f64 stays on the host deliberately. MISSION.md §3.2 records the decision:
    # the only way onto the GPU is boundary-casting through f32, "trading real
    # precision for a nicer table". `Nx.pow(f64, 0.5)` on 3.0 is
    # 1.7320508075688772 here and would be 1.7320507764816284 boundary-cast.
    #
    # INTEGER pow remains uncovered and is still admitted by DATA, not type —
    # see `nonneg_exponent?/1` and the note below.
    {{:pow, 3}, {:dtype, {:f, 64}},
     "f64 broadcasting/scalar-exponent pow: GLSL.std.450 has no f64 Pow and " <>
       "boundary-casting through f32 costs ~9 digits. MISSION.md §3.2 declines " <>
       "that trade. f32 broadcasting pow and equal-shape pow at both dtypes do " <>
       "run on the GPU."},
    {{:window_scatter_max, 6}, :always,
     "OVERLAPPING pooling backward only. One thread per input element is what " <>
       "avoids float atomics, and that only holds for non-overlapping windows; " <>
       "the overlapping case needs GL_EXT_shader_atomic_float, which the Kepler " <>
       "fleet does not guarantee. Non-overlapping runs on the GPU."},
    {{:reduce, 5}, :always,
     "arbitrary user fun, folded sequentially. A shader cannot express it, and " <>
       "the obvious workaround is SLOWER than the host path, which is why this " <>
       "is a decision rather than a gap. Vectorising the fold — one dispatch " <>
       "per step along the reduced axis, evaluating the fun on resident " <>
       "tensors — was prototyped and measured on super-io: 0.97ms vs 0.19ms at " <>
       "reduce_size 8, 39.8 vs 22.0 at 512, and 441ms vs 37ms at 4096. The gap " <>
       "widens with the axis. The mechanism is NOT launch overhead, though this " <>
       "entry said so until DTrace measured it (docs/DTRACE_VULKAN_PROFILING.md): " <>
       "dispatches batch 64 to a command buffer, so 4096 of them are ~64 queue " <>
       "waits, about 11ms against a measured 441ms. What scales is per-dispatch " <>
       "WORK - descriptor sets, buffer churn, GPU execution. The conclusion is " <>
       "unchanged; the reason was wrong. No implementation removes it without " <>
       "assuming the fun is " <>
       "ASSOCIATIVE (a log2-step tree reduce), which Nx.reduce does not " <>
       "guarantee — it is a left fold. Probing the fun to recognise `add` is " <>
       "the other tempting shortcut and is unsound for the same reason: a fun " <>
       "that agrees on probe values can differ elsewhere."},
    {{:sort, 3}, :always, "no sort shader and no plan for one; the host path is correct"},
    {{:argsort, 3}, :always, "no sort shader and no plan for one; the host path is correct"},
    {{:triangular_solve, 4}, :always,
     "the one Nx.LinAlg op still an Nx.Backend callback; a GPU solver is a " <>
       "project, and the host path is correct"},
    {{:transpose, 3}, {:rank_at_least, 5},
     "transpose_nd handles rank <= 4; rank 5+ is mechanical to add if a workload appears"},
    {{:reverse, 3}, {:rank_at_least, 5},
     "reverse_nd handles rank <= 4; rank 5+ is mechanical to add if a workload appears"},
    {{:broadcast, 4}, {:rank_at_least, 5},
     "broadcast_nd handles rank <= 4; rank 5+ is mechanical to add if a workload appears"},

    # --- block/4, keyed per Nx.Block struct (T13) -------------------------
    #
    # These are `{:block, Module}` rather than `{fun, arity}` deliberately. A
    # single `{:block, 4}` entry would exempt the whole family in one line —
    # the op-family wildcard this list forbids — and would silence a missing
    # `cumulative_sum` shader and `Nx.all_close` together. Per struct, each
    # carries its own reason and each can be deleted on its own.
    #
    # Only the *decided* ones are here. As of W4 all 21 `Nx.Block.*` structs in
    # nx 0.13 are decided: 13 allowlisted below, the other 8 routed on-device by
    # `VulkanoBackend.@device_blocks` so their constituent ops report for
    # themselves. A new `Nx.Block.*` still lands here undecided and still
    # raises, which is the intended default.
    {{:block, Nx.Block.LinAlg.SVD}, :always,
     "dense GPU SVD is iterative, convergence-sensitive, and awkward to make " <>
       "bit-reproducible across the fleet, which is a documented property here. " <>
       "ROADMAP withdraws any estimate for it. Correct on the host today."},
    {{:block, Nx.Block.LinAlg.QR}, :always,
     "Householder QR is sequential in the reflector loop; a GPU version is a " <>
       "project rather than a kernel, and nothing here has asked for one"},
    {{:block, Nx.Block.LinAlg.LU}, :always,
     "pivoted LU needs a device-side pivot search and row swaps per column; " <>
       "the host path is correct and no workload has made it hot"},
    {{:block, Nx.Block.LinAlg.Eigh}, :always,
     "symmetric eigendecomposition is iterative and convergence-sensitive, the " <>
       "same class of problem as SVD and with the same reproducibility concern"},
    {{:block, Nx.Block.LinAlg.Cholesky}, :always,
     "the factorisation is inherently sequential down the diagonal; at the " <>
       "matrix sizes this project sees, the host path is not the bottleneck"},
    {{:block, Nx.Block.LinAlg.Solve}, :always,
     "solve composes LU with two triangular solves, so it inherits their " <>
       "decisions; triangular_solve/4 is already allowlisted above for the same reason"},
    {{:block, Nx.Block.LinAlg.Determinant}, :always,
     "determinant is LU plus a product of the diagonal, so it is exactly as " <>
       "GPU-able as LU is, which is to say not yet and not urgently"},
    {{:block, Nx.Block.AllClose}, :always,
     "returns a scalar boolean from a comparison plus a reduction; there is no " <>
       "kernel worth writing, and it is this suite's own assertion helper — " <>
       "raising on it would make strict mode unusable in the tests that need it"},
    {{:block, Nx.Block.Phase}, :always,
     "complex-only, and the shader ISA is real-valued. LIMITATIONS.md lists it " <>
       "as a permanent skip: there is nothing to implement, not merely nothing done"},

    # --- W4: the four that had nothing to route to ------------------------
    #
    # The other eight undecided blocks are NOT here, and that is the decision
    # rather than an omission: `VulkanoBackend.@device_blocks` evaluates their
    # bodies on this backend, so `Nx.take/3` at axis 0 and float
    # `Nx.logical_not/1` are fully resident, and the rest report the op that is
    # actually missing (`concatenate/3`, `gather/4` off-prefix, integer
    # `equal/3`) instead of an opaque block. `Nx.top_k/2` reports `argsort/3`,
    # already allowlisted above — routing is what makes it inherit that
    # decision honestly rather than restate it.
    #
    # These four cannot do that. Their bodies are complex-valued throughout, so
    # routing would report `do_fft/4` — a rename of the same wall, not a gap
    # anyone can close while the ISA is real-valued.
    {{:block, Nx.Block.FFT2}, :always,
     "complex-valued, like Phase above; the shader ISA is real and the f64 " <>
       "FFT shaders here serve the real-input path only"},
    {{:block, Nx.Block.IFFT2}, :always,
     "the inverse of FFT2 and blocked on exactly the same thing"},
    {{:block, Nx.Block.RFFT}, :always,
     "real input, but complex output and a complex-valued body; blocked on " <>
       "complex dtype support, not on a kernel"},
    {{:block, Nx.Block.IRFFT}, :always,
     "complex input; the mirror of RFFT and blocked on the same dtype gap"}
  ]

  @doc """
  The strict-mode allowlist as `[{{fun, arity}, condition, reason}]`.

  Exposed so a test can assert on it — an allowlist that only exists in a
  module attribute is one nobody reviews.
  """
  @spec allowlist() :: [{{atom(), arity() | module()}, condition(), String.t()}]
  def allowlist, do: @allowlist

  @doc """
  Whether a fallback of `op` producing `meta` is permitted under `:raise`.
  """
  @spec allowed?({atom(), arity() | module()} | atom(), Nx.Tensor.t() | nil) :: boolean()
  def allowed?(op, meta \\ nil) do
    Enum.any?(@allowlist, fn {allowed_op, condition, _reason} ->
      allowed_op == op and condition_met?(condition, meta)
    end)
  end

  defp condition_met?(:always, _meta), do: true

  defp condition_met?({:rank_at_least, n}, %Nx.Tensor{shape: shape}),
    do: tuple_size(shape) >= n

  defp condition_met?({:rank_at_least, _n}, _meta), do: false

  # The reason attached to an entry has to actually apply to the case it
  # excuses. `pow` is the one entry here whose reason was written about floats
  # ("GLSL.std.450 has no f64 pow") while its condition matched every dtype, so
  # `Nx.pow(2, 4)` — s32 in, s32 out, nothing to do with fp64 — was permitted by
  # an argument that says nothing about it.
  #
  # That made integer pow invisible to BOTH censuses rather than merely
  # unimplemented: `enforce/3` short-circuits on an allowlisted op before it
  # logs, so pow reported zero hits under `:raise` AND zero under `:warn`. An
  # op nobody can see is an op nobody will fix.
  defp condition_met?({:dtype, type}, %Nx.Tensor{type: type}), do: true
  defp condition_met?({:dtype, _type}, _meta), do: false

  defp condition_met?(:float_output, %Nx.Tensor{type: {f, _}}) when f in [:f, :bf], do: true
  defp condition_met?(:float_output, _meta), do: false

  defp enforce(mode, op, meta) do
    if allowed?(op, meta) do
      :ok
    else
      message = refusal_message(mode, op, meta)

      case mode do
        :warn ->
          Logger.warning(message)
          :ok

        :raise ->
          {shape, type} = describe(meta)

          raise Nx.Vulkan.HostFallbackError,
            message: message,
            op: op,
            shape: shape,
            type: type
      end
    end
  end

  defp describe(%Nx.Tensor{shape: shape, type: type}), do: {shape, type}
  defp describe(_), do: {nil, nil}

  # `{:block, Nx.Block.CumulativeSum}` reads as "block Nx.Block.CumulativeSum",
  # not "block/Elixir.Nx.Block.CumulativeSum" — the Elixir. prefix and the
  # arity slash are both noise when the second element is a module.
  defp describe_op({:block, mod}) when is_atom(mod) and not is_nil(mod) do
    if Code.ensure_loaded?(mod), do: "block #{inspect(mod)}", else: "block/#{mod}"
  end

  defp describe_op({name, arity}), do: "#{name}/#{arity}"
  defp describe_op(name), do: "#{name}/?"

  defp refusal_message(mode, op, meta) do
    subject = describe_op(op)

    where =
      case describe(meta) do
        {nil, nil} -> "  (no output template)"
        {shape, type} -> "  output: #{inspect(shape)} #{inspect(type)}"
      end

    """
    host fallback #{if mode == :raise, do: "refused", else: "reported"}: #{subject}

    #{where}

    `host_fallback: #{inspect(mode)}` flags every fallback that is not on
    Nx.Vulkan.Fallback's allowlist. This op left the GPU and computed on
    Nx.BinaryBackend, which returns a bit-identical result — so no assertion on
    values could ever have seen it.

    The usual cause is a GPU fast path whose gate is narrower than its
    capability: six of the eight instances in docs/BACKWARD_PASS_AUDIT.md had
    the shader already and only the `if` was wrong. Gradients are where this
    shows up — Nx.Defn.Grad emits permuted, reversed, and {:s, 32}-typed
    versions of ops your forward pass never produces. See §1b of
    .claude/skills/vulkan-nx-compute/SKILL.md.

    If this fallback is correct and intended, add a one-line entry to
    @allowlist in lib/nx_vulkan/fallback.ex saying why. If it is not, widen
    the gate rather than the allowlist.

    To permit it for one block: Nx.Vulkan.Fallback.strict(:allow, fn -> ... end)
    """
  end
end
