defmodule Nx.Vulkan.ShaderTemplate do
  @moduledoc """
  Phase 1 — templated GLSL shader synthesis.

  Renders a chain shader from a per-family spec. The template skeleton is
  ~80 lines of GLSL identical across families (push-constants, bindings,
  per-thread state, leapfrog control flow, parallel reduction). The spec
  fills in three holes:

    * `push_fields`   — additional push-constant scalars
    * `grad_expr(qi)` — `dlogp/dq` formula referencing the position `qi`
    * `logp_expr(qi)` — per-element log-density contribution
    * `logp_final`    — how the workgroup-reduced sum becomes the
                         per-step `logp_chain[k]` (e.g. add a constant
                         host-side normalizer × n)

  Call `render/1` with a `%FamilySpec{}` to get a GLSL source string.
  Pipe to `Nx.Vulkan.Synthesis.compile/1` to land SPIR-V on
  disk + load via Vulkan.

  See `Nx.Vulkan.ChainShaderSpecs` for the catalog of family specs.

  ## Output contract — all four buffers are indexed by the SAME step

  A chain shader writes four buffers per step `k`. **Every one of them
  must describe the state *after* leapfrog step `k`:**

      q_chain[k]    = q_{k+1}
      p_chain[k]    = p_{k+1}
      grad_chain[k] = dlogp/dq at q_{k+1}
      logp_chain[k] = log p(q_{k+1})     <- NOT log p(q_k)

  Consumers slice all four at one index to build a single trajectory
  state, so a buffer that lags by a step silently corrupts that state
  rather than failing. The skeleton below satisfies the contract by
  placing `{{logp_block}}` *after* the position update — do not hoist it
  next to `{{grad_block}}`, which runs at the pre-update position.

  This is not hypothetical: eXMC's `MultiRvCustomSpec` generalised this
  skeleton to multi-RV models and, in doing so, moved the log-prob body
  above the position update. The resulting one-step lag in
  `logp_chain` made its NUTS sampler over-disperse on *every*
  distribution — Normal(0,1) posterior variance 8.55 against a CPU
  reference's 1.45 — and was misattributed to the GPU for a month. See
  `docs/T10_AMPERE_DISPERSION.md`.
  """

  defmodule FamilySpec do
    @moduledoc """
    Per-family hooks for the templated chain shader.

    All expressions reference local variable `qi` (current f32 position).
    """
    defstruct [
      :name,
      # Family parameter VALUES, baked into the rendered source as GLSL
      # literals: %{"alpha" => 2.0, "beta_param" => 5.0}. Every `pc.<key>` in
      # the blocks below is replaced by its literal.
      #
      # These used to be `push_fields`, extra members appended to the push
      # block. That could never have worked: the NIF pushes a fixed-size parsed
      # struct (sizeof(PushBlock) = 20 bytes), so anything declared past the
      # header was silently dropped. Nothing called the chain NIFs from this
      # repo, so nothing ever tripped it. exmc, the only real caller, bakes its
      # priors as OpConstants for the same reason — verified in its SPIR-V.
      #
      # Baking is free for MCMC: priors are fixed for a run, only q/p vary, and
      # Synthesis.compile/1 is content-addressed on the GLSL, so a given
      # parameter set compiles once and hits cache thereafter.
      :params,
      # GLSL block computing local `float grad_q` from `qi` + `in_bounds`.
      # The block sees: pc.*, qi, in_bounds. Must produce `float grad_q`.
      :grad_block,
      # Same shape but produces `float grad_qn`. For most families it is
      # the same expression with `q`→`qn` rename — the renderer can do
      # this automatically if you set `grad_block_n: nil`.
      :grad_block_n,
      # GLSL block computing local `float lp_i` from `qi` + `in_bounds`.
      :logp_block,
      # GLSL expression for `logp_chain[k]` value. Has `partial[0]`,
      # `pc.*` available. e.g. "partial[0]" or
      # "partial[0] + float(pc.d) * pc.logp_const".
      :logp_final,
      # :f32 (default) or :f64. ONE template serves both — the scalar type and
      # literal suffix are substituted. Keeping a second copy of the skeleton
      # is what this repo must not do: the multi-RV port diverged from it once
      # and moved the log-prob body above the position update, giving every
      # distribution a one-step lag in logp_chain that was blamed on the GPU
      # for a month. See the moduledoc.
      :dtype,
      # Optional GLSL emitted before main() — helper functions a family needs.
      # Weibull uses this; most families do not.
      :helpers
    ]
  end

  @template ~S"""
  #version 450
  {{ext}}

  // SYNTHESIZED by Nx.Vulkan.ShaderTemplate for family: {{name}}
  // Generated from a templated leapfrog-chain skeleton. Do not edit;
  // regenerate via Nx.Vulkan.Synthesis.compile/2.

  layout (local_size_x = 256) in;

  // Exactly Nx.Vulkan.NativeV's PushBlock: {k_steps, n_obs, d, _pad, eps},
  // 20 bytes. The NIF pushes sizeof(PushBlock) and nothing more, so a family
  // field declared here would never be written. Family parameters are BAKED
  // into this source as literals instead — see ShaderTemplate.render/1.
  layout (push_constant) uniform Push {
      uint  K;
      uint  n_obs;
      uint  d;
      uint  _pad;
      {{T}} eps;
  } pc;

  layout (std430, binding = 0) readonly  buffer In_q     { {{T}} q_init[]; };
  layout (std430, binding = 1) readonly  buffer In_p     { {{T}} p_init[]; };
  layout (std430, binding = 2) readonly  buffer In_mass  { {{T}} inv_mass[]; };
  layout (std430, binding = 3) writeonly buffer Out_q    { {{T}} q_chain[]; };
  layout (std430, binding = 4) writeonly buffer Out_p    { {{T}} p_chain[]; };
  layout (std430, binding = 5) writeonly buffer Out_grad { {{T}} grad_chain[]; };
  layout (std430, binding = 6) writeonly buffer Out_logp { {{T}} logp_chain[]; };

  shared {{T}} partial[256];

  {{helpers}}

  void main() {
      uint i   = gl_GlobalInvocationID.x;
      uint tid = gl_LocalInvocationIndex;
      bool in_bounds = (i < pc.d);

      {{T}} qi = in_bounds ? q_init[i] : 0.0{{S}};
      {{T}} pi = in_bounds ? p_init[i] : 0.0{{S}};
      {{T}} mi = in_bounds ? inv_mass[i] : 0.0{{S}};

      for (uint k = 0; k < pc.K; k++) {
          // Half-step momentum at q
          {
  {{grad_block}}
              {{T}} p_half = pi + 0.5{{S}} * pc.eps * grad_q;

              // Full-step position
              qi = qi + pc.eps * mi * p_half;

              {
  {{grad_block_n}}
                  pi = p_half + 0.5{{S}} * pc.eps * grad_qn;

                  if (in_bounds) {
                      q_chain[k * pc.d + i]    = qi;
                      p_chain[k * pc.d + i]    = pi;
                      grad_chain[k * pc.d + i] = grad_qn;
                  }
              }
          }

          // Output contract: lp_i is evaluated HERE, after `qi` has been
          // advanced, so logp_chain[k] describes the same state as
          // q_chain[k]. Moving this block above the update introduces a
          // one-step lag — see the moduledoc.
  {{logp_block}}
          partial[tid] = lp_i;
          barrier();

          for (uint s = 128u; s > 0u; s /= 2u) {
              if (tid < s) partial[tid] += partial[tid + s];
              barrier();
          }

          if (tid == 0u) {
              logp_chain[k] = {{logp_final}};
          }

          barrier();
      }
  }
  """

  @doc """
  Render a `%FamilySpec{}` to GLSL source.
  """
  def render(%FamilySpec{} = spec) do
    grad_block_n = spec.grad_block_n || derive_grad_n(spec.grad_block)

    dtype = spec.dtype || :f32

    {scalar, suffix, ext} =
      case dtype do
        :f32 -> {"float", "", ""}
        :f64 -> {"double", "LF", "#extension GL_ARB_gpu_shader_fp64 : require"}
      end

    @template
    |> String.replace("{{T}}", scalar)
    |> String.replace("{{S}}", suffix)
    |> String.replace("{{ext}}", ext)
    |> String.replace("{{helpers}}", spec.helpers || "")
    |> String.replace("{{name}}", spec.name)
    |> String.replace("{{grad_block}}", indent(spec.grad_block, 12))
    |> String.replace("{{grad_block_n}}", indent(grad_block_n, 16))
    |> String.replace("{{logp_block}}", indent(spec.logp_block, 8))
    |> String.replace("{{logp_final}}", spec.logp_final)
    |> bake(spec.params)
  end

  # Replace every `pc.<name>` with its literal value.
  #
  # Word-boundary anchored, not a bare String.replace: a parameter named `alpha`
  # would otherwise also rewrite the `alpha` inside `pc.alpha_scale`. This repo
  # has already paid for that once — Codegen's unary templates clobbered the `r`
  # inside `sqrt`/`round` before they were anchored the same way.
  defp bake(glsl, params) when is_map(params) do
    Enum.reduce(params, glsl, fn {name, value}, acc ->
      Regex.replace(~r/\bpc\.#{Regex.escape(to_string(name))}\b/, acc, glsl_literal(value))
    end)
  end

  defp bake(glsl, nil), do: glsl

  # GLSL needs a decimal point: `2` is an int literal and will not implicitly
  # convert in every position `2.0` is valid.
  defp glsl_literal(v) when is_float(v), do: Float.to_string(v)
  defp glsl_literal(v) when is_integer(v), do: Float.to_string(v * 1.0)

  # Auto-derive the second grad block by renaming locals to *_n. This
  # keeps simple specs DRY; complex families can override with grad_block_n.
  defp derive_grad_n(grad_block) do
    grad_block
    |> String.replace(~r/\bgrad_q\b/, "grad_qn")
    |> rename_intermediate_locals()
  end

  # Rename the most common intermediate local `q` → `qn` to avoid shadowing
  # the outer `qi`. Other locals (denom, diff, z2 etc.) are scoped per-block,
  # so shadowing inside the inner block is fine — but `q` specifically gets
  # used in logp_block too so we keep it distinct.
  defp rename_intermediate_locals(block) do
    Regex.replace(~r/\bfloat q\b/, block, "float qn")
    |> String.replace(~r/(?<![\w.])q(?![\w])/, "qn")
  end

  defp indent(text, spaces) do
    pad = String.duplicate(" ", spaces)

    text
    |> String.split("\n")
    |> Enum.map(fn
      "" -> ""
      line -> pad <> line
    end)
    |> Enum.join("\n")
  end
end
