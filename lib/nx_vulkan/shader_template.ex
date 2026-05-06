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
  Pipe to `Nx.Vulkan.Synthesis.compile/2` to land SPIR-V on
  disk + load via Vulkan.

  See `Nx.Vulkan.ChainShaderSpecs` for the catalog of family specs.
  """

  defmodule FamilySpec do
    @moduledoc """
    Per-family hooks for the templated chain shader.

    All expressions reference local variable `qi` (current f32 position).
    """
    defstruct [
      :name,
      # GLSL text for additional push-constant fields, one per line, with
      # leading 4 spaces. e.g. "    float alpha;\\n    float beta;"
      :push_fields,
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
      # "partial[0] + float(pc.n) * pc.logp_const".
      :logp_final
    ]
  end

  @template ~S"""
#version 450

// SYNTHESIZED by Nx.Vulkan.ShaderTemplate for family: {{name}}
// Generated from a templated leapfrog-chain skeleton. Do not edit;
// regenerate via Nx.Vulkan.Synthesis.compile/2.

layout (local_size_x = 256) in;

layout (push_constant) uniform Push {
    uint  n;
    uint  K;
    float eps;
{{push_fields}}
} pc;

layout (std430, binding = 0) readonly  buffer In_q     { float q_init[]; };
layout (std430, binding = 1) readonly  buffer In_p     { float p_init[]; };
layout (std430, binding = 2) readonly  buffer In_mass  { float inv_mass[]; };
layout (std430, binding = 3) writeonly buffer Out_q    { float q_chain[]; };
layout (std430, binding = 4) writeonly buffer Out_p    { float p_chain[]; };
layout (std430, binding = 5) writeonly buffer Out_grad { float grad_chain[]; };
layout (std430, binding = 6) writeonly buffer Out_logp { float logp_chain[]; };

shared float partial[256];

void main() {
    uint i   = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationIndex;
    bool in_bounds = (i < pc.n);

    float qi = in_bounds ? q_init[i] : 0.0;
    float pi = in_bounds ? p_init[i] : 0.0;
    float mi = in_bounds ? inv_mass[i] : 0.0;

    for (uint k = 0; k < pc.K; k++) {
        // Half-step momentum at q
        {
{{grad_block}}
            float p_half = pi + 0.5 * pc.eps * grad_q;

            // Full-step position
            qi = qi + pc.eps * mi * p_half;

            {
{{grad_block_n}}
                pi = p_half + 0.5 * pc.eps * grad_qn;

                if (in_bounds) {
                    q_chain[k * pc.n + i]    = qi;
                    p_chain[k * pc.n + i]    = pi;
                    grad_chain[k * pc.n + i] = grad_qn;
                }
            }
        }

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

    @template
    |> String.replace("{{name}}", spec.name)
    |> String.replace("{{push_fields}}", indent(spec.push_fields, 4))
    |> String.replace("{{grad_block}}", indent(spec.grad_block, 12))
    |> String.replace("{{grad_block_n}}", indent(grad_block_n, 16))
    |> String.replace("{{logp_block}}", indent(spec.logp_block, 8))
    |> String.replace("{{logp_final}}", spec.logp_final)
  end

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
