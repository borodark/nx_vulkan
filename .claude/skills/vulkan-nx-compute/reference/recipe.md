# Add-a-static-op recipe (copy-pasteable)

A worked minimal example: an elementwise unary `scale(x) = x * c` where the
scalar `c` rides in the push constant. Adapt the pattern; the four files are
always the same.

## 1. `glsl/scale_f32.comp`

```glsl
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly  buffer In  { float x[]; };  // input first
layout(std430, binding = 1) writeonly buffer Out { float o[]; };  // output last

layout(push_constant) uniform Push {
    uint  n;   // element count
    float c;   // scale factor
} p;

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.n) return;
    o[i] = x[i] * p.c;
}
```

Compile:

```sh
glslangValidator -V glsl/scale_f32.comp -o priv/shaders/scale_f32.spv
```

## 2. NIF in `native/nx_vulkan_vulkano/src/lib.rs`

Add the push struct near the other `PushN`/`PushMatmul` defs:

```rust
#[derive(Clone, Copy, BufferContents)]
#[repr(C)]
struct PushScale {
    n: u32,
    c: f32,
}
```

The NIF (model it on `dispatch_generated` — it already uses the shared
`run_single_dispatch` helper):

```rust
#[rustler::nif(schedule = "DirtyIo")]
fn scale<'a>(
    env: Env<'a>,
    out_ref: ResourceArc<VulkanoTensor>,
    a_ref: ResourceArc<VulkanoTensor>,
    n: u32,
    c: f32,
    spv_path: String,
) -> NifResult<Term<'a>> {
    let context = match ctx() {
        Ok(c) => c,
        Err(e) => return Ok((atoms::error(), atoms::vulkan_init_failed(), e).encode(env)),
    };

    let result = (|| -> Result<(), String> {
        // No spec constant -> None. Pipeline is cached by (spv_path, -1).
        let cached = get_or_create_pipeline(&spv_path, None)?;
        let set = PersistentDescriptorSet::new(
            &context.set_allocator,
            cached.layout.set_layouts()[0].clone(),
            [
                WriteDescriptorSet::buffer(0, a_ref.buf.clone()),   // inputs 0..k-1
                WriteDescriptorSet::buffer(1, out_ref.buf.clone()), // output last
            ],
            [],
        )
        .map_err(|e| format!("descriptor set: {e}"))?;

        run_single_dispatch(context, &cached, set, PushScale { n, c }, [n.div_ceil(256), 1, 1])
    })();

    match result {
        Ok(()) => Ok(rustler::types::atom::ok().encode(env)),
        Err(msg) => Ok((atoms::error(), atoms::dispatch_failed(), msg).encode(env)),
    }
}
```

Register it in `rustler::init!(...)` (add `scale,` to the list).

## 3. Stub in `lib/nx_vulkan/native_v.ex`

Arity MUST match the NIF exactly, or the whole NIF module fails to load:

```elixir
def scale(_out, _a, _n, _c, _spv_path), do: :erlang.nif_error(:nif_not_loaded)
```

## 4. Call it from `lib/nx_vulkan/vulkano_backend.ex`

```elixir
@scale_f32_spv Path.expand("../../priv/shaders/scale_f32.spv", __DIR__)

def scale(%T{shape: shape, type: {:f, 32} = type} = out, tensor, c)
    when is_number(c) do
  t = ensure_on_backend(tensor)

  if match?(%__MODULE__{}, t.data) and t.shape == shape do
    %T{data: %__MODULE__{ref: a_ref}} = t
    n = byte_size_of(shape)                        # element count helper in the module
    {:ok, out_ref} = Nx.Vulkan.NativeV.buf_alloc(n * element_bytes(type))
    :ok = Nx.Vulkan.NativeV.scale(out_ref, a_ref, n, c * 1.0, @scale_f32_spv)
    put_in(out.data, %__MODULE__{ref: out_ref, shape: shape, type: type})
  else
    # host fallback — always correct
    host_result(out, Nx.multiply(Nx.backend_transfer(t, Nx.BinaryBackend), c))
  end
end
```

## Verify

```elixir
Nx.Vulkan.NativeV.device_name()             # ensure real GPU, not llvmpipe
gpu  = Nx.multiply(x_on_vulkan, 3.0)         # or your Nx entrypoint
host = Nx.multiply(Nx.backend_transfer(x, Nx.BinaryBackend), 3.0)
# assert equal within f32 eps
```

## Reference NIFs to copy from (all in lib.rs)

- `dispatch_generated` — cleanest N-input/1-output, uses `run_single_dispatch`.
- `apply_binary` — spec-constant op selection (`get_or_create_pipeline(_, Some(op))`).
- `apply_binary_broadcast` / `apply_slice` — a `params` SSBO of packed int32 shape metadata.
- `matmul` — 2D tiled dispatch `[ceil(N/16), ceil(M/16), 1]` (NOTE: builds its
  pipeline per-call — legacy, do not copy that part; use the cache).
- `reduce_axis` — `(outer, reduce_size, inner, op)` push, per-slot dispatch.
- `transpose_nd` / `reverse_nd` / `broadcast_nd` — the **index-remap** pattern:
  one thread per *output* element, decompose its linear index into coordinates,
  map to an input index, copy. Shape metadata rides in a `params` SSBO (SKILL.md
  §5). Copy this whenever the op is "same data, different layout" — it is what
  keeps gradient-shaped tensors on the GPU.
- `window_scatter_max` — one thread per **input** element (not per output), which
  is how it avoids needing `GL_EXT_shader_atomic_float` for the scatter. Ties
  break on `>=` to match BinaryBackend's last-max-wins fold.

## Before you call it done

A green parity test does not mean your kernel ran — the host fallback returns
the same bytes. Assert residency too:

```elixir
{_out, counts} = Nx.Vulkan.Fallback.count(fn -> Nx.sum(Nx.multiply(a, b)) end)
assert counts == %{}
```

Put the assertion on the op you changed. A fallback anywhere upstream breaks the
chain, so a residency assertion on the wrong op measures something else — that
mistake is why the first conv-backward fix was believed complete when it was not.
