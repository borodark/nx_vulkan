defmodule Nx.Vulkan.Backend do
  @moduledoc """
  `Nx.Backend` implementation on top of the Vulkan compute primitives.

  Tensors are represented by:

      %Nx.Vulkan.Backend{ref: <ResourceArc<VulkanTensor>>, shape: tuple, type: {:f, 32}}

  Storage lives on the GPU; an Elixir reference to the `ResourceArc`
  keeps the GPU buffer alive. When the Elixir reference is GC'd, the
  Rust `Drop` impl frees the buffer.

  ## Status — v0.0.3 baseline

  The backend implements only the operators wired in earlier
  iterations:

    - `from_binary/3`, `to_binary/2`, `backend_copy/3`, `backend_transfer/3`
    - elementwise binary: `add`, `subtract`, `multiply`, `divide`, `pow`,
      `max`, `min`
    - elementwise unary:  `exp`, `log`, `sqrt`, `abs`, `negate`, `sigmoid`,
      `tanh`, `tanh`, `relu`-via-clamp-to-0…
    - reductions:         `sum`, `reduce_max`, `reduce_min` (all-axis only)
    - linear algebra:     `dot/6` (rank-2 × rank-2 matmul)

  Anything else falls back to `Nx.BinaryBackend` automatically via
  `Nx.backend_transfer/2`.

  ## Limitations

    - **Compute is f32 only.** Storage round-trips any type
      (f32/f64/s8..s64/u8..u64) via `from_binary`/`to_binary`/`as_type`.
      Per-element ops still dispatch f32 shaders. Use `as_type` to cast
      f64 accumulators to f32 before computing.
    - **No autograd.** `defn grad` won't work end-to-end. The forward
      path is what this backend proves.
    - **Per-axis reductions are host-materialized.** Full-axis
      reductions hit the GPU; per-axis go through download/walk/upload.
    - **No broadcasting in this backend.** Broadcasting happens in
      Nx's frontend before dispatch; we receive already-broadcast
      tensors. (Will swap to the broadcast shader later.)
  """

  @behaviour Nx.Backend

  @enforce_keys [:ref, :shape, :type]
  defstruct [:ref, :shape, :type]

  alias Nx.Tensor, as: T

  @impl true
  def init(opts), do: opts

  # ---------------------------------------------------------------- transfers

  # from_binary / to_binary accept any element type. The compute ops
  # below still gate on f32 (the shaders are f32-only). Non-f32 buffers
  # are transparent storage for things like integer indices that need
  # to round-trip through the backend before being consumed by gather
  # or indexed_put.
  @impl true
  def from_binary(%T{shape: shape, type: type} = tensor, binary, _opts) do
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(binary)
    put_in(tensor.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  @impl true
  def to_binary(%T{data: %__MODULE__{ref: ref}, type: type, shape: shape}, _limit) do
    n_bytes = byte_size_of(shape) * element_bytes(type)

    case Nx.Vulkan.Native.download_binary(ref, n_bytes) do
      {:ok, bin} -> bin
      {:error, reason} -> raise "Nx.Vulkan: download failed: #{inspect(reason)}"
    end
  end

  defp element_bytes({_, bits}), do: div(bits, 8)

  @impl true
  def backend_copy(%T{} = tensor, backend, opts) do
    bin = to_binary(tensor, byte_size_of(tensor.shape) * element_bytes(tensor.type))
    backend.from_binary(tensor, bin, opts)
  end

  @impl true
  def backend_transfer(%T{} = tensor, backend, opts) do
    backend_copy(tensor, backend, opts)
  end

  @impl true
  def backend_deallocate(%T{}), do: :ok

  # ---------------------------------------------------------------- creation

  @impl true
  def constant(%T{shape: shape, type: type} = tensor, scalar, _opts) do
    ensure_f32!(type)
    n = byte_size_of(shape)
    bin = :binary.copy(<<:erlang.float(scalar)::float-32-native>>, n)
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(tensor.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  # v0.1.5: iota and eye host-materialize. Both are tiny (mass-matrix
  # init, index broadcasts) and the GPU shader version would be a
  # one-liner that spends more on dispatch than compute. Upload once,
  # reuse the resulting tensor. Iota with axis=nil flattens the
  # shape; iota with axis=k counts along that axis.
  @impl true
  def iota(%T{shape: shape, type: type} = out, axis, _opts) do
    ensure_f32!(type)
    n = byte_size_of(shape)
    dims = Tuple.to_list(shape)

    floats =
      if axis == nil do
        for i <- 0..(n - 1)//1, do: i * 1.0
      else
        for flat <- 0..(n - 1)//1 do
          coords = unflatten(flat, dims)
          (Enum.at(coords, axis) || 0) * 1.0
        end
      end

    bin = floats |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  @impl true
  def eye(%T{shape: shape, type: type} = out, _opts) do
    ensure_f32!(type)
    dims = Tuple.to_list(shape)
    rank = length(dims)
    n = byte_size_of(shape)

    floats =
      for flat <- 0..(n - 1)//1 do
        coords = unflatten(flat, dims)
        # 1.0 where the last two coords are equal, 0.0 otherwise.
        # Rank-1 falls through to all-1.0 (degenerate but consistent).
        last = Enum.at(coords, rank - 1)
        prev = Enum.at(coords, rank - 2)
        if rank >= 2 and last == prev, do: 1.0, else: if(rank < 2, do: 1.0, else: 0.0)
      end

    bin = floats |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  # ---------------------------------------------------------------- elementwise binary

  for {op, _} <- %{
        add: 0,
        subtract: 0,
        multiply: 0,
        divide: 0,
        pow: 0,
        max: 0,
        min: 0
      } do
    @impl true
    def unquote(op)(out, a, b) do
      do_binary(out, a, b, unquote(op))
    end
  end

  defp do_binary(%T{shape: shape, type: type} = out, %T{} = a, %T{} = b, op) do
    ensure_f32!(type)
    a_data = to_vulkan!(a)
    b_data = to_vulkan!(b)

    apply_op =
      case op do
        :add -> &Nx.Vulkan.add/2
        :multiply -> &Nx.Vulkan.multiply/2
        :subtract -> &Nx.Vulkan.subtract/2
        :divide -> &Nx.Vulkan.divide/2
        :pow -> &Nx.Vulkan.pow/2
        :max -> &Nx.Vulkan.max/2
        :min -> &Nx.Vulkan.min/2
      end

    {:ok, ref} = apply_op.(a_data.ref, b_data.ref)
    put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  # ---------------------------------------------------------------- elementwise unary

  @unary_ops [
    :exp,
    :log,
    :sqrt,
    :abs,
    :negate,
    :sigmoid,
    :tanh,
    :ceil,
    :floor,
    :sign,
    :erf,
    :expm1
  ]

  for op <- @unary_ops do
    @impl true
    def unquote(op)(%T{shape: shape, type: type} = out, a) do
      ensure_f32!(type)
      a_data = to_vulkan!(a)
      apply_op = Function.capture(Nx.Vulkan, unquote(op), 1)
      {:ok, ref} = apply_op.(a_data.ref)
      put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
    end
  end

  # ---------------------------------------------------------------- reductions

  # v0.1.4: full-axis reductions hit the GPU scalar path; per-axis
  # reductions host-materialize. The shader version of partial-axis
  # reduce is a v0.2 item — for the autograd workloads we're chasing,
  # the per-axis reductions are typically over batch dims of modest
  # size, and the download/upload is the dominant cost anyway.
  @impl true
  def sum(out, t, opts), do: do_reduce(out, t, opts, :sum)

  @impl true
  def reduce_max(out, t, opts), do: do_reduce(out, t, opts, :reduce_max)

  @impl true
  def reduce_min(out, t, opts), do: do_reduce(out, t, opts, :reduce_min)

  defp do_reduce(%T{shape: out_shape, type: type} = out, %T{shape: in_shape} = t, opts, op) do
    ensure_f32!(type)
    rank = tuple_size(in_shape)
    axes = opts[:axes] || Enum.to_list(0..(rank - 1)//1)
    keep_axes = opts[:keep_axes] == true

    cond do
      length(axes) == rank ->
        t_data = to_vulkan!(t)
        {:ok, scalar} = gpu_full_reduce(op, t_data.ref)
        scalar_to_tensor(out, scalar)

      length(axes) == 1 ->
        # Single-axis reduction → GPU shader. Compute (outer, reduce, inner).
        [axis] = axes
        in_dims = Tuple.to_list(in_shape)
        outer = in_dims |> Enum.take(axis) |> Enum.reduce(1, &*/2)
        reduce_size = Enum.at(in_dims, axis)
        inner = in_dims |> Enum.drop(axis + 1) |> Enum.reduce(1, &*/2)

        t_data = to_vulkan!(t)
        op_const = case op do :sum -> 0; :reduce_max -> 1; :reduce_min -> 2 end
        {:ok, ref} = Nx.Vulkan.reduce_axis(t_data.ref, outer, reduce_size, inner, op_const)
        put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})

      true ->
        # Multi-axis reduction → host-materialize for v0.1.
        axis_reduce(out, t, axes, keep_axes, op)
    end
  end

  defp gpu_full_reduce(:sum, ref), do: Nx.Vulkan.sum(ref)
  defp gpu_full_reduce(:reduce_max, ref), do: Nx.Vulkan.reduce_max(ref)
  defp gpu_full_reduce(:reduce_min, ref), do: Nx.Vulkan.reduce_min(ref)

  defp axis_reduce(%T{shape: out_shape, type: type} = out,
                   %T{shape: in_shape} = tensor, axes, keep_axes, op) do
    in_bin = to_binary(tensor, byte_size_of(in_shape) * 4)
    input = for <<x::float-32-native <- in_bin>>, do: x

    in_dims = Tuple.to_list(in_shape)
    out_dims = Tuple.to_list(out_shape)
    rank = length(in_dims)

    reduced_sizes = Enum.map(axes, &Enum.at(in_dims, &1))
    n_reduce = Enum.reduce(reduced_sizes, 1, &*/2)

    reduce_set = MapSet.new(axes)
    kept_axes = Enum.reject(0..(rank - 1)//1, &MapSet.member?(reduce_set, &1))

    n_out = byte_size_of(out_shape)

    output =
      for flat_out <- 0..(n_out - 1) do
        out_coords = unflatten(flat_out, out_dims)

        kept_coord_map =
          if keep_axes do
            for k <- kept_axes, into: %{}, do: {k, Enum.at(out_coords, k)}
          else
            Enum.zip(kept_axes, out_coords) |> Map.new()
          end

        values =
          for red_flat <- 0..(n_reduce - 1) do
            red_coords = unflatten(red_flat, reduced_sizes)
            red_map = Enum.zip(axes, red_coords) |> Map.new()

            in_coords =
              for i <- 0..(rank - 1)//1 do
                Map.get(kept_coord_map, i) || Map.fetch!(red_map, i)
              end

            Enum.at(input, flatten(in_coords, in_dims))
          end

        reduce_values(op, values)
      end

    bin = output |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  defp reduce_values(:sum, values), do: Enum.sum(values)
  defp reduce_values(:reduce_max, values), do: Enum.max(values)
  defp reduce_values(:reduce_min, values), do: Enum.min(values)

  defp scalar_to_tensor(%T{shape: shape, type: type} = out, scalar) do
    bin = :binary.copy(<<scalar::float-32-native>>, byte_size_of(shape))
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  # ---------------------------------------------------------------- matmul

  @impl true
  def dot(%T{shape: out_shape, type: type} = out, %T{shape: a_shape} = a, _ca, _ba,
          %T{shape: b_shape} = b, _cb, _bb) do
    ensure_f32!(type)
    {m, k} = a_shape
    {^k, n} = b_shape
    {^m, ^n} = out_shape

    a_data = to_vulkan!(a)
    b_data = to_vulkan!(b)
    {:ok, ref} = Nx.Vulkan.matmul(a_data.ref, b_data.ref, m, n, k)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  # ---------------------------------------------------------------- reshape

  # Zero-copy on the buffer — the GPU stores byte_size + ResourceArc;
  # shape is metadata in the Elixir tensor. Reshape just rewraps the
  # existing reference under a new shape without touching the GPU.
  @impl true
  def reshape(%T{shape: new_shape, type: type} = out, %T{data: %__MODULE__{ref: ref}}) do
    put_in(out.data, %__MODULE__{ref: ref, shape: new_shape, type: type})
  end

  @impl true
  def squeeze(%T{shape: new_shape, type: type} = out, %T{data: %__MODULE__{ref: ref}}, _axes) do
    # Same shape — squeeze drops trivial axes, but the byte layout is
    # unchanged. Pure metadata update.
    put_in(out.data, %__MODULE__{ref: ref, shape: new_shape, type: type})
  end

  # ---------------------------------------------------------------- transpose

  # 2D only at v0.1.2. Higher-rank transpose materializes via host
  # fallback (defer to v0.1.3 when slicing lands and we can do
  # rank-N permutation natively).
  @impl true
  def transpose(%T{shape: out_shape, type: type} = out, %T{shape: in_shape} = a, axes) do
    ensure_f32!(type)

    case {Tuple.to_list(in_shape), axes} do
      {[m, n], [1, 0]} ->
        a_data = to_vulkan!(a)
        {:ok, ref} = Nx.Vulkan.transpose_2d(a_data.ref, m, n)
        put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})

      _ ->
        # Higher-rank or non-swap permutation — host materialize for now.
        bin = to_binary(a, byte_size_of(in_shape) * 4)
        floats = for <<x::float-32-native <- bin>>, do: x

        in_dims = Tuple.to_list(in_shape)
        out_dims = Tuple.to_list(out_shape)
        n_total = byte_size_of(in_shape)

        permuted =
          for flat_idx <- 0..(n_total - 1) do
            out_coords = unflatten(flat_idx, out_dims)
            in_coords = Enum.map(0..(length(in_dims) - 1)//1,
                                 fn i -> Enum.at(out_coords, Enum.find_index(axes, &(&1 == i))) end)
            Enum.at(floats, flatten(in_coords, in_dims))
          end

        new_bin = permuted |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
        {:ok, ref} = Nx.Vulkan.Native.upload_binary(new_bin)
        put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
    end
  end

  # ---------------------------------------------------------------- broadcast

  # v0.1.2 cut: download → replicate on host → upload. Slow for large
  # tensors but correct for any shape/axes combination. v0.2 wires the
  # broadcast shader (spirit already has it as
  # elementwise_binary_broadcast.spv) for the in-place fast path.
  @impl true
  def broadcast(%T{shape: out_shape, type: type} = out, %T{shape: in_shape} = tensor, _shape, axes) do
    ensure_f32!(type)
    in_bin = to_binary(tensor, byte_size_of(in_shape) * 4)

    # Decode input as a flat list — it'll be re-indexed during expansion.
    input = for <<x::float-32-native <- in_bin>>, do: x

    in_dims = Tuple.to_list(in_shape)
    out_dims = Tuple.to_list(out_shape)
    n_out = byte_size_of(out_shape)

    output =
      for flat_idx <- 0..(n_out - 1) do
        flat_in = flat_in_index(flat_idx, in_dims, out_dims, axes)
        Enum.at(input, flat_in)
      end

    bin = output |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  # Scalar input → every output position maps to flat_in=0.
  defp flat_in_index(_flat_idx, [], _out_dims, _axes), do: 0

  # General N-D: project the output coord onto the input coord using the
  # axes mapping (axes[i] = j means input axis i corresponds to output
  # axis j).
  defp flat_in_index(flat_idx, in_dims, out_dims, axes) do
    out_coords = unflatten(flat_idx, out_dims)
    in_coords = Enum.map(0..(length(in_dims) - 1)//1, fn i -> Enum.at(out_coords, Enum.at(axes, i)) end)
    flatten(in_coords, in_dims)
  end

  defp unflatten(flat, dims) do
    {coords, _} =
      Enum.reduce(Enum.reverse(dims), {[], flat}, fn d, {acc, rem} ->
        {[rem(rem, d) | acc], div(rem, d)}
      end)

    coords
  end

  defp flatten(coords, dims) do
    Enum.zip(coords, dims)
    |> Enum.reduce(0, fn {c, d}, acc -> acc * d + c end)
  end

  # ---------------------------------------------------------------- slicing

  # v0.1.3 cut: host-materialize. Walk output coords, project onto
  # input coords via start_indices + out_coord * strides, read from
  # input. v0.2 introduces a strided-copy shader; for now the small
  # slice patterns Nx.Defn produces (mostly axis-aligned, contiguous)
  # are bandwidth-bound on the download/upload anyway.
  @impl true
  def slice(%T{shape: out_shape, type: type} = out, %T{shape: in_shape} = tensor,
            start_indices, lengths, strides) do
    ensure_f32!(type)

    in_bin = to_binary(tensor, byte_size_of(in_shape) * 4)
    input = for <<x::float-32-native <- in_bin>>, do: x

    in_dims = Tuple.to_list(in_shape)
    out_dims = Tuple.to_list(out_shape)

    # Nx may pass start_indices as scalar tensors (for dynamic slices)
    # or as plain integers. For v0.1.3 we accept integers only — if
    # the caller passed tensor indices, materialize them to host
    # numbers via to_number/1.
    start_ints = Enum.map(start_indices, &to_int/1)
    length_ints = Enum.map(lengths, &to_int/1)
    stride_ints = Enum.map(strides, &to_int/1)

    n_out = byte_size_of(out_shape)

    output =
      for flat_idx <- 0..(n_out - 1) do
        out_coords = unflatten(flat_idx, out_dims)

        in_coords =
          Enum.zip([out_coords, start_ints, stride_ints])
          |> Enum.map(fn {c, s, st} -> s + c * st end)

        Enum.at(input, flatten(in_coords, in_dims))
      end

    bin = output |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  # put_slice writes `slice` into `target` starting at `start_indices`,
  # leaving the rest of `target` unchanged. Same host-materialize
  # strategy as slice/5 — read both, walk target coords, write either
  # source.
  @impl true
  def put_slice(%T{shape: out_shape, type: type} = out, %T{shape: target_shape} = target,
                start_indices, %T{shape: slice_shape} = slice_t) do
    ensure_f32!(type)
    ^out_shape = target_shape

    target_bin = to_binary(target, byte_size_of(target_shape) * 4)
    slice_bin  = to_binary(slice_t, byte_size_of(slice_shape) * 4)

    target_floats = for <<x::float-32-native <- target_bin>>, do: x
    slice_floats  = for <<x::float-32-native <- slice_bin>>, do: x

    target_dims = Tuple.to_list(target_shape)
    slice_dims  = Tuple.to_list(slice_shape)
    start_ints  = Enum.map(start_indices, &to_int/1)

    n = byte_size_of(target_shape)

    output =
      for flat_idx <- 0..(n - 1) do
        coords = unflatten(flat_idx, target_dims)
        # Each axis: in slice if start <= coord < start + slice_dim
        slice_coords =
          Enum.zip([coords, start_ints, slice_dims])
          |> Enum.map(fn {c, s, sd} ->
            if c >= s and c < s + sd, do: c - s, else: :outside
          end)

        if Enum.any?(slice_coords, &(&1 == :outside)) do
          Enum.at(target_floats, flat_idx)
        else
          Enum.at(slice_floats, flatten(slice_coords, slice_dims))
        end
      end

    bin = output |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  defp to_int(n) when is_integer(n), do: n
  defp to_int(%T{} = t), do: Nx.to_number(t)

  # ---------------------------------------------------------------- indexing (v0.1.6)

  # gather: indices is a tensor of shape `{..., d}` where d = length(axes)
  # is the number of input axes addressed. For each row of indices, pick
  # one element from input. Output shape = indices.shape minus the last
  # axis (when axes covers the full input rank). When axes is partial,
  # remaining input axes are preserved tail-side.
  @impl true
  def gather(%T{shape: out_shape, type: type} = out, %T{shape: in_shape} = input, indices, opts) do
    ensure_f32!(type)

    axes = opts[:axes] || Enum.to_list(0..(tuple_size(in_shape) - 1)//1)
    in_dims = Tuple.to_list(in_shape)

    in_floats = to_binary(input, byte_size_of(in_shape) * 4) |> decode_f32()
    idx_list = Nx.to_flat_list(indices)

    indices_shape = Tuple.to_list(indices.shape)
    d = List.last(indices_shape)
    n_picks = div(Enum.reduce(indices_shape, 1, &*/2), d)

    # Determine the "kept" input axes (not in `axes`) — those become
    # the trailing dims of each gathered slice.
    axes_set = MapSet.new(axes)
    kept_axes = Enum.reject(0..(tuple_size(in_shape) - 1)//1, &MapSet.member?(axes_set, &1))
    kept_sizes = Enum.map(kept_axes, &Enum.at(in_dims, &1))
    n_kept = Enum.reduce(kept_sizes, 1, &*/2)

    # For each pick: indices row gives values for `axes`. Iterate kept
    # coords for the slice. Total output count = n_picks * n_kept.
    output =
      for pick_idx <- 0..(n_picks - 1)//1, kept_flat <- 0..(n_kept - 1)//1 do
        coord_for_axis =
          for {ax, k} <- Enum.with_index(axes), into: %{} do
            {ax, Enum.at(idx_list, pick_idx * d + k)}
          end

        kept_coords = unflatten(kept_flat, kept_sizes)

        kept_for_axis =
          Enum.zip(kept_axes, kept_coords) |> Map.new()

        in_coords =
          for ax <- 0..(tuple_size(in_shape) - 1)//1 do
            Map.get(coord_for_axis, ax) || Map.fetch!(kept_for_axis, ax)
          end

        Enum.at(in_floats, flatten(in_coords, in_dims))
      end

    bin = output |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  # indexed_put writes scattered values: for each row of indices and
  # corresponding update, set target[idx_row] = update_value. Last write
  # wins on collisions (matching Nx.BinaryBackend semantics).
  @impl true
  def indexed_put(out, target, indices, updates, opts) do
    do_indexed(out, target, indices, updates, opts, :put)
  end

  # indexed_add: same as indexed_put but accumulates. Repeated indices
  # sum, which is what autograd's scatter dual needs.
  @impl true
  def indexed_add(out, target, indices, updates, opts) do
    do_indexed(out, target, indices, updates, opts, :add)
  end

  defp do_indexed(%T{shape: out_shape, type: type} = out, %T{shape: target_shape} = target,
                  indices, updates, opts, mode) do
    ensure_f32!(type)
    axes = opts[:axes] || Enum.to_list(0..(tuple_size(target_shape) - 1)//1)
    target_dims = Tuple.to_list(target_shape)

    target_floats = to_binary(target, byte_size_of(target_shape) * 4) |> decode_f32()
    update_floats = Nx.to_flat_list(updates)
    idx_list = Nx.to_flat_list(indices)

    indices_shape = Tuple.to_list(indices.shape)
    d = List.last(indices_shape)
    n_writes = div(Enum.reduce(indices_shape, 1, &*/2), d)

    axes_set = MapSet.new(axes)
    kept_axes = Enum.reject(0..(tuple_size(target_shape) - 1)//1, &MapSet.member?(axes_set, &1))
    kept_sizes = Enum.map(kept_axes, &Enum.at(target_dims, &1))
    n_kept = Enum.reduce(kept_sizes, 1, &*/2)

    output =
      Enum.reduce(0..(n_writes - 1)//1, target_floats, fn pick_idx, acc ->
        coord_for_axis =
          for {ax, k} <- Enum.with_index(axes), into: %{} do
            {ax, Enum.at(idx_list, pick_idx * d + k)}
          end

        Enum.reduce(0..(n_kept - 1)//1, acc, fn kept_flat, inner_acc ->
          kept_coords = unflatten(kept_flat, kept_sizes)
          kept_for_axis = Enum.zip(kept_axes, kept_coords) |> Map.new()

          target_coords =
            for ax <- 0..(tuple_size(target_shape) - 1)//1 do
              Map.get(coord_for_axis, ax) || Map.fetch!(kept_for_axis, ax)
            end

          flat = flatten(target_coords, target_dims)
          v = Enum.at(update_floats, pick_idx * n_kept + kept_flat)
          old = Enum.at(inner_acc, flat)

          new_val =
            case mode do
              :put -> v
              :add -> old + v
            end

          List.replace_at(inner_acc, flat, new_val)
        end)
      end)

    bin = output |> Enum.map(fn x -> <<x::float-32-native>> end) |> IO.iodata_to_binary()
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: out_shape, type: type})
  end

  defp decode_f32(bin), do: for <<x::float-32-native <- bin>>, do: x

  # ---------------------------------------------------------------- dense linalg (v0.1.9)

  # determinant, solve, cholesky, triangular_solve all host-materialize.
  # Compute is O(d^3); the upload/download is O(d^2). For exmc's mass
  # matrix sizes (d ≤ 50) the GPU offload would lose to the trip cost
  # by 10-100x — host BinaryBackend is the right answer here. A future
  # cuBLAS-equivalent path would need Vulkan KHR cooperative-matrix or
  # a hand-rolled LU shader; deferred.

  @impl true
  def determinant(%T{type: type} = out, %T{} = a) do
    ensure_f32!(type)
    a_host = Nx.backend_transfer(a, Nx.BinaryBackend)
    res = Nx.LinAlg.determinant(a_host)
    upload_host_tensor(out, res)
  end

  @impl true
  def solve(%T{type: type} = out, %T{} = a, %T{} = b) do
    ensure_f32!(type)
    a_host = Nx.backend_transfer(a, Nx.BinaryBackend)
    b_host = Nx.backend_transfer(b, Nx.BinaryBackend)
    res = Nx.LinAlg.solve(a_host, b_host)
    upload_host_tensor(out, res)
  end

  @impl true
  def cholesky(%T{type: type} = out, %T{} = a) do
    ensure_f32!(type)
    a_host = Nx.backend_transfer(a, Nx.BinaryBackend)
    res = Nx.LinAlg.cholesky(a_host)
    upload_host_tensor(out, res)
  end

  @impl true
  def triangular_solve(%T{type: type} = out, %T{} = a, %T{} = b, opts) do
    ensure_f32!(type)
    a_host = Nx.backend_transfer(a, Nx.BinaryBackend)
    b_host = Nx.backend_transfer(b, Nx.BinaryBackend)
    res = Nx.LinAlg.triangular_solve(a_host, b_host, opts)
    upload_host_tensor(out, res)
  end

  defp upload_host_tensor(%T{shape: shape, type: type} = out, host_tensor) do
    bin = Nx.to_binary(host_tensor)
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: type})
  end

  # ---------------------------------------------------------------- as_type (v0.1.8)

  # Round-trip cast between numeric element types. Same-type is a
  # zero-copy ref rewrap. Different types host-materialize: download
  # → decode per source type → re-encode per destination type → upload.
  #
  # The compute shaders are still f32-only; this path lets exmc's mass
  # matrix accumulate in f64 and convert back to f32 before per-step ops.
  @impl true
  def as_type(%T{type: dst_type, shape: shape} = out, %T{type: src_type, data: %__MODULE__{ref: ref}} = tensor) do
    n = byte_size_of(shape)

    cond do
      src_type == dst_type ->
        put_in(out.data, %__MODULE__{ref: ref, shape: shape, type: dst_type})

      src_type == {:f, 32} and dst_type == {:f, 64} ->
        {:ok, new_ref} = Nx.Vulkan.cast_f32_to_f64(ref, n)
        put_in(out.data, %__MODULE__{ref: new_ref, shape: shape, type: dst_type})

      src_type == {:f, 64} and dst_type == {:f, 32} ->
        {:ok, new_ref} = Nx.Vulkan.cast_f64_to_f32(ref, n)
        put_in(out.data, %__MODULE__{ref: new_ref, shape: shape, type: dst_type})

      true ->
        # Integer ↔ float casts host-materialize.
        n_bytes_src = n * element_bytes(src_type)
        {:ok, src_bin} = Nx.Vulkan.Native.download_binary(ref, n_bytes_src)

        values = decode_typed(src_bin, src_type)
        dst_bin = encode_typed(values, dst_type)

        {:ok, new_ref} = Nx.Vulkan.Native.upload_binary(dst_bin)
        put_in(out.data, %__MODULE__{ref: new_ref, shape: shape, type: dst_type})
    end
  end

  defp decode_typed(<<>>, _), do: []
  defp decode_typed(bin, {:f, 32}), do: for <<x::float-32-native <- bin>>, do: x
  defp decode_typed(bin, {:f, 64}), do: for <<x::float-64-native <- bin>>, do: x
  defp decode_typed(bin, {:s, 8}),  do: for <<x::signed-8 <- bin>>, do: x
  defp decode_typed(bin, {:s, 16}), do: for <<x::signed-16-native <- bin>>, do: x
  defp decode_typed(bin, {:s, 32}), do: for <<x::signed-32-native <- bin>>, do: x
  defp decode_typed(bin, {:s, 64}), do: for <<x::signed-64-native <- bin>>, do: x
  defp decode_typed(bin, {:u, 8}),  do: for <<x::unsigned-8 <- bin>>, do: x
  defp decode_typed(bin, {:u, 16}), do: for <<x::unsigned-16-native <- bin>>, do: x
  defp decode_typed(bin, {:u, 32}), do: for <<x::unsigned-32-native <- bin>>, do: x
  defp decode_typed(bin, {:u, 64}), do: for <<x::unsigned-64-native <- bin>>, do: x

  defp encode_typed(values, {:f, 32}),
    do: values |> Enum.map(fn v -> <<v / 1.0::float-32-native>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:f, 64}),
    do: values |> Enum.map(fn v -> <<v / 1.0::float-64-native>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:s, 8}),
    do: values |> Enum.map(fn v -> <<trunc(v)::signed-8>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:s, 16}),
    do: values |> Enum.map(fn v -> <<trunc(v)::signed-16-native>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:s, 32}),
    do: values |> Enum.map(fn v -> <<trunc(v)::signed-32-native>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:s, 64}),
    do: values |> Enum.map(fn v -> <<trunc(v)::signed-64-native>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:u, 8}),
    do: values |> Enum.map(fn v -> <<trunc(v)::unsigned-8>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:u, 16}),
    do: values |> Enum.map(fn v -> <<trunc(v)::unsigned-16-native>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:u, 32}),
    do: values |> Enum.map(fn v -> <<trunc(v)::unsigned-32-native>> end) |> IO.iodata_to_binary()
  defp encode_typed(values, {:u, 64}),
    do: values |> Enum.map(fn v -> <<trunc(v)::unsigned-64-native>> end) |> IO.iodata_to_binary()

  # ---------------------------------------------------------------- helpers

  defp ensure_f32!({:f, 32}), do: :ok

  defp ensure_f32!(other) do
    raise ArgumentError,
          "Nx.Vulkan.Backend currently supports only {:f, 32}; got #{inspect(other)}"
  end

  # If the tensor's data isn't already a Vulkan backend, materialise it.
  defp to_vulkan!(%T{data: %__MODULE__{} = data}), do: data
  defp to_vulkan!(%T{} = t) do
    bin = Nx.to_binary(t)
    {:ok, ref} = Nx.Vulkan.Native.upload_binary(bin)
    %__MODULE__{ref: ref, shape: t.shape, type: t.type}
  end

  defp byte_size_of(shape) do
    shape |> Tuple.to_list() |> Enum.reduce(1, &*/2)
  end

  # ---------------------------------------------------------------- inspect

  @impl true
  def inspect(%T{} = tensor, opts) do
    bin = to_binary(tensor, byte_size_of(tensor.shape) * element_bytes(tensor.type))
    Nx.Backend.inspect(tensor, bin, opts)
  end
end
