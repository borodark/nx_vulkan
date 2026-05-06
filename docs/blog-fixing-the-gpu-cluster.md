# Fixing the GPU Cluster: Five Problems Between "It Compiles" and "It Ships"

*The demo worked on slide 1. Getting it to work on two real machines
took five fixes that no tutorial covers.*

---

## The promise

```elixir
Zed.GPU.dispatch(:"gpu2@192.168.0.247",
  %{dist: :normal, mu: 0.0, sigma: 1.0},
  num_warmup: 200, num_samples: 200)
```

One function call. Two FreeBSD boxes. Two GPUs. Bayesian inference
via Vulkan compute shaders, dispatched over Erlang distribution.

The result:

```
GT 750M: 934ms, 200 samples
GT 650M: 1278ms, 200 samples
Combined 400 samples: mean=-0.081
```

Getting there required fixing five problems that don't appear in
any unit test, any type check, or any CI pipeline. They only
appear when you plug two physical machines together and press enter.

---

## Fix 1: The NIF That Compiled But Didn't Load

### The symptom

```
The on_load function for module Elixir.Nx.Vulkan.Native returned:
{:error, {:bad_lib, "Function not found 'Elixir.Nx.Vulkan.Native':upload_binary_into/2"}}
```

The Rust NIF source had the function. The `.so` compiled. The
Elixir stub existed. But Erlang's NIF loader rejected it.

### The cause

Rustler 0.36 with path deps shares the Cargo target directory
between the source project (`nx_vulkan`) and any consuming project
(`zed`). When `zed` runs `mix deps.compile nx_vulkan`, Cargo
sees the unchanged source files in `~/nx_vulkan/native/` and
uses its cached `.so` — which was compiled before the new NIF
functions were added.

The cached `.so` has the right size. The right symbols (visible
via `strings`). But it was compiled from an older version of the
source. Cargo's fingerprinting says "no change" because the source
*files* didn't change — only the *build context* (which dep
requested the build) changed.

### The fix

```sh
# After building nx_vulkan standalone:
cd ~/nx_vulkan && mix compile --force

# Copy the working .so into Zed's build:
cp ~/nx_vulkan/_build/dev/lib/nx_vulkan/native/nx_vulkan_native/release/libnx_vulkan_native.so \
   ~/zed/_build/dev/lib/nx_vulkan/priv/native/libnx_vulkan_native.so
```

Automated as `scripts/sync_gpu_nif.sh` — run after any NIF change.

### The lesson

**Build caches lie across project boundaries.** When project A
compiles a NIF and project B uses it as a path dep, the
cached artifact in B's `_build` may be from a different version
than A's source. Erlang's NIF loader catches this at load time
(function count mismatch), but the error message doesn't tell
you *which* function is missing or *which* `.so` was loaded.

---

## Fix 2: The OTP Version That Broke Silently

### The symptom

Same NIF, same source, same Cargo build — works on mac-248
(OTP 27), doesn't work on mac-247 (OTP 26). No compilation
error. No warning. The `.so` loads, most functions work, but
`upload_binary_into/2` is missing.

### The cause

OTP 26 and OTP 27 have different Erlang NIF ABI versions.
Rustler generates NIF initialization code that registers
functions with the BEAM. The registration table format
differs between OTP versions. A NIF compiled against OTP 26
headers works on OTP 26 but may silently drop functions on
OTP 27 (or vice versa).

The tricky part: `function_exported?/3` returns `false` for
the missing function, but calling the function directly
triggers module loading and the function *appears to work*
(throws `ArgumentError` from bad args, not `nif_not_loaded`).
This made the diagnosis harder — the function seemed to
exist in some code paths but not others.

### The fix

```elixir
# In Zed.GPU.Agent.init:
defp check_otp_version do
  otp = :erlang.system_info(:otp_release) |> List.to_integer()
  if otp >= 27, do: :ok,
    else: {:error, {:otp_version, "requires OTP 27+"}}
end
```

Plus `setup_freebsd_gpu_node.sh` now installs `erlang-runtime27`
and builds Elixir 1.18 from source against it. All nodes in the
cluster must run the same OTP major version.

### The lesson

**Pin your OTP version across the cluster.** Erlang distribution
connects nodes of different OTP versions without complaint. The
BEAM protocol is backwards-compatible. But NIF ABIs are not. A
cluster where node A runs OTP 26 and node B runs OTP 27 will
connect, exchange messages, and then crash when a NIF function
is called on the wrong node.

---

## Fix 3: The Elixir That Was Compiled Against the Wrong Erlang

### The symptom

```
Elixir 1.17.3 (compiled with Erlang/OTP 26)
```

...running on an OTP 27 runtime. The `erl` binary is OTP 27
(installed via `erlang-runtime27`), but the `elixir` binary was
compiled by FreeBSD's package system against OTP 26.

### The cause

FreeBSD's `pkg` installs Elixir 1.17 compiled against whatever
OTP version was current when the package was built (OTP 26).
Installing `erlang-runtime27` gives you a new `erl` binary at
`/usr/local/lib/erlang27/bin/erl`, but doesn't rebuild Elixir.

Running Elixir 1.17 (compiled with OTP 26) on OTP 27 runtime
*mostly works* — the BEAM bytecode is forward-compatible. But
mix tasks that compile NIFs use the Elixir version's idea of
the Erlang include paths, which point to OTP 26 headers. NIFs
compiled this way get OTP 26 ABI despite running on OTP 27.

### The fix

Build Elixir 1.18.4 from source against OTP 27:

```sh
export PATH=/usr/local/lib/erlang27/bin:$PATH
cd /tmp
wget https://github.com/elixir-lang/elixir/archive/v1.18.4.tar.gz
tar xf v1.18.4.tar.gz && cd elixir-1.18.4
gmake clean compile
# Use from /tmp/elixir-1.18.4/bin/
```

### The lesson

**Elixir's compiled-with-OTP version matters for NIF compilation.**
`elixir --version` shows both the Elixir version and which OTP
it was compiled against. If those don't match the running OTP
runtime, NIF compilation targets the wrong ABI. Check both lines.

---

## Fix 4: The Anonymous Node That Couldn't Connect

### The symptom

```elixir
iex> Node.connect(:"gpu2@192.168.0.247")
false
```

No error. No exception. Just `false`. The remote node was running.
EPMD was reachable. The cookie matched. But `Node.connect`
returned `false`.

### The cause

`mix run` starts an anonymous (unnamed) BEAM node. Anonymous
nodes cannot participate in Erlang distribution. They can't
connect to named nodes. They can't receive connections. The
BEAM doesn't warn you — `Node.connect` simply returns `false`.

This is documented in the Erlang docs, but it's easy to miss
when you're used to `iex -S mix` (which also starts unnamed by
default).

### The fix

```elixir
# In Zed.GPU.Agent.init:
defp check_named_node do
  if Node.alive?() do
    :ok
  else
    {:error, {:not_distributed, "start with --name node@ip"}}
  end
end
```

The Agent refuses to start if the node isn't named. Plus all
demo scripts now include `--name`:

```sh
elixir --name gpu1@192.168.0.248 --cookie zed_gpu_demo \
  --erl "-kernel inet_dist_listen_min 9100 inet_dist_listen_max 9200" \
  -S mix run --no-halt
```

### The lesson

**Check `Node.alive?()` before attempting distribution.** A
named node is a prerequisite for clustering, not an optimization.
If your GenServer needs RPC, fail fast at init when the node
isn't named — don't wait for the first `:rpc.call` to return
`:badrpc`.

---

## Fix 5: The GPU Tensor That Couldn't Cross the Wire

### The symptom

```
** (Protocol.UndefinedError) protocol Enumerable not implemented for type Atom
Got value: :badrpc
```

The RPC call to the remote GPU node returned `:badrpc`. But the
same sampling call worked locally on both machines. And simpler
RPC calls (like `Nx.Vulkan.Native.device_name()`) worked fine.

### The cause

The `ir` struct (the model's intermediate representation) contains
`Nx.Tensor` values for model parameters (`mu`, `sigma`). On the
local node, these tensors are backed by the Vulkan GPU backend —
their `data` field contains a `%Nx.Vulkan.Backend{ref: ref}`
where `ref` is a NIF resource (a pointer to GPU-allocated memory).

When `:rpc.call` serializes the `ir` struct to send it to the
remote node, the NIF resource ref is invalid on the remote
machine. The remote node tries to use the deserialized ref to
access GPU memory that doesn't exist on that machine's GPU.
Crash.

### The fix

Don't send IR across the wire. Send a **model spec** — a plain
map with no tensors:

```elixir
# Before (breaks across nodes):
Zed.GPU.dispatch(remote_node, ir, opts)

# After (works everywhere):
Zed.GPU.dispatch(remote_node, %{dist: :normal, mu: 0.0, sigma: 1.0}, opts)
```

The GPU Agent on the remote node builds the IR locally:

```elixir
def handle_call({:sample, model_spec, opts}, _from, state) do
  ir = build_ir(model_spec)  # builds Nx tensors on THIS node's GPU
  {trace, stats} = Exmc.NUTS.Sampler.sample(ir, %{}, opts)
  plain_trace = materialize_trace(trace)  # convert to plain floats
  {:reply, {:ok, plain_trace, stats, timing}, state}
end
```

And the response is also materialized — `Nx.to_flat_list/1`
converts GPU-backed tensors to plain Elixir lists before
returning over Erlang distribution.

### The lesson

**NIF resources are process-local, not cluster-local.** A
`ResourceArc<VulkanTensor>` on node A is a pointer to GPU memory
on A's physical GPU. Serializing it to node B gives B a pointer
to nothing. Design your RPC API so that GPU tensors never appear
in the request or response — use plain Elixir terms (maps, lists,
binaries) as the serialization boundary.

---

## The meta-lesson

These five fixes share a pattern: **they're all boundary
problems.** The code inside each boundary works perfectly — the
NIF compiles, the shader runs, the sampler converges, the nodes
connect. The bugs live at the seams:

- Build system boundary (Cargo cache across projects)
- ABI boundary (OTP 26 vs 27 NIF tables)
- Compilation boundary (Elixir compiled against wrong OTP)
- Distribution boundary (unnamed vs named nodes)
- Serialization boundary (GPU pointers vs plain data)

No unit test catches these. No type system prevents them. They
only appear when you cross a boundary that your development
environment doesn't have — because you develop on one machine,
with one OTP version, with one project, with named nodes already
running.

The fix in every case was the same: **make the boundary explicit
and check it at init time.** Don't discover at RPC time that your
node isn't named. Don't discover at NIF load time that your OTP
version is wrong. Don't discover at serialization time that your
tensors can't cross the wire. Check at startup. Fail fast. Print
the reason.

```elixir
def init(_opts) do
  with :ok <- check_named_node(),
       :ok <- check_otp_version(),
       :ok <- load_code_paths(),
       {:ok, state} <- init_vulkan() do
    {:ok, state}
  else
    {:error, reason} -> {:stop, reason}
  end
end
```

Four checks. Four potential failures. Four clear error messages.
The demo that took hours to debug now starts in seconds — or
tells you exactly why it can't.

---

*Two GPUs. Two FreeBSD boxes. Five fixes. One Erlang cluster.*

*The shaders were the easy part.*
