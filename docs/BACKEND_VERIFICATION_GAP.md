# The Backend Verification Gap

**Date:** 2026-08-02
**Scope:** what the Elixir Nx ecosystem does and does not verify about a third-party `Nx.Backend`, and what `nx_vulkan` has to build itself
**Status:** research complete; two of the recommendations are already implemented (`grad_test.exs`, `Nx.Vulkan.Fallback`), the rest are open

---

## TL;DR

The Nx ecosystem ships **no reusable conformance suite for a backend**, and in
particular **nothing that exercises the backward pass**. The community standard
for validating a backend is `doctest Nx` with the backend set as default. That
suite contains **zero gradient examples**, and it is value-based, which makes it
structurally incapable of detecting the failure mode that actually bit us: an op
silently leaving the GPU.

Upstream *does* have the two pieces we want — a 6031-line
`nx/test/nx/defn/grad_test.exs` (293 tests) and a finite-differences
`Nx.Helpers.check_grads!` — but both live under `nx/test/`, which the Hex package
does not ship. From `deps/nx` they do not exist.

Two things follow. First, every non-monorepo backend author reinvents gradient
testing or skips it. Second — the harder point — nobody upstream has the bug
class we have, because neither EXLA nor Torchx uses a blanket silent host
fallback. So even a shared conformance kit, if it only asserted values, would not
have caught our conv regression. The missing primitive is **residency
assertion**, not more numbers.

---

## 1. The gap

### 1.1 What is verified today

| Mechanism | Where | What it covers |
|---|---|---|
| `doctest Nx` with backend as default | `test/nx_vulkan/nx_doctest_test.exs` (mirrors `torchx/test/torchx/nx_doctest_test.exs`) | Forward-pass values and `inspect` strings for ~950 examples |
| `Nx.Testing.assert_equal/2`, `assert_all_close/3` | `deps/nx/lib/nx/testing.ex` | Assertion *helpers*. Not a suite — they only compare two tensors you produced yourself |
| Backend documentation convention | `nx/guides/backend_documentation/convention.md`, tested via `doctest_file/1` | Prose about divergent behaviour. Documentation, not verification |

That is the whole published surface. `Nx.Backend`'s `@moduledoc`
([hexdocs](https://hexdocs.pm/nx/Nx.Backend.html)) enumerates the 71 callbacks
and points at the documentation convention. It says nothing about how to test an
implementation of them. There is no "writing a backend" guide in
`nx/guides/{getting_started,advanced,cheatsheets,exercises}` — verified by
listing the directory on `main`.

### 1.2 Evidence, verified in this tree and on `elixir-nx/nx@main`

```
$ grep -c "Nx.Defn.grad\|Nx.Defn.value_and_grad" deps/nx/lib/nx.ex
0
```

The `Nx` module's doctests contain **zero** gradient examples. `doctest Nx`
therefore exercises a backend's forward path only. This is not an oversight —
`grad` lives in `Nx.Defn`, not `Nx` — but the consequence stands: the
community-standard backend validation never calls the autodiff transform.

```
$ ls deps/nx/test
ls: cannot access 'deps/nx/test': No such file or directory
```

Confirmed against `deps/nx/hex_metadata.config` (nx 0.13.0): the `files` list
contains `lib/**` only. No `test/`, no `guides/`. `nx/mix.exs` does not override
Hex's default file list, so this is not going to change by accident.

What is behind that wall, on `main`:

| File | Size | Reachable from a Hex dep? |
|---|---|---|
| `nx/test/nx/defn/grad_test.exs` | 6031 lines, **293 tests**, 33 uses of `check_grads!` | No |
| `nx/test/support/helpers.ex` — `Nx.Helpers.check_grads!/4` | finite-difference gradient checker | No |
| `nx/test/support/nx_case.ex` — `Nx.Case` | test case template | No |
| `nx/test/nx/doctest_test.exs` | `doctest Nx`, 3 lines | Only by copying it |

`Nx.Helpers.check_grads!/4` is the single most valuable thing upstream has for a
backend author and it is unreachable:

```elixir
def check_grads!(func, grad_func, x, opts \\ []) when is_list(opts) do
  atol = opts[:atol] || 1.0e-7
  rtol = opts[:rtol] || 1.0e-4
  step = opts[:step] || 1.0e-4
  est_grad = finite_differences(func, x, step)
  comp_grad = grad_func.(x)
  assert_all_close(comp_grad, est_grad, x, atol, rtol)
end
```

Central-difference estimate vs. the analytical gradient. That is JAX's
`check_grads` and PyTorch's `gradcheck`, in twelve lines, already written, and
not shipped.

### 1.3 Why the monorepo does not feel the gap

`.github/workflows/ci.yml` runs a matrix over `working_directory: ["nx", "exla",
"torchx"]`, each doing a plain `mix test` in its own directory. `mix test` only
runs that project's own `test/`. So:

- `nx/test/nx/defn/grad_test.exs` — 293 gradient tests — runs **only against
  `Nx.BinaryBackend` / `Nx.Defn.Evaluator`**. Grepping `nx/test/` for
  `default_backend` finds four hits, all in `tensor_test.exs`/`defn_test.exs`
  swapping in the toy `ProcessBackend`, none parametrising the suite.
- EXLA's `test/test_helper.exs` sets `Nx.Defn.global_default_options(compiler:
  EXLA)`, but that only affects EXLA's own test files.
- Torchx's `test/test_helper.exs` sets `Application.put_env(:nx,
  :default_backend, {Torchx.Backend, device: default_device})`, likewise scoped
  to Torchx's own test files.

Upstream's gradient suite is a test of `Nx.Defn.Grad`'s *rules*, not of any
backend's ability to execute what those rules emit. Nothing runs it against a
device backend, in the monorepo or out of it.

---

## 2. Why value-based tests are structurally blind to fallback regressions

This is the load-bearing argument, and it is specific to backends built the way
this one is.

`Nx.Vulkan.VulkanoBackend` implements a **universal host fallback**: any op it
cannot run natively does `backend_transfer` to `Nx.BinaryBackend`, computes
there, and transfers back. That is a good design for coverage — it is why the
backend can claim the full `Nx.Backend` surface (see `docs/NX_PARITY_RESEARCH.md`
and `test/nx_vulkan/parity_fallback_test.exs`). It is also the reason no
assertion on values can detect a fallback:

> **The fallback *is* the reference implementation.** A host fallback returns a
> result computed by `Nx.BinaryBackend` — the exact module every parity test
> compares against. The comparison is not "close": it is bit-identical, by
> construction, because it is the same code path producing both sides.

Consequences, in increasing order of unpleasantness:

1. **A fallback never fails a value test.** `assert_all_close` and `assert_equal`
   both pass trivially. Tightening tolerances does nothing — the diff is exactly
   zero.
2. **A fallback never fails a doctest.** Doctests compare `inspect` strings.
   BinaryBackend produces the canonical string. Falling back makes the doctest
   *more* likely to pass, since the native shader path is what produces last-ULP
   `inspect` mismatches (`@rounding` in `nx_doctest_test.exs` exists precisely
   for those).
3. **The pathological limit:** a backend that host-falls-back for *every*
   callback scores 100% on `doctest Nx`. The standard validation cannot
   distinguish a working GPU backend from an elaborate no-op wrapper around
   `Nx.BinaryBackend`. Correctness testing and acceleration testing are
   orthogonal, and the ecosystem only ships the first.
4. **A performance cliff presents as silence.** Not a slow test, not a warning —
   nothing. Wall-clock is the only signal, and wall-clock is exactly what nobody
   asserts on in a unit suite because it is flaky.

### 2.1 The case that proved it: conv's backward pass

`Nx.Defn.Grad` generates ops that nobody writes by hand. The gradient of a
convolution is itself a convolution — but with the first two axes swapped
(`conv_spec_transpose/1` in Nx), so it arrives with non-identity
`input_permutation` / `kernel_permutation` / `output_permutation`.

The GPU conv path was gated on an identity-permutation check, because every conv
a *forward* pass produces has identity permutations. Every gradient conv failed
the gate and host-fell-back. Result:

- the parity suite stayed green (bit-identical),
- `doctest Nx` stayed green (bit-identical),
- a CNN training step took ~30 seconds,
- and the entire backward pass of the flagship op ran on the CPU for the whole
  life of the conv shaders.

The generalisation, now encoded in `test/nx_vulkan/grad_test.exs`: **a fast path
is not covered until its gradient is covered**, because the gradient is a
*different distribution of shapes and options* over the same op. Fast-path gates
written against forward-pass shapes are the default failure mode, not an unlucky
one. A second instance of exactly this fell out once the counter existed: the
gradient seed for `Nx.sum` is materialised at Nx's default f32 while the input is
f64, so the kernel-gradient conv arrived as `f64 × f32` and failed a
`i.type == ot and k.type == ot` gate — a *dtype* gate rather than a permutation
gate, same class of bug, invisible for the same reason (fixed in `a680788` by
coercing on-device).

### 2.2 What a test would have to assert instead

Not values. One of:

- **Residency** — `match?(%VulkanoBackend{}, tensor.data)` on the result. Cheap,
  but only observes the *final* tensor of a graph; an intermediate that fell back
  and came home is invisible. Also unreliable for gradients, whose final tensor
  is often legitimately host-resident (e.g. `reverse/3` in the conv
  input-gradient chain is still a fallback).
- **Fallback count** — instrument the single exit point every fallback path
  shares and assert the count is zero for a given graph. This observes
  intermediates. This is what `Nx.Vulkan.Fallback` does.
- **Wall-clock** — real signal, unusable as an assertion.

---

## 3. What EXLA and Torchx actually do

All of the following was read from a shallow clone of `elixir-nx/nx@main`, not
inferred.

### 3.1 EXLA

`exla/test/` contains `exla_test.exs`, `test_helper.exs`, `support/`, and
`exla/`. Under `exla/test/exla/`: `backend_test.exs`, `backend_documentation_test.exs`,
`client_test.exs`, `custom_call_alias_test.exs`, `device_buffer_test.exs`,
`device_memory_sharing_test.exs`, `executable_test.exs`, `memory_tracking_test.exs`,
`nx_linalg_doctest_test.exs`, `random_test.exs`, `serving_test.exs`, plus `defn/`
and `mlir/`. `exla/test/exla/defn/` holds `api_test.exs`, `expr_test.exs`,
`lock_test.exs`, `locked_cache_test.exs`, `recompilation_warning_test.exs`,
`runtime_call_test.exs`, `sharding_test.exs`, `vectorize_test.exs`.

**There is no `exla/test/exla/defn/grad_test.exs`.** Grepping the whole of
`exla/test/` for `grad` returns hits in exactly two files:

- `exla/test/exla/defn/expr_test.exs` — `grad_if_tuple` (grad through `if`),
  `stop_grad`, and three `triangular_solve` gradient tests for complex/conjugate
  transforms. Six `grad(` call sites total.
- `exla/test/exla/backend_test.exs` — one local variable happens to be named
  `gradients`.

So EXLA's gradient coverage is roughly **four tests**, and all four are testing
*control-flow and linalg lowering corner cases*, not "does the backward pass of
op X work." EXLA gets away with this because XLA compiles the whole `Nx.Defn`
graph: a missing lowering is a compile error, not a silent detour. There is no
"fell back to the host" state to test for. `exla/lib/exla/backend.ex` touches
`Nx.BinaryBackend` in exactly three places (`constant`, `concatenate`, `stack`)
— not a general fallback policy.

### 3.2 Torchx

`torchx/test/torchx/` holds `backend_documentation_test.exs`, `complex_test.exs`,
`defn_test.exs`, `device_memory_sharing_test.exs`, `device_test.exs`,
`nx_block_test.exs`, `nx_doctest_test.exs`, `nx_linalg_doctest_test.exs`,
`nx_linalg_test.exs`, `nx_test.exs`, `random_test.exs`.

**`grep -rn "grad" torchx/test/` returns nothing.** Zero occurrences, in any
file, in any form. Torchx — the reference third-party-shaped backend, the one our
`nx_doctest_test.exs` is modelled on — has **no gradient test at all**.
`torchx/test/torchx/defn_test.exs` (68 lines) covers `iota`, scalar broadcast,
`while`, and `determinant`. That is the entire `defn` integration test.

Torchx's real validation is `nx_doctest_test.exs` (`doctest Nx` with
`Nx.default_backend(Torchx.Backend)` and four `@except` buckets:
`@rounding_error_doctests`, `@os_rounding_error_doctests`,
`@inherently_unsupported_doctests`, `@unrelated_doctests`) plus a hand-written
1256-line `nx_test.exs` re-covering what the doctests exclude. Our
`nx_doctest_test.exs` reproduces this structure faithfully, including the
bucketed excepts — which is correct, and also inherits the blind spot.

Torchx does *not* have our bug class either, for a different reason than EXLA:
`torchx/lib/torchx/backend.ex` **raises** on unsupported operations
(`raise "operation #{fun} is not supported on Torchx.Backend"`,
`unsupported_option!/3`, `"Torchx does not support complex values for atan2"`).
It has no universal host fallback. Its MPS handling is instructive: an explicit
`mps_unsupported` list of blocks that raises rather than silently routing
elsewhere.

### 3.3 Would either have caught our conv bug?

**No, and they couldn't have.** Neither project has a test that asserts *where*
an op executed, because neither has a mechanism by which an op can execute
somewhere unexpected. The gap is not laziness upstream; it is that the "universal
silent fallback" design — the thing that makes `nx_vulkan` usable while only ~33
of 71 callbacks are native — creates a failure mode that has no upstream
equivalent and therefore no upstream test.

Corollary for us: adopting EXLA's and Torchx's test strategies wholesale is
necessary but not sufficient. We have already done that (`nx_doctest_test.exs`).
The residency layer is ours to invent.

---

## 4. Prior art: PyTorch and JAX

### 4.1 Gradient correctness

**PyTorch —** `torch.autograd.gradcheck(func, inputs, *, eps=1e-06, atol=1e-05,
rtol=0.001, raise_exception=True, nondet_tol=0.0, check_undefined_grad=True,
check_grad_dtypes=False, check_batched_grad=False, check_batched_forward_grad=False,
check_forward_ad=False, check_backward_ad=True, fast_mode=False, masked=None)`.
It builds the numerical Jacobian by finite differences and compares it to the
analytical Jacobian obtained from `backward()`, elementwise under `allclose`.
`gradgradcheck` does the same one order up (gradient-of-gradient). Note
`check_undefined_grad` and `check_grad_dtypes` — the API treats "returned the
wrong dtype" and "mishandled an undefined grad" as first-class failures, not just
"wrong number."
([docs](https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.gradcheck.html))

**JAX —** `jax.test_util.check_grads(f, args, order, modes=('fwd', 'rev'),
atol=None, rtol=None, eps=None)`. Same finite-difference-vs-autodiff idea, with
two refinements worth stealing: it checks **both forward and reverse mode**, and
it projects onto *a single random direction* rather than materialising the full
Jacobian, which is what makes it affordable on realistic shapes. `order` requests
higher-order checks.
([docs](https://docs.jax.dev/en/latest/_autosummary/jax.test_util.check_grads.html))

**Nx analogue:** `Nx.Helpers.check_grads!/4` already *is* this — central
differences, `atol: 1.0e-7`, `rtol: 1.0e-4`, `step: 1.0e-4` — at order 1, reverse
mode only, full-tensor (not random-direction). It is used 33 times in
`nx/test/nx/defn/grad_test.exs` and shipped to nobody.

### 4.2 Op-level device conformance

**PyTorch OpInfo.** `torch/testing/_internal/common_methods_invocations.py`
defines an `OpInfo` per operator and collects them in `op_db`: each entry carries
`sample_inputs()`, supported dtypes per device type, autograd support flags, and
known-failure decorators. `test/test_ops.py` then uses the `@ops` decorator from
`torch/testing/_internal/common_device_type.py` to instantiate every test
template across the **cross-product of {operator} × {dtype} × {device}**. One
test template, thousands of instantiations. Adding a device means the whole
matrix runs against it; adding an operator means the whole matrix runs on it.
([overview](https://pytorch.org/blog/understanding-pytorchs-test-infrastructure/),
[op_db](https://github.com/pytorch/pytorch/blob/main/torch/testing/_internal/common_methods_invocations.py),
[test_ops.py](https://github.com/pytorch/pytorch/blob/main/test/test_ops.py))

**Nx analogue:** does not exist, at any level. There is no data structure
describing "here is `Nx.conv/3`, here are representative argument sets, here are
the dtypes it should support." Every backend hand-writes its own samples. The
closest thing in this repo is `docs/nx_parity_gap.csv` — a callback inventory,
not a sample-input database.

### 4.3 The fallback question, solved elsewhere

PyTorch's MPS backend is the closest analogue to our situation: a device backend
with partial op coverage and a CPU to fall back to. Its answer is the one worth
copying — **the fallback is opt-in and off by default**. An unimplemented op
raises `NotImplementedError: The operator 'aten::…' is not currently implemented
for the MPS device`, and only `PYTORCH_ENABLE_MPS_FALLBACK=1` turns on CPU
routing, documented as "this will be slower than running natively on MPS."
([MPS notes](https://docs.pytorch.org/docs/stable/notes/mps.html),
[#86195](https://github.com/pytorch/pytorch/issues/86195))

The design lesson is not "don't fall back." It is: **silence is the bug**. A
fallback that must be requested cannot happen by accident, so a regression that
introduces one turns into a hard failure at the exact call site, in every test
that touches it, without anyone writing a residency assertion.

JAX takes the strict version of the same position: there is no device fallback at
all. A primitive without a lowering for the target platform is a compile-time
error.

**Nx analogue:** we have the counter (`Nx.Vulkan.Fallback`), which is the
observability half. The enforcement half — a strict mode where `host_result/2`
raises instead of computing — does not exist yet. See §5.

---

## 5. Recommendations for `nx_vulkan`

### 5.1 Already built

**`test/nx_vulkan/grad_test.exs`** — 22 test cases, 28 `grad_parity/3`
assertions, covering elementwise chains, reductions, `dot`/`transpose`, softmax
and layernorm composites, five conv gradient cases (including strided+padded, f32,
and conv→tanh), `window_max`, and two end-to-end nets (MLP, conv→activation→dense
head). Each differentiates the same function on `VulkanoBackend` and
`Nx.BinaryBackend` via `Nx.Defn.jit_apply/3` and compares. Tolerances are split
deliberately: `1.0e-10` for pure arithmetic, `1.0e-6` where a transcendental is
involved, because SPIR-V's `GLSL.std.450` has no f64 transcendentals and those
boundary-cast through f32.

This is **gradient parity**, which is a strictly weaker property than PyTorch's
`gradcheck`: it verifies the backend agrees with the reference implementation of
the same autodiff rules, not that the rules are right. That is the correct
division of labour — the rules are `Nx.Defn.Grad`'s problem — but see 5.2(a).

**`lib/nx_vulkan/fallback.ex` + `test/nx_vulkan/fallback_test.exs`** — a
process-local host-fallback counter with compile-time op attribution, hooked at
`host_result/2`, the common exit point of every fallback path. `count/1` returns
`{result, %{{fun, arity} => n}}`; `count_total/1` collapses it. Off by default
(one `Process.get/2` on a path already doing a device→host copy); nests correctly.
The test file splits into **native** (must be 0 — a regression that reroutes to
the host fails here) and **known fallbacks** (pinned to exact counts, so
promoting an op on-device *fails the test*, which is the reminder to move it).
This is the only mechanism in the repo that can observe the conv bug class.

### 5.2 Still missing

**(a) A real `check_grads!` — finite differences, not parity.**
Port `Nx.Helpers.check_grads!/4` (it is twelve lines; §1.2) into
`test/support/`. Parity against BinaryBackend cannot catch a case where
`Nx.Defn.Grad` and both backends agree on a *wrong* answer, and more practically
it cannot catch the case where our backend's forward op is subtly wrong in a way
that makes both the forward value and its gradient consistently wrong. Add
JAX's random-direction projection so it stays affordable on conv-sized tensors.
Worth adding `order: 2` for the ops that appear in second-order workloads
(exmc's NUTS does not need it today; Scholar's optimisers might).

**(b) Fallback assertions on every gradient test, not a separate file.**
Right now `grad_test.exs` asserts numbers and `fallback_test.exs` asserts
residency, and only one conv gradient appears in both. The two should be fused:
`grad_parity/3` should take an expected fallback count (defaulting to a pinned
value per test) and assert it alongside the value comparison. Every gradient the
backend claims to accelerate then has both halves of its contract checked in one
place, and adding a new grad test cannot forget the residency half.

**(c) Strict mode — make the fallback opt-in, as MPS does.**
An application-env flag (`config :nx_vulkan, host_fallback: :raise | :allow`)
checked in `host_result/2`. In `:raise` mode, an unnative op raises with the
`{fun, arity}` the counter already attributes. Then a *new* CI job runs the whole
suite in strict mode with a documented allowlist of legitimate fallbacks, and any
newly-introduced fallback fails loudly at its call site without anyone having
written a test for it. This is the single highest-leverage item on this list: it
converts the failure mode from "invisible" to "impossible to miss," which is
qualitatively different from "detectable if you wrote the right assertion."

**(d) Cross-process fallback counting.**
`Nx.Vulkan.Fallback` is process-local by design, so work funnelled through
`Nx.Vulkan.Node` is invisible to a caller's `count/1`. Any multi-device or
serving-shaped workload therefore reports zero fallbacks regardless of the truth.
Needs either a `$callers`-style propagation of the recording flag or an ETS
counter keyed by a run id.

**(e) Coverage of what `Nx.Defn.Grad` actually emits, systematically.**
The conv bug was found by accident. The systematic version: enumerate the ops
`Nx.Defn.Grad` *generates* rather than the ops users call — `reverse`,
permuted `conv`, `window_scatter_max`, `pad` with negative padding, `select` with
broadcast conditions, transposed `dot` contractions — and assert each is native
*with the shapes and options the grad transform produces*, not the shapes a
forward pass produces. `reverse/3` and `window_max/4` are already pinned as known
fallbacks; the list should be derived from the grad rules, not from memory.

**(f) A fallback budget for a real training step.**
`assert Fallback.count_total(fn -> one_axon_training_step() end) <= N` with `N`
ratcheting down over time. This catches composition-level regressions that no
per-op test sees, and it is the metric that actually correlates with the 30-second
CNN step.

**(g) Higher-order and vectorized paths.**
No test differentiates twice, and none exercises `Nx.Defn.grad` inside
`Nx.vectorize`/`while`. Both are shape-and-option generators in the same way the
grad transform is, and both are therefore prime territory for gate mismatches.

**(h) f32 vs f64 as a test axis.**
Only one conv gradient test runs in f32 (`tol: 1.0e-3`). Since the fusion
compiler landed, f32 is a first-class path with different gates. Every gradient
test should run in both, or the f32 gates are unverified in the backward
direction — which is exactly the position the conv permutation gate was in.

---

## 6. Is any of this worth upstreaming?

**Partly, and the split matters.**

### 6.1 Upstream-shaped: expose what already exists

The cheapest, highest-value change to `elixir-nx/nx` is not new code — it is
**publishing `Nx.Helpers.check_grads!/4`**. Move it from `nx/test/support/helpers.ex`
into `lib/nx/testing.ex` (or a new `Nx.Testing.Grad`) so it ships in the Hex
package. `Nx.Testing` already exists, already ships, and already holds
`assert_equal/2` and `assert_all_close/3`; a gradient checker is the obvious
third member and needs no new concepts. Every backend author and every
`defn`-writing application gets JAX's `check_grads` for free. This is a small,
uncontroversial PR.

Second, smaller: **ship `Nx.Case`** (or document the three lines it contains) so
backends stop hand-rolling `Torchx.Case`/`EXLA.Case` clones that do nothing but
`import Nx.Testing`.

### 6.2 Upstream-shaped: a conformance kit

A `nx_conformance` package (separate Hex package, not part of `nx`, so its
version can move independently of Nx's) that a backend adds as a test-only dep:

```elixir
defmodule MyBackend.ConformanceTest do
  use Nx.Conformance.Case, backend: MyBackend, skip: [:complex, :f8]
end
```

and which generates, table-driven from an OpInfo-shaped registry:

- forward-value parity vs `Nx.BinaryBackend` per op × dtype × representative shapes,
- `check_grads!` on every differentiable op,
- **gradient-emitted shapes** — the transposed convs, the negative pads, the
  broadcast selects — which is the part upstream currently has nowhere,
- dtype-support declarations, so a backend states "no complex, no sub-byte" once
  instead of maintaining `@unsupported` except-lists in a doctest module.

The registry is the real artifact and the real work: an Nx `op_db`. It would
also immediately serve a purpose upstream, since `nx/test/nx/defn/grad_test.exs`
is 6031 hand-written lines that a table would compress substantially.

Realistic assessment: this is a multi-week piece of work, it needs buy-in from
the Nx maintainers to be worth anything (a conformance suite nobody blesses is
just another backend's test dir), and the honest opening move is 6.1 plus an
issue on `elixir-nx/nx` asking whether a shared kit is wanted before building it.

### 6.3 Not upstream-shaped

**The fallback counter should stay here.** Neither EXLA nor Torchx has a
universal host fallback — EXLA compiles the whole graph, Torchx raises — so
`Nx.Vulkan.Fallback` solves a problem no upstream backend has. There is no
generic hook it could attach to; it works precisely because it instruments *our*
single `host_result/2` chokepoint.

What *is* generalisable is the idea, and it belongs in prose rather than code: a
paragraph in `Nx.Backend`'s documentation stating that a backend which silently
delegates to `Nx.BinaryBackend` cannot be validated by value comparison, and
should expose either a counter or a strict mode. That is a docs PR, and it is the
part of this whole document that would help the next person most.

---

## Verified vs. inferred

**Verified** (read directly, this session):
`deps/nx` has no `test/` and its `hex_metadata.config` `files` list is `lib/**`
only; `grep -c` for grad in `deps/nx/lib/nx.ex` is 0; `Nx.Backend`'s and
`Nx.Testing`'s moduledocs; the full file listings and contents cited from a
shallow clone of `elixir-nx/nx@main` (`nx/test/`, `exla/test/`, `torchx/test/`,
`.github/workflows/ci.yml`, `nx/guides/`, `nx/test/support/helpers.ex`,
`exla/lib/exla/backend.ex`, `torchx/lib/torchx/backend.ex`); the grep counts
(`grad` → 0 hits in `torchx/test/`, 2 files in `exla/test/`; `check_grads` → 2
files repo-wide); `torch.autograd.gradcheck`'s signature and
`jax.test_util.check_grads`'s signature from their official docs; the local
counts in `test/nx_vulkan/grad_test.exs` (22 tests, 28 `grad_parity` calls) and
`lib/nx_vulkan/fallback.ex`.

**Inferred, and flagged as such:**
that upstream's lack of residency testing is *because* neither EXLA nor Torchx
has a silent-fallback design (strongly supported by their source, but it is an
attribution of motive); that a conformance kit would be accepted upstream
(unknown — no issue or discussion was found either way, and none was searched for
exhaustively); the PyTorch OpInfo details are from PyTorch's own blog and the
`op_db`/`test_ops.py` sources rather than from running the suite.

**Not confirmed:**
whether `elixir-nx/nx` has an open issue or discussion about backend conformance
testing. No search of the issue tracker was performed.

---

## See also

- `test/nx_vulkan/grad_test.exs` — the backward-pass parity suite
- `lib/nx_vulkan/fallback.ex`, `test/nx_vulkan/fallback_test.exs` — the residency counter
- `test/nx_vulkan/nx_doctest_test.exs` — the community-standard validation, and its except buckets
- `test/nx_vulkan/parity_fallback_test.exs` — coverage of the deliberately-host ops
- `docs/NX_PARITY_RESEARCH.md` — the callback-level gap analysis this complements
- `docs/VULKANO_BACKEND_ROADMAP.md` — where the native/fallback line currently sits
