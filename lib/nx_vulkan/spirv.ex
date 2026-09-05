defmodule Nx.Vulkan.Spirv do
  @moduledoc """
  Structural validation of a SPIR-V binary before it is cached or dispatched.

  ## Why this exists

  `glslangValidator` can exit **0** and still write a corrupt binary. SPIR-V's
  per-instruction word count is a **16-bit** field, so no single instruction may
  exceed 65535 words. A large `OpConstantComposite` — which is what a GLSL
  `const double[N] = double[N](...)` literal becomes — needs `N + 3` words, and
  past that ceiling glslang wraps the field instead of refusing.

  Measured on glslang 15.1.0, super-io, 2026-09-05:

      N        max instruction word count   glslang exit   binary
      65530    65533                        0              valid
      65533    7  (wrapped)                 0              CORRUPT
      65540    7  (wrapped)                 0              CORRUPT

  The consequence without this check is not a bad error message, it is two
  worse things. vulkano's parser hits `assert!(word_count >= 1)` and **panics**
  the NIF rather than returning its `ParseError`, and — because both compile
  paths cache by content hash — the corrupt artifact is written to the cache
  and reused on every subsequent run until someone deletes it by hand.

  A consumer that inlines tensor data as shader literals could cross this
  ceiling by growing its *data*, with no change to its code.

  **But do not assume this is the limit such a consumer will hit first.** The
  case that prompted this check — eXMC inlining closure-captured tensors as
  `const double[]` — turned out to sit two orders of magnitude BELOW the wrap:
  its largest single array was 1350 elements, and it failed at pipeline
  creation somewhere between 868 and 1302 elements summed across ~3 separate
  arrays. That is an aggregate driver/compiler limit, not this per-instruction
  one, and the two behave differently: below the wrap you get a clean
  `Validated` error, at or above it you get a corrupt binary glslang called
  fine. Splitting one large literal into several smaller ones evades THIS
  ceiling and does nothing for that one.

  ## What is checked

  Structure only, not semantics — that is the driver's job. This is the
  cheapest check that separates "glslang produced a file" from "glslang
  produced a SPIR-V module":

    * the file is a whole number of 32-bit words and has the 5-word header
    * the magic number is `0x07230203` (little-endian; a big-endian module
      would need byte-swapping and this project has never produced one)
    * every instruction declares `word_count >= 1` — the overflow case
    * the instruction walk lands EXACTLY on the end of the file, so a wrapped
      count that happens to stay positive is caught by the walk desynchronising
  """

  @magic 0x07230203
  @header_words 5

  @doc """
  Validate the SPIR-V at `path`.

  Returns `:ok`, or `{:error, reason}` where reason is a human-readable string
  naming the file and what is wrong with it.
  """
  def validate_file(path) do
    case File.read(path) do
      {:ok, bin} -> validate(bin, path)
      {:error, reason} -> {:error, "cannot read #{path}: #{:file.format_error(reason)}"}
    end
  end

  @doc "Validate a SPIR-V binary already in memory. `label` only appears in messages."
  def validate(bin, label \\ "<binary>")

  def validate(bin, label) when byte_size(bin) < @header_words * 4 do
    {:error, "#{label}: too short to be SPIR-V (#{byte_size(bin)} bytes, need at least 20)"}
  end

  def validate(bin, label) when rem(byte_size(bin), 4) != 0 do
    {:error, "#{label}: #{byte_size(bin)} bytes is not a whole number of 32-bit words"}
  end

  def validate(<<@magic::little-32, _rest::binary>> = bin, label) do
    <<_header::binary-size(@header_words * 4), body::binary>> = bin
    walk(body, label, 0, @header_words)
  end

  def validate(<<other::little-32, _::binary>>, label) do
    {:error,
     "#{label}: bad SPIR-V magic 0x#{Integer.to_string(other, 16)} (expected 0x07230203)"}
  end

  defp walk(<<>>, _label, _index, _word), do: :ok

  defp walk(<<word0::little-32, _rest::binary>> = body, label, index, word_offset) do
    word_count = Bitwise.bsr(word0, 16)
    opcode = Bitwise.band(word0, 0xFFFF)

    cond do
      word_count == 0 ->
        {:error,
         "#{label}: instruction #{index} (opcode #{opcode}) at word #{word_offset} declares a " <>
           "word count of 0. SPIR-V's word count is a 16-bit field, so this is almost " <>
           "certainly an instruction that needed more than 65535 words and wrapped — a " <>
           "const array of more than ~65532 elements does it. glslang emits this and " <>
           "still exits 0; the binary is corrupt and must not be cached."}

      word_count * 4 > byte_size(body) ->
        {:error,
         "#{label}: instruction #{index} (opcode #{opcode}) at word #{word_offset} declares " <>
           "#{word_count} words but only #{div(byte_size(body), 4)} remain — truncated or " <>
           "desynchronised"}

      true ->
        <<_consumed::binary-size(word_count * 4), rest::binary>> = body
        walk(rest, label, index + 1, word_offset + word_count)
    end
  end
end
