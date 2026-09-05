defmodule Nx.Vulkan.SpirvTest do
  @moduledoc """
  `glslangValidator` can exit 0 and still write a corrupt binary, because
  SPIR-V's per-instruction word count is a 16-bit field. Measured boundary on
  glslang 15.1.0 (a `const double[N]` literal needs `N + 3` words):

      N        max word count   glslang exit   binary
      65530    65533            0              valid
      65533    7  (wrapped)     0              CORRUPT

  Without this checker the corrupt artifact is written to a content-hash cache
  and reused forever, and vulkano's parser PANICS on it rather than returning
  its own ParseError.
  """
  use ExUnit.Case, async: true

  alias Nx.Vulkan.Spirv

  @magic 0x07230203

  defp header, do: <<@magic::little-32, 0x00010600::little-32, 8::little-32, 1::little-32, 0::little-32>>
  defp instr(opcode, words), do: <<Bitwise.bor(Bitwise.bsl(words, 16), opcode)::little-32>> <>
                                    :binary.copy(<<0::little-32>>, words - 1)

  describe "accepts what is actually valid" do
    test "every .spv this repo ships" do
      paths = Path.wildcard("priv/shaders/*.spv")
      assert length(paths) > 50, "expected the shipped shader set, found #{length(paths)}"

      for p <- paths do
        assert Spirv.validate_file(p) == :ok, "rejected a shipped shader: #{Path.basename(p)}"
      end
    end

    test "a minimal hand-built module" do
      assert Spirv.validate(header() <> instr(17, 2) <> instr(19, 3)) == :ok
    end

    test "a header with no instructions at all" do
      assert Spirv.validate(header()) == :ok
    end
  end

  describe "rejects the corruption glslang reports as success" do
    test "an instruction whose word count wrapped to zero" do
      # The real failure: word count 0. vulkano asserts on exactly this.
      bin = header() <> instr(17, 2) <> <<Bitwise.bsl(0, 16) |> Bitwise.bor(59)::little-32>>

      assert {:error, msg} = Spirv.validate(bin, "wrapped.spv")
      assert msg =~ "word count of 0"
      assert msg =~ "16-bit"
      assert msg =~ "wrapped.spv"
    end

    test "an instruction claiming more words than remain" do
      # Declare 900 words and supply two. `instr/2` pads to the declared length,
      # so it cannot express this — the count word is written by hand.
      bin = header() <> <<Bitwise.bor(Bitwise.bsl(900, 16), 17)::little-32>> <> <<0::little-32>>

      assert {:error, msg} = Spirv.validate(bin, "truncated.spv")
      assert msg =~ "declares 900 words"
      assert msg =~ "truncated or desynchronised"
    end

    test "a wrong magic number" do
      bin = <<0xDEADBEEF::little-32>> <> :binary.copy(<<0>>, 16)

      assert {:error, msg} = Spirv.validate(bin, "notspv.spv")
      assert msg =~ "bad SPIR-V magic"
    end

    test "a length that is not a whole number of words" do
      assert {:error, msg} = Spirv.validate(header() <> <<1, 2, 3>>, "ragged.spv")
      assert msg =~ "not a whole number of 32-bit words"
    end

    test "too short to be SPIR-V" do
      assert {:error, msg} = Spirv.validate(<<0, 0, 0, 0>>, "stub.spv")
      assert msg =~ "too short"
    end

    test "a missing file names itself" do
      assert {:error, msg} = Spirv.validate_file("/nonexistent/nope.spv")
      assert msg =~ "cannot read"
    end
  end

  describe "the walk must desynchronise, not merely bounds-check" do
    # A wrapped count that stays POSITIVE will not trip the zero check, and
    # need not overrun the buffer either — it is caught only because the walk
    # then lands mid-instruction and the next word decodes as nonsense. This
    # pins that the walk is exact rather than approximate.
    test "an instruction one word short leaves the walk misaligned" do
      # instr(17, 3) but declared as 2: the walk resumes inside it.
      bin = header() <> <<Bitwise.bor(Bitwise.bsl(2, 16), 17)::little-32>> <>
              <<0::little-32>> <> <<0::little-32>> <> instr(19, 2)

      # The stray zero word decodes as word_count 0 — caught.
      assert {:error, msg} = Spirv.validate(bin, "misaligned.spv")
      assert msg =~ "word count of 0"
    end
  end
end
