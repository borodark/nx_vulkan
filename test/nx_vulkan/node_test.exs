defmodule Nx.Vulkan.NodeTest do
  @moduledoc """
  Phase 2 — `Nx.Vulkan.Node` GenServer lifecycle and `with_node/2`
  contract. These tests exercise the GPU node directly without any
  MCMC dependency, which is the whole point of the Phase 2
  architectural split — `Nx.Vulkan` advertises a generic GPU-node
  API and these tests prove it works without `exmc`.
  """

  use ExUnit.Case, async: false

  # `whereis` then `stop` is check-then-act on a globally named process, and
  # every node here is `start_link`ed from the test process — so the node is
  # already being torn down by its link when `on_exit` runs in a *different*
  # process and tries to stop it. The pid is live at `whereis` and dead by
  # `stop`, which exits with :noproc. Nothing GPU-side is involved (the test
  # that trips it, `status/1`, never dispatches); it is pure BEAM lifecycle,
  # and it surfaces or hides depending on scheduler timing, which is why it
  # went unnoticed. Tolerate the pid dying inside the window.
  defp stop_node do
    case Process.whereis(Nx.Vulkan.Node) do
      nil -> :ok
      pid -> try_stop(pid)
    end
  end

  defp try_stop(pid) do
    GenServer.stop(pid, :normal)
  catch
    :exit, {:noproc, _} -> :ok
    :exit, :noproc -> :ok
  end

  setup do
    stop_node()

    on_exit(fn ->
      Application.delete_env(:nx_vulkan, :node_timeout_ms)
      stop_node()
    end)

    :ok
  end

  describe "lifecycle" do
    test "start_link/1 boots and registers under the default name" do
      {:ok, pid} = Nx.Vulkan.Node.start_link()
      assert Process.alive?(pid)
      assert Process.whereis(Nx.Vulkan.Node) == pid
      assert Nx.Vulkan.Node.alive?()
    end

    test "alive?/1 reports false when no node is registered" do
      refute Nx.Vulkan.Node.alive?()
    end

    test "status/1 reports uptime + exec_count" do
      {:ok, _pid} = Nx.Vulkan.Node.start_link()
      status = Nx.Vulkan.Node.status()

      assert is_map(status)
      assert status.uptime_ms >= 0
      assert status.exec_count == 0
    end

    test "stop/1 cleanly tears down" do
      {:ok, pid} = Nx.Vulkan.Node.start_link()
      assert :ok = GenServer.stop(pid, :normal)
      refute Nx.Vulkan.Node.alive?()
    end
  end

  describe "with_node/2 generic dispatch" do
    test "runs the function inside the GenServer process" do
      {:ok, _pid} = Nx.Vulkan.Node.start_link()
      result = Nx.Vulkan.Node.with_node(fn -> 42 end)
      assert result == 42
    end

    test "increments exec_count" do
      {:ok, _pid} = Nx.Vulkan.Node.start_link()
      Nx.Vulkan.Node.with_node(fn -> :first end)
      Nx.Vulkan.Node.with_node(fn -> :second end)
      Nx.Vulkan.Node.with_node(fn -> :third end)

      assert Nx.Vulkan.Node.status().exec_count == 3
    end

    test "function runs in the node's process, not the caller's" do
      {:ok, node_pid} = Nx.Vulkan.Node.start_link()
      caller = self()

      result = Nx.Vulkan.Node.with_node(fn -> {self(), caller} end)
      {actual_pid, expected_caller} = result

      assert actual_pid == node_pid
      assert expected_caller == caller
    end

    test "returns {:error, :node_dead} when no node is registered" do
      refute Nx.Vulkan.Node.alive?()
      assert Nx.Vulkan.Node.with_node(fn -> :unreached end) == {:error, :node_dead}
    end

    test "returns {:error, :node_timeout} when the function exceeds the timeout" do
      {:ok, _pid} = Nx.Vulkan.Node.start_link()
      Application.put_env(:nx_vulkan, :node_timeout_ms, 5)

      result =
        Nx.Vulkan.Node.with_node(fn ->
          Process.sleep(100)
          :unreached
        end)

      assert result == {:error, :node_timeout}
    end

    test "exceptions inside the function crash the GenServer" do
      # `with_node/2` doesn't trap exceptions — a raise inside the
      # function takes down the GenServer. The caller sees the exit
      # reason via `:exit` (not as a re-raised exception). Future
      # work could wrap fun.() in a try/rescue if we want a per-call
      # crash boundary, but for now the supervisor restart is the
      # recovery story.
      {:ok, _pid} = Nx.Vulkan.Node.start_link()

      Process.flag(:trap_exit, true)

      assert catch_exit(Nx.Vulkan.Node.with_node(fn -> raise "boom" end))
    end
  end
end
