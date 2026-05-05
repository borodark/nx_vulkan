#!/bin/sh
# diagnose_erlang_cluster.sh — Debug Erlang distribution connectivity
#
# Run on the node that CAN'T be reached:
#   sh diagnose_erlang_cluster.sh [peer_ip]
#
# Example: on mac-247, diagnosing why mac-248 can't connect:
#   sh diagnose_erlang_cluster.sh 192.168.0.248

PEER=${1:-"192.168.0.248"}
MY_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')

echo "=== Erlang Cluster Diagnostics ==="
echo "This host: $MY_IP"
echo "Peer host: $PEER"
echo ""

# ----------------------------------------------------------------
# 1. Basic connectivity
# ----------------------------------------------------------------
echo "--- 1. ICMP ping ---"
if ping -c 1 -t 2 "$PEER" >/dev/null 2>&1; then
    echo "[ok] ping $PEER reachable"
else
    echo "[FAIL] cannot ping $PEER — check cable/DHCP/routing"
fi
echo ""

# ----------------------------------------------------------------
# 2. EPMD
# ----------------------------------------------------------------
echo "--- 2. EPMD (port 4369) ---"
echo "  Local:"
epmd -names 2>&1 | sed 's/^/    /'
echo ""
echo "  Remote ($PEER):"
if nc -z -w 2 "$PEER" 4369 2>/dev/null; then
    echo "    [ok] port 4369 open on $PEER"
else
    echo "    [FAIL] port 4369 blocked on $PEER — epmd not reachable"
fi
echo ""

# ----------------------------------------------------------------
# 3. PF firewall
# ----------------------------------------------------------------
echo "--- 3. PF firewall ---"
if doas pfctl -s info 2>/dev/null | head -1 | grep -q "Status: Enabled"; then
    echo "  [on] pf is ENABLED"
    echo ""
    echo "  Active rules:"
    doas pfctl -sr 2>/dev/null | sed 's/^/    /'
    echo ""
    echo "  Blocked packets (last counters):"
    doas pfctl -s info 2>/dev/null | grep -i "block" | sed 's/^/    /'
else
    PF_STATUS=$(doas pfctl -s info 2>&1 | head -1)
    echo "  [off] pf is DISABLED or not running ($PF_STATUS)"
fi
echo ""

# ----------------------------------------------------------------
# 4. Bastille jails
# ----------------------------------------------------------------
echo "--- 4. Bastille jails ---"
if command -v bastille >/dev/null 2>&1; then
    JAILS=$(doas bastille list 2>/dev/null)
    if [ -n "$JAILS" ]; then
        echo "  Running jails:"
        echo "$JAILS" | sed 's/^/    /'
        echo ""
        echo "  Jail network interfaces:"
        ifconfig | grep -E "^(bastille|epair|bridge)" | sed 's/^/    /'
    else
        echo "  [ok] bastille installed but no jails running"
    fi
else
    echo "  [ok] bastille not installed"
fi
echo ""

# ----------------------------------------------------------------
# 5. BEAM distribution port
# ----------------------------------------------------------------
echo "--- 5. BEAM distribution ports ---"
BEAM_PORTS=$(epmd -names 2>/dev/null | grep "name " | awk '{print $5}')
if [ -n "$BEAM_PORTS" ]; then
    for PORT in $BEAM_PORTS; do
        NAME=$(epmd -names 2>/dev/null | grep "port $PORT" | awk '{print $2}')
        echo "  node '$NAME' listening on port $PORT"

        # Check if peer can reach this port
        echo "  testing if $PEER can reach port $PORT..."
        if nc -z -w 2 "$MY_IP" "$PORT" 2>/dev/null; then
            echo "    [ok] port $PORT open locally"
        else
            echo "    [warn] port $PORT not connectable even locally"
        fi
    done
else
    echo "  [info] no BEAM nodes registered with epmd"
    echo "  Start a node first: iex --name test@$MY_IP --cookie test"
fi
echo ""

# ----------------------------------------------------------------
# 6. /etc/pf.conf contents
# ----------------------------------------------------------------
echo "--- 6. /etc/pf.conf ---"
if [ -f /etc/pf.conf ]; then
    cat /etc/pf.conf | sed 's/^/    /'
else
    echo "  [info] no /etc/pf.conf"
fi
echo ""

# ----------------------------------------------------------------
# 7. Suggested fixes
# ----------------------------------------------------------------
echo "--- 7. Suggested fixes ---"
echo ""
echo "  A. Quick fix — allow all traffic from peer in pf:"
echo "     Add to /etc/pf.conf before any block rules:"
echo ""
echo "       pass quick from $PEER"
echo "       pass quick to $PEER"
echo ""
echo "     Then reload: doas pfctl -f /etc/pf.conf"
echo ""
echo "  B. Pin BEAM port range + open in pf:"
echo "     Start iex with:"
echo "       iex --erl \"-kernel inet_dist_listen_min 9100 inet_dist_listen_max 9200\" \\"
echo "           --name gpu2@$MY_IP --cookie zed_gpu_demo -S mix"
echo ""
echo "     Add to /etc/pf.conf:"
echo "       pass in proto tcp from $PEER to any port 4369"
echo "       pass in proto tcp from $PEER to any port 9100:9200"
echo ""
echo "     Then reload: doas pfctl -f /etc/pf.conf"
echo ""
echo "  C. Temporarily disable pf to test:"
echo "     doas pfctl -d"
echo "     # test Node.connect from peer"
echo "     doas pfctl -e"
echo ""
echo "  D. If running inside a Bastille jail:"
echo "     The jail's vnet/loopback may not route LAN traffic."
echo "     Run the BEAM on the HOST, not inside a jail."
echo ""
echo "=== Done ==="
