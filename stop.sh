#!/usr/bin/env bash
# Stop any running JARVIS voice agent and its in-flight Claude doer subprocesses.
# Safe to run anytime — it ONLY matches JARVIS's own processes, never your
# Claude Desktop / Claude Code app sessions.

set -uo pipefail

R='\033[0;31m'  G='\033[0;32m'  D='\033[2m'  NC='\033[0m'

killed=0

# 1) The voice agent itself.
for pid in $(pgrep -f "agent.py console" 2>/dev/null); do
    kill -TERM "$pid" 2>/dev/null && { echo -e "${G}stopped${NC} agent.py console ${D}(pid $pid)${NC}"; killed=$((killed+1)); }
done

# 2) JARVIS's headless Claude doers — scoped to the sonnet model JARVIS launches
#    with -p. This will NOT match your interactive Claude app (opus / no `-p`).
for pid in $(pgrep -f "claude -p .*claude-sonnet-4-6" 2>/dev/null); do
    kill -TERM "$pid" 2>/dev/null && { echo -e "${G}stopped${NC} claude doer ${D}(pid $pid)${NC}"; killed=$((killed+1)); }
done

sleep 1
# Force-kill any stragglers.
for pid in $(pgrep -f "agent.py console" 2>/dev/null); do
    kill -9 "$pid" 2>/dev/null && echo -e "${R}force-killed${NC} stray agent ${D}(pid $pid)${NC}"
done

if [ "$killed" -eq 0 ]; then
    echo -e "${D}No JARVIS processes were running.${NC}"
else
    echo -e "${G}Done — $killed process(es) stopped.${NC}"
fi
