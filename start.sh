#!/usr/bin/env bash

# ──────────────────────────────────────────────
# JARVIS — One-command startup
# ──────────────────────────────────────────────

set -euo pipefail
cd "$(dirname "$0")"

# Colors
R='\033[0;31m'  G='\033[0;32m'  C='\033[0;36m'  D='\033[2m'  B='\033[1m'  NC='\033[0m'

# ── Session mode ─────────────────────────────
# Usage: ./start.sh [defyner|personal]   (default: personal)
# Chosen before launch so the voice agent gets a domain-tailored system prompt
# and dispatches default to that context (Defyner repo vs life-OS root).
export JARVIS_MODE="${1:-${JARVIS_MODE:-personal}}"
if [ "$JARVIS_MODE" != "defyner" ] && [ "$JARVIS_MODE" != "personal" ]; then
    echo -e "${R}Invalid mode '$JARVIS_MODE'. Use: ./start.sh [defyner|personal]${NC}"
    exit 1
fi

echo -e "${C}${B}JARVIS${NC}  ${D}mode:${NC} ${B}${JARVIS_MODE}${NC}"
echo ""

# ── Virtual environment ──────────────────────
if [ ! -d "venv" ]; then
    echo -e "${C}Creating virtual environment...${NC}"
    python3 -m venv venv
    source venv/bin/activate
    echo -e "${C}Installing dependencies...${NC}"
    pip install -e . --quiet
    echo -e "${C}Downloading model files...${NC}"
    python src/agent.py download-files 2>/dev/null || true
    echo -e "${G}Setup complete.${NC}"
else
    source venv/bin/activate
fi

# ── Environment variables ────────────────────
if [ ! -f ".env" ]; then
    echo -e "${R}No .env file found.${NC}"
    echo -e "Run: ${B}cp .env.example .env${NC} and add your API keys."
    exit 1
fi

# Quick key validation
source .env 2>/dev/null || true
missing=()
[ -z "${LIVEKIT_URL:-}" ]        && missing+=("LIVEKIT_URL")
[ -z "${LIVEKIT_API_KEY:-}" ]    && missing+=("LIVEKIT_API_KEY")
[ -z "${LIVEKIT_API_SECRET:-}" ] && missing+=("LIVEKIT_API_SECRET")
[ -z "${GEMINI_API_KEY:-}" ]     && missing+=("GEMINI_API_KEY")
[ -z "${DEEPGRAM_API_KEY:-}" ]   && missing+=("DEEPGRAM_API_KEY")
[ -z "${ELEVENLABS_API_KEY:-}" ] && missing+=("ELEVENLABS_API_KEY")

if [ ${#missing[@]} -gt 0 ]; then
    echo -e "${R}Missing required API keys in .env:${NC}"
    for k in "${missing[@]}"; do
        echo -e "  ${R}✗${NC} $k"
    done
    echo -e "\nSee .env.example for details."
    exit 1
fi

# ── Launch ───────────────────────────────────
echo -e "${D}LLM:  ${GEMINI_MODEL:-gemini-3-flash-preview}${NC}"
echo -e "${D}TTS:  ElevenLabs (${ELEVENLABS_MODEL:-eleven_flash_v2_5})${NC}"
echo -e "${D}STT:  Deepgram${NC}"
echo -e "${D}Doer: Claude Code (${CLAUDE_MODEL:-claude-sonnet-5}) → ${JARVIS_MODE} context${NC}"
echo ""
echo -e "${G}${B}Starting...${NC} ${D}[m] mute | [l] logs | [v] paste context${NC}"
echo -e "${D}[s] Claude sessions  ·  ↑/↓ select  ·  ↵ enter session  ·  Esc release  ·  [c] copy resume${NC}"
echo ""

# exec so python replaces this shell and directly owns the TTY — no orphaned
# wrapper process, and terminal-close signals go straight to the agent.
exec env PYTHONPATH=src python src/agent.py console
