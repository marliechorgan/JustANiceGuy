#!/usr/bin/env bash

# JARVIS Start Script
# This script makes it easy to run JARVIS in one click.

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Colors for nice output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}Starting JARVIS...${NC}"

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "Virtual environment not found! Please run the setup steps in README.md first."
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Copying from .env.example..."
    cp .env.example .env
fi

echo -e "${GREEN}JARVIS is ready!${NC}"
echo -e "Press 'l' and Enter to toggle logs."
echo -e "Press 'm' and Enter to toggle microphone mute.\n"

# Run the agent in console mode (bypasses dev auto-restart so UI is cleaner)
PYTHONPATH=src python src/agent.py dev
