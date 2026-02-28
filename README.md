# JARVIS — Real-Time Voice Agent with Async Sub-Agents

A custom real-time voice assistant built on [LiveKit Agents v1.4](https://docs.livekit.io/agents/) with a **dual-trigger voice loop** — the agent responds to both user speech *and* background task results, creating a fluid, JARVIS-like conversational experience.

> **"I'll check your inbox and pull up the markets now, sir."**
> *(dispatches work in the background, speaks results naturally when they arrive)*

---

## What Makes This Different

Most voice agents are sequential: *user speaks → silence → response*. JARVIS breaks that pattern with a custom pipeline that separates **acknowledgement** from **results**:

1. **Instant acknowledgement** — the LLM streams a brief response to TTS within ~300ms
2. **Background execution** — tasks are dispatched to sub-agents via OpenClaw while JARVIS is already speaking
3. **Dual-trigger loop** — the agent re-activates when *either* the user speaks again *or* a sub-agent returns results
4. **Streaming context injection** — using `CONTINUE` mode, the LLM checks for new results between sentences, weaving them in naturally

```
User speaks → STT transcript → Voice LLM → Streamed TTS (instant)
                                    ↓
                          dispatch_openclaw(directive)  →  Sub-agents (async)
                          set_turn_mode(ACKWAIT)
                                    ↑
                          Queue ← sub-agent results
                          (flushed to chat history before next LLM run)
```

---

## Tech Stack

| Component | Implementation |
|-----------|----------------|
| **Transport** | [LiveKit](https://livekit.io/) (WebRTC) |
| **STT** | [Deepgram](https://deepgram.com/) (`nova-3`, streaming) |
| **LLM** | [Google Gemini](https://ai.google.dev/) (`gemini-3-flash-preview`, with thinking) |
| **TTS** | [ElevenLabs](https://elevenlabs.io/) (`eleven_flash_v2_5`, streaming) |
| **VAD** | [Silero](https://github.com/snakers4/silero-vad) |
| **Sub-agents** | [OpenClaw](https://github.com/marliechorgan/openclaw) (natural language dispatch) |

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- Node ≥ 22 (for [OpenClaw](https://github.com/marliechorgan/openclaw))
- API keys for: [LiveKit](https://cloud.livekit.io/), [Deepgram](https://console.deepgram.com/), [Google Gemini](https://aistudio.google.com/app/apikey), [ElevenLabs](https://elevenlabs.io/)

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/marliechorgan/JustANiceGuy.git
cd JustANiceGuy

# 2. Install OpenClaw (the agent system JARVIS dispatches to)
npm install -g openclaw@latest
openclaw onboard --install-daemon

# 3. Create and activate a Python virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install -e .

# 5. Download Silero VAD model files
python src/agent.py download-files

# 6. Configure environment variables
cp .env.example .env
# Edit .env and fill in your API keys

# 7. Start JARVIS
./start.sh
```

**Controls inside JARVIS:**
- Press `m` to toggle microphone mute.
- Press `l` to toggle the animated **CLI UI / Logs**.

> **Don't have OpenClaw?** Set `USE_OPENCLAW_STUB=true` in your `.env` to use
> a built-in demo that simulates agent responses with Gemini.

### Test in the LiveKit Playground

```bash
# Generate a LiveKit token
python scripts/get_token.py

# Open the playground and paste your URL + token:
# https://agents-playground.livekit.io/
```

---

## How It Works

### The Core Voice Loop

Unlike standard LiveKit `AgentSession` pipelines, JARVIS uses a **custom dual-trigger loop**:

```python
while True:
    # Wait for EITHER user speech OR sub-agent results
    trigger = await wait_for_either(user_speech, queue_update)

    # Flush any queued results into conversation history
    queue.flush_to_chat_ctx(chat_ctx)

    # Run LLM with full context → stream to TTS
    while True:
        speech, tools, mode = await stream_and_speak(llm, tts, chat_ctx)

        match mode:
            case "CONTINUE":  # LLM has more to say — re-run immediately
                queue.flush_to_chat_ctx(chat_ctx)  # check for new context
                continue
            case "ACKWAIT":   # Waiting for sub-agent results
                break
            case "CONV":      # Open mic for user
                break
            case "END":       # Close session
                return
```

### Turn Modes

The LLM controls conversation flow via the `set_turn_mode` tool:

| Mode | Behaviour |
|------|-----------|
| **ACKWAIT** | Wait for sub-agent results or user speech before speaking again |
| **CONTINUE** | Re-run LLM immediately (checks queue for new context between sentences) |
| **CONV** | Open mic — wait for user to speak |
| **END** | End the session |

### Two Tools Only

| Tool | Purpose |
|------|---------|
| `dispatch_openclaw(directive)` | Send a natural language task to the sub-agent system |
| `set_turn_mode(mode)` | Control what happens after the agent finishes speaking |

The LLM dispatches tasks in plain English — no structured parameters, no agent routing. OpenClaw handles all of that.

---

## OpenClaw Integration

JARVIS uses [OpenClaw](https://github.com/marliechorgan/openclaw) as its agent system. When you say *"Check my emails"*, JARVIS dispatches a natural language directive to OpenClaw, which routes it to the right sub-agent, executes the task, and returns results back to the voice loop.

### How it connects

JARVIS calls the `openclaw agent` CLI under the hood:

```bash
openclaw agent --agent main --message "Check Charlie's inbox" --json
```

OpenClaw's Gateway handles routing, agent selection, and execution. JARVIS never needs to know which sub-agent handles email, GitHub, calendar, etc. — it just describes what needs to happen in plain English.

### Without OpenClaw

If you don't have OpenClaw installed, set `USE_OPENCLAW_STUB=true` in your `.env`. The stub uses Gemini to simulate realistic agent responses — useful for testing the voice loop without a real agent system.

---

## Project Structure

```
├── src/
│   ├── agent.py              # Main entrypoint — voice loop, LLM streaming, TTS
│   ├── voice_queue.py        # Queue for sub-agent results (flush to chat history)
│   ├── openclaw_client.py    # Real OpenClaw integration
│   └── openclaw_stub.py      # LLM-powered simulation for demo/testing
├── scripts/
│   └── get_token.py          # Generate LiveKit tokens for playground testing
├── docs/
│   ├── architecture.md       # Full architecture specification
│   └── research.md           # Research brief and design decisions
├── pyproject.toml            # Dependencies and project config
├── .env.example              # Environment variable template
└── AGENTS.md                 # Notes for AI coding assistants
```

---

## Key Design Decisions

1. **Sub-agent results go through the LLM, not directly to TTS.** The LLM always controls *what* gets said and *how*. Sub-agents are just its hands.

2. **Results are appended to conversation history, not the system prompt.** This prevents amnesia and enables prompt caching.

3. **Natural language dispatch.** The LLM writes plain English directives — no structured tool schemas. New sub-agents can be added to OpenClaw with zero changes to the voice layer.

For the full architecture specification, see [`docs/architecture.md`](docs/architecture.md).

---

## Configuration

### Environment Variables

See [`.env.example`](.env.example) for all variables. Key settings:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LIVEKIT_URL` | ✅ | — | Your LiveKit Cloud project URL |
| `LIVEKIT_API_KEY` | ✅ | — | LiveKit API key |
| `LIVEKIT_API_SECRET` | ✅ | — | LiveKit API secret |
| `DEEPGRAM_API_KEY` | ✅ | — | Deepgram STT API key |
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key |
| `ELEVENLABS_API_KEY` | ✅ | — | ElevenLabs TTS API key |
| `GEMINI_MODEL` | | `gemini-3-flash-preview` | Gemini model to use |
| `ELEVENLABS_MODEL` | | `eleven_flash_v2_5` | ElevenLabs TTS model |
| `ELEVENLABS_VOICE_ID` | | `lUTamkMw7gOzZbFIwmq4` | Voice ID ([browse voices](https://elevenlabs.io/voice-library)) |
| `ELEVENLABS_SPEED` | | `1.14` | Speech speed (0.7–1.3) |
| `ELEVENLABS_STABILITY` | | `0.40` | Voice stability (0.0–1.0) |
| `ELEVENLABS_SIMILARITY` | | `0.75` | Voice similarity boost (0.0–1.0) |
| `USE_OPENCLAW_STUB` | | *(off)* | Set to `true` to use a simulated demo instead of real OpenClaw |
| `OPENCLAW_AGENT` | | `main` | OpenClaw agent to route directives to |
| `OPENCLAW_TIMEOUT` | | `600` | OpenClaw CLI timeout in seconds |

---

## License

[MIT](LICENSE)
