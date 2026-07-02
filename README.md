# JARVIS — A Real-Time Voice Console for Agentic AI

A custom real-time voice assistant built on [LiveKit Agents v1.4](https://docs.livekit.io/agents/) with a **dual-trigger voice loop**: the agent responds to both user speech *and* background task results. Under the voice layer sits a real agent runtime. By default, every task you speak is dispatched to a **headless Claude Code session** that runs with your files, your tools, and your context, then reports back through the conversation.

> **"I'll check your inbox and pull up the markets now, sir."**
> *(dispatches work to a live agent session, speaks results naturally when they arrive)*

---

## What Makes This Different

Most voice agents are sequential: *user speaks, silence, response*. JARVIS breaks that pattern with a pipeline that separates **acknowledgement** from **results**:

1. **Instant acknowledgement.** The voice LLM streams a brief response to TTS within ~300ms.
2. **Background execution.** Tasks are dispatched to an agent backend while JARVIS is already speaking.
3. **Dual-trigger loop.** The agent re-activates when *either* the user speaks again *or* a background task returns results.
4. **Streaming context injection.** Using `CONTINUE` mode, the LLM checks for new results between sentences and weaves them in naturally.

And because the default backend is a resumable headless Claude Code session, the thing doing the work is not a toy tool router. It is a full agent with file access, shell, MCP servers, and persistent cross-turn memory. The voice layer is a console over it.

```
User speaks → STT transcript → Voice LLM → Streamed TTS (instant)
                                    ↓
                          dispatch(directive, target)  →  Headless agent session (async)
                          set_turn_mode(ACKWAIT)
                                    ↑
                          Queue ← results + status updates
                          (flushed to chat history before next LLM run)
```

---

## Tech Stack

| Component | Implementation |
|-----------|----------------|
| **Transport** | [LiveKit](https://livekit.io/) (WebRTC) |
| **STT** | [Deepgram](https://deepgram.com/) (`nova-3`, streaming) |
| **Voice LLM** | [Google Gemini](https://ai.google.dev/) (`gemini-3.5-flash`, thinking LOW) |
| **TTS** | [ElevenLabs](https://elevenlabs.io/) (`eleven_flash_v2_5`, streaming) |
| **VAD** | [Silero](https://github.com/snakers4/silero-vad) |
| **Agent backend** | **Headless [Claude Code](https://claude.com/claude-code) (default)** · [OpenClaw](https://github.com/marliechorgan/openclaw) · built-in stub |

---

## Quick Start

### Prerequisites

- Python ≥ 3.10
- API keys for: [LiveKit](https://cloud.livekit.io/), [Deepgram](https://console.deepgram.com/), [Google Gemini](https://aistudio.google.com/app/apikey), [ElevenLabs](https://elevenlabs.io/)
- For real task dispatch: the [Claude Code CLI](https://claude.com/claude-code) on your PATH (or OpenClaw, or use the stub)

### Setup

```bash
# 1. Clone and configure
git clone https://github.com/marliechorgan/JustANiceGuy.git
cd JustANiceGuy
cp .env.example .env    # Fill in your API keys

# 2. Start JARVIS (auto-creates venv, installs deps on first run)
./start.sh
```

> **No Claude Code or OpenClaw?** Set `USE_OPENCLAW_STUB=true` in your `.env` to use
> a built-in demo that simulates agent responses with Gemini.

### Controls inside JARVIS

| Key | Action |
|-----|--------|
| `m` | Toggle microphone mute |
| `l` | Toggle between the animated 3D sphere UI and raw logs |
| `s` | Toggle the **live sessions view**: every background dispatch with status, elapsed time, tool calls and cost |
| `↑/↓` + `Enter` | Select and **enter** a running session; your voice now talks to that exact session. `Esc` releases |
| `c` | Copy a `claude --resume <id>` command for the selected session, so you can drop into it in a terminal and take over by keyboard |
| `v` | Paste clipboard text into the conversation as context (articles, diagrams, code) |

### The CLI UI

The 3D sphere visualization shows real-time system status: breathing when idle, audio-reactive ripples while you speak, an orbiting light sweep while it talks. The status bar shows LLM state, per-service error tallies, and time-to-first-byte.

---

## How It Works

### The Core Voice Loop

Unlike standard LiveKit `AgentSession` pipelines, JARVIS uses a custom dual-trigger loop:

```python
while True:
    # Wait for EITHER user speech OR background results
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
            case "ACKWAIT":   # Waiting for background results
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
| **ACKWAIT** | Wait for background results or user speech before speaking again |
| **CONTINUE** | Re-run LLM immediately (checks queue for new context between sentences) |
| **CONV** | Open mic, wait for the user to speak |
| **END** | End the session |

### Two Tools Only

| Tool | Purpose |
|------|---------|
| `dispatch_openclaw(directive, target)` | Send a natural language task to the agent backend (the name is historical; it routes to whichever backend is configured) |
| `set_turn_mode(mode)` | Control what happens after the agent finishes speaking |

The voice LLM dispatches tasks in plain English. No structured parameters, no agent routing schemas, no JSON to malform. The backend agent parses intent itself. It also asks one focused clarifying question first when a request is genuinely ambiguous, rather than dispatching a half-specified task.

---

## Agent Backends

### Headless Claude Code (default)

`src/claude_client.py` shells out to `claude -p <directive> --output-format stream-json` and parses the event stream live. What this buys:

- **Real capability.** The session runs with your working directory, project instructions, memory, MCP servers and skills. It can read files, run commands, search the web, and *explain* its findings, not just execute.
- **Per-target routing.** The voice LLM picks a `target` per task (for example `personal` vs a work project). Each target maps to a working directory and, if you use separate Claude Code config worlds, its own config and billing.
- **Cross-turn memory.** One resumable session per target persists across the conversation (`--resume`), so "what did you find earlier?" just works.
- **Live narration.** Tool-use events stream back as status updates in the UI while the task runs; the final result is spoken.

### OpenClaw

Set `USE_OPENCLAW=true` to route directives to the [OpenClaw](https://github.com/marliechorgan/openclaw) CLI instead. Same contract: one natural-language directive in, results pushed to the queue.

### Stub

Set `USE_OPENCLAW_STUB=true` for a Gemini-simulated backend. Useful for testing the voice loop with zero setup.

All three implement the same abstraction: `dispatch_fn(directive, queue)`. Adding a new backend means writing one function.

---

## Live Session Observability

The part that turns this from a demo into a usable console: you can **see and steer** what the background agents are doing.

- Press `s` for the sessions panel: every dispatch with status, target, elapsed time, tool count, cost, and a rolling feed of tool calls with their inputs ("Read src/agent.py", "Bash git log").
- **Enter a session** with the arrow keys and your voice now continues that exact session, not the default one.
- Every JARVIS session is a real on-disk Claude Code session. Press `c` to copy a `claude --resume` command and take over the same session in a terminal, mid-task, with full history.
- The LLM itself is session-aware: it receives a pruned summary of running sessions each turn and has a `peek_session` tool to read a session's recent turns without dispatching new work.

The handoff works in both directions: voice to terminal, terminal back to voice. The agent session is the durable thing; the interfaces are views onto it.

---

## Project Structure

```
├── src/
│   ├── agent.py              # Main entrypoint: voice loop, LLM streaming, TTS
│   ├── voice_queue.py        # Queue for background results (flush to chat history)
│   ├── claude_client.py      # Headless Claude Code backend (default)
│   ├── openclaw_client.py    # OpenClaw backend
│   ├── openclaw_stub.py      # LLM-powered simulation for demo/testing
│   └── cli_ui.py             # Sphere UI + live sessions panel
├── scripts/
│   └── get_token.py          # Generate LiveKit tokens for playground testing
├── docs/
│   ├── architecture.md       # Original architecture specification
│   └── research.md           # Historical build brief
├── start.sh / stop.sh        # Run + rescue scripts
├── .env.example              # Environment variable template (all config lives here)
└── AGENTS.md                 # Notes for AI coding assistants
```

---

## Key Design Decisions

1. **Background results go through the LLM, not directly to TTS.** The LLM always controls *what* gets said and *how*. The backend agents are its hands.

2. **Results are appended to conversation history, not the system prompt.** This prevents amnesia and keeps the system prompt static for prompt caching.

3. **Natural language dispatch.** The voice LLM writes plain English directives. No tool schemas to maintain, and new backend capability needs zero changes to the voice layer.

4. **The agent session is the source of truth.** Voice, the sessions panel, and a terminal `--resume` are all views onto the same persistent session. You can start a task by voice and finish it by keyboard.

5. **Speech is bounded, directives are not.** A spoken-character cap keeps answers to a few sentences (with `CONTINUE` for more), while tool-call output has a separate generous budget so dispatch directives can be long and precise.

For the original architecture specification, see [`docs/architecture.md`](docs/architecture.md).

---

## Configuration

All configuration is documented inline in [`.env.example`](.env.example): required API keys, voice tuning, Gemini model and thinking level, spoken-length caps, and backend selection (Claude Code model, timeout, permission mode and target directories; OpenClaw agent and timeout; stub mode).

A note on permissions: the headless backend defaults to `bypassPermissions` because a voice agent has no TTY to approve prompts on. The directive prefix constrains sessions to read-and-investigate work (no sends, pushes, deletes or purchases). Adjust `CLAUDE_PERMISSION_MODE` to your own risk tolerance.

---

## License

[MIT](LICENSE)
