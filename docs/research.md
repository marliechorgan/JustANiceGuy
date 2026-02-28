# Research Brief: JARVIS Voice Agent — Build from LiveKit Starter

**Purpose:** This document provides full context for a research AI agent to determine the optimal approach for building the JARVIS voice agent described in `jarvis-voice-architecture.md`, starting from the official LiveKit agent starter and avoiding overcomplication.

---

## 1. Target: What We Want to Build

The full specification is in **`jarvis-voice-architecture.md`**. Summary of the core requirements:

### Architecture Overview

```
User Voice → STT → Voice LLM → TTS (streamed)
                  ↓
            dispatch_openclaw(directive)  →  OpenClaw → sub-agents
            set_turn_mode(mode)            →  Turn controller
                  ↑
            Voice LLM Queue ← sub-agent results
            (flushed to conversation history before each LLM run)
```

### Key Behaviors (Non-Negotiable)

1. **Two triggers for the next LLM run:**
   - User speaks (STT final transcript)
   - Sub-agent returns results (queue has items)
   - Implemented as `wait_for_either(user_speech, queue.wait_for_items())`

2. **Turn modes** (LLM calls `set_turn_mode` or defaults apply):
   - `ACKWAIT` — Wait for queue or user; do NOT re-run LLM until one of those
   - `CONTINUE` — Re-run LLM immediately (after flushing queue); agent has more to say
   - `CONV` — Open mic; wait for user
   - `END` — Close session

3. **Sub-agent results go to conversation history**, not system prompt. Queue items are flushed as tool/user messages before each LLM run. This enables prompt caching and prevents amnesia.

4. **Two tools only:**
   - `dispatch_openclaw(directive: str)` — Fire-and-forget; OpenClaw runs async, pushes to queue
   - `set_turn_mode(mode: ACKWAIT|CONTINUE|CONV|END)` — Controls what happens after speech

5. **Streaming:** LLM text streams directly to TTS. Tool calls come after text. No JSON parsing.

6. **Barge-in:** VAD detects user speech during TTS → stop TTS, mark `[INTERRUPTED]`, process new user input.

### Tech Stack (from architecture)

| Component | Choice |
|-----------|--------|
| Transport | LiveKit (WebRTC) |
| STT | Deepgram (streaming) |
| LLM | Gemini 3 Flash (or Claude/GPT-4o — native tool calling) |
| TTS | ElevenLabs (streaming, voice `lUTamkMw7gOzZbFIwmq4`) |
| VAD | Silero |
| Sub-agent system | OpenClaw (stub for V1) |

---

## 2. Baseline: Official LiveKit Agent Starter

**Repo:** https://github.com/livekit-examples/agent-starter-python

### Structure

```
agent-starter-python/
├── src/
│   └── agent.py          # Single entrypoint
├── tests/
├── pyproject.toml
├── .env.example
├── AGENTS.md
├── Dockerfile
└── README.md
```

### How the Starter Works

```python
# agent.py (simplified)
from livekit.agents import AgentServer, AgentSession, Agent, JobContext, room_io
from livekit.plugins import silero, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice AI assistant.")

server = AgentServer()

@server.rtc_session(agent_name="my-agent")
async def my_agent(ctx: JobContext):
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language="multi"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(model="cartesia/sonic-3", voice="..."),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
        preemptive_generation=True,
    )
    await session.start(agent=Assistant(), room=ctx.room, room_options=...)
    await ctx.connect()
```

- Uses **LiveKit Inference** (hosted models via LiveKit Cloud) — no STT/LLM/TTS API keys
- **AgentSession** is a black-box pipeline: STT → LLM → TTS with automatic turn detection
- Flow: user speaks → turn detected → LLM generates → TTS speaks → repeat
- No explicit "wait for queue" or "wait for user" — it's always "wait for user turn"

### Starter Dependencies (pyproject.toml)

```toml
dependencies = [
  "livekit-agents[silero,turn-detector]~=1.4",
  "livekit-plugins-noise-cancellation~=0.2",
  "python-dotenv",
]
```

---

## 3. Critical Gap: AgentSession vs JARVIS Flow

| Aspect | LiveKit AgentSession | JARVIS Architecture |
|--------|----------------------|----------------------|
| Trigger for LLM run | User turn (VAD + turn detector) | User speech **OR** queue has items |
| After agent speaks | Waits for user | May wait for queue (ACKWAIT) or re-run immediately (CONTINUE) |
| Sub-agent results | N/A | Must be flushed to conversation history before next LLM run |
| Turn control | Implicit (turn detector) | Explicit (`set_turn_mode`) |
| Tools | Supported | `dispatch_openclaw`, `set_turn_mode` — tools control flow, not just data |

**Core question:** Can `AgentSession` be configured or extended to support:
- A second trigger (queue items) in addition to user speech?
- Explicit turn modes (ACKWAIT, CONTINUE) that change when the next LLM run happens?
- Injecting queue items into conversation history before each LLM run?

If not, the implementation will require a **custom voice loop** that does not use `AgentSession` as the main orchestrator, but may still use:
- `AgentServer` + `JobContext` for room connection and job lifecycle
- LiveKit plugins for STT, TTS, VAD (as streams)
- LLM plugin for `chat()` with tools

---

## 4. Model Choices: Inference vs Plugins

| | LiveKit Inference | Plugins |
|---|------------------|---------|
| STT | `inference.STT("deepgram/nova-3")` | `deepgram.STT(api_key=...)` |
| LLM | `inference.LLM("openai/gpt-4.1-mini")` | `google.LLM(model="gemini-3-flash-preview", api_key=...)` |
| TTS | `inference.TTS("cartesia/sonic-3", ...)` | `elevenlabs.TTS(api_key=..., voice_id=..., ...)` |
| API keys | Only LiveKit (LIVEKIT_URL, API_KEY, SECRET) | LiveKit + Deepgram + Gemini + ElevenLabs |

The user has `.env` with `DEEPGRAM_API_KEY`, `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`. So we need **plugins**, not LiveKit Inference.

---

## 5. Research Questions for the AI Agent

1. **AgentSession extensibility**
   - Does `AgentSession` support a custom "trigger" (e.g. queue) in addition to user speech?
   - Can we hook into the session to inject messages (queue items) before each LLM call?
   - Is there a way to implement ACKWAIT (don't re-run LLM until queue or user) and CONTINUE (re-run immediately) within the session model?

2. **Minimal customization path**
   - If AgentSession can be extended: what is the smallest set of changes (hooks, options, event handlers) to achieve the JARVIS flow?
   - If AgentSession cannot: what is the minimal code we can reuse from the starter (AgentServer, JobContext, room setup) and what must be custom (voice loop, queue, turn logic)?

3. **Official examples**
   - Are there LiveKit examples that implement:
     - Tools that affect control flow (not just return data)?
     - Waiting on external events (e.g. queue) before the next LLM run?
     - A custom pipeline that bypasses or wraps AgentSession?
   - Check: https://github.com/livekit/agents (main repo examples), https://github.com/livekit-examples

4. **Recommended approach**
   - Option A: Clone starter → add tools + queue + minimal session customization
   - Option B: Clone starter → replace `AgentSession` with custom loop, keep AgentServer/JobContext/plugins
   - Option C: Something else (e.g. use `AgentSession` for base pipeline but run a separate "queue watcher" task that triggers `session.generate_reply()` when queue has items)

5. **Avoiding overcomplication**
   - What is the simplest implementation that satisfies the architecture?
   - Are there features in the architecture we can defer to V2 (e.g. status updates, CONTINUE mid-thought) to get a working V1 faster?

---

## 6. Resources to Consult

- **LiveKit Docs MCP:** https://docs.livekit.io/mcp (if available)
- **Voice AI quickstart:** https://docs.livekit.io/agents/start/voice-ai-quickstart.md
- **Agent sessions:** https://docs.livekit.io/agents/logic/sessions.md
- **Tools:** https://docs.livekit.io/agents/logic/tools.md
- **Agent starter Python:** https://github.com/livekit-examples/agent-starter-python
- **LiveKit agents repo (examples):** https://github.com/livekit/agents/tree/main/examples
- **Architecture spec:** `jarvis-voice-architecture.md` (in this repo)

---

## 7. Constraints

- **Start from the official starter** — clone it, don't build from scratch
- **Don't overcomplicate** — prefer the simplest path that works
- **V1 scope:** Get a working voice agent that:
  - Connects via LiveKit
  - Uses Deepgram STT, Gemini LLM, ElevenLabs TTS
  - Has `dispatch_openclaw` and `set_turn_mode` tools
  - Implements the `wait_for_either` pattern (user or queue)
  - Flushes queue to conversation history
  - Handles turn modes (at least ACKWAIT, CONV; CONTINUE can be simplified for V1)
- **OpenClaw:** Use a stub that yields fake results; real integration later
- **Security:** No hardcoded secrets; use `.env` and `.gitignore`

---

## 8. Current Repo State

- **`jarvis-voice-architecture.md`** — Full architecture specification (target)
- **`RESEARCH-BRIEF.md`** — This document
- **`jarvis/`** — Previous custom implementation (to be discarded). The user wants to **undo all of this** and start fresh from the LiveKit starter.

**Plan:** Clone `livekit-examples/agent-starter-python` into a new directory (or replace `jarvis/`), then customize per the research agent's recommendation.

---

## 9. Deliverable for the Research Agent

After research, produce:

1. **Recommendation:** Option A, B, or C (or a refined variant), with justification
2. **Implementation plan:** Step-by-step changes to the cloned starter
3. **Risks and simplifications:** What to defer to V2 if needed
4. **File-level changes:** Which files to add/modify/remove

This brief should give the research agent everything needed to produce a clear, actionable plan.
