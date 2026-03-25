# AGENTS.md — JARVIS Voice Agent

## LiveKit Documentation

LiveKit Agents is a fast-evolving project with frequent API changes. **Always
consult the latest documentation before writing or modifying agent code.**

LiveKit provides an MCP server for AI coding assistants:

```
https://docs.livekit.io/mcp
```

Install it in Cursor by adding the following to your MCP config:

```json
{
  "livekit-docs": {
    "url": "https://docs.livekit.io/mcp"
  }
}
```

If the MCP server is unavailable, every docs page has a Markdown version
accessible by appending `.md` to the URL, e.g.:

- https://docs.livekit.io/agents/start/voice-ai-quickstart.md
- https://docs.livekit.io/agents/logic/tools.md
- https://docs.livekit.io/agents/models/tts/plugins/elevenlabs.md

---

## Project Overview

JARVIS is a custom real-time voice agent built on the **LiveKit Agents v1.4**
framework with a custom STT → LLM → TTS pipeline (not using `AgentSession`).

### Stack

| Component | Implementation |
|-----------|----------------|
| **Transport** | LiveKit WebRTC |
| **STT** | Deepgram (`nova-3`, streaming) |
| **LLM** | Google Gemini (`gemini-3-flash-preview`) |
| **TTS** | ElevenLabs (`eleven_flash_v2_5`, voice `lUTamkMw7gOzZbFIwmq4`) |
| **VAD** | Silero |
| **Agent system** | OpenClaw (stub for V1) |

### Key files

- `src/agent.py` — Agent entrypoint, plugin wiring, core voice loop
- `src/voice_queue.py` — Queue for sub-agent results
- `src/openclaw_client.py` — Real OpenClaw integration client
- `src/openclaw_stub.py` — OpenClaw stub (LLM-powered simulation)
- `scripts/get_token.py` — Helper to generate LiveKit access tokens for testing

### Documentation

- `docs/architecture.md` — Full architecture specification
- `docs/research.md` — Research brief and design decisions

---

## Architecture

The voice loop does NOT use `AgentSession`. It is a custom pipeline:

1. **Audio in** → `rtc.AudioStream` frames pushed to `stt_stream` and `vad_stream`
2. **STT feeder** (background task) → puts `FINAL_TRANSCRIPT` text into `speech_queue`
3. **VAD feeder** (background task) → sets `barge_in_event` on `START_OF_SPEECH`
4. **`voice_agent_loop`** waits for `speech_queue.get()` OR `queue.wait_for_items()`
5. **`stream_and_speak`** calls `llm.chat()`, streams text chunks to `tts_stream.push_text()`, checks `barge_in_event` for interruption
6. Tool calls (`dispatch_openclaw`, `set_turn_mode`) are extracted from `ChatChunk.delta.tool_calls`

### Turn modes (set by LLM via `set_turn_mode` tool)

| Mode | Meaning |
|------|---------|
| `ACKWAIT` | Dispatched work; wait for sub-agent results or user speech |
| `CONTINUE` | More to say; immediately re-run LLM (check queue first) |
| `CONV` | Finished; open mic for user |
| `END` | End session |

---

## API Notes (v1.4)

### LLM

```python
# llm.chat() returns LLMStream (not a coroutine — do NOT await it)
response = voice_llm.chat(chat_ctx=chat_ctx, tools=tools_context.flatten())
async for chunk in response:
    delta = chunk.delta          # ChoiceDelta | None
    if delta and delta.content:
        ...                      # str — text token
    if delta and delta.tool_calls:
        ...                      # list[FunctionToolCall]
        # FunctionToolCall.name: str
        # FunctionToolCall.arguments: str (JSON)
```

### TTS streaming

```python
stream = tts_plugin.stream()
stream.push_text(token)  # push text chunks as they arrive
stream.end_input()       # signal end of text (calls flush internally)
async for audio in stream:
    await audio_source.capture_frame(audio.frame)
# On barge-in:
await stream.aclose()    # cancel immediately
```

### STT stream

```python
stt_stream = stt_plugin.stream()
stt_stream.push_frame(audio_frame)   # from rtc.AudioStream events
async for event in stt_stream:
    if event.type == SpeechEventType.FINAL_TRANSCRIPT:
        text = event.alternatives[0].text
```

### VAD stream

```python
vad_stream = vad.stream()
vad_stream.push_frame(audio_frame)
async for event in vad_stream:
    if event.type == VADEventType.START_OF_SPEECH:
        ...
```

### Tools

```python
from livekit.agents import llm
from livekit.agents.llm.tool_context import ToolContext, find_function_tools

class MyTools:
    @llm.function_tool(description="...")
    def my_tool(self, param: str) -> None:
        pass

tools_context = ToolContext(tools=find_function_tools(MyTools()))
# Pass to LLM:
tools=tools_context.flatten()
```

### Audio source sample rate

ElevenLabs TTS default sample rate is **22050 Hz**. Always use:

```python
audio_source = rtc.AudioSource(tts_plugin.sample_rate, 1)
```

---

## Running locally

```bash
# 1. Create and activate venv
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -e .

# 3. Download Silero VAD model files
python src/agent.py download-files

# 4. Copy and fill in environment variables
cp .env.example .env

# 5. Run in dev mode
python src/agent.py dev
```

### Testing in the playground

```bash
# Generate a token with JARVIS dispatch attached
python scripts/get_token.py
# Open https://agents-playground.livekit.io/ and paste the URL + token
```

---

## Security

- **Never commit `.env`** — it is in `.gitignore`
- API keys are loaded via `python-dotenv` from `.env` in the project root
- `scripts/get_token.py` is a local dev helper only — do not expose token
  generation to the public
- The LiveKit API key and secret give full room control — treat them
  like database credentials
