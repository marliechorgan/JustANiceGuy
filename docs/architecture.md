# JARVIS Voice Architecture: Real-Time Voice Agent with Async Sub-Agents

## The Core Idea (in one paragraph)

When the user speaks, their audio is transcribed (STT) and sent to a Voice LLM. The LLM streams natural conversational text **directly to TTS** — no JSON wrapping, no parsing layer, no fragile streaming hacks. The user hears JARVIS respond in ~300ms. If the LLM needs to dispatch work, it calls a single native tool: `dispatch_openclaw(directive)`, passing a clean natural language task string. OpenClaw receives the directive, handles all routing, agent selection, and execution autonomously — the Voice LLM never needs to know the internal API surface of any sub-agent. Results flow back into the **conversation history** as tool messages, giving the LLM persistent memory of everything that's happened. The LLM controls pacing through a `set_turn_mode` tool: `ACKWAIT` (wait for sub-agent results), `CONTINUE` (re-run immediately, checking the queue for new context before speaking the next 1–2 sentences), `CONV` (open mic), or `END` (close session). The result: a voice agent that acknowledges instantly, works in the background via OpenClaw, and speaks results naturally — like JARVIS.

---

## Why This Is Different

Every voice agent today is sequential:

```
User speaks → Silence → STT → LLM thinks → TTS → User hears response
```

This architecture breaks that by separating **acknowledgement** from **results**. The LLM talks immediately ("I'll check that for you, sir"), the real work happens async via OpenClaw, and results flow back through the LLM — not directly to TTS — so the LLM always controls what gets said and how.

---

## The NL Handoff: Why There Are No Structured Tool Calls

Forcing a conversational Voice LLM to act as a strict JSON router (`{"agent": "github", "action": "recent_commits", "params": {"limit": 5}}`) is a massive waste of inference time. It increases Time-To-First-Token (TTFT) latency, burns tokens on schema generation, and introduces the risk of malformed JSON crashing the system.

OpenClaw already handles natural language prompts. It already knows how to parse intent, route to the correct sub-agent, and execute multi-step workflows. Making the Voice LLM duplicate that work — translating messy spoken words into rigid JSON structures with exact agent names, action strings, and parameter objects — is redundant and slow.

The Voice LLM acts solely as the **Front Desk Receptionist**. It hears the user, translates their spoken words into a clean natural language directive for OpenClaw, and goes back to talking. One tool. One string:

```python
dispatch_openclaw(
    directive="Check the user's recent GitHub commits and search their inbox "
              "for emails from the label manager about a contract"
)
```

OpenClaw takes it from there. The Voice LLM never needs to know which sub-agent handles email, what parameters a GitHub search takes, or how to format a calendar query. It just describes what needs to happen in plain English and lets the orchestrator do its job.

This cuts TTFT latency, eliminates malformed JSON errors entirely, and means adding new sub-agents to OpenClaw requires **zero changes** to the Voice LLM's tool schema.

---

## Native Streaming: Why There Is No JSON Parsing

A custom streaming JSON approach — forcing the LLM to output a JSON object like `{"speech": "...", "tool_calls": [...]}` and parsing it mid-flight with a streaming parser — is highly brittle. If the LLM generates an unescaped quote mark, a trailing comma, or any malformed JSON, the stream parser crashes and the voice agent dies. It also adds a parsing layer between the LLM and TTS that introduces latency and complexity for no gain.

Because we simplified the tool interface down to `dispatch_openclaw(directive)` and `set_turn_mode(mode)`, we can use **native LLM tool calling** — supported natively by Claude 3.5 Sonnet and GPT-4o.

Native tool calling works like this:

1. The LLM **streams standard conversational text first**. These raw text chunks are piped directly to the TTS engine with zero parsing logic.
2. **After the text is complete**, the LLM outputs tool call blocks (dispatch directives, turn mode) in its native structured format — handled by the API, not by us.

The text-first streaming behaviour is inherent to how these models handle tool use. There is no custom parser to crash, no JSON to malform, no `jiter` or `ijson` dependency. Latency drops to the absolute floor — the TTS receives the first token the instant the LLM generates it — and the system is 100% crash-proof.

---

## The Architecture

### System Flow

```
┌─────────────┐
│  User Voice  │
└──────┬──────┘
       │ audio stream (via LiveKit WebRTC)
       ▼
┌──────────────────┐
│  STT Engine      │
│  (+ LiveKit VAD  │
│   + barge-in     │
│   detection)     │
└──────┬───────────┘
       │ transcript
       ▼
┌──────────────────────────────────────────────────┐
│                 VOICE LLM                         │
│                                                  │
│  Inputs:                                         │
│   - Conversation history (includes all prior     │
│     agent results as tool messages)              │
│   - Static system prompt (cacheable)             │
│                                                  │
│  Outputs:                                        │
│   1. Streamed text → piped directly to TTS       │
│   2. Native tool calls (after text completes):   │
│      - dispatch_openclaw(directive: str)          │
│      - set_turn_mode(mode: str)                  │
│                                                  │
└──────┬──────────┬───────────────┬────────────────┘
       │          │               │
       ▼          ▼               ▼
  ┌─────────┐  ┌──────────┐  ┌────────────────────┐
  │   TTS   │  │ OpenClaw │  │  Turn Controller    │
  │         │  │          │  │                     │
  │ Speaks  │  │ Receives │  │ ACKWAIT:  wait for  │
  │ streamed│  │ NL       │  │   queue or user     │
  │ text as │  │ directive│  │ CONV: open mic      │
  │ chunks  │  │ & routes │  │ CONTINUE: re-run    │
  │ arrive  │  │ to sub-  │  │   LLM immediately   │
  │         │  │ agents   │  │ END: close session  │
  └─────────┘  └────┬─────┘  └────────────────────┘
                    │ sub-agents execute async
                    ▼
          ┌──────────────────┐
          │  Sub-Agent Pool   │
          │  (OpenClaw)       │
          │                  │
          │  email_agent()   │
          │  github_agent()  │
          │  coding_agent()  │
          │  calendar_agent()│
          └────────┬─────────┘
                   │
                   │ results + status updates
                   ▼
          ┌──────────────────────┐
          │   VOICE LLM QUEUE    │
          │                      │
          │ Accumulates results. │
          │ On flush, items are  │
          │ appended to conver-  │
          │ sation history as    │
          │ tool messages.       │
          └──────────┬───────────┘
                     │
                     │ triggers next LLM run
                     ▼
              ┌──────────────┐
              │  VOICE LLM   │  ← runs again with full history
              │  (next cycle) │
              └──────────────┘
```

### Three Key Design Decisions

**1. Sub-agent output goes through the LLM, not directly to TTS.**
Sub-agent results do not get spoken directly to the user. They enter the conversation history, and the LLM decides how to present the information — what to say, in what order, what to emphasise. The LLM is always the brain *and* the mouth. Sub-agents are just its hands.

**2. Agent results are appended to conversation history, not the system prompt.**
When the queue flushes, items are appended to `conversation_history` as tool messages — not injected into a mutating system prompt. This prevents the "amnesia bug" (the LLM forgetting data from two turns ago because the system prompt was overwritten). It also keeps the system prompt **static**, which enables prompt caching on Claude and GPT-4o — slashing token costs by ~80% and reducing latency on every subsequent turn.

**3. The Voice LLM dispatches natural language, not structured tool schemas.**
The LLM never constructs agent names, action strings, or parameter objects. It writes a single natural language directive and lets OpenClaw handle the rest. This eliminates an entire class of errors, reduces TTFT latency, and means new sub-agents can be added to OpenClaw with zero changes to the voice layer.

---

## The Voice LLM Tool Interface

### Native Tool Definitions

These are passed to Claude / GPT-4o as standard tool definitions. The LLM calls them natively — no custom JSON parsing required.

```python
VOICE_TOOLS = [
    {
        "name": "dispatch_openclaw",
        "description": (
            "Send a task to the OpenClaw agent system. Write a clear, "
            "natural language directive describing what needs to be done. "
            "OpenClaw will route to the appropriate sub-agent(s) automatically. "
            "You do not need to specify agent names or parameters — just "
            "describe the task in plain English."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "directive": {
                    "type": "string",
                    "description": "A clear natural language task description."
                }
            },
            "required": ["directive"]
        }
    },
    {
        "name": "set_turn_mode",
        "description": (
            "Control what happens after you finish speaking. Call this every "
            "turn to set the system's next state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["ACKWAIT", "CONTINUE", "CONV", "END"],
                    "description": (
                        "ACKWAIT: you dispatched agents and need to wait for "
                        "results before speaking again. CONTINUE: you have more "
                        "to say — the system will re-run you immediately and "
                        "inject any new context. CONV: you've finished your "
                        "thought, open the mic for the user. END: conversation over."
                    )
                }
            },
            "required": ["mode"]
        }
    }
]
```

### Turn Mode Defaults

If the LLM doesn't explicitly call `set_turn_mode`:
- **If `dispatch_openclaw` was called** → defaults to `ACKWAIT` (wait for results)
- **If no tools were called** → defaults to `CONV` (open mic for user)

This means for the two most common cases — dispatching work and casual conversation — the LLM doesn't even need to call `set_turn_mode`. It only needs it for `CONTINUE` (more to say) and `END` (close session).

### The Streaming Flow

```
LLM generates:

  "I'll pull your inbox and check the markets now, sir."  ← text tokens
                                                            streamed to TTS
                                                            in real-time

  [tool_call: dispatch_openclaw(                           ← native tool call
      directive="Check my inbox and look up                 processed after
                  current Tesla stock price"                            text completes
  )]

  [tool_call: set_turn_mode(mode="ACKWAIT")]               ← turn mode set
```

The user hears the acknowledgement within ~300ms of the LLM starting to generate. By the time the tool calls are processed, JARVIS has already spoken and OpenClaw is already working.

---

## Turn Modes Explained

### `ACKWAIT` — Acknowledge and Wait

The LLM has said something like *"Let me check that for you"* and dispatched a directive to OpenClaw. The system should **not** keep generating more speech. It waits until either:
- A sub-agent returns results (flushed from queue into conversation history → triggers next LLM run)
- The user speaks again (new transcript → triggers next LLM run)

This prevents the agent from rambling or filling silence with filler while it waits for data.

### `CONV` — Conversation (Open Mic)

Normal conversational flow. The LLM has finished its thought and yields the floor to the user. Used when:
- No directives were dispatched (pure conversation)
- The LLM has finished presenting results and is waiting for the user's next instruction

### `CONTINUE` — Keep Speaking (The Breathing Agent)

This is what makes the agent feel alive. The LLM has spoken 1–2 sentences but has more to say. Instead of dumping a wall of text into TTS, it sets `CONTINUE`, and the system **immediately re-runs the LLM** for the next chunk.

Why this matters: before generating sentences 3 and 4, the LLM checks the queue. If a sub-agent result arrived during sentences 1 and 2, the LLM sees it and can adapt mid-thought:

```
LLM run 1: "You've got three commits today on the ACI repo."
           set_turn_mode("CONTINUE")

           [Queue check: email_agent just returned a result]

LLM run 2: "Most recent was the webhook handler about an hour ago. 
            Oh — and the label manager just got back to you with a 
            revised contract. Want me to pull it up?"
           set_turn_mode("CONV")
```

The user hears a single, fluid response. But under the hood, the LLM paused between sentences, checked for new context, and wove it in seamlessly. This is the **stream-of-consciousness effect** — the agent sounds like it's thinking in real-time.

**The 1–2 sentence rule:** The Voice LLM's system prompt instructs it to output only 1–2 sentences per turn and use `CONTINUE` if it has more to say. This creates natural breathing room for context injection and prevents monotonous walls of TTS audio.

### `END` — End Session

The conversation is wrapping up. *"Alright, catch you later."* The system closes the voice session and tears down the LiveKit room.

### Why `turn_mode` Is Non-Negotiable

Without explicit turn mode control, the Python loop has to *guess* what state it should be in based on the presence of tool calls. But tool calls don't always mean "wait":

- *"Send that email to the label manager, and by the way, what's my next meeting?"* — The LLM dispatches the email task (background) **but** should answer the calendar question immediately (`CONV`, not `ACKWAIT`).
- *"Scan my entire codebase."* — The LLM says "Scanning now..." and should sit quietly until data arrives (`ACKWAIT`).

`set_turn_mode` gives the LLM explicit state-machine control. It prevents fragile `if dispatched and not should_continue and not...` logic in Python.

---

## The Voice LLM Queue

The Voice LLM Queue accumulates sub-agent results and status updates between LLM runs. When flushed, items are **appended to the conversation history as tool messages** — not injected into the system prompt.

### Why Conversation History, Not System Prompt

If you inject queue items into the system prompt:
- Turn 2: LLM reads the email data in `<PENDING_CONTEXT>`. Great.
- Turn 3: Queue is flushed, system prompt resets. The LLM has **completely forgotten** the email it just summarised. The user says "what was in that contract?" and the LLM has no idea.

By appending to conversation history instead:
- The data persists across all future turns — no amnesia.
- The system prompt stays **static** — enabling prompt caching (Claude's prompt caching, GPT-4o's cached prefixes). This slashes token costs by ~80% and reduces latency on every turn after the first.

### Queue Implementation

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class QueueItem:
    """A single item in the Voice LLM Queue."""
    type: Literal["agent_result", "status_update"]
    agent: str
    content: dict
    summary: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


class VoiceLLMQueue:
    def __init__(self):
        self.items: list[QueueItem] = []
        self._event = asyncio.Event()

    def push(self, item: QueueItem):
        """Add an item to the queue and signal any waiters."""
        self.items.append(item)
        self._event.set()

    def flush_to_history(self, conversation_history: list[dict]):
        """Drain the queue and append all items to conversation history
        as tool messages. Called before each LLM run."""
        for item in self.items:
            conversation_history.append({
                "role": "tool",
                "content": f"[{item.type.upper()} | {item.agent} | "
                           f"{item.timestamp.isoformat()}]\n{item.summary}"
            })
        self.items.clear()
        self._event.clear()

    def has_items(self) -> bool:
        return len(self.items) > 0

    async def wait_for_items(self):
        """Block until the queue receives at least one item."""
        await self._event.wait()
```

### What Queue Items Look Like in Conversation History

After flushing, the conversation history contains tool messages alongside normal user/assistant turns:

```python
[
    {"role": "user", "content": "Check my GitHub and inbox"},
    {"role": "assistant", "content": "I'll have a look now, sir."},
    {"role": "tool", "content": "[AGENT_RESULT | github_agent | 2026-02-20T14:32:03Z]\n3 commits today on ACI repo. Most recent: 'add webhook handler' (1 hour ago)."},
    {"role": "tool", "content": "[AGENT_RESULT | email_agent | 2026-02-20T14:32:01Z]\nFound reply from Label Manager. Subject: 'Revised Contract'. Attachment: contract_v2.pdf."},
    {"role": "assistant", "content": "Right — three commits today, most recent was the webhook handler..."},
    {"role": "user", "content": "What did they change in the contract?"},
    # The LLM can still see the email result from 4 messages ago — no amnesia.
]
```

---

## The Core Loop

### Pseudocode

```python
async def voice_agent_loop(
    stt: STTEngine,
    llm: VoiceLLM,
    tts: TTSEngine,
    openclaw: OpenClawOrchestrator,
    queue: VoiceLLMQueue,
    conversation_history: list[dict],
    livekit_room: LiveKitRoom
):
    while True:
        # ── WAIT FOR A TRIGGER ──────────────────────────────
        trigger = await wait_for_either(
            user_speech=stt.listen(),
            queue_update=queue.wait_for_items()
        )

        # ── HANDLE USER SPEECH ──────────────────────────────
        if trigger.type == "user_speech":
            conversation_history.append({
                "role": "user",
                "content": trigger.transcript
            })

        # ── FLUSH QUEUE INTO CONVERSATION HISTORY ───────────
        queue.flush_to_history(conversation_history)

        # ── RUN THE VOICE LLM (with CONTINUE loop) ─────────
        while True:
            speech, tool_calls, turn_mode = await stream_and_speak(
                llm=llm,
                tts=tts,
                system_prompt=VOICE_AGENT_PROMPT,
                messages=conversation_history,
                tools=VOICE_TOOLS,
                livekit_room=livekit_room
            )

            conversation_history.append({
                "role": "assistant",
                "content": speech
            })

            # Dispatch any OpenClaw directives (async, fire and forget)
            for tc in tool_calls:
                if tc.name == "dispatch_openclaw":
                    asyncio.create_task(
                        dispatch_to_openclaw(tc.directive, openclaw, queue)
                    )

            # ── TURN MODE LOGIC ─────────────────────────────
            match turn_mode:

                case "CONTINUE":
                    queue.flush_to_history(conversation_history)
                    continue

                case "ACKWAIT":
                    break

                case "CONV":
                    break

                case "END":
                    await livekit_room.disconnect()
                    return


async def stream_and_speak(
    llm, tts, system_prompt, messages, tools, livekit_room
):
    """
    Stream the LLM's text output directly to TTS. After text completes,
    extract native tool calls. Handles barge-in via LiveKit VAD.

    Returns (speech_text, tool_calls, turn_mode).
    """
    speech = ""

    response = llm.stream(
        system_prompt=system_prompt,
        messages=messages,
        tools=tools
    )

    # Text streams first — pipe every chunk straight to TTS
    async for text_chunk in response.text_stream:
        speech += text_chunk
        await tts.stream_chunk(text_chunk)

        if livekit_room.vad_triggered():
            await tts.stop()
            await response.cancel()
            return f"{speech} [INTERRUPTED]", [], "CONV"

    # Text done — extract tool calls from the response
    tool_calls = response.get_tool_calls()
    turn_mode = resolve_turn_mode(tool_calls)

    return speech, tool_calls, turn_mode


def resolve_turn_mode(tool_calls: list) -> str:
    """Determine turn mode from tool calls, applying defaults."""
    for tc in tool_calls:
        if tc.name == "set_turn_mode":
            return tc.mode

    # No explicit set_turn_mode — apply defaults
    has_dispatch = any(tc.name == "dispatch_openclaw" for tc in tool_calls)
    return "ACKWAIT" if has_dispatch else "CONV"


async def dispatch_to_openclaw(
    directive: str,
    openclaw: OpenClawOrchestrator,
    queue: VoiceLLMQueue
):
    """
    Send a natural language directive to OpenClaw. OpenClaw parses
    intent, routes to sub-agents, and executes. Status updates and
    final results are pushed into the Voice LLM Queue.
    """
    async for update in openclaw.execute(directive):

        if update.is_status:
            queue.push(QueueItem(
                type="status_update",
                agent=update.agent_name,
                content=update.data,
                summary=update.message
            ))

        elif update.is_result:
            queue.push(QueueItem(
                type="agent_result",
                agent=update.agent_name,
                content=update.data,
                summary=update.summary
            ))
```

### The `wait_for_either` Pattern

The system sits idle until **one of two things** happens:

```python
async def wait_for_either(user_speech, queue_update):
    """Wait for whichever comes first: user speaks or queue gets new items."""
    done, pending = await asyncio.wait(
        [asyncio.create_task(user_speech),
         asyncio.create_task(queue_update)],
        return_when=asyncio.FIRST_COMPLETED
    )

    for task in pending:
        task.cancel()

    return done.pop().result()
```

This is what makes `ACKWAIT` work. After the LLM says *"I'll check your emails, sir"*, the loop doesn't re-run the LLM. It waits. When the email agent returns results into the queue, the loop triggers, the queue is flushed into conversation history, and the LLM runs again with full context.

---

## LiveKit: Why WebRTC From Day 1

Do not build this over raw WebSockets. Use LiveKit (WebRTC) from the start. Three reasons:

### 1. Acoustic Echo Cancellation (AEC)
With raw WebSockets, the agent will hear its own TTS playing from the user's speakers, transcribe it, and talk to itself in an infinite loop. LiveKit provides AEC out of the box.

### 2. Voice Activity Detection (VAD)
LiveKit detects when the user starts speaking, even while TTS is playing. This is essential for barge-in handling.

### 3. Barge-In (User Interrupts the Agent)
When the user interrupts:

1. LiveKit's VAD triggers → the system sends `.stop()` to the TTS engine immediately.
2. The system calculates approximately how much text the TTS managed to speak.
3. Only the **spoken portion** is appended to conversation history (marked as `[INTERRUPTED]`).
4. The new user transcript is appended.
5. The LLM re-runs with accurate context — it knows it was interrupted and what the user said.

```
Conversation history after barge-in:

{"role": "assistant", "content": "You've got three commits today on the ACI-- [INTERRUPTED]"}
{"role": "user", "content": "Actually, skip GitHub. What about the contract?"}
```

The LLM sees it was cut off and what the user pivoted to. It responds naturally: *"Sure — the label manager sent over a revised contract this morning..."*

---

## Example Conversation (Full Lifecycle with CONTINUE)

```
──────────────────────────────────────────────────────
TURN 1: User speaks
──────────────────────────────────────────────────────

STT transcript: "Hey, check my recent GitHub commits and see if the
                 label manager emailed me back about that contract."

Voice LLM runs with:
  - static system prompt (cached)
  - conversation history + new user message
  - native tools: dispatch_openclaw, set_turn_mode

Voice LLM output:

  Streamed text (piped to TTS):
    "I'll have a look through your commits and check your inbox now, sir."

  Tool calls:
    dispatch_openclaw(
      directive="Check the user's recent GitHub commits on the ACI repo
                 and search their email inbox for messages from the label
                 manager about a contract"
    )
    set_turn_mode(mode="ACKWAIT")

→ TTS speaks (streaming, ~300ms to first audio)
→ Directive dispatched to OpenClaw
→ ACKWAIT — system pauses, does not re-run LLM

──────────────────────────────────────────────────────
Sub-agents return results (2 seconds later)
──────────────────────────────────────────────────────

Queue receives two items:
  QueueItem(type="agent_result", agent="github_agent", ...)
  QueueItem(type="agent_result", agent="email_agent", ...)

→ Queue has items → triggers next LLM run
→ Queue flushed into conversation history as tool messages

──────────────────────────────────────────────────────
TURN 2: LLM processes results (with CONTINUE)
──────────────────────────────────────────────────────

Voice LLM sees the tool messages in conversation history.

LLM run 2a:

  Streamed text:
    "Right — three commits today on the ACI repo. Most recent was the
     webhook handler about an hour ago."

  Tool calls:
    set_turn_mode(mode="CONTINUE")

→ TTS speaks sentence 1-2
→ CONTINUE — loop immediately re-runs LLM
→ Queue checked: empty, nothing new

LLM run 2b:

  Streamed text:
    "And yes, the label manager got back to you. They've sent a revised
     contract and are asking you to review it by Friday. Want me to
     pull it up?"

  Tool calls: (none — defaults to CONV)

→ TTS speaks sentence 3-4
→ CONV — open mic, wait for user

The user hears a single fluid response across two LLM runs.
Under the hood, the LLM checked for new context between chunks.

──────────────────────────────────────────────────────
TURN 3: User follows up
──────────────────────────────────────────────────────

STT transcript: "Yeah, pull up the contract and tell me what changed."

Voice LLM output:

  Streamed text:
    "On it — pulling the attachment and comparing it now."

  Tool calls:
    dispatch_openclaw(
      directive="Get the most recent email attachment from the label
                 manager (the revised contract) and produce a summary
                 of what changed compared to the previous version"
    )
    set_turn_mode(mode="ACKWAIT")

→ Cycle repeats...

──────────────────────────────────────────────────────
TURN 4: Mid-speech context injection (via CONTINUE)
──────────────────────────────────────────────────────

Document agent returns the diff summary.
Queue flushed into conversation history.

LLM run 4a:

  Streamed text:
    "The main changes are in clause 7 — they've bumped the royalty
     rate from 15 to 18 percent."

  Tool calls:
    set_turn_mode(mode="CONTINUE")

→ TTS speaks
→ CONTINUE — queue checked — email_agent just pushed a status
  update: "New email from label manager: 'Also see addendum re:
  territories'"

LLM run 4b:

  Streamed text:
    "Oh, and they've just sent a follow-up — there's an addendum
     about territories. Want me to read through that as well?"

  Tool calls: (none — defaults to CONV)

The agent seamlessly wove in a real-time email notification
mid-response. To the user, it sounds like one continuous thought.
```

---

## Integration with OpenClaw

This architecture **does not replace** OpenClaw's agent system. It wraps it with a voice interface:

```
┌─────────────────────────────────────────────────┐
│              VOICE LAYER                         │
│                                                  │
│  LiveKit ↔ STT → Voice LLM → TTS               │
│                     │                            │
│                     │ dispatch_openclaw(          │
│                     │   directive: str            │
│                     │ )                           │
│                     ▼                            │
│  ┌─────────────────────────────────────────────┐ │
│  │           OPENCLAW (existing)                │ │
│  │                                             │ │
│  │  NL directive → Orchestrator → sub-agents   │ │
│  │                                             │ │
│  │  ┌────────┐ ┌────────┐ ┌────────┐          │ │
│  │  │ Email  │ │ GitHub │ │ Coding │  ...      │ │
│  │  │ Agent  │ │ Agent  │ │ Agent  │          │ │
│  │  └────────┘ └────────┘ └────────┘          │ │
│  │                                             │ │
│  │  Returns results + status updates           │ │
│  └──────────────────────┬──────────────────────┘ │
│                         │                        │
│                         ▼                        │
│               Voice LLM Queue                    │
│                         │                        │
│                         │ flush to conv history   │
│                         ▼                        │
│               Voice LLM (next run)               │
└─────────────────────────────────────────────────┘
```

OpenClaw already handles:
- Parsing natural language directives into agent routing decisions
- Routing to the correct sub-agent(s)
- Agent execution, retries, error handling
- Multi-step agent workflows

The voice layer only needs to:
1. Call `dispatch_openclaw(directive)` with a natural language string
2. Receive results and status updates back
3. Push them into the Voice LLM Queue
4. Flush the queue into conversation history before each LLM run

```python
class OpenClawVoiceInterface:
    """Thin adapter between the voice layer and OpenClaw."""

    async def execute(
        self,
        directive: str
    ) -> AsyncIterator[AgentUpdate]:
        """
        Send a natural language directive to OpenClaw and yield
        status updates and the final result as they arrive.
        """
        async for update in self.orchestrator.run(directive):
            yield update
```

---

## Status Updates

Sub-agents can emit **status updates** during execution, not just final results. These flow into the Voice LLM Queue just like results:

```python
# Inside a sub-agent (e.g., email_agent)
async def search_inbox(self, query: str):
    yield AgentUpdate(is_status=True, message="Searching inbox...")

    results = await self.imap.search(query)

    yield AgentUpdate(is_status=True,
                      message=f"Found {len(results)} matching emails, reading...")

    details = await self.read_emails(results)

    yield AgentUpdate(is_result=True, data=details, summary="...")
```

Two handling modes (configurable):

**Option A: Hold and batch** (simpler, recommended initially)
Status updates accumulate in the queue silently. Only trigger the next LLM run when a final `agent_result` arrives. Status context is still in history, so the LLM sees what happened.

**Option B: Relay in real-time** (more JARVIS-like)
Each status update triggers a quick LLM run. The LLM might say *"Still searching..."* or might decide it's not worth speaking. More responsive but more LLM calls.

---

## Edge Cases

### User speaks during ACKWAIT (before sub-agent returns)
User speech always triggers an LLM run. The queue may be empty (agent hasn't returned yet) or partially filled. The LLM handles this naturally — it knows what it dispatched and can say *"Still waiting on that, but what's up?"*

### Multiple sub-agents return at different times
Each result is pushed into the queue. The first result triggers the LLM run. If the second arrives during the LLM's `CONTINUE` loop, the LLM picks it up on the next chunk via the queue flush. If it arrives after the LLM finishes, it triggers a fresh LLM run.

### Sub-agent fails or times out
OpenClaw handles retries and errors. On final failure, it pushes an error into the queue:
```python
QueueItem(type="agent_result", agent="github",
          content={"error": "timeout"},
          summary="Failed to reach GitHub API after 3 retries.")
```
The LLM says *"I wasn't able to reach GitHub just now — want me to try again?"*

### User interrupts (barge-in)
LiveKit VAD detects speech → TTS stops → partial speech logged as `[INTERRUPTED]` → user's new transcript processed → LLM re-runs with accurate context.

### Long-running agents (e.g., "scan my entire codebase")
The LLM outputs `ACKWAIT`. Status updates keep the queue alive with progress. The system can optionally relay progress to the user: *"About halfway through the codebase..."* The final result triggers the full LLM run.

---

## The Voice LLM System Prompt

The system prompt is **static** (enabling prompt caching) and instructs the LLM on the tool-calling contract:

```
You are JARVIS, a voice-first AI assistant. You respond in spoken English.

CRITICAL RULES:
- Output 1-2 sentences maximum per response. Call set_turn_mode("CONTINUE")
  if you have more to say. This creates natural pacing.
- When dispatching work, acknowledge immediately with speech ("I'll check 
  that now"), call dispatch_openclaw with a clear directive, and call
  set_turn_mode("ACKWAIT").
- When you have tool results to present and the information spans more 
  than 2 sentences, present 1-2 sentences and use CONTINUE.
- Synthesise and prioritise information. Don't parrot raw data.
- Be conversational, concise, and natural. You are speaking, not writing.

TOOLS:
- dispatch_openclaw(directive): Send a natural language task to the agent
  system. Describe what needs to be done in plain English. Do NOT try to
  specify agent names, action types, or parameters — just describe the task.
- set_turn_mode(mode): Control what happens next.
  - ACKWAIT: You've dispatched work and need to wait for results.
  - CONTINUE: You have more to say. System re-runs you immediately with
    any new context that arrived in the meantime.
  - CONV: You've finished your thought. Open the mic for the user.
  - END: The conversation is over.

You will receive tool results as messages in the conversation history
with the format: [AGENT_RESULT | agent_name | timestamp]
Treat these as data you've retrieved. Present them naturally.
```

---

## Tech Stack

| Component | Recommendation | Role |
|---|---|---|
| **Transport** | LiveKit (WebRTC) | Bidirectional audio, AEC, VAD, barge-in |
| **STT** | Deepgram (streaming) | Real-time transcription via LiveKit integration |
| **Voice LLM** | Claude Sonnet / GPT-4o | Native tool calling, conversational reasoning |
| **TTS** | ElevenLabs / Cartesia (streaming) | Low-latency voice synthesis |
| **Agent system** | OpenClaw (existing) | NL directive parsing, sub-agent routing + execution |
| **Queue** | Python asyncio.Queue + Event | Voice LLM Queue |
| **Tool interface** | Native LLM tool calling | Zero-parsing, crash-proof dispatch |

---

## Out of Scope (Revisit Later)

- **Speculative execution / mid-sentence intent parsing** — Parsing user words as they speak and pre-fetching data. Cool but marginal gain. Wait for complete utterances.

- **LangGraph / AutoGen** — OpenClaw handles agent orchestration. No additional framework needed in the voice layer.

- **Persistent memory across sessions** — Handled in OpenClaw's existing memory architecture, not the voice layer.

---

## The Market Opportunity

Recent market data from late 2025 and early 2026 shows that viral AI apps are pulling in staggering numbers. Solo developers leveraging TikTok have scaled novelty AI apps to $800,000 in revenue in a single year, and viral AI tools have hit $1 million in ARR in just 7 days.

JARVIS has two properties that make it uniquely positioned to exploit this: an undeniable demo (the kind of thing people film their reaction to) and an architecture that makes each session absurdly cheap to serve.

### The Hyper-Viral Weekend

Imagine you post a raw, unedited video of you walking around your room, interrupting JARVIS mid-sentence while it searches the live web, and it adapts perfectly.

| Metric | Value |
|---|---|
| **Viral Reach** | 10,000,000 views across TikTok + X |
| **Click-Through Rate** | 5% (highly engaging demo) → 500,000 visitors |
| **Impulse Buy Conversion** | 10% at £2 for 5 minutes of magic (zero sign-up friction, Apple Pay) → 50,000 paid sessions |
| **Gross Revenue (48 hours)** | **50,000 × £2.00 = £100,000** |

### The Unfair Profit Margin

Because this architecture appends sub-agent results to the conversation history as tool messages, the system prompt stays static. This enables prompt caching, which slashes token costs by ~80%.

Even using top-tier streaming TTS and LLMs, the cost for a 5-minute session is approximately **£0.38**.

| | |
|---|---|
| **Total API Cost** | 50,000 sessions × £0.38 = £19,000 |
| **Weekend Net Profit** | **£81,000** |

And that doesn't account for the fact that people will want to show their friends. If just 30% of those users drop another £2 to show their roommate how cool it is, you break the **£100,000 profit mark** in a matter of days.

### Why The Demo Will Actually Convert

JARVIS won't look like another ChatGPT wrapper — it will feel like a leap into the future because of the mechanics baked into this architecture:

**The 300ms Magic.** When users push the mic button, the LLM's text tokens stream to TTS immediately. They hear JARVIS respond in ~300ms. That instant reaction triggers the "wow" factor that no turn-based chatbot can replicate.

**The Stream-of-Consciousness.** Because of the `CONTINUE` turn mode, JARVIS speaks 1–2 sentences, checks the queue for new search results, and weaves them into its next sentence without awkward pauses. It sounds like it's thinking out loud — not reading from a script.

**The Flawless Interruption.** If they try to test the system by yelling over it, LiveKit's VAD triggers, stops the TTS immediately, and logs the partial speech as `[INTERRUPTED]`. JARVIS instantly adapts to their rudeness, which makes for incredible viral video moments.

---

## The Viral Demo Script

The secret to a viral tech demo: keep it under 30 seconds, show a "wait, what?" moment immediately, and focus on one mind-blowing feature at a time.

This script flexes the three most magical parts of the architecture: the ~300ms latency, the LiveKit VAD barge-in, and the `CONTINUE` state mid-thought weaving.

### Camera Setup

Film POV style (from your chest or holding your phone). Point the camera at your laptop screen showing two things side-by-side:

- **Left Side:** Your terminal, showing the raw streaming logs and the VoiceLLMQueue.
- **Right Side:** Your live email inbox or a blank code editor.

### The 30-Second Script

**[0:00 — The Hook]**
Don't say "Hey Jarvis." Just start talking fast while typing.

> **You:** "Jarvis, read my last email and run a web search on Tesla stock."

**[0:03 — The ACKWAIT Latency Flex]**
Instantly — within ~300ms — the terminal lights up with streaming tokens.

> **Jarvis (Audio):** "I'll pull your inbox and check the markets now, sir."

Point to the screen where the terminal shows `turn_mode: ACKWAIT`.

**[0:08 — The "Rude" Barge-In Flex]**
Jarvis starts reading the search results:

> **Jarvis:** "Tesla is currently trading at—"

Abruptly yell over him.

> **You:** "Actually, skip the stock! Just give me the email!"

The terminal instantly flashes `[INTERRUPTED]` as the LiveKit VAD detects your voice and stops the TTS.

**[0:15 — The CONTINUE Mid-Thought Flex]**
Because the system appended the interruption to the conversation history, Jarvis instantly understands he was cut off.

> **Jarvis:** "Right, skipping the markets. You have an email from your boss about the..."

Right as Jarvis says this, push a button on your keyboard that triggers a "Breaking News" alert into your VoiceLLMQueue. Because Jarvis uses the `CONTINUE` loop, he checks the queue between his first and second sentence.

> **Jarvis (without pausing):** "...Oh, and you actually just got a calendar invite for a meeting in 10 minutes. Want me to accept it?"

**[0:25 — The Call to Action]**
Look directly into the camera.

> **You:** "It actually thinks in real-time. Try it yourself right now for 2 quid. Link in bio."

### Why This Goes Viral

**It breaks the ChatGPT mold.** People are used to AI that forces you to wait in silence, listen to a wall of text, and start over if you make a mistake. Seeing an AI stop mid-sentence when interrupted and instantly pivot looks like magic.

**The visual proof.** Seeing the `[INTERRUPTED]` log and the sub-agent results flushing into the conversation history as tool messages proves this isn't a fake, pre-recorded video. It is raw, working code.

**The stream-of-consciousness effect.** When Jarvis weaves that calendar invite into his speech mid-thought, it creates a breathing, adaptive agent that sounds incredibly human.

---

## Summary

The architecture is five things:

1. **Natural language dispatch to OpenClaw** — The Voice LLM calls a single tool, `dispatch_openclaw(directive)`, passing a plain English task description. OpenClaw handles all routing, agent selection, and execution. The Voice LLM never constructs structured schemas, never risks malformed JSON, and never needs to know how any sub-agent works internally.

2. **Native text streaming with zero parsing** — The LLM streams conversational text directly to TTS using native tool calling. Text chunks are piped to the TTS engine the instant they're generated — no JSON wrapper, no streaming parser, no custom extraction logic. Latency hits the absolute floor at ~300ms to first audio, and the pipeline is crash-proof.

3. **The Voice LLM Queue** — Sub-agent results and status updates accumulate here. On flush, they're appended to **conversation history** as tool messages — not the system prompt. No amnesia. Prompt caching enabled. Token costs slashed by ~80%.

4. **The CONTINUE loop** — The LLM speaks 1–2 sentences at a time. On `CONTINUE`, it re-runs immediately, checking the queue for fresh context before generating the next chunk. This creates a breathing, adaptive agent that can weave in real-time data mid-thought — the stream-of-consciousness effect.

5. **LiveKit from Day 1** — WebRTC for AEC (no feedback loops), VAD (barge-in detection), and proper audio transport. When the user interrupts, TTS stops instantly, partial speech is logged as `[INTERRUPTED]`, and the LLM re-runs with accurate context. Not optional.

OpenClaw handles the agents. The voice layer is: **listen → LLM → stream speech → dispatch directive → check turn mode → repeat.**
