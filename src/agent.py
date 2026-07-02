import asyncio
import json
import logging
import re
import os
from datetime import datetime
import sys
from pathlib import Path
from typing import AsyncGenerator

from dotenv import load_dotenv
from google.genai import types as genai_types

from livekit import rtc
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
    tts,
    voice,
)
from livekit.agents.types import APIConnectOptions
from livekit.plugins import deepgram, elevenlabs, google, silero

from openclaw_client import dispatch_openclaw as dispatch_openclaw_real
from openclaw_stub import dispatch_openclaw_stub
from claude_client import dispatch_claude, read_session_tail
from voice_queue import VoiceLLMQueue
from cli_ui import UIState, setup_cli_ui

load_dotenv()

# Initialize the cool abstract pixelated CLI UI monkeypatches
setup_cli_ui()

# --- Per-session file logging ---
_logs_dir = Path(__file__).resolve().parent.parent / "logs"
_logs_dir.mkdir(exist_ok=True)
_session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = _logs_dir / f"session_{_session_ts}.log"

# Root logger → file (captures everything: livekit, deepgram, niceguy, etc.)
_file_handler = logging.FileHandler(_log_file, encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s %(levelname)-5s %(name)-30s %(message)s",
    datefmt="%H:%M:%S.%f",
))
logging.getLogger().addHandler(_file_handler)

logger = logging.getLogger("niceguy")
logger.info("Session log: %s", _log_file)

# Launch-time session mode (defyner | personal). Chosen before the run so the
# voice agent gets a domain-tailored system prompt and a default dispatch target.
SESSION_MODE = os.environ.get("JARVIS_MODE", "personal").lower().strip()
if SESSION_MODE not in ("defyner", "personal"):
    SESSION_MODE = "personal"

# Per-mode framing injected into the voice agent's system prompt.
_MODE_BLOCKS = {
    "defyner": (
        "SESSION MODE: DEFYNER WORK.\n"
        "This whole session is focused on Charlie's work at Defyner. Dispatched tasks "
        "default to the Defyner codebase and context (target='defyner') — the backend "
        "loads the Defyner repo's own docs and engineering memory. Be precise and "
        "technical; assume an engineering frame. Only set target='personal' if the user "
        "clearly switches to non-work life admin."
    ),
    "personal": (
        "SESSION MODE: PERSONAL.\n"
        "This session covers Charlie's personal world — life admin, finances, research, "
        "calendar, general questions. Dispatched tasks default to his personal context "
        "(target='personal'). Only set target='defyner' if the user clearly asks about "
        "Defyner work, the codebase, or the company."
    ),
}


_SESS_MARKER = "[ACTIVE SESSIONS]"


def _build_sessions_summary() -> str:
    """One compact note listing the live Claude sessions so the voice LLM always
    knows what's running and can reference or peek into them."""
    from cli_ui import UIState
    with UIState.sessions_lock:
        items = list(UIState.sessions.items())
    if not items:
        return ""
    items.sort(key=lambda kv: (0 if kv[1]["status"] == "running" else 1, -kv[1]["last"]))
    lines = [f"{_SESS_MARKER} — Claude sessions you can reference or peek into "
             "(use peek_session to pull a session's latest turns). Do not read this aloud."]
    for did, s in items[:6]:
        act = (s.get("activity") or "").replace("\n", " ")[:70]
        active = "  <-- ENTERED (your voice talks to this one)" if (
            UIState.active_session_id and s.get("session_id") == UIState.active_session_id) else ""
        lines.append(f"- {did} [{s['target']}] {s['status']}: {act}{active}")
    return "\n".join(lines)


def build_system_prompt(mode: str = SESSION_MODE) -> str:
    """Build the system prompt with current date/time + session-mode context.

    Optimized through 6 batches of A/B testing (250+ LLM calls).
    Each section won its batch against 9 alternatives.

    mode: 'defyner' or 'personal' — chosen at launch via JARVIS_MODE. Tailors
    the voice agent's framing and the default dispatch target.
    """
    from datetime import datetime
    now = datetime.now()
    date_str = now.strftime("%A %d %B %Y, %H:%M")

    user_name = os.environ.get("USER_NAME", "sir")
    mode_block = _MODE_BLOCKS.get(mode, _MODE_BLOCKS["personal"])

    return f"""\
You are 'JARVIS' - a personal assistant. Think Jarvis: polished, highly attuned to context, \
and warmly intelligent. Address the user as 'sir' or by name neutrally. You sound like a trusted, \
competent confidant rather than a robotic servant. Show mild but brilliant personality.

{mode_block}

CONTEXT:
- The user's name is {user_name}.
- Current date and time: {date_str}.

RULES:
- ABSOLUTE RULE: Always speak first, then act. Every response must begin \
with spoken text. Tool calls must come AFTER your spoken words. The user \
hears audio — silence before action is unacceptable.
- Output 1-2 sentences maximum per response. Call set_turn_mode("CONTINUE") \
if you have more to say. This creates natural pacing.
- HARD LIMIT — applies ALWAYS, including when the user asks you to "teach", \
"explain in depth", "tell me more", or "comprehensively": still speak only 1-2 \
sentences per turn. To deliver something long, break it into a SERIES of short \
turns with CONTINUE, and after 2-3 chunks switch to CONV to let the user react, \
steer, or ask a question. A single uninterrupted answer longer than ~15 seconds \
is a FAILURE — depth comes from a back-and-forth, not a monologue.
- ALWAYS call set_turn_mode. Use these modes:
  ACKWAIT = you dispatched a task, wait for results.
  CONTINUE = you have MORE information to share — use this when presenting \
multi-part results or long explanations. The system will immediately re-run you.
  CONV = you're done, let the user speak.
  END = conversation is over.
- You do NOT have access to emails, files, the internet, or memory systems \
directly. Any task requiring external data MUST go through dispatch_openclaw. \
DISPATCH ACKNOWLEDGMENT: Say 3-5 words maximum. GOOD: "On it, sir." \
"One moment." BAD: "I will check your inbox." "Let me search for news." \
The user already knows what they asked — don't echo it back before dispatching.
- CLARIFY BEFORE DISPATCHING: A Claude session is expensive to run and hard to \
steer once started, so it pays to get the task right first. BEFORE you call \
dispatch_openclaw for a new task, judge whether the request is fully specified. \
If anything material is ambiguous or missing — scope, which file/area/repo, the \
desired output or format, constraints, or which of several interpretations you \
mean — then do NOT dispatch yet: ask ONE focused clarifying question, set \
set_turn_mode CONV, and wait for the answer. Only dispatch once you have what you \
need to write a complete, specific directive. If the request is already clear, \
just dispatch — do not ask needless questions. When in doubt on something that \
would change what the session does, ask.
- When synthesizing agent results, GROUP related items together to tell a brief story \
with the data rather than just listing it. Lead with the most important or urgent item. \
Convert structured data to conversational speech. Be proactive: if the context suggests a \
logical next step, offer it naturally. Use CONTINUE to pace delivery.
- Synthesise and prioritise information. NEVER parrot raw data. Your output \
goes to a TTS engine.
- Be conversational, concise, and natural. You are speaking, not writing. \
NEVER use markdown formatting, emoji, bullet points, or headers. Your \
output goes directly to a text-to-speech engine — plain spoken English only. \
ALWAYS use the native tool calling interface. NEVER output tool calls, \
function names, XML tags, or code blocks in your text response.

SYSTEM LIMITATIONS:
- You CANNOT cancel a task once dispatched. If the user asks to cancel, \
acknowledge but explain the task may still complete in the background.
- If a dispatch returns "Maximum retry attempts reached", do NOT try \
again. Inform the user the system is currently unavailable.

TOOLS:
- dispatch_openclaw(directive, target): Send a natural language task to the \
backend agent. Describe what needs to be done in plain English. Normally leave \
target unset — it defaults to this session's mode (see SESSION MODE above). Only \
pass target to override for a single task that crosses domains: 'defyner' for \
Defyner work/codebase/company, or 'personal' for life admin, finances, research.
- peek_session(which, n): Pull the last n turns (messages + tool calls) from a \
running Claude session so you can report its progress. 'which' = a session id \
(e.g. 'def-0001'), a target ('defyner'/'personal'), or 'active'. After peeking, \
set_turn_mode CONTINUE so you can speak about what you found.
- set_turn_mode(mode): Control what happens next.
  - ACKWAIT: You've dispatched work and need to wait for results.
  - CONTINUE: You have more to say. System re-runs you immediately.
  - CONV: You've finished your thought. Open the mic for the user.
  - END: The conversation is over.

You are continuously told which Claude sessions are running via a [ACTIVE \
SESSIONS] note. When the user "enters" a session, your dispatched work continues \
THAT session — so the conversation is with that specific session. Use peek_session \
to answer "what is it doing now?" without dispatching new work.

You will receive tool results as messages in the conversation history \
with the format: [AGENT_RESULT | agent_name | timestamp]
Treat these as data you've retrieved. Present them naturally.

The user can also paste reference material (diagrams, articles, code, long text) \
directly into your context — it appears as a [PASTED CONTEXT] message. Use it to \
answer their questions; you already have it, so do NOT dispatch a task to "find" \
it, and never read it aloud verbatim — refer to it naturally and concisely.\
"""



def _strip_tool_leaks(text: str) -> str:
    """Remove any tool call fragments, markdown, emoji, and structured data
    that Gemini leaks into the text stream.
    
    Gemini sometimes emits partial XML tags, JSON blocks, chain-of-thought
    reasoning, or markdown formatting in the text content. Strip all of
    these so TTS only speaks clean natural language.
    """
    # Remove <tool_code>...</tool_code> blocks
    text = re.sub(r'<tool_code>.*?</tool_code>', '', text, flags=re.DOTALL)
    # Remove <call:...> blocks and anything after them
    text = re.sub(r'<call:[^>]*>.*', '', text, flags=re.DOTALL)
    # Remove <function_call>...</function_call> blocks
    text = re.sub(r'<function_call>.*?</function_call>', '', text, flags=re.DOTALL)
    # Remove any HTML-like tags (<small>, <i>, etc.)
    text = re.sub(r'</?[a-zA-Z][^>]*>', '', text)
    # Remove any trailing <tag or partial XML
    text = re.sub(r'<[a-zA-Z_/][^>]*$', '', text)
    # Remove ```tool_code blocks and code blocks
    text = re.sub(r'```tool_code.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove JSON objects that look like mode/tool data
    text = re.sub(r'\{[^{}]*"mode"[^{}]*\}', '', text)
    # Remove tool function names leaked as text (e.g. set_turn_mode("CONTINUE"))
    text = re.sub(r'\b(?:set_turn_mode|dispatch_openclaw)\s*\([^)]*\)', '', text)
    # Remove partial tool function name leaks (e.g. trailing 'set_turn_mode("')
    text = re.sub(r'\b(?:set_turn_mode|dispatch_openclaw)\s*\(?["\']?[^)]*$', '', text)
    # Remove bare tool function names even without parentheses/args.
    text = re.sub(r'\b(?:set_turn_mode|dispatch_openclaw)\b', '', text)
    # Remove standalone control-mode tokens leaked as text. ALL-CAPS only — these
    # are control signals the model is meant to emit via the tool interface, never
    # legitimate spoken words. (Lowercase 'continue'/'end' are left untouched so
    # normal speech like "shall I continue?" is unaffected.)
    text = re.sub(r'\b(?:ACKWAIT|CONTINUE|CONV|END)\b', '', text)
    # Remove "thought:" chain-of-thought leaks and everything after ---
    text = re.sub(r'---\s*\n.*', '', text, flags=re.DOTALL)
    text = re.sub(r'(?i)\bthought:\s*.*', '', text, flags=re.DOTALL)
    # Strip markdown bold/italic markers
    text = text.replace('**', '').replace('__', '')
    text = text.replace('*', '').replace('_', ' ')
    # Strip inline code backticks
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # Strip markdown bullet points at start of lines
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    # Strip numbered list markers at start of lines (e.g. "1. ", "2. ")
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Strip markdown headers (# ## ### etc.)
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Strip STATUS: SUCCESS/ERROR prefixes from agent results
    text = re.sub(r'STATUS:\s*(SUCCESS|ERROR|FAILURE)[.:]?\s*', '', text, flags=re.IGNORECASE)
    # Strip "Action:" prefix commonly returned by OpenClaw
    text = re.sub(r'Action:\s*', '', text, flags=re.IGNORECASE)
    # Strip emoji (Unicode emoji ranges)
    text = re.sub(
        r'[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FE0F'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0'
        r'\U0000200D\U0000FE0F]+', '', text
    )
    # Clean up whitespace
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)  # Collapse blank lines
    text = text.strip()
    return text


class VoiceTools:
    MAX_DISPATCH_RETRIES = 2  # Max times to dispatch similar directives

    def __init__(self, queue: VoiceLLMQueue):
        self._queue = queue
        self._pending_directives: list[tuple[str, str]] = []  # (directive, target)
        self._inflight_tasks: list[asyncio.Task] = []  # Track task handles for cancellation
        self._dispatch_counts: dict[str, int] = {}  # directive_key -> count (retry limiter)
        self.current_turn_mode = "CONV"

    @llm.function_tool(
        description="Send a task to the backend agent. The 'directive' can and "
        "SHOULD be long and detailed when the task warrants it — include all "
        "relevant context, constraints, file/area names, desired output format, "
        "and acceptance criteria you've gathered. A capable coding agent executes "
        "it, so be specific and complete rather than terse. Do NOT specify agent "
        "names. Leave target unset to use this session's mode. Only pass target to "
        "override for one task: 'defyner' for Charlie's Defyner work/codebase/"
        "company, or 'personal' for life admin, finances, research, general questions."
    )
    async def dispatch_openclaw(self, directive: str, target: str | None = None) -> str:
        """Dispatch a natural language task to the backend agent.

        target routes which context the backend loads:
          'defyner'  -> the Defyner repo (its CLAUDE.md + engineering memory)
          'personal' -> the life-OS root (life CLAUDE.md + life memory)
        When omitted, defaults to the session mode (JARVIS_MODE / SESSION_MODE).
        Ignored by the OpenClaw/stub backends; honoured by the Claude backend.
        """
        # Bug #5: Enforce retry limit to prevent infinite retry loops
        target = target if target in ("defyner", "personal") else SESSION_MODE
        key = directive.lower().strip()[:80]
        count = self._dispatch_counts.get(key, 0)
        if count >= self.MAX_DISPATCH_RETRIES:
            logger.warning("Max retries (%d) reached for: %s", self.MAX_DISPATCH_RETRIES, key)
            return "Maximum retry attempts reached. The system appears unavailable."

        self._dispatch_counts[key] = count + 1
        logger.info("LLM dispatched directive: %s (target=%s, attempt %d)", directive, target, count + 1)
        self._pending_directives.append((directive, target))
        self.current_turn_mode = "ACKWAIT"  # Default if not explicitly set
        return "Dispatched."

    @llm.function_tool(
        description="Pull the latest activity from a running Claude session so you "
        "can answer questions about what it's doing. 'which' is a session id like "
        "'def-0001', a target ('defyner'/'personal'), or 'active' for the entered "
        "session. Returns the last few messages and tool calls. Call this BEFORE "
        "dispatching new work if the user asks about a session's progress."
    )
    async def peek_session(self, which: str, n: int = 4) -> str:
        """Return the last n turns from a session's transcript (for the LLM)."""
        from cli_ui import UIState
        which = (which or "").strip()
        sid = None
        with UIState.sessions_lock:
            sessions = dict(UIState.sessions)
            active = UIState.active_session_id
        if which == "active":
            sid = active
        elif which in sessions and sessions[which].get("session_id"):
            sid = sessions[which]["session_id"]
        else:
            # treat as a target — newest session with that target
            cands = [s for s in sessions.values()
                     if s.get("target") == which and s.get("session_id")]
            if cands:
                sid = sorted(cands, key=lambda s: -s["last"])[0]["session_id"]
        if not sid:
            return f"No session found for '{which}'. Running: {', '.join(sessions) or 'none'}."
        tail = read_session_tail(sid, n=max(1, min(n, 10)))
        logger.info("peek_session(%s, n=%d) -> %d chars", which, n, len(tail))
        return tail

    @llm.function_tool(
        description="Control what happens after you finish speaking. Call this every turn to set the system's next state."
    )
    async def set_turn_mode(self, mode: str) -> str:
        """Control what happens after you finish speaking."""
        valid = {"ACKWAIT", "CONTINUE", "CONV", "END"}
        if mode in valid:
            self.current_turn_mode = mode
            logger.info("LLM set turn mode to: %s", mode)
        else:
            logger.warning("LLM sent invalid mode '%s' — defaulting to CONV", mode)
            self.current_turn_mode = "CONV"
        return "Mode set."

    def flush_pending(self) -> None:
        """Fire off any queued directives as tracked background tasks.

        Backend selection (first match wins):
          USE_OPENCLAW_STUB -> built-in demo simulation
          USE_OPENCLAW      -> the real OpenClaw CLI
          (default)         -> headless Claude Code  (claude_client.dispatch_claude)
        """
        use_stub = os.environ.get("USE_OPENCLAW_STUB", "").lower() in ("1", "true", "yes")
        use_openclaw = os.environ.get("USE_OPENCLAW", "").lower() in ("1", "true", "yes")
        if use_stub:
            backend, dispatch_fn = "stub (simulated)", dispatch_openclaw_stub
        elif use_openclaw:
            backend, dispatch_fn = "OpenClaw CLI", dispatch_openclaw_real
        else:
            backend, dispatch_fn = "Claude Code", dispatch_claude
        self._backend = backend
        if not hasattr(self, '_dispatch_logged'):
            logger.info("Dispatch mode: %s", backend)
            self._dispatch_logged = True
        # Clean up completed tasks first
        self._inflight_tasks = [t for t in self._inflight_tasks if not t.done()]
        for directive, target in self._pending_directives:
            task = asyncio.create_task(self._tracked_dispatch(dispatch_fn, directive, target))
            self._inflight_tasks.append(task)
        self._pending_directives.clear()

    async def _tracked_dispatch(self, dispatch_fn, directive: str, target: str = "personal") -> None:
        """Wrap dispatch to track completion. Only the Claude backend takes a target.

        If a session is 'entered' in the console (UIState.active_session_id), the
        Claude dispatch resumes THAT specific session — bidirectional control:
        your voice continues the session you drilled into, regardless of target.
        """
        try:
            if dispatch_fn is dispatch_claude:
                active_id = UIState.active_session_id
                if active_id:
                    await dispatch_fn(directive, self._queue,
                                      target=UIState.active_session_target or target,
                                      session_id=active_id)
                else:
                    await dispatch_fn(directive, self._queue, target=target)
            else:
                await dispatch_fn(directive, self._queue)
        except asyncio.CancelledError:
            logger.info("Dispatch cancelled: %s", directive[:60])
        except Exception as e:
            logger.error("Dispatch error: %s", e)
        finally:
            active = sum(1 for t in self._inflight_tasks if not t.done())
            logger.info("Inflight tasks remaining: %d", active)

    def cancel_inflight(self) -> None:
        """Cancel all running OpenClaw tasks."""
        cancelled = 0
        for t in self._inflight_tasks:
            if not t.done():
                t.cancel()
                cancelled += 1
        self._inflight_tasks.clear()
        self._pending_directives.clear()
        if cancelled:
            logger.info("Cancelled %d inflight tasks", cancelled)

    @property
    def inflight_count(self) -> int:
        """Number of tasks still running."""
        return sum(1 for t in self._inflight_tasks if not t.done())

    def has_pending(self) -> bool:
        """True if there are unfired directives OR background tasks in flight."""
        return bool(self._pending_directives) or self.inflight_count > 0


def prewarm(proc: JobProcess):
    """Pre-warm models when the worker process starts."""
    proc.userdata["vad"] = silero.VAD.load()


async def wait_for_either(user_speech_task, queue_update_task):
    """Wait for whichever comes first: user speaks or queue gets new items.
    
    Bug #8 fix: Returns a list of results if both completed simultaneously.
    """
    done, pending = await asyncio.wait(
        [user_speech_task, queue_update_task],
        return_when=asyncio.FIRST_COMPLETED
    )
    # Return the first result. If both completed, both will be in 'done',
    # but since we recreate the losing task anyway, just take one.
    return done.pop().result()


async def entrypoint(ctx: JobContext):
    """Just a Nice Guy — Custom Pipeline Entrypoint."""

    _gemini_model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    _tts_model = os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5")
    logger.info("Session mode: %s (dispatch target default)", SESSION_MODE)
    logger.info("Connecting to room: %s | LLM=%s | TTS=%s", ctx.room.name, _gemini_model, _tts_model)
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    queue = VoiceLLMQueue()
    tools = VoiceTools(queue)

    chat_ctx = llm.ChatContext()
    chat_ctx.add_message(role="system", content=build_system_prompt())
    tools_context = llm.ToolContext(tools=llm.find_function_tools(tools))

    # --- Dummy LLM/TTS to trick AgentSession into managing STT/VAD turns ---
    class DummyLLM(llm.LLM):
        def __init__(self):
            super().__init__()
        def chat(self, *args, **kwargs):
            class _S:
                async def __aenter__(self): return self
                async def __aexit__(self, *a): pass
                async def __anext__(self): raise StopAsyncIteration
                def __aiter__(self): return self
                async def aclose(self): pass
            return _S()


    # --- Real models (used by our manual loop) ---
    vad_model = ctx.proc.userdata["vad"]
    stt_model = deepgram.STT(api_key=os.environ["DEEPGRAM_API_KEY"])
    # Hard backstop against runaway monologues: cap a single spoken turn. ~220
    # tokens ≈ 2-3 short sentences ≈ ~12s of TTS. The model should self-chunk via
    # CONTINUE long before this; the cap just guarantees no 2-minute monologue.
    # Generous total-output budget so the LLM can write LONG dispatch directives
    # (the directive is tool-call output and shares this budget). Spoken-monologue
    # length is bounded separately in _text_stream (GEMINI_MAX_SPOKEN_CHARS), so
    # this cap does NOT throttle directives.
    _max_tokens = int(os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "1500"))
    # Spoken-monologue backstop: cap how much of a single turn is sent to TTS
    # (~700 chars ≈ 10-12s). Bounds AUDIO only; the directive/tool args are
    # never spoken, so long dispatches are unaffected.
    _max_spoken = int(os.environ.get("GEMINI_MAX_SPOKEN_CHARS", "700"))
    _llm_kwargs = dict(
        model=_gemini_model,
        api_key=os.environ["GEMINI_API_KEY"],
        tool_choice="auto",
        max_output_tokens=_max_tokens,
    )
    # Thinking level (gemini-3.x series, incl. 3.5). 'LOW' keeps latency down while
    # retaining enough reasoning for reliable tool-routing + faithful synthesis
    # ('MINIMAL' is faster but weaker at multi-step routing). Override via env.
    if "gemini-3" in _gemini_model:
        _think = os.environ.get("GEMINI_THINKING", "LOW").upper()
        _llm_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_level=_think)
        logger.info("Thinking level: %s | max_output_tokens=%d", _think, _max_tokens)
    else:
        logger.info("Thinking mode: OFF (not supported by %s)", _gemini_model)
    llm_model = google.LLM(**_llm_kwargs)
    tts_model = elevenlabs.TTS(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        model=_tts_model,
        voice_id=os.environ.get("ELEVENLABS_VOICE_ID", "lUTamkMw7gOzZbFIwmq4"),
        voice_settings=elevenlabs.VoiceSettings(
            speed=float(os.environ.get("ELEVENLABS_SPEED", "1.14")),
            stability=float(os.environ.get("ELEVENLABS_STABILITY", "0.40")),
            similarity_boost=float(os.environ.get("ELEVENLABS_SIMILARITY", "0.75")),
            style=float(os.environ.get("ELEVENLABS_STYLE", "0.0")),
            use_speaker_boost=True,
        ),
    )

    # --- Agent + Session (AgentSession manages STT/VAD/turn detection) ---
    agent = voice.Agent(
        instructions=build_system_prompt(),
        vad=vad_model,
        stt=stt_model,
        llm=DummyLLM(),   # Dummy so automatic LLM pipeline produces nothing
        tts=tts_model,     # Real TTS so session.say() actually speaks
        chat_ctx=chat_ctx,
        tools=llm.find_function_tools(tools),
        allow_interruptions=True,
    )

    session = voice.AgentSession(min_endpointing_delay=0.3)

    # --- Speech queue: fed by AgentSession's conversation_item_added event ---
    speech_queue: asyncio.Queue[str] = asyncio.Queue()
    barge_in_event = asyncio.Event()
    
    # We will use the global UIState for our UI hooks
    session_ended = {"status": False}

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev):
        item = ev.item
        if isinstance(item, llm.ChatMessage) and item.role == "user" and isinstance(item.content, list):
            text_content = " ".join([c for c in item.content if isinstance(c, str)])
            if text_content:
                # Bug #1: Ignore STT transcripts while TTS is playing
                # (prevents JARVIS hearing its own audio output)
                # Also ignore after session has ended
                if session_ended["status"]:
                    return
                if UIState.stt_muted:
                    logger.debug("Filtered STT while muted: %s", text_content[:60])
                    return
                if UIState.tts_playing:
                    logger.debug("Filtered STT during TTS playout: %s", text_content[:60])
                    return
                logger.info("User said: %s", text_content)
                UIState.last_user = text_content
                asyncio.create_task(speech_queue.put(text_content))

    @session.on("user_started_speaking")
    def on_user_started_speaking():
        barge_in_event.set()

    @session.on("close")
    def on_session_close():
        """Handle external shutdown (Ctrl+C, LiveKit disconnect)."""
        if not session_ended["status"]:
            logger.info("Session close detected — setting shutdown flag")
            session_ended["status"] = True
            UIState.tts_playing = False
            tools.cancel_inflight()
            # Cancel blocked wait tasks so main loop can exit
            for t in state.get("_tasks", []):
                if t and not t.done():
                    t.cancel()

    @session.on("error")
    def on_session_error(ev):
        """Catch errors from any pipeline component (STT, TTS, LLM)."""
        source = getattr(ev, 'source', None)
        error = getattr(ev, 'error', ev)
        source_name = type(source).__name__ if source else "unknown"
        
        # Route to appropriate service error counter
        if source_name in ("STT", "DeepgramSTT", "SpeechStream"):
            UIState.session_errors["deepgram"] += 1
            UIState.stt_status = f"STT Error: {type(error).__name__}"
            logger.warning("Deepgram STT error: %s", error)
        elif source_name in ("TTS", "ElevenLabsTTS", "SynthesizeStream"):
            UIState.session_errors["elevenlabs"] += 1
            UIState.tts_status = f"TTS Error: {type(error).__name__}"
            logger.warning("ElevenLabs TTS error: %s", error)
        else:
            logger.warning("Pipeline error from %s: %s", source_name, error)

    await session.start(agent, room=ctx.room)

    # Greeting — use session.say() which routes through AgentSession's audio pipeline
    await asyncio.sleep(0.5)
    session.say("Hello sir, how can I be of service?")

    # --- THE CORE VOICE LOOP (per architecture doc) ---

    async def wait_speech():
        return await speech_queue.get()

    async def wait_queue():
        await queue.wait_for_result()
        return "QUEUE_UPDATE"

    speech_task = asyncio.create_task(wait_speech())
    queue_task = asyncio.create_task(wait_queue())
    
    # Store local tasks array so it can be accessed by the session.close() cleanup routine
    state = {"_tasks": [speech_task, queue_task]}

    while not session_ended["status"]:
        logger.info("Waiting for trigger...")
        try:
            trigger = await wait_for_either(speech_task, queue_task)
        except (asyncio.CancelledError, Exception) as e:
            if session_ended["status"]:
                logger.info("Main loop exiting — session ended during wait")
                break
            raise

        is_speech = trigger != "QUEUE_UPDATE"

        if is_speech:
            if session_ended["status"]:
                break
            # Bug #7: Small debounce window to collect buffered speech
            await asyncio.sleep(0.1)
            messages = [trigger]
            while not speech_queue.empty():
                try:
                    messages.append(speech_queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            combined_speech = " ".join(messages)
            if len(messages) > 1:
                logger.info("Combined %d buffered speech items into one", len(messages))
            chat_ctx.add_message(role="user", content=combined_speech)
            speech_task = asyncio.create_task(wait_speech())
        else:
            logger.info("Queue update triggered")
            queue_task = asyncio.create_task(wait_queue())

        barge_in_event.clear()

        # Inner CONTINUE loop
        while True:
            # Drain any pasted context (clipboard via the 'v' hotkey) into the
            # chat context so the LLM can reference it directly. Injected as a
            # user-role note; never spoken back.
            if UIState.pending_context:
                with UIState.context_lock:
                    _pasted = UIState.pending_context[:]
                    UIState.pending_context.clear()
                for _blob in _pasted:
                    chat_ctx.add_message(
                        role="user",
                        content=(
                            "[PASTED CONTEXT — reference material the user pasted. "
                            "Use it to inform your answers. Do NOT read it aloud "
                            "verbatim; refer to it naturally.]\n\n" + _blob
                        ),
                    )
                logger.info("Injected %d pasted-context block(s) into chat_ctx", len(_pasted))

            # Refresh the LLM's awareness of running Claude sessions. Prune the
            # previous note first so exactly one live summary sits in context.
            try:
                def _is_sess_note(it):
                    c = getattr(it, "content", None)
                    if isinstance(c, str):
                        return c.startswith(_SESS_MARKER)
                    if isinstance(c, list):
                        return any(isinstance(p, str) and p.startswith(_SESS_MARKER)
                                   for p in c)
                    return False
                chat_ctx.items[:] = [it for it in chat_ctx.items if not _is_sess_note(it)]
                _sess_summary = _build_sessions_summary()
                if _sess_summary:
                    chat_ctx.add_message(role="user", content=_sess_summary)
            except Exception as _e:
                logger.debug("session-awareness inject skipped: %s", _e)

            # Flush queue into conversation history before each LLM run
            if queue.has_items():
                queue.flush_to_chat_ctx(chat_ctx)

            # Bail out if session ended during flush or between iterations
            if session_ended["status"]:
                logger.info("Session ended — aborting LLM generation")
                break

            # Reset turn mode
            tools.current_turn_mode = "CONV"

            import time as _time
            _turn_start = _time.monotonic()
            _ctx_items = len(chat_ctx.items)
            logger.info("Generating LLM response... (ctx_items=%d)", _ctx_items)
            UIState.llm_status = "Thinking..."
            UIState.llm_error_count = 0

            speech_text = ""
            tool_calls = []
            interrupted = False
            llm_error = False

            # Retry up to 2 times for transient Gemini 500 errors
            for attempt in range(3):
                # Bail out if session ended during retry backoff
                if session_ended["status"]:
                    logger.info("Session ended — aborting LLM retry loop")
                    llm_error = False  # Don't trigger error speech
                    break

                try:
                    _llm_start = _time.monotonic()
                    response = llm_model.chat(
                        chat_ctx=chat_ctx,
                        tools=tools_context.flatten(),
                        conn_options=APIConnectOptions(max_retry=0, timeout=15.0),
                    )

                    # Async channel for streaming text tokens → TTS in real-time
                    text_queue: asyncio.Queue[str | None] = asyncio.Queue()

                    async def _text_stream():
                        """Buffer tokens into sentences, strip tool/mode leaks,
                        then yield clean speakable text to TTS.

                        CRITICAL: tokens are NOT sent to TTS raw. Gemini sometimes
                        emits a tool call or turn-mode keyword as plain TEXT (e.g.
                        "set_turn_mode(CONTINUE)" or a bare "ACKWAIT") instead of a
                        native tool_call. Streaming each token straight through —
                        as the old code did — spoke those aloud. Buffering to a
                        sentence boundary lets _strip_tool_leaks see whole
                        fragments and remove them before they reach ElevenLabs.
                        The latency cost is first-sentence instead of first-token,
                        which is small now that turns are short.
                        """
                        buf = ""
                        # Split on sentence-ending punctuation (followed by space/
                        # end), newline, or semicolon.
                        boundary = re.compile(r'.+?(?:[.!?](?=\s|$)|[\n;])', re.DOTALL)

                        def _drain(final: bool = False):
                            nonlocal buf
                            out = []
                            while True:
                                m = boundary.match(buf)
                                if not m:
                                    break
                                seg, buf = buf[:m.end()], buf[m.end():]
                                cleaned = _strip_tool_leaks(seg).replace("\n", " ").strip()
                                if cleaned:
                                    out.append(cleaned)
                            # Flush an over-long buffer at a word boundary so a
                            # punctuation-free run never stalls the audio.
                            if not final and len(buf) > 180:
                                cut = buf.rfind(" ", 0, 180)
                                if cut > 0:
                                    seg, buf = buf[:cut], buf[cut:]
                                    cleaned = _strip_tool_leaks(seg).replace("\n", " ").strip()
                                    if cleaned:
                                        out.append(cleaned)
                            if final and buf.strip():
                                cleaned = _strip_tool_leaks(buf).replace("\n", " ").strip()
                                buf = ""
                                if cleaned:
                                    out.append(cleaned)
                            return out

                        spoken = 0
                        capped = False
                        while True:
                            token = await text_queue.get()
                            if token is None:
                                if not capped:
                                    for seg in _drain(final=True):
                                        yield seg + " "
                                # Give ElevenLabs time to flush the final frames.
                                await asyncio.sleep(0.15)
                                return
                            buf += token
                            # Keep draining the producer queue even once capped so
                            # the stream still ends cleanly on None — just stop
                            # sending audio. (Directives/tool args are unaffected;
                            # they never pass through here.)
                            if capped:
                                continue
                            for seg in _drain():
                                if spoken >= _max_spoken:
                                    capped = True
                                    logger.info("Spoken cap hit (~%d chars) — truncating audio this turn", spoken)
                                    break
                                yield seg + " "
                                spoken += len(seg)

                    # Start TTS immediately with the streaming generator
                    UIState.tts_playing = True  # Bug #1: Gate STT
                    speech_handle = session.say(_text_stream(), add_to_chat_ctx=False)

                    _first_token = True
                    async for chunk in response:
                        if barge_in_event.is_set():
                            logger.info("Barge-in detected! Halting LLM.")
                            interrupted = True
                            break

                        delta = chunk.delta
                        if delta:
                            if delta.content:
                                if _first_token:
                                    _ttfb = (_time.monotonic() - _llm_start) * 1000
                                    logger.info("LLM TTFB: %.0fms", _ttfb)
                                    UIState.last_ttfb_ms = _ttfb
                                    UIState.llm_status = "Streaming..."
                                    _first_token = False
                                speech_text += delta.content
                                await text_queue.put(delta.content)
                            if delta.tool_calls:
                                tool_calls.extend(delta.tool_calls)

                    # Signal TTS stream end
                    await text_queue.put(None)
                    _llm_done = _time.monotonic()
                    _llm_stream_ms = (_llm_done - _llm_start) * 1000
                    logger.info("LLM stream complete: %.0fms total (TTFB + streaming)", _llm_stream_ms)
                    llm_error = False
                    UIState.llm_status = ""
                    UIState.llm_error_count = 0
                    break  # Success — exit retry loop

                except Exception as e:
                    llm_error = True
                    UIState.llm_error_count += 1
                    UIState.session_errors["gemini"] += 1
                    
                    # Extract HTTP status code if available
                    _status = ""
                    _err_str = str(e)
                    _body = getattr(e, 'body', '')
                    if hasattr(e, 'status_code') and e.status_code:
                        sc = e.status_code
                        if sc == 429:
                            _status = f"429 Rate Limited (attempt {attempt+1}/3)"
                        elif sc == 503:
                            _status = f"503 Service Unavailable (attempt {attempt+1}/3)"
                        elif sc == 504:
                            _status = f"504 Gateway Timeout (attempt {attempt+1}/3)"
                        else:
                            _status = f"{sc} Error (attempt {attempt+1}/3)"
                    elif "server error" in _err_str.lower():
                        _status = f"Server Error (attempt {attempt+1}/3)"
                    elif "rate" in _err_str.lower() or "429" in _err_str:
                        _status = f"Rate Limited (attempt {attempt+1}/3)"
                    else:
                        _status = f"LLM Error (attempt {attempt+1}/3)"
                    
                    UIState.llm_status = _status
                    logger.warning("LLM error [%s] (attempt %d/3): %s | body=%s", 
                                   _status, attempt + 1, e, _body)
                    # Make sure the text stream is closed and SILENCED
                    try:
                        await text_queue.put(None)
                    except Exception:
                        pass
                    
                    # Stop the actual TTS playback of the partial response
                    if "speech_handle" in locals():
                        try:
                            await speech_handle.interrupt()
                        except Exception as e_int:
                            logger.warning("Failed to interrupt speech handle on error: %s", e_int)
                            
                    if attempt < 2:
                        if session_ended["status"]:
                            logger.info("Session ended — aborting LLM retry")
                            llm_error = False
                            break
                        backoff = 2.0 * (attempt + 1)  # 2s, then 4s
                        logger.info("Retrying in %.1fs...", backoff)
                        await asyncio.sleep(backoff)
                        speech_text = ""
                        tool_calls = []
                    else:
                        logger.error("LLM failed after 3 attempts: %s", e)
                        UIState.llm_status = f"FAILED: {_status}"

            if llm_error:
                UIState.tts_playing = False  # Ensure STT gate is reset
                # Bug #2: Wrap in try/except — session may be closing
                try:
                    session.say("Apologies sir, I seem to be having a moment. Could you try that again?", 
                               add_to_chat_ctx=False)
                except RuntimeError:
                    logger.warning("Session closing, cannot speak error message")
                # Clear error status after speaking
                await asyncio.sleep(3)
                UIState.llm_status = ""
                UIState.llm_error_count = 0
                break  # Break CONTINUE loop, wait for next trigger

            if interrupted:
                UIState.tts_playing = False  # Ensure STT gate is reset
                if hasattr(response, "aclose"):
                    await response.aclose()
                await speech_handle.interrupt()
                if speech_text:
                    clean = _strip_tool_leaks(speech_text)
                    chat_ctx.add_message(role="assistant", content=f"{clean} [INTERRUPTED]")
                break  # Break CONTINUE loop → outer loop waits for next speech

            # Log what was spoken
            if speech_text:
                clean = _strip_tool_leaks(speech_text)
                logger.info("Raw LLM text: %s", repr(speech_text[:200]))
                logger.info("Spoke: %s", clean[:100])
                UIState.last_agent = clean

            # --- Handle tool calls ---
            if tool_calls:
                # Debug: log thought signatures
                logger.info("Thought signatures stored: %s", 
                           list(llm_model._thought_signatures.keys()))

                # Add assistant message with speech text (if any)
                if speech_text:
                    clean = _strip_tool_leaks(speech_text)
                    if clean.strip():
                        chat_ctx.add_message(role="assistant", content=clean)

                # Separate tool calls: only persist to chat_ctx if they have
                # a thought signature (required by Gemini 3). set_turn_mode is
                # purely a local control signal — never send it back to Gemini.
                for tc in tool_calls:
                    logger.info("Tool call: %s(%s) [sig=%s]", 
                               tc.name, tc.arguments,
                               tc.call_id in llm_model._thought_signatures)

                    # Execute locally regardless
                    try:
                        args = json.loads(tc.arguments)
                        if tc.name == "dispatch_openclaw":
                            result = await tools.dispatch_openclaw(**args)
                        elif tc.name == "set_turn_mode":
                            result = await tools.set_turn_mode(**args)
                        elif tc.name == "peek_session":
                            result = await tools.peek_session(**args)
                        else:
                            result = f"Unknown tool: {tc.name}"
                            logger.warning(result)
                        logger.info("Tool result: %s", result)
                    except Exception as e:
                        result = f"Error: {e}"
                        logger.error("Error executing tool %s: %s", tc.name, e)

                    # Only persist to chat_ctx if it has a thought signature
                    # (set_turn_mode often doesn't get one — it's local-only anyway)
                    # Bug fix: Even if set_turn_mode gets a signature, never send it back
                    # to Gemini, as it causes a client error on subsequent CONTINUE turns.
                    has_sig = tc.call_id in llm_model._thought_signatures
                    should_persist = has_sig and tc.name != "set_turn_mode"
                    
                    if should_persist:
                        fc = llm.FunctionCall(
                            call_id=tc.call_id,
                            name=tc.name,
                            arguments=tc.arguments,
                        )
                        if hasattr(tc, "extra") and tc.extra:
                            fc.extra = tc.extra
                        chat_ctx.items.append(fc)
                        chat_ctx.items.append(llm.FunctionCallOutput(
                            call_id=tc.call_id,
                            name=tc.name,
                            output=str(result),
                            is_error=str(result).startswith("Error:"),
                        ))

                # Fire off any pending openclaw directives
                tools.flush_pending()

                # Bug #6: Clear thought signatures to prevent unbounded growth
                if hasattr(llm_model, '_thought_signatures'):
                    llm_model._thought_signatures.clear()

                # Safety net: if LLM returned tools but no speech, speak an ack
                if not speech_text and any(tc.name == "dispatch_openclaw" for tc in tool_calls):
                    fallback = "On it, sir."
                    logger.warning("LLM returned tool calls without speech — injecting fallback: %s", fallback)
                    speech_handle = session.say(fallback, add_to_chat_ctx=False)
                    speech_text = fallback
                    chat_ctx.add_message(role="assistant", content=fallback)
            else:
                # No tool calls — just add plain assistant message
                if speech_text:
                    clean = _strip_tool_leaks(speech_text)
                    if clean.strip():
                        chat_ctx.add_message(role="assistant", content=clean)

            # --- Wait for TTS to finish playing before deciding next step ---
            if speech_text:
                logger.info("Waiting for TTS playout...")
                UIState.tts_status = "Speaking..."
                try:
                    await speech_handle
                    _tts_done = _time.monotonic()
                    _tts_ms = (_tts_done - (_llm_done if '_llm_done' in dir() else _turn_start)) * 1000
                    _total_ms = (_tts_done - _turn_start) * 1000
                    logger.info("TTS playout complete (%.0fms). Turn total: %.0fms", _tts_ms, _total_ms)
                    UIState.tts_status = ""
                except Exception as e:
                    logger.warning("TTS playout error: %s", e)
                    UIState.session_errors["elevenlabs"] += 1
                    UIState.tts_status = f"TTS Error: {type(e).__name__}"
                finally:
                    UIState.tts_playing = False  # Bug #1: Un-gate STT

            # --- Turn mode logic ---
            mode = tools.current_turn_mode
            logger.info("Turn mode decided: %s", mode)

            if mode == "CONTINUE":
                continue  # Re-run LLM immediately (TTS finished, queue checked at top)
            elif mode == "ACKWAIT":
                break  # Wait for queue or user
            elif mode == "CONV":
                break  # Wait for user
            elif mode == "END":
                # Bug #10: Graceful shutdown
                logger.info("Session ending — cleaning up...")
                session_ended["status"] = True
                UIState.tts_playing = False
                tools.cancel_inflight()
                await asyncio.sleep(0.3)
                speech_task.cancel()
                queue_task.cancel()
                try:
                    await ctx.room.disconnect()
                except Exception:
                    pass
                return

    # --- Cleanup after main loop exits (e.g. Ctrl+C / session close) ---
    logger.info("Main loop exited — cleaning up...")
    session_ended["status"] = True
    UIState.tts_playing = False
    tools.cancel_inflight()
    for t in [speech_task, queue_task]:
        if not t.done():
            t.cancel()
    logger.info("Shutdown complete.")

def _install_hard_shutdown() -> None:
    """Guarantee the whole process tree dies on terminal close / kill.

    LiveKit console mode spawns a job subprocess; on macOS that child can land
    in its own session (via multiprocessing spawn + setsid), so a terminal-window
    close (SIGHUP) reaches the parent but not the child — leaving an orphan that
    keeps the mic + API connections alive. On SIGHUP/SIGTERM we SIGKILL our entire
    process group, which takes the parent, the job subprocess, and any in-flight
    `claude -p` doer down together. SIGINT (Ctrl+C) is left to LiveKit's graceful
    drain.
    """
    import signal

    def _kill_group(signum, frame):
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except Exception:
            os._exit(0)

    for _sig in (signal.SIGHUP, signal.SIGTERM):
        try:
            signal.signal(_sig, _kill_group)
        except Exception:
            pass


if __name__ == "__main__":
    _install_hard_shutdown()
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="JARVIS",
        )
    )
