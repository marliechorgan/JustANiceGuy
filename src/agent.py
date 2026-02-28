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
from livekit.plugins import deepgram, elevenlabs, google, silero

from openclaw_client import dispatch_openclaw as dispatch_openclaw_real
from openclaw_stub import dispatch_openclaw_stub
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

def build_system_prompt() -> str:
    """Build the system prompt with current date/time context."""
    from datetime import datetime
    now = datetime.now()
    date_str = now.strftime("%A %d %B %Y, %H:%M")

    return f"""\
You are "JARVIS" - a general personal assistant. Think Jarvis: polished, \
respectful, and efficient. Address the user as "sir" naturally. You're well-spoken \
and a touch posh, but modern — not archaic or stuffy.

CONTEXT:
- The user's name is Charlie.
- Current date and time: {date_str}.

RULES:
- CRITICAL: You MUST ALWAYS output spoken text BEFORE any tool calls. \
Never return tool calls without speaking first. The user must always hear \
something — even a brief acknowledgement. Silent tool calls are NOT allowed.
- Output 1-2 sentences maximum per response. Call set_turn_mode("CONTINUE") \
if you have more to say. This creates natural pacing.
- AVOID DOUBLE-SPEAKING: When dispatching work, give ONLY a very brief \
acknowledgment ("On it, sir." or "One moment." or "Let me check."). Do \
NOT describe what you're about to do — the user already asked for it. \
When the result comes back, SYNTHESIZE the key information into natural \
spoken English. NEVER read raw status codes (like "STATUS: SUCCESS"), \
headers, section titles, bullet point markers, or structured formatting \
verbatim. Extract the important facts and present them conversationally \
as if briefing the user. If the result is an error, state the error \
clearly and concisely.
- When dispatching work, call dispatch_openclaw with a clear directive, \
and call set_turn_mode("ACKWAIT").
- When you have tool results to present and the information spans more \
than 2 sentences, present 1-2 sentences and use CONTINUE.
- Synthesise and prioritise information. NEVER parrot raw data, status \
codes, or machine-readable formatting. Your output goes to a TTS engine.
- Be conversational, concise, and natural. You are speaking, not writing.
- NEVER use markdown formatting (no **bold**, no `backticks`, no bullet \
points, no headers). NEVER use emoji. Your output goes directly to a \
text-to-speech engine — it must be plain spoken English only.
- When mentioning file paths or technical terms, say them naturally \
(e.g. "the guide file in your docs folder" not "docs forward slash GUIDE \
dot md").
- ALWAYS use the native tool calling interface. NEVER output tool calls, \
function names, XML tags, or code blocks in your text response to the user for TTS. Your \
text output is spoken aloud — it must be pure natural language.

SYSTEM LIMITATIONS (be honest about these):
- You CANNOT cancel a task once dispatched. If the user asks to cancel, \
acknowledge but explain the task may still complete in the background.
- You CANNOT change system timeouts or configurations at runtime. If \
something is timing out, tell the user honestly rather than pretending to fix it.
- If a dispatch returns "Maximum retry attempts reached", do NOT try \
again. Inform the user the system is currently unavailable and suggest \
trying later or rephrasing.

TOOLS:
- dispatch_openclaw(directive): Send a natural language task to the agent \
system. Describe what needs to be done in plain English. Do NOT try to \
specify agent names, action types, or parameters — just describe the task.
- set_turn_mode(mode): Control what happens next.
  - ACKWAIT: You've dispatched work and need to wait for results.
  - CONTINUE: You have more to say. System re-runs you immediately with \
any new context that arrived in the meantime.
  - CONV: You've finished your thought. Open the mic for the user.
  - END: The conversation is over.

You will receive tool results as messages in the conversation history \
with the format: [AGENT_RESULT | agent_name | timestamp]
Treat these as data you've retrieved. Present them naturally.\
"""



def _strip_tool_leaks(text: str) -> str:
    """Remove any tool call fragments, markdown formatting, and emoji that
    Gemini leaks into the text stream.
    
    Gemini sometimes emits partial XML-like tags (<call:...>, <function_call>, etc.)
    or trailing tool metadata in the text content. It also passes through markdown
    formatting (**bold**, `code`) and emoji from OpenClaw responses. Strip all
    of these so TTS only speaks clean natural language.
    """
    # Remove <call:...> blocks and anything after them
    text = re.sub(r'<call:[^>]*>.*', '', text, flags=re.DOTALL)
    # Remove <function_call>...</function_call> blocks
    text = re.sub(r'<function_call>.*?</function_call>', '', text, flags=re.DOTALL)
    # Remove any trailing <tag or partial XML
    text = re.sub(r'<[a-zA-Z_/][^>]*$', '', text)
    # Remove ```tool_code blocks
    text = re.sub(r'```tool_code.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
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
        self._pending_directives: list[str] = []
        self._inflight_tasks: list[asyncio.Task] = []  # Track task handles for cancellation
        self._dispatch_counts: dict[str, int] = {}  # directive_key -> count (retry limiter)
        self.current_turn_mode = "CONV"

    @llm.function_tool(
        description="Send a task to the agent system. Write a clear natural "
        "language directive. Do NOT specify agent names or parameters."
    )
    async def dispatch_openclaw(self, directive: str) -> str:
        """Dispatch a natural language task to the agent system."""
        # Bug #5: Enforce retry limit to prevent infinite retry loops
        key = directive.lower().strip()[:80]
        count = self._dispatch_counts.get(key, 0)
        if count >= self.MAX_DISPATCH_RETRIES:
            logger.warning("Max retries (%d) reached for: %s", self.MAX_DISPATCH_RETRIES, key)
            return "Maximum retry attempts reached. The system appears unavailable."

        self._dispatch_counts[key] = count + 1
        logger.info("LLM dispatched directive: %s (attempt %d)", directive, count + 1)
        self._pending_directives.append(directive)
        self.current_turn_mode = "ACKWAIT"  # Default if not explicitly set
        return "Dispatched."

    @llm.function_tool(
        description="Control what happens after you finish speaking. Call this every turn to set the system's next state."
    )
    async def set_turn_mode(self, mode: str) -> str:
        """Control what happens after you finish speaking."""
        if mode in ["ACKWAIT", "CONTINUE", "CONV", "END"]:
            self.current_turn_mode = mode
            logger.info("LLM set turn mode to: %s", mode)
        return "Mode set."

    def flush_pending(self) -> None:
        """Fire off any queued directives as tracked background tasks."""
        use_stub = os.environ.get("USE_OPENCLAW_STUB", "").lower() in ("1", "true", "yes")
        dispatch_fn = dispatch_openclaw_stub if use_stub else dispatch_openclaw_real
        if not hasattr(self, '_dispatch_logged'):
            mode = "stub (simulated)" if use_stub else "OpenClaw CLI"
            logger.info("Dispatch mode: %s", mode)
            self._dispatch_logged = True
        # Clean up completed tasks first
        self._inflight_tasks = [t for t in self._inflight_tasks if not t.done()]
        for d in self._pending_directives:
            task = asyncio.create_task(self._tracked_dispatch(dispatch_fn, d))
            self._inflight_tasks.append(task)
        self._pending_directives.clear()

    async def _tracked_dispatch(self, dispatch_fn, directive: str) -> None:
        """Wrap dispatch to track completion."""
        try:
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

    logger.info("Connecting to room: %s", ctx.room.name)
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
    llm_model = google.LLM(
        model=os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview"),
        api_key=os.environ["GEMINI_API_KEY"],
        thinking_config=genai_types.ThinkingConfig(thinking_level="LOW"),
        tool_choice="auto",
    )
    tts_model = elevenlabs.TTS(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        model=os.environ.get("ELEVENLABS_MODEL", "eleven_flash_v2_5"),
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
            # Flush queue into conversation history before each LLM run
            if queue.has_items():
                queue.flush_to_chat_ctx(chat_ctx)

            # Bail out if session ended during flush or between iterations
            if session_ended["status"]:
                logger.info("Session ended — aborting LLM generation")
                break

            # Reset turn mode
            tools.current_turn_mode = "CONV"

            logger.info("Generating LLM response...")

            speech_text = ""
            tool_calls = []
            interrupted = False
            llm_error = False

            # Retry up to 2 times for transient Gemini 500 errors
            for attempt in range(3):
                try:
                    response = llm_model.chat(chat_ctx=chat_ctx, tools=tools_context.flatten())

                    # Async channel for streaming text tokens → TTS in real-time
                    text_queue: asyncio.Queue[str | None] = asyncio.Queue()

                    async def _text_stream():
                        """Async generator that yields text chunks as they arrive."""
                        while True:
                            token = await text_queue.get()
                            if token is None:
                                return
                            yield token

                    # Start TTS immediately with the streaming generator
                    UIState.tts_playing = True  # Bug #1: Gate STT
                    speech_handle = session.say(_text_stream(), add_to_chat_ctx=False)

                    async for chunk in response:
                        if barge_in_event.is_set():
                            logger.info("Barge-in detected! Halting LLM.")
                            interrupted = True
                            break

                        delta = chunk.delta
                        if delta:
                            if delta.content:
                                speech_text += delta.content
                                await text_queue.put(delta.content)
                            if delta.tool_calls:
                                tool_calls.extend(delta.tool_calls)

                    # Signal TTS stream end
                    await text_queue.put(None)
                    llm_error = False
                    break  # Success — exit retry loop

                except Exception as e:
                    llm_error = True
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
                        logger.warning("LLM error (attempt %d/3), retrying: %s", attempt + 1, e)
                        await asyncio.sleep(0.5 * (attempt + 1))
                        speech_text = ""
                        tool_calls = []
                    else:
                        logger.error("LLM failed after 3 attempts: %s", e)

            if llm_error:
                UIState.tts_playing = False  # Ensure STT gate is reset
                # Bug #2: Wrap in try/except — session may be closing
                try:
                    session.say("Apologies sir, I seem to be having a moment. Could you try that again?", 
                               add_to_chat_ctx=False)
                except RuntimeError:
                    logger.warning("Session closing, cannot speak error message")
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
                try:
                    await speech_handle
                    logger.info("TTS playout complete.")
                except Exception as e:
                    logger.warning("TTS playout error: %s", e)
                finally:
                    state["tts_playing"] = False  # Bug #1: Un-gate STT

            # --- Turn mode logic ---
            mode = tools.current_turn_mode
            logger.info("Turn mode decided: %s", mode)

            if mode == "CONTINUE":
                continue  # Re-run LLM immediately (TTS already finished)
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

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="JARVIS",
        )
    )
