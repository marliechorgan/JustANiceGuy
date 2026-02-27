import asyncio
import json
import logging
import re
import os
from datetime import datetime
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

load_dotenv()

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
something — even a brief acknowledgement like "On it, sir" or "Let me \
check that for you." Silent tool calls are NOT allowed.
- Output 1-2 sentences maximum per response. Call set_turn_mode("CONTINUE") \
if you have more to say. This creates natural pacing.
- When dispatching work, acknowledge immediately with speech ("I'll check \
that now"), call dispatch_openclaw with a clear directive, and call \
set_turn_mode("ACKWAIT").
- When you have tool results to present and the information spans more \
than 2 sentences, present 1-2 sentences and use CONTINUE.
- Synthesise and prioritise information. Don't parrot raw data.
- Be conversational, concise, and natural. You are speaking, not writing.
- ALWAYS use the native tool calling interface. NEVER output tool calls, \
function names, XML tags, or code blocks in your text response to the user for TTS. Your \
text output is spoken aloud — it must be pure natural language.

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
    """Remove any tool call fragments Gemini leaks into the text stream.
    
    Gemini sometimes emits partial XML-like tags (<call:...>, <function_call>, etc.)
    or trailing tool metadata in the text content. Strip these so TTS only speaks
    clean natural language.
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
    # Clean up whitespace
    text = text.strip()
    return text


class VoiceTools:
    def __init__(self, queue: VoiceLLMQueue):
        self._queue = queue
        self._pending_directives: list[str] = []
        self._inflight_count = 0  # Track background tasks still running
        self.current_turn_mode = "CONV"

    @llm.function_tool(
        description="Send a task to the agent system. Write a clear natural "
        "language directive. Do NOT specify agent names or parameters."
    )
    async def dispatch_openclaw(self, directive: str) -> str:
        """Dispatch a natural language task to the agent system."""
        logger.info("LLM dispatched directive: %s", directive)
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
        for d in self._pending_directives:
            self._inflight_count += 1
            asyncio.create_task(self._tracked_dispatch(dispatch_fn, d))
        self._pending_directives.clear()

    async def _tracked_dispatch(self, dispatch_fn, directive: str) -> None:
        """Wrap dispatch to decrement inflight counter when done."""
        try:
            await dispatch_fn(directive, self._queue)
        finally:
            self._inflight_count -= 1
            logger.info("Inflight tasks remaining: %d", self._inflight_count)

    def has_pending(self) -> bool:
        """True if there are unfired directives OR background tasks in flight."""
        return bool(self._pending_directives) or self._inflight_count > 0


def prewarm(proc: JobProcess):
    """Pre-warm models when the worker process starts."""
    proc.userdata["vad"] = silero.VAD.load()


async def wait_for_either(user_speech_task, queue_update_task):
    """Wait for whichever comes first: user speaks or queue gets new items."""
    done, pending = await asyncio.wait(
        [user_speech_task, queue_update_task],
        return_when=asyncio.FIRST_COMPLETED
    )
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
        model="gemini-3-flash-preview",
        api_key=os.environ["GEMINI_API_KEY"],
        thinking_config=genai_types.ThinkingConfig(thinking_level="LOW"),
        tool_choice="auto",
    )
    tts_model = elevenlabs.TTS(
        api_key=os.environ["ELEVENLABS_API_KEY"],
        model="eleven_flash_v2_5",
        voice_id="lUTamkMw7gOzZbFIwmq4",
        voice_settings=elevenlabs.VoiceSettings(
            speed=1.14, stability=0.40, similarity_boost=0.75, style=0.0, use_speaker_boost=True
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

    @session.on("conversation_item_added")
    def on_conversation_item_added(ev):
        item = ev.item
        if isinstance(item, llm.ChatMessage) and item.role == "user" and isinstance(item.content, list):
            text_content = " ".join([c for c in item.content if isinstance(c, str)])
            if text_content:
                logger.info("User said: %s", text_content)
                asyncio.create_task(speech_queue.put(text_content))

    @session.on("user_started_speaking")
    def on_user_started_speaking():
        barge_in_event.set()

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

    while True:
        logger.info("Waiting for trigger...")
        trigger = await wait_for_either(speech_task, queue_task)

        is_speech = trigger != "QUEUE_UPDATE"

        if is_speech:
            chat_ctx.add_message(role="user", content=trigger)
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
                    # Make sure the text stream is closed
                    try:
                        await text_queue.put(None)
                    except Exception:
                        pass
                    if attempt < 2:
                        logger.warning("LLM error (attempt %d/3), retrying: %s", attempt + 1, e)
                        await asyncio.sleep(0.5 * (attempt + 1))
                        speech_text = ""
                        tool_calls = []
                    else:
                        logger.error("LLM failed after 3 attempts: %s", e)

            if llm_error:
                session.say("Apologies sir, I seem to be having a moment. Could you try that again?", 
                           add_to_chat_ctx=False)
                break  # Break CONTINUE loop, wait for next trigger

            if interrupted:
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
                    has_sig = tc.call_id in llm_model._thought_signatures
                    if has_sig:
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
                logger.info("Session ended by agent.")
                await ctx.room.disconnect()
                return

if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="JARVIS",
        )
    )
