"""OpenClaw integration — dispatch directives to the real OpenClaw agent system.

Uses the `openclaw agent` CLI to send directives and receive responses.
The Gateway runs locally at ws://127.0.0.1:18789. The CLI handles auth,
device signing, and session management.

Status updates and final results are pushed into the VoiceLLMQueue.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid

from voice_queue import QueueItem, VoiceLLMQueue

logger = logging.getLogger("niceguy.openclaw")

# Configurable via .env (see .env.example)
DEFAULT_AGENT = os.environ.get("OPENCLAW_AGENT", "main")
AGENT_TIMEOUT = int(os.environ.get("OPENCLAW_TIMEOUT", "600"))

# Voice-specific context prepended to every directive.
# Tells the main agent to handle simple tasks directly instead of
# spawning sub-agents, which saves 1-2 LLM round-trips.
VOICE_DIRECTIVE_PREFIX = (
    "[VOICE COMMAND FROM JARVIS ROUTER]\n"
    "CRITICAL INSTRUCTION: Do NOT output conversational filler, polite phrases, or emojis. "
    "Another AI is acting as the voice interface and will speak to the user. "
    "You are a backend execution tool. Output ONLY raw data, facts, or a terse status code "
    "(e.g., 'STATUS: SUCCESS. Action: Music paused.'). Output nothing else. "
    "Do NOT spawn sub-agents unless the task requires specialist tools.\n\n"
)


async def dispatch_openclaw(directive: str, queue: VoiceLLMQueue) -> None:
    """Send a directive to OpenClaw and push results into the queue."""

    # Phase 1: Immediate status update
    queue.push(
        QueueItem(
            type="status_update",
            agent="openclaw_router",
            content={},
            summary=f"Dispatching to OpenClaw: {directive[:100]}...",
        )
    )

    start = time.monotonic()

    # Wrap directive with voice-specific context
    wrapped_directive = VOICE_DIRECTIVE_PREFIX + directive

    # Use a persistent session ID so OpenClaw caches the system prompt.
    # This avoids 14s cold-starts and massive context resubmissions
    # which trigger LLM API rate limits (causing "fetch failed" errors).
    session_id = "jarvis-voice-persistent"

    # Build CLI command
    cmd = [
        "openclaw", "agent",
        "--agent", DEFAULT_AGENT,
        "--message", wrapped_directive,
        "--session-id", session_id,
        "--json",
        "--timeout", str(AGENT_TIMEOUT),
    ]
    logger.info("OpenClaw session: %s", session_id)
    logger.info("OpenClaw directive (full): %s", directive)

    try:
        # Run the OpenClaw CLI as a subprocess
        t_spawn = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        t_spawned = time.monotonic()

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=AGENT_TIMEOUT + 10,  # Extra buffer for CLI overhead
        )
        t_done = time.monotonic()

        # Timing telemetry for latency diagnostics
        logger.info(
            "OpenClaw timing: spawn=%.0fms exec=%.0fms total=%.0fms",
            (t_spawned - t_spawn) * 1000,
            (t_done - t_spawned) * 1000,
            (t_done - start) * 1000,
        )

        # Log stderr (always at INFO so it shows in session logs)
        if stderr:
            stderr_text = stderr.decode().strip()
            if stderr_text:
                logger.info("OpenClaw stderr:\n%s", stderr_text[:2000])

        elapsed = time.monotonic() - start
        logger.info("OpenClaw completed in %.1fs (exit=%s)", elapsed, proc.returncode)

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() or f"Exit code {proc.returncode}"
            logger.error("OpenClaw error: %s", error_msg)
            # Log raw stdout too in case it has useful error info
            raw_out = stdout.decode().strip()
            if raw_out:
                logger.error("OpenClaw error stdout: %s", raw_out[:1000])
            queue.push(
                QueueItem(
                    type="agent_result",
                    agent="openclaw",
                    content={"status": "error", "error": error_msg},
                    summary=f"OpenClaw encountered an error: {error_msg[:200]}",
                )
            )
            return

        # Parse the JSON response
        raw = stdout.decode().strip()
        logger.info("OpenClaw raw response length: %d bytes", len(raw))
        
        # Intercept common text-based errors from Gateway before JSON parsing
        if "fetch failed" in raw.lower():
            logger.error("OpenClaw hit upstream AI rate limit: fetch failed")
            queue.push(
                QueueItem(
                    type="agent_result",
                    agent="openclaw",
                    content={"status": "error", "raw": "fetch failed"},
                    summary="The upstream AI provider is currently rate-limited. Please try again in a moment.",
                )
            )
            return
            
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("OpenClaw returned invalid JSON: %s", raw[:500])
            queue.push(
                QueueItem(
                    type="agent_result",
                    agent="openclaw",
                    content={"status": "error", "raw": raw[:500]},
                    summary=f"OpenClaw returned an unparseable response.",
                )
            )
            return

        # Extract the response text from payloads
        status = result.get("status", "unknown")
        payloads = result.get("result", {}).get("payloads", [])
        meta = result.get("result", {}).get("meta", {})

        # Build summary from all text payloads
        texts = [p.get("text", "") for p in payloads if p.get("text")]
        summary = "\n\n".join(texts) if texts else "No response text from OpenClaw."

        # Extract agent metadata
        agent_meta = meta.get("agentMeta", {})
        duration_ms = meta.get("durationMs", 0)
        model_used = agent_meta.get("model", "unknown")
        agent_name = agent_meta.get("agent", "unknown")
        actions_taken = agent_meta.get("actions", [])

        logger.info(
            "OpenClaw result: status=%s, payloads=%d, model=%s, agent=%s, duration=%dms",
            status, len(payloads), model_used, agent_name, duration_ms,
        )
        if actions_taken:
            logger.info("OpenClaw actions: %s", json.dumps(actions_taken, default=str)[:500])
        
        # Log full summary (not truncated) so everything shows in session logs
        logger.info("OpenClaw summary: %s", summary[:500])
        
        # Log each payload's details
        for i, p in enumerate(payloads):
            p_type = p.get("type", "text")
            p_text = p.get("text", "")[:300]
            p_data = {k: v for k, v in p.items() if k not in ("text",)}
            logger.info("OpenClaw payload[%d] type=%s: %s", i, p_type, p_text)
            if p_data:
                logger.info("OpenClaw payload[%d] metadata: %s", i, json.dumps(p_data, default=str)[:300])

        queue.push(
            QueueItem(
                type="agent_result",
                agent="openclaw",
                content={
                    "status": status,
                    "payloads": payloads,
                    "model": model_used,
                    "duration_ms": duration_ms,
                },
                summary=summary,
            )
        )

    except asyncio.TimeoutError:
        elapsed = time.monotonic() - start
        logger.error("OpenClaw timed out after %.1fs", elapsed)
        # Kill the orphaned subprocess
        try:
            proc.kill()
            await proc.wait()
        except Exception:
            pass
        queue.push(
            QueueItem(
                type="agent_result",
                agent="openclaw",
                content={"status": "error", "error": "timeout"},
                summary=f"OpenClaw timed out after {elapsed:.0f} seconds.",
            )
        )

    except Exception as e:
        logger.error("OpenClaw dispatch error: %s", e, exc_info=True)
        queue.push(
            QueueItem(
                type="agent_result",
                agent="openclaw",
                content={"status": "error", "error": str(e)},
                summary=f"Failed to dispatch to OpenClaw: {e}",
            )
        )

