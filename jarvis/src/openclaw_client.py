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
import time

from voice_queue import QueueItem, VoiceLLMQueue

logger = logging.getLogger("niceguy.openclaw")

# Default agent to route to (OpenClaw's "main" agent handles routing)
DEFAULT_AGENT = "main"

# Timeout for agent execution (seconds)
AGENT_TIMEOUT = 60


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

    try:
        # Run the OpenClaw CLI as a subprocess
        proc = await asyncio.create_subprocess_exec(
            "openclaw", "agent",
            "--agent", DEFAULT_AGENT,
            "--message", directive,
            "--json",
            "--timeout", str(AGENT_TIMEOUT),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            proc.communicate(),
            timeout=AGENT_TIMEOUT + 10,  # Extra buffer for CLI overhead
        )

        elapsed = time.monotonic() - start
        logger.info("OpenClaw completed in %.1fs (exit=%s)", elapsed, proc.returncode)

        if proc.returncode != 0:
            error_msg = stderr.decode().strip() or f"Exit code {proc.returncode}"
            logger.error("OpenClaw error: %s", error_msg)
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
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("OpenClaw returned invalid JSON: %s", raw[:200])
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

        logger.info(
            "OpenClaw result: status=%s, payloads=%d, model=%s, duration=%dms",
            status, len(payloads), model_used, duration_ms,
        )
        logger.info("OpenClaw summary: %s", summary[:200])

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
        queue.push(
            QueueItem(
                type="agent_result",
                agent="openclaw",
                content={"status": "error", "error": "timeout"},
                summary=f"OpenClaw timed out after {elapsed:.0f} seconds.",
            )
        )

    except Exception as e:
        logger.error("OpenClaw dispatch error: %s", e)
        queue.push(
            QueueItem(
                type="agent_result",
                agent="openclaw",
                content={"status": "error", "error": str(e)},
                summary=f"Failed to dispatch to OpenClaw: {e}",
            )
        )
