"""OpenClaw stub — LLM-powered simulation of async sub-agent execution.

Uses a lightweight Gemini call to generate realistic, contextual responses
to directives. Simulates real-world latency with progressive status updates.
Replace with real OpenClaw orchestrator integration in production.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random

from google import genai
from google.genai import types as genai_types

from voice_queue import QueueItem, VoiceLLMQueue

logger = logging.getLogger("niceguy.stub")

# Lazy-init the genai client
_client: genai.Client | None = None

def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


# Agent routing based on directive content
AGENT_ROUTING = {
    "email": {"agent": "email_agent", "status": "Connecting to email service and scanning inbox..."},
    "calendar": {"agent": "calendar_agent", "status": "Accessing calendar and retrieving events..."},
    "schedule": {"agent": "calendar_agent", "status": "Accessing calendar and retrieving events..."},
    "meeting": {"agent": "calendar_agent", "status": "Checking meeting details..."},
    "flight": {"agent": "travel_agent", "status": "Accessing travel booking system..."},
    "check-in": {"agent": "travel_agent", "status": "Processing check-in request..."},
    "check in": {"agent": "travel_agent", "status": "Processing check-in request..."},
    "weather": {"agent": "weather_agent", "status": "Fetching weather data..."},
    "news": {"agent": "news_agent", "status": "Scanning news sources..."},
    "research": {"agent": "research_agent", "status": "Conducting research..."},
    "code": {"agent": "code_agent", "status": "Analyzing codebase..."},
    "music": {"agent": "music_agent", "status": "Accessing music library..."},
}


def _route_directive(directive: str) -> tuple[str, str]:
    """Route a directive to the appropriate agent based on keywords."""
    directive_lower = directive.lower()
    for keyword, info in AGENT_ROUTING.items():
        if keyword in directive_lower:
            return info["agent"], info["status"]
    return "general_agent", "Processing request..."


STUB_SYSTEM_PROMPT = """\
You are simulating an AI sub-agent that executes tasks on behalf of a voice assistant.
Given a directive, generate a REALISTIC and DETAILED result as if you actually performed the task.

Rules:
- Return a JSON object with "status" (success/error) and "data" (the result details)
- Include realistic names, times, subjects, amounts, etc.
- Make it contextually appropriate for a UK-based professional user
- Keep it concise but informative — this will be summarized by another LLM for voice output
- Be creative and varied — don't repeat the same data patterns

Example for "Check user's emails":
{
  "status": "success",
  "data": {
    "total_unread": 7,
    "important": [
      {"from": "James Henderson", "subject": "Dinner Friday?", "preview": "Are you free Friday? Thinking of trying the new Italian place on King's Road.", "time": "2 hours ago", "priority": "normal"},
      {"from": "British Airways", "subject": "Check-in open — BA287 to NYC", "preview": "Your flight departs at 09:15 tomorrow. Check in now.", "time": "3 hours ago", "priority": "high"}
    ]
  }
}
"""


async def dispatch_openclaw_stub(directive: str, queue: VoiceLLMQueue) -> None:
    """LLM-powered OpenClaw simulation with realistic latency."""
    agent_name, status_msg = _route_directive(directive)

    # Phase 1: Router analysis (instant)
    queue.push(
        QueueItem(
            type="status_update",
            agent="router",
            content={},
            summary=f"Analyzed directive and routing to {agent_name}...",
        )
    )

    # Phase 2: Agent connecting (0.5-1.5s delay)
    await asyncio.sleep(random.uniform(0.5, 1.5))
    queue.push(
        QueueItem(
            type="status_update",
            agent=agent_name,
            content={},
            summary=status_msg,
        )
    )

    # Phase 3: LLM-powered result generation (simulates real work)
    await asyncio.sleep(random.uniform(1.0, 2.5))

    try:
        client = _get_client()
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=f"Directive: {directive}\n\nGenerate a realistic JSON result for this task.",
            config=genai_types.GenerateContentConfig(
                system_instruction=STUB_SYSTEM_PROMPT,
                temperature=0.9,
                max_output_tokens=500,
                response_mime_type="application/json",
            ),
        )
        result_text = response.text.strip() if response.text else '{"status": "success", "data": {}}'

        # Parse and validate JSON
        try:
            result_data = json.loads(result_text)
        except json.JSONDecodeError:
            result_data = {"status": "success", "raw": result_text}

        # Build a human-readable summary from the result
        summary = _summarize_result(directive, agent_name, result_data)

        queue.push(
            QueueItem(
                type="agent_result",
                agent=agent_name,
                content=result_data,
                summary=summary,
            )
        )
        logger.info("Stub agent '%s' completed: %s", agent_name, summary[:120])

    except Exception as e:
        logger.error("Stub LLM error: %s", e)
        # Fallback: push a generic success result
        queue.push(
            QueueItem(
                type="agent_result",
                agent=agent_name,
                content={"status": "success", "note": "Simulated result"},
                summary=f"Successfully completed: {directive}",
            )
        )


def _summarize_result(directive: str, agent: str, data: dict) -> str:
    """Build a concise text summary from structured result data."""
    status = data.get("status", "success")

    if status == "error":
        return f"Error executing task: {data.get('error', 'Unknown error')}"

    inner = data.get("data", data)

    # Email results
    if "important" in inner or "emails" in inner:
        emails = inner.get("important", inner.get("emails", []))
        if isinstance(emails, list) and emails:
            parts = []
            for i, e in enumerate(emails[:5], 1):
                sender = e.get("from", e.get("sender", "Unknown"))
                subject = e.get("subject", "No subject")
                priority = e.get("priority", "normal")
                time = e.get("time", "")
                parts.append(f"({i}) {sender} — {subject} [{priority}] {time}")
            return f"Found {len(emails)} messages: " + ". ".join(parts)

    # Calendar results
    if "events" in inner:
        events = inner.get("events", [])
        if isinstance(events, list) and events:
            parts = []
            for e in events[:5]:
                title = e.get("title", e.get("name", "Event"))
                time = e.get("time", e.get("start", ""))
                parts.append(f"{title} at {time}")
            return f"Found {len(events)} upcoming events: " + "; ".join(parts)

    # Travel / check-in results
    if "booking" in inner or "confirmation" in inner or "boarding_pass" in inner:
        conf = inner.get("confirmation", inner.get("booking", {}))
        if isinstance(conf, dict):
            return f"Travel update: {json.dumps(conf)}"

    # Generic: just dump the summary
    if isinstance(inner, dict):
        # Try to build something readable
        readable_parts = []
        for k, v in inner.items():
            if isinstance(v, (str, int, float, bool)):
                readable_parts.append(f"{k}: {v}")
        if readable_parts:
            return "; ".join(readable_parts[:5])

    return f"Task completed successfully: {directive}"
