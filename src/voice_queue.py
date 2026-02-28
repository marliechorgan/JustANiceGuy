"""Voice LLM Queue — accumulates sub-agent results between LLM runs.

Items are flushed into conversation history as tool messages (not system
prompt) to enable prompt caching and prevent amnesia.

Per the architecture doc (Option A — Hold and batch):
  Status updates accumulate silently. Only an agent_result triggers
  the next LLM run. Status context is still in history so the LLM
  sees what happened.
"""

from __future__ import annotations
import re

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


@dataclass
class QueueItem:
    """A single item in the Voice LLM Queue."""

    type: Literal["agent_result", "status_update"]
    agent: str
    content: dict
    summary: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VoiceLLMQueue:
    """Queue for sub-agent results. Flushes to conversation history before each LLM run."""

    def __init__(self) -> None:
        self.items: list[QueueItem] = []
        self._result_event = asyncio.Event()  # Only set on agent_result

    def push(self, item: QueueItem) -> None:
        """Add an item. Only signal waiters when an agent_result arrives."""
        self.items.append(item)
        if item.type == "agent_result":
            self._result_event.set()

    def flush_to_chat_ctx(self, chat_ctx) -> None:
        """Drain the queue and append agent_result items to the ChatContext.
        
        Only agent_result items are flushed — status_update items are noise
        (e.g. "Dispatching to OpenClaw...") that waste LLM tokens.
        Items are added as 'user' role so the LLM treats them as input
        requiring a response (system messages may be deprioritized).
        """
        import logging
        _logger = logging.getLogger("niceguy")

        results = [item for item in self.items if item.type == "agent_result"]
        status_count = len(self.items) - len(results)

        parts = []
        for item in results:
            cleaned_summary = _clean_for_voice(item.summary)
            parts.append(
                f"[AGENT_RESULT | {item.agent} | "
                f"{item.timestamp.isoformat()}]\n{cleaned_summary}"
            )
        
        if parts:
            combined = "\n\n".join(parts)
            _logger.info("Flushing %d results to chat_ctx (dropped %d status updates):\n%s",
                        len(results), status_count, combined[:300])
            chat_ctx.add_message(
                role="user",
                content=f"[SYSTEM — Sub-agent results below. Present these to the user naturally.]\n\n{combined}",
            )
        elif status_count:
            _logger.debug("Dropped %d status-only queue items (no results to flush)", status_count)
        
        self.items.clear()
        self._result_event.clear()

    def has_items(self) -> bool:
        return bool(self.items)

    def has_results(self) -> bool:
        """True if there's at least one agent_result (not just status updates)."""
        return any(i.type == "agent_result" for i in self.items)

    async def wait_for_result(self) -> None:
        """Block until an agent_result arrives. Status updates accumulate silently."""
        await self._result_event.wait()


def _clean_for_voice(text: str) -> str:
    """Strip markdown and machine-readable formatting from agent results.
    
    Cleans the text BEFORE it enters chat_ctx so the LLM receives
    natural prose instead of structured data it might parrot.
    """
    # Strip markdown bold/italic
    text = text.replace('**', '').replace('__', '')
    # Strip bullet points and list markers
    text = re.sub(r'^\s*[-*•]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    # Strip markdown headers
    text = re.sub(r'^\s*#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Strip STATUS: SUCCESS/ERROR prefixes
    text = re.sub(r'STATUS:\s*(SUCCESS|ERROR|FAILURE)[.:]?\s*', '', text, flags=re.IGNORECASE)
    # Strip "Action:" prefix
    text = re.sub(r'^Action:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)
    # Strip inline code backticks
    text = re.sub(r'`([^`]*)`', r'\1', text)
    # Strip emoji
    text = re.sub(
        r'[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FE0F'
        r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0'
        r'\U0000200D\U0000FE0F]+', '', text
    )
    # Collapse excess whitespace
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()
