"""Voice LLM Queue — accumulates sub-agent results between LLM runs.

Items are flushed into conversation history as tool messages (not system
prompt) to enable prompt caching and prevent amnesia.

Per the architecture doc (Option A — Hold and batch):
  Status updates accumulate silently. Only an agent_result triggers
  the next LLM run. Status context is still in history so the LLM
  sees what happened.
"""

from __future__ import annotations

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
        """Drain the queue and append items to the ChatContext.
        
        Items are added as 'user' role so the LLM treats them as input
        requiring a response (system messages may be deprioritized).
        """
        import logging
        _logger = logging.getLogger("niceguy")

        parts = []
        for item in self.items:
            parts.append(
                f"[{item.type.upper()} | {item.agent} | "
                f"{item.timestamp.isoformat()}]\n{item.summary}"
            )
        
        if parts:
            combined = "\n\n".join(parts)
            _logger.info("Flushing %d queue items to chat_ctx:\n%s", len(self.items), combined[:300])
            chat_ctx.add_message(
                role="user",
                content=f"[SYSTEM — Sub-agent results below. Present these to the user naturally.]\n\n{combined}",
            )
        
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
