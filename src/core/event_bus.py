"""비동기 이벤트 버스 — 컴포넌트 간 느슨한 결합 통신."""

import asyncio
import logging
from collections import defaultdict
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

Handler = Callable[[Any], Awaitable[None]]


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Handler]] = defaultdict(list)
        self._queues: dict[str, list[asyncio.Queue]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Handler) -> None:
        self._subscribers[event_type].append(callback)

    def create_queue(self, event_type: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._queues[event_type].append(queue)
        return queue

    async def publish(self, event_type: str, data: Any = None) -> None:
        for callback in self._subscribers[event_type]:
            try:
                await callback(data)
            except Exception as e:
                logger.error(
                    "Event handler error for %s: %s", event_type, e, exc_info=True
                )
        for queue in self._queues[event_type]:
            await queue.put(data)
