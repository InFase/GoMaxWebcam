"""
events.py — Async SSE event bus for GoMaxWebcam v2.

A lightweight publish/subscribe event bus for broadcasting typed events
to SSE (Server-Sent Events) subscribers on the FastAPI dashboard.

Event types:
  - connection: Transport state changes (DISCONNECTED -> CONNECTING -> STREAMING)
  - transport: Transport type changes (USB -> WiFi AP, failover)
  - battery: Battery level updates from camera polling
  - pipeline: Pipeline state changes (STOPPED -> STREAMING -> FREEZE_FRAME)
  - error: Error events (connection failures, decode errors)
  - status: Full status snapshots (periodic aggregation)

Architecture:
  Publishers (camera manager, transport, pipeline) call bus.publish(event).
  Each SSE subscriber gets an asyncio.Queue fed by the bus.
  Backpressure: if a subscriber's queue is full, the oldest event is dropped.

Thread safety:
  - publish() is safe to call from any thread (schedules onto asyncio loop)
  - subscribe/unsubscribe must be called from the asyncio thread
  - The bus stores a reference to the event loop for cross-thread publish
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, replace as dataclass_replace
from enum import Enum
from typing import Any, Optional

log = logging.getLogger("gomaxwebcam.events")


class EventType(str, Enum):
    """Event types published on the SSE bus."""
    CONNECTION = "connection"      # Transport connection state changed
    TRANSPORT = "transport"        # Active transport type changed
    BATTERY = "battery"            # Battery level update
    PIPELINE = "pipeline"          # Pipeline state changed
    ERROR = "error"                # Error event
    STATUS = "status"              # Full status snapshot (periodic)
    KEEPALIVE = "keepalive"        # SSE keepalive ping (empty comment)
    COHN_REPROVISION = "cohn_reprovision"  # Cached COHN credentials invalid, BLE re-provisioning needed
    USB_REDETECTED = "usb_redetected"      # USB device found while COHN is active (fail-back trigger)


@dataclass(frozen=True)
class Event:
    """A single event published on the SSE bus.

    Attributes:
        type: Event category (maps to SSE 'event:' field).
        data: Event payload (JSON-serializable dict).
        timestamp: Monotonic timestamp of event creation.
        id: Sequential event ID assigned by the EventBus on publish.
            A value of 0 means no ID has been assigned yet.
            Clients can use the Last-Event-ID header to resume from this ID.
    """
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)
    id: int = 0

    def to_sse(self) -> str:
        """Format as an SSE message string.

        Returns:
            SSE-formatted string with id, event type, and data fields.
            If the event has an assigned ID (> 0), the ``id:`` field is
            included so browsers can track the last received event and
            send ``Last-Event-ID`` on reconnect.

        Example output::

            id: 42
            event: battery
            data: {"level": 85, "charging": false}

        """
        json_data = json.dumps(self.data, default=str)
        if self.id:
            return f"id: {self.id}\nevent: {self.type.value}\ndata: {json_data}\n\n"
        return f"event: {self.type.value}\ndata: {json_data}\n\n"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        d: dict[str, Any] = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        if self.id:
            d["id"] = self.id
        return d


# -- Convenience event constructors --

def connection_event(
    old_state: str,
    new_state: str,
    transport_name: str = "",
    detail: str = "",
) -> Event:
    """Create a connection state change event."""
    return Event(
        type=EventType.CONNECTION,
        data={
            "old_state": old_state,
            "new_state": new_state,
            "transport": transport_name,
            "detail": detail,
        },
    )


def transport_event(
    old_transport: str,
    new_transport: str,
    reason: str = "",
) -> Event:
    """Create a transport change event (e.g. failover)."""
    return Event(
        type=EventType.TRANSPORT,
        data={
            "old_transport": old_transport,
            "new_transport": new_transport,
            "reason": reason,
        },
    )


def battery_event(level: int, charging: bool = False) -> Event:
    """Create a battery level update event."""
    return Event(
        type=EventType.BATTERY,
        data={"level": level, "charging": charging},
    )


def pipeline_event(
    old_state: str,
    new_state: str,
    detail: str = "",
) -> Event:
    """Create a pipeline state change event."""
    return Event(
        type=EventType.PIPELINE,
        data={
            "old_state": old_state,
            "new_state": new_state,
            "detail": detail,
        },
    )


def error_event(
    source: str,
    message: str,
    recoverable: bool = True,
) -> Event:
    """Create an error event."""
    return Event(
        type=EventType.ERROR,
        data={
            "source": source,
            "message": message,
            "recoverable": recoverable,
        },
    )


def status_event(status_data: dict[str, Any]) -> Event:
    """Create a full status snapshot event."""
    return Event(
        type=EventType.STATUS,
        data=status_data,
    )


def pipeline_metrics_event(
    fps: float,
    decode_ms: float,
    drop_rate: float,
    jitter_ms: float,
    health: str,
    frames_decoded: int = 0,
    frames_dropped: int = 0,
) -> Event:
    """Create a pipeline metrics snapshot event.

    Published periodically (every ~1s) by the status tracker to feed
    dashboard diagnostics charts and health indicators.
    """
    return Event(
        type=EventType.PIPELINE,
        data={
            "metrics": True,
            "fps": round(fps, 1),
            "decode_ms": round(decode_ms, 2),
            "drop_rate": round(drop_rate, 4),
            "jitter_ms": round(jitter_ms, 2),
            "health": health,
            "frames_decoded": frames_decoded,
            "frames_dropped": frames_dropped,
        },
    )


def cohn_reprovision_event(
    reason: str = "",
    camera_serial: str = "",
    camera_ip: str = "",
) -> Event:
    """Create a COHN re-provisioning needed event.

    Published when cached COHN credentials fail (expired, camera reset, etc.)
    and BLE re-provisioning is required to restore COHN connectivity.
    """
    return Event(
        type=EventType.COHN_REPROVISION,
        data={
            "reason": reason,
            "camera_serial": camera_serial,
            "camera_ip": camera_ip,
        },
    )


def usb_redetected_event(
    camera_serial: str = "",
    camera_ip: str = "",
) -> Event:
    """Create a USB re-detected event (fail-back trigger).

    Published by the USB polling loop when a GoPro USB device is found
    on the bus while the COHN transport is active.  The orchestrator or
    TransportManager can use this to initiate a fail-back to USB.
    """
    return Event(
        type=EventType.USB_REDETECTED,
        data={
            "camera_serial": camera_serial,
            "camera_ip": camera_ip,
        },
    )


# ---------------------------------------------------------------------------
# SSE Event Bus
# ---------------------------------------------------------------------------

#: Default max events buffered per subscriber before dropping oldest
_DEFAULT_QUEUE_SIZE = 64

#: Sentinel value to signal subscriber to stop
_STOP_SENTINEL = None


class EventBus:
    """Async event bus for broadcasting events to SSE subscribers.

    Usage:
        bus = EventBus()

        # Publisher side (any thread):
        bus.publish(battery_event(85))

        # Subscriber side (async):
        async for event in bus.subscribe():
            yield event.to_sse()

        # Shutdown:
        await bus.shutdown()

    The bus is designed for a single asyncio event loop. Call set_loop()
    to configure the loop reference for cross-thread publishing.
    """

    def __init__(self, max_queue_size: int = _DEFAULT_QUEUE_SIZE) -> None:
        self._max_queue_size = max_queue_size
        self._subscribers: list[asyncio.Queue[Event | None]] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._event_count: int = 0
        self._history: list[Event] = []
        self._history_max: int = 100  # Keep last 100 events for late joiners

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Set the asyncio event loop for cross-thread publishing."""
        self._loop = loop

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        return len(self._subscribers)

    @property
    def event_count(self) -> int:
        """Total events published since bus creation."""
        return self._event_count

    @property
    def recent_events(self) -> list[Event]:
        """Return recent event history (newest last)."""
        return list(self._history)

    # -- Publishing --

    def publish(self, event: Event) -> None:
        """Publish an event to all subscribers.

        Safe to call from any thread. If called from a non-asyncio thread,
        the event is scheduled on the event loop via call_soon_threadsafe.

        A sequential integer ID is assigned to the event before dispatch so
        clients can reference it via ``Last-Event-ID`` on reconnect.
        """
        self._event_count += 1

        # Assign a unique sequential ID so clients can resume with Last-Event-ID
        if event.id == 0:
            event = dataclass_replace(event, id=self._event_count)

        # Maintain bounded history
        self._history.append(event)
        if len(self._history) > self._history_max:
            self._history = self._history[-self._history_max:]

        # Try to detect if we're on the event loop thread
        loop = self._loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
                self._loop = loop
            except RuntimeError:
                # No running loop — try to push directly
                self._dispatch(event)
                return

        try:
            # Check if we're on the loop's thread
            if loop.is_running():
                try:
                    asyncio.get_running_loop()
                    # We're on the async thread — dispatch directly
                    self._dispatch(event)
                except RuntimeError:
                    # We're on a different thread — schedule
                    loop.call_soon_threadsafe(self._dispatch, event)
            else:
                self._dispatch(event)
        except RuntimeError:
            # Loop is closed — just dispatch in place
            self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        """Push event to all subscriber queues (must run on loop thread)."""
        dead: list[asyncio.Queue] = []

        for q in self._subscribers:
            try:
                if q.full():
                    # Backpressure: drop oldest event
                    try:
                        q.get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                q.put_nowait(event)
            except Exception:
                dead.append(q)

        # Clean up dead subscribers
        for q in dead:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

        if dead:
            log.debug("Cleaned up %d dead subscribers", len(dead))

    # -- Subscription --

    def subscribe(
        self,
        event_types: Optional[set[EventType]] = None,
        include_history: bool = False,
        since_event_id: Optional[int] = None,
    ) -> EventSubscription:
        """Create a new SSE subscription.

        Args:
            event_types: If provided, only receive events of these types.
                         None = receive all events.
            include_history: If True, replay all recent events on subscribe.
            since_event_id: If provided, replay only history events whose
                ``id`` is strictly greater than this value.  Clients should
                pass the value received from the ``Last-Event-ID`` header on
                reconnect.  Takes precedence over ``include_history`` for
                the history replay logic: when ``since_event_id`` is set,
                only missed events (id > since_event_id) are replayed even
                if ``include_history`` is False.

        Returns:
            An async iterator that yields Event objects.
        """
        q: asyncio.Queue[Event | None] = asyncio.Queue(
            maxsize=self._max_queue_size,
        )
        self._subscribers.append(q)

        # Determine which history events to replay
        if since_event_id is not None:
            # Resume mode: only replay events the client hasn't seen yet
            for evt in self._history:
                if evt.id > since_event_id:
                    if event_types is None or evt.type in event_types:
                        try:
                            q.put_nowait(evt)
                        except asyncio.QueueFull:
                            break
        elif include_history:
            # Full history replay
            for evt in self._history:
                if event_types is None or evt.type in event_types:
                    try:
                        q.put_nowait(evt)
                    except asyncio.QueueFull:
                        break

        log.debug(
            "New subscriber (total=%d, filter=%s, since_id=%s)",
            len(self._subscribers),
            event_types or "all",
            since_event_id,
        )

        return EventSubscription(q, self._subscribers, event_types)

    def unsubscribe(self, subscription: EventSubscription) -> None:
        """Remove a subscription (also happens automatically on iteration exit)."""
        subscription._cleanup()

    # -- Lifecycle --

    async def shutdown(self) -> None:
        """Signal all subscribers to stop and clear state."""
        log.info("Shutting down event bus (%d subscribers)", len(self._subscribers))

        for q in self._subscribers:
            try:
                q.put_nowait(_STOP_SENTINEL)
            except asyncio.QueueFull:
                # Force by draining and re-adding sentinel
                try:
                    q.get_nowait()
                    q.put_nowait(_STOP_SENTINEL)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        self._subscribers.clear()
        self._history.clear()


class EventSubscription:
    """Async iterator for SSE events.

    Supports optional type filtering and automatic cleanup.

    Usage:
        async for event in bus.subscribe(event_types={EventType.BATTERY}):
            yield event.to_sse()
    """

    def __init__(
        self,
        queue: asyncio.Queue[Event | None],
        subscribers: list[asyncio.Queue],
        event_types: Optional[set[EventType]] = None,
    ):
        self._queue = queue
        self._subscribers = subscribers
        self._event_types = event_types

    def __aiter__(self):
        return self

    async def __anext__(self) -> Event:
        while True:
            item = await self._queue.get()
            if item is None:
                # Sentinel: bus shutdown or unsubscribe
                self._cleanup()
                raise StopAsyncIteration

            # Apply type filter
            if self._event_types is not None and item.type not in self._event_types:
                continue

            return item

    def _cleanup(self) -> None:
        """Remove this subscription from the bus."""
        try:
            self._subscribers.remove(self._queue)
        except ValueError:
            pass
