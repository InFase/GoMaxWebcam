"""
failover.py -- Auto-failover state machine for GoMaxWebcam v2.

Manages the failover chain between transports (USB -> COHN -> freeze-frame)
with confirmation-based detection, exponential backoff, and network change
detection.

Architecture:
    TransportFailover
      - Monitors active transport health via frame timeout + health_check
      - On failure: stop_stream on old transport -> start_stream on new one
      - Freeze-frame covers the gap during hot-switch
      - Subscribes to EventBus for transport state changes
      - Publishes failover events to EventBus
      - Exponential backoff with cap, reset on network change

Thread model:
    All methods run on the asyncio event loop.
    Network polling runs as an asyncio task.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from gomaxwebcam.events import (
    EventBus,
    EventType,
    transport_event,
    error_event,
    connection_event,
)
from gomaxwebcam.transport.base import Transport, TransportState

log = logging.getLogger("gomaxwebcam.failover")

# Defaults
DEFAULT_FRAME_TIMEOUT_S = 3.0
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
NETWORK_POLL_INTERVAL_S = 2.5


class FailoverState(Enum):
    """State of the failover state machine."""
    IDLE = auto()              # No active monitoring
    MONITORING = auto()        # Watching active transport health
    DETECTING = auto()         # Frame timeout hit, confirming failure
    SWITCHING = auto()         # Hot-switching to next transport
    EXHAUSTED = auto()         # All transports failed, on freeze-frame


@dataclass
class FailoverConfig:
    """Configuration for the failover state machine."""
    priority: list[str] = field(default_factory=lambda: ["USB", "COHN"])
    frame_timeout_s: float = DEFAULT_FRAME_TIMEOUT_S
    backoff_base_s: float = BACKOFF_BASE_S
    backoff_cap_s: float = BACKOFF_CAP_S
    network_poll_interval_s: float = NETWORK_POLL_INTERVAL_S


class TransportFailover:
    """Auto-failover state machine for transport hot-switching.

    Monitors the active transport and fails over to the next in the
    priority chain when failures are confirmed. Freeze-frame covers
    the gap during hot-switch.

    Usage:
        failover = TransportFailover(event_bus, config)
        failover.register_transport("USB", usb_transport)
        failover.register_transport("COHN", cohn_transport)
        await failover.start()

        # On shutdown:
        await failover.stop()

    Args:
        event_bus: SSE event bus for publishing failover events.
        config: Failover configuration (priority, timeouts, backoff).
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[FailoverConfig] = None,
    ) -> None:
        self._bus = event_bus
        self._config = config or FailoverConfig()

        # Registered transports by name
        self._transports: dict[str, Transport] = {}

        # Current active transport name (or empty)
        self._active_name: str = ""

        # State machine
        self._state = FailoverState.IDLE

        # Backoff state
        self._backoff_attempt: int = 0
        self._last_backoff_reset: float = 0.0

        # Network change detection
        self._last_interfaces: set[str] = set()

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None
        self._network_task: Optional[asyncio.Task] = None
        self._backoff_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Timestamp of last received frame (updated externally)
        self._last_frame_time: float = 0.0

        # Freeze-frame callback (set by orchestrator to trigger pipeline freeze)
        self.on_freeze: Optional[Any] = None
        self.on_unfreeze: Optional[Any] = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_transport(self, name: str, transport: Transport) -> None:
        """Register a transport in the failover chain."""
        self._transports[name] = transport
        log.info("Registered transport '%s' for failover", name)

    def unregister_transport(self, name: str) -> None:
        """Remove a transport from the failover chain."""
        self._transports.pop(name, None)
        log.info("Unregistered transport '%s' from failover", name)

    # ------------------------------------------------------------------
    # Frame heartbeat
    # ------------------------------------------------------------------

    def notify_frame_received(self) -> None:
        """Called by the pipeline when a decoded frame arrives.

        Resets the frame timeout detector.
        """
        self._last_frame_time = time.monotonic()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, active_name: str = "") -> None:
        """Start failover monitoring.

        Args:
            active_name: Name of the currently active transport.
                         If empty, uses the first in priority list.
        """
        if self._running:
            return

        self._running = True
        self._active_name = active_name or (
            self._config.priority[0] if self._config.priority else ""
        )
        self._last_frame_time = time.monotonic()
        self._backoff_attempt = 0
        self._state = FailoverState.MONITORING

        self._monitor_task = asyncio.create_task(
            self._monitor_loop(),
            name="failover-monitor",
        )
        self._network_task = asyncio.create_task(
            self._network_poll_loop(),
            name="failover-network-poll",
        )

        log.info(
            "Failover started (active=%s, priority=%s)",
            self._active_name,
            self._config.priority,
        )

    async def stop(self) -> None:
        """Stop failover monitoring."""
        self._running = False
        self._state = FailoverState.IDLE

        for task in (self._monitor_task, self._network_task, self._backoff_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._monitor_task = None
        self._network_task = None
        self._backoff_task = None
        log.info("Failover stopped")

    @property
    def state(self) -> FailoverState:
        return self._state

    @property
    def active_transport_name(self) -> str:
        return self._active_name

    @property
    def active_transport(self) -> Optional[Transport]:
        return self._transports.get(self._active_name)

    # ------------------------------------------------------------------
    # Monitor loop
    # ------------------------------------------------------------------

    async def _monitor_loop(self) -> None:
        """Periodically check transport health and trigger failover."""
        while self._running:
            try:
                await asyncio.sleep(1.0)

                if self._state != FailoverState.MONITORING:
                    continue

                transport = self.active_transport
                if transport is None:
                    continue

                # Check frame timeout
                elapsed = time.monotonic() - self._last_frame_time
                if elapsed < self._config.frame_timeout_s:
                    continue

                # Frame timeout hit -- confirm with health check
                log.warning(
                    "Frame timeout (%.1fs > %.1fs) on '%s', confirming...",
                    elapsed, self._config.frame_timeout_s, self._active_name,
                )
                self._state = FailoverState.DETECTING

                healthy = False
                try:
                    healthy = await asyncio.wait_for(
                        transport.health_check(),
                        timeout=5.0,
                    )
                except (asyncio.TimeoutError, Exception) as e:
                    log.warning("Health check failed: %s", e)

                if healthy:
                    # False alarm -- transport responded, reset
                    log.info("Health check passed, resuming monitoring")
                    self._state = FailoverState.MONITORING
                    self._last_frame_time = time.monotonic()
                    continue

                # Confirmed failure -- trigger failover
                log.warning(
                    "Transport '%s' confirmed dead, triggering failover",
                    self._active_name,
                )
                await self._execute_failover()

            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("Error in failover monitor loop")
                await asyncio.sleep(2.0)

    # ------------------------------------------------------------------
    # Failover execution
    # ------------------------------------------------------------------

    async def _execute_failover(self) -> None:
        """Execute the failover to the next transport in the chain."""
        self._state = FailoverState.SWITCHING
        old_name = self._active_name

        # Trigger freeze-frame to cover the gap
        if self.on_freeze is not None:
            try:
                self.on_freeze()
            except Exception:
                log.debug("on_freeze callback error", exc_info=True)

        self._bus.publish(error_event(
            source="failover",
            message=f"Transport '{old_name}' failed, switching...",
            recoverable=True,
        ))

        # Stop stream on old transport
        old_transport = self._transports.get(old_name)
        if old_transport is not None:
            try:
                await asyncio.wait_for(old_transport.stop_stream(), timeout=5.0)
            except Exception as e:
                log.warning("Error stopping stream on '%s': %s", old_name, e)

        # Find next transport in priority list
        next_name = self._find_next_transport(old_name)

        if next_name is None:
            # All transports exhausted
            self._state = FailoverState.EXHAUSTED
            self._bus.publish(error_event(
                source="failover",
                message="All transports exhausted, staying on freeze-frame",
                recoverable=False,
            ))
            log.error("All transports exhausted")
            # Start backoff retry loop — store task so stop() can cancel it
            self._backoff_task = asyncio.ensure_future(self._backoff_retry_loop())
            return

        # Try to start stream on next transport
        success = await self._try_start_transport(next_name)
        if success:
            self._active_name = next_name
            self._state = FailoverState.MONITORING
            self._last_frame_time = time.monotonic()
            self._backoff_attempt = 0

            self._bus.publish(transport_event(
                old_transport=old_name,
                new_transport=next_name,
                reason="failover",
            ))

            # Unfreeze
            if self.on_unfreeze is not None:
                try:
                    self.on_unfreeze()
                except Exception:
                    log.debug("on_unfreeze callback error", exc_info=True)

            log.info("Failover complete: %s -> %s", old_name, next_name)
        else:
            # Next transport also failed, recurse
            self._active_name = next_name
            await self._execute_failover()

    def _find_next_transport(self, current_name: str) -> Optional[str]:
        """Find the next transport in the priority list after current."""
        priority = self._config.priority
        if current_name not in priority:
            return priority[0] if priority else None

        idx = priority.index(current_name)
        # Try each transport after current in priority order
        for i in range(1, len(priority)):
            candidate = priority[(idx + i) % len(priority)]
            if candidate != current_name and candidate in self._transports:
                return candidate

        return None

    async def _try_start_transport(self, name: str) -> bool:
        """Attempt to discover, connect, and start stream on a transport."""
        transport = self._transports.get(name)
        if transport is None:
            return False

        log.info("Attempting failover to '%s'", name)

        try:
            discovered = await asyncio.wait_for(
                transport.discover(), timeout=10.0,
            )
            if not discovered:
                log.warning("Failover to '%s': discovery failed", name)
                return False

            connected = await asyncio.wait_for(
                transport.connect(), timeout=15.0,
            )
            if not connected:
                log.warning("Failover to '%s': connect failed", name)
                return False

            stream_info = await asyncio.wait_for(
                transport.start_stream(), timeout=20.0,
            )
            if stream_info is None:
                log.warning("Failover to '%s': start_stream failed", name)
                return False

            log.info("Failover to '%s' succeeded", name)
            return True

        except (asyncio.TimeoutError, Exception) as e:
            log.warning("Failover to '%s' failed: %s", name, e)
            return False

    # ------------------------------------------------------------------
    # Backoff retry
    # ------------------------------------------------------------------

    async def _backoff_retry_loop(self) -> None:
        """Retry all transports with exponential backoff."""
        while self._running and self._state == FailoverState.EXHAUSTED:
            delay = min(
                self._config.backoff_base_s * (2 ** self._backoff_attempt),
                self._config.backoff_cap_s,
            )
            log.info(
                "Backoff retry in %.1fs (attempt %d)",
                delay, self._backoff_attempt + 1,
            )
            await asyncio.sleep(delay)

            if not self._running:
                break

            # Try each transport in priority order
            for name in self._config.priority:
                if name not in self._transports:
                    continue

                success = await self._try_start_transport(name)
                if success:
                    old_name = self._active_name
                    self._active_name = name
                    self._state = FailoverState.MONITORING
                    self._last_frame_time = time.monotonic()
                    self._backoff_attempt = 0

                    self._bus.publish(transport_event(
                        old_transport=old_name,
                        new_transport=name,
                        reason="backoff_recovery",
                    ))

                    if self.on_unfreeze is not None:
                        try:
                            self.on_unfreeze()
                        except Exception:
                            pass

                    log.info("Backoff recovery: connected to '%s'", name)
                    return

            self._backoff_attempt += 1

    # ------------------------------------------------------------------
    # Network change detection
    # ------------------------------------------------------------------

    async def _network_poll_loop(self) -> None:
        """Poll network interfaces for changes (resets backoff)."""
        while self._running:
            try:
                await asyncio.sleep(self._config.network_poll_interval_s)

                current = await self._get_interface_names()
                if self._last_interfaces and current != self._last_interfaces:
                    added = current - self._last_interfaces
                    removed = self._last_interfaces - current
                    log.info(
                        "Network change detected (added=%s, removed=%s)",
                        added or "none", removed or "none",
                    )
                    # Reset backoff on network change
                    self._backoff_attempt = 0
                    self._last_backoff_reset = time.monotonic()

                self._last_interfaces = current

            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("Network poll error", exc_info=True)

    @staticmethod
    async def _get_interface_names() -> set[str]:
        """Get current network interface names via psutil (if available)."""
        try:
            import psutil
            addrs = psutil.net_if_addrs()
            return set(addrs.keys())
        except ImportError:
            return set()
        except Exception:
            return set()
