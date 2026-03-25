"""
transport_manager.py — Event-driven transport manager with USB→COHN failover.

Listens for USB transport disconnect events and automatically attempts
COHN transport connection as failover. Integrates with the EventBus for
SSE status updates and provides freeze-frame callbacks during transitions.

Architecture:
    TransportManager
      ├── USBTransport (primary)
      │     └── disconnect_listener → triggers COHN failover
      ├── COHNTransport (fallback)
      │     └── disconnect_listener → triggers USB recovery / exhausted
      ├── EventBus (SSE status publishing)
      └── Freeze-frame callbacks (cover the gap during hot-switch)

Thread model:
    All methods run on the asyncio event loop. Disconnect listeners are
    synchronous callbacks invoked on the event loop thread — they schedule
    async failover via asyncio.ensure_future.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional

from gomaxwebcam.events import (
    EventBus,
    EventType,
    Event,
    transport_event,
    connection_event,
    error_event,
    usb_redetected_event,
)
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo

log = logging.getLogger("gomaxwebcam.transport_manager")

# Timeouts for failover operations
FAILOVER_DISCOVER_TIMEOUT_S = 10.0
FAILOVER_CONNECT_TIMEOUT_S = 15.0
FAILOVER_STREAM_TIMEOUT_S = 20.0

# Backoff for recovery retries when all transports exhausted
RECOVERY_BACKOFF_BASE_S = 2.0
RECOVERY_BACKOFF_CAP_S = 60.0

# USB re-detection polling (fail-back from COHN)
USB_POLL_INTERVAL_S = 5.0       # How often to check USB bus when COHN is active
USB_POLL_INITIAL_DELAY_S = 3.0  # Wait before first poll after failover settles


class ManagerState(Enum):
    """State of the transport manager."""
    IDLE = auto()           # Not started, no active transport
    ACTIVE = auto()         # A transport is connected and streaming
    FAILING_OVER = auto()   # Switching from one transport to another
    RECOVERING = auto()     # All transports failed, retrying with backoff
    STOPPED = auto()        # Manager shut down


@dataclass
class TransportManagerConfig:
    """Configuration for the transport manager."""
    # Ordered list of transport names to try (first = highest priority)
    priority: list[str] = field(default_factory=lambda: ["USB", "COHN"])
    # Timeouts for failover operations
    discover_timeout_s: float = FAILOVER_DISCOVER_TIMEOUT_S
    connect_timeout_s: float = FAILOVER_CONNECT_TIMEOUT_S
    stream_timeout_s: float = FAILOVER_STREAM_TIMEOUT_S
    # Backoff for recovery retries
    backoff_base_s: float = RECOVERY_BACKOFF_BASE_S
    backoff_cap_s: float = RECOVERY_BACKOFF_CAP_S
    # Enable automatic recovery retries when all transports exhausted
    auto_recovery: bool = True
    # USB re-detection polling interval when COHN is active
    usb_poll_interval_s: float = USB_POLL_INTERVAL_S


class TransportManager:
    """Event-driven transport manager with automatic USB→COHN failover.

    Registers as a disconnect listener on all transports. When the active
    transport reports an unexpected disconnection, the manager:

      1. Triggers freeze-frame (via on_freeze callback)
      2. Cleans up the failed transport
      3. Attempts to connect the next transport in priority order
      4. On success: unfreezes and publishes transport change event
      5. On failure: tries remaining transports, then enters recovery mode

    Usage::

        bus = EventBus()
        manager = TransportManager(bus)

        manager.register_transport("USB", usb_transport)
        manager.register_transport("COHN", cohn_transport)

        # Start with the highest-priority transport
        await manager.start()

        # On shutdown:
        await manager.stop()

    Args:
        event_bus: SSE event bus for publishing failover events.
        config: Optional manager configuration.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[TransportManagerConfig] = None,
    ) -> None:
        self._bus = event_bus
        self._config = config or TransportManagerConfig()

        # Registered transports by name
        self._transports: dict[str, Transport] = {}

        # Current active transport name
        self._active_name: str = ""

        # State machine
        self._state = ManagerState.IDLE

        # Backoff state for recovery
        self._backoff_attempt: int = 0

        # Background recovery task
        self._recovery_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Failover lock — prevents concurrent failover attempts
        self._failover_lock = asyncio.Lock()

        # USB re-detection polling task (active when COHN is the active transport)
        self._usb_poll_task: Optional[asyncio.Task] = None

        # Freeze-frame callbacks (set by orchestrator or pipeline)
        self.on_freeze: Optional[Callable[[], None]] = None
        self.on_unfreeze: Optional[Callable[[], None]] = None

        # Transport-switch callback (set by orchestrator to rewire pipeline)
        self.on_transport_switch: Optional[Callable[[str, Transport], Any]] = None

        # Stats
        self._failover_count: int = 0
        self._last_failover_time: float = 0.0
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_transport(self, name: str, transport: Transport) -> None:
        """Register a transport and wire up the disconnect listener.

        The manager automatically listens for unexpected disconnections
        on all registered transports.

        Args:
            name: Transport name (e.g. "USB", "COHN"). Must match
                  a name in the priority list to participate in failover.
            transport: Transport instance to register.
        """
        self._transports[name] = transport
        transport.add_disconnect_listener(self._on_transport_disconnect)
        log.info("Registered transport '%s' for failover management", name)

    def unregister_transport(self, name: str) -> None:
        """Remove a transport from management.

        Note: disconnect listeners cannot be removed from the Transport
        base class, but the manager ignores events from unregistered transports.
        """
        self._transports.pop(name, None)
        log.info("Unregistered transport '%s'", name)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> ManagerState:
        return self._state

    @property
    def active_transport_name(self) -> str:
        return self._active_name

    @property
    def active_transport(self) -> Optional[Transport]:
        return self._transports.get(self._active_name)

    @property
    def failover_count(self) -> int:
        return self._failover_count

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def registered_transports(self) -> dict[str, Transport]:
        """Return a copy of the registered transports dict."""
        return dict(self._transports)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, preferred: str = "") -> bool:
        """Start the transport manager.

        Attempts to connect the preferred transport (or the first in
        priority order). Returns True if a transport was successfully
        connected and streaming.

        Args:
            preferred: Name of the preferred transport to start with.
                       If empty, uses the first in the priority list.

        Returns:
            True if a transport is connected and streaming.
        """
        if self._running:
            return self._state == ManagerState.ACTIVE

        self._running = True
        self._start_time = time.monotonic()

        start_name = preferred or (
            self._config.priority[0] if self._config.priority else ""
        )

        if not start_name or start_name not in self._transports:
            log.error(
                "Cannot start: transport '%s' not registered (available: %s)",
                start_name,
                list(self._transports.keys()),
            )
            self._state = ManagerState.IDLE
            return False

        log.info("TransportManager starting with '%s'", start_name)

        # Publish CONNECTING state so tray icon turns yellow during startup
        self._bus.publish(connection_event(
            old_state="DISCONNECTED",
            new_state="CONNECTING",
            transport_name=start_name,
            detail="Initial transport startup",
        ))

        success = await self._try_start_transport(start_name)
        if success:
            self._active_name = start_name
            self._state = ManagerState.ACTIVE
            self._bus.publish(transport_event(
                old_transport="none",
                new_transport=start_name,
                reason="startup",
            ))

            # Notify orchestrator so CameraManager observes this transport
            await self._notify_transport_switch(start_name)

            # Publish STREAMING state so tray icon turns green
            self._bus.publish(connection_event(
                old_state="CONNECTING",
                new_state="STREAMING",
                transport_name=start_name,
            ))
            log.info("TransportManager active on '%s'", start_name)
            return True

        # Primary failed — try other transports
        for name in self._config.priority:
            if name == start_name or name not in self._transports:
                continue

            self._bus.publish(connection_event(
                old_state="CONNECTING",
                new_state="CONNECTING",
                transport_name=name,
                detail=f"Trying fallback transport '{name}'",
            ))

            success = await self._try_start_transport(name)
            if success:
                self._active_name = name
                self._state = ManagerState.ACTIVE
                self._bus.publish(transport_event(
                    old_transport="none",
                    new_transport=name,
                    reason="startup_fallback",
                ))

                # Notify orchestrator so CameraManager observes this transport
                await self._notify_transport_switch(name)

                # Publish STREAMING state so tray icon turns green
                self._bus.publish(connection_event(
                    old_state="CONNECTING",
                    new_state="STREAMING",
                    transport_name=name,
                ))
                log.info("TransportManager active on '%s' (fallback)", name)
                return True

        log.warning("TransportManager: no transports available at startup")
        self._state = ManagerState.IDLE
        return False

    async def stop(self) -> None:
        """Stop the transport manager and disconnect all transports."""
        old_name = self._active_name
        self._running = False
        self._state = ManagerState.STOPPED

        # Cancel USB poll task
        self._stop_usb_poll()

        # Cancel recovery task
        if self._recovery_task is not None:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass
            self._recovery_task = None

        # Disconnect active transport
        active = self.active_transport
        if active is not None:
            try:
                await active.disconnect()
            except Exception as e:
                log.warning("Error disconnecting '%s': %s", self._active_name, e)

        self._active_name = ""

        # Publish STOPPED state so tray icon turns grey
        self._bus.publish(connection_event(
            old_state="STREAMING" if old_name else "DISCONNECTED",
            new_state="STOPPED",
            transport_name="none",
        ))

        log.info(
            "TransportManager stopped (failovers=%d)",
            self._failover_count,
        )

    # ------------------------------------------------------------------
    # Disconnect listener (synchronous callback from Transport)
    # ------------------------------------------------------------------

    def _on_transport_disconnect(self, transport: Transport, reason: str) -> None:
        """Called when a transport reports an unexpected disconnection.

        This is a synchronous callback invoked by the Transport base class.
        We schedule the async failover operation on the event loop.

        Args:
            transport: The transport that disconnected.
            reason: Human-readable reason (e.g. "health_check_failed").
        """
        if not self._running:
            return

        # Find the name of the disconnected transport
        disconnected_name = ""
        for name, t in self._transports.items():
            if t is transport:
                disconnected_name = name
                break

        if not disconnected_name:
            log.debug("Disconnect event from unregistered transport, ignoring")
            return

        # Only react if the disconnected transport is the active one
        if disconnected_name != self._active_name:
            log.info(
                "Transport '%s' disconnected (reason=%s) but it's not active ('%s'), ignoring",
                disconnected_name, reason, self._active_name,
            )
            return

        log.warning(
            "Active transport '%s' disconnected (reason=%s), initiating failover",
            disconnected_name, reason,
        )

        self._bus.publish(error_event(
            source=f"transport.{disconnected_name}",
            message=f"Transport disconnected: {reason}",
            recoverable=True,
        ))

        # Schedule async failover on the event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.error("No running event loop — cannot schedule failover")
            return

        asyncio.ensure_future(
            self._handle_disconnect(disconnected_name, reason),
        )

    # ------------------------------------------------------------------
    # Failover logic
    # ------------------------------------------------------------------

    async def _handle_disconnect(self, failed_name: str, reason: str) -> None:
        """Handle a transport disconnect by failing over to the next transport.

        Uses a lock to prevent concurrent failover attempts (e.g. if
        multiple disconnect events fire in quick succession).
        """
        async with self._failover_lock:
            # Re-check state under lock
            if not self._running or self._active_name != failed_name:
                return

            if self._state == ManagerState.FAILING_OVER:
                log.debug("Failover already in progress, skipping")
                return

            self._state = ManagerState.FAILING_OVER
            self._failover_count += 1
            self._last_failover_time = time.monotonic()

            log.info(
                "Failover #%d: '%s' failed (%s), switching...",
                self._failover_count, failed_name, reason,
            )

            # Step 1: Trigger freeze-frame and publish FREEZE_FRAME state
            self._trigger_freeze()
            self._bus.publish(connection_event(
                old_state="STREAMING",
                new_state="FREEZE_FRAME",
                transport_name=failed_name,
                detail=f"Transport '{failed_name}' failed ({reason}), switching...",
            ))

            # Step 2: Clean up the failed transport
            await self._cleanup_transport(failed_name)

            # Step 3: Try next transports in priority order
            next_name = await self._try_failover_chain(failed_name)

            if next_name is not None:
                # Success — new transport is connected and streaming
                old_name = self._active_name
                self._active_name = next_name
                self._state = ManagerState.ACTIVE
                self._backoff_attempt = 0

                self._bus.publish(transport_event(
                    old_transport=old_name,
                    new_transport=next_name,
                    reason="failover",
                ))

                # Notify orchestrator of transport switch
                await self._notify_transport_switch(next_name)

                # Step 4: Unfreeze and publish STREAMING state
                self._trigger_unfreeze()
                self._bus.publish(connection_event(
                    old_state="FREEZE_FRAME",
                    new_state="STREAMING",
                    transport_name=next_name,
                    detail=f"Failover to '{next_name}' succeeded",
                ))

                log.info(
                    "Failover complete: '%s' -> '%s' (took %.1fs)",
                    old_name, next_name,
                    time.monotonic() - self._last_failover_time,
                )
            else:
                # All transports exhausted
                log.error("All transports exhausted after failover")
                self._state = ManagerState.RECOVERING

                self._bus.publish(connection_event(
                    old_state="FREEZE_FRAME",
                    new_state="DISCONNECTED",
                    transport_name="none",
                    detail="All transports exhausted",
                ))
                self._bus.publish(error_event(
                    source="transport_manager",
                    message="All transports exhausted, entering recovery mode",
                    recoverable=True,
                ))

                # Start recovery retry loop
                if self._config.auto_recovery:
                    self._start_recovery_loop()

    async def _try_failover_chain(self, failed_name: str) -> Optional[str]:
        """Try each transport after the failed one in priority order.

        Returns the name of the transport that was successfully started,
        or None if all transports failed.
        """
        priority = self._config.priority

        # Build ordered list: transports after the failed one, wrapping around
        # but excluding the failed transport itself
        candidates: list[str] = []
        if failed_name in priority:
            idx = priority.index(failed_name)
            for i in range(1, len(priority)):
                candidate = priority[(idx + i) % len(priority)]
                if candidate != failed_name and candidate in self._transports:
                    candidates.append(candidate)
        else:
            # Failed transport not in priority — try all registered
            candidates = [
                n for n in priority
                if n != failed_name and n in self._transports
            ]

        for name in candidates:
            log.info("Attempting failover to '%s'", name)
            success = await self._try_start_transport(name)
            if success:
                return name
            log.warning("Failover to '%s' failed, trying next", name)

        return None

    async def _try_start_transport(self, name: str) -> bool:
        """Attempt to discover, connect, and start stream on a transport.

        Returns True if the transport is now streaming.
        """
        transport = self._transports.get(name)
        if transport is None:
            return False

        cfg = self._config

        try:
            # Discover
            discovered = await asyncio.wait_for(
                transport.discover(),
                timeout=cfg.discover_timeout_s,
            )
            if not discovered:
                log.warning("Transport '%s' discovery failed", name)
                return False

            # Connect
            connected = await asyncio.wait_for(
                transport.connect(),
                timeout=cfg.connect_timeout_s,
            )
            if not connected:
                log.warning("Transport '%s' connect failed", name)
                return False

            # Start stream
            stream_info = await asyncio.wait_for(
                transport.start_stream(),
                timeout=cfg.stream_timeout_s,
            )
            if stream_info is None:
                log.warning("Transport '%s' start_stream failed", name)
                await transport.disconnect()
                return False

            log.info(
                "Transport '%s' started successfully (stream=%s:%d)",
                name, stream_info.host, stream_info.port,
            )
            return True

        except asyncio.TimeoutError:
            log.warning("Transport '%s' timed out during startup", name)
            try:
                await transport.disconnect()
            except Exception:
                pass
            return False
        except Exception as e:
            log.warning("Transport '%s' startup error: %s", name, e)
            try:
                await transport.disconnect()
            except Exception:
                pass
            return False

    async def _cleanup_transport(self, name: str) -> None:
        """Clean up a failed transport (stop stream, disconnect)."""
        transport = self._transports.get(name)
        if transport is None:
            return

        try:
            if transport.is_streaming:
                await asyncio.wait_for(transport.stop_stream(), timeout=5.0)
        except Exception as e:
            log.debug("Error stopping stream on '%s': %s", name, e)

        try:
            await asyncio.wait_for(transport.disconnect(), timeout=5.0)
        except Exception as e:
            log.debug("Error disconnecting '%s': %s", name, e)

    # ------------------------------------------------------------------
    # Recovery loop (backoff retries when all transports exhausted)
    # ------------------------------------------------------------------

    def _start_recovery_loop(self) -> None:
        """Start the background recovery retry loop."""
        if self._recovery_task is not None and not self._recovery_task.done():
            return

        self._recovery_task = asyncio.create_task(
            self._recovery_loop(),
            name="transport-manager-recovery",
        )

    async def _recovery_loop(self) -> None:
        """Retry all transports with exponential backoff."""
        while self._running and self._state == ManagerState.RECOVERING:
            delay = min(
                self._config.backoff_base_s * (2 ** self._backoff_attempt),
                self._config.backoff_cap_s,
            )
            log.info(
                "Recovery retry in %.1fs (attempt %d)",
                delay, self._backoff_attempt + 1,
            )
            await asyncio.sleep(delay)

            if not self._running or self._state != ManagerState.RECOVERING:
                break

            # Publish CONNECTING state so tray shows yellow during recovery
            self._bus.publish(connection_event(
                old_state="DISCONNECTED",
                new_state="CONNECTING",
                transport_name="",
                detail=f"Recovery attempt {self._backoff_attempt + 1}",
            ))

            # Try each transport in priority order
            for name in self._config.priority:
                if name not in self._transports:
                    continue

                success = await self._try_start_transport(name)
                if success:
                    self._active_name = name
                    self._state = ManagerState.ACTIVE
                    self._backoff_attempt = 0

                    self._bus.publish(transport_event(
                        old_transport="none",
                        new_transport=name,
                        reason="recovery",
                    ))

                    # Notify orchestrator
                    await self._notify_transport_switch(name)

                    self._trigger_unfreeze()

                    # Publish STREAMING state so tray turns green
                    self._bus.publish(connection_event(
                        old_state="CONNECTING",
                        new_state="STREAMING",
                        transport_name=name,
                        detail="Recovery succeeded",
                    ))

                    log.info("Recovery succeeded: connected to '%s'", name)
                    return

            # All transports failed this round — back to DISCONNECTED
            self._bus.publish(connection_event(
                old_state="CONNECTING",
                new_state="DISCONNECTED",
                transport_name="",
                detail=f"Recovery attempt {self._backoff_attempt + 1} failed",
            ))

            self._backoff_attempt += 1

    # ------------------------------------------------------------------
    # Transport switch notification helper
    # ------------------------------------------------------------------

    async def _notify_transport_switch(self, name: str) -> None:
        """Notify the orchestrator of a transport switch (handles sync + async callbacks).

        Also manages the USB re-detection polling loop:
          - If switching TO a non-USB transport (e.g. COHN), starts USB polling
            so we can fail-back when the USB device reappears.
          - If switching TO USB, stops polling (USB is already active).
        """
        # Manage USB poll lifecycle based on new active transport
        if name == "USB":
            self._stop_usb_poll()
        else:
            self._start_usb_poll()

        if self.on_transport_switch is not None:
            try:
                result = self.on_transport_switch(name, self._transports[name])
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                log.exception("Error in on_transport_switch callback")

    # ------------------------------------------------------------------
    # Freeze-frame helpers
    # ------------------------------------------------------------------

    def _trigger_freeze(self) -> None:
        """Trigger freeze-frame mode via callback."""
        if self.on_freeze is not None:
            try:
                self.on_freeze()
                log.debug("Freeze-frame triggered")
            except Exception:
                log.exception("Error in on_freeze callback")

    def _trigger_unfreeze(self) -> None:
        """Release freeze-frame mode via callback."""
        if self.on_unfreeze is not None:
            try:
                self.on_unfreeze()
                log.debug("Freeze-frame released")
            except Exception:
                log.exception("Error in on_unfreeze callback")

    # ------------------------------------------------------------------
    # Manual failover
    # ------------------------------------------------------------------

    async def force_failover(self, target_name: str = "") -> bool:
        """Manually trigger failover to a specific transport.

        If target_name is empty, fails over to the next in priority order.

        Returns True if the failover succeeded.
        """
        if not self._running:
            return False

        old_name = self._active_name

        if target_name and target_name not in self._transports:
            log.error("Cannot failover to '%s': not registered", target_name)
            return False

        async with self._failover_lock:
            self._state = ManagerState.FAILING_OVER
            self._trigger_freeze()
            self._bus.publish(connection_event(
                old_state="STREAMING",
                new_state="FREEZE_FRAME",
                transport_name=old_name,
                detail="Manual failover initiated",
            ))

            # Clean up current transport
            if old_name:
                await self._cleanup_transport(old_name)

            if target_name:
                success = await self._try_start_transport(target_name)
                if success:
                    self._active_name = target_name
                    self._state = ManagerState.ACTIVE
                    self._trigger_unfreeze()
                    self._bus.publish(transport_event(
                        old_transport=old_name,
                        new_transport=target_name,
                        reason="manual",
                    ))
                    await self._notify_transport_switch(target_name)
                    self._bus.publish(connection_event(
                        old_state="FREEZE_FRAME",
                        new_state="STREAMING",
                        transport_name=target_name,
                    ))
                    return True
                self._state = ManagerState.RECOVERING
                self._bus.publish(connection_event(
                    old_state="FREEZE_FRAME",
                    new_state="DISCONNECTED",
                    transport_name="none",
                    detail=f"Manual failover to '{target_name}' failed",
                ))
                return False
            else:
                # Try next in chain
                next_name = await self._try_failover_chain(old_name)
                if next_name:
                    self._active_name = next_name
                    self._state = ManagerState.ACTIVE
                    self._trigger_unfreeze()
                    self._bus.publish(transport_event(
                        old_transport=old_name,
                        new_transport=next_name,
                        reason="manual",
                    ))
                    await self._notify_transport_switch(next_name)
                    self._bus.publish(connection_event(
                        old_state="FREEZE_FRAME",
                        new_state="STREAMING",
                        transport_name=next_name,
                    ))
                    return True
                self._state = ManagerState.RECOVERING
                self._bus.publish(connection_event(
                    old_state="FREEZE_FRAME",
                    new_state="DISCONNECTED",
                    transport_name="none",
                    detail="All transports exhausted during manual failover",
                ))
                return False

    # ------------------------------------------------------------------
    # USB re-detection polling (fail-back from COHN to USB)
    # ------------------------------------------------------------------

    def _start_usb_poll(self) -> None:
        """Start background USB presence polling.

        Called when COHN becomes the active transport (after failover from
        USB).  Polls the USB bus periodically via ``find_gopro_device()``
        and, when a GoPro is detected, publishes a ``USB_REDETECTED`` event
        on the EventBus and triggers an automatic fail-back to USB.

        If the poll task is already running, this is a no-op.
        """
        if self._usb_poll_task is not None and not self._usb_poll_task.done():
            return  # Already polling

        # Only poll if USB is a registered transport
        if "USB" not in self._transports:
            return

        self._usb_poll_task = asyncio.create_task(
            self._usb_poll_loop(),
            name="usb-redetect-poll",
        )
        log.info("USB re-detection polling started (interval=%.1fs)", self._config.usb_poll_interval_s)

    def _stop_usb_poll(self) -> None:
        """Cancel the USB presence polling task."""
        if self._usb_poll_task is not None and not self._usb_poll_task.done():
            self._usb_poll_task.cancel()
            log.debug("USB re-detection polling stopped")
        self._usb_poll_task = None

    async def _usb_poll_loop(self) -> None:
        """Background loop that checks USB bus for GoPro device presence.

        Runs while COHN is the active transport.  When a GoPro is found:
          1. Publishes ``USB_REDETECTED`` event on the EventBus
          2. Triggers ``force_failover("USB")`` to switch back

        The loop exits when:
          - USB is re-detected and failback initiated
          - COHN is no longer the active transport
          - Manager is stopped
          - Task is cancelled
        """
        try:
            # Initial delay — let failover settle before first poll
            await asyncio.sleep(USB_POLL_INITIAL_DELAY_S)

            while self._running and self._active_name != "USB":
                try:
                    device = await self._check_usb_presence()
                except Exception as exc:
                    log.debug("USB poll error (non-fatal): %s", exc)
                    device = None

                if device is not None:
                    serial = getattr(device, "serial_number", "") or ""
                    ip = getattr(device, "camera_ip", "") or ""
                    log.info(
                        "USB device re-detected! serial=%s ip=%s — initiating fail-back",
                        serial or "unknown", ip or "unknown",
                    )

                    # Publish event so dashboard/SSE subscribers are notified
                    self._bus.publish(usb_redetected_event(
                        camera_serial=serial,
                        camera_ip=ip,
                    ))

                    # Trigger fail-back to USB (runs under failover lock)
                    success = await self.force_failover("USB")
                    if success:
                        log.info("USB fail-back succeeded")
                    else:
                        log.warning("USB fail-back failed — will retry on next poll")
                        # Don't exit loop; keep polling so we retry
                        await asyncio.sleep(self._config.usb_poll_interval_s)
                        continue

                    # Failback succeeded — stop polling
                    return

                await asyncio.sleep(self._config.usb_poll_interval_s)

        except asyncio.CancelledError:
            log.debug("USB poll loop cancelled")
            raise
        except Exception:
            log.exception("USB poll loop unexpected error")

    async def _check_usb_presence(self):
        """Check if a GoPro USB device is present on the bus.

        Returns a GoProDeviceInfo if found, None otherwise.
        Uses the discovery module's find_gopro_device() which runs
        WMI/pnputil via asyncio.to_thread().
        """
        from gomaxwebcam.discovery import find_gopro_device
        return await find_gopro_device()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current transport manager status as a dict."""
        active = self.active_transport
        return {
            "state": self._state.name,
            "active_transport": self._active_name,
            "active_transport_state": (
                active.state.name if active else "NONE"
            ),
            "registered_transports": list(self._transports.keys()),
            "priority": self._config.priority,
            "failover_count": self._failover_count,
            "last_failover_time": self._last_failover_time,
            "backoff_attempt": self._backoff_attempt,
            "usb_poll_active": (
                self._usb_poll_task is not None
                and not self._usb_poll_task.done()
            ),
            "uptime_s": (
                round(time.monotonic() - self._start_time, 1)
                if self._start_time else 0.0
            ),
        }
