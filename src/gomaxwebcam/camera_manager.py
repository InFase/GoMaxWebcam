"""
camera_manager.py — Camera lifecycle orchestrator for GoMaxWebcam v2.

The CameraManager owns the transport and pipeline lifecycle, and publishes
all status updates (battery polling, transport changes, connection events)
to the SSE event bus for live dashboard consumption.

Responsibilities:
  1. Hold the active transport and pipeline references
  2. Listen for transport state changes and publish connection events
  3. Listen for pipeline state changes and publish pipeline events
  4. Run periodic battery polling and publish battery events
  5. Publish full status snapshots at regular intervals
  6. Handle transport failover and publish transport change events
  7. Wire everything to the CameraStatusTracker for SSE broadcasting

Architecture:
    CameraManager
      ├── Transport (USB / COHN / WiFi AP)
      │     └── state listener → publishes connection events to EventBus
      ├── FramePipeline
      │     └── on_state_change → publishes pipeline events to EventBus
      ├── Battery poller (asyncio task, periodic)
      │     └── publishes battery events to EventBus
      ├── Status poller (asyncio task, periodic)
      │     └── publishes full status snapshots to EventBus
      └── CameraStatusTracker
            └── aggregates all above for SSE subscribers

Thread model:
    All CameraManager methods run on the asyncio event loop thread.
    The transport state listener callbacks are synchronous (called from
    the transport on the same thread), and we use them to publish events.

Usage:
    bus = EventBus()
    manager = CameraManager(bus)

    # Set transport
    manager.set_transport(usb_transport)
    manager.set_pipeline(pipeline)

    # Start status publishing
    await manager.start()

    # SSE endpoint uses bus.subscribe()

    # Cleanup
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from gomaxwebcam.events import (
    EventBus,
    Event,
    EventType,
    connection_event,
    transport_event,
    battery_event,
    pipeline_event,
    error_event,
    status_event,
)
from gomaxwebcam.transport.base import Transport, TransportState, TransportStats
from gomaxwebcam.pipeline.frame_pipeline import FramePipeline, PipelineState, PipelineStats

log = logging.getLogger("gomaxwebcam.camera_manager")

# Polling intervals
_BATTERY_POLL_INTERVAL_S = 30.0     # Battery changes slowly
_STATUS_POLL_INTERVAL_S = 1.0       # Full status snapshot every second


@dataclass
class CameraManagerConfig:
    """Configuration for the camera manager."""
    battery_poll_interval: float = _BATTERY_POLL_INTERVAL_S
    status_poll_interval: float = _STATUS_POLL_INTERVAL_S
    battery_poll_enabled: bool = True


class CameraManager:
    """Orchestrates transport + pipeline lifecycle and publishes to SSE event bus.

    The camera manager is the single integration point that wires:
      - Transport state changes → SSE connection events
      - Pipeline state changes → SSE pipeline events
      - Battery polling → SSE battery events
      - Full status snapshots → SSE status events
      - Transport failover → SSE transport events

    It does NOT own the transport or pipeline instances — those are set
    externally by the orchestrator. It observes them and publishes events.
    """

    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[CameraManagerConfig] = None,
    ) -> None:
        self._bus = event_bus
        self._config = config or CameraManagerConfig()

        # Observed components (set externally)
        self._transport: Optional[Transport] = None
        self._pipeline: Optional[FramePipeline] = None

        # Active transport name for change detection
        self._active_transport_name: str = "none"

        # Battery state
        self._battery_level: int = -1
        self._battery_charging: bool = False

        # Background tasks
        self._battery_task: Optional[asyncio.Task] = None
        self._status_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Stats
        self._start_time: Optional[float] = None
        self._events_published: int = 0

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def set_transport(self, transport: Optional[Transport]) -> None:
        """Set or change the active transport.

        Publishes a transport change event if the transport type changes.
        Wires up the transport state listener for connection events.
        """
        old_name = self._active_transport_name

        # Remove listener from old transport
        if self._transport is not None:
            # Transport doesn't have remove_listener, but the list is accessible
            try:
                self._transport._state_listeners.remove(self._on_transport_state_change)
            except (ValueError, AttributeError):
                pass

        self._transport = transport

        if transport is not None:
            new_name = transport.name
            self._active_transport_name = new_name

            # Wire up state listener
            transport.add_state_listener(self._on_transport_state_change)

            # Publish transport change event if type changed
            if old_name != new_name:
                self._publish(transport_event(
                    old_transport=old_name,
                    new_transport=new_name,
                    reason="manual" if old_name == "none" else "failover",
                ))
                log.info("Transport set: %s -> %s", old_name, new_name)

            # Publish the current connection state so tray/dashboard reflect it
            # immediately (the state listener only fires on *future* changes)
            current_state = transport.state.name
            self._publish(connection_event(
                old_state="DISCONNECTED" if old_name == "none" else "CONNECTING",
                new_state=current_state,
                transport_name=new_name,
                detail=f"Transport '{new_name}' assigned",
            ))
        else:
            if old_name != "none":
                self._active_transport_name = "none"
                self._publish(transport_event(
                    old_transport=old_name,
                    new_transport="none",
                    reason="disconnected",
                ))
                log.info("Transport cleared (was %s)", old_name)

    def set_pipeline(self, pipeline: Optional[FramePipeline]) -> None:
        """Set or change the active pipeline.

        Wires up pipeline state change and freeze/unfreeze callbacks.
        """
        # Remove callbacks from old pipeline
        if self._pipeline is not None:
            self._pipeline.on_state_change = None
            self._pipeline.on_freeze = None
            self._pipeline.on_unfreeze = None

        self._pipeline = pipeline

        if pipeline is not None:
            pipeline.on_state_change = self._on_pipeline_state_change
            pipeline.on_freeze = self._on_pipeline_freeze
            pipeline.on_unfreeze = self._on_pipeline_unfreeze
            log.info("Pipeline set")

    def set_battery_level(self, level: int, charging: bool = False) -> None:
        """Update battery level from an external source (e.g. BLE notification).

        Publishes a battery event if the level changed.
        """
        if level != self._battery_level or charging != self._battery_charging:
            self._battery_level = level
            self._battery_charging = charging
            self._publish(battery_event(level=level, charging=charging))
            log.debug("Battery updated: %d%% (charging=%s)", level, charging)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start periodic polling tasks (battery, status snapshots)."""
        if self._running:
            return

        self._running = True
        self._start_time = time.monotonic()
        self._bus.set_loop(asyncio.get_running_loop())

        # Start battery polling task
        if self._config.battery_poll_enabled:
            self._battery_task = asyncio.create_task(
                self._battery_poll_loop(),
                name="camera-manager-battery-poll",
            )

        # Start status snapshot task
        self._status_task = asyncio.create_task(
            self._status_poll_loop(),
            name="camera-manager-status-poll",
        )

        log.info(
            "Camera manager started (battery_poll=%.0fs, status_poll=%.1fs)",
            self._config.battery_poll_interval,
            self._config.status_poll_interval,
        )

    async def stop(self) -> None:
        """Stop all polling tasks and clean up."""
        self._running = False

        for task in (self._battery_task, self._status_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        self._battery_task = None
        self._status_task = None

        log.info(
            "Camera manager stopped (published %d events)",
            self._events_published,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def events_published(self) -> int:
        return self._events_published

    # ------------------------------------------------------------------
    # Transport state listener (synchronous callback)
    # ------------------------------------------------------------------

    def _on_transport_state_change(
        self,
        old_state: TransportState,
        new_state: TransportState,
    ) -> None:
        """Called by the Transport when its state changes.

        Publishes a connection event to the SSE bus.
        """
        transport_name = self._active_transport_name

        detail = ""
        if new_state == TransportState.ERROR:
            # Include last error from transport stats
            if self._transport is not None:
                detail = self._transport.stats.last_error or ""

        self._publish(connection_event(
            old_state=old_state.name,
            new_state=new_state.name,
            transport_name=transport_name,
            detail=detail,
        ))

        # Also publish error event for ERROR state
        if new_state == TransportState.ERROR and detail:
            self._publish(error_event(
                source=f"transport.{transport_name}",
                message=detail,
                recoverable=True,
            ))

        log.debug(
            "Transport state event: %s -> %s (transport=%s)",
            old_state.name, new_state.name, transport_name,
        )

    # ------------------------------------------------------------------
    # Pipeline state callbacks
    # ------------------------------------------------------------------

    def _on_pipeline_state_change(self, new_state: PipelineState) -> None:
        """Called by the FramePipeline when its state changes."""
        # We don't have the old state here, so we track it
        old_name = getattr(self, "_last_pipeline_state", "UNKNOWN")
        self._last_pipeline_state = new_state.name

        self._publish(pipeline_event(
            old_state=old_name,
            new_state=new_state.name,
        ))

        log.debug("Pipeline state event: %s -> %s", old_name, new_state.name)

    def _on_pipeline_freeze(self) -> None:
        """Called when the pipeline enters freeze-frame mode."""
        self._publish(pipeline_event(
            old_state="STREAMING",
            new_state="FREEZE_FRAME",
            detail="Stream lost, showing last good frame",
        ))
        log.info("Pipeline freeze event published")

    def _on_pipeline_unfreeze(self) -> None:
        """Called when the pipeline exits freeze-frame mode."""
        self._publish(pipeline_event(
            old_state="FREEZE_FRAME",
            new_state="STREAMING",
            detail="Live frames resumed",
        ))
        log.info("Pipeline unfreeze event published")

    # ------------------------------------------------------------------
    # Battery polling
    # ------------------------------------------------------------------

    async def _battery_poll_loop(self) -> None:
        """Periodically poll battery level from the camera.

        Uses the transport's SDK handle or HTTP API to query battery.
        Publishes battery events when the level changes.
        """
        while self._running:
            try:
                await asyncio.sleep(self._config.battery_poll_interval)

                level = await self._query_battery()
                if level is not None and level != self._battery_level:
                    old = self._battery_level
                    self._battery_level = level
                    self._publish(battery_event(level=level, charging=self._battery_charging))
                    log.debug("Battery poll: %d%% (was %d%%)", level, old)

            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("Battery poll error", exc_info=True)

    async def _query_battery(self) -> Optional[int]:
        """Query battery level from the active transport.

        Returns battery percentage (0-100) or None if unavailable.
        Tries USB SDK status query, then HTTP status endpoint.
        """
        if self._transport is None or not self._transport.is_connected:
            return None

        try:
            # For USB transport: try open-gopro SDK status query
            gopro = getattr(self._transport, "_gopro", None)
            if gopro is not None:
                # Try the SDK's status accessor for battery level
                # open-gopro uses StatusId.INT_BATT_PER for battery percentage
                try:
                    from open_gopro import constants
                    batt = await asyncio.wait_for(
                        gopro.ble_status.int_batt_per.get_value(),
                        timeout=5.0,
                    )
                    if isinstance(batt, int):
                        return batt
                except (ImportError, AttributeError, asyncio.TimeoutError):
                    pass

                # Fallback: try HTTP status endpoint
                try:
                    resp = await asyncio.wait_for(
                        gopro.http_command.get_camera_state(),
                        timeout=5.0,
                    )
                    if resp.ok and hasattr(resp.data, 'status'):
                        # Battery percentage is status ID 70
                        batt_pct = getattr(resp.data.status, 'int_batt_per', None)
                        if batt_pct is not None:
                            return int(batt_pct)
                except (AttributeError, asyncio.TimeoutError):
                    pass

            # For WiFi AP transport: try HTTP status endpoint
            session = getattr(self._transport, "_session", None)
            base_url = getattr(self._transport, "_base_url", None)
            if session is not None and base_url is not None:
                try:
                    import aiohttp
                    url = f"{base_url}/gopro/camera/state"
                    async with session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=5.0),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json(content_type=None)
                            # Battery percentage is in status dict under key "70"
                            status_dict = data.get("status", {})
                            batt = status_dict.get("70") or status_dict.get(70)
                            if batt is not None:
                                return int(batt)
                except (ImportError, Exception):
                    pass

        except Exception:
            log.debug("Battery query failed", exc_info=True)

        return None

    # ------------------------------------------------------------------
    # Status snapshot polling
    # ------------------------------------------------------------------

    async def _status_poll_loop(self) -> None:
        """Periodically publish full status snapshots to the event bus."""
        while self._running:
            try:
                await asyncio.sleep(self._config.status_poll_interval)
                snapshot = self._collect_status_snapshot()
                self._publish(status_event(snapshot))
            except asyncio.CancelledError:
                raise
            except Exception:
                log.debug("Status poll error", exc_info=True)

    def _collect_status_snapshot(self) -> dict:
        """Build a full status snapshot from transport + pipeline state."""
        snapshot: dict = {
            "timestamp": time.monotonic(),
            "transport_type": self._active_transport_name,
            "battery_level": self._battery_level,
            "battery_charging": self._battery_charging,
        }

        # Transport info
        if self._transport is not None:
            snapshot["connection_state"] = self._transport.state.name
            snapshot["is_connected"] = self._transport.is_connected
            snapshot["is_streaming"] = self._transport.is_streaming

            stats = self._transport.stats
            snapshot["camera_model"] = stats.camera_model or ""
            snapshot["camera_serial"] = stats.camera_serial or ""
            snapshot["keepalives_sent"] = stats.keepalives_sent
            snapshot["keepalives_failed"] = stats.keepalives_failed
            snapshot["last_error"] = stats.last_error or ""
            snapshot["connection_uptime_s"] = round(stats.connection_uptime_s, 1)
        else:
            snapshot["connection_state"] = "DISCONNECTED"
            snapshot["is_connected"] = False
            snapshot["is_streaming"] = False

        # Pipeline info
        if self._pipeline is not None:
            snapshot["pipeline_state"] = self._pipeline.state.name
            snapshot["is_frozen"] = self._pipeline.is_frozen

            try:
                ps = self._pipeline.get_stats()
                snapshot["decoder_fps"] = ps.decoder_fps
                snapshot["vcam_fps"] = ps.vcam_fps
                snapshot["decoder_frames"] = ps.decoder_frames
                snapshot["vcam_frames"] = ps.vcam_frames
                snapshot["pipeline_uptime_s"] = ps.uptime_s
            except Exception:
                pass
        else:
            snapshot["pipeline_state"] = "STOPPED"
            snapshot["is_frozen"] = False

        # Manager info
        snapshot["manager_uptime_s"] = round(
            time.monotonic() - self._start_time, 1
        ) if self._start_time else 0.0
        snapshot["events_published"] = self._events_published
        snapshot["subscribers"] = self._bus.subscriber_count

        return snapshot

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _publish(self, event: Event) -> None:
        """Publish an event to the bus and increment counter."""
        self._bus.publish(event)
        self._events_published += 1

    # ------------------------------------------------------------------
    # Public status accessors
    # ------------------------------------------------------------------

    @property
    def transport(self) -> Optional[Transport]:
        return self._transport

    @property
    def pipeline(self) -> Optional[FramePipeline]:
        return self._pipeline

    @property
    def battery_level(self) -> int:
        return self._battery_level

    @property
    def active_transport_name(self) -> str:
        return self._active_transport_name

    def get_status_snapshot(self) -> dict:
        """Return a current status snapshot (synchronous)."""
        return self._collect_status_snapshot()
