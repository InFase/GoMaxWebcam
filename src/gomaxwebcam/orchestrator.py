"""
orchestrator.py — AppOrchestrator: top-level lifecycle coordinator for GoMaxWebcam v2.

The AppOrchestrator is the single entry point that wires together all v2
components and manages their startup/shutdown lifecycle:

  - CameraManager: transport + pipeline observation and SSE event publishing
  - TransportManager: USB/COHN failover chain with freeze-frame callbacks
  - EventBus: SSE event distribution
  - FramePipeline: decoder → virtual camera sink
  - CameraStatusTracker: dashboard status aggregation
  - FastAPI dashboard: SSE endpoints, status REST API, BLE wizard

Architecture:
    AppOrchestrator
      ├── EventBus (shared)
      ├── CameraManager (observes transport + pipeline, publishes SSE events)
      ├── TransportManager (owns failover chain, freeze/unfreeze callbacks)
      │     ├── USBTransport (primary)
      │     └── COHNTransport (fallback)
      ├── FramePipeline (decoder → virtual camera)
      ├── CameraStatusTracker (dashboard status polling)
      └── FastAPI app (lifespan-managed startup/shutdown)

The orchestrator owns the wiring — it connects TransportManager callbacks
(on_transport_switch, on_freeze, on_unfreeze) to CameraManager and
FramePipeline so that failover events propagate correctly.

FastAPI integration:
    create_dashboard() builds a FastAPI app whose lifespan context manager
    calls orch.start() on startup and orch.stop() on shutdown. This ensures
    all components are wired and running before the first HTTP request is
    served, and cleanly torn down when the server exits.

Thread model:
    All orchestrator methods run on the asyncio event loop thread.
    The orchestrator is created and driven from __main__.py.

Usage:
    bus = EventBus()
    camera_mgr = CameraManager(bus)
    transport_mgr = TransportManager(bus)

    orch = AppOrchestrator(
        event_bus=bus,
        camera_manager=camera_mgr,
        transport_manager=transport_mgr,
    )

    app = orch.create_dashboard(auth_token="...", host="127.0.0.1", port=8080)
    # Run with uvicorn — lifespan handles start/stop automatically
"""

from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from fastapi import FastAPI

from gomaxwebcam.camera_manager import CameraManager
from gomaxwebcam.dashboard.status_tracker import CameraStatusTracker
from gomaxwebcam.events import EventBus, cohn_reprovision_event
from gomaxwebcam.pipeline.frame_pipeline import FramePipeline, PipelineConfig
from gomaxwebcam.transport.base import Transport
from gomaxwebcam.transport.cohn_persistence import CohnCredentialStore
from gomaxwebcam.transport_manager import TransportManager

log = logging.getLogger("gomaxwebcam.orchestrator")


class AppOrchestrator:
    """Top-level lifecycle coordinator for GoMaxWebcam v2.

    Accepts and stores the core component instances, wires their callbacks,
    and coordinates ordered startup and shutdown.

    Args:
        event_bus: Shared SSE event bus for all components.
        camera_manager: Manages transport/pipeline observation and SSE publishing.
        transport_manager: Manages USB/COHN failover chain.
        status_tracker: Optional dashboard status aggregator.
    """

    def __init__(
        self,
        event_bus: EventBus,
        camera_manager: CameraManager,
        transport_manager: TransportManager,
        status_tracker: Optional[CameraStatusTracker] = None,
    ) -> None:
        self._bus = event_bus
        self._camera_manager = camera_manager
        self._transport_manager = transport_manager
        self._status_tracker = status_tracker

        # Pipeline reference (set when streaming starts)
        self._pipeline: Optional[FramePipeline] = None

        # FastAPI app (created by create_dashboard)
        self._app: Optional[FastAPI] = None
        self._auth_token: Optional[str] = None
        self._dashboard_host: Optional[str] = None
        self._dashboard_port: Optional[int] = None

        # Running state
        self._running: bool = False
        self._shutdown: bool = False

        # Background COHN reconnection task (eager startup)
        self._cohn_reconnect_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    @property
    def camera_manager(self) -> CameraManager:
        return self._camera_manager

    @property
    def transport_manager(self) -> TransportManager:
        return self._transport_manager

    @property
    def status_tracker(self) -> Optional[CameraStatusTracker]:
        return self._status_tracker

    @property
    def pipeline(self) -> Optional[FramePipeline]:
        return self._pipeline

    @property
    def app(self) -> Optional[FastAPI]:
        """The FastAPI app, or None if create_dashboard() hasn't been called."""
        return self._app

    @property
    def auth_token(self) -> Optional[str]:
        """The dashboard auth token, set by create_dashboard()."""
        return self._auth_token

    @property
    def dashboard_url(self) -> Optional[str]:
        """Full dashboard URL including auth token, or None if not configured."""
        if self._dashboard_host and self._dashboard_port and self._auth_token:
            return (
                f"http://{self._dashboard_host}:{self._dashboard_port}"
                f"/?token={self._auth_token}"
            )
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # FastAPI dashboard creation
    # ------------------------------------------------------------------

    def create_dashboard(
        self,
        auth_token: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> FastAPI:
        """Create and configure the FastAPI dashboard app with lifespan management.

        The returned app uses FastAPI's lifespan context manager to call
        ``self.start()`` on startup and ``self.stop()`` on shutdown. This
        guarantees that all components (EventBus, CameraManager,
        TransportManager, StatusTracker) are fully wired before the first
        HTTP request is served.

        Args:
            auth_token: Dashboard authentication token. If None, a random
                        token is generated.
            host: Bind address for dashboard URL construction.
            port: Bind port for dashboard URL construction.

        Returns:
            A configured FastAPI app ready to be served by uvicorn.

        Raises:
            RuntimeError: If create_dashboard() was already called.
        """
        if self._app is not None:
            raise RuntimeError("Dashboard already created — call create_dashboard() only once")

        self._auth_token = auth_token or secrets.token_urlsafe(32)
        self._dashboard_host = host
        self._dashboard_port = port

        # Import here to avoid circular imports at module level
        from gomaxwebcam.dashboard.app import create_app

        # Build the lifespan that ties orchestrator lifecycle to FastAPI
        @asynccontextmanager
        async def _lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
            log.info("FastAPI lifespan: starting AppOrchestrator...")
            await self.start()
            try:
                yield
            finally:
                log.info("FastAPI lifespan: stopping AppOrchestrator...")
                await self.stop()

        # Create the FastAPI app via the existing factory, then attach lifespan
        app = create_app(
            tracker=self._status_tracker,
            event_bus=self._bus,
            auth_token=self._auth_token,
        )

        # Attach the lifespan to the app
        app.router.lifespan_context = _lifespan

        # Store orchestrator reference on app.state for middleware/endpoints
        app.state.orchestrator = self

        self._app = app
        log.info(
            "Dashboard created (host=%s, port=%d, token=%s...)",
            host,
            port,
            self._auth_token[:8],
        )
        return app

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all components in dependency order.

        Startup sequence:
          1. Configure the event loop on the EventBus
          2. Wire TransportManager callbacks (freeze, unfreeze, transport switch)
          3. Start CameraManager (begins SSE polling tasks)
          4. Start CameraStatusTracker (if provided)
          5. Start TransportManager (discovers and connects first transport)

        Raises:
            RuntimeError: If the orchestrator is already running.
        """
        if self._shutdown:
            raise RuntimeError("AppOrchestrator has been shut down and cannot be restarted")

        if self._running:
            raise RuntimeError("AppOrchestrator is already running")

        log.info("AppOrchestrator starting...")

        # Step 1: Ensure EventBus knows the current loop
        self._bus.set_loop(asyncio.get_running_loop())

        # Step 2: Wire TransportManager callbacks
        self._wire_callbacks()

        # Step 3: Start CameraManager
        await self._camera_manager.start()

        # Step 4: Start StatusTracker
        if self._status_tracker is not None:
            await self._status_tracker.start(poll_interval=1.0)

        # Step 5: Start TransportManager (attempts first transport connection)
        await self._transport_manager.start()

        self._running = True
        log.info("AppOrchestrator started")

        # Step 6: Eager COHN background reconnection (non-blocking)
        # If cached COHN credentials exist, attempt to pre-warm the
        # COHN transport connection in the background so it's ready
        # for fast failover if USB drops.
        self._cohn_reconnect_task = asyncio.create_task(
            self._try_eager_cohn_reconnect(),
            name="eager-cohn-reconnect",
        )

    async def stop(self) -> None:
        """Stop all components in reverse dependency order.

        Shutdown sequence:
          1. Stop TransportManager (disconnects transports, cancels recovery)
          2. Stop FramePipeline (if active)
          3. Stop CameraStatusTracker (if provided)
          4. Stop CameraManager (cancels polling tasks)
          5. Shutdown EventBus (signals all SSE subscribers)

        Safe to call multiple times; subsequent calls are no-ops.
        The orchestrator can be restarted with ``start()`` after ``stop()``.
        For terminal cleanup (no restart), use ``shutdown()`` instead.
        """
        if not self._running:
            return

        log.info("AppOrchestrator stopping...")
        self._running = False

        # Cancel eager COHN reconnect if still running
        if self._cohn_reconnect_task is not None and not self._cohn_reconnect_task.done():
            self._cohn_reconnect_task.cancel()
            try:
                await self._cohn_reconnect_task
            except asyncio.CancelledError:
                pass
            log.debug("Eager COHN reconnect task cancelled")
        self._cohn_reconnect_task = None

        # Step 1: Stop TransportManager
        try:
            await self._transport_manager.stop()
        except Exception:
            log.exception("Error stopping TransportManager")

        # Step 2: Stop pipeline
        if self._pipeline is not None:
            try:
                await self._pipeline.stop()
            except Exception:
                log.exception("Error stopping FramePipeline")
            self._pipeline = None

        # Step 3: Stop StatusTracker
        if self._status_tracker is not None:
            try:
                await self._status_tracker.stop()
            except Exception:
                log.exception("Error stopping CameraStatusTracker")

        # Step 4: Stop CameraManager
        try:
            await self._camera_manager.stop()
        except Exception:
            log.exception("Error stopping CameraManager")

        # Step 5: Shutdown EventBus
        try:
            await self._bus.shutdown()
        except Exception:
            log.exception("Error shutting down EventBus")

        log.info("AppOrchestrator stopped")

    async def shutdown(self) -> None:
        """Terminal shutdown — stop all components and release all resources.

        Like ``stop()`` but additionally:
          - Best-effort ``stop_stream()`` on the active transport (3s timeout)
          - Clears the dashboard app reference
          - Unwires TransportManager callbacks
          - Marks the orchestrator as shut down (cannot be restarted)

        This is the method to call from signal handlers and application
        exit paths where the process is about to terminate.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        if self._shutdown:
            return

        log.info("AppOrchestrator shutting down (terminal)...")

        # Best-effort stop_stream before full teardown (3s timeout)
        active = self._transport_manager.active_transport
        if active is not None and active.is_streaming:
            try:
                await asyncio.wait_for(active.stop_stream(), timeout=3.0)
                log.debug("stop_stream completed during shutdown")
            except Exception:
                log.debug("stop_stream during shutdown failed", exc_info=True)

        # Full component teardown via stop()
        await self.stop()

        # Unwire callbacks so dangling references don't fire
        self._transport_manager.on_transport_switch = None
        self._transport_manager.on_freeze = None
        self._transport_manager.on_unfreeze = None

        # Clear dashboard reference
        self._app = None

        # Mark as terminally shut down
        self._shutdown = True

        log.info("AppOrchestrator shutdown complete")

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    def _wire_callbacks(self) -> None:
        """Wire TransportManager callbacks to CameraManager and pipeline.

        Connects:
          - on_transport_switch → updates CameraManager's observed transport
          - on_freeze → triggers pipeline freeze-frame
          - on_unfreeze → releases pipeline freeze-frame
        """
        self._transport_manager.on_transport_switch = self._on_transport_switch
        self._transport_manager.on_freeze = self._on_freeze
        self._transport_manager.on_unfreeze = self._on_unfreeze
        log.debug("TransportManager callbacks wired")

    async def _on_transport_switch(
        self, name: str, transport: Transport
    ) -> None:
        """Handle transport switch from TransportManager.

        Updates the CameraManager's observed transport so SSE events
        reflect the new active transport.  When the transport is already
        streaming, starts (or restarts) the FramePipeline so decoded
        frames flow to the virtual camera automatically.
        """
        log.info("Transport switched to '%s'", name)
        self._camera_manager.set_transport(transport)

        # Update StatusTracker if available
        if self._status_tracker is not None:
            self._status_tracker.set_transport(transport)

        # Start / restart the pipeline if the transport is streaming
        await self._ensure_pipeline(transport)

    async def _ensure_pipeline(self, transport: Transport) -> None:
        """Start or restart the FramePipeline for the given transport.

        If the transport is streaming and a stream_info is available, the
        pipeline is started or hot-swapped to the new stream source.

        **Freeze-frame preservation**: when a pipeline is already running
        (e.g. during USB→COHN failover), the decoder is swapped via
        ``pipeline.switch_stream()`` while the virtual camera sink stays
        alive.  This keeps the freeze-frame visible to downstream apps
        (Zoom, Teams, NVIDIA Broadcast) throughout the transition —
        there is no gap where the virtual camera device disappears.

        If no pipeline exists yet, a new one is created from scratch.

        The pipeline config is derived from the transport's StreamInfo via
        ``PipelineConfig.from_stream_info()`` which validates the codec and
        maps stream dimensions into the decoder + virtual camera setup.
        """
        if not transport.is_streaming:
            log.debug("Transport not streaming — skipping pipeline start")
            return

        stream_info = getattr(transport, "stream_info", None)
        if stream_info is None:
            log.debug("Transport has no stream_info — skipping pipeline start")
            return

        # Hot-swap: reuse existing pipeline's vcam sink for freeze-frame continuity
        if self._pipeline is not None and self._pipeline.is_running:
            log.info(
                "Hot-swapping pipeline decoder to new transport '%s' "
                "(vcam sink preserved for freeze-frame continuity)",
                transport.name,
            )
            try:
                ok = await self._pipeline.switch_stream(stream_info)
                if ok:
                    log.info(
                        "Pipeline decoder swapped to transport '%s' "
                        "(port=%d), freeze-frame preserved",
                        transport.name,
                        stream_info.port,
                    )
                    return
                else:
                    log.warning(
                        "Hot-swap failed, falling back to full pipeline restart"
                    )
            except Exception:
                log.exception("Error during pipeline hot-swap, falling back")

            # Hot-swap failed — stop old pipeline and create new one
            try:
                await self._pipeline.stop()
            except Exception:
                log.exception("Error stopping pipeline after failed hot-swap")

        elif self._pipeline is not None:
            # Pipeline exists but not running — stop it cleanly
            try:
                await self._pipeline.stop()
            except Exception:
                log.exception("Error stopping previous pipeline")

        # Build config from the transport's stream metadata
        try:
            config = PipelineConfig.from_stream_info(stream_info)
        except ValueError as exc:
            log.error("Cannot create pipeline config: %s", exc)
            return

        pipeline = FramePipeline(config)
        started = await pipeline.start(stream_info)
        if started:
            self._pipeline = pipeline
            self._camera_manager.set_pipeline(pipeline)
            if self._status_tracker is not None:
                self._status_tracker.set_pipeline(pipeline)
            log.info(
                "Pipeline started for transport '%s' "
                "(codec=%s, %dx%d@%dfps, port=%d, fmt=%s)",
                transport.name,
                stream_info.codec,
                stream_info.width,
                stream_info.height,
                stream_info.fps,
                stream_info.port,
                config.pixel_format,
            )
        else:
            log.warning("Pipeline failed to start (virtual camera may not be available)")

    def set_pipeline(self, pipeline: FramePipeline) -> None:
        """Manually set the pipeline (e.g. when created externally).

        Wires the pipeline into CameraManager and StatusTracker.
        """
        self._pipeline = pipeline
        self._camera_manager.set_pipeline(pipeline)
        if self._status_tracker is not None:
            self._status_tracker.set_pipeline(pipeline)

    def _on_freeze(self) -> None:
        """Handle freeze notification from TransportManager.

        Explicitly tells the FramePipeline to enter freeze-frame mode
        *immediately*, rather than waiting for the decoder's frame-timeout
        to fire.  This gives sub-second visual continuity — the vcam sink
        keeps re-sending the last good frame at 30 fps while the failover
        completes in the background.

        If no pipeline is active yet, this is a no-op (the virtual camera
        isn't running, so there is nothing to freeze).
        """
        log.info("Transport freeze signalled — freezing pipeline")
        if self._pipeline is not None:
            self._pipeline.freeze()

    def _on_unfreeze(self) -> None:
        """Handle unfreeze notification from TransportManager.

        Called when failover completes and a new transport is streaming.
        Signals the pipeline that fresh frames are expected — the actual
        state transition back to STREAMING happens when the first decoded
        frame arrives from the new transport.
        """
        log.info("Transport unfreeze signalled — pipeline will resume on first frame")
        if self._pipeline is not None:
            self._pipeline.unfreeze()

    # ------------------------------------------------------------------
    # Eager COHN background reconnection
    # ------------------------------------------------------------------

    async def _try_eager_cohn_reconnect(self) -> None:
        """Check for cached COHN credentials and pre-warm the COHN transport.

        Runs as a fire-and-forget background task launched at the end of
        ``start()``.  If the COHN transport is already registered with the
        TransportManager and cached credentials exist in the credential
        store, this method injects the credentials into the transport and
        attempts discover + connect (but *not* start_stream) so that the
        transport is ready for near-instant failover if USB drops.

        This method never blocks the main startup flow — it runs entirely
        in the background.  Errors are logged but never propagated.

        Flow:
          1. Look up the COHN transport from TransportManager
          2. Load cached credentials from CohnCredentialStore
          3. Inject credentials into the COHNTransport
          4. Call discover() + connect() to pre-warm the HTTPS session
          5. Log success/failure — do NOT start streaming (that's failover's job)
        """
        try:
            # Step 1: Find the COHN transport in the TransportManager
            cohn_transport = self._transport_manager.registered_transports.get("COHN")
            if cohn_transport is None:
                log.debug("Eager COHN reconnect: no COHN transport registered, skipping")
                return

            # Track which credentials we're using (for invalidation if they fail)
            creds = None

            # If the transport already has credentials, skip the store lookup
            if getattr(cohn_transport, "has_credentials", False):
                log.debug("Eager COHN reconnect: transport already has credentials")
            else:
                # Step 2: Check the credential store for cached credentials
                store = CohnCredentialStore()
                cameras = store.list_cameras()
                if not cameras:
                    log.debug("Eager COHN reconnect: no cached COHN credentials found")
                    return

                # Use the first valid credential set
                for c in cameras:
                    if c.is_valid:
                        creds = c
                        break

                if creds is None:
                    log.debug("Eager COHN reconnect: no valid cached credentials")
                    return

                # Step 3: Inject credentials into the transport
                log.info(
                    "Eager COHN reconnect: found cached credentials for %s (ip=%s)",
                    creds.camera_serial, creds.ip_address,
                )
                cohn_transport._inject_credentials(creds.username, creds.password)
                cohn_transport._inject_camera_ip(creds.ip_address)

            # Step 4: Pre-warm with discover + connect
            log.info("Eager COHN reconnect: attempting background connection...")

            discovered = await cohn_transport.discover(timeout=10.0)
            if not discovered:
                log.info("Eager COHN reconnect: discover failed (camera not reachable)")
                return

            connected = await cohn_transport.connect()
            if not connected:
                # Check if the failure was due to invalid credentials
                if getattr(cohn_transport, "credentials_invalid", False):
                    log.warning(
                        "Eager COHN reconnect: credentials invalid — "
                        "invalidating cache and signalling re-provisioning"
                    )
                    self._handle_cohn_credential_failure(
                        camera_serial=getattr(creds, "camera_serial", "") if creds else "",
                        camera_ip=getattr(creds, "ip_address", "") if creds else "",
                    )
                else:
                    log.info("Eager COHN reconnect: connect failed")
                return

            # Step 5: Success — transport is pre-warmed for failover
            log.info(
                "Eager COHN reconnect: COHN transport pre-warmed and ready for failover"
            )

        except asyncio.CancelledError:
            log.debug("Eager COHN reconnect: cancelled")
            raise  # Re-raise so the task registers as cancelled
        except Exception:
            # Never let background reconnect crash the app
            log.debug(
                "Eager COHN reconnect: failed (non-fatal)",
                exc_info=True,
            )

    def _handle_cohn_credential_failure(
        self,
        camera_serial: str = "",
        camera_ip: str = "",
    ) -> None:
        """Handle a COHN credential failure: invalidate cache and signal re-provisioning.

        Called when eager COHN reconnect detects that cached credentials are
        invalid (401, TLS errors, etc.).  This method:

          1. Removes the stale credentials from CohnCredentialStore
          2. Publishes a ``COHN_REPROVISION`` event on the EventBus so the
             dashboard can prompt the user to re-run BLE provisioning

        This is a synchronous, non-throwing method — errors in invalidation
        or event publishing are logged but never propagated.

        Args:
            camera_serial: Serial of the camera whose credentials failed.
            camera_ip: IP address of the camera (for event payload).
        """
        # Step 1: Invalidate cached credentials
        try:
            store = CohnCredentialStore()
            if camera_serial:
                store.invalidate_credentials(camera_serial)
                log.info(
                    "Invalidated cached COHN credentials for serial %s",
                    camera_serial,
                )
            else:
                # No serial — try to invalidate all cached credentials
                cameras = store.list_cameras()
                for cam in cameras:
                    store.invalidate_credentials(cam.camera_serial)
                log.info(
                    "Invalidated all cached COHN credentials (%d entries)",
                    len(cameras),
                )
        except Exception:
            log.debug("Error invalidating COHN credentials", exc_info=True)

        # Step 2: Publish re-provisioning event on the EventBus
        try:
            self._bus.publish(cohn_reprovision_event(
                reason="Cached COHN credentials failed during eager reconnect",
                camera_serial=camera_serial,
                camera_ip=camera_ip,
            ))
            log.info("Published COHN_REPROVISION event to dashboard")
        except Exception:
            log.debug("Error publishing COHN_REPROVISION event", exc_info=True)
