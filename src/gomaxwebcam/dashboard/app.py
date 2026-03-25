"""
dashboard/app.py — FastAPI application with SSE endpoint for camera status.

Creates a FastAPI app with:
  - GET /api/status/stream — SSE endpoint for live status events (tracker-based)
  - GET /api/events/stream — SSE endpoint for typed reactive events (EventBus)
  - GET /api/status — Single JSON snapshot of current camera status
  - Startup token auth via query parameter or Authorization header

The /api/events/stream endpoint is the primary real-time channel, streaming
typed events (connection, transport, battery, pipeline, error, status) from
the reactive EventBus. Each event carries its own type and data payload.

The /api/status/stream endpoint streams CameraStatus snapshots for backward
compatibility — each event is a complete status snapshot.

Architecture:
    EventBus receives typed events from CameraManager, TransportFailover,
    and other components. SSE subscribers receive events as they happen
    with no polling delay.

    CameraStatusTracker polls Transport + Pipeline every ~1 second for the
    legacy /api/status/stream endpoint and /api/status REST snapshot.

Security:
    A random token is generated at startup and required for all API access.
    Pass it as ?token=<value> or Authorization: Bearer <value>.
    The token is displayed in the system tray / logs for the local user.
"""

from __future__ import annotations

import logging
import secrets
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from gomaxwebcam.dashboard.status_tracker import CameraStatusTracker, CameraStatus
from gomaxwebcam.dashboard.wizard import WizardState, create_wizard_router
from gomaxwebcam.events import EventBus, Event, EventType

log = logging.getLogger("gomaxwebcam.dashboard.app")

# Directory containing static frontend files (index.html, etc.)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    tracker: Optional[CameraStatusTracker] = None,
    event_bus: Optional[EventBus] = None,
    auth_token: Optional[str] = None,
    wizard_state: Optional[WizardState] = None,
) -> FastAPI:
    """Create the FastAPI dashboard application.

    Args:
        tracker: CameraStatusTracker instance. If None, creates a new one.
        event_bus: EventBus for reactive SSE events. If None, creates a new one.
        auth_token: Startup auth token. If None, generates a random one.
        wizard_state: WizardState for BLE setup wizard. If None, creates a new one.

    Returns:
        Configured FastAPI app with SSE, REST, and wizard endpoints.
    """
    if tracker is None:
        tracker = CameraStatusTracker()

    if event_bus is None:
        event_bus = EventBus()

    if auth_token is None:
        auth_token = secrets.token_urlsafe(32)

    if wizard_state is None:
        wizard_state = WizardState(event_bus=event_bus)
    elif wizard_state._event_bus is None:
        wizard_state.set_event_bus(event_bus)

    app = FastAPI(
        title="GoMaxWebcam Dashboard",
        version="2.0.0",
        docs_url=None,    # Disable Swagger UI in production
        redoc_url=None,
    )

    # Store on app state for access in dependencies
    app.state.tracker = tracker
    app.state.event_bus = event_bus
    app.state.auth_token = auth_token
    app.state.wizard = wizard_state

    # -- Auth dependency --

    async def verify_token(
        request: Request,
        token: Optional[str] = Query(None),
    ) -> None:
        """Verify the startup auth token.

        Accepts token via:
          - Query parameter: ?token=<value>
          - Authorization header: Bearer <value>
        """
        expected = request.app.state.auth_token

        # Check query parameter first
        if token and secrets.compare_digest(token, expected):
            return

        # Check Authorization header
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            bearer_token = auth_header[7:]
            if secrets.compare_digest(bearer_token, expected):
                return

        raise HTTPException(status_code=401, detail="Invalid or missing auth token")

    # -- Reactive SSE endpoint (EventBus) --

    @app.get("/api/events/stream")
    async def events_stream(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> StreamingResponse:
        """Server-Sent Events endpoint for real-time typed events.

        Streams typed events from the reactive EventBus as they occur.
        Event types: connection, transport, battery, pipeline, error, status.

        SSE format:
            event: connection
            data: {"old_state": "DISCONNECTED", "new_state": "CONNECTING", ...}

            event: battery
            data: {"level": 85, "charging": false}

            event: transport
            data: {"old_transport": "USB", "new_transport": "COHN", "reason": "failover"}

        Clients receive events immediately — no polling delay.
        Use include_history=true query parameter to replay recent events on connect.
        """
        bus: EventBus = request.app.state.event_bus
        include_history = request.query_params.get("include_history", "").lower() == "true"

        # Parse optional event type filter from query parameter
        # e.g. ?types=connection,battery,pipeline
        types_param = request.query_params.get("types", "")
        event_types: Optional[set[EventType]] = None
        if types_param:
            try:
                event_types = {EventType(t.strip()) for t in types_param.split(",") if t.strip()}
            except ValueError:
                pass  # Invalid type names are ignored, subscribe to all

        async def event_generator() -> AsyncGenerator[str, None]:
            subscription = bus.subscribe(
                event_types=event_types,
                include_history=include_history,
            )
            try:
                async for event in subscription:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break
                    yield event.to_sse()
            except Exception:
                log.debug("SSE events client disconnected")
            finally:
                bus.unsubscribe(subscription)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # -- Legacy SSE endpoint (tracker-based) --

    @app.get("/api/status/stream")
    async def status_stream(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> StreamingResponse:
        """Server-Sent Events endpoint for live camera status snapshots.

        Streams camera status as JSON events whenever the status changes.
        Each event contains a complete CameraStatus snapshot.

        SSE format:
            event: status
            data: {"transport_type": "USB", "connection_state": "STREAMING", ...}

        The first event is sent immediately with current status.
        Subsequent events are sent on status change (typically every 1s if changed).

        Note: Prefer /api/events/stream for real-time typed events.
        This endpoint is kept for backward compatibility.
        """
        tracker_: CameraStatusTracker = request.app.state.tracker

        async def event_generator() -> AsyncGenerator[str, None]:
            subscription = tracker_.subscribe()
            try:
                async for status in subscription:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        break
                    yield format_sse_event("status", status.to_sse_data())
            except Exception:
                log.debug("SSE client disconnected")
            finally:
                subscription._cleanup()

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # -- REST endpoints --

    @app.get("/api/status")
    async def status_snapshot(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Return current camera status as JSON.

        Single snapshot — use /api/events/stream for live updates.
        """
        tracker_: CameraStatusTracker = request.app.state.tracker
        status = tracker_.current
        return JSONResponse(
            content={"status": _status_to_dict(status)},
        )

    @app.get("/api/events/recent")
    async def recent_events(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Return recent event history from the EventBus.

        Returns the last N events (up to 100) as a JSON array.
        Useful for clients that want to catch up on missed events.
        """
        bus: EventBus = request.app.state.event_bus
        events = [e.to_dict() for e in bus.recent_events]
        return JSONResponse(
            content={"events": events, "count": len(events)},
        )

    # -- Dashboard HTML --

    @app.get("/")
    async def dashboard_page(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> HTMLResponse:
        """Serve the main dashboard HTML page.

        The page uses htmx + Alpine.js + Pico CSS + uPlot (all from CDN,
        no build step). The auth token is injected so JS can connect to SSE.
        """
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard HTML not found")
        html = index_path.read_text(encoding="utf-8")
        # Inject the auth token so the frontend can authenticate API calls
        html = html.replace("__AUTH_TOKEN__", request.app.state.auth_token)
        return HTMLResponse(content=html)

    # -- Pipeline metrics endpoints --

    @app.get("/api/pipeline/health")
    async def pipeline_health(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Return pipeline health assessment against v1 performance baselines.

        Evaluates decode FPS, latency, drop rate, and jitter against
        thresholds derived from v1 proven behavior.

        Returns:
            JSON with health status (healthy/degraded/unhealthy/unknown)
            and per-metric breakdown.
        """
        from dataclasses import asdict

        orch = getattr(request.app.state, "orchestrator", None)
        pipeline = getattr(orch, "pipeline", None) if orch else None

        if pipeline is None:
            return JSONResponse(content={
                "health": {
                    "status": "unknown",
                    "details": "Pipeline not available",
                    "fps_ok": True,
                    "latency_ok": True,
                    "drop_rate_ok": True,
                },
            })

        health = pipeline.get_health()
        return JSONResponse(content={"health": asdict(health)})

    @app.get("/api/pipeline/stats")
    async def pipeline_stats(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Return detailed pipeline statistics for diagnostics.

        Includes decoder stats, vcam stats, latency metrics, and health
        assessment. Used by the dashboard diagnostics panel.
        """
        orch = getattr(request.app.state, "orchestrator", None)
        pipeline = getattr(orch, "pipeline", None) if orch else None

        if pipeline is None:
            return JSONResponse(content={"stats": None, "detail": "Pipeline not running"})

        detailed = pipeline.get_detailed_stats()
        return JSONResponse(content={"stats": detailed})

    # -- Health (no auth required) --

    @app.get("/health")
    async def health() -> JSONResponse:
        """Health check endpoint (no auth required)."""
        return JSONResponse(content={"ok": True})

    # -- Wizard router (BLE setup endpoints) --
    wizard_router = create_wizard_router()
    # Wizard endpoints use the same auth dependency
    app.include_router(wizard_router, dependencies=[Depends(verify_token)])

    # -- Static file mount --
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def format_sse_event(event: str, data: str) -> str:
    """Format a server-sent event string.

    Args:
        event: Event type name.
        data: JSON data string.

    Returns:
        SSE-formatted string with event type and data fields.
    """
    return f"event: {event}\ndata: {data}\n\n"


def _status_to_dict(status: CameraStatus) -> dict:
    """Convert CameraStatus to a JSON-serializable dict."""
    from dataclasses import asdict
    return asdict(status)
