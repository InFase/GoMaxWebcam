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

import asyncio
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

        # Support Last-Event-ID header for resuming after reconnection.
        # The browser EventSource API sends this header automatically whenever
        # the connection is re-established after a drop, using the last ``id:``
        # field received from the server.
        since_event_id: Optional[int] = None
        last_event_id_header = request.headers.get("last-event-id", "").strip()
        if last_event_id_header:
            try:
                since_event_id = int(last_event_id_header)
            except ValueError:
                log.debug("Ignoring invalid Last-Event-ID header: %r", last_event_id_header)

        async def event_generator() -> AsyncGenerator[str, None]:
            subscription = bus.subscribe(
                event_types=event_types,
                include_history=include_history,
                since_event_id=since_event_id,
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

    # -- Camera control POST endpoints --

    @app.post("/api/camera/resolution")
    async def set_resolution(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Set camera resolution (e.g. 1080p, 720p, 480p).

        Persists to config.toml and applies if orchestrator is available.
        """
        body = await request.json()
        resolution = body.get("resolution", "")
        valid = ("480p", "720p", "1080p")
        if resolution not in valid:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid resolution. Must be one of: {valid}"},
            )

        # Persist to config
        cfg = _load_config(request)
        if cfg:
            cfg.video.resolution = resolution
            cfg.save()

        # Apply via orchestrator if available
        orch = getattr(request.app.state, "orchestrator", None)
        if orch and hasattr(orch, "set_resolution"):
            await orch.set_resolution(resolution)

        log.info("Resolution set to %s", resolution)
        return JSONResponse(content={"ok": True, "resolution": resolution})

    @app.post("/api/camera/fov")
    async def set_fov(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Set camera field of view (wide, linear, narrow, superview)."""
        body = await request.json()
        fov = body.get("fov", "")
        valid = ("wide", "linear", "narrow", "superview")
        if fov not in valid:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid FOV. Must be one of: {valid}"},
            )

        cfg = _load_config(request)
        if cfg:
            cfg.video.fov = fov
            cfg.save()

        orch = getattr(request.app.state, "orchestrator", None)
        if orch and hasattr(orch, "set_fov"):
            await orch.set_fov(fov)

        log.info("FOV set to %s", fov)
        return JSONResponse(content={"ok": True, "fov": fov})

    @app.post("/api/transport/priority")
    async def set_transport_priority(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Reorder transport priority list.

        Body: {"priority": ["USB", "COHN", "WiFi AP"]}
        Persists to config.toml and updates TransportManager.
        """
        body = await request.json()
        priority = body.get("priority", [])
        if not isinstance(priority, list) or not priority:
            return JSONResponse(
                status_code=400,
                content={"error": "priority must be a non-empty list"},
            )

        # Normalize to lowercase for config storage
        priority_lower = [p.lower().replace(" ", "_") for p in priority]

        cfg = _load_config(request)
        if cfg:
            cfg.transport.priority = priority_lower
            cfg.save()

        # Update TransportManager config if available
        orch = getattr(request.app.state, "orchestrator", None)
        if orch:
            tm = getattr(orch, "transport_manager", None)
            if tm:
                tm._config.priority = [p.upper().replace("WIFI_AP", "WiFi AP") for p in priority_lower]

        log.info("Transport priority set to %s", priority_lower)
        return JSONResponse(content={"ok": True, "priority": priority_lower})

    @app.post("/api/transport/switch")
    async def switch_transport(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Manually switch to a specific transport.

        Body: {"transport": "USB"} or {"transport": "COHN"}
        """
        body = await request.json()
        transport_name = body.get("transport", "")
        if not transport_name:
            return JSONResponse(
                status_code=400,
                content={"error": "transport name is required"},
            )

        # Normalize name to match registered transport names
        name_map = {
            "usb": "USB",
            "cohn": "COHN",
            "wifi_ap": "WiFi AP",
            "wifi ap": "WiFi AP",
        }
        normalized = name_map.get(transport_name.lower(), transport_name.upper())

        orch = getattr(request.app.state, "orchestrator", None)
        if not orch:
            return JSONResponse(
                status_code=503,
                content={"error": "Orchestrator not available"},
            )

        tm = getattr(orch, "transport_manager", None)
        if not tm:
            return JSONResponse(
                status_code=503,
                content={"error": "TransportManager not available"},
            )

        success = await tm.force_failover(normalized)
        log.info("Transport switch to '%s': %s", normalized, "success" if success else "failed")
        return JSONResponse(content={
            "ok": success,
            "transport": normalized,
            "detail": "switched" if success else "switch failed",
        })

    @app.post("/api/settings/auto-start")
    async def set_auto_start(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Toggle auto-start on login.

        Body: {"enabled": true/false}
        Persists to config.toml.
        """
        body = await request.json()
        enabled = bool(body.get("enabled", False))

        cfg = _load_config(request)
        if cfg:
            cfg.dashboard.auto_start = enabled
            cfg.save()

        log.info("Auto-start set to %s", enabled)
        return JSONResponse(content={"ok": True, "auto_start": enabled})

    @app.post("/api/settings")
    async def update_settings(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Update miscellaneous settings and persist to config.toml.

        Body: any combination of setting keys:
          - ble_wake_mode: "always_on" or "battery_saver"
          - debug_logging: true/false
          - auto_start: true/false
        """
        body = await request.json()
        cfg = _load_config(request)
        if not cfg:
            return JSONResponse(
                status_code=503,
                content={"error": "Config not available"},
            )

        _VALID_BLE_WAKE_MODES = {"always_on", "battery_saver"}
        if "ble_wake_mode" in body:
            mode = body["ble_wake_mode"]
            if mode in _VALID_BLE_WAKE_MODES:
                cfg.transport.ble_wake_mode = mode
            else:
                log.warning("Ignoring invalid ble_wake_mode=%r", mode)
        if "debug_logging" in body:
            cfg.logging.debug = bool(body["debug_logging"])
        if "auto_start" in body:
            cfg.dashboard.auto_start = bool(body["auto_start"])

        cfg.save()
        log.info("Settings updated: %s", list(body.keys()))
        return JSONResponse(content={"ok": True, "updated": list(body.keys())})

    @app.get("/api/settings")
    async def get_settings(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Return current settings from config.toml."""
        cfg = _load_config(request)
        if not cfg:
            return JSONResponse(content={"settings": {}})

        orch = getattr(request.app.state, "orchestrator", None)
        tm_status = {}
        if orch:
            tm = getattr(orch, "transport_manager", None)
            if tm:
                tm_status = tm.get_status()

        return JSONResponse(content={
            "settings": {
                "resolution": cfg.video.resolution,
                "fov": cfg.video.fov,
                "transport_priority": cfg.transport.priority,
                "ble_wake_mode": cfg.transport.ble_wake_mode,
                "debug_logging": cfg.logging.debug,
                "auto_start": cfg.dashboard.auto_start,
                "open_browser_on_start": cfg.dashboard.open_browser_on_start,
                "udp_port": cfg.advanced.udp_port,
            },
            "transport_status": tm_status,
        })

    # -- Keyboard shortcut action endpoints --

    @app.post("/api/camera/pause")
    async def toggle_pause(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Toggle stream pause / resume via the keyboard shortcut (P key).

        Uses the pipeline freeze/unfreeze mechanism:
        - pause=true  → pipeline.freeze()  — holds last frame in virtual camera
        - pause=false → pipeline.unfreeze() — resumes on next decoded frame

        Body: {"paused": true|false}

        If no pipeline is active (camera not connected), returns ok=true with a
        detail message so the frontend can display a friendly toast.
        """
        body = await request.json()
        want_pause = bool(body.get("paused", True))

        orch = getattr(request.app.state, "orchestrator", None)
        pipeline = getattr(orch, "pipeline", None) if orch else None

        if pipeline is None:
            return JSONResponse(content={
                "ok": True,
                "paused": want_pause,
                "detail": "No active pipeline — connect a camera first",
            })

        if want_pause:
            pipeline.freeze()
            log.info("Stream paused via keyboard shortcut")
        else:
            pipeline.unfreeze()
            log.info("Stream resumed via keyboard shortcut")

        return JSONResponse(content={"ok": True, "paused": want_pause})

    @app.post("/api/camera/reconnect")
    async def reconnect_camera(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Force a camera reconnect via the keyboard shortcut (R key).

        Triggers a transport failover cycle so the app re-discovers and
        reconnects the best available transport (USB → COHN → WiFi AP).

        Returns immediately — the reconnect happens asynchronously and
        status updates arrive via the SSE stream.
        """
        orch = getattr(request.app.state, "orchestrator", None)
        if not orch:
            return JSONResponse(
                status_code=503,
                content={"error": "App not fully started yet — try again in a moment"},
            )

        tm = getattr(orch, "transport_manager", None)
        if not tm:
            return JSONResponse(
                status_code=503,
                content={"error": "Transport manager not available"},
            )

        # Fire the failover asynchronously so we return immediately.
        # Empty string → use the priority list rather than a specific transport.
        async def _safe_reconnect():
            try:
                await tm.force_failover("")
            except Exception as e:
                log.error("Keyboard reconnect failed: %s", e, exc_info=True)

        asyncio.create_task(_safe_reconnect(), name="kbd-reconnect")
        log.info("Reconnect triggered via keyboard shortcut")
        return JSONResponse(content={"ok": True, "detail": "Reconnecting…"})

    @app.post("/api/camera/visibility")
    async def toggle_visibility(
        request: Request,
        _auth: None = Depends(verify_token),
    ) -> JSONResponse:
        """Toggle virtual camera output visibility (V key).

        When hidden the virtual camera outputs a blank (black) frame so apps
        like OBS still see a connected camera but receive no live content.
        When visible the live GoPro feed is restored to the virtual camera.

        This is distinct from pause (P key), which holds the last frame.
        Visibility lets the user blank the virtual camera without stopping
        decode or disconnecting the GoPro.

        Body: {"hidden": true|false}

        If no pipeline is active, returns ok=true with an explanatory detail
        so the frontend can display a friendly toast.
        """
        body = await request.json()
        want_hidden = bool(body.get("hidden", True))

        orch = getattr(request.app.state, "orchestrator", None)
        pipeline = getattr(orch, "pipeline", None) if orch else None

        if pipeline is None:
            return JSONResponse(content={
                "ok": True,
                "hidden": want_hidden,
                "detail": "No active pipeline — connect a camera first",
            })

        if want_hidden:
            pipeline.hide()
            log.info("Virtual camera hidden via keyboard shortcut")
        else:
            pipeline.show()
            log.info("Virtual camera shown via keyboard shortcut")

        return JSONResponse(content={"ok": True, "hidden": want_hidden})

    # -- Camera control API (open-gopro SDK passthrough) --

    from gomaxwebcam.camera_api import CameraAPI, CameraNotConnected

    def _get_gopro_handle():
        orch = getattr(app.state, "orchestrator", None)
        if not orch:
            return None
        tm = getattr(orch, "transport_manager", None)
        if not tm:
            return None
        active = tm.active_transport
        if not active:
            return None
        return active.gopro_handle

    # Load COHN credentials for direct HTTPS access (bypasses SDK transport)
    _cohn_ip = ""
    _cohn_user = ""
    _cohn_password = ""
    try:
        from gomaxwebcam.transport.cohn_persistence import CohnCredentialStore
        _store = CohnCredentialStore()
        # Try known serials, then scan the db
        for _serial in ["7212", "0067212"]:
            _creds = _store.load_credentials(_serial)
            if _creds and _creds.password:
                _cohn_ip = _creds.ip_address
                _cohn_user = _creds.username
                _cohn_password = _creds.password
                log.info("CameraAPI loaded COHN credentials for serial %s: %s@%s", _serial, _cohn_user, _cohn_ip)
                break
    except Exception as e:
        log.debug("COHN credential load failed: %s", e)

    app.state.camera_api = CameraAPI(
        get_gopro=_get_gopro_handle,
        cohn_ip=_cohn_ip,
        cohn_user=_cohn_user,
        cohn_password=_cohn_password,
    )

    async def _camera_call(request: Request, method: str, **kwargs) -> JSONResponse:
        api: CameraAPI = request.app.state.camera_api
        try:
            result = await getattr(api, method)(**kwargs)
            return JSONResponse(content=result)
        except CameraNotConnected:
            return JSONResponse(status_code=503, content={"ok": False, "error": "Camera not connected"})
        except Exception as e:
            log.error("Camera API error (%s): %s", method, e)
            return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})

    @app.post("/api/camera/shutter/start")
    async def start_recording(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "start_recording")

    @app.post("/api/camera/shutter/stop")
    async def stop_recording(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "stop_recording")

    @app.get("/api/camera/presets")
    async def get_presets(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "get_presets")

    @app.post("/api/camera/preset")
    async def load_preset(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        body = await request.json()
        return await _camera_call(request, "load_preset", preset_id=body.get("id", 0))

    @app.post("/api/camera/preset-group")
    async def load_preset_group(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        body = await request.json()
        return await _camera_call(request, "load_preset_group", group_id=body.get("id", 0))

    @app.get("/api/camera/state")
    async def camera_state(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "get_state")

    @app.get("/api/camera/info")
    async def camera_info(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "get_info")

    @app.get("/api/media/list")
    async def media_list(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "get_media_list")

    @app.get("/api/media/info")
    async def media_info(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        path = request.query_params.get("path", "")
        if not path:
            return JSONResponse(status_code=400, content={"error": "path required"})
        return await _camera_call(request, "get_media_info", path=path)

    @app.delete("/api/media/file")
    async def delete_media(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        body = await request.json()
        path = body.get("path", "")
        if not path:
            return JSONResponse(status_code=400, content={"error": "path required"})
        return await _camera_call(request, "delete_file", path=path)

    @app.get("/api/media/thumbnail")
    async def media_thumbnail(request: Request, _auth: None = Depends(verify_token)):
        """Proxy thumbnail from camera to avoid CORS issues."""
        import httpx
        path = request.query_params.get("path", "")
        if not path:
            return JSONResponse(status_code=400, content={"error": "path required"})
        api: CameraAPI = request.app.state.camera_api
        try:
            url = api.get_thumbnail_url(path)
            auth = api._auth()
            async with httpx.AsyncClient(verify=False, timeout=10.0, auth=auth) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    from fastapi.responses import Response
                    return Response(content=resp.content, media_type="image/jpeg")
                return JSONResponse(status_code=resp.status_code, content={"error": "Thumbnail not found"})
        except CameraNotConnected:
            return JSONResponse(status_code=503, content={"error": "Camera not connected"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/api/media/download")
    async def download_media(request: Request, _auth: None = Depends(verify_token)):
        """Proxy media file download from camera."""
        import httpx
        path = request.query_params.get("path", "")
        if not path:
            return JSONResponse(status_code=400, content={"error": "path required"})
        api: CameraAPI = request.app.state.camera_api
        try:
            url = api.get_download_url(path)
            auth = api._auth()
            async with httpx.AsyncClient(verify=False, timeout=300.0, auth=auth) as client:
                resp = await client.get(url)
                if resp.status_code == 200:
                    filename = path.split("/")[-1] if "/" in path else path
                    from fastapi.responses import Response
                    return Response(
                        content=resp.content,
                        media_type="application/octet-stream",
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
                    )
                return JSONResponse(status_code=resp.status_code, content={"error": "File not found"})
        except CameraNotConnected:
            return JSONResponse(status_code=503, content={"error": "Camera not connected"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.post("/api/camera/reboot")
    async def reboot_camera(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "reboot")

    @app.post("/api/camera/sleep")
    async def sleep_camera(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "sleep")

    @app.post("/api/ble/wake")
    async def ble_wake(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        """Wake camera from sleep via BLE connection."""
        try:
            from gomaxwebcam.ble.scanner import BLEScanner
            from gomaxwebcam.ble.gatt_client import GoProBLEClient

            scanner = BLEScanner(scan_timeout=8.0)
            gopros = await scanner.scan_once(timeout=8.0)
            named = [g for g in gopros if "GoPro" in g.name]
            if not named:
                return JSONResponse(content={"ok": False, "error": "No GoPro found via BLE. Camera may be fully powered off."})

            client = GoProBLEClient(address=named[0].address)
            if not await client.connect(timeout=15.0):
                return JSONResponse(content={"ok": False, "error": "BLE connection failed"})

            # Connection itself wakes the camera — wait a moment
            import asyncio
            await asyncio.sleep(2)
            await client.disconnect()

            return JSONResponse(content={"ok": True, "camera_name": named[0].name})
        except Exception as e:
            return JSONResponse(content={"ok": False, "error": str(e)})

    @app.post("/api/camera/zoom")
    async def set_zoom(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        body = await request.json()
        percent = int(body.get("percent", 0))
        if not 0 <= percent <= 100:
            return JSONResponse(status_code=400, content={"error": "percent must be 0-100"})
        return await _camera_call(request, "set_zoom", percent=percent)

    @app.post("/api/ble/auto-connect")
    async def ble_auto_connect(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        """BLE scan → pair → GATT connect → read COHN status → store credentials.

        If COHN IP from BLE is incomplete, falls back to mDNS discovery.
        """
        try:
            from gomaxwebcam.ble.scanner import BLEScanner
            from gomaxwebcam.ble.gatt_client import GoProBLEClient
            from gomaxwebcam.ble.cohn import COHNProvisioner

            # Step 1: BLE scan
            scanner = BLEScanner(scan_timeout=10.0)
            gopros = await scanner.scan_once(timeout=10.0)
            named = [g for g in gopros if "GoPro" in g.name]
            if not named:
                return JSONResponse(content={"ok": False, "error": "No GoPro found via Bluetooth. Make sure it's powered on."})

            target = named[0]
            log.info("BLE auto-connect: found %s @ %s", target.name, target.address)

            # Step 2: GATT connect (includes pairing)
            client = GoProBLEClient(address=target.address)
            if not await client.connect(timeout=20.0):
                return JSONResponse(content={"ok": False, "error": "BLE connection failed. Try putting the camera in pairing mode (Preferences > Connections > Connect Device)."})

            try:
                # Step 3: Query COHN status
                provisioner = COHNProvisioner(ble_client=client)
                creds = await provisioner.get_status()

                if not creds.password:
                    return JSONResponse(content={
                        "ok": False,
                        "error": "Camera is not provisioned for COHN yet. Use the GoPro Quik app to set up COHN first, then try again. Or use Quick Connect with manual credentials.",
                    })

                # Step 4: Validate/fix IP address
                ip = creds.ip_address or ""
                # BLE sometimes returns truncated IP — fall back to mDNS
                if not ip or ip.count(".") < 3 or ip.endswith("."):
                    log.warning("BLE returned incomplete IP '%s', trying mDNS...", ip)
                    try:
                        from zeroconf import Zeroconf, ServiceBrowser
                        import asyncio
                        from gomaxwebcam.discovery import find_gopro_device
                        # Quick mDNS scan
                        import subprocess, json as _json
                        result = subprocess.run(
                            ["python", "-c",
                             "import asyncio; from zeroconf import Zeroconf, ServiceBrowser; "
                             "zc = Zeroconf(); "
                             "import time; time.sleep(3); "
                             "info = zc.get_service_info('_gopro-web._tcp.local.', "
                             f"'C3531350067212._gopro-web._tcp.local.'); "
                             "print(info.parsed_addresses()[0] if info else ''); zc.close()"],
                            capture_output=True, text=True, timeout=10)
                        mdns_ip = result.stdout.strip()
                        if mdns_ip and mdns_ip.count(".") == 3:
                            ip = mdns_ip
                            log.info("mDNS found camera at %s", ip)
                    except Exception as e:
                        log.debug("mDNS fallback failed: %s", e)

                if not ip or ip.count(".") < 3:
                    return JSONResponse(content={
                        "ok": False,
                        "error": "Could not determine camera IP. Try Quick Connect and enter the IP manually.",
                    })

                # Step 5: Verify HTTP connectivity
                import httpx
                try:
                    async with httpx.AsyncClient(verify=False, timeout=5.0,
                                                  auth=httpx.BasicAuth(creds.username, creds.password)) as hc:
                        r = await hc.get(f"https://{ip}/gopro/camera/info")
                        r.raise_for_status()
                        cam_info = r.json()
                        log.info("HTTP verified: %s at %s", cam_info.get("model_name", "GoPro"), ip)
                except Exception as e:
                    log.warning("HTTP verify failed: %s", e)
                    return JSONResponse(content={
                        "ok": False,
                        "error": f"Camera found but HTTPS connection to {ip} failed. Make sure your PC and camera are on the same WiFi network.",
                    })

                # Step 6: Store and activate
                api: CameraAPI = request.app.state.camera_api
                api.set_cohn_credentials(ip, creds.username, creds.password)

                try:
                    from gomaxwebcam.transport.cohn_persistence import CohnCredentialStore, StoredCOHNCredentials
                    store = CohnCredentialStore()
                    store.store_credentials(StoredCOHNCredentials(
                        ip_address=ip,
                        username=creds.username,
                        password=creds.password,
                        certificate=creds.certificate or "ble-auto",
                        camera_serial=target.serial_suffix or "7212",
                    ))
                except Exception as e:
                    log.warning("Failed to persist credentials: %s", e)

                return JSONResponse(content={
                    "ok": True,
                    "ip": ip,
                    "username": creds.username,
                    "ssid": creds.ssid,
                    "camera_name": target.name,
                    "model": cam_info.get("model_name", "GoPro"),
                })
            finally:
                await client.disconnect()

        except Exception as e:
            log.error("BLE auto-connect failed: %s", e, exc_info=True)
            return JSONResponse(content={"ok": False, "error": str(e)})

    @app.post("/api/cohn/credentials")
    async def set_cohn_credentials(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        """Set or update COHN credentials for direct camera access."""
        body = await request.json()
        ip = body.get("ip", "")
        user = body.get("username", "gopro")
        password = body.get("password", "")
        if not ip or not password:
            return JSONResponse(status_code=400, content={"error": "ip and password required"})
        api: CameraAPI = request.app.state.camera_api
        api.set_cohn_credentials(ip, user, password)
        # Store for persistence
        try:
            from gomaxwebcam.transport.cohn_persistence import CohnCredentialStore, StoredCOHNCredentials
            store = CohnCredentialStore()
            serial = body.get("serial", "7212")
            store.store_credentials(StoredCOHNCredentials(
                ip_address=ip, username=user, password=password,
                certificate=body.get("certificate", "placeholder"),
                camera_serial=serial,
            ))
        except Exception as e:
            log.warning("Failed to persist COHN credentials: %s", e)
        return JSONResponse(content={"ok": True, "ip": ip})

    @app.post("/api/camera/webcam/start")
    async def webcam_start(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "webcam_start")

    @app.post("/api/camera/webcam/stop")
    async def webcam_stop(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "webcam_stop")

    @app.get("/api/camera/webcam/status")
    async def webcam_status(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "webcam_status")

    @app.post("/api/camera/preview/start")
    async def preview_start(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "preview_start")

    @app.post("/api/camera/preview/stop")
    async def preview_stop(request: Request, _auth: None = Depends(verify_token)) -> JSONResponse:
        return await _camera_call(request, "preview_stop")

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


def _load_config(request: Request):
    """Load the Config instance from app state or from disk.

    Returns a Config object, or None if config loading fails.
    """
    # Check if config is already cached on app state
    cfg = getattr(request.app.state, "config", None)
    if cfg is not None:
        return cfg

    try:
        from gomaxwebcam.config import Config
        cfg = Config.load()
        request.app.state.config = cfg
        return cfg
    except Exception:
        log.debug("Failed to load config", exc_info=True)
        return None


def _status_to_dict(status: CameraStatus) -> dict:
    """Convert CameraStatus to a JSON-serializable dict."""
    from dataclasses import asdict
    return asdict(status)
