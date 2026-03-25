"""
dashboard/wizard.py — FastAPI router for the BLE setup wizard.

Provides endpoints for the 4-step COHN provisioning wizard:
  Step 1: POST /api/wizard/scan      — Scan/discover BLE devices
  Step 2: POST /api/wizard/select    — Pair/connect (select camera)
  Step 3: POST /api/wizard/provision — Provision COHN with WiFi creds
  Step 4: POST /api/wizard/confirm   — Confirm and persist credentials

Supporting endpoints:
  - POST /api/wizard/cancel    — Cancel an in-progress provisioning
  - GET  /api/wizard/status    — Current wizard status (JSON snapshot)
  - GET  /api/wizard/progress  — SSE stream of wizard progress events

The wizard delegates to COHNOrchestrator for the actual BLE/COHN work,
publishing progress events to both the SSE progress stream and the
system-wide EventBus.

BLE scanning uses bleak directly (a dependency of open-gopro) rather
than the hand-rolled ble/scanner.py module which is dead code.
open-gopro owns COHN credential persistence via its cohn_db (TinyDB).

Architecture:
    Dashboard wizard UI
      → POST /api/wizard/scan (discover cameras via bleak)
      → POST /api/wizard/select (pick camera)
      → POST /api/wizard/provision (WiFi creds → COHN setup)
      → POST /api/wizard/confirm (verify credentials persisted)
      → GET  /api/wizard/progress (SSE for live progress)
      ← COHNOrchestrator (BLE scan → BLE connect → provision → COHN connect)
      ← EventBus (progress events published for SSE)

Thread model:
    All endpoints run on the asyncio event loop.
    Long-running provisioning is launched as a background task.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Optional

from fastapi import APIRouter, Request, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse, JSONResponse

from gomaxwebcam.dashboard.wizard_models import (
    BLEScanRequest,
    BLEScanResponse,
    CameraSelectRequest,
    CameraSelectResponse,
    ConfirmResponse,
    DiscoveredCamera,
    ProvisionRequest,
    ProvisionResponse,
    WizardCancelResponse,
    WizardPhase,
    WizardStatus,
)
from gomaxwebcam.events import EventBus, Event, EventType

log = logging.getLogger("gomaxwebcam.dashboard.wizard")

# GoPro BLE service UUID and name prefixes for scanning
# (constants shared with open-gopro / bleak — not importing from dead ble/ module)
_GOPRO_SERVICE_UUID = "0000fea6-0000-1000-8000-00805f9b34fb"
_GOPRO_NAME_PREFIXES = ("GoPro ", "GP-")


# ---------------------------------------------------------------------------
# Wizard state (module-level singleton, attached to app state on include)
# ---------------------------------------------------------------------------


class WizardState:
    """Holds the current state of the setup wizard.

    Attached to app.state.wizard for access from endpoints.
    Manages the lifecycle of BLE scanning and COHN provisioning.
    Publishes wizard events to the system EventBus for other components.
    """

    def __init__(self, event_bus: Optional[EventBus] = None) -> None:
        self.status = WizardStatus()
        self.discovered_cameras: list[DiscoveredCamera] = []
        self.selected_camera: Optional[DiscoveredCamera] = None
        self._orchestrator: Any = None  # COHNOrchestrator
        self._provision_task: Optional[asyncio.Task] = None
        self._progress_queues: list[asyncio.Queue] = []
        self._lock = asyncio.Lock()
        self._event_bus: Optional[EventBus] = event_bus
        # Credentials from last successful provisioning (for confirm step)
        self._last_credentials: Any = None

    def set_event_bus(self, bus: EventBus) -> None:
        """Attach an EventBus for publishing wizard events system-wide."""
        self._event_bus = bus

    def _notify_progress(self, status: WizardStatus) -> None:
        """Push a status update to all progress SSE subscribers and EventBus."""
        self.status = status
        for q in list(self._progress_queues):
            try:
                q.put_nowait(status)
            except asyncio.QueueFull:
                # Drop oldest and retry
                try:
                    q.get_nowait()
                    q.put_nowait(status)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

        # Also publish to EventBus for system-wide visibility
        if self._event_bus is not None:
            try:
                self._event_bus.publish(Event(
                    type=EventType.CONNECTION,
                    data={
                        "source": "wizard",
                        "phase": status.phase if isinstance(status.phase, str) else status.phase.value,
                        "progress_pct": status.progress_pct,
                        "message": status.message,
                        "camera_name": status.camera_name,
                        "camera_ip": status.camera_ip,
                    },
                ))
            except Exception:
                log.debug("Failed to publish wizard event to EventBus", exc_info=True)

    def subscribe_progress(self) -> asyncio.Queue:
        """Create a new progress subscription queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        self._progress_queues.append(q)
        return q

    def unsubscribe_progress(self, q: asyncio.Queue) -> None:
        """Remove a progress subscription queue."""
        if q in self._progress_queues:
            self._progress_queues.remove(q)

    def reset(self) -> None:
        """Reset wizard to initial state."""
        self.status = WizardStatus()
        self.discovered_cameras = []
        self.selected_camera = None
        self._orchestrator = None
        self._last_credentials = None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------


def create_wizard_router() -> APIRouter:
    """Create the wizard API router.

    Returns an APIRouter with all wizard endpoints.
    The router expects app.state.wizard (WizardState) and
    app.state.auth_token to be set.
    """
    router = APIRouter(prefix="/api/wizard", tags=["wizard"])

    # -- Step 1: Scan endpoint --

    @router.post("/scan", response_model=BLEScanResponse)
    async def wizard_scan(
        request: Request,
        body: BLEScanRequest,
    ) -> BLEScanResponse:
        """Step 1: Scan for GoPro cameras via Bluetooth.

        Uses bleak (open-gopro's BLE dependency) to scan for GoPro
        cameras advertising the GoPro BLE service UUID.  The hand-rolled
        ble/scanner.py module is dead code and NOT imported here.

        Performs a synchronous scan — the response is returned after
        the scan completes (up to timeout_s seconds).
        """
        wizard: WizardState = request.app.state.wizard
        start_time = time.monotonic()

        wizard._notify_progress(WizardStatus(
            phase=WizardPhase.SCANNING,
            progress_pct=5,
            message="Scanning for GoPro cameras via Bluetooth...",
        ))

        try:
            from bleak import BleakScanner

            scanner = BleakScanner(
                service_uuids=[_GOPRO_SERVICE_UUID],
            )
            devices = await scanner.discover(timeout=body.timeout_s)

            elapsed = time.monotonic() - start_time
            cameras: list[DiscoveredCamera] = []
            for device in devices:
                name = device.name or ""
                if not any(name.startswith(p) for p in _GOPRO_NAME_PREFIXES):
                    # Also check metadata for service UUID match
                    meta_uuids = getattr(device, "metadata", {}).get("uuids", [])
                    if _GOPRO_SERVICE_UUID not in meta_uuids:
                        continue

                # Extract serial suffix from name
                serial_suffix = ""
                if name.startswith("GoPro "):
                    serial_suffix = name[6:].strip()
                elif name.startswith("GP-"):
                    serial_suffix = name[3:].strip()

                rssi = getattr(device, "rssi", -100) or -100

                cameras.append(DiscoveredCamera(
                    address=str(device.address),
                    name=name or f"GoPro ({str(device.address)[:8]}...)",
                    rssi=rssi,
                    serial_suffix=serial_suffix,
                    model_hint="",
                ))

            # Sort by signal strength (strongest first)
            cameras.sort(key=lambda c: c.rssi, reverse=True)

            wizard.discovered_cameras = cameras

            if cameras:
                phase = WizardPhase.SCAN_COMPLETE
                msg = f"Found {len(cameras)} GoPro camera(s)"
            else:
                phase = WizardPhase.SCAN_COMPLETE
                msg = "No GoPro cameras found. Make sure camera is on and nearby."

            wizard._notify_progress(WizardStatus(
                phase=phase,
                progress_pct=10,
                message=msg,
                elapsed_s=round(elapsed, 1),
            ))

            return BLEScanResponse(
                cameras=cameras,
                scan_duration_s=round(elapsed, 1),
            )

        except ImportError:
            error = "Bluetooth (bleak) not installed — BLE scanning unavailable"
            log.error(error)
            wizard._notify_progress(WizardStatus(
                phase=WizardPhase.FAILED,
                message=error,
                error=error,
            ))
            return BLEScanResponse(
                scan_duration_s=round(time.monotonic() - start_time, 1),
                error=error,
            )

        except Exception as e:
            error = f"BLE scan failed: {e}"
            log.error(error, exc_info=True)
            wizard._notify_progress(WizardStatus(
                phase=WizardPhase.FAILED,
                message=error,
                error=error,
            ))
            return BLEScanResponse(
                scan_duration_s=round(time.monotonic() - start_time, 1),
                error=error,
            )

    # -- Step 2: Select/pair endpoint --

    @router.post("/select", response_model=CameraSelectResponse)
    async def wizard_select(
        request: Request,
        body: CameraSelectRequest,
    ) -> CameraSelectResponse:
        """Step 2: Select a discovered camera for COHN provisioning.

        Pairs the wizard session with a specific camera by BLE address.
        Must be called after a successful scan, or with a known address.
        The selected camera will be used in the subsequent provision step.
        """
        wizard: WizardState = request.app.state.wizard

        # Validate camera was discovered
        matching = [
            c for c in wizard.discovered_cameras
            if c.address == body.address
        ]

        if not matching:
            # Allow selecting by address even without a prior scan
            # (e.g., user knows the BLE address)
            if body.address:
                camera = DiscoveredCamera(
                    address=body.address,
                    name=body.name or f"GoPro ({body.address[:8]}...)",
                )
                wizard.selected_camera = camera
                log.info("Selected camera by address: %s", body.address)

                wizard._notify_progress(WizardStatus(
                    phase=WizardPhase.SCAN_COMPLETE,
                    progress_pct=15,
                    message=f"Selected {camera.name}",
                    camera_name=camera.name,
                    camera_address=camera.address,
                ))

                return CameraSelectResponse(
                    selected=True,
                    address=camera.address,
                    name=camera.name,
                )
            else:
                return CameraSelectResponse(
                    selected=False,
                    error="No camera address provided",
                )

        camera = matching[0]
        wizard.selected_camera = camera
        log.info("Selected camera: %s (%s)", camera.name, camera.address)

        wizard._notify_progress(WizardStatus(
            phase=WizardPhase.SCAN_COMPLETE,
            progress_pct=15,
            message=f"Selected {camera.name}",
            camera_name=camera.name,
            camera_address=camera.address,
        ))

        return CameraSelectResponse(
            selected=True,
            address=camera.address,
            name=camera.name,
        )

    # -- Step 3: Provision endpoint --

    @router.post("/provision", response_model=ProvisionResponse)
    async def wizard_provision(
        request: Request,
        body: ProvisionRequest,
    ) -> ProvisionResponse:
        """Step 3: Start COHN provisioning on the selected camera.

        Launches provisioning as a background task. The provisioning
        sequence is:
          1. BLE connect to camera
          2. Send WiFi credentials
          3. Camera connects to WiFi
          4. Create TLS certificate
          5. Retrieve COHN credentials
          6. Persist credentials via open-gopro's cohn_db

        Credentials are automatically stored by open-gopro's SDK in
        cohn_db.json. Use POST /api/wizard/confirm (Step 4) to verify.

        Use GET /api/wizard/progress (SSE) for real-time progress,
        or GET /api/wizard/status for a JSON snapshot.
        """
        wizard: WizardState = request.app.state.wizard

        # Check if already provisioning
        if (
            wizard._provision_task is not None
            and not wizard._provision_task.done()
        ):
            return ProvisionResponse(
                started=False,
                phase=WizardPhase.PROVISIONING,
                message="Provisioning already in progress",
                error="Provisioning already in progress",
            )

        # Determine camera address
        camera_address = body.camera_address
        camera_name = body.camera_name

        if not camera_address and wizard.selected_camera:
            camera_address = wizard.selected_camera.address
            camera_name = camera_name or wizard.selected_camera.name

        if not camera_address:
            return ProvisionResponse(
                started=False,
                phase=WizardPhase.FAILED,
                error="No camera selected. Run scan and select first.",
            )

        # Start provisioning in background
        async def _do_provision() -> None:
            try:
                from gomaxwebcam.transport.cohn_orchestrator import (
                    COHNOrchestrator,
                    OrchestratorStatus,
                    OrchestratorPhase,
                )

                orchestrator = COHNOrchestrator()
                wizard._orchestrator = orchestrator

                # Map orchestrator phases to wizard phases
                phase_map = {
                    OrchestratorPhase.IDLE: WizardPhase.IDLE,
                    OrchestratorPhase.BLE_SCANNING: WizardPhase.PROVISIONING,
                    OrchestratorPhase.BLE_CONNECTING: WizardPhase.PROVISIONING,
                    OrchestratorPhase.PROVISIONING: WizardPhase.PROVISIONING,
                    OrchestratorPhase.COHN_DISCOVERING: WizardPhase.PROVISIONING,
                    OrchestratorPhase.COHN_CONNECTING: WizardPhase.PROVISIONING,
                    OrchestratorPhase.COHN_STREAMING: WizardPhase.PROVISIONING,
                    OrchestratorPhase.COMPLETE: WizardPhase.COMPLETE,
                    OrchestratorPhase.FAILED: WizardPhase.FAILED,
                    OrchestratorPhase.CANCELLED: WizardPhase.CANCELLED,
                }

                # Track the COHN username from orchestrator status
                _cohn_username = ""

                def _on_status(orch_status: OrchestratorStatus) -> None:
                    nonlocal _cohn_username
                    wizard_phase = phase_map.get(
                        orch_status.phase, WizardPhase.PROVISIONING,
                    )
                    # Extract username from the camera_ip / transport info
                    # (populated when COHN credentials are retrieved)
                    if hasattr(orch_status, "camera_ip") and orch_status.camera_ip:
                        _cohn_username = getattr(orch_status, "_cohn_username", _cohn_username)

                    wizard._notify_progress(WizardStatus(
                        phase=wizard_phase,
                        progress_pct=orch_status.progress_pct,
                        message=orch_status.message,
                        error=orch_status.error,
                        camera_name=orch_status.camera_name or camera_name,
                        camera_address=orch_status.camera_address or camera_address,
                        wifi_ssid=orch_status.wifi_ssid or body.wifi_ssid,
                        camera_ip=orch_status.camera_ip,
                        elapsed_s=round(orch_status.elapsed_s, 1),
                        cohn_provisioned=(
                            orch_status.phase == OrchestratorPhase.COMPLETE
                        ),
                        cohn_username=_cohn_username,
                    ))

                transport = await orchestrator.provision_and_connect(
                    wifi_ssid=body.wifi_ssid,
                    wifi_password=body.wifi_password,
                    ble_address=camera_address,
                    on_status=_on_status,
                )

                if transport is not None:
                    log.info("COHN provisioning complete via wizard")

                    # Read back credentials from open-gopro's cohn_db
                    # to populate wizard status with username
                    try:
                        from gomaxwebcam.provisioning import BLEProvisioningService
                        svc = BLEProvisioningService(
                            cohn_db_path=orchestrator._get_cohn_db_path(),
                        )
                        serial = wizard.selected_camera.serial_suffix if wizard.selected_camera else ""
                        stored = svc.get_stored_credentials(serial)
                        if stored and stored.provisioned:
                            wizard._last_credentials = stored
                            _cohn_username = stored.username
                            wizard._notify_progress(WizardStatus(
                                phase=WizardPhase.COMPLETE,
                                progress_pct=100,
                                message=f"COHN provisioned! Camera at {stored.ip_address}",
                                camera_name=camera_name or wizard.status.camera_name,
                                camera_address=camera_address,
                                wifi_ssid=body.wifi_ssid,
                                camera_ip=stored.ip_address,
                                cohn_provisioned=True,
                                cohn_username=stored.username,
                            ))
                            log.info(
                                "Credentials confirmed in cohn_db: ip=%s, user=%s",
                                stored.ip_address, stored.username,
                            )
                    except Exception as e:
                        log.debug("Could not read back cohn_db: %s", e)
                        # Not fatal — orchestrator already reported COMPLETE
                else:
                    log.warning("COHN provisioning did not produce a transport")

            except Exception as e:
                log.error("Wizard provisioning failed: %s", e, exc_info=True)
                wizard._notify_progress(WizardStatus(
                    phase=WizardPhase.FAILED,
                    message=f"Provisioning failed: {e}",
                    error=str(e),
                    camera_name=camera_name,
                    camera_address=camera_address,
                    wifi_ssid=body.wifi_ssid,
                ))

        wizard._provision_task = asyncio.create_task(_do_provision())

        wizard._notify_progress(WizardStatus(
            phase=WizardPhase.PROVISIONING,
            progress_pct=5,
            message="Starting COHN provisioning...",
            camera_name=camera_name,
            camera_address=camera_address,
            wifi_ssid=body.wifi_ssid,
        ))

        return ProvisionResponse(
            started=True,
            phase=WizardPhase.PROVISIONING,
            message=f"Provisioning started for {camera_name or camera_address}",
        )

    # -- Cancel endpoint --

    @router.post("/cancel", response_model=WizardCancelResponse)
    async def wizard_cancel(
        request: Request,
    ) -> WizardCancelResponse:
        """Cancel an in-progress provisioning operation.

        Signals the orchestrator to stop and cleans up BLE resources.
        """
        wizard: WizardState = request.app.state.wizard

        if wizard._orchestrator is not None and hasattr(wizard._orchestrator, "cancel"):
            wizard._orchestrator.cancel()
            log.info("Wizard provisioning cancelled by user")

            wizard._notify_progress(WizardStatus(
                phase=WizardPhase.CANCELLED,
                message="Provisioning cancelled by user",
                camera_name=wizard.status.camera_name,
                camera_address=wizard.status.camera_address,
            ))

            return WizardCancelResponse(
                cancelled=True,
                message="Provisioning cancelled",
            )

        if (
            wizard._provision_task is not None
            and not wizard._provision_task.done()
        ):
            wizard._provision_task.cancel()
            log.info("Wizard provision task cancelled")

            wizard._notify_progress(WizardStatus(
                phase=WizardPhase.CANCELLED,
                message="Provisioning cancelled by user",
            ))

            return WizardCancelResponse(
                cancelled=True,
                message="Provisioning task cancelled",
            )

        return WizardCancelResponse(
            cancelled=False,
            message="No provisioning in progress to cancel",
        )

    # -- Step 4: Confirm and persist credentials --

    @router.post("/confirm", response_model=ConfirmResponse)
    async def wizard_confirm(
        request: Request,
    ) -> ConfirmResponse:
        """Step 4: Confirm COHN credentials are persisted.

        Verifies that open-gopro's cohn_db has stored the credentials
        from the provisioning step. Returns the final camera connection
        details for the dashboard to display.

        Should be called after provisioning completes (phase == COMPLETE).
        This is the final wizard step — after confirmation, the COHN
        transport can be used for camera streaming.
        """
        wizard: WizardState = request.app.state.wizard

        # Check if provisioning actually completed
        phase = wizard.status.phase
        if isinstance(phase, WizardPhase):
            phase = phase.value
        if phase != WizardPhase.COMPLETE and phase != "complete":
            return ConfirmResponse(
                confirmed=False,
                error="Provisioning has not completed yet. Current phase: " + str(phase),
                message="Run provisioning first (Step 3).",
            )

        # Try cached credentials first
        if wizard._last_credentials and wizard._last_credentials.provisioned:
            creds = wizard._last_credentials
            log.info("Credentials confirmed (cached): ip=%s, user=%s", creds.ip_address, creds.username)
            return ConfirmResponse(
                confirmed=True,
                camera_ip=creds.ip_address,
                cohn_username=creds.username,
                camera_serial=creds.camera_serial,
                message=f"COHN credentials confirmed for camera at {creds.ip_address}",
            )

        # Fall back to reading from the credential cache (CohnCredentialStore)
        # which persists credentials after BLE provisioning for restart survival.
        serial = ""
        if wizard.selected_camera:
            serial = wizard.selected_camera.serial_suffix

        # Determine cohn_db path
        db_path = None
        if wizard._orchestrator and hasattr(wizard._orchestrator, "_get_cohn_db_path"):
            db_path = wizard._orchestrator._get_cohn_db_path()

        # Try CohnCredentialStore (local cache) first
        try:
            from gomaxwebcam.transport.cohn_persistence import CohnCredentialStore

            store = CohnCredentialStore(db_path=db_path) if db_path else CohnCredentialStore()
            cached = store.load_credentials(serial) if serial else None
            if cached and cached.is_valid:
                log.info("Credentials confirmed (CohnCredentialStore): ip=%s, user=%s", cached.ip_address, cached.username)

                wizard._notify_progress(WizardStatus(
                    phase=WizardPhase.COMPLETE,
                    progress_pct=100,
                    message=f"COHN credentials confirmed for camera at {cached.ip_address}",
                    camera_name=wizard.status.camera_name,
                    camera_address=wizard.status.camera_address,
                    wifi_ssid=wizard.status.wifi_ssid,
                    camera_ip=cached.ip_address,
                    cohn_provisioned=True,
                    cohn_username=cached.username,
                ))

                return ConfirmResponse(
                    confirmed=True,
                    camera_ip=cached.ip_address,
                    cohn_username=cached.username,
                    camera_serial=cached.camera_serial,
                    message=f"COHN credentials confirmed for camera at {cached.ip_address}",
                )
        except Exception as e:
            log.debug("CohnCredentialStore lookup failed: %s", e)

        # Fall back to reading from open-gopro's cohn_db via provisioning service
        try:
            from gomaxwebcam.provisioning import BLEProvisioningService

            if db_path:
                svc = BLEProvisioningService(cohn_db_path=db_path)
            else:
                svc = BLEProvisioningService()

            stored = svc.get_stored_credentials(serial)
            if stored and stored.provisioned:
                wizard._last_credentials = stored
                log.info("Credentials confirmed (cohn_db): ip=%s, user=%s", stored.ip_address, stored.username)

                # Update wizard status with username
                wizard._notify_progress(WizardStatus(
                    phase=WizardPhase.COMPLETE,
                    progress_pct=100,
                    message=f"COHN credentials confirmed for camera at {stored.ip_address}",
                    camera_name=wizard.status.camera_name,
                    camera_address=wizard.status.camera_address,
                    wifi_ssid=wizard.status.wifi_ssid,
                    camera_ip=stored.ip_address,
                    cohn_provisioned=True,
                    cohn_username=stored.username,
                ))

                return ConfirmResponse(
                    confirmed=True,
                    camera_ip=stored.ip_address,
                    cohn_username=stored.username,
                    camera_serial=stored.camera_serial,
                    message=f"COHN credentials confirmed for camera at {stored.ip_address}",
                )

        except Exception as e:
            log.warning("Failed to verify credentials in cohn_db: %s", e)

        # Credentials not found in cohn_db, but provisioning reported success
        # Return what we have from the wizard status
        if wizard.status.camera_ip:
            return ConfirmResponse(
                confirmed=True,
                camera_ip=wizard.status.camera_ip,
                cohn_username=wizard.status.cohn_username,
                message=f"Provisioning complete (camera at {wizard.status.camera_ip}). "
                        "Credentials managed by open-gopro.",
            )

        return ConfirmResponse(
            confirmed=False,
            error="Credentials not found in open-gopro cohn_db",
            message="Provisioning may have failed to persist credentials. Try reprovisioning.",
        )

    # -- Status snapshot endpoint --

    @router.get("/status", response_model=WizardStatus)
    async def wizard_status(
        request: Request,
    ) -> WizardStatus:
        """Return current wizard status as JSON.

        Single snapshot — use /api/wizard/progress for live SSE updates.
        """
        wizard: WizardState = request.app.state.wizard
        return wizard.status

    # -- Progress SSE endpoint --

    @router.get("/progress")
    async def wizard_progress(
        request: Request,
    ) -> StreamingResponse:
        """Server-Sent Events stream of wizard progress.

        Streams WizardStatus updates as SSE events in real-time.
        Each event is a complete status snapshot.

        SSE format:
            event: wizard_progress
            data: {"phase": "provisioning", "progress_pct": 45, ...}
        """
        wizard: WizardState = request.app.state.wizard

        async def event_generator() -> AsyncGenerator[str, None]:
            q = wizard.subscribe_progress()
            try:
                # Send current status immediately
                import json
                current = wizard.status.model_dump()
                yield f"event: wizard_progress\ndata: {json.dumps(current)}\n\n"

                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        status = await asyncio.wait_for(q.get(), timeout=30.0)
                        data = status.model_dump()
                        yield f"event: wizard_progress\ndata: {json.dumps(data)}\n\n"

                        # Stop streaming after terminal states
                        if status.phase in (
                            WizardPhase.COMPLETE,
                            WizardPhase.FAILED,
                            WizardPhase.CANCELLED,
                        ):
                            break
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield ": keepalive\n\n"
            except Exception:
                log.debug("Wizard progress SSE client disconnected")
            finally:
                wizard.unsubscribe_progress(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return router
