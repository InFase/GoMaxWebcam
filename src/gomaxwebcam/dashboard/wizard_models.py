"""
dashboard/wizard_models.py — Pydantic request/response models for BLE setup wizard.

Defines typed schemas for the COHN provisioning wizard endpoints:
  - BLE scan: discover GoPro cameras via Bluetooth
  - Camera select: choose a discovered camera for provisioning
  - COHN provision: provide WiFi credentials and start COHN setup
  - Wizard status: current provisioning progress

These models are used by the FastAPI wizard endpoints in app.py and
consumed by the dashboard frontend wizard UI.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class WizardPhase(str, Enum):
    """Current phase of the setup wizard."""
    IDLE = "idle"
    SCANNING = "scanning"
    SCAN_COMPLETE = "scan_complete"
    PROVISIONING = "provisioning"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# BLE Scan models
# ---------------------------------------------------------------------------


class BLEScanRequest(BaseModel):
    """Request to start a BLE scan for GoPro cameras.

    Attributes:
        timeout_s: Scan duration in seconds (default 10, max 30).
    """
    timeout_s: float = Field(
        default=10.0,
        ge=1.0,
        le=30.0,
        description="BLE scan duration in seconds",
    )


class DiscoveredCamera(BaseModel):
    """A GoPro camera discovered via BLE scan.

    Attributes:
        address: BLE MAC address (or UUID on macOS).
        name: Advertised BLE name (e.g. "GoPro 1234").
        rssi: Signal strength in dBm (more negative = weaker).
        serial_suffix: Last 4 chars of camera name (serial hint).
        model_hint: Parsed model string, if available.
    """
    address: str = Field(..., description="BLE MAC address")
    name: str = Field(..., description="BLE advertised name")
    rssi: int = Field(default=-100, description="Signal strength in dBm")
    serial_suffix: str = Field(default="", description="Serial number suffix")
    model_hint: str = Field(default="", description="Camera model hint")


class BLEScanResponse(BaseModel):
    """Response from a BLE scan operation.

    Attributes:
        cameras: List of discovered GoPro cameras.
        scan_duration_s: Actual scan duration in seconds.
        error: Error message if scan failed, empty on success.
    """
    cameras: list[DiscoveredCamera] = Field(
        default_factory=list,
        description="Discovered GoPro cameras",
    )
    scan_duration_s: float = Field(
        default=0.0,
        description="Actual scan duration in seconds",
    )
    error: str = Field(default="", description="Error message if scan failed")


# ---------------------------------------------------------------------------
# Camera selection models
# ---------------------------------------------------------------------------


class CameraSelectRequest(BaseModel):
    """Request to select a discovered camera for provisioning.

    Attributes:
        address: BLE address of the camera to select.
        name: Camera name (for display purposes).
    """
    address: str = Field(..., description="BLE address of camera to select")
    name: str = Field(default="", description="Camera name for display")


class CameraSelectResponse(BaseModel):
    """Response after selecting a camera.

    Attributes:
        selected: Whether the camera was successfully selected.
        address: BLE address of the selected camera.
        name: Camera name.
        error: Error message if selection failed.
    """
    selected: bool = Field(default=False, description="Whether selection succeeded")
    address: str = Field(default="", description="Selected camera BLE address")
    name: str = Field(default="", description="Selected camera name")
    error: str = Field(default="", description="Error message if selection failed")


# ---------------------------------------------------------------------------
# COHN Provision models
# ---------------------------------------------------------------------------


class ProvisionRequest(BaseModel):
    """Request to provision COHN on a selected camera.

    Requires WiFi credentials for the home network. The camera will
    connect to this network and enable COHN for HTTP API access.

    Attributes:
        wifi_ssid: Home WiFi network SSID.
        wifi_password: Home WiFi network password.
        camera_address: BLE address of the camera (from scan/select step).
        camera_name: Camera name (for display/logging).
    """
    wifi_ssid: str = Field(
        ...,
        min_length=1,
        max_length=32,
        description="WiFi network SSID",
    )
    wifi_password: str = Field(
        ...,
        min_length=1,
        max_length=63,
        description="WiFi network password",
    )
    camera_address: str = Field(
        default="",
        description="BLE address (empty = use previously selected camera)",
    )
    camera_name: str = Field(
        default="",
        description="Camera name for display/logging",
    )


class ProvisionResponse(BaseModel):
    """Response from a COHN provisioning operation.

    On success, contains the COHN credentials needed for wireless access.
    The provisioning runs asynchronously — use /api/wizard/status or
    /api/wizard/progress SSE for real-time updates.

    Attributes:
        started: Whether provisioning was successfully started.
        phase: Current wizard phase.
        message: Human-readable status message.
        error: Error message if provisioning failed to start.
    """
    model_config = ConfigDict(use_enum_values=True)

    started: bool = Field(default=False, description="Whether provisioning started")
    phase: WizardPhase = Field(
        default=WizardPhase.IDLE,
        description="Current wizard phase",
    )
    message: str = Field(default="", description="Human-readable status message")
    error: str = Field(default="", description="Error if provisioning failed to start")


# ---------------------------------------------------------------------------
# Wizard status models
# ---------------------------------------------------------------------------


class WizardStatus(BaseModel):
    """Full status of the setup wizard.

    Published via the /api/wizard/status endpoint and
    /api/wizard/progress SSE stream.

    Attributes:
        phase: Current wizard phase.
        progress_pct: Overall progress (0-100).
        message: Human-readable status message.
        error: Error detail if phase is FAILED.
        camera_name: BLE name of the selected camera.
        camera_address: BLE address of the selected camera.
        wifi_ssid: Target WiFi SSID.
        camera_ip: Camera IP after COHN provisioning.
        elapsed_s: Total elapsed seconds.
        cohn_provisioned: Whether COHN is fully provisioned.
        cohn_username: COHN HTTP username (if provisioned).
    """
    model_config = ConfigDict(use_enum_values=True)

    phase: WizardPhase = Field(
        default=WizardPhase.IDLE,
        description="Current wizard phase",
    )
    progress_pct: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Overall progress percentage",
    )
    message: str = Field(default="", description="Human-readable status")
    error: str = Field(default="", description="Error detail if failed")
    camera_name: str = Field(default="", description="Selected camera BLE name")
    camera_address: str = Field(default="", description="Selected camera BLE address")
    wifi_ssid: str = Field(default="", description="Target WiFi SSID")
    camera_ip: str = Field(default="", description="Camera IP after provisioning")
    elapsed_s: float = Field(default=0.0, description="Total elapsed seconds")
    cohn_provisioned: bool = Field(
        default=False,
        description="Whether COHN is fully provisioned",
    )
    cohn_username: str = Field(default="", description="COHN HTTP username")


class WizardCancelResponse(BaseModel):
    """Response from cancelling the wizard.

    Attributes:
        cancelled: Whether cancellation was accepted.
        message: Human-readable message.
    """
    cancelled: bool = Field(default=False, description="Whether cancellation accepted")
    message: str = Field(default="", description="Human-readable message")


class ConfirmResponse(BaseModel):
    """Response from the credential confirmation step (Step 4).

    After COHN provisioning completes, the confirm endpoint verifies
    that credentials were persisted by open-gopro's cohn_db and returns
    the final connection details.

    Attributes:
        confirmed: Whether credentials are persisted and valid.
        camera_ip: Camera IP address on the home network.
        cohn_username: COHN HTTP basic auth username.
        camera_serial: Camera serial suffix.
        message: Human-readable confirmation message.
        error: Error message if confirmation failed.
    """
    confirmed: bool = Field(default=False, description="Whether credentials are confirmed")
    camera_ip: str = Field(default="", description="Camera IP on home network")
    cohn_username: str = Field(default="", description="COHN HTTP username")
    camera_serial: str = Field(default="", description="Camera serial suffix")
    message: str = Field(default="", description="Confirmation message")
    error: str = Field(default="", description="Error if confirmation failed")
