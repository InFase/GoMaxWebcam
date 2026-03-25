# Dashboard: FastAPI web dashboard with SSE live data + REST control actions + BLE wizard

from gomaxwebcam.dashboard.status_tracker import CameraStatus, CameraStatusTracker
from gomaxwebcam.dashboard.app import create_app
from gomaxwebcam.dashboard.wizard import WizardState, create_wizard_router
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

__all__ = [
    "CameraStatus",
    "CameraStatusTracker",
    "create_app",
    "WizardState",
    "create_wizard_router",
    "BLEScanRequest",
    "BLEScanResponse",
    "CameraSelectRequest",
    "CameraSelectResponse",
    "ConfirmResponse",
    "DiscoveredCamera",
    "ProvisionRequest",
    "ProvisionResponse",
    "WizardCancelResponse",
    "WizardPhase",
    "WizardStatus",
]
