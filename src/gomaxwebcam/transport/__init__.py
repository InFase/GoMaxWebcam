# Transport layer: abstract base + concrete implementations (USB, COHN, WiFi AP)
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo, TransportStats
from gomaxwebcam.transport.usb import USBTransport
from gomaxwebcam.transport.cohn import COHNTransport
from gomaxwebcam.transport.wifi_ap import WiFiAPTransport
from gomaxwebcam.transport.wifi_manager import (
    WiFiManager,
    WiFiNetwork,
    WiFiConnectionState,
    WiFiManagerStats,
    WiFiManagerError,
    WiFiScanError,
    WiFiConnectError,
    WiFiPlatformError,
    is_gopro_ssid,
    GOPRO_AP_IP,
    GOPRO_AP_PORT,
)
from gomaxwebcam.transport.cohn_orchestrator import (
    COHNOrchestrator,
    OrchestratorPhase,
    OrchestratorStatus,
)
from gomaxwebcam.transport.cohn_persistence import (
    CohnCredentialStore,
    StoredCOHNCredentials,
    get_default_db_path,
)

__all__ = [
    "Transport",
    "TransportState",
    "StreamInfo",
    "TransportStats",
    "USBTransport",
    "COHNTransport",
    "WiFiAPTransport",
    # WiFi AP manager
    "WiFiManager",
    "WiFiNetwork",
    "WiFiConnectionState",
    "WiFiManagerStats",
    "WiFiManagerError",
    "WiFiScanError",
    "WiFiConnectError",
    "WiFiPlatformError",
    "is_gopro_ssid",
    "GOPRO_AP_IP",
    "GOPRO_AP_PORT",
    # COHN orchestrator
    "COHNOrchestrator",
    "OrchestratorPhase",
    "OrchestratorStatus",
    # COHN persistence
    "CohnCredentialStore",
    "StoredCOHNCredentials",
    "get_default_db_path",
]
