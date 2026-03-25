"""
config.py -- TOML config loader for GoMaxWebcam v2.

Handles loading/saving config.toml from the platform config directory,
COHN credential storage via keyring, and CA cert management.

Config directory: platformdirs.user_config_dir("GoMaxWebcam-v2")
Config file: config.toml

Sections:
  [camera]    - BLE address, camera name/serial, last known IP, model
  [transport] - Priority list, BLE wake mode
  [video]     - Resolution, FOV
  [dashboard] - No persistent settings (port is dynamic)
  [wifi_ap]   - AP name/password (plaintext, not sensitive)
  [advanced]  - Frame timeout, backoff, health check interval
  [logging]   - Debug flag, log retention, max log size

COHN credentials (Basic Auth password, WiFi password) are stored via
keyring, not in the TOML file.  COHN Root CA cert is stored as a PEM
file alongside config.toml.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import keyring
import tomli_w
from platformdirs import user_config_dir

log = logging.getLogger("gomaxwebcam.config")

_APP_NAME = "GoMaxWebcam-v2"
_CONFIG_FILENAME = "config.toml"
_CA_CERT_FILENAME = "cohn_root_ca.pem"

# Keyring service / username constants
_KR_SERVICE = "GoMaxWebcam-v2"
_KR_COHN_PASSWORD = "cohn_password"
_KR_WIFI_PASSWORD = "wifi_password"


# ---------------------------------------------------------------------------
# Section dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CameraSection:
    ble_address: str = ""
    camera_name: str = ""
    camera_serial: str = ""
    last_known_ip: str = ""
    model: str = ""


@dataclass
class TransportSection:
    priority: list[str] = field(default_factory=lambda: ["usb", "cohn", "wifi_ap"])
    ble_wake_mode: str = "always_on"


@dataclass
class VideoSection:
    resolution: str = "1080p"
    fov: str = "wide"


@dataclass
class DashboardSection:
    """No persistent settings -- port is dynamic."""
    pass


@dataclass
class WifiApSection:
    ap_name: str = ""
    ap_password: str = ""


@dataclass
class AdvancedSection:
    frame_timeout_seconds: int = 3
    backoff_base_seconds: int = 2
    backoff_cap_seconds: int = 60
    health_check_interval: int = 5


@dataclass
class LoggingSection:
    debug: bool = False
    log_retention_days: int = 7
    log_max_size_mb: int = 50


# ---------------------------------------------------------------------------
# Default TOML content (with comments for first-run creation)
# ---------------------------------------------------------------------------

_DEFAULT_TOML = """\
# GoMaxWebcam v2 configuration
# https://github.com/your-repo/GoMaxWebcam

[camera]
# BLE MAC address of your GoPro (discovered automatically on first connect)
ble_address = ""
camera_name = ""
camera_serial = ""
last_known_ip = ""
model = ""

[transport]
# Ordered list of transports to try: usb, cohn, wifi_ap
priority = ["usb", "cohn", "wifi_ap"]
# BLE wake mode: "always_on" or "on_demand"
ble_wake_mode = "always_on"

[video]
resolution = "1080p"
fov = "wide"

[dashboard]
# No persistent settings -- port is assigned dynamically at startup.

[wifi_ap]
# GoPro WiFi AP credentials (not security-sensitive)
ap_name = ""
ap_password = ""

[advanced]
frame_timeout_seconds = 3
backoff_base_seconds = 2
backoff_cap_seconds = 60
health_check_interval = 5

[logging]
debug = false
log_retention_days = 7
log_max_size_mb = 50
"""


# ---------------------------------------------------------------------------
# Config class
# ---------------------------------------------------------------------------

class Config:
    """TOML-backed configuration for GoMaxWebcam v2.

    Usage:
        cfg = Config.load()
        cfg.camera.ble_address = "AA:BB:CC:DD:EE:FF"
        cfg.save()

        # COHN credentials via keyring
        cfg.set_cohn_password("my-secret")
        pw = cfg.get_cohn_password()
    """

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        self._config_dir = Path(config_dir) if config_dir else Path(
            user_config_dir(_APP_NAME)
        )
        self._config_path = self._config_dir / _CONFIG_FILENAME

        # Sections
        self.camera = CameraSection()
        self.transport = TransportSection()
        self.video = VideoSection()
        self.dashboard = DashboardSection()
        self.wifi_ap = WifiApSection()
        self.advanced = AdvancedSection()
        self.logging = LoggingSection()

    # -- Properties --

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    @property
    def config_path(self) -> Path:
        return self._config_path

    # -- Load / Save --

    @classmethod
    def load(cls, config_dir: Optional[Path] = None) -> Config:
        """Load config from disk, creating default if none exists.

        Args:
            config_dir: Override config directory (mainly for testing).

        Returns:
            Populated Config instance.
        """
        cfg = cls(config_dir=config_dir)

        if not cfg._config_path.exists():
            log.info("No config found, creating default at %s", cfg._config_path)
            cfg._config_dir.mkdir(parents=True, exist_ok=True)
            cfg._config_path.write_text(_DEFAULT_TOML, encoding="utf-8")
            return cfg

        try:
            raw = cfg._config_path.read_bytes()
            data = tomllib.loads(raw.decode("utf-8"))
        except Exception:
            log.exception("Failed to parse %s, using defaults", cfg._config_path)
            return cfg

        cfg._apply_dict(data)
        log.info("Config loaded from %s", cfg._config_path)
        return cfg

    def save(self) -> None:
        """Write current config to disk as TOML."""
        self._config_dir.mkdir(parents=True, exist_ok=True)
        data = self._to_dict()
        raw = tomli_w.dumps(data)
        self._config_path.write_text(raw, encoding="utf-8")
        log.info("Config saved to %s", self._config_path)

    # -- COHN credential management (keyring) --

    def get_cohn_password(self) -> Optional[str]:
        """Retrieve COHN Basic Auth password from OS keyring."""
        try:
            return keyring.get_password(_KR_SERVICE, _KR_COHN_PASSWORD)
        except Exception:
            log.debug("Failed to read COHN password from keyring", exc_info=True)
            return None

    def set_cohn_password(self, password: str) -> None:
        """Store COHN Basic Auth password in OS keyring."""
        try:
            keyring.set_password(_KR_SERVICE, _KR_COHN_PASSWORD, password)
        except Exception:
            log.exception("Failed to store COHN password in keyring")

    def get_wifi_password(self) -> Optional[str]:
        """Retrieve WiFi password from OS keyring."""
        try:
            return keyring.get_password(_KR_SERVICE, _KR_WIFI_PASSWORD)
        except Exception:
            log.debug("Failed to read WiFi password from keyring", exc_info=True)
            return None

    def set_wifi_password(self, password: str) -> None:
        """Store WiFi password in OS keyring."""
        try:
            keyring.set_password(_KR_SERVICE, _KR_WIFI_PASSWORD, password)
        except Exception:
            log.exception("Failed to store WiFi password in keyring")

    # -- CA cert management --

    def get_ca_cert_path(self) -> Path:
        """Return path to the COHN Root CA PEM file.

        The file may or may not exist -- callers should check .exists().
        """
        return self._config_dir / _CA_CERT_FILENAME

    # -- Internal helpers --

    def _apply_dict(self, data: dict[str, Any]) -> None:
        """Apply a parsed TOML dict onto the section dataclasses."""
        section_map: dict[str, Any] = {
            "camera": self.camera,
            "transport": self.transport,
            "video": self.video,
            "wifi_ap": self.wifi_ap,
            "advanced": self.advanced,
            "logging": self.logging,
        }
        for section_name, section_obj in section_map.items():
            toml_section = data.get(section_name, {})
            if not isinstance(toml_section, dict):
                continue
            for f in fields(section_obj):
                if f.name in toml_section:
                    try:
                        setattr(section_obj, f.name, toml_section[f.name])
                    except Exception:
                        log.warning(
                            "Ignoring invalid value for [%s].%s",
                            section_name, f.name,
                        )

    def _to_dict(self) -> dict[str, Any]:
        """Serialize all sections to a plain dict for TOML writing."""
        from dataclasses import asdict

        result: dict[str, Any] = {}
        section_map: dict[str, Any] = {
            "camera": self.camera,
            "transport": self.transport,
            "video": self.video,
            "dashboard": {},
            "wifi_ap": self.wifi_ap,
            "advanced": self.advanced,
            "logging": self.logging,
        }
        for name, obj in section_map.items():
            if isinstance(obj, dict):
                result[name] = obj
            else:
                result[name] = asdict(obj)
        return result
