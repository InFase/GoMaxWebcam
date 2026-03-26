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
  [dashboard] - auto_start (port is always dynamic)
  [wifi_ap]   - AP name/password (plaintext, not sensitive)
  [advanced]  - Stream port, timeouts, backoff, health check tuning
  [logging]   - Debug flag, log retention, max log size

COHN credentials (Basic Auth password, WiFi password) are stored via
keyring, not in the TOML file.  COHN Root CA cert is stored as a PEM
file alongside config.toml.

Valid values reference
----------------------
transport.priority entries : "usb", "cohn", "wifi_ap"
transport.ble_wake_mode    : "always_on", "battery_saver"
video.resolution           : "480p", "720p", "1080p"
video.fov                  : "wide", "linear", "narrow", "superview"

All settings have safe out-of-the-box defaults — you never need to edit
the config file to get started.  Delete the file to restore all defaults.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar, NamedTuple, Optional

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
# Validation result type
# ---------------------------------------------------------------------------

class ConfigValidationIssue(NamedTuple):
    """A single validation problem found in the config.

    Attributes:
        field:   Dotted field path, e.g. ``"advanced.udp_port"``.
        message: Plain-English explanation a non-technical user can act on.
        hint:    Optional suggestion for how to fix the problem.
    """
    field: str
    message: str
    hint: str = ""


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
    # Ordered list of transports to try (first = highest priority).
    # Valid entries: "usb", "cohn", "wifi_ap"
    priority: list[str] = field(default_factory=lambda: ["usb", "cohn", "wifi_ap"])
    # BLE power management mode.
    # "always_on"     — BLE adapter stays active for instant reconnect (recommended)
    # "battery_saver" — BLE adapter sleeps between connections (reduces PC power draw)
    ble_wake_mode: str = "always_on"

    # Maps lowercase TOML names to the uppercase names used by TransportManager.
    _MANAGER_NAMES: ClassVar[dict[str, str]] = {
        "usb": "USB", "cohn": "COHN", "wifi_ap": "WIFI_AP",
    }

    def priority_for_manager(self) -> list[str]:
        """Return priority list using TransportManager uppercase names.

        Converts lowercase TOML config names (e.g. "usb") to the uppercase
        names registered with TransportManager (e.g. "USB").

        Returns:
            List of uppercase transport names in priority order.
            Unknown entries are converted to uppercase as a fallback.
        """
        return [self._MANAGER_NAMES.get(t.lower(), t.upper()) for t in self.priority]


@dataclass
class VideoSection:
    # Webcam output resolution: "480p", "720p", "1080p"
    resolution: str = "1080p"
    # Field of view: "wide", "linear", "narrow", "superview"
    fov: str = "wide"

    # --- Conversion helpers (ClassVar = not serialised to TOML) ---

    # Maps human-readable resolution strings to Open GoPro webcam API integer codes.
    # These match WebcamResolution enum values in open-gopro SDK.
    _RESOLUTION_CODES: ClassVar[dict[str, int]] = {
        "480p": 4, "720p": 7, "1080p": 12,
    }

    # Maps human-readable FOV strings to Open GoPro webcam API integer codes.
    # These match WebcamFOV enum values in open-gopro SDK.
    _FOV_CODES: ClassVar[dict[str, int]] = {
        "wide": 0, "narrow": 2, "superview": 3, "linear": 4,
    }

    def resolution_code(self) -> int:
        """Return the Open GoPro integer code for the configured resolution.

        Returns:
            Integer webcam resolution code (e.g. 12 for "1080p").
            Defaults to 1080p (12) if the configured string is unrecognised.
        """
        return self._RESOLUTION_CODES.get(self.resolution, 12)

    def fov_code(self) -> int:
        """Return the Open GoPro integer code for the configured FOV.

        Returns:
            Integer webcam FOV code (e.g. 0 for "wide").
            Defaults to wide (0) if the configured string is unrecognised.
        """
        return self._FOV_CODES.get(self.fov, 0)


@dataclass
class DashboardSection:
    """Dashboard persistent settings (port is always assigned dynamically)."""
    # Launch GoMaxWebcam automatically when the user logs in.
    auto_start: bool = False
    # Open the dashboard browser window automatically on startup.
    open_browser_on_start: bool = True


@dataclass
class WifiApSection:
    # GoPro WiFi AP SSID (e.g. "GP24500123"). Leave empty to auto-scan.
    ap_name: str = ""
    # GoPro WiFi AP password — stored as plain text here because it is
    # displayed on the camera screen and is not considered a secret.
    # Sensitive credentials (COHN password) are stored via OS keyring.
    ap_password: str = ""


@dataclass
class AdvancedSection:
    # UDP port the GoPro sends the MPEG-TS preview stream to.
    # Change this only if port 8554 is in use by another application.
    udp_port: int = 8554

    # Seconds without a frame before the pipeline enters freeze-frame mode.
    # Increase to 8-10 on slow WiFi connections where jitter is common.
    frame_timeout_seconds: int = 5

    # How often to send keep-alive pings to the camera (seconds).
    # Lower values detect disconnects faster but use more network bandwidth.
    keepalive_interval: float = 2.5

    # Number of consecutive failed keep-alive pings before the USB transport
    # declares the camera disconnected and triggers failover.
    max_consecutive_failures: int = 3

    # Exponential back-off base (seconds) for transport reconnect retries.
    backoff_base_seconds: int = 2

    # Maximum back-off cap (seconds) — retries never wait longer than this.
    backoff_cap_seconds: int = 60

    # How often to run the transport health-check loop (seconds).
    health_check_interval: float = 5.0

    # How often to poll the USB bus for a reconnected GoPro when running
    # on COHN/WiFi fallback (seconds). Lower = faster fail-back to USB.
    usb_poll_interval: float = 5.0


@dataclass
class LoggingSection:
    # Enable verbose debug logging (writes significantly more log data).
    debug: bool = False
    # Delete log files older than this many days.
    log_retention_days: int = 7
    # Roll over the log file when it exceeds this size (megabytes).
    log_max_size_mb: int = 50


# ---------------------------------------------------------------------------
# Default TOML content (with comments for first-run creation)
# ---------------------------------------------------------------------------

_DEFAULT_TOML = """\
# GoMaxWebcam v2 configuration
# Edit values here to customise behaviour.  Unknown keys are ignored.
# Delete this file to reset everything to factory defaults.

# ---------------------------------------------------------------------------
# Camera identity (populated automatically on first successful connect)
# ---------------------------------------------------------------------------
[camera]
ble_address  = ""  # BLE MAC address, e.g. "AA:BB:CC:DD:EE:FF"
camera_name  = ""  # Friendly name shown in the dashboard, e.g. "GoPro 1234"
camera_serial = "" # Full serial number from the camera
last_known_ip = "" # IP address from the last successful COHN/WiFi connection
model         = "" # Camera model string, e.g. "HERO13 Black"

# ---------------------------------------------------------------------------
# Transport selection and BLE behaviour
# ---------------------------------------------------------------------------
[transport]
# Priority order to try when connecting.
# Valid values (lowercase): "usb", "cohn", "wifi_ap"
# USB is fastest and most reliable when the camera is plugged in.
priority = ["usb", "cohn", "wifi_ap"]

# BLE power mode used for waking the camera and provisioning COHN.
# "always_on"     — adapter stays active, instant reconnect (recommended)
# "battery_saver" — adapter sleeps between uses, saves a little PC power
ble_wake_mode = "always_on"

# ---------------------------------------------------------------------------
# Video output settings
# ---------------------------------------------------------------------------
[video]
# Webcam resolution sent to virtual camera.  Valid: "480p", "720p", "1080p"
resolution = "1080p"

# Field of view mode.  Valid: "wide", "linear", "narrow", "superview"
fov = "wide"

# ---------------------------------------------------------------------------
# Dashboard behaviour
# ---------------------------------------------------------------------------
[dashboard]
# Start GoMaxWebcam automatically when you log in (Windows Task Scheduler /
# macOS LaunchAgent / Linux systemd user unit).
auto_start = false

# Open the dashboard in your default browser when the app starts.
open_browser_on_start = true

# ---------------------------------------------------------------------------
# GoPro WiFi AP credentials (only needed for WiFi AP transport)
# ---------------------------------------------------------------------------
[wifi_ap]
# GoPro WiFi AP SSID, e.g. "GP24500123".  Leave empty to auto-scan.
ap_name = ""
# GoPro WiFi AP password.  This is visible on the camera screen and is NOT
# treated as a secret — it is stored here in plain text.
ap_password = ""

# ---------------------------------------------------------------------------
# Advanced tuning — change only if you know what you are doing
# ---------------------------------------------------------------------------
[advanced]
# UDP port the GoPro streams MPEG-TS to.
# Only change this if port 8554 is already in use on your machine.
udp_port = 8554

# Seconds of silence before the pipeline switches to freeze-frame mode.
# Increase to 8–10 on slow or congested WiFi to avoid false freeze triggers.
frame_timeout_seconds = 5

# Seconds between keep-alive pings sent to the camera.
keepalive_interval = 2.5

# USB: consecutive failed keep-alive pings before declaring the camera lost.
max_consecutive_failures = 3

# Reconnect back-off: first retry waits backoff_base_seconds, each subsequent
# retry doubles up to backoff_cap_seconds.
backoff_base_seconds = 2
backoff_cap_seconds  = 60

# Health-check loop interval (seconds).
health_check_interval = 5.0

# How often to check whether a USB GoPro has reappeared while running on
# COHN or WiFi fallback (seconds).  Lower = faster fail-back to USB.
usb_poll_interval = 5.0

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
[logging]
# Set to true to enable verbose DEBUG logging.
debug = false
# Delete log files older than this many days.
log_retention_days = 7
# Roll the log file when it exceeds this size in megabytes.
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
        except Exception as exc:
            log.warning(
                "Config file at %s could not be read (%s). "
                "All settings have been reset to safe defaults. "
                "To fix: delete %s and restart — a fresh default file will be created.",
                cfg._config_path,
                exc,
                cfg._config_path,
            )
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

    def reset_to_defaults(self) -> None:
        """Reset all sections to their built-in defaults and save.

        Rewrites the config file with the annotated default TOML template so
        users get helpful comments back.  Camera identity fields (BLE address,
        name, serial) are preserved so the camera does not need to be
        re-discovered after a settings reset.

        Useful after a bad edit leaves the config in an unusable state.
        """
        preserved_camera = self.camera
        # Re-initialise all sections to defaults
        self.transport = TransportSection()
        self.video = VideoSection()
        self.dashboard = DashboardSection()
        self.wifi_ap = WifiApSection()
        self.advanced = AdvancedSection()
        self.logging = LoggingSection()
        # Restore camera identity so the user doesn't lose pairing info
        self.camera = preserved_camera
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._config_path.write_text(_DEFAULT_TOML, encoding="utf-8")
        log.info("Config reset to defaults at %s", self._config_path)

    def validate(self) -> list[ConfigValidationIssue]:
        """Check the current config for problems and return friendly messages.

        Validates each section and runs cross-field checks.  Returns a list of
        :class:`ConfigValidationIssue` items — one per problem found.  An empty
        list means the config is valid.

        This is intended for display in the dashboard Settings page so users
        see plain-English guidance instead of raw exception traces.

        Returns:
            List of issues (may be empty).
        """
        issues: list[ConfigValidationIssue] = []

        # --- [transport] ---
        _VALID_TRANSPORTS = {"usb", "cohn", "wifi_ap"}
        _VALID_BLE_WAKE_MODES = {"always_on", "battery_saver"}

        if not self.transport.priority:
            issues.append(ConfigValidationIssue(
                field="transport.priority",
                message="No connection methods are enabled — GoMaxWebcam won't be able to reach your camera.",
                hint='Set transport.priority to at least ["usb"] in config.toml.',
            ))
        else:
            unknown = [t for t in self.transport.priority if t not in _VALID_TRANSPORTS]
            if unknown:
                issues.append(ConfigValidationIssue(
                    field="transport.priority",
                    message=f'Unknown connection method(s): {unknown!r}.',
                    hint='Valid methods are "usb", "cohn", and "wifi_ap".',
                ))

        if self.transport.ble_wake_mode not in _VALID_BLE_WAKE_MODES:
            issues.append(ConfigValidationIssue(
                field="transport.ble_wake_mode",
                message=f'BLE wake mode "{self.transport.ble_wake_mode}" is not recognised.',
                hint='Use "always_on" (recommended) or "battery_saver".',
            ))

        # --- [video] ---
        _VALID_RESOLUTIONS = {"480p", "720p", "1080p"}
        _VALID_FOVS = {"wide", "linear", "narrow", "superview"}

        if self.video.resolution not in _VALID_RESOLUTIONS:
            issues.append(ConfigValidationIssue(
                field="video.resolution",
                message=f'Resolution "{self.video.resolution}" is not supported.',
                hint='Valid options are "480p", "720p", or "1080p".',
            ))

        if self.video.fov not in _VALID_FOVS:
            issues.append(ConfigValidationIssue(
                field="video.fov",
                message=f'Field of view "{self.video.fov}" is not recognised.',
                hint='Valid options are "wide", "linear", "narrow", or "superview".',
            ))

        # --- [advanced] ---
        adv = self.advanced

        if not (1024 <= adv.udp_port <= 65535):
            issues.append(ConfigValidationIssue(
                field="advanced.udp_port",
                message=f"UDP port {adv.udp_port} is outside the allowed range (1024–65535).",
                hint="Leave it at the default 8554 unless another app is using that port.",
            ))

        if adv.frame_timeout_seconds <= 0:
            issues.append(ConfigValidationIssue(
                field="advanced.frame_timeout_seconds",
                message="Frame timeout must be greater than zero.",
                hint="The default of 5 seconds works well for most connections.",
            ))

        if adv.keepalive_interval <= 0:
            issues.append(ConfigValidationIssue(
                field="advanced.keepalive_interval",
                message="Keep-alive interval must be greater than zero.",
                hint="The default of 2.5 seconds is recommended.",
            ))

        if adv.max_consecutive_failures <= 0:
            issues.append(ConfigValidationIssue(
                field="advanced.max_consecutive_failures",
                message="max_consecutive_failures must be at least 1.",
                hint="The default of 3 gives a good balance of speed and stability.",
            ))

        if adv.backoff_base_seconds <= 0:
            issues.append(ConfigValidationIssue(
                field="advanced.backoff_base_seconds",
                message="Reconnect back-off base must be greater than zero.",
                hint="The default of 2 seconds is recommended.",
            ))

        if adv.backoff_cap_seconds <= 0:
            issues.append(ConfigValidationIssue(
                field="advanced.backoff_cap_seconds",
                message="Reconnect back-off cap must be greater than zero.",
                hint="The default of 60 seconds is recommended.",
            ))

        # Cross-field: base must be <= cap
        if (adv.backoff_base_seconds > 0 and adv.backoff_cap_seconds > 0
                and adv.backoff_base_seconds > adv.backoff_cap_seconds):
            issues.append(ConfigValidationIssue(
                field="advanced.backoff_base_seconds",
                message=(
                    f"Reconnect back-off base ({adv.backoff_base_seconds}s) is larger than "
                    f"the cap ({adv.backoff_cap_seconds}s) — retries will never back off correctly."
                ),
                hint="Set backoff_base_seconds to a value smaller than backoff_cap_seconds.",
            ))

        if adv.health_check_interval <= 0:
            issues.append(ConfigValidationIssue(
                field="advanced.health_check_interval",
                message="Health check interval must be greater than zero.",
                hint="The default of 5.0 seconds is recommended.",
            ))

        if adv.usb_poll_interval <= 0:
            issues.append(ConfigValidationIssue(
                field="advanced.usb_poll_interval",
                message="USB poll interval must be greater than zero.",
                hint="The default of 5.0 seconds is recommended.",
            ))

        # --- [logging] ---
        if self.logging.log_retention_days <= 0:
            issues.append(ConfigValidationIssue(
                field="logging.log_retention_days",
                message="Log retention must be at least 1 day.",
                hint="The default of 7 days keeps a week of history without wasting disk space.",
            ))

        if self.logging.log_max_size_mb <= 0:
            issues.append(ConfigValidationIssue(
                field="logging.log_max_size_mb",
                message="Log max size must be at least 1 MB.",
                hint="The default of 50 MB is usually plenty.",
            ))

        return issues

    @property
    def is_valid(self) -> bool:
        """Return True if the config passes all validation checks."""
        return len(self.validate()) == 0

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
        """Apply a parsed TOML dict onto the section dataclasses.

        Unknown keys are silently ignored.  Invalid values (wrong type or
        out-of-range) are logged and replaced with the section's default.
        """
        section_map: dict[str, Any] = {
            "camera": self.camera,
            "transport": self.transport,
            "video": self.video,
            "dashboard": self.dashboard,
            "wifi_ap": self.wifi_ap,
            "advanced": self.advanced,
            "logging": self.logging,
        }
        for section_name, section_obj in section_map.items():
            toml_section = data.get(section_name, {})
            if not isinstance(toml_section, dict):
                continue
            for f in fields(section_obj):
                if f.name not in toml_section:
                    continue
                raw = toml_section[f.name]
                try:
                    validated = self._validate_field(section_name, f.name, raw, f.default)
                    setattr(section_obj, f.name, validated)
                except Exception as exc:
                    log.warning(
                        "Ignoring invalid value for [%s].%s = %r — %s. "
                        "The built-in default will be used instead.",
                        section_name, f.name, raw, exc,
                    )

    def _validate_field(
        self,
        section: str,
        name: str,
        value: Any,
        default: Any,
    ) -> Any:
        """Validate and coerce a single config field value.

        Applies type coercion and range/enum checks.  Returns the validated
        value or raises ValueError / TypeError to signal rejection.

        Args:
            section: TOML section name (e.g. "transport").
            name:    Field name (e.g. "ble_wake_mode").
            value:   Raw value from TOML parsing.
            default: Dataclass field default (used as fallback reference).

        Returns:
            Validated (and possibly coerced) value.

        Raises:
            ValueError: Value is outside the allowed set.
            TypeError:  Value cannot be coerced to the expected type.
        """
        # --- [transport] ---
        if section == "transport":
            if name == "ble_wake_mode":
                _VALID_BLE_WAKE_MODES = {"always_on", "battery_saver"}
                if value not in _VALID_BLE_WAKE_MODES:
                    log.warning(
                        "[transport].ble_wake_mode=%r is not valid "
                        "(valid: %s); using default 'always_on'",
                        value, sorted(_VALID_BLE_WAKE_MODES),
                    )
                    return "always_on"
            if name == "priority":
                _VALID_TRANSPORTS = {"usb", "cohn", "wifi_ap"}
                if not isinstance(value, list):
                    raise TypeError(f"priority must be a list, got {type(value).__name__}")
                cleaned = [str(v).lower().strip() for v in value]
                valid = [t for t in cleaned if t in _VALID_TRANSPORTS]
                invalid = [t for t in cleaned if t not in _VALID_TRANSPORTS]
                if invalid:
                    log.warning(
                        "[transport].priority contains unknown entries %r "
                        "(valid: %s) — they will be ignored",
                        invalid, sorted(_VALID_TRANSPORTS),
                    )
                return valid or list(default)

        # --- [video] ---
        if section == "video":
            if name == "resolution":
                _VALID_RESOLUTIONS = {"480p", "720p", "1080p"}
                if value not in _VALID_RESOLUTIONS:
                    log.warning(
                        "[video].resolution=%r is not valid "
                        "(valid: %s); using default '1080p'",
                        value, sorted(_VALID_RESOLUTIONS),
                    )
                    return "1080p"
            if name == "fov":
                _VALID_FOVS = {"wide", "linear", "narrow", "superview"}
                if value not in _VALID_FOVS:
                    log.warning(
                        "[video].fov=%r is not valid "
                        "(valid: %s); using default 'wide'",
                        value, sorted(_VALID_FOVS),
                    )
                    return "wide"

        # --- [advanced] numeric range guards ---
        if section == "advanced":
            if name == "udp_port":
                port = int(value)
                if not (1024 <= port <= 65535):
                    raise ValueError(f"udp_port {port} out of range [1024, 65535]")
                return port
            if name in ("frame_timeout_seconds", "health_check_interval",
                        "keepalive_interval", "usb_poll_interval"):
                v = float(value)
                if v <= 0:
                    raise ValueError(f"{name} must be positive, got {v}")
                return v
            if name in ("max_consecutive_failures", "backoff_base_seconds",
                        "backoff_cap_seconds"):
                v = int(value)
                if v <= 0:
                    raise ValueError(f"{name} must be positive, got {v}")
                return v

        # --- [logging] numeric range guards ---
        if section == "logging":
            if name in ("log_retention_days", "log_max_size_mb"):
                v = int(value)
                if v <= 0:
                    raise ValueError(f"{name} must be positive, got {v}")
                return v

        # Default: return as-is (TOML already provides basic typing)
        return value

    def _to_dict(self) -> dict[str, Any]:
        """Serialize all sections to a plain dict for TOML writing."""
        from dataclasses import asdict

        result: dict[str, Any] = {}
        section_map: dict[str, Any] = {
            "camera": self.camera,
            "transport": self.transport,
            "video": self.video,
            "dashboard": self.dashboard,
            "wifi_ap": self.wifi_ap,
            "advanced": self.advanced,
            "logging": self.logging,
        }
        for name, obj in section_map.items():
            result[name] = asdict(obj)
        return result
