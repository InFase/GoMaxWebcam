"""
config.py — Settings management for GoPro Bridge

Loads/saves configuration from %APPDATA%/GoProBridge/config.json.
All timing and recovery parameters are user-tunable via config.json
while the GUI stays simple.
"""

import json
import logging
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict

# Use a module-level logger under the gopro_bridge namespace.
# NOTE: We use logging.getLogger directly here because config.py is loaded
# before setup_logger() runs — the logger inherits handlers once setup completes.
_log = logging.getLogger("gopro_bridge.config")


def _appdata_dir() -> Path:
    """Return the app data directory: %APPDATA%/GoProBridge/"""
    base = os.environ.get("APPDATA", os.path.expanduser("~"))
    return Path(base) / "GoProBridge"


def _log_dir() -> Path:
    """Return the log directory: %APPDATA%/GoProBridge/logs/"""
    return _appdata_dir() / "logs"


@dataclass
class Config:
    """All app settings with sensible defaults.

    Power users can edit config.json directly to tune timing/recovery params.
    The GUI only exposes a subset of these.
    """

    # --- Camera settings ---
    resolution: int = 4          # 4=1080p, 7=720p, 12=4K
    fov: int = 4                 # 0=wide, 2=narrow, 3=superview, 4=linear
    anti_flicker: int = 1        # 0=60Hz (NTSC), 1=50Hz (PAL)
    virtual_camera_name: str = "GoPro Webcam"

    # --- Startup behavior ---
    auto_start_webcam: bool = True
    auto_connect_on_launch: bool = True

    # --- Stream settings ---
    udp_port: int = 8554
    stream_width: int = 1920
    stream_height: int = 1080
    stream_fps: int = 30
    ffmpeg_path: str = "ffmpeg"

    # --- Discovery & connection timing (seconds) ---
    discovery_timeout: float = 5.0        # How long to wait for HTTP response during discovery
    discovery_retry_interval: float = 3.0 # Seconds between discovery retries
    discovery_max_retries: int = 10       # Max retries before giving up
    discovery_overall_timeout: float = 30.0  # Max time for full discovery when USB detected but IP not resolved

    # --- Keep-alive & health ---
    keepalive_interval: float = 2.5       # Seconds between keep-alive pings
    health_check_interval: float = 5.0    # Seconds between connection health checks

    # --- Recovery timing ---
    reconnect_delay: float = 2.0          # Seconds to wait before reconnect attempt
    reconnect_max_retries: int = 0        # 0 = infinite retries
    ncm_adapter_wait: float = 5.0         # Seconds to wait for NCM network adapter after USB detection
    idle_reset_delay: float = 1.0         # Seconds to wait during IDLE workaround
    stream_startup_timeout: float = 10.0  # Max seconds to wait for first frame after ffmpeg restart
    ffmpeg_port_release_delay: float = 0.5  # Seconds to wait after stopping ffmpeg for UDP port release

    # --- Logging ---
    log_level: str = "INFO"
    log_max_session_files: int = 5         # Max session log files before cleanup
    log_max_total_bytes: int = 50 * 1024 * 1024  # 50 MB total cap across all logs
    log_max_file_bytes: int = 10 * 1024 * 1024   # 10 MB per individual log file

    # --- Internal (not saved) ---
    _config_path: str = field(default="", repr=False)

    @classmethod
    def load(cls) -> "Config":
        """Load config from disk, creating defaults if file doesn't exist."""
        config_path = _appdata_dir() / "config.json"
        config = cls()
        config._config_path = str(config_path)

        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    data = json.load(f)
                # Only set fields that exist on the dataclass
                for key, value in data.items():
                    if hasattr(config, key) and not key.startswith("_"):
                        setattr(config, key, value)
                _log.info("[EVENT:config] Loaded config from %s", config_path)
                _log.debug(
                    "[EVENT:config] Key settings: resolution=%s, fov=%s, "
                    "udp_port=%s, stream=%sx%s@%sfps, keepalive=%.1fs, "
                    "reconnect_delay=%.1fs, ncm_adapter_wait=%.1fs, log_level=%s",
                    config.resolution, config.fov,
                    config.udp_port, config.stream_width, config.stream_height,
                    config.stream_fps, config.keepalive_interval,
                    config.reconnect_delay, config.ncm_adapter_wait,
                    config.log_level,
                )
            except (json.JSONDecodeError, OSError) as e:
                # If config is corrupted, use defaults and overwrite
                _log.warning(
                    "[EVENT:config] Could not read config (%s), using defaults", e
                )
        else:
            _log.info("[EVENT:config] No config file found, creating with defaults at %s", config_path)

        # Always ensure directories exist
        _appdata_dir().mkdir(parents=True, exist_ok=True)
        _log_dir().mkdir(parents=True, exist_ok=True)

        # Save to create the file with any new default fields
        config.save()
        return config

    def save(self):
        """Write current config to disk."""
        config_path = Path(self._config_path) if self._config_path else _appdata_dir() / "config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict, excluding internal fields
        data = {k: v for k, v in asdict(self).items() if not k.startswith("_")}

        with open(config_path, "w") as f:
            json.dump(data, f, indent=2)
        _log.debug("[EVENT:config] Config saved to %s", config_path)

    @property
    def log_dir(self) -> Path:
        return _log_dir()

    @property
    def appdata_dir(self) -> Path:
        return _appdata_dir()
