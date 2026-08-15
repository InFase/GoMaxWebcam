# GoMaxWebcam

**Turn your GoPro into a high-quality wireless or USB webcam.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/InFase/GoMaxWebcam/actions/workflows/ci.yml/badge.svg)](https://github.com/InFase/GoMaxWebcam/actions)
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)]()

---

## What It Does

GoMaxWebcam connects your GoPro camera to your computer and presents it as a standard virtual webcam that works with Zoom, Teams, OBS, NVIDIA Broadcast, and any other video application.

- **USB + WiFi (COHN) dual transport** with automatic failover -- if the USB cable is unplugged, GoMaxWebcam switches to WiFi without dropping your call.
- **Freeze-frame during transitions** -- the last good frame is held while transports switch, so your video never goes black.
- **Browser-based dashboard** -- control everything from a GoPro-inspired dark-theme UI. No desktop GUI framework needed.
- **Fully local** -- no cloud, no telemetry, no account. Everything runs on your machine.

---

## Supported Cameras

Any GoPro with the Open GoPro webcam protocol:

- GoPro Hero 13 Black
- GoPro Hero 12 Black
- GoPro Hero 11 Black / Mini
- GoPro Hero 10 Black
- GoPro Hero 9 Black

Older models (Hero 8 and earlier) do not support the Open GoPro USB webcam API.

---

## Quick Start

### Prerequisites

- **Python 3.12 or later**
- **GoPro Hero 12 or 13** (recommended; older models work over USB only)
- **Virtual camera driver:**
  - Windows: [Unity Capture](https://github.com/schellingb/UnityCapture) (bundled in release builds)
  - macOS: OBS Virtual Camera
  - Linux: v4l2loopback
- **USB-C data cable** (the cable that came with your GoPro works)

### Install from PyPI

```bash
pip install gomaxwebcam
gomaxwebcam
```

### Install from Source

```bash
git clone https://github.com/InFase/GoMaxWebcam.git
cd GoMaxWebcam
pip install -e .
python -m gomaxwebcam
```

GoMaxWebcam auto-detects your GoPro over USB, starts the webcam stream, and opens the dashboard in your default browser. Select **Unity Video Capture** (Windows) or the equivalent virtual camera in your video app.

### CLI Options

| Flag | Description |
|------|-------------|
| `--headless` | Run without system tray (dashboard only, binds to `0.0.0.0`) |
| `--debug` | Enable verbose debug logging |

---

## Features

### Dual Transport with Automatic Failover

GoMaxWebcam tries transports in priority order (configurable):

1. **USB** -- lowest latency, most reliable when physically connected.
2. **COHN (Camera on the Home Network)** -- WiFi streaming via the camera's built-in HTTPS server. Requires one-time BLE provisioning.
3. **WiFi AP** -- direct WiFi connection to the camera's own access point.

If the active transport drops, GoMaxWebcam automatically fails over to the next available transport. When a higher-priority transport reappears (e.g., USB cable plugged back in), it fails back automatically.

### Freeze-Frame

During transport transitions, the pipeline holds the last decoded frame and continues sending it to the virtual camera. Your video call sees a brief freeze instead of a black screen.

### Browser Dashboard

The dashboard opens automatically at `http://127.0.0.1:<port>/?token=<token>` and has six pages:

| Page | What It Shows |
|------|---------------|
| **Overview** | Plain-English status banner, camera info, and a live view of the actual webcam feed |
| **Camera** | Recording, modes/presets, digital zoom, power (sleep/wake/reboot), webcam stream |
| **Media** | Browse, view, download, and delete photos and videos on the camera |
| **Connect** | Wireless setup — one-tap Bluetooth, a guided Wi-Fi wizard, or manual credentials |
| **Settings** | Resolution, field of view, connection priority, auto-start |
| **Advanced** | Live diagnostics, transport state, frame-pipeline health, raw camera settings, export |

The **Live View** (also the `L` shortcut) shows the exact image GoMaxWebcam is sending to your virtual camera — i.e. what Zoom, Teams, and OBS receive.

The dashboard uses a GoPro-inspired dark theme with `#00BCE4` accent color. It is a single-page Alpine.js application served by FastAPI with SSE for real-time updates.

### System Tray

On desktop (non-headless) mode, a system tray icon provides quick access to:

- Open the dashboard
- View connection status
- Quit the application

### Single Instance

Only one copy of GoMaxWebcam runs at a time. Launching a second instance opens the existing dashboard URL and exits.

---

## COHN WiFi Setup

COHN (Camera on the Home Network) lets your GoPro stream over your home WiFi without a USB cable. Setup requires a one-time BLE (Bluetooth Low Energy) provisioning step:

1. Open the GoMaxWebcam dashboard and go to the **Setup** page.
2. Follow the wizard: it scans for your GoPro via BLE, pairs with it, and provisions COHN credentials.
3. Credentials are stored securely in your OS keyring (Windows Credential Manager / macOS Keychain / Linux Secret Service).
4. Once provisioned, COHN is available as a fallback transport automatically.

---

## Keyboard Shortcuts

Press **?** in the dashboard to see the shortcuts overlay.

### Navigation

| Key | Action |
|-----|--------|
| `H` | Go to Home page |
| `S` | Go to Settings page |
| `D` | Go to Diagnostics page |
| `U` | Go to Setup page |

### Camera Actions

| Key | Action |
|-----|--------|
| `L` | Toggle live preview |
| `P` | Pause / resume stream |
| `V` | Toggle camera visibility |
| `R` | Reconnect to camera |

### UI

| Key | Action |
|-----|--------|
| `?` | Toggle keyboard shortcuts overlay |
| `Escape` | Close overlay / preview |

---

## Configuration

Settings are stored in a TOML file at the platform config directory:

| OS | Path |
|----|------|
| Windows | `%APPDATA%\GoMaxWebcam-v2\config.toml` |
| macOS | `~/Library/Application Support/GoMaxWebcam-v2/config.toml` |
| Linux | `~/.config/GoMaxWebcam-v2/config.toml` |

The file is created automatically on first run with safe defaults. Delete it to reset all settings.

### Key Settings

```toml
[transport]
priority = ["usb", "cohn", "wifi_ap"]   # Connection order
ble_wake_mode = "always_on"              # "always_on" or "battery_saver"

[video]
resolution = "1080p"                     # "480p", "720p", "1080p"
fov = "wide"                             # "wide", "linear", "narrow", "superview"

[dashboard]
auto_start = false                       # Launch on login
open_browser_on_start = true             # Open dashboard automatically

[advanced]
udp_port = 8554                          # GoPro stream port
frame_timeout_seconds = 5                # Seconds before freeze-frame
keepalive_interval = 2.5                 # Keep-alive ping interval
```

COHN credentials (passwords, CA certificates) are stored in the OS keyring, not in the config file.

See the full annotated default config in [`src/gomaxwebcam/config.py`](src/gomaxwebcam/config.py).

---

## Building Standalone Executables

GoMaxWebcam includes a cross-platform PyInstaller spec file.

```bash
pip install pyinstaller
pyinstaller gomaxwebcam.spec --noconfirm
```

Output goes to `dist/GoMaxWebcam/`. The spec handles:

- Dashboard static assets (HTML/CSS/JS)
- pyvirtualcam native backends per platform
- BLE (bleak) data files
- Unity Capture DLLs on Windows (if present in `UnityCapture/`)
- macOS `.app` bundle with Bluetooth and network usage descriptions

---

## Development

### Setup

```bash
git clone https://github.com/InFase/GoMaxWebcam.git
cd GoMaxWebcam
pip install -e ".[dev]"
```

### Run Tests

Tests run inside a Windows Job Object with a 2 GB RAM cap to prevent runaway allocations from crashing the machine:

```bash
python run_tests.py                         # Default: 2 GB limit
python run_tests.py --limit-mb 1500         # Custom limit
python run_tests.py -- -k test_frame        # Pass args to pytest
python run_tests.py --force-run             # Run without GoPro connected
```

On non-Windows platforms, pytest runs normally without the Job Object wrapper.

Tests are in `tests/v2/` and use `pytest-asyncio`. Hardware-dependent tests are marked `@pytest.mark.hardware` and skipped when no GoPro is connected.

### Lint

```bash
ruff check src/gomaxwebcam/
```

Ruff is configured for Python 3.12, line length 100, with `E`, `F`, `I`, `N`, `W` rules enabled.

### CI

GitHub Actions runs on every push to `v2-rewrite` and `main`:

- Lint with ruff
- Tests on Python 3.12 and 3.13 across Ubuntu, macOS, and Windows
- Tagged releases build standalone executables for all three platforms

---

## Architecture

GoMaxWebcam v2 is built on an async-first architecture with clear separation of concerns:

```
                    +-----------+
                    | Dashboard |  (FastAPI + Alpine.js SPA)
                    | SSE push  |
                    +-----+-----+
                          |
                    +-----+-----+
                    |Orchestrator|  Lifecycle & coordination
                    +-----+-----+
                          |
              +-----------+-----------+
              |                       |
     +--------+--------+    +--------+--------+
     |TransportManager  |    |  FramePipeline  |
     | USB / COHN /     |    |  PyAV decode    |
     | WiFi AP failover |    |  freeze-frame   |
     +--------+---------+    |  BGR24 output   |
              |              +--------+--------+
              |                       |
     +--------+---------+   +--------+--------+
     |   EventBus       |   | VirtualCamera   |
     | async pub/sub    |   | pyvirtualcam    |
     +------------------+   +-----------------+
```

**Key modules:**

| Module | Purpose |
|--------|---------|
| `__main__` | CLI, single-instance lock, server bootstrap |
| `orchestrator` | App lifecycle, dashboard creation, graceful shutdown |
| `transport_manager` | Priority-based transport selection, failover, fail-back |
| `transport/usb` | USB webcam via Open GoPro SDK |
| `transport/cohn` | COHN (WiFi HTTPS) webcam transport |
| `transport/wifi_ap` | Direct WiFi AP transport |
| `pipeline/frame_pipeline` | Frame decoding (PyAV primary, ffmpeg fallback), freeze-frame |
| `pipeline/virtual_camera_sink` | pyvirtualcam output (1080p 30fps BGR24) |
| `events` | Async EventBus for decoupled component communication |
| `config` | TOML config with validation, keyring credential storage |
| `dashboard/app` | FastAPI routes, SSE streaming, static file serving |
| `ble/` | BLE scanner, GATT client, COHN provisioning |
| `tray` | pystray system tray icon |

---

## License

MIT -- see [LICENSE](LICENSE).

---

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on setting up the development environment, running tests, and submitting pull requests.

---

## Acknowledgments

- [Open GoPro](https://gopro.github.io/OpenGoPro/) -- GoPro's open API for camera control
- [PyAV](https://github.com/PyAV-Org/PyAV) -- Python bindings for ffmpeg libraries
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) -- virtual camera output
- [Unity Capture](https://github.com/schellingb/UnityCapture) -- DirectShow virtual camera driver (Windows)
- [FastAPI](https://fastapi.tiangolo.com/) -- async web framework for the dashboard
- [Alpine.js](https://alpinejs.dev/) -- lightweight JS framework for the dashboard SPA
- [bleak](https://github.com/hbldh/bleak) -- cross-platform BLE library
