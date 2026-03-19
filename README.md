# GoProBridge

Turn your GoPro into a high-quality USB webcam for Zoom, Teams, OBS, and NVIDIA Broadcast.

---

## What It Does

GoProBridge connects your GoPro over USB, enables its webcam mode, and streams the video feed into a virtual camera device that any application can use -- just like a regular webcam.

Key behaviors:

- **Auto-reconnect** — if the camera disconnects mid-call, GoProBridge detects it, holds a freeze-frame, and resumes streaming automatically when the camera comes back.
- **Zero-latency pipeline** — the stream goes directly from the camera to the virtual device with no intermediate recording or buffering overhead.
- **Fully offline** — no cloud, no telemetry, no account required. Everything runs locally.

---

## Supported Cameras

GoProBridge works with any GoPro that supports the Open GoPro USB webcam protocol:

- GoPro Hero 13 Black
- GoPro Hero 12 Black
- GoPro Hero 11 Black / Mini
- GoPro Hero 10 Black
- GoPro Hero 9 Black

Older models (Hero 8 and earlier) do not support USB webcam mode via the Open GoPro API.

## Requirements

- Windows 10 or Windows 11 (64-bit)
- A supported GoPro camera (see above)
- USB-C data cable (the cable that came with your GoPro works)

---

## Quick Start

1. Download the latest release zip from the [Releases](https://github.com/InFase/GoMaxWebcam/releases) page.
2. Extract the zip to any folder.
3. Run `GoProBridge.exe`.
4. On first run, a setup wizard will check for and install any missing dependencies (ffmpeg, Unity Capture virtual camera driver).
5. Connect your GoPro to your PC via USB-C. The dashboard will show the camera as active.
6. Open Zoom, Teams, OBS, or NVIDIA Broadcast and select "GoProBridge" or "Unity Video Capture" as your camera source.

---

## Features

- **Automatic camera detection** — plug in the GoPro and it is found without any manual configuration.
- **1080p at 30 fps by default**, with 720p also available. Resolution options depend on your GoPro model and firmware.
- **Freeze-frame recovery** — the last good frame is held during brief disconnects so your video feed does not go black mid-call.
- **Broad app compatibility** — tested with Zoom, Microsoft Teams, OBS Studio, and NVIDIA Broadcast.
- **Virtual camera via Unity Capture** — a low-overhead DirectShow virtual device that works with virtually any Windows application.
- **Configurable resolution, frame rate, and stream parameters** via a JSON config file.
- **Dark dashboard UI** — a lightweight system-tray-accessible window showing camera status, active port, and stream health.
- **No internet required** — GoProBridge never phones home. All processing is local.

---

## Configuration

Settings are stored in:

```
%APPDATA%\GoProBridge\config.json
```

You can edit this file directly. Changes take effect the next time GoProBridge starts. The file is created automatically on first run with sensible defaults.

Common settings include resolution, frame rate, the virtual camera device index, and reconnect retry behavior. See the comments inside the generated config file for descriptions of each option.

---

## Building from Source

**Prerequisites:** Python 3.11 or later, Git

```bash
git clone https://github.com/InFase/GoMaxWebcam.git
cd GoMaxWebcam
pip install -r requirements.txt
python src/main.py
```

**To build a standalone executable:**

```bash
pyinstaller GoProBridge.spec
```

The output will be in `dist/GoProBridge/`. The spec file handles all hidden imports and data bundling.

---

## Troubleshooting

**GoPro not found**
Confirm the USB-C cable supports data transfer (not just charging). On the GoPro, go to Connections > USB Connection and set it to "Webcam" or "GoPro Connect". Some cables are charge-only.

**Virtual camera not available in Zoom / Teams**
The Unity Capture driver may not be installed or may need to be re-registered. Run GoProBridge as Administrator once to allow the first-run wizard to install the driver. If the problem persists, manually run `register_vcam.bat` found in the installation folder.

**ffmpeg not found**
GoProBridge expects ffmpeg to be either bundled in the installation directory or available on your system PATH. The first-run wizard normally handles this. If you built from source, download a Windows ffmpeg build from [ffmpeg.org](https://ffmpeg.org/download.html) and place `ffmpeg.exe` in the project root or add it to PATH.

**Windows Camera app does not show the virtual camera**
The built-in Windows Camera app on Windows 11 uses the Media Foundation API, which cannot see DirectShow-based virtual cameras like Unity Capture or OBS Virtual Camera. This is a platform limitation that affects all DirectShow virtual cameras, not just GoProBridge. Use any other app instead -- Zoom, Teams, OBS, NVIDIA Broadcast, Google Meet, Discord, and most other video apps use DirectShow and will detect the camera normally.

**Windows Firewall prompt**
GoProBridge communicates with the GoPro over a local USB network interface (NCM/RNDIS). If Windows Firewall asks for permission, allow access on Private networks. No external network access is made.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgments

- [ffmpeg](https://ffmpeg.org/) — video capture and stream processing.
- [Unity Capture](https://github.com/schellingb/UnityCapture) — virtual DirectShow camera device.
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) — Python interface for virtual camera output.
