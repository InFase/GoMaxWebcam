# GoPro Bridge — Open Source GoPro Webcam Replacement

## GOAL

Build a Windows desktop application called **GoPro Bridge** that replaces the official GoPro Webcam Utility. It connects to a GoPro camera over USB, puts it into webcam mode using the official Open GoPro API, captures the video stream, and exposes it as a virtual camera that NVIDIA Broadcast (or any other app) can use.

**The key differentiator from GoPro's official app:** This app gives the user full control over camera visibility. The user can choose whether the virtual camera is registered publicly (visible to all apps) or kept private (only accessible to whitelisted apps like NVIDIA Broadcast). This solves the problem where multiple virtual cameras confuse proctoring/exam software.

## THE PROBLEM THIS SOLVES

The user has:
- GoPro Hero 12+ connected via USB-C through a hub
- NVIDIA Broadcast configured to use GoPro as input, outputting a processed feed
- The official GoPro Webcam Utility registers a public virtual camera called "GoProWebcam"
- This causes exam/interview proctoring apps to see BOTH the GoPro and NVIDIA Broadcast cameras
- The proctoring app tries to handshake with both, causing the GoPro to power-cycle repeatedly
- Neither camera feed works during the exam

## HOW THE GOPRO WEBCAM PROTOCOL WORKS

This is documented in GoPro's official Open GoPro API (MIT licensed): https://gopro.github.io/OpenGoPro/

### USB Connection
1. GoPro connects via USB-C and creates a virtual network interface using NCM (Network Control Model) protocol
2. The GoPro gets an IP address on this virtual network — typically `172.2x.1xx.51` (varies)
3. The camera runs an HTTP server on port 8080 at this IP
4. All commands are sent as HTTP GET requests to `http://{camera_ip}:8080/gopro/...`

### Webcam Mode Sequence
```
1. Connect USB cable → GoPro appears as a network adapter (shows as "enx..." or "GoPro" in network adapters)
2. Discover GoPro's IP address on the USB network interface
3. GET http://{ip}:8080/gopro/webcam/status    → check current webcam state
4. GET http://{ip}:8080/gopro/webcam/start     → put camera into webcam mode
5. Camera starts streaming on UDP port 8554     → TS (MPEG Transport Stream) format
6. Read the UDP stream and decode video frames
7. GET http://{ip}:8080/gopro/webcam/stop      → stop webcam mode when done
```

### Webcam Start Parameters (optional query params)
- `res`: Resolution — 4 (1080p), 7 (720p), 12 (4K)
- `fov`: Field of view — 0 (wide), 2 (narrow), 3 (superview), 4 (linear)

Example: `GET http://{ip}:8080/gopro/webcam/start?res=4&fov=4` starts 1080p linear

### Webcam Status Response
```json
{
    "error": 0,
    "status": 0    // 0=OFF, 1=IDLE, 2=READY, 3=STREAMING, 4=...
}
```

### Known Quirk (from GoPro FAQ)
After a new USB connection, the webcam status may incorrectly report IDLE instead of OFF. The workaround is to call webcam/start followed by webcam/stop after connecting USB to reset the state machine.

### Stream Details
- Protocol: UDP on port 8554
- Format: MPEG-TS (Transport Stream)
- Codec: H.264 (AVC)
- The stream is sent TO the host machine (the GoPro pushes it)
- Can be read with ffmpeg: `ffmpeg -f mpegts -i udp://@:8554 ...`

## ARCHITECTURE

```
┌──────────────┐     USB-C      ┌─────────────────┐
│   GoPro      │◄──────────────►│  GoPro Bridge    │
│   Hero 12+   │   NCM network  │  (our app)       │
│              │                │                   │
│  HTTP server │◄── commands ───│  1. Discover IP   │
│  :8080       │                │  2. Start webcam  │
│              │                │  3. Read UDP stream│
│  UDP stream  │──── video ────►│  4. Decode frames │
│  :8554       │                │  5. → Virtual Cam │
└──────────────┘                └───────┬───────────┘
                                        │
                                        │ video frames
                                        ▼
                               ┌─────────────────┐
                               │  Virtual Camera  │
                               │  (pyvirtualcam)  │
                               └───────┬───────────┘
                                       │
                          ┌────────────┼────────────┐
                          ▼            ▼            ▼
                    ┌──────────┐ ┌──────────┐ ┌──────────┐
                    │  NVIDIA  │ │  Zoom    │ │  Chrome  │
                    │Broadcast │ │  Teams   │ │  etc.    │
                    └──────────┘ └──────────┘ └──────────┘
```

## TECHNOLOGY STACK

### Python (primary language)
The user is comfortable with Python and can read/modify the code. Use Python 3.11+.

### Key Libraries
1. **open-gopro** — GoPro's official Python SDK (pip install open-gopro)
   - Handles USB connection, device discovery, HTTP commands
   - https://gopro.github.io/OpenGoPro/python_sdk/
   - BUT: we may not need the full SDK. Simple HTTP requests via `requests` library may be simpler and more transparent. Evaluate both approaches.

2. **pyvirtualcam** — creates a virtual webcam that apps can see (pip install pyvirtualcam)
   - Cross-platform virtual camera output
   - Uses OBS Virtual Camera backend on Windows
   - https://github.com/letmaik/pyvirtualcam
   - NOTE: requires OBS VirtualCam plugin to be installed, OR use the Unity Capture backend
   - IMPORTANT: Research which backend works best on Windows without requiring OBS installed

3. **ffmpeg (via subprocess or ffmpeg-python)** — decode the MPEG-TS UDP stream from GoPro
   - `ffmpeg-python` for pythonic interface, OR
   - `subprocess.Popen` with raw ffmpeg for more control
   - ffmpeg reads from `udp://@:8554` and outputs raw frames

4. **OpenCV (cv2)** — alternative to ffmpeg for stream reading and frame manipulation
   - Can read from UDP stream directly: `cv2.VideoCapture("udp://@:8554")`
   - Provides frame manipulation if needed

5. **tkinter or PyQt** — GUI for settings, status display, camera preview
   - tkinter is simpler and built-in
   - PyQt is more polished

6. **psutil + netifaces** — to discover GoPro's USB network interface and IP address

### Virtual Camera Backend — Unity Capture (NO OBS needed)
pyvirtualcam needs a "backend" — a virtual camera driver installed on the system. We will use **Unity Capture**:
- Download from: https://github.com/schellingb/UnityCapture/releases
- It's a standalone lightweight virtual camera driver — does NOT need OBS or Unity installed
- Supports custom device names (can show as "GoPro Bridge" instead of generic names)
- pyvirtualcam uses it via: `pyvirtualcam.Camera(..., backend='unitycapture')`
- Supports RGBA frames and custom resolutions

If Unity Capture causes issues, the fallback options are:
- **OBS VirtualCam DLL only** (without OBS Studio): download obs-virtual-cam zip from https://github.com/CatxFish/obs-virtual-cam/releases and register the DLLs with `regsvr32`
- **softcam** (C++ library, https://github.com/tshino/softcam): creates a DirectShow filter from scratch — full control but requires C++ compilation

## PROJECT STRUCTURE

```
gopro-bridge/
├── src/
│   ├── main.py                 # Entry point, launches GUI
│   ├── gopro_connection.py     # USB discovery, HTTP API communication
│   ├── stream_reader.py        # Read and decode UDP MPEG-TS stream
│   ├── virtual_camera.py       # Output frames to virtual camera
│   ├── gui.py                  # Main application window
│   ├── camera_visibility.py    # Registry management for camera visibility
│   ├── config.py               # Settings management (resolution, FOV, etc.)
│   └── utils.py                # Logging, helpers
│
├── assets/
│   └── icon.ico                # App icon
│
├── requirements.txt
├── setup.py                    # For building distributable
├── README.md
├── ARCHITECTURE.md
└── BUILD.md
```

## DETAILED COMPONENT SPECIFICATIONS

### 1. gopro_connection.py — GoPro Discovery and Control

**USB Network Interface Discovery:**
- When GoPro connects via USB, Windows creates a new network adapter
- The adapter typically has a name containing "GoPro" or uses a specific USB vendor ID
- Use `psutil.net_if_addrs()` or `netifaces` to list network interfaces
- Filter for the GoPro interface by checking:
  - Interface name patterns
  - IP address range (GoPro typically assigns 172.2x.1xx.xx)
  - Or use `ipconfig /all` parsing to find the GoPro RNDIS/NCM adapter
- Alternative: Use mDNS discovery — GoPro registers `_gopro-web` service

**HTTP API wrapper:**
```python
class GoProConnection:
    def __init__(self):
        self.ip = None
        self.base_url = None
    
    def discover(self) -> bool:
        """Find GoPro on USB network interface, set self.ip"""
        pass
    
    def webcam_status(self) -> dict:
        """GET /gopro/webcam/status"""
        pass
    
    def webcam_start(self, resolution=4, fov=4) -> bool:
        """GET /gopro/webcam/start?res={resolution}&fov={fov}"""
        pass
    
    def webcam_stop(self) -> bool:
        """GET /gopro/webcam/stop"""
        pass
    
    def keep_alive(self):
        """Send periodic keep-alive to prevent camera sleep"""
        pass
    
    def get_camera_info(self) -> dict:
        """Get camera model, firmware version, battery level etc."""
        pass
```

**Important:** After USB connection, perform the IDLE workaround:
```python
# GoPro FAQ workaround: webcam status may be wrong after fresh USB connect
self.webcam_start()
time.sleep(1)
self.webcam_stop()
time.sleep(1)
# Now the webcam state machine is properly reset
```

### 2. stream_reader.py — Video Stream Capture

After `webcam_start()` is called, the GoPro streams MPEG-TS over UDP port 8554.

**Option A — ffmpeg subprocess (recommended for reliability):**
```python
import subprocess

class StreamReader:
    def __init__(self):
        self.process = None
        self.width = 1920
        self.height = 1080
    
    def start(self):
        """Launch ffmpeg to decode UDP stream into raw frames"""
        cmd = [
            'ffmpeg',
            '-f', 'mpegts',
            '-i', 'udp://@:8554',
            '-pix_fmt', 'rgb24',        # Output as RGB
            '-f', 'rawvideo',            # Raw frame output
            '-an',                        # No audio
            '-sn',                        # No subtitles
            '-vf', f'scale={self.width}:{self.height}',
            'pipe:1'                      # Output to stdout
        ]
        self.process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
    
    def read_frame(self) -> numpy.ndarray:
        """Read one frame from ffmpeg stdout"""
        frame_size = self.width * self.height * 3
        raw = self.process.stdout.read(frame_size)
        if len(raw) != frame_size:
            return None
        frame = numpy.frombuffer(raw, dtype=numpy.uint8)
        return frame.reshape((self.height, self.width, 3))
    
    def stop(self):
        if self.process:
            self.process.terminate()
```

**Option B — OpenCV (simpler but may have buffering issues):**
```python
cap = cv2.VideoCapture("udp://@:8554", cv2.CAP_FFMPEG)
ret, frame = cap.read()
```

Evaluate both and use whichever gives lower latency and more reliable frame delivery.

**IMPORTANT:** ffmpeg must be installed on the system and in PATH. Add this to the README as a prerequisite, or bundle ffmpeg with the app.

### 3. virtual_camera.py — Virtual Camera Output

**Using pyvirtualcam with Unity Capture backend:**
```python
import pyvirtualcam

class VirtualCamera:
    def __init__(self, width=1920, height=1080, fps=30):
        self.cam = pyvirtualcam.Camera(
            width=width, height=height, fps=fps,
            fmt=pyvirtualcam.PixelFormat.RGB,
            backend='unitycapture',
            device="GoPro Bridge"  # Custom name visible in app camera lists
        )
    
    def send_frame(self, frame: numpy.ndarray):
        self.cam.send(frame)
        self.cam.sleep_until_next_frame()
    
    def close(self):
        self.cam.close()
```

**Note:** Unity Capture allows custom device names, so our camera will show up as "GoPro Bridge" in app camera dropdowns — not some generic "OBS Virtual Camera" or "Unity Video Capture" name.

### 4. camera_visibility.py — Registry-based Camera Visibility Control

This is the piece that makes our app special. It manages the DirectShow registry to control which cameras are visible to apps.

```python
import winreg

DIRECTSHOW_64 = r"SOFTWARE\Classes\CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance"
DIRECTSHOW_32 = r"SOFTWARE\WOW6432Node\Classes\CLSID\{860BB310-5D01-11d0-BD3B-00A0C911CE86}\Instance"

class CameraVisibilityManager:
    def list_cameras(self) -> list:
        """Read all registered DirectShow cameras from registry"""
        pass
    
    def hide_camera(self, clsid: str) -> bool:
        """Remove a camera from the DirectShow Instance list (backs up first)"""
        pass
    
    def show_camera(self, clsid: str) -> bool:
        """Restore a camera to the DirectShow Instance list from backup"""
        pass
    
    def hide_gopro_official(self):
        """Specifically hide the official GoPro Webcam virtual camera
        CLSID (64-bit): {FDB60968-EC75-4CF9-BC63-7A2C7FFBF210}
        CLSID (32-bit): {506868A5-8940-4F2E-9E49-BEC693B7075D}
        """
        pass
    
    def backup_exists(self, clsid: str) -> bool:
        """Check if a backup exists for a hidden camera"""
        pass
    
    def restore_all(self):
        """Restore all hidden cameras (safety net)"""
        pass
```

Store backups at: `HKLM\SOFTWARE\CamHide\Backups\{clsid}`

This component REQUIRES admin elevation. The app should request UAC elevation on startup for registry operations.

### 5. gui.py — Application Interface

Use tkinter (simpler) or PyQt5. The GUI should have:

**Main Window:**
- Camera preview panel (shows what the GoPro sees)
- Connection status indicator (disconnected / connected / streaming)
- Start/Stop button
- Settings panel:
  - Resolution dropdown: 720p, 1080p, 4K
  - FOV dropdown: Wide, Linear, Narrow, Superview
  - Virtual camera name (editable)
- Camera visibility section:
  - List of all registered cameras with checkboxes
  - Toggle to hide/show each camera
  - "Hide official GoPro Webcam" quick button
- Battery level / camera info display
- System tray icon with:
  - Quick start/stop
  - Show/hide window
  - Exit

**Status Bar:**
- Current resolution and FPS
- Frame count
- Latency estimate

### 6. config.py — Settings Persistence

Store settings in a JSON file at `%APPDATA%\GoProBridge\config.json`:
```json
{
    "resolution": 4,
    "fov": 4,
    "auto_start_webcam": true,
    "auto_hide_gopro_official": true,
    "virtual_camera_name": "GoPro Bridge",
    "hidden_cameras": [
        "{FDB60968-EC75-4CF9-BC63-7A2C7FFBF210}"
    ],
    "ffmpeg_path": "ffmpeg",
    "log_level": "INFO"
}
```

## MAIN APPLICATION FLOW

```python
# main.py pseudocode

def main():
    # 1. Initialize config
    config = Config.load()
    
    # 2. Check prerequisites
    check_ffmpeg_installed()
    check_unitycapture_installed()
    
    # 3. Discover GoPro
    gopro = GoProConnection()
    if not gopro.discover():
        show_error("GoPro not found. Make sure it's connected via USB and powered on.")
        return
    
    # 4. Optionally hide official GoPro Webcam from registry
    if config.auto_hide_gopro_official:
        visibility = CameraVisibilityManager()
        visibility.hide_gopro_official()
    
    # 5. Start webcam mode on camera
    gopro.webcam_start(resolution=config.resolution, fov=config.fov)
    
    # 6. Start reading the stream
    reader = StreamReader()
    reader.start()
    
    # 7. Create virtual camera
    vcam = VirtualCamera(width=1920, height=1080, fps=30)
    
    # 8. Frame loop (in a separate thread)
    while running:
        frame = reader.read_frame()
        if frame is not None:
            vcam.send_frame(frame)
            gui.update_preview(frame)
    
    # 9. Cleanup on exit
    reader.stop()
    gopro.webcam_stop()
    vcam.close()
    if config.auto_hide_gopro_official:
        visibility.restore_all()  # Restore hidden cameras on exit
```

## BUILD AND DISTRIBUTION

### For development:
```bash
pip install -r requirements.txt
python src/main.py
```

### For distribution (single .exe):
Use PyInstaller:
```bash
pyinstaller --onefile --windowed --icon=assets/icon.ico --name="GoPro Bridge" src/main.py
```

Consider bundling ffmpeg.exe with the distribution using `--add-binary`.

## PREREQUISITES FOR THE USER

1. Python 3.11+ (for development) OR the bundled .exe (for end users)
2. ffmpeg installed and in PATH (download from https://ffmpeg.org/download.html)
3. Unity Capture virtual camera driver (download from https://github.com/schellingb/UnityCapture/releases — install the driver, no OBS or Unity needed)
4. GoPro Hero 12+ with firmware v01.10.00 or later
5. USB-C cable connected to the PC

## IMPORTANT NOTES FOR CLAUDE CODE

1. **Start with the simplest working version first.** Get GoPro discovery + webcam start + stream reading working in a single Python script before building the full app. A proof-of-concept that prints "got frame: 1920x1080" is the first milestone.

2. **The user has a Hero 12 Black.** Test with the Hero 12's specific USB behavior. The camera's USB vendor ID is 2672 (GoPro Inc.) — this helps with network interface discovery.

3. **ffmpeg is the most reliable stream reader.** OpenCV's UDP handling can be flaky. Default to the ffmpeg subprocess approach.

4. **Unity Capture is the virtual camera backend.** It's lightweight, doesn't need OBS, and allows custom device names. pyvirtualcam talks to it. The user must install the Unity Capture driver first (just running an installer from the GitHub releases page).

5. **Admin elevation is needed** for the camera visibility registry management. The app should either request UAC on startup or have a separate "admin mode" for registry operations.

6. **The GoPro USB network interface discovery is the trickiest part.** The interface may be named differently on different systems. Be robust — try multiple discovery methods (interface name matching, IP range scanning, mDNS).

7. **Handle the GoPro going to sleep.** The camera will sleep if no keep-alive is sent. Implement a background thread that periodically pings the camera (e.g., GET /gopro/webcam/status every 10 seconds).

8. **The user is NOT a professional programmer.** They can read Python and understand what's happening, but don't use complex architectural patterns without explaining them. Add comments that explain the "why" behind decisions, especially for networking and threading code.

9. **Error handling is critical.** The GoPro can disconnect at any time (cable pulled, battery dies, camera crashes). Every network call should have timeouts and try/except blocks. The app should recover gracefully and show clear error messages.

10. **This project is NOT about reverse engineering.** Everything uses GoPro's official, public, MIT-licensed Open GoPro API. This is completely legitimate.

## DELIVERABLES

1. Working Python application with all components above
2. requirements.txt with all dependencies
3. README.md with setup and usage instructions
4. ARCHITECTURE.md explaining how everything works
5. A proof-of-concept script (poc.py) that demonstrates the core flow in a single file

## TESTING PLAN

1. **POC Phase:** Single script that discovers GoPro, starts webcam, reads 10 frames, prints dimensions, stops webcam
2. **Stream Phase:** Add ffmpeg decoding, display frames in an OpenCV window
3. **Virtual Camera Phase:** Add pyvirtualcam output, verify NVIDIA Broadcast can see it
4. **Visibility Phase:** Add registry management, verify hiding/showing cameras works
5. **GUI Phase:** Build the full interface
6. **Integration Phase:** Everything working together end-to-end

## STRETCH GOALS

- Auto-detect GoPro connection/disconnection via USB event monitoring
- Preview window with low-latency rendering
- Recording the stream to a local file
- Multiple GoPro support
- Tray icon with quick controls
- Auto-start with Windows
- Dark mode GUI
