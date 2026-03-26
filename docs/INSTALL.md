# Installation Guide

Detailed installation instructions for GoMaxWebcam v2 on all supported platforms.

---

## Table of Contents

- [Windows](#windows)
- [macOS](#macos)
- [Linux](#linux)
- [Verify Installation](#verify-installation)
- [Troubleshooting](#troubleshooting)

---

## Windows

Windows is the primary supported platform. GoMaxWebcam uses Unity Capture as its virtual camera driver.

### Step 1: Install Python 3.12+

Download from [python.org](https://www.python.org/downloads/). During installation, check **"Add Python to PATH"**.

Verify:

```
python --version
```

### Step 2: Install the Virtual Camera Driver

GoMaxWebcam outputs video through [Unity Capture](https://github.com/schellingb/UnityCapture), a lightweight DirectShow virtual camera.

1. Download the latest release from [Unity Capture releases](https://github.com/schellingb/UnityCapture/releases).
2. Extract the zip.
3. Right-click `Install.bat` and select **Run as Administrator**.
4. Confirm the UAC prompt.

After installation, "Unity Video Capture" will appear as a camera source in Zoom, Teams, OBS, and other apps.

> **Note:** The built-in Windows Camera app cannot see DirectShow virtual cameras. This is a Windows platform limitation, not a GoMaxWebcam bug. Use any other video application instead.

### Step 3: Install GoMaxWebcam

**Option A -- From PyPI:**

```
pip install gomaxwebcam
```

**Option B -- From source:**

```
git clone https://github.com/InFase/GoMaxWebcam.git
cd GoMaxWebcam
pip install -e .
```

### Step 4: Connect Your GoPro

1. Plug your GoPro into your PC with a USB-C data cable.
2. On the GoPro, go to **Connections > USB Connection** and set it to **GoPro Connect**.
3. The camera should appear as a USB Ethernet adapter (NCM/RNDIS).

### Step 5: Run

```
gomaxwebcam
```

Or from source:

```
python -m gomaxwebcam
```

The dashboard opens in your browser. Your GoPro feed is now available as "Unity Video Capture" in any video application.

---

## macOS

macOS support uses OBS Virtual Camera as the output device.

### Step 1: Install Python 3.12+

Using Homebrew:

```bash
brew install python@3.12
```

Or download from [python.org](https://www.python.org/downloads/).

### Step 2: Install OBS Virtual Camera

1. Install [OBS Studio](https://obsproject.com/download) (version 26+ includes the virtual camera).
2. You do not need to run OBS -- GoMaxWebcam uses the virtual camera driver directly via pyvirtualcam.

After installation, "OBS Virtual Camera" will appear as a camera source in video apps.

### Step 3: Install GoMaxWebcam

```bash
pip install gomaxwebcam
```

Or from source:

```bash
git clone https://github.com/InFase/GoMaxWebcam.git
cd GoMaxWebcam
pip install -e .
```

### Step 4: Run

```bash
gomaxwebcam
```

> **Note:** macOS may prompt for Bluetooth and local network permissions. Allow both -- Bluetooth is needed for COHN provisioning, and local network access is needed for camera streaming.

---

## Linux

Linux support uses v4l2loopback to create a virtual camera device.

### Step 1: Install Python 3.12+

Most distributions provide Python 3.12+ in their package manager:

```bash
# Ubuntu/Debian
sudo apt install python3.12 python3.12-venv python3-pip

# Fedora
sudo dnf install python3.12

# Arch
sudo pacman -S python
```

### Step 2: Install v4l2loopback

```bash
# Ubuntu/Debian
sudo apt install v4l2loopback-dkms v4l2loopback-utils

# Fedora
sudo dnf install v4l2loopback

# Arch
sudo pacman -S v4l2loopback-dkms
```

Load the kernel module:

```bash
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="GoMaxWebcam" exclusive_caps=1
```

To load automatically on boot, add to `/etc/modules-load.d/v4l2loopback.conf`:

```
v4l2loopback
```

And configure options in `/etc/modprobe.d/v4l2loopback.conf`:

```
options v4l2loopback devices=1 video_nr=10 card_label="GoMaxWebcam" exclusive_caps=1
```

### Step 3: Install GoMaxWebcam

```bash
pip install gomaxwebcam
```

Or from source:

```bash
git clone https://github.com/InFase/GoMaxWebcam.git
cd GoMaxWebcam
pip install -e .
```

### Step 4: Run

```bash
gomaxwebcam
```

On headless Linux (no `DISPLAY` or `WAYLAND_DISPLAY`), GoMaxWebcam automatically runs in headless mode, binding the dashboard to `0.0.0.0` so you can access it from another machine.

```bash
gomaxwebcam --headless
```

---

## Verify Installation

After starting GoMaxWebcam with a GoPro connected:

1. **Dashboard opens** in your browser showing the Home page with camera status.
2. **Status shows "Streaming"** once the camera connects and the stream starts.
3. **Virtual camera is visible** in your video app:
   - Windows: look for "Unity Video Capture"
   - macOS: look for "OBS Virtual Camera"
   - Linux: look for "GoMaxWebcam" (the v4l2loopback device)
4. **Video appears** when you select the virtual camera in Zoom, Teams, OBS, etc.

---

## Troubleshooting

### GoPro Not Detected

- **Check the USB cable.** Many USB-C cables are charge-only. Use the cable that came with your GoPro, or a cable you have verified supports data transfer.
- **Check USB Connection mode.** On the GoPro: Connections > USB Connection > set to "GoPro Connect" (not "MTP").
- **Try a different USB port.** Some USB hubs do not support the NCM/RNDIS Ethernet adapter that the GoPro creates.
- **Check the GoPro firmware.** Update to the latest firmware via the GoPro Quik app.

### Virtual Camera Not Visible in Video Apps

- **Windows:** Ensure Unity Capture is installed (run `Install.bat` as Administrator). Some apps require a restart after driver installation.
- **macOS:** Ensure OBS Studio is installed. You do not need to run OBS, but the virtual camera driver must be present.
- **Linux:** Ensure v4l2loopback is loaded: `lsmod | grep v4l2loopback`. If not loaded, run `sudo modprobe v4l2loopback`.

### Windows Firewall Prompt

GoMaxWebcam communicates with the GoPro over a local USB network interface (NCM/RNDIS). If Windows Firewall asks for permission, allow access on **Private networks**. No external network access is made.

### Dashboard Does Not Open

- GoMaxWebcam prints the dashboard URL to the console: `Dashboard: http://127.0.0.1:<port>/?token=<token>`. Copy this URL and open it manually.
- If another instance is already running, the second launch opens the existing dashboard. Check the system tray for an existing GoMaxWebcam icon.
- To disable auto-open, set `open_browser_on_start = false` in your config.toml.

### COHN (WiFi) Not Connecting

- COHN requires one-time BLE provisioning. Go to the **Setup** page in the dashboard and follow the wizard.
- Ensure your PC has a Bluetooth adapter. COHN provisioning uses BLE (Bluetooth Low Energy).
- The GoPro and your PC must be on the same WiFi network for COHN to work.
- Check that COHN credentials are stored: the Setup wizard stores them in your OS keyring.

### Stream Freezes or Stutters

- **On WiFi (COHN):** Increase `frame_timeout_seconds` to 8-10 in config.toml to avoid false freeze triggers on congested networks.
- **On USB:** Try a different USB port or cable. USB 3.0 ports are preferred.
- **High CPU usage:** Lower the resolution to 720p in Settings or config.toml.

### Port 8554 Already in Use

If another application uses port 8554 (common with RTSP servers), change the UDP port in config.toml:

```toml
[advanced]
udp_port = 8555
```
