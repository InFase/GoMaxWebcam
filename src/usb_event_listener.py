"""
usb_event_listener.py — Windows USB device notification listener for GoPro detection

Uses Windows Device Notification APIs (RegisterDeviceNotification via ctypes)
to receive real-time USB attach/detach events for GoPro devices.

This provides instant notification when the GoPro is plugged in or unplugged,
complementing the polling-based discovery in discovery.py with event-driven
detection for faster response times.

Architecture:
  - Creates a hidden message-only window to receive WM_DEVICECHANGE messages
  - Registers for DBT_DEVICEARRIVAL and DBT_DEVICEREMOVECOMPLETE notifications
  - Filters events by GoPro USB vendor ID (0x2672)
  - Fires callbacks on a background thread so the GUI stays responsive
  - Thread-safe start/stop lifecycle

Usage:
    from usb_event_listener import USBEventListener

    def on_attach(device_id: str):
        print(f"GoPro attached: {device_id}")

    def on_detach(device_id: str):
        print(f"GoPro detached: {device_id}")

    listener = USBEventListener(on_attach=on_attach, on_detach=on_detach)
    listener.start()
    # ... later ...
    listener.stop()
"""

import ctypes
import ctypes.wintypes as wintypes
import re
import struct
import threading
import time
from typing import Callable, Optional

from logger import get_logger

log = get_logger(__name__)

# --- GoPro USB identifiers (imported from discovery.py for single source of truth) ---
from discovery import GOPRO_VENDOR_IDS

GOPRO_VID_STRINGS = {f"VID_{vid:04X}" for vid in GOPRO_VENDOR_IDS}  # {"VID_2672"}
# Build a regex alternation pattern for all known GoPro vendor IDs
_GOPRO_VID_PATTERN = re.compile(
    "|".join(f"VID_{vid:04X}" for vid in GOPRO_VENDOR_IDS),
    re.IGNORECASE,
)

# --- Windows constants for WM_DEVICECHANGE ---

# Window message for device change events
WM_DEVICECHANGE = 0x0219

# Device change event types
DBT_DEVICEARRIVAL = 0x8000          # A device has been inserted and is now available
DBT_DEVICEREMOVECOMPLETE = 0x8004   # A device has been removed
DBT_DEVNODES_CHANGED = 0x0007      # Device tree changed (generic notification)

# Device broadcast header device types
DBT_DEVTYP_DEVICEINTERFACE = 0x00000005

# Notification filter flags
DEVICE_NOTIFY_WINDOW_HANDLE = 0x00000000
DEVICE_NOTIFY_ALL_INTERFACE_CLASSES = 0x00000004

# GUID for USB device interface class
# {A5DCBF10-6530-11D2-901F-00C04FB951ED} — standard USB device interface
GUID_DEVINTERFACE_USB_DEVICE = (
    0xA5DCBF10, 0x6530, 0x11D2,
    (0x90, 0x1F, 0x00, 0xC0, 0x4F, 0xB9, 0x51, 0xED)
)

# Window message to request thread shutdown
WM_QUIT = 0x0012
WM_USER = 0x0400
WM_CLOSE = 0x0010

# Window style for message-only window
HWND_MESSAGE = -3


# --- ctypes structure definitions ---

class GUID(ctypes.Structure):
    """COM GUID structure for device interface class identification."""
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class DEV_BROADCAST_HDR(ctypes.Structure):
    """Header for all WM_DEVICECHANGE broadcast structures.

    The dbch_devicetype field determines which specific structure
    follows (e.g. DEV_BROADCAST_DEVICEINTERFACE for USB devices).
    """
    _fields_ = [
        ("dbch_size", wintypes.DWORD),
        ("dbch_devicetype", wintypes.DWORD),
        ("dbch_reserved", wintypes.DWORD),
    ]


class DEV_BROADCAST_DEVICEINTERFACE_W(ctypes.Structure):
    """Device interface notification structure (wide/Unicode version).

    Contains the device interface class GUID and a variable-length
    device name string that includes VID/PID information.
    """
    _fields_ = [
        ("dbcc_size", wintypes.DWORD),
        ("dbcc_devicetype", wintypes.DWORD),
        ("dbcc_reserved", wintypes.DWORD),
        ("dbcc_classguid", GUID),
        ("dbcc_name", ctypes.c_wchar * 1),  # Variable-length, extends beyond struct
    ]


class DEV_BROADCAST_DEVICEINTERFACE_FILTER(ctypes.Structure):
    """Registration filter structure for RegisterDeviceNotification.

    Used to specify which device interface class we want notifications for.
    """
    _fields_ = [
        ("dbcc_size", wintypes.DWORD),
        ("dbcc_devicetype", wintypes.DWORD),
        ("dbcc_reserved", wintypes.DWORD),
        ("dbcc_classguid", GUID),
    ]


class WNDCLASSEXW(ctypes.Structure):
    """Extended window class structure for RegisterClassEx."""
    _fields_ = [
        ("cbSize", ctypes.c_uint),
        ("style", ctypes.c_uint),
        ("lpfnWndProc", ctypes.WINFUNCTYPE(
            ctypes.c_long,          # return LRESULT
            wintypes.HWND,          # hWnd
            ctypes.c_uint,          # uMsg
            wintypes.WPARAM,        # wParam
            wintypes.LPARAM,        # lParam
        )),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE),
        ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE),
        ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR),
        ("lpszClassName", wintypes.LPCWSTR),
        ("hIconSm", wintypes.HICON),
    ]


# Type aliases for readability
LRESULT = ctypes.c_long

# WndProc callback type
WNDPROC = ctypes.WINFUNCTYPE(
    LRESULT,            # return type
    wintypes.HWND,      # hWnd
    ctypes.c_uint,      # uMsg
    wintypes.WPARAM,    # wParam
    wintypes.LPARAM,    # lParam
)


def _make_guid(data1, data2, data3, data4_tuple):
    """Create a GUID structure from components."""
    guid = GUID()
    guid.Data1 = data1
    guid.Data2 = data2
    guid.Data3 = data3
    for i, b in enumerate(data4_tuple):
        guid.Data4[i] = b
    return guid


def _extract_device_name(lparam: int) -> str:
    """Extract the device name string from a DEV_BROADCAST_DEVICEINTERFACE.

    The device name is a variable-length wide string that extends beyond
    the fixed fields of the structure. We read the full buffer to get it.

    Args:
        lparam: The lParam from WM_DEVICECHANGE, pointing to the broadcast struct.

    Returns:
        The device path/name string, or empty string on failure.
    """
    if not lparam or not isinstance(lparam, int) or lparam <= 0:
        return ""

    try:
        hdr = ctypes.cast(lparam, ctypes.POINTER(DEV_BROADCAST_HDR)).contents
        if hdr.dbch_devicetype != DBT_DEVTYP_DEVICEINTERFACE:
            return ""

        # Read the total size from the header
        total_size = hdr.dbch_size
        if total_size < ctypes.sizeof(DEV_BROADCAST_HDR):
            return ""

        # The name starts after the fixed fields (size + type + reserved + GUID = 28 bytes)
        name_offset = 28  # 4 + 4 + 4 + 16 bytes
        name_bytes = total_size - name_offset

        if name_bytes <= 0:
            return ""

        # Read the raw bytes starting at the name offset
        buffer = (ctypes.c_byte * total_size)()
        ctypes.memmove(buffer, lparam, total_size)

        # Convert the name portion from wide chars (UTF-16LE)
        name_data = bytes(buffer[name_offset:])
        device_name = name_data.decode("utf-16-le", errors="ignore").rstrip("\x00")
        return device_name

    except Exception as e:
        log.debug("Failed to extract device name from notification: %s", e)
        return ""


def _is_gopro_device(device_name: str) -> bool:
    """Check if a device notification string matches a GoPro device.

    Device names from WM_DEVICECHANGE look like:
        \\\\?\\USB#VID_2672&PID_0059#...#{guid}   (webcam mode)
        \\\\?\\USB#VID_2672&PID_000D#...#{guid}   (RNDIS mode)
    We check for any known GoPro vendor ID pattern.

    Args:
        device_name: The device path from the notification.

    Returns:
        True if this is a GoPro device (any known vendor ID matches).
    """
    if not device_name:
        return False
    # Case-insensitive search for any known GoPro VID in the device path
    return bool(_GOPRO_VID_PATTERN.search(device_name))


class USBEventListener:
    """Listens for GoPro USB attach/detach events via Windows device notifications.

    Creates a hidden message-only window on a background thread that receives
    WM_DEVICECHANGE messages. When a GoPro device (VID_2672)
    is detected, the appropriate callback fires.

    Thread safety:
      - The message loop runs on its own daemon thread
      - Callbacks fire from the message loop thread
      - start()/stop() are safe to call from any thread
      - The listener can be started and stopped multiple times

    Attributes:
        on_attach: Called when a GoPro USB device is connected.
                   Signature: fn(device_id: str) -> None
        on_detach: Called when a GoPro USB device is disconnected.
                   Signature: fn(device_id: str) -> None
        on_devnodes_changed: Called on generic device tree changes.
                             Signature: fn() -> None (optional, for broader detection)
    """

    def __init__(
        self,
        on_attach: Optional[Callable[[str], None]] = None,
        on_detach: Optional[Callable[[str], None]] = None,
        on_devnodes_changed: Optional[Callable[[], None]] = None,
    ):
        self.on_attach = on_attach
        self.on_detach = on_detach
        self.on_devnodes_changed = on_devnodes_changed

        self._thread: Optional[threading.Thread] = None
        self._hwnd: Optional[int] = None
        self._hdev_notify: Optional[int] = None
        self._running = False
        self._ready_event = threading.Event()
        self._lock = threading.Lock()

        # Store the WndProc reference to prevent garbage collection
        self._wndproc: Optional[WNDPROC] = None

    @property
    def is_running(self) -> bool:
        """True if the listener thread is active and processing events."""
        with self._lock:
            return self._running and self._thread is not None and self._thread.is_alive()

    def start(self, timeout: float = 5.0) -> bool:
        """Start listening for USB device events.

        Creates a background thread with a hidden window and message loop
        that receives WM_DEVICECHANGE notifications.

        Args:
            timeout: Max seconds to wait for the listener thread to initialize.

        Returns:
            True if the listener started successfully.
        """
        with self._lock:
            if self._running:
                log.warning("[EVENT:usb_listener] start() called but already running")
                return True

        self._ready_event.clear()

        self._thread = threading.Thread(
            target=self._message_loop,
            name="USBEventListener",
            daemon=True,
        )
        self._thread.start()

        # Wait for the window to be created and device notification registered
        if not self._ready_event.wait(timeout=timeout):
            log.error("[EVENT:usb_listener] Listener thread failed to start within %.1fs", timeout)
            self.stop()
            return False

        with self._lock:
            if not self._running:
                log.error("[EVENT:usb_listener] Listener thread started but failed initialization")
                return False

        log.info("[EVENT:usb_listener] USB event listener started successfully")
        return True

    def stop(self):
        """Stop the USB event listener and clean up resources.

        Sends WM_CLOSE to the hidden window to trigger cleanup and exit
        the message loop. Safe to call from any thread.
        """
        with self._lock:
            if not self._running:
                return
            hwnd = self._hwnd

        if hwnd:
            try:
                # Post WM_CLOSE to trigger window destruction from the message loop thread
                user32 = ctypes.windll.user32
                user32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
            except Exception as e:
                log.debug("Error posting WM_CLOSE: %s", e)

        # Wait for the thread to finish
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                log.warning("[EVENT:usb_listener] Listener thread did not stop cleanly")

        with self._lock:
            self._running = False
            self._hwnd = None
            self._hdev_notify = None

        log.info("[EVENT:usb_listener] USB event listener stopped")

    def _message_loop(self):
        """Main message loop for the listener thread.

        Creates a message-only window, registers for USB device notifications,
        and pumps Windows messages until WM_CLOSE is received.
        """
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        # --- Create window class ---
        class_name = "GoProBridge_USBListener"
        hinstance = kernel32.GetModuleHandleW(None)

        # Create the WndProc callback (must keep reference to prevent GC)
        self._wndproc = WNDPROC(self._wnd_proc)

        wc = WNDCLASSEXW()
        wc.cbSize = ctypes.sizeof(WNDCLASSEXW)
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = hinstance
        wc.lpszClassName = class_name

        atom = user32.RegisterClassExW(ctypes.byref(wc))
        if not atom:
            error = kernel32.GetLastError()
            # Error 1410 = class already registered, which is OK
            if error != 1410:
                log.error(
                    "[EVENT:usb_listener] RegisterClassExW failed (error %d)", error
                )
                self._ready_event.set()
                return

        # --- Create message-only window ---
        hwnd = user32.CreateWindowExW(
            0,                      # dwExStyle
            class_name,             # lpClassName
            "GoPro USB Listener",   # lpWindowName
            0,                      # dwStyle
            0, 0, 0, 0,            # x, y, width, height
            HWND_MESSAGE,           # hWndParent (message-only)
            None,                   # hMenu
            hinstance,              # hInstance
            None,                   # lpParam
        )

        if not hwnd:
            error = kernel32.GetLastError()
            log.error(
                "[EVENT:usb_listener] CreateWindowExW failed (error %d)", error
            )
            self._ready_event.set()
            return

        with self._lock:
            self._hwnd = hwnd

        # --- Register for USB device interface notifications ---
        dev_filter = DEV_BROADCAST_DEVICEINTERFACE_FILTER()
        dev_filter.dbcc_size = ctypes.sizeof(DEV_BROADCAST_DEVICEINTERFACE_FILTER)
        dev_filter.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE
        dev_filter.dbcc_classguid = _make_guid(*GUID_DEVINTERFACE_USB_DEVICE)

        hdev_notify = user32.RegisterDeviceNotificationW(
            hwnd,
            ctypes.byref(dev_filter),
            DEVICE_NOTIFY_WINDOW_HANDLE,
        )

        if not hdev_notify:
            error = kernel32.GetLastError()
            log.error(
                "[EVENT:usb_listener] RegisterDeviceNotificationW failed (error %d)",
                error,
            )
            user32.DestroyWindow(hwnd)
            self._ready_event.set()
            return

        with self._lock:
            self._hdev_notify = hdev_notify
            self._running = True

        log.debug("[EVENT:usb_listener] Hidden window created (hwnd=%s), device notification registered", hwnd)
        self._ready_event.set()

        # --- Message pump ---
        msg = wintypes.MSG()
        while True:
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret <= 0:
                # 0 = WM_QUIT received, -1 = error
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        # --- Cleanup ---
        self._cleanup(user32, hwnd, hdev_notify, class_name, hinstance)

    def _cleanup(self, user32, hwnd, hdev_notify, class_name, hinstance):
        """Clean up Windows resources after the message loop exits."""
        try:
            if hdev_notify:
                user32.UnregisterDeviceNotification(hdev_notify)
        except Exception as e:
            log.debug("Error unregistering device notification: %s", e)

        try:
            if hwnd:
                user32.DestroyWindow(hwnd)
        except Exception as e:
            log.debug("Error destroying window: %s", e)

        try:
            kernel32 = ctypes.windll.kernel32
            hinstance = kernel32.GetModuleHandleW(None)
            user32.UnregisterClassW(class_name, hinstance)
        except Exception as e:
            log.debug("Error unregistering window class: %s", e)

        with self._lock:
            self._running = False
            self._hwnd = None
            self._hdev_notify = None

        log.debug("[EVENT:usb_listener] Message loop resources cleaned up")

    def _wnd_proc(self, hwnd, msg, wparam, lparam):
        """Window procedure that processes WM_DEVICECHANGE messages.

        This is the core event handler. Windows calls this for every message
        sent to our hidden window. We only care about WM_DEVICECHANGE.

        Args:
            hwnd: Window handle
            msg: Message type
            wparam: Event subtype for WM_DEVICECHANGE
            lparam: Pointer to device broadcast structure

        Returns:
            0 if handled, or result from DefWindowProcW for unhandled messages.
        """
        user32 = ctypes.windll.user32

        if msg == WM_DEVICECHANGE:
            self._handle_device_change(wparam, lparam)
            return 0

        if msg == WM_CLOSE:
            # Post WM_QUIT to exit the GetMessageW loop
            user32.PostQuitMessage(0)
            return 0

        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _handle_device_change(self, event_type: int, lparam: int):
        """Process a WM_DEVICECHANGE event.

        Filters for GoPro devices and fires the appropriate callback.

        Args:
            event_type: The wparam value (DBT_DEVICEARRIVAL, etc.)
            lparam: Pointer to DEV_BROADCAST_HDR structure.
        """
        if event_type == DBT_DEVNODES_CHANGED:
            log.debug("[EVENT:usb_listener] Device tree changed (DBT_DEVNODES_CHANGED)")
            if self.on_devnodes_changed:
                try:
                    self.on_devnodes_changed()
                except Exception:
                    log.exception("Error in on_devnodes_changed callback")
            return

        if event_type not in (DBT_DEVICEARRIVAL, DBT_DEVICEREMOVECOMPLETE):
            return

        if not lparam:
            return

        # Check if this is a device interface notification
        try:
            hdr = ctypes.cast(lparam, ctypes.POINTER(DEV_BROADCAST_HDR)).contents
        except Exception:
            return

        if hdr.dbch_devicetype != DBT_DEVTYP_DEVICEINTERFACE:
            return

        # Extract and check the device name
        device_name = _extract_device_name(lparam)

        if not device_name:
            log.debug("[EVENT:usb_listener] Device event with empty name (type=%d)", event_type)
            return

        is_gopro = _is_gopro_device(device_name)
        event_label = "ARRIVAL" if event_type == DBT_DEVICEARRIVAL else "REMOVAL"

        if is_gopro:
            log.info(
                "[EVENT:usb_listener] GoPro USB %s detected: %s",
                event_label, device_name,
            )
        else:
            log.debug(
                "[EVENT:usb_listener] Non-GoPro USB %s: %s",
                event_label, device_name[:80],
            )
            return

        # Fire the appropriate callback
        if event_type == DBT_DEVICEARRIVAL and self.on_attach:
            try:
                self.on_attach(device_name)
            except Exception:
                log.exception("Error in on_attach callback")

        elif event_type == DBT_DEVICEREMOVECOMPLETE and self.on_detach:
            try:
                self.on_detach(device_name)
            except Exception:
                log.exception("Error in on_detach callback")


# --- Module self-test ---
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s")

    print("=== USB Event Listener Test ===")
    print(f"Watching for GoPro USB events (VIDs: {GOPRO_VID_STRINGS})...")
    print("Plug/unplug a GoPro to see events. Press Ctrl+C to exit.\n")

    def on_attach(device_id):
        print(f"\n  >>> GoPro ATTACHED: {device_id}")

    def on_detach(device_id):
        print(f"\n  >>> GoPro DETACHED: {device_id}")

    def on_devnodes_changed():
        print("  ... Device tree changed")

    listener = USBEventListener(
        on_attach=on_attach,
        on_detach=on_detach,
        on_devnodes_changed=on_devnodes_changed,
    )

    if listener.start():
        print("Listener running. Waiting for events...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            listener.stop()
            print("Done.")
    else:
        print("ERROR: Listener failed to start!")
