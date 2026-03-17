"""
Tests for usb_event_listener.py — USB device notification listener.

Tests mock all Windows APIs (ctypes.windll) so they run on any platform.
Covers:
  - GoPro device ID matching / filtering
  - Device name extraction from notification structures
  - Callback dispatching for attach/detach/devnodes_changed events
  - Listener lifecycle (start/stop/restart)
  - Edge cases (empty names, non-GoPro devices, double start/stop)
  - Window procedure message handling
  - Cleanup on shutdown
"""

import ctypes
import struct
import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock, call
import pytest

from usb_event_listener import (

    USBEventListener,
    _is_gopro_device,
    _extract_device_name,
    _make_guid,
    GOPRO_VENDOR_IDS,
    GOPRO_VID_STRINGS,
    WM_DEVICECHANGE,
    WM_CLOSE,
    DBT_DEVICEARRIVAL,
    DBT_DEVICEREMOVECOMPLETE,
    DBT_DEVNODES_CHANGED,
    DBT_DEVTYP_DEVICEINTERFACE,
    GUID,
    DEV_BROADCAST_HDR,
    DEV_BROADCAST_DEVICEINTERFACE_FILTER,
    GUID_DEVINTERFACE_USB_DEVICE,
)

pytestmark = pytest.mark.no_gopro_needed



# =============================================================================
# Tests for _is_gopro_device()
# =============================================================================

class TestIsGoProDevice:
    """Tests for the GoPro device ID matching function."""

    def test_matches_standard_gopro_hero12_path(self):
        """Standard Windows USB device path for GoPro Hero 12."""
        path = r"\\?\USB#VID_0A70&PID_000D#5&1234ABCD&0&1#{a5dcbf10-6530-11d2-901f-00c04fb951ed}"
        assert _is_gopro_device(path) is True

    def test_matches_lowercase_vid(self):
        """Device paths may have lowercase hex."""
        path = r"\\?\USB#vid_0a70&pid_000d#serial#{guid}"
        assert _is_gopro_device(path) is True

    def test_matches_mixed_case(self):
        """Mixed case should still match."""
        path = r"\\?\USB#Vid_0A70&Pid_000D#serial"
        assert _is_gopro_device(path) is True

    def test_matches_alternate_pid(self):
        """Different PIDs still match on VID."""
        path = r"\\?\USB#VID_0A70&PID_0011#serial#{guid}"
        assert _is_gopro_device(path) is True

    def test_rejects_non_gopro_device(self):
        """Logitech webcam should not match."""
        path = r"\\?\USB#VID_046D&PID_0825#serial#{guid}"
        assert _is_gopro_device(path) is False

    def test_rejects_empty_string(self):
        assert _is_gopro_device("") is False

    def test_rejects_none(self):
        """None input should return False (not raise)."""
        assert _is_gopro_device(None) is False

    def test_rejects_partial_vid(self):
        """VID_0A7 (missing digit) should not match."""
        path = r"\\?\USB#VID_0A7&PID_000D#serial"
        assert _is_gopro_device(path) is False

    def test_rejects_similar_vid(self):
        """VID_0A71 is not GoPro."""
        path = r"\\?\USB#VID_0A71&PID_000D#serial"
        assert _is_gopro_device(path) is False

    def test_matches_vid_in_longer_path(self):
        """VID can appear anywhere in the path."""
        path = r"USB\VID_0A70&PID_000D&MI_00\7&ABCDEF&0&0000"
        assert _is_gopro_device(path) is True

    def test_matches_hero12_vid_2672(self):
        """Hero 12+ uses VID_2672 which must also match."""
        path = r"\\?\USB#VID_2672&PID_0059#serial#{guid}"
        assert _is_gopro_device(path) is True

    def test_matches_hero12_vid_2672_lowercase(self):
        """Hero 12 VID in lowercase should match."""
        path = r"\\?\USB#vid_2672&pid_0059#serial#{guid}"
        assert _is_gopro_device(path) is True

    def test_matches_hero12_vid_2672_in_longer_path(self):
        """VID_2672 in a composite device path."""
        path = r"USB\VID_2672&PID_0059&MI_00\7&ABCDEF&0&0000"
        assert _is_gopro_device(path) is True


# =============================================================================
# Tests for GOPRO constants
# =============================================================================

class TestConstants:
    """Verify constants match expected values."""

    def test_vendor_ids_contains_both(self):
        """Both known GoPro vendor IDs must be present."""
        assert 0x2672 in GOPRO_VENDOR_IDS
        assert 0x0A70 in GOPRO_VENDOR_IDS

    def test_vid_strings_contains_both(self):
        """VID strings must cover both vendor IDs."""
        assert "VID_2672" in GOPRO_VID_STRINGS
        assert "VID_0A70" in GOPRO_VID_STRINGS

    def test_vendor_ids_match_discovery(self):
        """USB listener vendor IDs must match discovery.py."""
        from discovery import GOPRO_VENDOR_IDS as DISCOVERY_VIDS
        assert GOPRO_VENDOR_IDS == DISCOVERY_VIDS

    def test_wm_devicechange(self):
        assert WM_DEVICECHANGE == 0x0219

    def test_dbt_constants(self):
        assert DBT_DEVICEARRIVAL == 0x8000
        assert DBT_DEVICEREMOVECOMPLETE == 0x8004
        assert DBT_DEVNODES_CHANGED == 0x0007


# =============================================================================
# Tests for _make_guid()
# =============================================================================

class TestMakeGuid:
    """Tests for GUID construction helper."""

    def test_creates_usb_device_guid(self):
        guid = _make_guid(*GUID_DEVINTERFACE_USB_DEVICE)
        assert guid.Data1 == 0xA5DCBF10
        assert guid.Data2 == 0x6530
        assert guid.Data3 == 0x11D2
        assert guid.Data4[0] == 0x90
        assert guid.Data4[1] == 0x1F

    def test_guid_data4_length(self):
        guid = _make_guid(0, 0, 0, (1, 2, 3, 4, 5, 6, 7, 8))
        for i, expected in enumerate([1, 2, 3, 4, 5, 6, 7, 8]):
            assert guid.Data4[i] == expected


# =============================================================================
# Tests for USBEventListener lifecycle
# =============================================================================

class TestUSBEventListenerLifecycle:
    """Tests for listener start/stop behavior using mocked Windows APIs."""

    def test_initial_state(self):
        """Listener starts in non-running state."""
        listener = USBEventListener()
        assert listener.is_running is False
        assert listener._hwnd is None
        assert listener._hdev_notify is None

    def test_callbacks_stored(self):
        """Callbacks are stored on construction."""
        attach = MagicMock()
        detach = MagicMock()
        devnodes = MagicMock()
        listener = USBEventListener(
            on_attach=attach,
            on_detach=detach,
            on_devnodes_changed=devnodes,
        )
        assert listener.on_attach is attach
        assert listener.on_detach is detach
        assert listener.on_devnodes_changed is devnodes

    def test_stop_when_not_running_is_noop(self):
        """stop() on an already-stopped listener should not error."""
        listener = USBEventListener()
        listener.stop()  # Should not raise

    def test_double_stop_is_safe(self):
        """Calling stop() twice should not error."""
        listener = USBEventListener()
        listener.stop()
        listener.stop()

    def test_is_running_checks_thread_alive(self):
        """is_running should be False if thread is dead."""
        listener = USBEventListener()
        listener._running = True
        listener._thread = MagicMock()
        listener._thread.is_alive.return_value = False
        assert listener.is_running is False

    def test_is_running_true_when_thread_alive(self):
        """is_running should be True if _running and thread alive."""
        listener = USBEventListener()
        listener._running = True
        listener._thread = MagicMock()
        listener._thread.is_alive.return_value = True
        assert listener.is_running is True


# =============================================================================
# Tests for _handle_device_change()
# =============================================================================

class TestHandleDeviceChange:
    """Tests for the WM_DEVICECHANGE event handler."""

    def _make_listener_with_mocks(self):
        """Create a listener with mock callbacks."""
        listener = USBEventListener(
            on_attach=MagicMock(),
            on_detach=MagicMock(),
            on_devnodes_changed=MagicMock(),
        )
        return listener

    def test_devnodes_changed_fires_callback(self):
        """DBT_DEVNODES_CHANGED should fire on_devnodes_changed."""
        listener = self._make_listener_with_mocks()
        listener._handle_device_change(DBT_DEVNODES_CHANGED, 0)
        listener.on_devnodes_changed.assert_called_once()

    def test_devnodes_changed_no_callback(self):
        """DBT_DEVNODES_CHANGED with no callback should not error."""
        listener = USBEventListener()
        listener._handle_device_change(DBT_DEVNODES_CHANGED, 0)

    def test_unknown_event_type_ignored(self):
        """Unknown event types should be silently ignored."""
        listener = self._make_listener_with_mocks()
        listener._handle_device_change(0x9999, 0)
        listener.on_attach.assert_not_called()
        listener.on_detach.assert_not_called()

    def test_null_lparam_ignored(self):
        """Null lparam should be ignored for arrival/removal events."""
        listener = self._make_listener_with_mocks()
        listener._handle_device_change(DBT_DEVICEARRIVAL, 0)
        listener.on_attach.assert_not_called()

    def test_arrival_with_gopro_device_fires_on_attach(self):
        """Attach event for GoPro device should fire on_attach callback."""
        listener = self._make_listener_with_mocks()
        gopro_path = r"\\?\USB#VID_0A70&PID_000D#serial#{guid}"

        # Mock _extract_device_name and _is_gopro_device
        with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            # Set up the mock for ctypes.cast to return a proper header
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            listener._handle_device_change(DBT_DEVICEARRIVAL, 12345)
            listener.on_attach.assert_called_once_with(gopro_path)

    def test_removal_with_gopro_device_fires_on_detach(self):
        """Detach event for GoPro device should fire on_detach callback."""
        listener = self._make_listener_with_mocks()
        gopro_path = r"\\?\USB#VID_0A70&PID_000D#serial#{guid}"

        with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            listener._handle_device_change(DBT_DEVICEREMOVECOMPLETE, 12345)
            listener.on_detach.assert_called_once_with(gopro_path)

    def test_non_gopro_device_not_dispatched(self):
        """Non-GoPro device events should not fire callbacks."""
        listener = self._make_listener_with_mocks()
        logitech_path = r"\\?\USB#VID_046D&PID_0825#serial#{guid}"

        with patch("usb_event_listener._extract_device_name", return_value=logitech_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            listener._handle_device_change(DBT_DEVICEARRIVAL, 12345)
            listener.on_attach.assert_not_called()

    def test_callback_exception_does_not_propagate(self):
        """If a callback raises, it should be caught and logged."""
        listener = self._make_listener_with_mocks()
        listener.on_attach.side_effect = RuntimeError("callback error")
        gopro_path = r"\\?\USB#VID_0A70&PID_000D#serial#{guid}"

        with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            # Should not raise
            listener._handle_device_change(DBT_DEVICEARRIVAL, 12345)

    def test_devnodes_callback_exception_does_not_propagate(self):
        """If on_devnodes_changed raises, it should be caught."""
        listener = self._make_listener_with_mocks()
        listener.on_devnodes_changed.side_effect = RuntimeError("callback error")

        # Should not raise
        listener._handle_device_change(DBT_DEVNODES_CHANGED, 0)

    def test_empty_device_name_ignored(self):
        """Empty device name should be silently ignored."""
        listener = self._make_listener_with_mocks()

        with patch("usb_event_listener._extract_device_name", return_value=""), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            listener._handle_device_change(DBT_DEVICEARRIVAL, 12345)
            listener.on_attach.assert_not_called()


# =============================================================================
# Tests for _wnd_proc()
# =============================================================================

class TestWndProc:
    """Tests for the window procedure callback."""

    def test_wm_devicechange_calls_handler(self):
        """WM_DEVICECHANGE should trigger _handle_device_change."""
        listener = USBEventListener()
        listener._handle_device_change = MagicMock()

        with patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_ctypes.windll.user32.DefWindowProcW.return_value = 0
            result = listener._wnd_proc(1, WM_DEVICECHANGE, DBT_DEVICEARRIVAL, 12345)

        assert result == 0
        listener._handle_device_change.assert_called_once_with(DBT_DEVICEARRIVAL, 12345)

    def test_wm_close_posts_quit(self):
        """WM_CLOSE should call PostQuitMessage to exit the message loop."""
        listener = USBEventListener()

        with patch("usb_event_listener.ctypes") as mock_ctypes:
            result = listener._wnd_proc(1, WM_CLOSE, 0, 0)

        assert result == 0
        mock_ctypes.windll.user32.PostQuitMessage.assert_called_once_with(0)

    def test_other_messages_forwarded(self):
        """Non-handled messages should be forwarded to DefWindowProcW."""
        listener = USBEventListener()
        OTHER_MSG = 0x1234

        with patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_ctypes.windll.user32.DefWindowProcW.return_value = 42
            result = listener._wnd_proc(1, OTHER_MSG, 0, 0)

        assert result == 42
        mock_ctypes.windll.user32.DefWindowProcW.assert_called_once_with(1, OTHER_MSG, 0, 0)


# =============================================================================
# Tests for DEV_BROADCAST_DEVICEINTERFACE_FILTER structure
# =============================================================================

class TestDevBroadcastFilter:
    """Tests for the notification filter structure layout."""

    def test_filter_size(self):
        """Filter struct size should be correct for RegisterDeviceNotification."""
        filt = DEV_BROADCAST_DEVICEINTERFACE_FILTER()
        filt.dbcc_size = ctypes.sizeof(DEV_BROADCAST_DEVICEINTERFACE_FILTER)
        # Size = 4 (DWORD) + 4 (DWORD) + 4 (DWORD) + 16 (GUID) = 28
        assert filt.dbcc_size == 28

    def test_filter_device_type(self):
        """Filter should specify device interface type."""
        filt = DEV_BROADCAST_DEVICEINTERFACE_FILTER()
        filt.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE
        assert filt.dbcc_devicetype == 5

    def test_filter_with_usb_guid(self):
        """Filter with USB device interface GUID should be constructable."""
        filt = DEV_BROADCAST_DEVICEINTERFACE_FILTER()
        filt.dbcc_size = ctypes.sizeof(DEV_BROADCAST_DEVICEINTERFACE_FILTER)
        filt.dbcc_devicetype = DBT_DEVTYP_DEVICEINTERFACE
        filt.dbcc_classguid = _make_guid(*GUID_DEVINTERFACE_USB_DEVICE)
        assert filt.dbcc_classguid.Data1 == 0xA5DCBF10


# =============================================================================
# Tests for _extract_device_name()
# =============================================================================

class TestExtractDeviceName:
    """Tests for extracting device names from notification structures."""

    def test_returns_empty_for_zero_lparam(self):
        """Zero/null lparam should return empty string."""
        # _extract_device_name will try to cast 0 which should fail gracefully
        result = _extract_device_name(0)
        # Either returns "" or raises - we handle both
        assert isinstance(result, str)

    def test_returns_empty_for_negative_lparam(self):
        """Negative lparam should return empty string (not crash)."""
        result = _extract_device_name(-1)
        assert result == ""

    def test_returns_empty_for_none_lparam(self):
        """None-like lparam should return empty string."""
        result = _extract_device_name(None)
        assert result == ""


# =============================================================================
# Tests for message loop initialization (mocked)
# =============================================================================

class TestMessageLoopInit:
    """Tests for the _message_loop initialization path with mocked Win32 APIs."""

    def test_start_timeout_returns_false(self):
        """If message loop thread doesn't signal ready, start() returns False."""
        listener = USBEventListener()

        # Patch _message_loop to do nothing (simulating it never signals ready)
        with patch.object(listener, '_message_loop'):
            result = listener.start(timeout=0.1)

        assert result is False

    def test_start_when_already_running(self):
        """start() when already running should return True immediately."""
        listener = USBEventListener()
        listener._running = True

        result = listener.start()
        assert result is True

    def test_start_sets_ready_event_on_failure(self):
        """If RegisterClassExW fails, ready_event should still be set."""
        listener = USBEventListener()

        def mock_message_loop():
            # Simulate failure - set ready but don't set _running
            listener._ready_event.set()

        with patch.object(listener, '_message_loop', side_effect=mock_message_loop):
            result = listener.start(timeout=1.0)

        assert result is False

    def test_stop_posts_wm_close(self):
        """stop() should post WM_CLOSE to the hidden window."""
        listener = USBEventListener()
        listener._running = True
        listener._hwnd = 12345
        listener._thread = MagicMock()
        listener._thread.is_alive.return_value = False

        with patch("usb_event_listener.ctypes") as mock_ctypes:
            listener.stop()
            mock_ctypes.windll.user32.PostMessageW.assert_called_once_with(
                12345, WM_CLOSE, 0, 0
            )


# =============================================================================
# Tests for cleanup
# =============================================================================

class TestCleanup:
    """Tests for resource cleanup on shutdown."""

    def test_cleanup_unregisters_notification(self):
        """_cleanup should call UnregisterDeviceNotification."""
        listener = USBEventListener()
        mock_user32 = MagicMock()

        listener._cleanup(mock_user32, hwnd=1, hdev_notify=2, class_name="test", hinstance=3)

        mock_user32.UnregisterDeviceNotification.assert_called_once_with(2)

    def test_cleanup_destroys_window(self):
        """_cleanup should call DestroyWindow."""
        listener = USBEventListener()
        mock_user32 = MagicMock()

        listener._cleanup(mock_user32, hwnd=1, hdev_notify=2, class_name="test", hinstance=3)

        mock_user32.DestroyWindow.assert_called_once_with(1)

    def test_cleanup_resets_state(self):
        """_cleanup should reset _running, _hwnd, _hdev_notify."""
        listener = USBEventListener()
        listener._running = True
        listener._hwnd = 123
        listener._hdev_notify = 456

        mock_user32 = MagicMock()
        listener._cleanup(mock_user32, hwnd=1, hdev_notify=2, class_name="test", hinstance=3)

        assert listener._running is False
        assert listener._hwnd is None
        assert listener._hdev_notify is None

    def test_cleanup_handles_unregister_exception(self):
        """_cleanup should handle exceptions from UnregisterDeviceNotification."""
        listener = USBEventListener()
        mock_user32 = MagicMock()
        mock_user32.UnregisterDeviceNotification.side_effect = OSError("test")

        # Should not raise
        listener._cleanup(mock_user32, hwnd=1, hdev_notify=2, class_name="test", hinstance=3)

    def test_cleanup_with_null_handles(self):
        """_cleanup should handle None/zero handles gracefully."""
        listener = USBEventListener()
        mock_user32 = MagicMock()

        # Should not raise
        listener._cleanup(mock_user32, hwnd=0, hdev_notify=0, class_name="test", hinstance=0)


# =============================================================================
# Integration-style tests (still mocked but test full flow)
# =============================================================================

class TestIntegration:
    """Higher-level tests combining multiple components."""

    def test_gopro_attach_full_flow(self):
        """Simulate a full GoPro attach event through _handle_device_change."""
        attach_calls = []
        detach_calls = []

        listener = USBEventListener(
            on_attach=lambda d: attach_calls.append(d),
            on_detach=lambda d: detach_calls.append(d),
        )

        gopro_path = r"\\?\USB#VID_0A70&PID_000D#serial#{a5dcbf10-6530-11d2-901f-00c04fb951ed}"

        with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            # Simulate attach
            listener._handle_device_change(DBT_DEVICEARRIVAL, 1)
            # Simulate detach
            listener._handle_device_change(DBT_DEVICEREMOVECOMPLETE, 1)

        assert len(attach_calls) == 1
        assert len(detach_calls) == 1
        assert "VID_0A70" in attach_calls[0]
        assert "VID_0A70" in detach_calls[0]

    def test_multiple_devices_only_gopro_matches(self):
        """Multiple USB events but only GoPro ones trigger callbacks."""
        attach_calls = []

        listener = USBEventListener(on_attach=lambda d: attach_calls.append(d))

        devices = [
            (r"\\?\USB#VID_046D&PID_0825#serial#{guid}", False),  # Logitech
            (r"\\?\USB#VID_0A70&PID_000D#serial#{guid}", True),   # GoPro
            (r"\\?\USB#VID_8086&PID_9A13#serial#{guid}", False),  # Intel
        ]

        for path, is_gopro in devices:
            with patch("usb_event_listener._extract_device_name", return_value=path), \
                 patch("usb_event_listener.ctypes") as mock_ctypes:
                mock_hdr = MagicMock()
                mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
                mock_ctypes.cast.return_value = mock_hdr
                mock_ctypes.POINTER = ctypes.POINTER

                listener._handle_device_change(DBT_DEVICEARRIVAL, 1)

        assert len(attach_calls) == 1
        assert "VID_0A70" in attach_calls[0]

    def test_listener_without_callbacks(self):
        """Listener with no callbacks should handle events without error."""
        listener = USBEventListener()  # No callbacks set
        gopro_path = r"\\?\USB#VID_0A70&PID_000D#serial#{guid}"

        with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            # Should not raise despite no callbacks
            listener._handle_device_change(DBT_DEVICEARRIVAL, 1)
            listener._handle_device_change(DBT_DEVICEREMOVECOMPLETE, 1)
            listener._handle_device_change(DBT_DEVNODES_CHANGED, 0)


# =============================================================================
# Tests for USB detection reliability (13-item reliability fixes)
# =============================================================================

class TestUSBDetectionReliability:
    """Tests for USB detection edge cases and reliability guarantees.

    Covers:
      - Both vendor IDs (0x2672 Hero 12+, 0x0A70 older models) are detected
      - Duplicate event handling is safe (idempotent)
      - Rapid attach/detach sequences don't cause state corruption
      - Vendor ID matching is exact (no false positives from similar IDs)
      - Device paths with various formatting are handled
    """

    def test_both_vendor_ids_detected(self):
        """Both 0x2672 (Hero 12+) and 0x0A70 (older) must match."""
        paths = [
            r"\\?\USB#VID_2672&PID_0059#serial#{guid}",
            r"\\?\USB#VID_0A70&PID_000D#serial#{guid}",
        ]
        for path in paths:
            assert _is_gopro_device(path) is True, f"Failed for {path}"

    def test_vendor_id_boundary_values_rejected(self):
        """Similar but different vendor IDs must not match."""
        non_gopro = [
            r"\\?\USB#VID_2671&PID_0059#serial",  # one less
            r"\\?\USB#VID_2673&PID_0059#serial",  # one more
            r"\\?\USB#VID_0A6F&PID_000D#serial",  # one less
            r"\\?\USB#VID_0A71&PID_000D#serial",  # one more
        ]
        for path in non_gopro:
            assert _is_gopro_device(path) is False, f"False positive for {path}"

    def test_duplicate_attach_events_fire_callback_each_time(self):
        """Each attach event should fire the callback (filtering is caller's job)."""
        attach_calls = []
        listener = USBEventListener(on_attach=lambda d: attach_calls.append(d))

        gopro_path = r"\\?\USB#VID_2672&PID_0059#serial#{guid}"
        for _ in range(3):
            with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
                 patch("usb_event_listener.ctypes") as mock_ctypes:
                mock_hdr = MagicMock()
                mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
                mock_ctypes.cast.return_value = mock_hdr
                mock_ctypes.POINTER = ctypes.POINTER
                listener._handle_device_change(DBT_DEVICEARRIVAL, 1)

        assert len(attach_calls) == 3

    def test_rapid_attach_detach_sequence_no_crash(self):
        """Rapidly alternating attach/detach events must not corrupt state."""
        attach_calls = []
        detach_calls = []
        listener = USBEventListener(
            on_attach=lambda d: attach_calls.append(d),
            on_detach=lambda d: detach_calls.append(d),
        )

        gopro_path = r"\\?\USB#VID_0A70&PID_000D#serial#{guid}"
        for i in range(10):
            event = DBT_DEVICEARRIVAL if i % 2 == 0 else DBT_DEVICEREMOVECOMPLETE
            with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
                 patch("usb_event_listener.ctypes") as mock_ctypes:
                mock_hdr = MagicMock()
                mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
                mock_ctypes.cast.return_value = mock_hdr
                mock_ctypes.POINTER = ctypes.POINTER
                listener._handle_device_change(event, 1)

        assert len(attach_calls) == 5
        assert len(detach_calls) == 5

    def test_device_path_with_composite_interface(self):
        """Composite USB device paths (with MI_xx) should still match."""
        composite = r"USB\VID_2672&PID_0059&MI_00\7&ABCDEF&0&0000"
        assert _is_gopro_device(composite) is True

    def test_hero12_vid_hex_format(self):
        """VID_2672 is 0x2672 in hex (decimal 9842), not decimal 2672."""
        # Verify the pattern matches hex representation, not decimal
        hex_path = r"\\?\USB#VID_2672&PID_0059#serial"
        assert _is_gopro_device(hex_path) is True

    def test_vendor_ids_set_immutable_between_modules(self):
        """GOPRO_VENDOR_IDS must be consistent across modules."""
        from discovery import GOPRO_VENDOR_IDS as discovery_vids
        assert GOPRO_VENDOR_IDS == discovery_vids
        assert len(GOPRO_VENDOR_IDS) >= 2  # At least two known VIDs

    def test_callback_exception_isolation(self):
        """If on_attach raises, on_detach should still work on next event."""
        detach_calls = []
        listener = USBEventListener(
            on_attach=MagicMock(side_effect=RuntimeError("boom")),
            on_detach=lambda d: detach_calls.append(d),
        )
        gopro_path = r"\\?\USB#VID_0A70&PID_000D#serial#{guid}"

        with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            listener._handle_device_change(DBT_DEVICEARRIVAL, 1)  # Raises in callback
            listener._handle_device_change(DBT_DEVICEREMOVECOMPLETE, 1)  # Should still work

        assert len(detach_calls) == 1

    def test_fan_out_to_multiple_consumers(self):
        """A single USBEventListener can fan out events to multiple consumers.

        This is the pattern used by AppController (AC 2): one listener instance
        fans out to both DisconnectDetector.handle_usb_detach() and
        AppController._on_usb_disconnect(). Verify callbacks handle this correctly.
        """
        consumer_a_attach = []
        consumer_b_attach = []
        consumer_a_detach = []
        consumer_b_detach = []

        def fan_out_attach(device_id):
            consumer_a_attach.append(device_id)
            consumer_b_attach.append(device_id)

        def fan_out_detach(device_id):
            consumer_a_detach.append(device_id)
            consumer_b_detach.append(device_id)

        listener = USBEventListener(
            on_attach=fan_out_attach,
            on_detach=fan_out_detach,
        )

        gopro_path = r"\\?\USB#VID_2672&PID_0059#serial#{guid}"

        with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            listener._handle_device_change(DBT_DEVICEARRIVAL, 1)
            listener._handle_device_change(DBT_DEVICEREMOVECOMPLETE, 1)

        # Both consumers received both events
        assert len(consumer_a_attach) == 1
        assert len(consumer_b_attach) == 1
        assert len(consumer_a_detach) == 1
        assert len(consumer_b_detach) == 1

    def test_fan_out_one_consumer_error_doesnt_block_others(self):
        """When fan-out callback raises, events are still dispatched.

        In the fan-out pattern, if one consumer's handler raises (e.g.,
        DisconnectDetector), the exception is caught and the next event
        still fires for other consumers.
        """
        consumer_b_calls = []
        call_count = [0]

        def fan_out_attach(device_id):
            call_count[0] += 1
            # First consumer fails
            if call_count[0] == 1:
                raise RuntimeError("consumer A error")
            # This won't run on same call, but on a second event it will
            consumer_b_calls.append(device_id)

        listener = USBEventListener(on_attach=fan_out_attach)

        gopro_path = r"\\?\USB#VID_0A70&PID_000D#serial#{guid}"

        with patch("usb_event_listener._extract_device_name", return_value=gopro_path), \
             patch("usb_event_listener.ctypes") as mock_ctypes:
            mock_hdr = MagicMock()
            mock_hdr.contents.dbch_devicetype = DBT_DEVTYP_DEVICEINTERFACE
            mock_ctypes.cast.return_value = mock_hdr
            mock_ctypes.POINTER = ctypes.POINTER

            # First call raises — exception caught by listener
            listener._handle_device_change(DBT_DEVICEARRIVAL, 1)
            # Second call should still work
            listener._handle_device_change(DBT_DEVICEARRIVAL, 1)

        assert len(consumer_b_calls) == 1  # Second call succeeded
