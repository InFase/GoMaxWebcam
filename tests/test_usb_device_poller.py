"""
test_usb_device_poller.py — Tests for USB device polling/detection loop

Tests cover:
  1. Polling loop detects GoPro device appearance (absent → present transition)
  2. Polling loop detects GoPro device disappearance (present → absent transition)
  3. Callbacks fire correctly on state transitions
  4. Debounce: rapid transitions don't cause duplicate callbacks
  5. Lifecycle: start/stop/restart
  6. force_poll() for on-demand device checks
  7. Status reporting (get_status)
  8. Integration with AppController: poller signals _usb_reconnect_event
  9. Settling time delay before callback fires
  10. Error handling in enumerate_devices

IMPORTANT: USBEventListener uses raw Win32 APIs that crash in pytest.
All tests that instantiate AppController MUST mock usb_event_listener.USBEventListener.
"""

import sys
import os
import time
import threading
from unittest.mock import MagicMock, patch, PropertyMock, call
from dataclasses import dataclass

import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from usb_device_poller import USBDevicePoller
from config import Config


pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class MockGoProDevice:
    """Minimal GoProDevice mock for testing."""
    vendor_id: int = 0x0A70
    product_id: int = 0x000D
    description: str = "GoPro Hero 12 Black"
    camera_ip: str = "172.20.145.51"

    @property
    def usb_id_str(self) -> str:
        return f"VID_{self.vendor_id:04X}&PID_{self.product_id:04X}"


def make_test_config(**overrides) -> Config:
    """Create a config with fast timings for testing."""
    config = Config()
    config.reconnect_delay = 0.1
    config.ncm_adapter_wait = 0.1
    config.discovery_timeout = 0.5
    config.discovery_retry_interval = 0.1
    config.discovery_max_retries = 2
    config.keepalive_interval = 0.5
    config.idle_reset_delay = 0.1
    config._config_path = ""
    for k, v in overrides.items():
        setattr(config, k, v)
    return config


# Mock USBEventListener globally for AppController tests
_usb_listener_patch = patch(
    'usb_event_listener.USBEventListener',
    **{'return_value.start.return_value': False, 'return_value.stop.return_value': None,
       'return_value.is_running': False},
)


# ===========================================================================
# Tests: USBDevicePoller lifecycle
# ===========================================================================

class TestUSBDevicePollerLifecycle:
    """Tests for poller start/stop behavior."""

    def test_initial_state(self):
        """Poller starts in non-running state."""
        poller = USBDevicePoller()
        assert poller.is_running is False
        assert poller.is_device_present is False
        assert poller.poll_count == 0
        assert poller.transition_count == 0
        assert poller.last_seen_time is None

    def test_start_returns_true(self):
        """start() should return True and set running."""
        poller = USBDevicePoller(poll_interval=10.0)
        # Mock enumerate to avoid actual USB scan
        poller._enumerate_devices = MagicMock(return_value=[])

        assert poller.start() is True
        assert poller.is_running is True
        poller.stop()

    def test_stop_cleans_up(self):
        """stop() should set running to False."""
        poller = USBDevicePoller(poll_interval=10.0)
        poller._enumerate_devices = MagicMock(return_value=[])

        poller.start()
        assert poller.is_running is True

        poller.stop()
        assert poller.is_running is False

    def test_double_start_is_safe(self):
        """Double start() should return True without error."""
        poller = USBDevicePoller(poll_interval=10.0)
        poller._enumerate_devices = MagicMock(return_value=[])

        poller.start()
        result = poller.start()
        assert result is True
        poller.stop()

    def test_stop_without_start_is_noop(self):
        """stop() before start() should not crash."""
        poller = USBDevicePoller()
        poller.stop()  # Should not raise

    def test_double_stop_is_safe(self):
        """Double stop() should not crash."""
        poller = USBDevicePoller(poll_interval=10.0)
        poller._enumerate_devices = MagicMock(return_value=[])

        poller.start()
        poller.stop()
        poller.stop()  # Should not raise


# ===========================================================================
# Tests: Device detection (absent → present transition)
# ===========================================================================

class TestDeviceAppearance:
    """Tests that the poller detects when a GoPro device appears."""

    def test_device_appearance_fires_callback(self):
        """When GoPro appears on USB, on_device_appeared should fire."""
        appeared_events = []
        device = MockGoProDevice()

        poller = USBDevicePoller(
            on_device_appeared=lambda devs: appeared_events.append(devs),
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        poller.start()
        time.sleep(0.2)
        poller.stop()

        assert len(appeared_events) == 1
        assert appeared_events[0] == [device]

    def test_device_appearance_sets_present_flag(self):
        """Device appearing should set is_device_present to True."""
        device = MockGoProDevice()
        poller = USBDevicePoller(
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        assert poller.is_device_present is False

        poller.start()
        time.sleep(0.2)
        poller.stop()

        assert poller.is_device_present is True

    def test_device_appearance_updates_last_seen_time(self):
        """Device appearing should update last_seen_time."""
        device = MockGoProDevice()
        poller = USBDevicePoller(
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        assert poller.last_seen_time is None

        poller.start()
        time.sleep(0.2)
        poller.stop()

        assert poller.last_seen_time is not None

    def test_device_appearance_increments_transition_count(self):
        """Device appearing should increment transition_count."""
        device = MockGoProDevice()
        poller = USBDevicePoller(
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        poller.start()
        time.sleep(0.2)
        poller.stop()

        assert poller.transition_count == 1

    def test_stable_device_fires_callback_only_once(self):
        """A device that stays present should fire callback only once."""
        appeared_events = []
        device = MockGoProDevice()

        poller = USBDevicePoller(
            on_device_appeared=lambda devs: appeared_events.append(devs),
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        poller.start()
        time.sleep(0.3)  # Multiple poll cycles
        poller.stop()

        # Should only fire once (absent → present), not every poll
        assert len(appeared_events) == 1


# ===========================================================================
# Tests: Device disappearance (present → absent transition)
# ===========================================================================

class TestDeviceDisappearance:
    """Tests that the poller detects when a GoPro device disappears."""

    def test_device_disappearance_fires_callback(self):
        """When GoPro disappears from USB, on_device_disappeared should fire."""
        disappeared_events = []
        device = MockGoProDevice()
        poll_count = [0]

        def mock_enumerate():
            poll_count[0] += 1
            # First 2 polls: device present. After: device absent.
            if poll_count[0] <= 2:
                return [device]
            return []

        poller = USBDevicePoller(
            on_device_disappeared=lambda: disappeared_events.append(True),
            on_device_appeared=MagicMock(),
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = mock_enumerate

        poller.start()
        time.sleep(0.5)
        poller.stop()

        assert len(disappeared_events) >= 1

    def test_device_disappearance_clears_present_flag(self):
        """Device disappearing should set is_device_present to False."""
        device = MockGoProDevice()
        poll_count = [0]

        def mock_enumerate():
            poll_count[0] += 1
            if poll_count[0] <= 2:
                return [device]
            return []

        poller = USBDevicePoller(
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = mock_enumerate

        poller.start()
        time.sleep(0.5)
        poller.stop()

        assert poller.is_device_present is False


# ===========================================================================
# Tests: Full cycle (appear → disappear → reappear)
# ===========================================================================

class TestFullCycle:
    """Tests a full reconnection cycle: appear → disappear → reappear."""

    def test_full_reconnection_cycle(self):
        """Device appear → disappear → reappear fires callbacks correctly."""
        appeared_events = []
        disappeared_events = []
        device = MockGoProDevice()
        poll_count = [0]

        def mock_enumerate():
            poll_count[0] += 1
            # Polls 1-3: present, 4-6: absent, 7+: present again
            if poll_count[0] <= 3:
                return [device]
            elif poll_count[0] <= 6:
                return []
            else:
                return [device]

        poller = USBDevicePoller(
            on_device_appeared=lambda devs: appeared_events.append(devs),
            on_device_disappeared=lambda: disappeared_events.append(True),
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = mock_enumerate

        poller.start()
        time.sleep(0.8)
        poller.stop()

        # Should see: initial appear → disappear → reappear = 2 appears, 1 disappear
        assert len(appeared_events) == 2
        assert len(disappeared_events) == 1
        assert poller.transition_count == 3


# ===========================================================================
# Tests: Settling time
# ===========================================================================

class TestSettlingTime:
    """Tests that settling_time delays the callback after device detection."""

    def test_settling_time_delays_callback(self):
        """Callback should fire after settling_time, not immediately."""
        appeared_events = []
        device = MockGoProDevice()

        poller = USBDevicePoller(
            on_device_appeared=lambda devs: appeared_events.append(time.monotonic()),
            poll_interval=0.05,
            settling_time=0.15,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        start_time = time.monotonic()
        poller.start()
        time.sleep(0.4)
        poller.stop()

        assert len(appeared_events) >= 1
        # The callback should have fired at least settling_time after start
        elapsed = appeared_events[0] - start_time
        assert elapsed >= 0.1, f"Callback fired too early: {elapsed:.3f}s"


# ===========================================================================
# Tests: Error handling
# ===========================================================================

class TestErrorHandling:
    """Tests that polling errors don't crash the loop."""

    def test_enumerate_error_doesnt_crash(self):
        """If enumerate_devices raises, polling should continue."""
        appeared_events = []
        device = MockGoProDevice()
        poll_count = [0]

        def mock_enumerate():
            poll_count[0] += 1
            if poll_count[0] <= 2:
                raise OSError("WMI failed")
            return [device]

        poller = USBDevicePoller(
            on_device_appeared=lambda devs: appeared_events.append(devs),
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = mock_enumerate

        poller.start()
        time.sleep(0.4)
        poller.stop()

        # Should recover and detect the device after errors
        assert len(appeared_events) >= 1

    def test_callback_error_doesnt_crash(self):
        """If callback raises, polling should continue."""
        device = MockGoProDevice()
        call_count = [0]

        def bad_callback(devs):
            call_count[0] += 1
            raise RuntimeError("callback error")

        poller = USBDevicePoller(
            on_device_appeared=bad_callback,
            poll_interval=0.05,
            settling_time=0.0,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        poller.start()
        time.sleep(0.2)
        poller.stop()

        # Callback was called despite raising
        assert call_count[0] >= 1
        # Poller should still be functional (stopped normally)


# ===========================================================================
# Tests: force_poll()
# ===========================================================================

class TestForcePoll:
    """Tests for on-demand polling."""

    def test_force_poll_returns_true_when_device_found(self):
        """force_poll() should return True when GoPro is found."""
        device = MockGoProDevice()
        poller = USBDevicePoller()
        poller._enumerate_devices = MagicMock(return_value=[device])

        assert poller.force_poll() is True
        assert poller.poll_count == 1

    def test_force_poll_returns_false_when_no_device(self):
        """force_poll() should return False when no GoPro found."""
        poller = USBDevicePoller()
        poller._enumerate_devices = MagicMock(return_value=[])

        assert poller.force_poll() is False
        assert poller.poll_count == 1

    def test_force_poll_handles_error(self):
        """force_poll() should return False on error."""
        poller = USBDevicePoller()
        poller._enumerate_devices = MagicMock(side_effect=OSError("fail"))

        assert poller.force_poll() is False

    def test_force_poll_updates_last_seen(self):
        """force_poll() should update last_seen_time when device found."""
        device = MockGoProDevice()
        poller = USBDevicePoller()
        poller._enumerate_devices = MagicMock(return_value=[device])

        assert poller.last_seen_time is None
        poller.force_poll()
        assert poller.last_seen_time is not None


# ===========================================================================
# Tests: get_status()
# ===========================================================================

class TestGetStatus:
    """Tests for status reporting."""

    def test_get_status_returns_expected_keys(self):
        """get_status() should return a dict with all expected fields."""
        poller = USBDevicePoller()
        status = poller.get_status()

        assert isinstance(status, dict)
        assert "running" in status
        assert "device_present" in status
        assert "poll_count" in status
        assert "transition_count" in status
        assert "poll_interval" in status
        assert "last_seen_time" in status

    def test_get_status_initial_values(self):
        """Initial status should reflect non-running, no device state."""
        poller = USBDevicePoller(poll_interval=5.0)
        status = poller.get_status()

        assert status["running"] is False
        assert status["device_present"] is False
        assert status["poll_count"] == 0
        assert status["transition_count"] == 0
        assert status["poll_interval"] == 5.0
        assert status["last_seen_time"] is None

    def test_get_status_after_detection(self):
        """Status after detection should reflect device present."""
        device = MockGoProDevice()
        poller = USBDevicePoller(poll_interval=0.05, settling_time=0.0)
        poller._enumerate_devices = MagicMock(return_value=[device])

        poller.start()
        time.sleep(0.2)
        poller.stop()

        status = poller.get_status()
        assert status["device_present"] is True
        assert status["poll_count"] >= 1
        assert status["transition_count"] == 1
        assert status["last_seen_time"] is not None


# ===========================================================================
# Tests: Integration with AppController
# ===========================================================================

class TestAppControllerPollerIntegration:
    """Tests that AppController correctly integrates the USB device poller."""

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_usb_poller_attribute_initialized(self, mock_gopro_cls, mock_usb):
        """AppController should have _usb_poller = None initially."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        assert ctrl._usb_poller is None
        assert ctrl.usb_poller is None

    @_usb_listener_patch
    @patch('usb_device_poller.USBDevicePoller')
    @patch('app_controller.GoProConnection')
    def test_start_usb_poller_creates_instance(self, mock_gopro_cls, mock_poller_cls, mock_usb):
        """_start_usb_poller should create and start a USBDevicePoller."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        mock_poller = MagicMock()
        mock_poller.is_running = False
        mock_poller.start.return_value = True
        mock_poller_cls.return_value = mock_poller

        ctrl._start_usb_poller()

        mock_poller_cls.assert_called_once()
        mock_poller.start.assert_called_once()
        assert ctrl._usb_poller is mock_poller

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_stop_usb_poller_cleans_up(self, mock_gopro_cls, mock_usb):
        """_stop_usb_poller should stop and clear the poller."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        mock_poller = MagicMock()
        ctrl._usb_poller = mock_poller

        ctrl._stop_usb_poller()

        mock_poller.stop.assert_called_once()
        assert ctrl._usb_poller is None

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_stop_usb_poller_when_none(self, mock_gopro_cls, mock_usb):
        """_stop_usb_poller with no poller should not crash."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        ctrl._stop_usb_poller()  # Should not raise

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_app_stop_stops_poller(self, mock_gopro_cls, mock_usb):
        """AppController.stop() should stop the USB device poller."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        mock_poller = MagicMock()
        ctrl._usb_poller = mock_poller

        ctrl.stop()

        mock_poller.stop.assert_called_once()

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_poller_callback_sets_reconnect_event(self, mock_gopro_cls, mock_usb):
        """Poller's on_device_appeared callback should signal _usb_reconnect_event."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        # Simulate what _start_usb_poller wires up
        assert not ctrl._usb_reconnect_event.is_set()

        # Start poller with mock enumerate
        device = MockGoProDevice()

        with patch('usb_device_poller.USBDevicePoller') as mock_cls:
            mock_poller = MagicMock()
            mock_poller.is_running = False
            mock_poller.start.return_value = True

            # Capture the on_device_appeared callback
            captured_callback = None
            def capture_init(**kwargs):
                nonlocal captured_callback
                captured_callback = kwargs.get('on_device_appeared')
                return mock_poller
            mock_cls.side_effect = capture_init

            ctrl._start_usb_poller()

            # Verify callback was wired
            assert captured_callback is not None

            # Fire the callback
            captured_callback([device])

            # Verify reconnect event is set
            assert ctrl._usb_reconnect_event.is_set()

    @_usb_listener_patch
    @patch('app_controller.GoProConnection')
    def test_start_usb_poller_skip_if_already_running(self, mock_gopro_cls, mock_usb):
        """_start_usb_poller should skip if poller is already running."""
        from app_controller import AppController
        config = make_test_config()
        ctrl = AppController(config)

        mock_poller = MagicMock()
        mock_poller.is_running = True
        ctrl._usb_poller = mock_poller

        with patch('usb_device_poller.USBDevicePoller') as mock_cls:
            ctrl._start_usb_poller()
            mock_cls.assert_not_called()  # Should NOT create a new one


# ===========================================================================
# Tests: Polling loop thread safety
# ===========================================================================

class TestThreadSafety:
    """Tests for thread safety of poller operations."""

    def test_concurrent_status_reads(self):
        """Reading status from multiple threads should be safe."""
        device = MockGoProDevice()
        poller = USBDevicePoller(poll_interval=0.05, settling_time=0.0)
        poller._enumerate_devices = MagicMock(return_value=[device])

        poller.start()
        time.sleep(0.1)

        errors = []

        def read_status():
            try:
                for _ in range(50):
                    _ = poller.is_running
                    _ = poller.is_device_present
                    _ = poller.poll_count
                    _ = poller.get_status()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_status) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        poller.stop()
        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_concurrent_stop_calls(self):
        """Multiple concurrent stop() calls should be safe."""
        poller = USBDevicePoller(poll_interval=0.05)
        poller._enumerate_devices = MagicMock(return_value=[])

        poller.start()
        time.sleep(0.1)

        errors = []

        def try_stop():
            try:
                poller.stop()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=try_stop) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(errors) == 0


# ===========================================================================
# Tests: USB detection reliability and idempotency
# ===========================================================================

class TestUSBPollerReliability:
    """Tests for USB poller reliability edge cases.

    Covers:
      - Idempotent callback handling (duplicate events are safe)
      - No false triggers during transient enumeration failures
      - Poller recovery after enumeration errors
      - Memory-safe operation under 2GB cap (no unbounded state growth)
    """

    def test_stable_device_no_duplicate_callbacks(self):
        """A continuously present device should fire appeared callback exactly once."""
        appeared_count = [0]
        device = MockGoProDevice()

        poller = USBDevicePoller(
            on_device_appeared=lambda devs: appeared_count.__setitem__(0, appeared_count[0] + 1),
            poll_interval=0.03,
            settling_time=0.0,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        poller.start()
        time.sleep(0.3)  # ~10 polls
        poller.stop()

        assert appeared_count[0] == 1, f"Expected exactly 1 callback, got {appeared_count[0]}"

    def test_transient_enumeration_failure_no_false_disappeared(self):
        """A single poll failure should NOT trigger on_device_disappeared."""
        disappeared_count = [0]
        device = MockGoProDevice()
        poll_count = [0]

        def mock_enumerate():
            poll_count[0] += 1
            # Poll 3 fails, all others succeed
            if poll_count[0] == 3:
                raise OSError("transient WMI error")
            return [device]

        poller = USBDevicePoller(
            on_device_appeared=MagicMock(),
            on_device_disappeared=lambda: disappeared_count.__setitem__(0, disappeared_count[0] + 1),
            poll_interval=0.03,
            settling_time=0.0,
        )
        poller._enumerate_devices = mock_enumerate

        poller.start()
        time.sleep(0.3)
        poller.stop()

        # The transient error should NOT trigger a disappearance callback
        assert disappeared_count[0] == 0, f"False disappearance triggered {disappeared_count[0]} times"

    def test_poller_state_bounded_after_many_cycles(self):
        """Poller internal state should not grow unboundedly over many cycles."""
        device = MockGoProDevice()
        cycle_count = [0]

        def cycling_enumerate():
            cycle_count[0] += 1
            # Alternate between present and absent every 2 polls
            if (cycle_count[0] // 2) % 2 == 0:
                return [device]
            return []

        poller = USBDevicePoller(
            on_device_appeared=MagicMock(),
            on_device_disappeared=MagicMock(),
            poll_interval=0.02,
            settling_time=0.0,
        )
        poller._enumerate_devices = cycling_enumerate

        poller.start()
        time.sleep(0.5)  # ~25 polls
        poller.stop()

        # Verify state is bounded (no list growth, etc.)
        status = poller.get_status()
        assert status["poll_count"] >= 10
        # transition_count should be reasonable (not exponential)
        assert status["transition_count"] <= status["poll_count"]

    def test_force_poll_does_not_trigger_transition_callbacks(self):
        """force_poll() should NOT trigger appeared/disappeared callbacks."""
        appeared_calls = []
        device = MockGoProDevice()

        poller = USBDevicePoller(
            on_device_appeared=lambda devs: appeared_calls.append(devs),
            settling_time=0.0,
        )
        poller._enumerate_devices = MagicMock(return_value=[device])

        # force_poll should just return True/False without callbacks
        result = poller.force_poll()
        assert result is True
        assert len(appeared_calls) == 0

    def test_poller_restart_after_stop(self):
        """Poller should work correctly after stop → start cycle."""
        appeared_calls = []
        device = MockGoProDevice()
        poll_seq = [0]

        def cycling_enumerate():
            poll_seq[0] += 1
            return [device]

        poller = USBDevicePoller(
            on_device_appeared=lambda devs: appeared_calls.append(devs),
            poll_interval=0.03,
            settling_time=0.0,
        )
        poller._enumerate_devices = cycling_enumerate

        # First run — device appears once
        poller.start()
        time.sleep(0.15)
        poller.stop()
        assert len(appeared_calls) == 1

        # After stop, the device was present. On restart, the poller
        # resumes with _device_present=True, so no new appeared callback
        # unless the device goes away and comes back. This is correct behavior.
        appeared_calls.clear()

        # Simulate device gone then back for second run
        poller._device_present = False
        poller._consecutive_present = 0

        poller.start()
        time.sleep(0.15)
        poller.stop()
        assert len(appeared_calls) == 1
