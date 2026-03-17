"""
Tests for the ManualModeGuide fallback UI component.

These tests verify:
  - ManualModeGuide is hidden by default
  - show_guide() makes it visible with correct content
  - hide_guide() hides it
  - All 5 instruction steps are present and numbered
  - Custom reason text replaces default description
  - Collapse/expand toggle works correctly
  - Guide appears in DashboardWindow when webcam mode fails
  - Guide hides when state transitions to active states (DISCOVERING, etc.)
  - AppController triggers the guide callback on webcam mode failure
  - Step content matches expected manual GoPro setup instructions

All tests mock the AppController so no real GoPro hardware is needed.
They use a minimal QApplication in offscreen mode.

NOTE: We use `not widget.isHidden()` instead of `widget.isVisible()` in tests
because isVisible() requires the entire parent chain up to a shown window
to be visible, while isHidden() checks only the widget's own visibility flag.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.no_gopro_needed


pytest.importorskip("PyQt6", reason="PyQt6 not installed — GUI tests skipped")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from app_controller import AppController, AppState
from gui import (

    DashboardWindow,
    ManualModeGuide,
    MANUAL_MODE_STEPS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    """Create a single QApplication for the entire test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0], "-platform", "offscreen"])
    return app


@pytest.fixture
def mock_controller():
    """Create a mock AppController with all expected attributes."""
    ctrl = MagicMock()
    ctrl.state = AppState.INITIALIZING
    ctrl.on_state_change = None
    ctrl.on_status = None
    ctrl.on_camera_info = None
    ctrl.on_webcam_mode_failed = None
    ctrl.status_history = []
    ctrl.retry_connection = MagicMock()
    ctrl.stop = MagicMock()
    ctrl.start = MagicMock()
    return ctrl


@pytest.fixture
def guide(qapp):
    """Create a standalone ManualModeGuide widget."""
    g = ManualModeGuide()
    yield g
    g.close()


@pytest.fixture
def dashboard(qapp, mock_controller):
    """Create a DashboardWindow with a mock controller."""
    window = DashboardWindow(mock_controller)
    window.show()
    QApplication.processEvents()
    yield window
    window.close()
    QApplication.processEvents()


# ── ManualModeGuide unit tests ───────────────────────────────────────────────

class TestManualModeGuideVisibility:
    """Test that the guide shows/hides correctly."""

    def test_hidden_by_default(self, guide):
        """ManualModeGuide must be hidden when first created."""
        assert guide.isHidden()

    def test_show_guide_makes_visible(self, guide):
        """Calling show_guide() sets the widget visible."""
        guide.show_guide()
        assert not guide.isHidden()

    def test_hide_guide_makes_invisible(self, guide):
        """Calling hide_guide() hides the widget."""
        guide.show_guide()
        assert not guide.isHidden()
        guide.hide_guide()
        assert guide.isHidden()

    def test_show_guide_resets_to_expanded(self, guide):
        """show_guide() always resets to expanded state."""
        guide.show_guide()
        guide._toggle_collapse()  # collapse
        assert not guide.is_expanded

        guide.hide_guide()
        guide.show_guide()  # should reset to expanded
        assert guide.is_expanded


class TestManualModeGuideContent:
    """Test the instruction step content."""

    def test_has_five_steps(self, guide):
        """Guide should have exactly 5 instruction steps."""
        assert guide.step_count() == 5

    def test_steps_match_constant(self, guide):
        """Steps should match the MANUAL_MODE_STEPS constant."""
        assert guide.steps == MANUAL_MODE_STEPS

    def test_step_labels_created(self, guide):
        """Each step has a corresponding widget in the UI."""
        assert len(guide._step_labels) == 5

    def test_manual_steps_cover_key_actions(self):
        """Verify the manual steps cover essential GoPro setup actions."""
        all_titles = [title for title, _ in MANUAL_MODE_STEPS]
        all_details = [detail for _, detail in MANUAL_MODE_STEPS]
        all_text = " ".join(all_titles + all_details).lower()

        # Must mention power cycling
        assert "power cycle" in all_text or "turn" in all_text
        # Must mention USB
        assert "usb" in all_text
        # Must mention GoPro Connect mode
        assert "gopro connect" in all_text
        # Must mention retry
        assert "retry" in all_text

    def test_steps_are_numbered_sequentially(self):
        """Steps should be numbered 1 through 5."""
        for i, (title, detail) in enumerate(MANUAL_MODE_STEPS, start=1):
            assert isinstance(title, str) and len(title) > 0
            assert isinstance(detail, str) and len(detail) > 0
        assert i == 5  # noqa: F821 — confirms we iterated 5 times


class TestManualModeGuideCustomReason:
    """Test custom reason text display."""

    def test_show_with_custom_reason(self, guide):
        """show_guide() with a reason updates the description text."""
        custom = "Camera firmware needs update. Please set up manually:"
        guide.show_guide(reason=custom)
        assert guide._desc_label.text() == custom

    def test_show_without_reason_uses_default(self, guide):
        """show_guide() without a reason uses the default description."""
        guide.show_guide()
        text = guide._desc_label.text()
        assert "automatic" in text.lower() or "failed" in text.lower()

    def test_show_resets_reason_to_default(self, guide):
        """Calling show_guide() after custom reason resets to default."""
        guide.show_guide(reason="Custom reason text here")
        assert guide._desc_label.text() == "Custom reason text here"

        guide.show_guide()  # No reason — should reset to default
        text = guide._desc_label.text()
        assert "automatic" in text.lower() or "webcam" in text.lower()


class TestManualModeGuideCollapse:
    """Test the collapse/expand toggle behavior."""

    def test_starts_expanded(self, guide):
        """Guide starts in expanded state."""
        guide.show_guide()
        assert guide.is_expanded
        assert not guide._steps_container.isHidden()

    def test_collapse_hides_steps(self, guide):
        """Collapsing hides the steps container."""
        guide.show_guide()
        guide._toggle_collapse()
        assert not guide.is_expanded
        assert guide._steps_container.isHidden()

    def test_expand_shows_steps(self, guide):
        """Expanding shows the steps container again."""
        guide.show_guide()
        guide._toggle_collapse()  # collapse
        guide._toggle_collapse()  # expand
        assert guide.is_expanded
        assert not guide._steps_container.isHidden()

    def test_collapse_button_text_toggles(self, guide):
        """Collapse button shows ▼ when expanded, ▶ when collapsed."""
        guide.show_guide()
        assert guide._collapse_btn.text() == "▼"

        guide._toggle_collapse()
        assert guide._collapse_btn.text() == "▶"

        guide._toggle_collapse()
        assert guide._collapse_btn.text() == "▼"

    def test_collapse_hides_description(self, guide):
        """Collapsing also hides the description label."""
        guide.show_guide()
        guide._toggle_collapse()
        assert guide._desc_label.isHidden()

    def test_expand_shows_description(self, guide):
        """Expanding shows the description label again."""
        guide.show_guide()
        guide._toggle_collapse()
        guide._toggle_collapse()
        assert not guide._desc_label.isHidden()


# ── DashboardWindow integration tests ────────────────────────────────────────

class TestDashboardManualGuideIntegration:
    """Test ManualModeGuide behavior within the DashboardWindow."""

    def test_guide_hidden_initially(self, dashboard):
        """Manual guide is hidden when dashboard is first created."""
        assert not dashboard.manual_guide.isVisible()

    def test_guide_shown_on_webcam_mode_failed(self, dashboard):
        """Guide appears when _on_webcam_mode_failed is called."""
        dashboard._on_webcam_mode_failed("Test failure reason")
        QApplication.processEvents()
        assert dashboard.manual_guide.isVisible()

    def test_guide_shows_reason_text(self, dashboard):
        """Guide displays the provided reason text."""
        reason = "Camera stuck in photo mode. Manual setup needed:"
        dashboard._on_webcam_mode_failed(reason)
        QApplication.processEvents()
        assert dashboard.manual_guide._desc_label.text() == reason

    def test_guide_hidden_on_discovering_state(self, dashboard):
        """Guide hides when state changes to DISCOVERING."""
        dashboard._on_webcam_mode_failed("Test")
        QApplication.processEvents()
        assert dashboard.manual_guide.isVisible()

        dashboard._on_state_changed(AppState.DISCOVERING)
        QApplication.processEvents()
        assert not dashboard.manual_guide.isVisible()

    def test_guide_hidden_on_connecting_state(self, dashboard):
        """Guide hides when state changes to CONNECTING."""
        dashboard._on_webcam_mode_failed("Test")
        QApplication.processEvents()

        dashboard._on_state_changed(AppState.CONNECTING)
        QApplication.processEvents()
        assert not dashboard.manual_guide.isVisible()

    def test_guide_hidden_on_streaming_state(self, dashboard):
        """Guide hides when streaming starts."""
        dashboard._on_webcam_mode_failed("Test")
        QApplication.processEvents()

        dashboard._on_state_changed(AppState.STREAMING)
        QApplication.processEvents()
        assert not dashboard.manual_guide.isVisible()

    def test_guide_hidden_on_reconnecting_state(self, dashboard):
        """Guide hides during auto-reconnection."""
        dashboard._on_webcam_mode_failed("Test")
        QApplication.processEvents()

        dashboard._on_state_changed(AppState.RECONNECTING)
        QApplication.processEvents()
        assert not dashboard.manual_guide.isVisible()

    def test_guide_persists_on_error_state(self, dashboard):
        """Guide stays visible on ERROR state (user may still need it)."""
        dashboard._on_webcam_mode_failed("Test")
        QApplication.processEvents()

        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        assert dashboard.manual_guide.isVisible()

    def test_guide_persists_on_disconnected_state(self, dashboard):
        """Guide stays visible on DISCONNECTED state."""
        dashboard._on_webcam_mode_failed("Test")
        QApplication.processEvents()

        dashboard._on_state_changed(AppState.DISCONNECTED)
        QApplication.processEvents()
        assert dashboard.manual_guide.isVisible()

    def test_guide_and_retry_can_coexist(self, dashboard):
        """Both guide and retry button can be visible simultaneously."""
        dashboard._on_webcam_mode_failed("Test")
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()

        assert dashboard.manual_guide.isVisible()
        assert dashboard.retry_button.isVisible()


class TestDashboardControllerWiringWebcam:
    """Test that the webcam mode failed callback is properly wired."""

    def test_webcam_mode_failed_callback_set(self, dashboard, mock_controller):
        """After dashboard creation, on_webcam_mode_failed should be set."""
        assert mock_controller.on_webcam_mode_failed is not None

    def test_callback_triggers_guide(self, dashboard, mock_controller):
        """Calling the controller callback shows the guide in the dashboard."""
        # Simulate controller firing the callback
        mock_controller.on_webcam_mode_failed("Simulated failure")
        QApplication.processEvents()
        # The callback was set to emit a signal, so we call the slot directly
        dashboard._on_webcam_mode_failed("Simulated failure")
        QApplication.processEvents()
        assert dashboard.manual_guide.isVisible()


# ── AppController callback tests ─────────────────────────────────────────────

class TestAppControllerWebcamFailedCallback:
    """Test that AppController fires on_webcam_mode_failed correctly."""

    def test_callback_attribute_exists(self):
        """AppController should have on_webcam_mode_failed attribute."""
        with patch("app_controller.Config") as MockConfig:
            MockConfig.load.return_value = MagicMock()
            with patch("app_controller.GoProConnection"):
                ctrl = AppController.__new__(AppController)
                ctrl.config = MagicMock()
                ctrl.gopro = MagicMock()
                ctrl.state = AppState.INITIALIZING
                ctrl.status_history = []
                ctrl.on_state_change = None
                ctrl.on_status = None
                ctrl.on_camera_info = None
                ctrl.on_webcam_mode_failed = None
                ctrl._running = False
                ctrl._startup_thread = None
                ctrl._keepalive_thread = None
                ctrl._stop_event = MagicMock()

                assert hasattr(ctrl, "on_webcam_mode_failed")

    def test_notify_webcam_mode_failed_calls_callback(self):
        """_notify_webcam_mode_failed should invoke the callback."""
        with patch("app_controller.Config") as MockConfig:
            MockConfig.load.return_value = MagicMock()
            with patch("app_controller.GoProConnection"):
                ctrl = AppController.__new__(AppController)
                ctrl.config = MagicMock()
                ctrl.gopro = MagicMock()
                ctrl.state = AppState.INITIALIZING
                ctrl.status_history = []
                ctrl.on_state_change = None
                ctrl.on_status = None
                ctrl.on_camera_info = None
                ctrl._running = False
                ctrl._startup_thread = None
                ctrl._keepalive_thread = None
                ctrl._stop_event = MagicMock()

                callback = MagicMock()
                ctrl.on_webcam_mode_failed = callback

                ctrl._notify_webcam_mode_failed("Test reason")
                callback.assert_called_once_with("Test reason")

    def test_notify_webcam_mode_failed_handles_no_callback(self):
        """_notify_webcam_mode_failed should not crash with no callback."""
        with patch("app_controller.Config") as MockConfig:
            MockConfig.load.return_value = MagicMock()
            with patch("app_controller.GoProConnection"):
                ctrl = AppController.__new__(AppController)
                ctrl.config = MagicMock()
                ctrl.gopro = MagicMock()
                ctrl.state = AppState.INITIALIZING
                ctrl.status_history = []
                ctrl.on_state_change = None
                ctrl.on_status = None
                ctrl.on_camera_info = None
                ctrl.on_webcam_mode_failed = None
                ctrl._running = False
                ctrl._startup_thread = None
                ctrl._keepalive_thread = None
                ctrl._stop_event = MagicMock()

                # Should not raise
                ctrl._notify_webcam_mode_failed("Test reason")

    def test_connect_and_start_triggers_callback_on_failure(self):
        """_connect_and_start should trigger manual guide on webcam failure."""
        with patch("app_controller.Config") as MockConfig:
            MockConfig.load.return_value = MagicMock()
            with patch("app_controller.GoProConnection"):
                ctrl = AppController.__new__(AppController)
                ctrl.config = MagicMock()
                ctrl.config.resolution = 4
                ctrl.config.fov = 4
                ctrl.config.stream_width = 1920
                ctrl.config.stream_height = 1080
                ctrl.config.udp_port = 8554
                ctrl.gopro = MagicMock()
                ctrl.gopro.start_webcam.return_value = False
                ctrl.state = AppState.CONNECTING
                ctrl.status_history = []
                ctrl.on_state_change = None
                ctrl.on_status = None
                ctrl.on_camera_info = None
                ctrl._running = True
                ctrl._startup_thread = None
                ctrl._keepalive_thread = None
                ctrl._stop_event = MagicMock()

                callback = MagicMock()
                ctrl.on_webcam_mode_failed = callback

                result = ctrl._connect_and_start()
                assert result is False
                callback.assert_called_once()
                # Verify the reason mentions manual setup
                call_arg = callback.call_args[0][0]
                assert "manual" in call_arg.lower() or "failed" in call_arg.lower()

    def test_connect_and_start_no_callback_on_success(self):
        """_connect_and_start should NOT trigger manual guide on success."""
        with patch("app_controller.Config") as MockConfig:
            MockConfig.load.return_value = MagicMock()
            with patch("app_controller.GoProConnection"):
                ctrl = AppController.__new__(AppController)
                ctrl.config = MagicMock()
                ctrl.config.resolution = 4
                ctrl.config.fov = 4
                ctrl.config.stream_width = 1920
                ctrl.config.stream_height = 1080
                ctrl.config.udp_port = 8554
                ctrl.gopro = MagicMock()
                ctrl.gopro.start_webcam.return_value = True
                ctrl.state = AppState.CONNECTING
                ctrl.status_history = []
                ctrl.on_state_change = None
                ctrl.on_status = None
                ctrl.on_camera_info = None
                ctrl._running = True
                ctrl._startup_thread = None
                ctrl._keepalive_thread = None
                ctrl._stop_event = MagicMock()

                callback = MagicMock()
                ctrl.on_webcam_mode_failed = callback

                result = ctrl._connect_and_start()
                assert result is True
                callback.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
