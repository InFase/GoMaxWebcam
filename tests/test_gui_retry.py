"""
Tests for the manual retry button UI component.

These tests verify:
  - RetryButton is hidden by default
  - RetryButton appears when discovery times out (ERROR/DISCONNECTED states)
  - RetryButton hides when discovery restarts (DISCOVERING state)
  - Clicking retry triggers controller.retry_connection()
  - Button is disabled while retrying (prevents spam clicks)
  - Custom messages display correctly
  - Thread safety: controller callbacks update UI via signals

All tests mock the AppController so no real GoPro hardware is needed.
They also use a minimal QApplication to test Qt widget behavior.

NOTE: We use `not widget.isHidden()` instead of `widget.isVisible()` in tests
because isVisible() requires the entire parent chain up to a shown window
to be visible, while isHidden() checks only the widget's own visibility flag.
This lets us test widget state without needing to show the window on screen.
"""

import sys
import time
from unittest.mock import MagicMock

import pytest

pytestmark = pytest.mark.no_gopro_needed


# We need to check if PyQt6 is available before importing
pytest.importorskip("PyQt6", reason="PyQt6 not installed — GUI tests skipped")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app_controller import AppState
from gui import (

    DashboardWindow,
    RetryButton,
    RETRY_VISIBLE_STATES,
    STATE_COLORS,
    STATE_LABELS,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def qapp():
    """Create a single QApplication for the entire test session.

    Qt requires exactly one QApplication instance. Creating multiple
    causes a crash, so we share one across all tests.
    """
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
    ctrl.on_active_port = None
    ctrl.status_history = []
    ctrl.retry_connection = MagicMock()
    ctrl.stop = MagicMock()
    ctrl.start = MagicMock()
    return ctrl


@pytest.fixture
def retry_button(qapp):
    """Create a standalone RetryButton widget."""
    btn = RetryButton()
    yield btn
    btn.close()


@pytest.fixture
def dashboard(qapp, mock_controller):
    """Create a DashboardWindow with a mock controller."""
    window = DashboardWindow(mock_controller)
    window.show()  # Show window so isVisible() works for child widgets
    QApplication.processEvents()
    yield window
    window.close()
    QApplication.processEvents()


# ── RetryButton unit tests ──────────────────────────────────────────────────

class TestRetryButtonVisibility:
    """Test that the retry button shows/hides correctly."""

    def test_hidden_by_default(self, retry_button):
        """RetryButton must be hidden when first created."""
        assert retry_button.isHidden()

    def test_show_retry_makes_visible(self, retry_button):
        """Calling show_retry() sets the widget visible."""
        retry_button.show_retry()
        assert not retry_button.isHidden()

    def test_hide_retry_makes_invisible(self, retry_button):
        """Calling hide_retry() hides the button."""
        retry_button.show_retry()
        assert not retry_button.isHidden()
        retry_button.hide_retry()
        assert retry_button.isHidden()

    def test_show_with_custom_message(self, retry_button):
        """show_retry() with a message updates the label text."""
        custom_msg = "Camera powered off. Plug in and retry."
        retry_button.show_retry(message=custom_msg)
        assert not retry_button.isHidden()
        assert retry_button._label.text() == custom_msg

    def test_show_without_message_keeps_default(self, retry_button):
        """show_retry() without message keeps the default text."""
        default_text = retry_button._label.text()
        retry_button.show_retry()
        assert retry_button._label.text() == default_text


class TestRetryButtonState:
    """Test button enabled/disabled states."""

    def test_button_enabled_after_show(self, retry_button):
        """Button is enabled when show_retry() is called."""
        retry_button.show_retry()
        assert retry_button._button.isEnabled()
        assert retry_button._button.text() == "🔄  Retry Discovery"

    def test_set_retrying_disables_button(self, retry_button):
        """set_retrying() disables the button and changes text."""
        retry_button.show_retry()
        retry_button.set_retrying()
        assert not retry_button._button.isEnabled()
        assert retry_button._button.text() == "Retrying…"

    def test_show_retry_resets_from_retrying(self, retry_button):
        """Calling show_retry() again after retrying re-enables the button."""
        retry_button.show_retry()
        retry_button.set_retrying()
        assert not retry_button._button.isEnabled()

        retry_button.show_retry()
        assert retry_button._button.isEnabled()
        assert retry_button._button.text() == "🔄  Retry Discovery"


class TestRetryButtonClick:
    """Test click behavior."""

    def test_click_emits_signal(self, retry_button):
        """Clicking the button emits retry_requested signal."""
        signal_received = []
        retry_button.retry_requested.connect(lambda: signal_received.append(True))

        retry_button.show_retry()
        QTest.mouseClick(retry_button._button, Qt.MouseButton.LeftButton)

        assert len(signal_received) == 1

    def test_click_disables_button(self, retry_button):
        """Button is disabled after click to prevent spam."""
        retry_button.show_retry()
        QTest.mouseClick(retry_button._button, Qt.MouseButton.LeftButton)

        assert not retry_button._button.isEnabled()
        assert retry_button._button.text() == "Retrying…"

    def test_disabled_button_does_not_emit(self, retry_button):
        """A disabled button click should not emit the signal again."""
        signal_count = []
        retry_button.retry_requested.connect(lambda: signal_count.append(1))

        retry_button.show_retry()
        # First click
        QTest.mouseClick(retry_button._button, Qt.MouseButton.LeftButton)
        assert len(signal_count) == 1

        # Second click on disabled button
        QTest.mouseClick(retry_button._button, Qt.MouseButton.LeftButton)
        assert len(signal_count) == 1  # No additional signal


# ── DashboardWindow integration tests ───────────────────────────────────────

class TestDashboardRetryIntegration:
    """Test retry button behavior within the DashboardWindow."""

    def test_retry_hidden_initially(self, dashboard):
        """Retry button is hidden when dashboard is first created."""
        assert not dashboard.retry_button.isVisible()

    def test_retry_shown_on_error_state(self, dashboard):
        """Retry button appears when state changes to ERROR."""
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        assert dashboard.retry_button.isVisible()

    def test_retry_shown_on_disconnected_state(self, dashboard):
        """Retry button appears when state changes to DISCONNECTED."""
        dashboard._on_state_changed(AppState.DISCONNECTED)
        QApplication.processEvents()
        assert dashboard.retry_button.isVisible()

    def test_retry_shown_on_stopped_state(self, dashboard):
        """Retry button appears when state changes to STOPPED."""
        dashboard._on_state_changed(AppState.STOPPED)
        QApplication.processEvents()
        assert dashboard.retry_button.isVisible()

    def test_retry_hidden_on_discovering(self, dashboard):
        """Retry button hides when discovery starts (DISCOVERING state)."""
        # First show it
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        assert dashboard.retry_button.isVisible()

        # Then state changes to DISCOVERING
        dashboard._on_state_changed(AppState.DISCOVERING)
        QApplication.processEvents()
        assert not dashboard.retry_button.isVisible()

    def test_retry_hidden_on_streaming(self, dashboard):
        """Retry button hides when streaming starts."""
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        assert dashboard.retry_button.isVisible()

        dashboard._on_state_changed(AppState.STREAMING)
        QApplication.processEvents()
        assert not dashboard.retry_button.isVisible()

    def test_retry_hidden_on_connecting(self, dashboard):
        """Retry button hides during connection phase."""
        dashboard._on_state_changed(AppState.DISCONNECTED)
        QApplication.processEvents()
        assert dashboard.retry_button.isVisible()

        dashboard._on_state_changed(AppState.CONNECTING)
        QApplication.processEvents()
        assert not dashboard.retry_button.isVisible()

    def test_retry_hidden_on_reconnecting(self, dashboard):
        """Retry button hides during auto-reconnection."""
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        assert dashboard.retry_button.isVisible()

        dashboard._on_state_changed(AppState.RECONNECTING)
        QApplication.processEvents()
        assert not dashboard.retry_button.isVisible()

    def test_error_state_shows_specific_message(self, dashboard):
        """ERROR state shows a message about retries being exhausted."""
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        text = dashboard.retry_button._label.text()
        assert "not found" in text.lower() or "retries" in text.lower()

    def test_disconnected_state_shows_specific_message(self, dashboard):
        """DISCONNECTED state shows a message about reconnecting."""
        dashboard._on_state_changed(AppState.DISCONNECTED)
        QApplication.processEvents()
        text = dashboard.retry_button._label.text()
        assert "disconnect" in text.lower() or "reconnect" in text.lower()


class TestDashboardRetryAction:
    """Test that clicking retry in the dashboard triggers controller actions."""

    def test_retry_calls_controller(self, dashboard, mock_controller):
        """Clicking retry triggers controller.retry_connection()."""
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()

        # Click the retry button
        QTest.mouseClick(
            dashboard.retry_button._button,
            Qt.MouseButton.LeftButton,
        )
        QApplication.processEvents()

        # Give the background thread a moment to start
        time.sleep(0.15)
        QApplication.processEvents()

        mock_controller.retry_connection.assert_called_once()

    def test_retry_button_disabled_while_retrying(self, dashboard):
        """Button is disabled immediately after clicking retry."""
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()

        QTest.mouseClick(
            dashboard.retry_button._button,
            Qt.MouseButton.LeftButton,
        )
        QApplication.processEvents()

        assert not dashboard.retry_button._button.isEnabled()

    def test_retry_button_re_enabled_on_subsequent_error(self, dashboard):
        """If retry fails and we go back to ERROR, button is re-enabled."""
        # First failure
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        QTest.mouseClick(
            dashboard.retry_button._button,
            Qt.MouseButton.LeftButton,
        )
        QApplication.processEvents()
        assert not dashboard.retry_button._button.isEnabled()

        # Discovery starts
        dashboard._on_state_changed(AppState.DISCOVERING)
        QApplication.processEvents()
        assert not dashboard.retry_button.isVisible()

        # Discovery fails again
        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        assert dashboard.retry_button.isVisible()
        assert dashboard.retry_button._button.isEnabled()


class TestDashboardStateIndicator:
    """Test the state indicator updates correctly."""

    def test_all_states_have_colors(self):
        """Every AppState should have a defined color."""
        for state in AppState:
            assert state in STATE_COLORS, f"Missing color for {state.name}"

    def test_all_states_have_labels(self):
        """Every AppState should have a human-readable label."""
        for state in AppState:
            assert state in STATE_LABELS, f"Missing label for {state.name}"

    def test_state_indicator_updates(self, dashboard):
        """State indicator changes color and text on state change."""
        dashboard._on_state_changed(AppState.STREAMING)
        QApplication.processEvents()
        assert "Streaming" in dashboard._state_label.text()

        dashboard._on_state_changed(AppState.ERROR)
        QApplication.processEvents()
        assert "Error" in dashboard._state_label.text()

    def test_retry_visible_states_are_correct(self):
        """Only ERROR, DISCONNECTED, STOPPED should show the retry button."""
        assert RETRY_VISIBLE_STATES == {
            AppState.ERROR,
            AppState.DISCONNECTED,
            AppState.STOPPED,
        }


class TestDashboardStatusLog:
    """Test the status log display."""

    def test_status_message_appended(self, dashboard):
        """Status messages appear in the log area."""
        dashboard._on_status_message("Test message", "info")
        QApplication.processEvents()
        content = dashboard._log_area.toPlainText()
        assert "Test message" in content

    def test_error_message_styled(self, dashboard):
        """Error messages have red styling in HTML."""
        dashboard._on_status_message("Something broke", "error")
        QApplication.processEvents()
        html = dashboard._log_area.toHtml()
        assert "Something broke" in html
        # Qt lowercases hex colors in output
        assert "#f44336" in html.lower()

    def test_success_message_styled(self, dashboard):
        """Success messages have green styling."""
        dashboard._on_status_message("Connected!", "success")
        QApplication.processEvents()
        html = dashboard._log_area.toHtml()
        assert "Connected!" in html
        assert "#4caf50" in html.lower()


class TestDashboardCameraInfo:
    """Test camera info display."""

    def test_battery_shown(self, dashboard):
        """Battery level is displayed when camera info arrives."""
        dashboard._on_camera_info({"battery_level": 75})
        QApplication.processEvents()
        assert not dashboard._battery_label.isHidden()
        assert "75%" in dashboard._battery_label.text()

    def test_battery_hidden_initially(self, dashboard):
        """Battery label is hidden before any camera info."""
        assert dashboard._battery_label.isHidden()

    def test_no_battery_key_doesnt_show(self, dashboard):
        """Camera info without battery_level doesn't show the label."""
        dashboard._on_camera_info({"some_other_key": 42})
        QApplication.processEvents()
        assert dashboard._battery_label.isHidden()


class TestDashboardControllerWiring:
    """Test that controller callbacks are properly wired to signals."""

    def test_controller_callbacks_set(self, dashboard, mock_controller):
        """After dashboard creation, controller callbacks should be set."""
        # The callbacks should be callables that emit signals
        assert mock_controller.on_state_change is not None
        assert mock_controller.on_status is not None
        assert mock_controller.on_camera_info is not None
        assert mock_controller.on_active_port is not None

    def test_stop_button_calls_controller(self, dashboard, mock_controller):
        """Clicking stop calls controller.stop()."""
        QTest.mouseClick(dashboard._stop_btn, Qt.MouseButton.LeftButton)
        QApplication.processEvents()
        mock_controller.stop.assert_called_once()


class TestDashboardActivePortIndicator:
    """Test the read-only active port indicator in the dashboard header."""

    def test_port_label_hidden_initially(self, dashboard):
        """Port label is hidden before any port is selected."""
        assert dashboard.port_label.isHidden()

    def test_port_label_shown_after_active_port(self, dashboard):
        """Port label becomes visible when active port is set."""
        dashboard._on_active_port(8554)
        QApplication.processEvents()
        assert not dashboard.port_label.isHidden()
        assert "8554" in dashboard.port_label.text()

    def test_port_label_shows_auto_selected_port(self, dashboard):
        """Port label shows the auto-selected port (not the configured one)."""
        dashboard._on_active_port(8555)
        QApplication.processEvents()
        assert "8555" in dashboard.port_label.text()

    def test_port_label_updates_on_new_port(self, dashboard):
        """Port label updates when active port changes."""
        dashboard._on_active_port(8554)
        QApplication.processEvents()
        assert "8554" in dashboard.port_label.text()

        dashboard._on_active_port(8556)
        QApplication.processEvents()
        assert "8556" in dashboard.port_label.text()
        assert "8554" not in dashboard.port_label.text()

    def test_port_label_is_read_only(self, dashboard):
        """Port label is a QLabel (not editable), separate from config spinner."""
        from PyQt6.QtWidgets import QLabel
        assert isinstance(dashboard.port_label, QLabel)

    def test_port_label_accessible_via_property(self, dashboard):
        """Port label is accessible via the public port_label property."""
        assert dashboard.port_label is dashboard._port_label


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
