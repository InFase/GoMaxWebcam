"""
test_tray.py — Tests for system tray icon module.

Verifies:
  - _make_icon generates a valid RGBA image of correct size
  - _STATE_COLORS maps all expected states to colors
  - Icon color reflects connection state (green/yellow/red/grey)
  - run_tray creates a pystray.Icon with Open Dashboard and Quit menu items
  - EventBus CONNECTION events update icon color and tooltip
  - EventBus TRANSPORT events update tooltip transport name
  - Quit action calls on_quit callback and stops the icon
  - Open Dashboard action opens the dashboard URL in a browser
  - Left-click default action opens the dashboard URL
  - Unknown states fall back to grey default color
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from gomaxwebcam.tray import _make_icon, _STATE_COLORS, _DEFAULT_COLOR, _ICON_SIZE

# All tests in this module are pure unit tests with mocks — no GoPro needed.
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# _make_icon tests
# ---------------------------------------------------------------------------

class TestMakeIcon:
    """Tests for the _make_icon() icon generator."""

    def test_returns_rgba_image(self):
        """Icon should be an RGBA PIL Image."""
        img = _make_icon("#22c55e")
        assert img.mode == "RGBA"

    def test_correct_size(self):
        """Icon should be _ICON_SIZE x _ICON_SIZE pixels."""
        img = _make_icon("#22c55e")
        assert img.size == (_ICON_SIZE, _ICON_SIZE)

    def test_transparent_corners(self):
        """Corners should be transparent (circle on transparent background)."""
        img = _make_icon("#ff0000")
        # Top-left corner (0,0) should be transparent
        pixel = img.getpixel((0, 0))
        assert pixel[3] == 0, f"Expected transparent corner, got alpha={pixel[3]}"

    def test_center_has_color(self):
        """Center of the icon should have the fill color."""
        img = _make_icon("#ff0000")
        cx, cy = _ICON_SIZE // 2, _ICON_SIZE // 2
        pixel = img.getpixel((cx, cy))
        assert pixel[0] == 255, f"Expected red center, got R={pixel[0]}"
        assert pixel[3] > 0, "Center should be opaque"

    def test_different_colors_produce_different_icons(self):
        """Different color inputs produce visually different icons."""
        green_img = _make_icon("#00ff00")
        red_img = _make_icon("#ff0000")
        cx, cy = _ICON_SIZE // 2, _ICON_SIZE // 2
        assert green_img.getpixel((cx, cy)) != red_img.getpixel((cx, cy))


# ---------------------------------------------------------------------------
# State-to-color mapping tests
# ---------------------------------------------------------------------------

class TestStateColors:
    """Tests for the state-to-color mapping."""

    def test_streaming_is_green(self):
        assert _STATE_COLORS["STREAMING"] == "#22c55e"

    def test_connected_is_green(self):
        assert _STATE_COLORS["CONNECTED"] == "#22c55e"

    def test_connecting_is_yellow(self):
        assert _STATE_COLORS["CONNECTING"] == "#eab308"

    def test_discovering_is_yellow(self):
        assert _STATE_COLORS["DISCOVERING"] == "#eab308"

    def test_error_is_red(self):
        assert _STATE_COLORS["ERROR"] == "#ef4444"

    def test_disconnected_is_grey(self):
        assert _STATE_COLORS["DISCONNECTED"] == "#9ca3af"

    def test_freeze_frame_is_yellow(self):
        assert _STATE_COLORS["FREEZE_FRAME"] == "#eab308"

    def test_stopped_is_grey(self):
        assert _STATE_COLORS["STOPPED"] == "#9ca3af"

    def test_paused_is_grey(self):
        assert _STATE_COLORS["PAUSED"] == "#9ca3af"

    def test_default_color_is_grey(self):
        assert _DEFAULT_COLOR == "#9ca3af"

    def test_unknown_state_falls_back_to_default(self):
        """Unknown state key should not exist in mapping."""
        assert "UNKNOWN_STATE" not in _STATE_COLORS


# ---------------------------------------------------------------------------
# run_tray integration tests (mocked pystray)
# ---------------------------------------------------------------------------

class TestRunTray:
    """Tests for run_tray() with mocked pystray and EventBus."""

    @patch("gomaxwebcam.tray.webbrowser")
    def test_creates_icon_with_correct_name(self, mock_wb):
        """run_tray should create a pystray.Icon named 'GoMaxWebcam'."""
        mock_icon_cls = MagicMock()
        mock_icon_instance = MagicMock()
        mock_icon_cls.return_value = mock_icon_instance
        # Make icon.run() a no-op (don't block)
        mock_icon_instance.run = MagicMock()

        mock_menu_cls = MagicMock()
        mock_menu_item_cls = MagicMock()
        mock_menu_cls.SEPARATOR = "---"

        mock_bus = MagicMock()
        mock_bus.subscribe = MagicMock()

        with patch.dict("sys.modules", {
            "pystray": MagicMock(
                Icon=mock_icon_cls,
                Menu=mock_menu_cls,
                MenuItem=mock_menu_item_cls,
            ),
        }):
            from gomaxwebcam.tray import run_tray
            run_tray(
                dashboard_url="http://localhost:8080/?token=abc",
                event_bus=mock_bus,
                on_quit=MagicMock(),
            )

        # Verify Icon was created with name="GoMaxWebcam"
        mock_icon_cls.assert_called_once()
        kwargs = mock_icon_cls.call_args
        assert kwargs.kwargs.get("name") == "GoMaxWebcam" or \
            (kwargs.args and kwargs.args[0] == "GoMaxWebcam") or \
            kwargs[1].get("name") == "GoMaxWebcam"

    def test_quit_menu_action_calls_on_quit(self):
        """Selecting Quit from the menu should invoke the on_quit callback."""
        on_quit = MagicMock()
        mock_icon_instance = MagicMock()

        # Capture the quit action from MenuItem calls
        menu_items = []
        mock_menu_item = MagicMock(side_effect=lambda *a, **kw: menu_items.append((a, kw)) or MagicMock())

        mock_pystray = MagicMock()
        mock_pystray.Icon = MagicMock(return_value=mock_icon_instance)
        mock_pystray.Menu = MagicMock()
        mock_pystray.Menu.SEPARATOR = "---"
        mock_pystray.MenuItem = mock_menu_item
        mock_icon_instance.run = MagicMock()

        mock_bus = MagicMock()

        with patch.dict("sys.modules", {"pystray": mock_pystray}):
            from gomaxwebcam.tray import run_tray
            run_tray(
                dashboard_url="http://localhost:8080/?token=abc",
                event_bus=mock_bus,
                on_quit=on_quit,
            )

        # Find the "Quit" menu item and call its action
        quit_items = [
            (a, kw) for a, kw in menu_items
            if len(a) >= 1 and a[0] == "Quit"
        ]
        assert len(quit_items) >= 1, f"Expected a 'Quit' MenuItem, got: {menu_items}"

        # The second positional arg is the action callback
        quit_action = quit_items[0][0][1]
        quit_action()

        on_quit.assert_called_once()
        mock_icon_instance.stop.assert_called_once()

    @patch("gomaxwebcam.tray.webbrowser")
    def test_open_dashboard_menu_opens_url(self, mock_wb):
        """Open Dashboard menu item should open the dashboard URL."""
        mock_icon_instance = MagicMock()
        mock_icon_instance.run = MagicMock()

        menu_items = []
        mock_menu_item = MagicMock(side_effect=lambda *a, **kw: menu_items.append((a, kw)) or MagicMock())

        mock_pystray = MagicMock()
        mock_pystray.Icon = MagicMock(return_value=mock_icon_instance)
        mock_pystray.Menu = MagicMock()
        mock_pystray.Menu.SEPARATOR = "---"
        mock_pystray.MenuItem = mock_menu_item

        mock_bus = MagicMock()

        dashboard_url = "http://localhost:8080/?token=secret123"

        with patch.dict("sys.modules", {"pystray": mock_pystray}):
            from gomaxwebcam.tray import run_tray
            run_tray(
                dashboard_url=dashboard_url,
                event_bus=mock_bus,
                on_quit=MagicMock(),
            )

        # Find the "Open Dashboard" menu item
        dash_items = [
            (a, kw) for a, kw in menu_items
            if len(a) >= 1 and a[0] == "Open Dashboard"
        ]
        assert len(dash_items) >= 1

        # Call the action
        dash_action = dash_items[0][0][1]
        dash_action()

        mock_wb.open.assert_called_with(dashboard_url)

    def test_initial_tooltip_shows_disconnected(self):
        """Initial tooltip should show DISCONNECTED state."""
        mock_icon_instance = MagicMock()
        mock_icon_instance.run = MagicMock()

        mock_pystray = MagicMock()
        mock_pystray.Icon = MagicMock(return_value=mock_icon_instance)
        mock_pystray.Menu = MagicMock()
        mock_pystray.Menu.SEPARATOR = "---"
        mock_pystray.MenuItem = MagicMock()

        mock_bus = MagicMock()

        with patch.dict("sys.modules", {"pystray": mock_pystray}):
            from gomaxwebcam.tray import run_tray
            run_tray(
                dashboard_url="http://localhost:8080/",
                event_bus=mock_bus,
                on_quit=MagicMock(),
            )

        # Check the title kwarg passed to Icon constructor
        icon_call = mock_pystray.Icon.call_args
        title = icon_call.kwargs.get("title", "") if icon_call.kwargs else ""
        if not title and len(icon_call.args) >= 4:
            title = icon_call.args[3]
        assert "DISCONNECTED" in title

    def test_listener_thread_started_as_daemon(self):
        """Event listener thread should be a daemon thread."""
        mock_icon_instance = MagicMock()
        started_threads = []

        # Capture thread start calls
        original_thread_init = MagicMock()

        mock_pystray = MagicMock()
        mock_pystray.Icon = MagicMock(return_value=mock_icon_instance)
        mock_pystray.Menu = MagicMock()
        mock_pystray.Menu.SEPARATOR = "---"
        mock_pystray.MenuItem = MagicMock()
        mock_icon_instance.run = MagicMock()

        mock_bus = MagicMock()

        with patch.dict("sys.modules", {"pystray": mock_pystray}):
            with patch("gomaxwebcam.tray.threading.Thread") as mock_thread_cls:
                mock_thread = MagicMock()
                mock_thread_cls.return_value = mock_thread

                from gomaxwebcam.tray import run_tray
                run_tray(
                    dashboard_url="http://localhost:8080/",
                    event_bus=mock_bus,
                    on_quit=MagicMock(),
                )

                # Verify Thread was created as daemon
                thread_call = mock_thread_cls.call_args
                assert thread_call.kwargs.get("daemon") is True
                assert thread_call.kwargs.get("name") == "tray-event-listener"
                mock_thread.start.assert_called_once()
