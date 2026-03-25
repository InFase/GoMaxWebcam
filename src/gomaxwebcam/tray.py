"""
tray.py -- System tray integration for GoMaxWebcam v2.

Uses pystray for the system tray icon with:
  - Left-click: open dashboard URL in default browser
  - Right-click menu: Status, Open Dashboard, Quit
  - Icon color changes by state:
      green  = streaming / connected
      yellow = connecting / discovering / freeze-frame
      red    = error
      grey   = paused / stopped / disconnected
  - Tooltip: "GoMaxWebcam - [state] | [transport]"

Icons are generated programmatically as colored circles using Pillow
(no bundled icon files).

The tray subscribes to the EventBus for CONNECTION and TRANSPORT events
to update icon color and tooltip in real time.  Cross-thread communication
between the EventBus (asyncio loop thread) and pystray (main thread) uses
a thread-safe ``queue.Queue`` to avoid asyncio.Queue cross-loop issues.
"""

from __future__ import annotations

import logging
import queue
import threading
import webbrowser
from typing import Callable, Optional

from PIL import Image, ImageDraw

from gomaxwebcam.events import EventBus, EventType

log = logging.getLogger("gomaxwebcam.tray")

# Icon size (pixels)
_ICON_SIZE = 64

# State-to-color mapping
_STATE_COLORS: dict[str, str] = {
    "STREAMING": "#22c55e",    # green
    "CONNECTED": "#22c55e",    # green (connected, ready)
    "CONNECTING": "#eab308",   # yellow
    "DISCOVERING": "#eab308",  # yellow
    "ERROR": "#ef4444",        # red
    "DISCONNECTED": "#9ca3af", # grey
    "FREEZE_FRAME": "#eab308", # yellow (degraded)
    "STOPPED": "#9ca3af",      # grey
    "PAUSED": "#9ca3af",       # grey
}

_DEFAULT_COLOR = "#9ca3af"  # grey


# ---------------------------------------------------------------------------
# Icon generation
# ---------------------------------------------------------------------------

def _make_icon(color: str) -> Image.Image:
    """Generate a simple colored circle icon.

    Args:
        color: Hex color string (e.g. "#22c55e").

    Returns:
        PIL Image suitable for pystray.
    """
    img = Image.new("RGBA", (_ICON_SIZE, _ICON_SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse(
        [margin, margin, _ICON_SIZE - margin, _ICON_SIZE - margin],
        fill=color,
    )
    return img


# ---------------------------------------------------------------------------
# Tray runner
# ---------------------------------------------------------------------------

def run_tray(
    dashboard_url: str,
    event_bus: EventBus,
    on_quit: Callable[[], None],
) -> None:
    """Create and run the system tray icon (blocks on main thread).

    Args:
        dashboard_url: Full URL including auth token for the dashboard.
        event_bus: EventBus to subscribe to for state updates.
        on_quit: Callback invoked when the user selects Quit.
    """
    import pystray
    from pystray import MenuItem, Menu

    # Mutable state shared between event listener and menu
    state_info: dict[str, str] = {
        "connection": "DISCONNECTED",
        "transport": "none",
    }

    icon: Optional[pystray.Icon] = None

    # Thread-safe queue for forwarding EventBus events to the tray thread
    _event_queue: queue.Queue = queue.Queue(maxsize=64)

    def _open_dashboard() -> None:
        webbrowser.open(dashboard_url)

    def _on_left_click(icon_obj: pystray.Icon, item: object) -> None:
        _open_dashboard()

    def _get_status_label() -> str:
        return f"{state_info['connection']} | {state_info['transport']}"

    def _update_icon_state(connection_state: str, transport_name: str) -> None:
        """Update icon color and tooltip from event data."""
        state_info["connection"] = connection_state
        state_info["transport"] = transport_name

        color = _STATE_COLORS.get(connection_state, _DEFAULT_COLOR)
        new_icon = _make_icon(color)
        tooltip = f"GoMaxWebcam - {connection_state} | {transport_name}"

        if icon is not None:
            icon.icon = new_icon
            icon.title = tooltip

    def _drain_event_queue() -> None:
        """Process any pending events from the thread-safe queue.

        Called from the tray's icon polling or after each menu update.
        """
        while True:
            try:
                event = _event_queue.get_nowait()
            except queue.Empty:
                break

            if event.type == EventType.CONNECTION:
                new_state = event.data.get("new_state", "DISCONNECTED")
                transport = event.data.get("transport", state_info["transport"])
                _update_icon_state(new_state, transport)
            elif event.type == EventType.TRANSPORT:
                new_transport = event.data.get("new_transport", "none")
                _update_icon_state(state_info["connection"], new_transport)

    def _quit_action() -> None:
        if icon is not None:
            icon.stop()
        on_quit()

    # Build menu
    menu = Menu(
        MenuItem(
            lambda _text: f"Status: {_get_status_label()}",
            action=None,
            enabled=False,
        ),
        Menu.SEPARATOR,
        MenuItem("Open Dashboard", lambda: _open_dashboard()),
        Menu.SEPARATOR,
        MenuItem("Quit", lambda: _quit_action()),
    )

    icon = pystray.Icon(
        name="GoMaxWebcam",
        icon=_make_icon(_DEFAULT_COLOR),
        title="GoMaxWebcam - DISCONNECTED",
        menu=menu,
    )

    # On platforms that support default action (left-click), wire it up
    icon.default = _on_left_click

    # -- Subscribe to EventBus in a background thread --
    # Uses its own asyncio loop to iterate the async EventSubscription,
    # then forwards events to the thread-safe _event_queue for the tray
    # thread to consume.  This avoids asyncio.Queue cross-loop issues.
    def _event_listener() -> None:
        """Listen for connection/transport events and forward to thread-safe queue."""
        import asyncio

        async def _listen() -> None:
            subscription = event_bus.subscribe(
                event_types={EventType.CONNECTION, EventType.TRANSPORT},
            )
            try:
                async for event in subscription:
                    try:
                        _event_queue.put_nowait(event)
                    except queue.Full:
                        # Drop oldest and retry
                        try:
                            _event_queue.get_nowait()
                        except queue.Empty:
                            pass
                        try:
                            _event_queue.put_nowait(event)
                        except queue.Full:
                            pass

                    # Update icon from the event on this thread too,
                    # since pystray icon property updates are thread-safe
                    if event.type == EventType.CONNECTION:
                        new_state = event.data.get("new_state", "DISCONNECTED")
                        transport = event.data.get(
                            "transport", state_info["transport"]
                        )
                        _update_icon_state(new_state, transport)
                    elif event.type == EventType.TRANSPORT:
                        new_transport = event.data.get("new_transport", "none")
                        _update_icon_state(state_info["connection"], new_transport)
            except Exception:
                log.debug("Tray event listener stopped", exc_info=True)

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_listen())
        except Exception:
            pass
        finally:
            loop.close()

    listener_thread = threading.Thread(
        target=_event_listener,
        name="tray-event-listener",
        daemon=True,
    )
    listener_thread.start()

    # Blocks until icon.stop() is called
    log.info("System tray started")
    icon.run()
    log.info("System tray stopped")
