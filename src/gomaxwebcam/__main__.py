"""
__main__.py -- Application entry point for GoMaxWebcam v2.

Handles:
  - CLI arg parsing (--headless, --debug)
  - Headless detection (DISPLAY/WAYLAND_DISPLAY on Linux)
  - Single-instance enforcement via OS file lock
  - asyncio event loop on background thread
  - FastAPI/uvicorn on the async loop (dynamic port, random token)
  - pystray on main thread (desktop mode) or asyncio block (headless)
  - SIGTERM/SIGINT graceful shutdown with best-effort stop_stream
  - Second-instance detection: open dashboard URL and exit

Lock file format (one value per line):
  Line 1: dashboard URL (e.g. http://127.0.0.1:54321/?token=abc)
  Line 2: PID
"""

from __future__ import annotations

import argparse
import asyncio
import atexit
import logging
import os
import secrets
import signal
import socket
import sys
import threading
import webbrowser
from pathlib import Path
from typing import Optional

import uvicorn
from platformdirs import user_runtime_dir

log = logging.getLogger("gomaxwebcam")

_APP_NAME = "GoMaxWebcam-v2"
_LOCK_FILENAME = "gomaxwebcam.lock"


# ---------------------------------------------------------------------------
# Headless detection
# ---------------------------------------------------------------------------

def _detect_headless() -> bool:
    """Return True when no desktop session is available."""
    if sys.platform == "win32" or sys.platform == "darwin":
        return False
    # Linux / other: check X11 or Wayland
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


# ---------------------------------------------------------------------------
# Single-instance lock (OS-level file lock)
# ---------------------------------------------------------------------------

class _InstanceLock:
    """OS-level file lock for single-instance enforcement.

    On Windows uses msvcrt.locking; on POSIX uses fcntl.flock.
    The lock file also stores the dashboard URL and PID so a second
    instance can open the existing dashboard.
    """

    def __init__(self, lock_path: Path) -> None:
        self._path = lock_path
        self._fd: Optional[int] = None

    def try_acquire(self) -> bool:
        """Try to acquire the lock. Returns True on success."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Open or create the lock file
            flags = os.O_RDWR | os.O_CREAT
            self._fd = os.open(str(self._path), flags)
        except OSError:
            return False

        if sys.platform == "win32":
            import msvcrt
            try:
                msvcrt.locking(self._fd, msvcrt.LK_NBLCK, 1)
                return True
            except (OSError, IOError):
                os.close(self._fd)
                self._fd = None
                return False
        else:
            import fcntl
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except (OSError, IOError):
                os.close(self._fd)
                self._fd = None
                return False

    def write_info(self, dashboard_url: str) -> None:
        """Write dashboard URL and PID to the lock file."""
        if self._fd is None:
            return
        content = f"{dashboard_url}\n{os.getpid()}\n"
        # Truncate and write
        os.lseek(self._fd, 0, os.SEEK_SET)
        os.ftruncate(self._fd, 0)
        os.write(self._fd, content.encode("utf-8"))

    def read_existing_url(self) -> Optional[str]:
        """Read the dashboard URL from an existing lock file."""
        try:
            text = self._path.read_text(encoding="utf-8").strip()
            lines = text.splitlines()
            if lines:
                return lines[0]
        except Exception:
            pass
        return None

    def release(self) -> None:
        """Release the lock and remove the file."""
        if self._fd is not None:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    try:
                        os.lseek(self._fd, 0, os.SEEK_SET)
                        msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
                    except (OSError, IOError):
                        pass
                else:
                    import fcntl
                    fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

        try:
            self._path.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Find a free port
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Bind to port 0 and return the OS-assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gomaxwebcam",
        description="GoMaxWebcam v2 -- GoPro virtual camera bridge",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without system tray (dashboard only)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # -- Logging setup --
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    headless = args.headless or _detect_headless()

    # -- Single-instance check --
    lock_dir = Path(user_runtime_dir(_APP_NAME, ensure_exists=True))
    lock = _InstanceLock(lock_dir / _LOCK_FILENAME)

    if not lock.try_acquire():
        # Another instance is running -- open its dashboard and exit
        existing_url = lock.read_existing_url()
        if existing_url:
            log.info("Another instance is running. Opening dashboard: %s", existing_url)
            webbrowser.open(existing_url)
        else:
            log.info("Another instance is running (no dashboard URL found).")
        sys.exit(0)

    atexit.register(lock.release)

    # -- Config --
    from gomaxwebcam.config import Config
    cfg = Config.load()
    if args.debug:
        cfg.logging.debug = True

    # -- Component creation --
    from gomaxwebcam.events import EventBus
    from gomaxwebcam.camera_manager import CameraManager
    from gomaxwebcam.dashboard.status_tracker import CameraStatusTracker
    from gomaxwebcam.orchestrator import AppOrchestrator
    from gomaxwebcam.transport_manager import TransportManager, TransportManagerConfig
    from gomaxwebcam.transport.usb import USBTransport
    from gomaxwebcam.transport.cohn import COHNTransport

    event_bus = EventBus()
    camera_manager = CameraManager(event_bus)
    status_tracker = CameraStatusTracker()

    # Build TransportManager config from user settings.
    # priority_for_manager() converts lowercase TOML names ("usb") to the
    # uppercase names registered with TransportManager ("USB").
    tm_config = TransportManagerConfig(
        priority=cfg.transport.priority_for_manager(),
        backoff_base_s=float(cfg.advanced.backoff_base_seconds),
        backoff_cap_s=float(cfg.advanced.backoff_cap_seconds),
        usb_poll_interval_s=cfg.advanced.usb_poll_interval,
    )
    transport_manager = TransportManager(event_bus, config=tm_config)
    auth_token = secrets.token_urlsafe(32)

    # Convert video config string values to the integer codes the transports use.
    # e.g. "1080p" → 12,  "wide" → 0
    _res = cfg.video.resolution_code()
    _fov = cfg.video.fov_code()

    # Serial hint: use the stored serial suffix if one was saved from a previous
    # connection, otherwise let the USB transport auto-discover.
    _serial = cfg.camera.camera_serial or None

    # -- Register transports for failover management (USB primary, COHN fallback) --
    transport_manager.register_transport("USB", USBTransport(
        serial=_serial,
        udp_port=cfg.advanced.udp_port,
        resolution=_res,
        fov=_fov,
        keepalive_interval=cfg.advanced.keepalive_interval,
        max_consecutive_failures=cfg.advanced.max_consecutive_failures,
    ))
    transport_manager.register_transport("COHN", COHNTransport(
        # last_known_ip populated from a previous session if available;
        # None lets the COHN transport fall back to mDNS discovery.
        ip_address=cfg.camera.last_known_ip or None,
        udp_port=cfg.advanced.udp_port,
        resolution=_res,
        fov=_fov,
        keepalive_interval=cfg.advanced.keepalive_interval,
    ))

    bind_host = "0.0.0.0" if headless else "127.0.0.1"
    port = _find_free_port()

    # -- AppOrchestrator owns the lifecycle --
    orchestrator = AppOrchestrator(
        event_bus=event_bus,
        camera_manager=camera_manager,
        transport_manager=transport_manager,
        status_tracker=status_tracker,
    )

    # create_dashboard() builds the FastAPI app with lifespan-managed
    # startup/shutdown tied to the orchestrator's start()/stop().
    # Use 127.0.0.1 for URL construction even when binding 0.0.0.0
    # so the dashboard URL is always browsable.
    app = orchestrator.create_dashboard(
        auth_token=auth_token,
        host="127.0.0.1",
        port=port,
    )

    dashboard_url = orchestrator.dashboard_url
    lock.write_info(dashboard_url)

    # -- asyncio loop on background thread --
    loop = asyncio.new_event_loop()
    event_bus.set_loop(loop)

    shutdown_event = threading.Event()

    async def _run_server() -> None:
        uvi_config = uvicorn.Config(
            app,
            host=bind_host,
            port=port,
            log_level="debug" if args.debug else "warning",
            loop="asyncio",
        )
        server = uvicorn.Server(uvi_config)

        # The orchestrator's lifespan handles start()/stop() automatically.
        # TransportManager.start() discovers and connects the first available
        # transport in priority order (USB then COHN), with automatic failover.
        await server.serve()

    server_task: Optional[asyncio.Task] = None

    def _start_loop() -> None:
        nonlocal server_task
        asyncio.set_event_loop(loop)
        server_task = loop.create_task(_run_server())
        loop.run_until_complete(server_task)
        shutdown_event.set()

    loop_thread = threading.Thread(target=_start_loop, name="asyncio-loop", daemon=True)
    loop_thread.start()

    log.info("Dashboard: %s", dashboard_url)
    if headless:
        print(f"Dashboard URL: {dashboard_url}", flush=True)

    # Open the dashboard automatically on startup when configured and not headless.
    # Poll until the server is actually accepting connections before opening browser.
    if cfg.dashboard.open_browser_on_start and not headless:
        def _open_browser_when_ready() -> None:
            import time as _time
            for _ in range(30):  # max 15 seconds
                _time.sleep(0.5)
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=1):
                        break  # Server is ready
                except (ConnectionRefusedError, OSError):
                    continue
            webbrowser.open(dashboard_url)
        threading.Thread(
            target=_open_browser_when_ready,
            name="browser-open",
            daemon=True,
        ).start()

    # -- Signal handling for graceful shutdown --
    _shutdown_requested = False  # Track first vs second signal

    def _signal_handler(signum: int, frame: object) -> None:
        nonlocal _shutdown_requested

        if _shutdown_requested:
            # Second signal: force-quit immediately (user hit Ctrl+C twice)
            log.warning("Second signal received — forcing immediate exit")
            shutdown_event.set()
            os._exit(1)

        _shutdown_requested = True
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        log.info("Received %s, initiating graceful shutdown...", sig_name)

        async def _graceful_stop() -> None:
            # Use orchestrator.shutdown() for proper teardown of all components
            # (stop_stream, stop transports, pipeline, event bus, etc.)
            try:
                await orchestrator.shutdown()
            except Exception:
                log.debug("Orchestrator shutdown failed", exc_info=True)
            # Cancel the server task to unblock the loop
            if server_task is not None:
                server_task.cancel()

        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(_graceful_stop(), loop)
            # Don't block the signal handler — let shutdown_event + join handle it
            # The future will complete on the loop thread; if the loop dies,
            # the daemon thread exits and shutdown_event.wait() unblocks.
            try:
                future.result(timeout=10.0)
            except Exception:
                log.debug("Graceful shutdown future did not complete in 10s", exc_info=True)
        shutdown_event.set()

    signal.signal(signal.SIGINT, _signal_handler)
    # SIGTERM may not be available on Windows in all contexts
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    # -- Tray icon on a daemon thread (desktop mode) --
    if not headless:
        try:
            from gomaxwebcam.tray import run_tray

            def _tray_thread() -> None:
                try:
                    run_tray(
                        dashboard_url=dashboard_url,
                        event_bus=event_bus,
                        on_quit=lambda: _signal_handler(signal.SIGTERM, None),
                    )
                except Exception:
                    log.debug("Tray thread exited with error", exc_info=True)
                finally:
                    # If tray exits (e.g. user quit), ensure shutdown propagates
                    shutdown_event.set()

            tray_thread = threading.Thread(
                target=_tray_thread,
                name="system-tray",
                daemon=True,
            )
            tray_thread.start()
            log.info("System tray started on daemon thread")
        except ImportError:
            log.warning("pystray not available, running without tray")

    # -- Main thread: block until shutdown --
    shutdown_event.wait()

    # Wait for the loop thread to finish (give graceful shutdown time)
    loop_thread.join(timeout=10.0)
    if loop_thread.is_alive():
        log.warning("Asyncio loop thread did not exit within 10s")

    # Close the event loop if it's still open
    if not loop.is_closed():
        loop.close()

    lock.release()
    log.info("GoMaxWebcam shutdown complete")


if __name__ == "__main__":
    main()
