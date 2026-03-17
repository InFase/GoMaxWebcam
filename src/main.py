"""
main.py -- Entry point for GoPro Bridge

Launches the application with automatic startup lifecycle:
  1. Initialize logging and config
  2. Create the AppController (discovery-connect-initialize pipeline)
  3. Launch GUI (PyQt6) or fall back to console mode
  4. Auto-start the discovery-connect-initialize pipeline on startup

The startup pipeline runs automatically when the app launches:
  config → AppController → start() → _startup_flow() (background thread):
    1. Check prerequisites (ffmpeg on PATH)
    2. Discover GoPro on USB (with retries)
    3. Open control connection (API verify, USB control, IDLE workaround)
    4. Start webcam mode (resolution + FOV)
    5. Initialize streaming pipeline (VirtualCamera + StreamReader + FramePipeline)
    6. Launch keep-alive monitoring
    7. Start USB event listener for fast reconnect detection

This file can be run directly:
    python src/main.py          # Launch with GUI (default)
    python src/main.py --nogui  # Console-only mode (for debugging)
"""

import sys
import os
import signal
import threading
import traceback

# Add src/ to path so imports work when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _frozen_crash_handler():
    """When running as a PyInstaller .exe, show a dialog on unhandled exceptions
    instead of silently dying (since there's no console window)."""
    if not getattr(sys, 'frozen', False):
        return  # Only needed for frozen .exe

    _original_excepthook = sys.excepthook

    def _excepthook(exc_type, exc_value, exc_tb):
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        # Try to write to a crash log next to the .exe
        try:
            crash_path = os.path.join(os.path.dirname(sys.executable), "crash.log")
            with open(crash_path, "w") as f:
                f.write(msg)
        except OSError:
            pass
        # Try to show a message box
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"GoPro Bridge crashed:\n\n{msg}", "GoPro Bridge Error", 0x10
            )
        except Exception:
            pass
        _original_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook


_frozen_crash_handler()

from config import Config
from logger import setup_logger, get_logger
from app_controller import AppController, AppState


def _run_console_mode(controller: AppController, config: Config, log):
    """Run in console mode (no GUI) — used for debugging or --nogui flag.

    Sets up console-based callbacks and blocks until Ctrl+C.
    The discovery-connect-initialize pipeline runs automatically via
    controller.start() which triggers _startup_flow() on a background thread.
    """
    # Set up status callbacks for console output
    def on_state_change(state: AppState):
        state_labels = {
            AppState.INITIALIZING:          "[....] Initializing...",
            AppState.CHECKING_PREREQUISITES:"[CHCK] Checking prerequisites...",
            AppState.DISCOVERING:           "[SCAN] Searching for GoPro...",
            AppState.CONNECTING:            "[CONN] Connecting...",
            AppState.STREAMING:             "[ OK ] Streaming",
            AppState.RECONNECTING:          "[RETR] Reconnecting...",
            AppState.DISCONNECTED:          "[DISC] Disconnected",
            AppState.ERROR:                 "[ERR ] Error",
            AppState.STOPPED:               "[STOP] Stopped",
        }
        label = state_labels.get(state, state.name)
        print(f"\n{'-' * 40}")
        print(f"  STATE: {label}")
        print(f"{'-' * 40}")

    def on_status(message: str, level: str):
        prefix = {
            "info":    "  [i]",
            "success": "  [+]",
            "warning": "  [!]",
            "error":   "  [x]",
        }.get(level, "  ")
        print(f"{prefix} {message}")

    def on_camera_info(info: dict):
        battery = info.get("battery_level")
        if battery is not None:
            print(f"  [B] Battery: {battery}%")

    controller.on_state_change = on_state_change
    controller.on_status = on_status
    controller.on_camera_info = on_camera_info

    # Handle Ctrl+C gracefully
    shutdown_event = threading.Event()

    def signal_handler(sig, frame):
        print("\n\nShutting down GoPro Bridge...")
        controller.stop()
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)

    # Banner
    print()
    print("+======================================+")
    print("|         GoPro Bridge v0.1.0          |")
    print("+======================================+")
    print()

    # Auto-start the discovery-connect-initialize pipeline
    if config.auto_connect_on_launch:
        log.info("[EVENT:startup] Auto-connect enabled — starting discovery-connect-initialize pipeline")
        controller.start()
    else:
        log.info("[EVENT:startup] Auto-connect disabled — waiting for manual start")
        print("  Auto-connect is disabled. Edit config.json to enable.")
        print("  Or press Enter to start manually...")
        input()
        controller.start()

    # Block until shutdown
    print("\n  Press Ctrl+C to stop\n")
    try:
        shutdown_event.wait()
    except KeyboardInterrupt:
        signal_handler(None, None)


def _run_gui_mode(controller: AppController, config: Config, log):
    """Run with PyQt6 GUI — the default mode.

    The GUI wires the AppController callbacks to Qt signals for thread-safe
    UI updates. The discovery-connect-initialize pipeline starts automatically
    via a QTimer.singleShot(100ms) after the window is shown, so the user
    immediately sees status updates as the pipeline progresses.

    Flow:
      1. QApplication created
      2. DashboardWindow created (wires controller callbacks → Qt signals)
      3. Window shown
      4. 100ms later: controller.start() fires → _startup_flow() on background thread
      5. Qt event loop runs until window close or controller.stop()
    """
    from gui import run_gui

    log.info("[EVENT:startup] Launching GUI mode — pipeline will auto-start after window is shown")
    exit_code = run_gui(controller, config=config)
    return exit_code


def main():
    """GoPro Bridge application entry point.

    Wires the complete startup lifecycle:
      1. Load config from %APPDATA%/GoProBridge/config.json
      2. Initialize session logging
      3. Create AppController (owns GoProConnection + pipeline components)
      4. Launch UI (GUI or console)
      5. Auto-start discovery-connect-initialize pipeline

    The pipeline runs automatically on startup — no user action needed.
    When auto_connect_on_launch is True (default), the app immediately
    begins searching for the GoPro on USB, connects, enables webcam mode,
    and starts the streaming pipeline.
    """
    # Determine launch mode from command-line flags
    nogui = "--nogui" in sys.argv or "--console" in sys.argv

    # -- Step 1: Load configuration --
    config = Config.load()

    # -- Step 2: Set up logging --
    setup_logger(config.log_dir, level=config.log_level)
    log = get_logger("main")
    log.info("=" * 60)
    log.info("[EVENT:startup] GoPro Bridge starting up")
    log.info("=" * 60)
    log.info("[EVENT:startup] Config loaded from: %s", config._config_path)
    log.info("[EVENT:startup] Logs directory: %s", config.log_dir)
    log.info("[EVENT:startup] Launch mode: %s", "console" if nogui else "GUI")
    log.info(
        "[EVENT:startup] Stream: %dx%d@%dfps, UDP port %d, vcam='%s'",
        config.stream_width, config.stream_height, config.stream_fps,
        config.udp_port, config.virtual_camera_name,
    )
    log.info(
        "[EVENT:startup] Recovery: keepalive=%.1fs, reconnect_delay=%.1fs, "
        "reconnect_max_delay=%.1fs, ncm_adapter_wait=%.1fs, "
        "stale_poll_interval=%.1fs, max_retries=%s, discovery_timeout=%.1fs",
        config.keepalive_interval, config.reconnect_delay,
        config.reconnect_max_delay, config.ncm_adapter_wait,
        config.stale_poll_interval,
        config.reconnect_max_retries if config.reconnect_max_retries > 0 else "infinite",
        config.discovery_overall_timeout,
    )

    # -- Step 3: Create the app controller --
    # AppController owns the full pipeline: discovery → connection → streaming
    # Calling controller.start() triggers the entire lifecycle automatically
    controller = AppController(config)

    # -- Step 3b: Register cleanup for all exit paths --
    # atexit handles normal exit and unhandled exceptions.
    # Signal handlers catch Ctrl+C and SIGTERM (task manager "End Task").
    import atexit

    _cleanup_done = False

    def _cleanup():
        nonlocal _cleanup_done
        if _cleanup_done:
            return
        _cleanup_done = True
        log.info("[EVENT:shutdown] Running cleanup (atexit/signal)")
        try:
            controller.stop()
        except Exception:
            pass

    atexit.register(_cleanup)

    def _signal_shutdown(sig, frame):
        log.info("[EVENT:shutdown] Received signal %s", sig)
        _cleanup()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _signal_shutdown)
    signal.signal(signal.SIGINT, _signal_shutdown)

    # -- Step 4: Launch UI and auto-start pipeline --
    if nogui:
        _run_console_mode(controller, config, log)
    else:
        try:
            exit_code = _run_gui_mode(controller, config, log)
        except ImportError as e:
            log.warning("[EVENT:startup] PyQt6 not available (%s), falling back to console mode", e)
            print(f"  [!] PyQt6 not available: {e}")
            print("  [i] Falling back to console mode (install PyQt6 for GUI)")
            print()
            _run_console_mode(controller, config, log)
            exit_code = 0

    _cleanup()
    log.info("[EVENT:shutdown] GoPro Bridge exited cleanly")
    sys.exit(exit_code if exit_code else 0)


if __name__ == "__main__":
    main()
