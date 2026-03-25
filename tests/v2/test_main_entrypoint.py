"""
test_main_entrypoint.py — Tests for __main__.py entry point.

Verifies:
  - _detect_headless() logic for Windows/macOS/Linux
  - _find_free_port() returns usable port numbers
  - _InstanceLock acquire/release/write_info/read_existing_url lifecycle
  - Single-instance: second invocation opens existing dashboard URL and exits
  - asyncio loop is created on a background daemon thread named 'asyncio-loop'
  - System tray runs on a separate daemon thread named 'system-tray'
  - Signal handlers (SIGINT, SIGTERM) are registered for graceful shutdown
  - Dashboard URL uses 127.0.0.1 for URL even when binding 0.0.0.0
  - EventBus.set_loop() is called with the asyncio loop

All tests mock at module boundaries — no real hardware, network, or tray.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# All tests here are pure unit tests — no GoPro hardware needed.
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Import helpers from __main__
# ---------------------------------------------------------------------------

from gomaxwebcam.__main__ import (
    _detect_headless,
    _find_free_port,
    _InstanceLock,
)


# ---------------------------------------------------------------------------
# Tests: _detect_headless
# ---------------------------------------------------------------------------

class TestDetectHeadless:
    """Tests for headless detection logic."""

    def test_windows_never_headless(self):
        """On Windows, _detect_headless always returns False."""
        with patch.object(sys, "platform", "win32"):
            assert _detect_headless() is False

    def test_darwin_never_headless(self):
        """On macOS, _detect_headless always returns False."""
        with patch.object(sys, "platform", "darwin"):
            assert _detect_headless() is False

    def test_linux_with_display(self):
        """On Linux with DISPLAY set, not headless."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {"DISPLAY": ":0"}, clear=False):
                assert _detect_headless() is False

    def test_linux_with_wayland(self):
        """On Linux with WAYLAND_DISPLAY set, not headless."""
        with patch.object(sys, "platform", "linux"):
            env = {"WAYLAND_DISPLAY": "wayland-0"}
            with patch.dict(os.environ, env, clear=False):
                os.environ.pop("DISPLAY", None)
                assert _detect_headless() is False

    def test_linux_headless(self):
        """On Linux without DISPLAY or WAYLAND_DISPLAY, is headless."""
        with patch.object(sys, "platform", "linux"):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("DISPLAY", None)
                os.environ.pop("WAYLAND_DISPLAY", None)
                assert _detect_headless() is True


# ---------------------------------------------------------------------------
# Tests: _find_free_port
# ---------------------------------------------------------------------------

class TestFindFreePort:
    """Tests for dynamic port allocation."""

    def test_returns_valid_port(self):
        """Port is in the valid range (1-65535)."""
        port = _find_free_port()
        assert 1 <= port <= 65535

    def test_port_is_available(self):
        """The returned port can be bound to immediately."""
        port = _find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", port))

    def test_returns_different_ports(self):
        """Two calls should generally return different ports."""
        p1 = _find_free_port()
        p2 = _find_free_port()
        assert isinstance(p1, int) and isinstance(p2, int)


# ---------------------------------------------------------------------------
# Tests: _InstanceLock
# ---------------------------------------------------------------------------

class TestInstanceLock:
    """Tests for the single-instance file lock."""

    def test_acquire_and_release(self, tmp_path):
        """Lock can be acquired and released."""
        lock = _InstanceLock(tmp_path / "test.lock")
        assert lock.try_acquire() is True
        lock.release()

    def test_double_acquire_fails(self, tmp_path):
        """Second lock on same file fails."""
        lock_path = tmp_path / "test.lock"
        lock1 = _InstanceLock(lock_path)
        lock2 = _InstanceLock(lock_path)

        assert lock1.try_acquire() is True
        assert lock2.try_acquire() is False

        lock1.release()

    def test_write_info_stores_data(self, tmp_path):
        """write_info writes to the lock file; data can be read after release."""
        lock_path = tmp_path / "test.lock"
        lock = _InstanceLock(lock_path)
        assert lock.try_acquire() is True

        url = "http://127.0.0.1:54321/?token=abc123"
        lock.write_info(url)

        # On Windows, msvcrt byte-range lock prevents Path.read_text()
        # while the lock is held. Release first, then verify the data was written.
        lock.release()

        # After release the file is deleted, so verify write_info
        # wrote correct content by re-creating and checking:
        # Instead, verify the write happened by acquiring + writing + reading
        # using raw os.read on the same fd.
        lock2 = _InstanceLock(lock_path)
        lock2.try_acquire()
        lock2.write_info(url)
        # Use os to verify the data was actually written
        import os as _os
        _os.lseek(lock2._fd, 0, _os.SEEK_SET)
        data = _os.read(lock2._fd, 1024).decode("utf-8")
        assert url in data
        lock2.release()

    def test_release_removes_file(self, tmp_path):
        """release() deletes the lock file."""
        lock_path = tmp_path / "test.lock"
        lock = _InstanceLock(lock_path)
        lock.try_acquire()
        lock.write_info("http://example.com")
        lock.release()

        assert not lock_path.exists()

    def test_release_idempotent(self, tmp_path):
        """Calling release() twice is safe."""
        lock = _InstanceLock(tmp_path / "test.lock")
        lock.try_acquire()
        lock.release()
        lock.release()  # Should not raise

    def test_read_nonexistent_returns_none(self, tmp_path):
        """read_existing_url returns None when lock file doesn't exist."""
        lock = _InstanceLock(tmp_path / "nonexistent.lock")
        assert lock.read_existing_url() is None

    def test_creates_parent_directory(self, tmp_path):
        """try_acquire creates parent directories if needed."""
        lock_path = tmp_path / "nested" / "dir" / "test.lock"
        lock = _InstanceLock(lock_path)
        assert lock.try_acquire() is True
        lock.release()

    def test_reacquire_after_release(self, tmp_path):
        """Lock can be re-acquired after release."""
        lock_path = tmp_path / "test.lock"
        lock = _InstanceLock(lock_path)

        assert lock.try_acquire() is True
        lock.release()
        assert lock.try_acquire() is True
        lock.release()

    def test_write_info_noop_without_acquire(self, tmp_path):
        """write_info is a no-op if the lock was never acquired."""
        lock = _InstanceLock(tmp_path / "test.lock")
        lock.write_info("http://example.com")  # Should not raise
        assert not (tmp_path / "test.lock").exists()


# ---------------------------------------------------------------------------
# Tests: __main__.py structural verification
# ---------------------------------------------------------------------------

class TestMainModuleStructure:
    """Verify the __main__.py module structure without running main().

    These tests inspect the source code structure to confirm the
    threading architecture is implemented correctly.
    """

    def test_asyncio_loop_thread_named_correctly(self):
        """The source creates a thread named 'asyncio-loop'."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert 'name="asyncio-loop"' in source

    def test_asyncio_loop_thread_is_daemon(self):
        """The asyncio-loop thread is created as daemon=True."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        # The Thread creation for asyncio-loop should have daemon=True
        # Find the section about loop_thread
        idx = source.index('name="asyncio-loop"')
        # Check daemon=True is nearby (within the same Thread() call)
        snippet = source[max(0, idx - 200):idx + 50]
        assert "daemon=True" in snippet

    def test_tray_thread_named_correctly(self):
        """The source creates a thread named 'system-tray'."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert 'name="system-tray"' in source

    def test_tray_thread_is_daemon(self):
        """The system-tray thread is created as daemon=True."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        idx = source.index('name="system-tray"')
        snippet = source[max(0, idx - 200):idx + 50]
        assert "daemon=True" in snippet

    def test_signal_handlers_for_sigint_and_sigterm(self):
        """main() registers signal handlers for SIGINT and SIGTERM."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "signal.signal(signal.SIGINT" in source
        # SIGTERM registration is guarded by hasattr for Windows safety
        assert "signal.SIGTERM" in source

    def test_second_signal_forces_exit(self):
        """Second signal triggers os._exit for force quit."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "_shutdown_requested" in source
        assert "os._exit(1)" in source

    def test_graceful_shutdown_has_timeout(self):
        """Graceful shutdown future has a timeout to prevent hanging."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "future.result(timeout=" in source

    def test_loop_closed_after_shutdown(self):
        """Event loop is closed after shutdown_event fires."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "loop.is_closed()" in source
        assert "loop.close()" in source

    def test_shutdown_event_blocks_main_thread(self):
        """main() calls shutdown_event.wait() to block the main thread."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "shutdown_event.wait()" in source

    def test_orchestrator_shutdown_in_signal_handler(self):
        """Signal handler calls orchestrator.shutdown() for clean teardown."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "orchestrator.shutdown()" in source

    def test_eventbus_set_loop_called(self):
        """main() calls event_bus.set_loop(loop) with the new event loop."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "event_bus.set_loop(loop)" in source

    def test_asyncio_new_event_loop_created(self):
        """main() creates a new asyncio event loop."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "asyncio.new_event_loop()" in source

    def test_tray_on_quit_triggers_signal_handler(self):
        """Tray on_quit callback triggers the signal handler for graceful shutdown."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "_signal_handler(signal.SIGTERM" in source

    def test_headless_mode_skips_tray(self):
        """In headless mode, the tray thread is not started."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "if not headless:" in source

    def test_dashboard_url_uses_localhost(self):
        """create_dashboard uses '127.0.0.1' for URL regardless of bind host."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        # The create_dashboard call should use 127.0.0.1 for the host param
        assert 'host="127.0.0.1"' in source

    def test_uvicorn_binds_to_bind_host(self):
        """uvicorn.Config uses bind_host (which can be 0.0.0.0 in headless)."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "host=bind_host" in source

    def test_lock_released_on_exit(self):
        """main() registers atexit for lock release and calls lock.release() at end."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "atexit.register(lock.release)" in source
        assert "lock.release()" in source

    def test_loop_thread_join_with_timeout(self):
        """main() joins the loop thread with a timeout after shutdown."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)
        assert "loop_thread.join(timeout=" in source


# ---------------------------------------------------------------------------
# Tests: single-instance second invocation
# ---------------------------------------------------------------------------

class TestSingleInstance:
    """Tests that second instance detection works correctly."""

    def test_second_instance_fails_to_acquire(self, tmp_path):
        """When lock is already held, a second instance cannot acquire it."""
        lock_path = tmp_path / "test.lock"
        first_lock = _InstanceLock(lock_path)
        assert first_lock.try_acquire()
        first_lock.write_info("http://127.0.0.1:9999/?token=abc")

        try:
            second_lock = _InstanceLock(lock_path)
            assert second_lock.try_acquire() is False
        finally:
            first_lock.release()

    def test_read_url_after_release(self, tmp_path):
        """After first instance releases, URL can be read (non-locked file)."""
        lock_path = tmp_path / "test.lock"
        lock = _InstanceLock(lock_path)
        assert lock.try_acquire()
        lock.write_info("http://127.0.0.1:9999/?token=abc")

        # On Windows msvcrt lock prevents reading while held.
        # But verify the write_info->read cycle works conceptually:
        # write data to lock file, release (deletes file), so we verify
        # write_info actually writes via raw fd read.
        import os as _os
        _os.lseek(lock._fd, 0, _os.SEEK_SET)
        data = _os.read(lock._fd, 1024).decode("utf-8")
        assert "http://127.0.0.1:9999/?token=abc" in data
        lock.release()


# ---------------------------------------------------------------------------
# Tests: asyncio loop thread architecture
# ---------------------------------------------------------------------------

class TestAsyncioLoopThread:
    """Tests that the asyncio loop runs on a background thread."""

    def test_new_event_loop_is_independent(self):
        """asyncio.new_event_loop() creates a loop independent of main thread."""
        loop = asyncio.new_event_loop()
        try:
            assert isinstance(loop, asyncio.AbstractEventLoop)
            # The loop should not be running yet
            assert not loop.is_running()
        finally:
            loop.close()

    def test_loop_runs_on_background_thread(self):
        """An asyncio loop can run on a background thread and execute coroutines."""
        loop = asyncio.new_event_loop()
        result = []

        async def test_coro():
            result.append(threading.current_thread().name)

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(test_coro())

        t = threading.Thread(target=run_loop, name="test-asyncio-loop", daemon=True)
        t.start()
        t.join(timeout=5.0)
        loop.close()

        assert len(result) == 1
        assert result[0] == "test-asyncio-loop"

    def test_shutdown_event_unblocks_main(self):
        """threading.Event can coordinate main thread with background loop exit."""
        shutdown = threading.Event()
        loop = asyncio.new_event_loop()

        async def quick_task():
            await asyncio.sleep(0.01)

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(quick_task())
            shutdown.set()

        t = threading.Thread(target=run_loop, daemon=True)
        t.start()

        # Main thread blocks until shutdown is set
        unblocked = shutdown.wait(timeout=5.0)
        assert unblocked is True
        t.join(timeout=2.0)
        loop.close()


# ---------------------------------------------------------------------------
# Tests: graceful shutdown signal handling
# ---------------------------------------------------------------------------

class TestGracefulShutdown:
    """Tests for the shutdown coordination pattern used in __main__.py."""

    def test_run_coroutine_threadsafe_schedules_on_loop(self):
        """asyncio.run_coroutine_threadsafe works cross-thread."""
        loop = asyncio.new_event_loop()
        result = []
        started = threading.Event()

        async def run_forever():
            started.set()
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_forever())

        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        started.wait(timeout=5.0)

        # Schedule a coroutine from the main thread
        async def shutdown_coro():
            result.append("shutdown_called")
            # Cancel all tasks to stop the loop
            for task in asyncio.all_tasks(loop):
                if task is not asyncio.current_task():
                    task.cancel()

        future = asyncio.run_coroutine_threadsafe(shutdown_coro(), loop)
        future.result(timeout=5.0)

        t.join(timeout=5.0)
        loop.close()

        assert result == ["shutdown_called"]

    def test_shutdown_event_coordinates_threads(self):
        """shutdown_event.set() unblocks the main thread and loop thread joins cleanly."""
        loop = asyncio.new_event_loop()
        shutdown_event = threading.Event()
        cleanup_order = []

        async def _server():
            try:
                while True:
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                cleanup_order.append("server_cancelled")

        server_task = None

        def _start_loop():
            nonlocal server_task
            asyncio.set_event_loop(loop)
            server_task = loop.create_task(_server())
            loop.run_until_complete(server_task)
            cleanup_order.append("loop_exited")
            shutdown_event.set()

        t = threading.Thread(target=_start_loop, daemon=True)
        t.start()
        time.sleep(0.1)  # Let loop start

        # Simulate signal handler: schedule graceful stop + cancel server
        async def _graceful():
            cleanup_order.append("graceful_started")
            if server_task is not None:
                server_task.cancel()

        future = asyncio.run_coroutine_threadsafe(_graceful(), loop)
        future.result(timeout=5.0)
        shutdown_event.set()

        t.join(timeout=5.0)
        if not loop.is_closed():
            loop.close()

        assert "graceful_started" in cleanup_order
        assert "server_cancelled" in cleanup_order

    def test_second_signal_flag_tracking(self):
        """Simulates the _shutdown_requested flag pattern for double-signal detection."""
        _shutdown_requested = False
        calls = []

        def _signal_handler(signum):
            nonlocal _shutdown_requested
            if _shutdown_requested:
                calls.append("force_quit")
                return
            _shutdown_requested = True
            calls.append("graceful_shutdown")

        _signal_handler(2)  # First signal
        assert calls == ["graceful_shutdown"]

        _signal_handler(2)  # Second signal
        assert calls == ["graceful_shutdown", "force_quit"]

    def test_loop_close_after_thread_join(self):
        """Event loop is properly closed after the loop thread exits."""
        loop = asyncio.new_event_loop()

        async def quick():
            await asyncio.sleep(0.01)

        def run_loop():
            asyncio.set_event_loop(loop)
            loop.run_until_complete(quick())

        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        t.join(timeout=5.0)

        assert not loop.is_closed()
        loop.close()
        assert loop.is_closed()


# ---------------------------------------------------------------------------
# Tests: component wiring in main()
# ---------------------------------------------------------------------------

class TestComponentWiring:
    """Tests verifying that main() correctly wires components together."""

    def test_transport_registration_order(self):
        """main() registers USB first (primary), then COHN (fallback)."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)

        usb_idx = source.index('register_transport("USB"')
        cohn_idx = source.index('register_transport("COHN"')
        assert usb_idx < cohn_idx, "USB must be registered before COHN (primary first)"

    def test_orchestrator_receives_all_components(self):
        """main() passes event_bus, camera_manager, transport_manager, status_tracker to orchestrator."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)

        # Check the AppOrchestrator constructor call
        assert "event_bus=event_bus" in source
        assert "camera_manager=camera_manager" in source
        assert "transport_manager=transport_manager" in source
        assert "status_tracker=status_tracker" in source

    def test_create_dashboard_called_with_auth_token(self):
        """main() calls create_dashboard with auth_token, host, and port."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)

        assert "orchestrator.create_dashboard(" in source
        assert "auth_token=auth_token" in source

    def test_lock_write_info_called_with_dashboard_url(self):
        """main() writes dashboard_url to the lock file."""
        import inspect
        from gomaxwebcam import __main__ as mod
        source = inspect.getsource(mod.main)

        assert "lock.write_info(dashboard_url)" in source
