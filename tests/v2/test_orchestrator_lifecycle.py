"""
test_orchestrator_lifecycle.py — Tests for AppOrchestrator start/stop coordination.

Verifies:
  - start() calls components in correct dependency order:
    EventBus.set_loop → wire callbacks → CameraManager.start → StatusTracker.start → TransportManager.start
  - stop() calls components in reverse order:
    TransportManager.stop → Pipeline.stop → StatusTracker.stop → CameraManager.stop → EventBus.shutdown
  - stop() is idempotent (no-op on second call)
  - start() raises RuntimeError if already running
  - Callbacks are wired from TransportManager to orchestrator handlers
  - _on_transport_switch updates CameraManager and StatusTracker
  - _on_transport_switch starts pipeline when transport is streaming
  - stop() handles component errors gracefully without aborting
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock, call

import pytest

from gomaxwebcam.orchestrator import AppOrchestrator

# All tests in this module are pure unit tests with mocks — no GoPro needed.
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_event_bus():
    bus = MagicMock()
    bus.set_loop = MagicMock()
    bus.shutdown = AsyncMock()
    bus.subscribe = MagicMock()
    bus.recent_events = []
    return bus


@pytest.fixture
def mock_camera_manager():
    cm = MagicMock()
    cm.start = AsyncMock()
    cm.stop = AsyncMock()
    cm.set_transport = MagicMock()
    cm.set_pipeline = MagicMock()
    cm.transport = None
    return cm


@pytest.fixture
def mock_transport_manager():
    tm = MagicMock()
    tm.start = AsyncMock(return_value=True)
    tm.stop = AsyncMock()
    tm.on_transport_switch = None
    tm.on_freeze = None
    tm.on_unfreeze = None
    return tm


@pytest.fixture
def mock_status_tracker():
    st = MagicMock()
    st.start = AsyncMock()
    st.stop = AsyncMock()
    st.set_transport = MagicMock()
    st.set_pipeline = MagicMock()
    return st


@pytest.fixture
def orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager, mock_status_tracker):
    return AppOrchestrator(
        event_bus=mock_event_bus,
        camera_manager=mock_camera_manager,
        transport_manager=mock_transport_manager,
        status_tracker=mock_status_tracker,
    )


@pytest.fixture
def orchestrator_no_tracker(mock_event_bus, mock_camera_manager, mock_transport_manager):
    """Orchestrator without a status tracker, to test optional components."""
    return AppOrchestrator(
        event_bus=mock_event_bus,
        camera_manager=mock_camera_manager,
        transport_manager=mock_transport_manager,
        status_tracker=None,
    )


# ---------------------------------------------------------------------------
# Tests: start() dependency order
# ---------------------------------------------------------------------------

class TestStartOrder:
    """Tests that start() initializes components in the correct order."""

    @pytest.mark.asyncio
    async def test_start_sets_event_loop_on_bus(self, orchestrator, mock_event_bus):
        """Step 1: EventBus.set_loop() is called with the running loop."""
        await orchestrator.start()
        mock_event_bus.set_loop.assert_called_once()
        # The argument should be the running loop
        loop_arg = mock_event_bus.set_loop.call_args[0][0]
        assert loop_arg is asyncio.get_running_loop()

    @pytest.mark.asyncio
    async def test_start_wires_callbacks(self, orchestrator, mock_transport_manager):
        """Step 2: TransportManager callbacks are wired to orchestrator handlers."""
        await orchestrator.start()
        assert mock_transport_manager.on_transport_switch is not None
        assert mock_transport_manager.on_freeze is not None
        assert mock_transport_manager.on_unfreeze is not None

    @pytest.mark.asyncio
    async def test_start_starts_camera_manager(self, orchestrator, mock_camera_manager):
        """Step 3: CameraManager.start() is called."""
        await orchestrator.start()
        mock_camera_manager.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_starts_status_tracker(self, orchestrator, mock_status_tracker):
        """Step 4: CameraStatusTracker.start() is called with poll_interval."""
        await orchestrator.start()
        mock_status_tracker.start.assert_awaited_once_with(poll_interval=1.0)

    @pytest.mark.asyncio
    async def test_start_starts_transport_manager(self, orchestrator, mock_transport_manager):
        """Step 5: TransportManager.start() is called."""
        await orchestrator.start()
        mock_transport_manager.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_start_order_camera_before_transport(
        self, orchestrator, mock_camera_manager, mock_transport_manager
    ):
        """CameraManager starts before TransportManager.

        This ensures SSE event publishing is ready before transports connect.
        """
        call_order = []
        mock_camera_manager.start = AsyncMock(side_effect=lambda: call_order.append("camera"))
        mock_transport_manager.start = AsyncMock(side_effect=lambda: call_order.append("transport"))

        await orchestrator.start()
        assert call_order.index("camera") < call_order.index("transport")

    @pytest.mark.asyncio
    async def test_start_order_tracker_before_transport(
        self, orchestrator, mock_status_tracker, mock_transport_manager
    ):
        """StatusTracker starts before TransportManager.

        Dashboard status must be flowing before transports attempt connection.
        """
        call_order = []
        mock_status_tracker.start = AsyncMock(
            side_effect=lambda poll_interval=1.0: call_order.append("tracker")
        )
        mock_transport_manager.start = AsyncMock(side_effect=lambda: call_order.append("transport"))

        await orchestrator.start()
        assert call_order.index("tracker") < call_order.index("transport")

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, orchestrator):
        """Orchestrator is_running is True after start()."""
        assert not orchestrator.is_running
        await orchestrator.start()
        assert orchestrator.is_running

    @pytest.mark.asyncio
    async def test_start_raises_if_already_running(self, orchestrator):
        """start() raises RuntimeError if called twice."""
        await orchestrator.start()
        with pytest.raises(RuntimeError, match="already running"):
            await orchestrator.start()

    @pytest.mark.asyncio
    async def test_start_skips_tracker_when_none(self, orchestrator_no_tracker):
        """start() works without a status tracker (optional component)."""
        await orchestrator_no_tracker.start()
        assert orchestrator_no_tracker.is_running


# ---------------------------------------------------------------------------
# Tests: stop() reverse order
# ---------------------------------------------------------------------------

class TestStopOrder:
    """Tests that stop() shuts down components in reverse dependency order."""

    @pytest.mark.asyncio
    async def test_stop_stops_transport_manager_first(
        self, orchestrator, mock_transport_manager
    ):
        """Step 1: TransportManager.stop() is called."""
        await orchestrator.start()
        await orchestrator.stop()
        mock_transport_manager.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_stops_status_tracker(
        self, orchestrator, mock_status_tracker
    ):
        """Step 3: StatusTracker.stop() is called."""
        await orchestrator.start()
        await orchestrator.stop()
        mock_status_tracker.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_stops_camera_manager(
        self, orchestrator, mock_camera_manager
    ):
        """Step 4: CameraManager.stop() is called."""
        await orchestrator.start()
        await orchestrator.stop()
        mock_camera_manager.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_shuts_down_event_bus(
        self, orchestrator, mock_event_bus
    ):
        """Step 5: EventBus.shutdown() is called."""
        await orchestrator.start()
        await orchestrator.stop()
        mock_event_bus.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_order_transport_before_camera(
        self, orchestrator, mock_transport_manager, mock_camera_manager
    ):
        """TransportManager stops before CameraManager (reverse of start)."""
        call_order = []
        mock_transport_manager.stop = AsyncMock(side_effect=lambda: call_order.append("transport"))
        mock_camera_manager.stop = AsyncMock(side_effect=lambda: call_order.append("camera"))

        await orchestrator.start()
        await orchestrator.stop()
        assert call_order.index("transport") < call_order.index("camera")

    @pytest.mark.asyncio
    async def test_stop_order_camera_before_bus(
        self, orchestrator, mock_camera_manager, mock_event_bus
    ):
        """CameraManager stops before EventBus shuts down."""
        call_order = []
        mock_camera_manager.stop = AsyncMock(side_effect=lambda: call_order.append("camera"))
        mock_event_bus.shutdown = AsyncMock(side_effect=lambda: call_order.append("bus"))

        await orchestrator.start()
        await orchestrator.stop()
        assert call_order.index("camera") < call_order.index("bus")

    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self, orchestrator):
        """is_running is False after stop()."""
        await orchestrator.start()
        assert orchestrator.is_running
        await orchestrator.stop()
        assert not orchestrator.is_running

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self, orchestrator, mock_transport_manager):
        """Calling stop() twice is safe — second call is a no-op."""
        await orchestrator.start()
        await orchestrator.stop()
        await orchestrator.stop()  # Should not raise
        # stop() should only be called once
        mock_transport_manager.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_noop_before_start(self, orchestrator, mock_transport_manager):
        """stop() before start() is a safe no-op."""
        await orchestrator.stop()
        mock_transport_manager.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_stop_skips_tracker_when_none(self, orchestrator_no_tracker):
        """stop() works without a status tracker."""
        await orchestrator_no_tracker.start()
        await orchestrator_no_tracker.stop()  # Should not raise
        assert not orchestrator_no_tracker.is_running


# ---------------------------------------------------------------------------
# Tests: stop() error resilience
# ---------------------------------------------------------------------------

class TestStopErrorResilience:
    """Tests that stop() continues even when individual components fail."""

    @pytest.mark.asyncio
    async def test_stop_continues_after_transport_error(
        self, orchestrator, mock_transport_manager, mock_camera_manager, mock_event_bus
    ):
        """If TransportManager.stop() raises, CameraManager and EventBus still stop."""
        mock_transport_manager.stop = AsyncMock(side_effect=RuntimeError("USB error"))

        await orchestrator.start()
        await orchestrator.stop()  # Should not raise

        mock_camera_manager.stop.assert_awaited_once()
        mock_event_bus.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_continues_after_camera_error(
        self, orchestrator, mock_camera_manager, mock_event_bus
    ):
        """If CameraManager.stop() raises, EventBus still shuts down."""
        mock_camera_manager.stop = AsyncMock(side_effect=RuntimeError("poll error"))

        await orchestrator.start()
        await orchestrator.stop()  # Should not raise

        mock_event_bus.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_continues_after_tracker_error(
        self, orchestrator, mock_status_tracker, mock_camera_manager, mock_event_bus
    ):
        """If StatusTracker.stop() raises, remaining components still stop."""
        mock_status_tracker.stop = AsyncMock(side_effect=RuntimeError("tracker error"))

        await orchestrator.start()
        await orchestrator.stop()  # Should not raise

        mock_camera_manager.stop.assert_awaited_once()
        mock_event_bus.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_pipeline_error_does_not_block(self, orchestrator):
        """If pipeline.stop() raises, remaining components still stop."""
        mock_pipeline = AsyncMock()
        mock_pipeline.stop = AsyncMock(side_effect=RuntimeError("vcam error"))
        orchestrator._pipeline = mock_pipeline

        await orchestrator.start()
        await orchestrator.stop()  # Should not raise
        assert not orchestrator.is_running


# ---------------------------------------------------------------------------
# Tests: callback wiring
# ---------------------------------------------------------------------------

class TestCallbackWiring:
    """Tests that TransportManager callbacks are wired to correct handlers."""

    @pytest.mark.asyncio
    async def test_on_transport_switch_updates_camera_manager(
        self, orchestrator, mock_camera_manager
    ):
        """_on_transport_switch calls camera_manager.set_transport()."""
        await orchestrator.start()

        mock_transport = MagicMock()
        mock_transport.is_streaming = False
        mock_transport.name = "USB"
        await orchestrator._on_transport_switch("USB", mock_transport)

        mock_camera_manager.set_transport.assert_called_with(mock_transport)

    @pytest.mark.asyncio
    async def test_on_transport_switch_updates_status_tracker(
        self, orchestrator, mock_status_tracker
    ):
        """_on_transport_switch calls status_tracker.set_transport()."""
        await orchestrator.start()

        mock_transport = MagicMock()
        mock_transport.is_streaming = False
        mock_transport.name = "COHN"
        await orchestrator._on_transport_switch("COHN", mock_transport)

        mock_status_tracker.set_transport.assert_called_with(mock_transport)

    @pytest.mark.asyncio
    async def test_on_transport_switch_starts_pipeline_when_streaming(
        self, orchestrator, mock_camera_manager, mock_status_tracker
    ):
        """_on_transport_switch starts FramePipeline when transport is streaming."""
        await orchestrator.start()

        mock_transport = MagicMock()
        mock_transport.is_streaming = True
        mock_transport.name = "USB"
        # Provide a real StreamInfo so PipelineConfig.from_stream_info() can
        # validate the codec field (MagicMock codec fails validation).
        from gomaxwebcam.transport.base import StreamInfo
        mock_transport.stream_info = StreamInfo(
            protocol="udp", host="0.0.0.0", port=8554,
            width=1920, height=1080, fps=30, codec="h264",
        )

        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPipeline:
            mock_pipeline_instance = AsyncMock()
            mock_pipeline_instance.start = AsyncMock(return_value=True)
            mock_pipeline_instance.stop = AsyncMock()
            MockPipeline.return_value = mock_pipeline_instance

            await orchestrator._on_transport_switch("USB", mock_transport)

            MockPipeline.assert_called_once()
            mock_pipeline_instance.start.assert_awaited_once_with(mock_transport.stream_info)
            mock_camera_manager.set_pipeline.assert_called_with(mock_pipeline_instance)
            mock_status_tracker.set_pipeline.assert_called_with(mock_pipeline_instance)

    @pytest.mark.asyncio
    async def test_on_transport_switch_skips_pipeline_when_not_streaming(
        self, orchestrator
    ):
        """_on_transport_switch does not start pipeline when transport isn't streaming."""
        await orchestrator.start()

        mock_transport = MagicMock()
        mock_transport.is_streaming = False
        mock_transport.name = "USB"

        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPipeline:
            await orchestrator._on_transport_switch("USB", mock_transport)
            MockPipeline.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_transport_switch_hot_swaps_running_pipeline(
        self, orchestrator
    ):
        """_on_transport_switch hot-swaps the decoder when pipeline is running.

        When a pipeline is already running (is_running=True), the orchestrator
        uses switch_stream() to swap the decoder while keeping the vcam sink
        alive for freeze-frame continuity.  The old pipeline is NOT stopped.
        """
        await orchestrator.start()

        old_pipeline = MagicMock()
        old_pipeline.is_running = True
        old_pipeline.switch_stream = AsyncMock(return_value=True)
        old_pipeline.stop = AsyncMock()
        orchestrator._pipeline = old_pipeline

        mock_transport = MagicMock()
        mock_transport.is_streaming = True
        mock_transport.name = "COHN"
        from gomaxwebcam.transport.base import StreamInfo
        mock_transport.stream_info = StreamInfo(
            protocol="udp", host="0.0.0.0", port=8554,
            width=1920, height=1080, fps=30, codec="h264",
        )

        await orchestrator._on_transport_switch("COHN", mock_transport)

        # Hot-swap used instead of stop + create new
        old_pipeline.switch_stream.assert_awaited_once()
        old_pipeline.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_on_transport_switch_stops_non_running_pipeline(
        self, orchestrator
    ):
        """_on_transport_switch stops old pipeline when it's not running."""
        await orchestrator.start()

        old_pipeline = MagicMock()
        old_pipeline.is_running = False
        old_pipeline.stop = AsyncMock()
        orchestrator._pipeline = old_pipeline

        mock_transport = MagicMock()
        mock_transport.is_streaming = True
        mock_transport.name = "COHN"
        from gomaxwebcam.transport.base import StreamInfo
        mock_transport.stream_info = StreamInfo(
            protocol="udp", host="0.0.0.0", port=8554,
            width=1920, height=1080, fps=30, codec="h264",
        )

        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPipeline:
            mock_new_pipeline = MagicMock()
            mock_new_pipeline.start = AsyncMock(return_value=True)
            MockPipeline.return_value = mock_new_pipeline

            await orchestrator._on_transport_switch("COHN", mock_transport)

            old_pipeline.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_freeze_callback_is_callable(self, orchestrator, mock_transport_manager):
        """on_freeze callback is wired and callable."""
        await orchestrator.start()
        # Should not raise
        mock_transport_manager.on_freeze()

    @pytest.mark.asyncio
    async def test_unfreeze_callback_is_callable(self, orchestrator, mock_transport_manager):
        """on_unfreeze callback is wired and callable."""
        await orchestrator.start()
        # Should not raise
        mock_transport_manager.on_unfreeze()

    @pytest.mark.asyncio
    async def test_on_transport_switch_skips_tracker_when_none(
        self, orchestrator_no_tracker, mock_camera_manager
    ):
        """_on_transport_switch works without a status tracker."""
        await orchestrator_no_tracker.start()

        mock_transport = MagicMock()
        mock_transport.is_streaming = False
        mock_transport.name = "USB"
        # Should not raise
        await orchestrator_no_tracker._on_transport_switch("USB", mock_transport)
        mock_camera_manager.set_transport.assert_called_with(mock_transport)


# ---------------------------------------------------------------------------
# Tests: full start→stop lifecycle
# ---------------------------------------------------------------------------

class TestFullLifecycle:
    """Integration-style tests for the complete start→stop cycle."""

    @pytest.mark.asyncio
    async def test_start_stop_roundtrip(self, orchestrator):
        """Full start/stop cycle completes without errors."""
        assert not orchestrator.is_running
        await orchestrator.start()
        assert orchestrator.is_running
        await orchestrator.stop()
        assert not orchestrator.is_running

    @pytest.mark.asyncio
    async def test_start_stop_start_again(
        self, orchestrator, mock_event_bus, mock_camera_manager, mock_transport_manager
    ):
        """Orchestrator can be restarted after stop (fresh cycle)."""
        await orchestrator.start()
        await orchestrator.stop()

        # Reset mocks for the second cycle
        mock_event_bus.set_loop.reset_mock()
        mock_camera_manager.start.reset_mock()
        mock_transport_manager.start.reset_mock()

        # Re-enable start by clearing _running
        await orchestrator.start()
        assert orchestrator.is_running
        mock_camera_manager.start.assert_awaited_once()
        mock_transport_manager.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# Tests: shutdown() terminal teardown
# ---------------------------------------------------------------------------

class TestShutdown:
    """Tests for AppOrchestrator.shutdown() — terminal cleanup."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_stop(
        self, orchestrator, mock_transport_manager, mock_camera_manager, mock_event_bus
    ):
        """shutdown() delegates to stop() for component teardown."""
        await orchestrator.start()
        await orchestrator.shutdown()

        mock_transport_manager.stop.assert_awaited_once()
        mock_camera_manager.stop.assert_awaited_once()
        mock_event_bus.shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_best_effort_stop_stream(self, orchestrator, mock_transport_manager):
        """shutdown() attempts stop_stream on the active transport."""
        mock_transport = MagicMock()
        mock_transport.is_streaming = True
        mock_transport.stop_stream = AsyncMock()
        mock_transport_manager.active_transport = mock_transport

        await orchestrator.start()
        await orchestrator.shutdown()

        mock_transport.stop_stream.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_stop_stream_failure_nonfatal(
        self, orchestrator, mock_transport_manager
    ):
        """shutdown() continues even if stop_stream fails."""
        mock_transport = MagicMock()
        mock_transport.is_streaming = True
        mock_transport.stop_stream = AsyncMock(side_effect=RuntimeError("stream error"))
        mock_transport_manager.active_transport = mock_transport

        await orchestrator.start()
        await orchestrator.shutdown()  # Should not raise
        assert not orchestrator.is_running

    @pytest.mark.asyncio
    async def test_shutdown_unwires_callbacks(self, orchestrator, mock_transport_manager):
        """shutdown() clears TransportManager callbacks."""
        await orchestrator.start()
        assert mock_transport_manager.on_transport_switch is not None
        assert mock_transport_manager.on_freeze is not None
        assert mock_transport_manager.on_unfreeze is not None

        await orchestrator.shutdown()

        assert mock_transport_manager.on_transport_switch is None
        assert mock_transport_manager.on_freeze is None
        assert mock_transport_manager.on_unfreeze is None

    @pytest.mark.asyncio
    async def test_shutdown_clears_app_reference(self, orchestrator):
        """shutdown() clears the FastAPI app reference."""
        orchestrator.create_dashboard(auth_token="test", host="127.0.0.1", port=9999)
        assert orchestrator.app is not None

        await orchestrator.start()
        await orchestrator.shutdown()

        assert orchestrator.app is None

    @pytest.mark.asyncio
    async def test_shutdown_prevents_restart(self, orchestrator):
        """start() raises RuntimeError after shutdown() (terminal operation)."""
        await orchestrator.start()
        await orchestrator.shutdown()

        with pytest.raises(RuntimeError, match="shut down"):
            await orchestrator.start()

    @pytest.mark.asyncio
    async def test_shutdown_is_idempotent(self, orchestrator, mock_transport_manager):
        """Calling shutdown() twice is safe — second call is a no-op."""
        await orchestrator.start()
        await orchestrator.shutdown()
        await orchestrator.shutdown()  # Should not raise
        # stop() should only have been called once (from first shutdown)
        mock_transport_manager.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_without_start(self, orchestrator, mock_transport_manager):
        """shutdown() before start() performs terminal cleanup without errors."""
        mock_transport_manager.active_transport = None
        await orchestrator.shutdown()
        # stop() is a no-op when not running, but shutdown still marks terminal
        with pytest.raises(RuntimeError, match="shut down"):
            await orchestrator.start()

    @pytest.mark.asyncio
    async def test_shutdown_skips_stop_stream_when_not_streaming(
        self, orchestrator, mock_transport_manager
    ):
        """shutdown() skips stop_stream if no transport is streaming."""
        mock_transport = MagicMock()
        mock_transport.is_streaming = False
        mock_transport.stop_stream = AsyncMock()
        mock_transport_manager.active_transport = mock_transport

        await orchestrator.start()
        await orchestrator.shutdown()

        mock_transport.stop_stream.assert_not_awaited()
