"""
test_freeze_frame_wiring.py — Freeze-frame during transport transitions.

Proves that the virtual camera displays a frozen last-good-frame during
transport transitions, and that on_freeze/on_unfreeze callbacks are wired
end-to-end from TransportManager through AppOrchestrator to FramePipeline.

AC 5: Freeze-frame displayed in virtual camera during transport transitions
via wired on_freeze/on_unfreeze callbacks.

Test scenarios:
  1. Pipeline freeze/unfreeze state machine transitions
  2. VirtualCameraSink re-sends last frame during freeze
  3. TransportManager → Orchestrator → Pipeline callback chain
  4. Pipeline.switch_stream() hot-swaps decoder while vcam stays alive
  5. Orchestrator._ensure_pipeline() uses hot-swap during failover
  6. Full integration: USB disconnect → freeze → COHN failover → unfreeze
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

from gomaxwebcam.pipeline.frame_pipeline import (
    FramePipeline,
    PipelineConfig,
    PipelineState,
)
from gomaxwebcam.pipeline.virtual_camera_sink import VirtualCameraSink
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo
from gomaxwebcam.events import EventBus, EventType
from gomaxwebcam.transport_manager import (
    TransportManager,
    TransportManagerConfig,
    ManagerState,
)
from gomaxwebcam.orchestrator import AppOrchestrator
from gomaxwebcam.camera_manager import CameraManager

# All tests are mocked -- no GoPro or virtual camera hardware needed.
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stream_info(port: int = 8554) -> StreamInfo:
    return StreamInfo(
        protocol="udp", host="0.0.0.0", port=port,
        width=1920, height=1080, fps=30, codec="h264",
    )


def _make_1080p_frame(fill: int = 128) -> np.ndarray:
    return np.full((1080, 1920, 3), fill, dtype=np.uint8)


def _make_mock_transport(
    name: str, *, stream_ok: bool = True, port: int = 8554,
) -> MagicMock:
    """Create a mock Transport with controllable behavior."""
    t = MagicMock(spec=Transport)
    t.name = name
    t._name = name
    t.state = TransportState.DISCONNECTED
    t.is_streaming = stream_ok
    t.is_connected = False
    t.stream_info = _make_stream_info(port)

    t.discover = AsyncMock(return_value=True)
    t.connect = AsyncMock(return_value=True)
    t.start_stream = AsyncMock(
        return_value=_make_stream_info(port) if stream_ok else None,
    )
    t.stop_stream = AsyncMock()
    t.disconnect = AsyncMock()

    t._disconnect_listeners = []
    t.add_disconnect_listener = MagicMock(
        side_effect=lambda cb: t._disconnect_listeners.append(cb),
    )
    return t


def _fast_config() -> TransportManagerConfig:
    return TransportManagerConfig(
        priority=["USB", "COHN"],
        discover_timeout_s=2.0,
        connect_timeout_s=2.0,
        stream_timeout_s=2.0,
        backoff_base_s=0.05,
        backoff_cap_s=0.1,
        auto_recovery=False,
    )


# ---------------------------------------------------------------------------
# 1. Pipeline freeze/unfreeze state machine
# ---------------------------------------------------------------------------

class TestPipelineFreezeUnfreezeStateMachine:
    """Pipeline state transitions during freeze-frame mode."""

    def test_freeze_transitions_to_freeze_frame_state(self):
        """freeze() moves pipeline from STREAMING to FREEZE_FRAME."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.STREAMING

        pipeline.freeze()

        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert pipeline.is_frozen is True
        assert pipeline._was_frozen is True

    def test_freeze_fires_on_freeze_callback(self):
        """freeze() calls the on_freeze callback."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.STREAMING

        callback_log = []
        pipeline.on_freeze = lambda: callback_log.append("frozen")

        pipeline.freeze()

        assert callback_log == ["frozen"]

    def test_freeze_is_noop_when_already_frozen(self):
        """freeze() is a no-op when already in FREEZE_FRAME state."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.FREEZE_FRAME

        callback_log = []
        pipeline.on_freeze = lambda: callback_log.append("frozen")

        pipeline.freeze()

        assert callback_log == []  # No duplicate callback

    def test_freeze_is_noop_when_stopped(self):
        """freeze() is a no-op when pipeline is STOPPED."""
        pipeline = FramePipeline()
        assert pipeline.state == PipelineState.STOPPED

        pipeline.freeze()

        assert pipeline.state == PipelineState.STOPPED

    def test_unfreeze_signals_pending_resume(self):
        """unfreeze() signals that frames are expected but doesn't change state."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.FREEZE_FRAME
        pipeline._was_frozen = True

        pipeline.unfreeze()

        # State stays FREEZE_FRAME until a real frame arrives
        assert pipeline.state == PipelineState.FREEZE_FRAME

    def test_unfreeze_is_noop_when_not_frozen(self):
        """unfreeze() is a no-op when not in FREEZE_FRAME."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.STREAMING

        pipeline.unfreeze()  # Should not raise

        assert pipeline.state == PipelineState.STREAMING

    def test_first_frame_after_freeze_transitions_to_streaming(self):
        """First decoded frame after freeze transitions back to STREAMING."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.FREEZE_FRAME
        pipeline._was_frozen = True

        # Mock vcam sink
        mock_sink = MagicMock()
        pipeline._vcam_sink = mock_sink

        callback_log = []
        pipeline.on_unfreeze = lambda: callback_log.append("unfrozen")

        # Simulate first frame arriving
        frame = _make_1080p_frame()
        pipeline._on_decoded_frame(frame)

        assert pipeline.state == PipelineState.STREAMING
        assert pipeline._was_frozen is False
        assert callback_log == ["unfrozen"]
        mock_sink.submit_frame.assert_called_once_with(frame)

    def test_decoder_error_triggers_freeze(self):
        """Decoder error callback triggers freeze-frame mode."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.STREAMING

        callback_log = []
        pipeline.on_freeze = lambda: callback_log.append("frozen")

        pipeline._on_decoder_error(RuntimeError("stream lost"))

        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert callback_log == ["frozen"]

    def test_decoder_stopped_triggers_freeze(self):
        """Decoder stop callback triggers freeze-frame mode."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.STREAMING

        callback_log = []
        pipeline.on_freeze = lambda: callback_log.append("frozen")

        pipeline._on_decoder_stopped()

        assert pipeline.state == PipelineState.FREEZE_FRAME
        assert callback_log == ["frozen"]


# ---------------------------------------------------------------------------
# 2. VirtualCameraSink freeze-frame behavior
# ---------------------------------------------------------------------------

class TestVCamSinkFreezeFrame:
    """VirtualCameraSink re-sends last frame when no new frames arrive."""

    def test_last_frame_stored_after_submit(self):
        """submit_frame stores the frame for freeze-frame re-sending."""
        sink = VirtualCameraSink()
        # Don't start (no real device) — just test the queue/frame storage
        sink._stop_event.clear()

        frame = _make_1080p_frame(fill=200)
        sink.submit_frame(frame)

        # Frame should be in the queue (not yet consumed)
        assert not sink._frame_queue.empty()

    def test_send_freeze_frame_resends_last_frame(self):
        """_send_freeze_frame re-sends the stored last frame."""
        sink = VirtualCameraSink()
        frame = _make_1080p_frame(fill=150)

        # Manually set last frame (simulating a previously sent frame)
        sink._last_frame = frame

        # Mock the camera device
        mock_cam = MagicMock()
        sink._cam = mock_cam

        result = sink._send_freeze_frame()

        assert result is True
        mock_cam.send.assert_called_once()
        sent_frame = mock_cam.send.call_args[0][0]
        np.testing.assert_array_equal(sent_frame, frame)

    def test_send_freeze_frame_sends_placeholder_when_no_frame(self):
        """_send_freeze_frame sends placeholder when no frame has been received."""
        sink = VirtualCameraSink()
        sink._last_frame = None

        mock_cam = MagicMock()
        sink._cam = mock_cam

        result = sink._send_freeze_frame()

        assert result is True
        mock_cam.send.assert_called_once()


# ---------------------------------------------------------------------------
# 3. TransportManager → Orchestrator → Pipeline callback chain
# ---------------------------------------------------------------------------

class TestFreezeCallbackChain:
    """on_freeze/on_unfreeze propagate from TransportManager through Orchestrator to Pipeline."""

    def test_orchestrator_wires_freeze_to_pipeline(self):
        """_wire_callbacks connects TM.on_freeze → orch._on_freeze → pipeline.freeze()."""
        bus = EventBus()
        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        tm = TransportManager(bus)

        orch = AppOrchestrator(
            event_bus=bus,
            camera_manager=cm,
            transport_manager=tm,
        )

        # Set up mock pipeline
        mock_pipeline = MagicMock()
        mock_pipeline.freeze = MagicMock()
        mock_pipeline.unfreeze = MagicMock()
        orch._pipeline = mock_pipeline

        # Wire callbacks
        orch._wire_callbacks()

        # Trigger freeze via TransportManager's callback
        tm.on_freeze()

        mock_pipeline.freeze.assert_called_once()

    def test_orchestrator_wires_unfreeze_to_pipeline(self):
        """_wire_callbacks connects TM.on_unfreeze → orch._on_unfreeze → pipeline.unfreeze()."""
        bus = EventBus()
        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        tm = TransportManager(bus)

        orch = AppOrchestrator(
            event_bus=bus,
            camera_manager=cm,
            transport_manager=tm,
        )

        mock_pipeline = MagicMock()
        mock_pipeline.freeze = MagicMock()
        mock_pipeline.unfreeze = MagicMock()
        orch._pipeline = mock_pipeline

        orch._wire_callbacks()

        tm.on_unfreeze()

        mock_pipeline.unfreeze.assert_called_once()

    def test_freeze_is_noop_without_pipeline(self):
        """_on_freeze is a no-op when no pipeline exists (not yet started)."""
        bus = EventBus()
        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        tm = TransportManager(bus)

        orch = AppOrchestrator(
            event_bus=bus,
            camera_manager=cm,
            transport_manager=tm,
        )

        orch._wire_callbacks()

        # Should not raise even without a pipeline
        tm.on_freeze()
        tm.on_unfreeze()


# ---------------------------------------------------------------------------
# 4. Pipeline.switch_stream() hot-swaps decoder
# ---------------------------------------------------------------------------

class TestPipelineSwitchStream:
    """switch_stream() swaps decoder while keeping vcam sink alive."""

    @pytest.mark.asyncio
    async def test_switch_stream_keeps_vcam_alive(self):
        """switch_stream() stops old decoder, creates new, keeps vcam sink running."""
        pipeline = FramePipeline(PipelineConfig())
        pipeline._state = PipelineState.FREEZE_FRAME
        pipeline._was_frozen = True

        # Set up existing vcam sink (mocked)
        mock_sink = MagicMock(spec=VirtualCameraSink)
        mock_sink.is_running = True
        pipeline._vcam_sink = mock_sink

        # Set up existing decoder (mocked)
        old_decoder = MagicMock()
        pipeline._decoder = old_decoder

        new_stream = _make_stream_info(port=9999)

        with patch("gomaxwebcam.pipeline.frame_pipeline.UDPDecoder") as MockDecoder:
            mock_new_decoder = MagicMock()
            mock_new_decoder.start.return_value = True
            MockDecoder.return_value = mock_new_decoder

            result = await pipeline.switch_stream(new_stream)

        assert result is True

        # Old decoder was stopped
        old_decoder.stop.assert_called_once()

        # New decoder created with correct port
        MockDecoder.assert_called_once()
        call_kwargs = MockDecoder.call_args[1]
        assert call_kwargs["udp_port"] == 9999

        # Vcam sink was NOT stopped — still alive
        mock_sink.stop.assert_not_called()
        assert pipeline._vcam_sink is mock_sink

    @pytest.mark.asyncio
    async def test_switch_stream_falls_back_on_no_vcam(self):
        """switch_stream() falls back to full restart when no vcam sink exists."""
        pipeline = FramePipeline(PipelineConfig())
        pipeline._vcam_sink = None
        pipeline._state = PipelineState.STOPPED

        new_stream = _make_stream_info(port=9999)

        # Mock restart to verify it's called
        pipeline.restart = AsyncMock(return_value=True)

        result = await pipeline.switch_stream(new_stream)

        assert result is True
        pipeline.restart.assert_awaited_once_with(new_stream)

    @pytest.mark.asyncio
    async def test_switch_stream_fails_if_decoder_start_fails(self):
        """switch_stream() returns False and sets ERROR state when decoder fails to start."""
        pipeline = FramePipeline(PipelineConfig())
        pipeline._state = PipelineState.FREEZE_FRAME

        mock_sink = MagicMock(spec=VirtualCameraSink)
        mock_sink.is_running = True
        pipeline._vcam_sink = mock_sink

        pipeline._decoder = MagicMock()

        new_stream = _make_stream_info(port=9999)

        with patch("gomaxwebcam.pipeline.frame_pipeline.UDPDecoder") as MockDecoder:
            mock_new_decoder = MagicMock()
            mock_new_decoder.start.return_value = False
            MockDecoder.return_value = mock_new_decoder

            result = await pipeline.switch_stream(new_stream)

        assert result is False
        assert pipeline.state == PipelineState.ERROR


# ---------------------------------------------------------------------------
# 5. Orchestrator._ensure_pipeline() uses hot-swap during failover
# ---------------------------------------------------------------------------

class TestOrchestratorHotSwap:
    """_ensure_pipeline uses switch_stream when pipeline is already running."""

    @pytest.mark.asyncio
    async def test_ensure_pipeline_hot_swaps_when_running(self):
        """When pipeline is running, _ensure_pipeline uses switch_stream instead of restart."""
        bus = EventBus()
        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        cm.set_transport = MagicMock()
        cm.set_pipeline = MagicMock()
        tm = TransportManager(bus)

        orch = AppOrchestrator(
            event_bus=bus,
            camera_manager=cm,
            transport_manager=tm,
        )

        # Set up running pipeline with mock switch_stream
        mock_pipeline = MagicMock(spec=FramePipeline)
        mock_pipeline.is_running = True
        mock_pipeline.switch_stream = AsyncMock(return_value=True)
        mock_pipeline.stop = AsyncMock()
        orch._pipeline = mock_pipeline

        # Create mock transport
        transport = _make_mock_transport("COHN", port=9999)

        await orch._ensure_pipeline(transport)

        # switch_stream should be called, NOT stop+start
        mock_pipeline.switch_stream.assert_awaited_once()
        mock_pipeline.stop.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ensure_pipeline_creates_new_when_no_pipeline(self):
        """When no pipeline exists, _ensure_pipeline creates a new one."""
        bus = EventBus()
        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        cm.set_transport = MagicMock()
        cm.set_pipeline = MagicMock()
        tm = TransportManager(bus)

        orch = AppOrchestrator(
            event_bus=bus,
            camera_manager=cm,
            transport_manager=tm,
        )

        transport = _make_mock_transport("USB")

        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPipeline:
            mock_pl = MagicMock()
            mock_pl.start = AsyncMock(return_value=True)
            MockPipeline.return_value = mock_pl

            await orch._ensure_pipeline(transport)

            MockPipeline.assert_called_once()
            mock_pl.start.assert_awaited_once()
            assert orch._pipeline is mock_pl

    @pytest.mark.asyncio
    async def test_ensure_pipeline_falls_back_on_hot_swap_failure(self):
        """When switch_stream fails, _ensure_pipeline falls back to full restart."""
        bus = EventBus()
        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        cm.set_transport = MagicMock()
        cm.set_pipeline = MagicMock()
        tm = TransportManager(bus)

        orch = AppOrchestrator(
            event_bus=bus,
            camera_manager=cm,
            transport_manager=tm,
        )

        # Running pipeline where switch_stream fails
        old_pipeline = MagicMock(spec=FramePipeline)
        old_pipeline.is_running = True
        old_pipeline.switch_stream = AsyncMock(return_value=False)
        old_pipeline.stop = AsyncMock()
        orch._pipeline = old_pipeline

        transport = _make_mock_transport("COHN", port=9999)

        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPipeline:
            mock_pl = MagicMock()
            mock_pl.start = AsyncMock(return_value=True)
            MockPipeline.return_value = mock_pl

            await orch._ensure_pipeline(transport)

            # Old pipeline stopped after failed hot-swap
            old_pipeline.stop.assert_awaited()
            # New pipeline created
            MockPipeline.assert_called_once()
            mock_pl.start.assert_awaited_once()


# ---------------------------------------------------------------------------
# 6. Full integration: USB disconnect → freeze → COHN → unfreeze
# ---------------------------------------------------------------------------

class TestFreezeFrameFullIntegration:
    """End-to-end: freeze-frame is displayed during USB→COHN failover."""

    @pytest.mark.asyncio
    async def test_freeze_frame_through_full_failover_path(self):
        """USB disconnect → freeze callback → COHN failover → unfreeze callback.

        Proves the complete freeze-frame chain from TransportManager
        disconnect detection through AppOrchestrator wiring to the
        FramePipeline state machine.
        """
        bus = EventBus()

        usb = _make_mock_transport("USB")
        cohn = _make_mock_transport("COHN", port=9999)

        tm = TransportManager(bus, config=_fast_config())
        tm.register_transport("USB", usb)
        tm.register_transport("COHN", cohn)

        cm = MagicMock(spec=CameraManager)
        cm.start = AsyncMock()
        cm.stop = AsyncMock()
        cm.set_transport = MagicMock()
        cm.set_pipeline = MagicMock()

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore") as mock_store_cls:
            mock_store_cls.return_value.list_cameras.return_value = []

            orch = AppOrchestrator(
                event_bus=bus,
                camera_manager=cm,
                transport_manager=tm,
            )

        # Install mock pipeline to track freeze/unfreeze calls
        mock_pipeline = MagicMock(spec=FramePipeline)
        mock_pipeline.freeze = MagicMock()
        mock_pipeline.unfreeze = MagicMock()
        mock_pipeline.stop = AsyncMock()
        mock_pipeline.is_running = True
        mock_pipeline.switch_stream = AsyncMock(return_value=True)
        orch._pipeline = mock_pipeline

        # Wire callbacks (normally done in start())
        orch._wire_callbacks()

        # Start TransportManager on USB
        ok = await tm.start()
        assert ok is True
        assert tm.active_transport_name == "USB"

        # Reset call counts after startup (startup also calls switch_stream)
        mock_pipeline.freeze.reset_mock()
        mock_pipeline.unfreeze.reset_mock()
        mock_pipeline.switch_stream.reset_mock()

        # Simulate USB cable pull
        listener = usb._disconnect_listeners[0]
        listener(usb, "cable_pull")
        await asyncio.sleep(0.3)

        # Verify freeze was called FIRST (before COHN discovery)
        mock_pipeline.freeze.assert_called_once()

        # Verify unfreeze was called AFTER COHN success
        mock_pipeline.unfreeze.assert_called_once()

        # Verify hot-swap was used (switch_stream, not full restart)
        mock_pipeline.switch_stream.assert_awaited_once()

        # Verify final state
        assert tm.active_transport_name == "COHN"
        assert tm.state == ManagerState.ACTIVE

        await tm.stop()

    @pytest.mark.asyncio
    async def test_freeze_fires_before_any_failover_transport_calls(self):
        """Freeze callback fires BEFORE any COHN discover/connect/start_stream calls."""
        bus = EventBus()

        usb = _make_mock_transport("USB")
        cohn = _make_mock_transport("COHN")

        call_order = []

        tm = TransportManager(bus, config=_fast_config())
        tm.register_transport("USB", usb)
        tm.register_transport("COHN", cohn)

        # Track callback ordering
        tm.on_freeze = lambda: call_order.append("freeze")
        tm.on_unfreeze = lambda: call_order.append("unfreeze")

        # Instrument COHN to track ordering
        orig_discover = cohn.discover

        async def tracked_discover():
            call_order.append("cohn_discover")
            return await orig_discover()

        cohn.discover = AsyncMock(side_effect=tracked_discover)

        await tm.start()

        listener = usb._disconnect_listeners[0]
        listener(usb, "cable_pull")
        await asyncio.sleep(0.3)

        # freeze must be first
        assert call_order[0] == "freeze", f"Expected freeze first, got: {call_order}"
        # unfreeze must be last
        assert call_order[-1] == "unfreeze", f"Expected unfreeze last, got: {call_order}"
        # cohn_discover must be between freeze and unfreeze
        assert "cohn_discover" in call_order
        freeze_idx = call_order.index("freeze")
        discover_idx = call_order.index("cohn_discover")
        unfreeze_idx = call_order.index("unfreeze")
        assert freeze_idx < discover_idx < unfreeze_idx

        await tm.stop()

    @pytest.mark.asyncio
    async def test_connection_events_include_freeze_frame_state(self):
        """EventBus receives FREEZE_FRAME connection state during failover."""
        bus = EventBus()

        usb = _make_mock_transport("USB")
        cohn = _make_mock_transport("COHN")

        tm = TransportManager(bus, config=_fast_config())
        tm.register_transport("USB", usb)
        tm.register_transport("COHN", cohn)
        tm.on_freeze = lambda: None
        tm.on_unfreeze = lambda: None

        await tm.start()

        listener = usb._disconnect_listeners[0]
        listener(usb, "disconnect")
        await asyncio.sleep(0.3)

        # Check for FREEZE_FRAME in connection events
        recent = bus.recent_events
        conn_events = [
            e for e in recent
            if e.type == EventType.CONNECTION
        ]

        freeze_events = [
            e for e in conn_events
            if e.data.get("new_state") == "FREEZE_FRAME"
        ]
        assert len(freeze_events) >= 1, (
            f"Expected FREEZE_FRAME connection event, got: "
            f"{[e.data for e in conn_events]}"
        )

        await tm.stop()

    @pytest.mark.asyncio
    async def test_pipeline_state_change_callback_fires_on_freeze(self):
        """Pipeline on_state_change callback fires when entering FREEZE_FRAME."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.STREAMING

        state_log = []
        pipeline.on_state_change = lambda s: state_log.append(s)

        pipeline.freeze()

        assert PipelineState.FREEZE_FRAME in state_log

    @pytest.mark.asyncio
    async def test_pipeline_state_change_callback_fires_on_resume(self):
        """Pipeline on_state_change fires STREAMING when first frame arrives after freeze."""
        pipeline = FramePipeline()
        pipeline._state = PipelineState.FREEZE_FRAME
        pipeline._was_frozen = True

        mock_sink = MagicMock()
        pipeline._vcam_sink = mock_sink

        state_log = []
        pipeline.on_state_change = lambda s: state_log.append(s)

        frame = _make_1080p_frame()
        pipeline._on_decoded_frame(frame)

        assert PipelineState.STREAMING in state_log
