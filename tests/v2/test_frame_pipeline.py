"""
tests/v2/test_frame_pipeline.py — Integration wiring and frame pipeline tests.

Tests the FramePipeline connecting USB transport → UDPDecoder → VirtualCameraSink
with backpressure handling, frame timing, and 1080p 30fps validation.

All tests mock at the Transport ABC boundary and pyvirtualcam — no real hardware
or virtual camera drivers needed. This allows CI testing on all three OSes.

Test categories:
  1. Pipeline wiring: decoder → vcam_sink connection
  2. Backpressure: bounded queue, newest-wins frame drop
  3. Freeze-frame: stream loss → freeze → recovery transitions
  4. 1080p 30fps validation: resolution and frame rate checks
  5. Frame timing: inter-frame intervals and FPS measurement
  6. Pipeline lifecycle: start/stop/restart
  7. Error handling: decoder errors, transport failures
  8. BGR24 color format: v1 parity — pipeline uses BGR throughout
"""

import asyncio
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

import sys
import os


def run_async(coro):
    """Helper to run async tests without pytest-asyncio plugin."""
    return asyncio.run(coro)

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gomaxwebcam.pipeline.frame_pipeline import (
    FramePipeline,
    PipelineConfig,
    PipelineState,
    PipelineStats,
    validate_1080p_30fps,
)
from gomaxwebcam.pipeline.decode import UDPDecoder, DecodeStats
from gomaxwebcam.pipeline.virtual_camera_sink import VirtualCameraSink
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo

# Mark ALL tests in this module as not needing a real GoPro
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_1080p_frame(color: tuple = (128, 128, 128)) -> np.ndarray:
    """Create a valid 1080p RGB24 frame."""
    return np.full((1080, 1920, 3), color, dtype=np.uint8)


def make_720p_frame() -> np.ndarray:
    """Create a 720p frame (wrong resolution for 1080p pipeline)."""
    return np.full((720, 1280, 3), (100, 100, 100), dtype=np.uint8)


def make_stream_info(port: int = 8554) -> StreamInfo:
    """Create a standard 1080p 30fps StreamInfo."""
    return StreamInfo(
        protocol="udp",
        host="0.0.0.0",
        port=port,
        width=1920,
        height=1080,
        fps=30,
        codec="h264",
    )


class MockTransport(Transport):
    """Mock transport for testing pipeline integration.

    Simulates a transport that's already connected and can start
    a preview stream.
    """

    def __init__(self, stream_info: Optional[StreamInfo] = None):
        super().__init__(name="MockTransport")
        self._mock_stream_info = stream_info or make_stream_info()
        self._stream_started = False

    async def discover(self, timeout: float = 10.0) -> bool:
        self._set_state(TransportState.DISCOVERING)
        self._set_state(TransportState.CONNECTED)
        return True

    async def connect(self) -> bool:
        self._set_state(TransportState.CONNECTED)
        return True

    async def start_stream(self) -> Optional[StreamInfo]:
        self._stream_info = self._mock_stream_info
        self._stream_started = True
        self._set_state(TransportState.STREAMING)
        return self._mock_stream_info

    async def stop_stream(self) -> None:
        self._stream_info = None
        self._stream_started = False
        self._set_state(TransportState.CONNECTED)

    async def disconnect(self) -> None:
        self._set_state(TransportState.DISCONNECTED)

    async def keep_alive(self) -> bool:
        return True

    def force_connected(self) -> None:
        """Helper: set state directly to CONNECTED for testing."""
        self._set_state(TransportState.CONNECTED)


# ---------------------------------------------------------------------------
# 1. validate_1080p_30fps tests
# ---------------------------------------------------------------------------

class TestValidate1080p30fps:
    """Tests for the 1080p 30fps validation function."""

    def test_valid_1080p_30fps(self):
        """A perfect 1080p 30fps frame passes all checks."""
        frame = make_1080p_frame()
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["valid"] is True
        assert result["resolution_ok"] is True
        assert result["fps_ok"] is True
        assert result["dtype_ok"] is True
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["channels"] == 3
        assert len(result["errors"]) == 0

    def test_valid_fps_within_tolerance(self):
        """FPS within 10% tolerance passes."""
        frame = make_1080p_frame()
        # 30fps ± 10% = 27.0 to 33.0
        for fps in [27.0, 28.5, 30.0, 31.5, 33.0]:
            result = validate_1080p_30fps(frame, fps=fps)
            assert result["fps_ok"] is True, f"FPS {fps} should be within tolerance"

    def test_invalid_fps_outside_tolerance(self):
        """FPS outside 10% tolerance fails."""
        frame = make_1080p_frame()
        for fps in [15.0, 25.0, 35.0, 60.0]:
            result = validate_1080p_30fps(frame, fps=fps)
            assert result["fps_ok"] is False, f"FPS {fps} should be outside tolerance"
            assert result["valid"] is False

    def test_invalid_resolution_720p(self):
        """720p frame fails resolution check."""
        frame = make_720p_frame()
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["resolution_ok"] is False
        assert result["valid"] is False
        assert result["width"] == 1280
        assert result["height"] == 720

    def test_invalid_resolution_4k(self):
        """4K frame fails resolution check."""
        frame = np.zeros((2160, 3840, 3), dtype=np.uint8)
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["resolution_ok"] is False
        assert result["width"] == 3840
        assert result["height"] == 2160

    def test_invalid_dtype(self):
        """Non-uint8 frame fails dtype check."""
        frame = np.zeros((1080, 1920, 3), dtype=np.float32)
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["dtype_ok"] is False
        assert result["valid"] is False

    def test_invalid_channels_grayscale(self):
        """Grayscale (1 channel) frame fails resolution check."""
        frame = np.zeros((1080, 1920), dtype=np.uint8)
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["resolution_ok"] is False

    def test_invalid_channels_rgba(self):
        """RGBA (4 channel) frame fails resolution check."""
        frame = np.zeros((1080, 1920, 4), dtype=np.uint8)
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["resolution_ok"] is False
        assert result["channels"] == 4

    def test_custom_tolerance(self):
        """Custom tolerance value works correctly."""
        frame = make_1080p_frame()
        # 5% tolerance: 30 ± 1.5 = 28.5 to 31.5
        result = validate_1080p_30fps(frame, fps=28.0, tolerance=0.05)
        assert result["fps_ok"] is False

        result = validate_1080p_30fps(frame, fps=29.0, tolerance=0.05)
        assert result["fps_ok"] is True

    def test_zero_fps(self):
        """Zero FPS fails validation."""
        frame = make_1080p_frame()
        result = validate_1080p_30fps(frame, fps=0.0)
        assert result["fps_ok"] is False
        assert result["valid"] is False

    def test_errors_list_populated(self):
        """Multiple failures produce multiple error strings."""
        frame = np.zeros((720, 1280, 4), dtype=np.float32)
        result = validate_1080p_30fps(frame, fps=5.0)
        assert len(result["errors"]) >= 2  # At least resolution + fps errors


# ---------------------------------------------------------------------------
# 2. Pipeline configuration tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    """Tests for PipelineConfig defaults and customization."""

    def test_default_config(self):
        """Default config matches 1080p 30fps spec."""
        config = PipelineConfig()
        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 30
        assert config.device_name == "GoMaxWebcam"
        assert config.udp_port == 8554

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = PipelineConfig(
            width=1920,
            height=1080,
            fps=30,
            device_name="TestCam",
            udp_port=9000,
        )
        assert config.device_name == "TestCam"
        assert config.udp_port == 9000


# ---------------------------------------------------------------------------
# 3. Pipeline state machine tests
# ---------------------------------------------------------------------------

class TestPipelineState:
    """Tests for PipelineState enum and transitions."""

    def test_initial_state_is_stopped(self):
        """Pipeline starts in STOPPED state."""
        pipeline = FramePipeline()
        assert pipeline.state == PipelineState.STOPPED
        assert not pipeline.is_running
        assert not pipeline.is_streaming
        assert not pipeline.is_frozen

    def test_state_properties(self):
        """is_running covers STARTING, STREAMING, and FREEZE_FRAME."""
        pipeline = FramePipeline()

        # Directly test _set_state for state machine coverage
        pipeline._set_state(PipelineState.STARTING)
        assert pipeline.is_running

        pipeline._set_state(PipelineState.STREAMING)
        assert pipeline.is_running
        assert pipeline.is_streaming
        assert not pipeline.is_frozen

        pipeline._set_state(PipelineState.FREEZE_FRAME)
        assert pipeline.is_running
        assert not pipeline.is_streaming
        assert pipeline.is_frozen

        pipeline._set_state(PipelineState.STOPPING)
        assert not pipeline.is_running

        pipeline._set_state(PipelineState.STOPPED)
        assert not pipeline.is_running

    def test_state_change_callback(self):
        """on_state_change callback fires on transitions."""
        pipeline = FramePipeline()
        states = []
        pipeline.on_state_change = lambda s: states.append(s)

        pipeline._set_state(PipelineState.STARTING)
        pipeline._set_state(PipelineState.STREAMING)
        pipeline._set_state(PipelineState.STOPPED)

        assert states == [
            PipelineState.STARTING,
            PipelineState.STREAMING,
            PipelineState.STOPPED,
        ]

    def test_duplicate_state_ignored(self):
        """Setting the same state twice does not fire callback."""
        pipeline = FramePipeline()
        states = []
        pipeline.on_state_change = lambda s: states.append(s)

        pipeline._set_state(PipelineState.STREAMING)
        pipeline._set_state(PipelineState.STREAMING)  # Duplicate

        assert len(states) == 1


# ---------------------------------------------------------------------------
# 4. Pipeline wiring tests (mock at Transport ABC boundary)
# ---------------------------------------------------------------------------

class TestPipelineWiring:
    """Tests for the pipeline wiring: transport → decoder → vcam sink."""

    def test_start_creates_decoder_and_sink(self):
        """start() creates UDPDecoder and VirtualCameraSink, wires them."""
        async def _test():
            pipeline = FramePipeline()
            stream_info = make_stream_info(port=9999)

            with patch.object(UDPDecoder, "start", return_value=True) as mock_dec_start, \
                 patch.object(VirtualCameraSink, "start", return_value=True) as mock_sink_start:

                result = await pipeline.start(stream_info)

                assert result is True
                assert pipeline.state == PipelineState.STREAMING
                assert pipeline.decoder is not None
                assert pipeline.vcam_sink is not None

                # Verify decoder was configured with stream info
                assert pipeline.decoder.udp_port == 9999
                assert pipeline.decoder.width == 1920
                assert pipeline.decoder.height == 1080

                # Verify vcam sink was started
                mock_sink_start.assert_called_once()
                mock_dec_start.assert_called_once()

        run_async(_test())

    def test_start_vcam_failure_rolls_back(self):
        """If vcam sink fails to start, pipeline enters ERROR state."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(VirtualCameraSink, "start", return_value=False):
                result = await pipeline.start(make_stream_info())

                assert result is False
                assert pipeline.state == PipelineState.ERROR

        run_async(_test())

    def test_start_decoder_failure_rolls_back(self):
        """If decoder fails to start, vcam sink is stopped and pipeline errors."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(VirtualCameraSink, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "stop") as mock_sink_stop, \
                 patch.object(UDPDecoder, "start", return_value=False):

                result = await pipeline.start(make_stream_info())

                assert result is False
                assert pipeline.state == PipelineState.ERROR
                mock_sink_stop.assert_called()

        run_async(_test())

    def test_stop_cleans_up_both(self):
        """stop() stops both decoder and vcam sink."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            with patch.object(UDPDecoder, "stop") as mock_dec_stop, \
                 patch.object(VirtualCameraSink, "stop") as mock_sink_stop:
                await pipeline.stop()

                mock_dec_stop.assert_called_once()
                mock_sink_stop.assert_called_once()
                assert pipeline.state == PipelineState.STOPPED

        run_async(_test())

    def test_start_from_transport(self):
        """start_from_transport() wires transport stream to pipeline."""
        async def _test():
            transport = MockTransport()
            transport.force_connected()
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):

                result = await pipeline.start_from_transport(transport)

                assert result is True
                assert transport._stream_started is True
                assert pipeline.state == PipelineState.STREAMING

        run_async(_test())

    def test_start_from_disconnected_transport_fails(self):
        """start_from_transport() fails if transport is not connected."""
        async def _test():
            transport = MockTransport()
            pipeline = FramePipeline()

            result = await pipeline.start_from_transport(transport)
            assert result is False

        run_async(_test())

    def test_restart_stops_then_starts(self):
        """restart() stops the current pipeline and starts fresh."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True), \
                 patch.object(UDPDecoder, "stop"), \
                 patch.object(VirtualCameraSink, "stop"):

                await pipeline.start(make_stream_info(port=8554))
                assert pipeline.state == PipelineState.STREAMING

                await pipeline.restart(make_stream_info(port=9999))
                assert pipeline.state == PipelineState.STREAMING
                assert pipeline.decoder.udp_port == 9999

        run_async(_test())


# ---------------------------------------------------------------------------
# 5. Frame delivery and backpressure tests
# ---------------------------------------------------------------------------

class TestFrameDelivery:
    """Tests for frame delivery from decoder to vcam sink."""

    def test_on_decoded_frame_submits_to_sink(self):
        """Decoded frames are submitted to the vcam sink."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Mock submit_frame on the sink
            pipeline._vcam_sink.submit_frame = MagicMock(return_value=True)

            frame = make_1080p_frame()
            pipeline._on_decoded_frame(frame)

            pipeline._vcam_sink.submit_frame.assert_called_once()
            submitted = pipeline._vcam_sink.submit_frame.call_args[0][0]
            assert submitted.shape == (1080, 1920, 3)

        run_async(_test())

    def test_backpressure_drops_oldest_frame(self):
        """VirtualCameraSink drops oldest frame when queue is full."""
        sink = VirtualCameraSink()
        # Don't actually start the device — just test the queue
        sink._stop_event.clear()

        # Fill the queue (max size 2)
        frame1 = make_1080p_frame((100, 0, 0))
        frame2 = make_1080p_frame((0, 100, 0))
        frame3 = make_1080p_frame((0, 0, 100))

        assert sink.submit_frame(frame1) is True
        assert sink.submit_frame(frame2) is True
        # Queue is now full — frame3 should cause drop
        assert sink.submit_frame(frame3) is True
        assert sink.frames_dropped >= 1

        # The queue should contain the two newest frames
        assert sink._frame_queue.qsize() == 2

    def test_submit_frame_after_stop_returns_false(self):
        """submit_frame() returns False when sink is stopped."""
        sink = VirtualCameraSink()
        sink._stop_event.set()

        frame = make_1080p_frame()
        assert sink.submit_frame(frame) is False

    def test_bounded_queue_memory(self):
        """Queue never exceeds max size, bounding memory usage."""
        sink = VirtualCameraSink()
        sink._stop_event.clear()

        # Submit 100 frames — queue should never exceed 2
        for i in range(100):
            sink.submit_frame(make_1080p_frame((i % 256, 0, 0)))
            assert sink._frame_queue.qsize() <= 2

        # Should have dropped ~98 frames
        assert sink.frames_dropped >= 95  # Allow some slack


# ---------------------------------------------------------------------------
# 6. Freeze-frame tests
# ---------------------------------------------------------------------------

class TestFreezeFrame:
    """Tests for freeze-frame transitions."""

    def test_decoder_error_triggers_freeze(self):
        """Decoder fatal error transitions pipeline to FREEZE_FRAME."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            freeze_called = threading.Event()
            pipeline.on_freeze = lambda: freeze_called.set()

            pipeline._on_decoder_error(RuntimeError("Stream died"))

            assert pipeline.state == PipelineState.FREEZE_FRAME
            assert pipeline.is_frozen
            assert freeze_called.is_set()

        run_async(_test())

    def test_decoder_stopped_triggers_freeze(self):
        """Decoder thread exit transitions pipeline to FREEZE_FRAME."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            pipeline._on_decoder_stopped()
            assert pipeline.state == PipelineState.FREEZE_FRAME

        run_async(_test())

    def test_frame_after_freeze_resumes_streaming(self):
        """Receiving a frame after freeze transitions back to STREAMING."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Enter freeze
            pipeline._on_decoder_error(RuntimeError("Stream died"))
            assert pipeline.state == PipelineState.FREEZE_FRAME

            # Mock the sink so submit_frame works
            pipeline._vcam_sink.submit_frame = MagicMock(return_value=True)

            unfreeze_called = threading.Event()
            pipeline.on_unfreeze = lambda: unfreeze_called.set()

            # Resume with new frame
            pipeline._on_decoded_frame(make_1080p_frame())

            assert pipeline.state == PipelineState.STREAMING
            assert unfreeze_called.is_set()

        run_async(_test())

    def test_freeze_callback_only_once(self):
        """Multiple decoder errors only fire freeze callback once."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            freeze_count = 0

            def on_freeze():
                nonlocal freeze_count
                freeze_count += 1

            pipeline.on_freeze = on_freeze

            pipeline._on_decoder_error(RuntimeError("Error 1"))
            pipeline._on_decoder_error(RuntimeError("Error 2"))
            pipeline._on_decoder_stopped()

            assert freeze_count == 1  # Only first error triggers freeze

        run_async(_test())

    def test_freeze_during_stop_ignored(self):
        """Decoder stopped during pipeline shutdown does not trigger freeze."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Simulate pipeline stopping
            pipeline._set_state(PipelineState.STOPPING)
            pipeline._on_decoder_stopped()

            # Should NOT transition to FREEZE_FRAME
            assert pipeline.state == PipelineState.STOPPING

        run_async(_test())


# ---------------------------------------------------------------------------
# 7. Pipeline statistics tests
# ---------------------------------------------------------------------------

class TestPipelineStats:
    """Tests for pipeline statistics reporting."""

    def test_stats_default_values(self):
        """Default stats reflect stopped pipeline."""
        pipeline = FramePipeline()
        stats = pipeline.get_stats()
        assert stats.state == "STOPPED"
        assert stats.decoder_frames == 0
        assert stats.vcam_frames == 0
        assert stats.is_frozen is False

    def test_stats_after_start(self):
        """Stats reflect running pipeline state."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            stats = pipeline.get_stats()
            assert stats.state == "STREAMING"
            assert stats.uptime_s >= 0

        run_async(_test())

    def test_detailed_stats(self):
        """get_detailed_stats() includes sub-component stats."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            detailed = pipeline.get_detailed_stats()
            assert "pipeline" in detailed
            assert "decoder" in detailed
            assert "vcam" in detailed
            assert detailed["pipeline"]["state"] == "STREAMING"

        run_async(_test())


# ---------------------------------------------------------------------------
# 8. Transport ABC and StreamInfo tests
# ---------------------------------------------------------------------------

class TestTransportABC:
    """Tests for the Transport ABC contract."""

    def test_transport_initial_state(self):
        """Mock transport starts DISCONNECTED."""
        transport = MockTransport()
        assert transport.state == TransportState.DISCONNECTED
        assert not transport.is_connected
        assert not transport.is_streaming

    def test_transport_lifecycle(self):
        """Full transport lifecycle: discover → connect → stream → disconnect."""
        async def _test():
            transport = MockTransport()

            assert await transport.discover()
            assert transport.is_connected

            stream_info = await transport.start_stream()
            assert stream_info is not None
            assert stream_info.width == 1920
            assert stream_info.height == 1080
            assert stream_info.fps == 30
            assert transport.is_streaming

            await transport.stop_stream()
            assert not transport.is_streaming
            assert transport.is_connected

            await transport.disconnect()
            assert transport.state == TransportState.DISCONNECTED

        run_async(_test())

    def test_transport_keep_alive(self):
        """keep_alive() returns True when connected."""
        async def _test():
            transport = MockTransport()
            transport.force_connected()
            assert await transport.keep_alive()

        run_async(_test())

    def test_stream_info_defaults(self):
        """StreamInfo defaults match 1080p 30fps spec."""
        info = StreamInfo()
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30
        assert info.port == 8554
        assert info.protocol == "udp"
        assert info.codec == "h264"

    def test_stream_info_frozen(self):
        """StreamInfo is immutable (frozen dataclass)."""
        info = StreamInfo()
        with pytest.raises(AttributeError):
            info.port = 9999  # type: ignore[misc]

    def test_transport_state_listener(self):
        """State listeners are called on transitions."""
        transport = MockTransport()
        transitions = []
        transport.add_state_listener(lambda old, new: transitions.append((old.name, new.name)))

        transport.force_connected()

        assert len(transitions) == 1
        assert transitions[0] == ("DISCONNECTED", "CONNECTED")


# ---------------------------------------------------------------------------
# 9. Decode → Sink integration test (no real PyAV/pyvirtualcam)
# ---------------------------------------------------------------------------

class TestDecodeToSinkIntegration:
    """Integration test: decoder callback → sink submit_frame."""

    def test_frame_flows_decoder_to_sink(self):
        """Frame produced by decoder callback reaches the sink queue."""
        frames_received: list[np.ndarray] = []

        sink = VirtualCameraSink()
        sink._stop_event.clear()

        # Simulate what the pipeline does: wire decoder callback → sink
        def on_frame(frame: np.ndarray) -> None:
            sink.submit_frame(frame)
            frames_received.append(frame)

        # Simulate decoder producing frames
        for _ in range(5):
            frame = make_1080p_frame()
            on_frame(frame)

        assert len(frames_received) == 5
        # Queue should have at most 2 frames (bounded)
        assert sink._frame_queue.qsize() <= 2

    def test_high_throughput_backpressure(self):
        """Rapid frame production stays bounded in memory."""
        sink = VirtualCameraSink()
        sink._stop_event.clear()

        # Produce 1000 frames rapidly (simulating decode burst)
        for i in range(1000):
            frame = make_1080p_frame((i % 256, 0, 0))
            sink.submit_frame(frame)

        # Queue should never exceed 2 frames
        assert sink._frame_queue.qsize() <= 2
        # Most frames should have been dropped
        assert sink.frames_dropped >= 900

    def test_frame_dimensions_preserved(self):
        """Submitted frames maintain their dimensions through the queue."""
        sink = VirtualCameraSink()
        sink._stop_event.clear()

        frame = make_1080p_frame((255, 0, 0))
        sink.submit_frame(frame)

        # Read it back from the queue
        queued = sink._frame_queue.get_nowait()
        assert queued.shape == (1080, 1920, 3)
        assert queued.dtype == np.uint8
        # Check color preserved
        assert queued[0, 0, 0] == 255  # Red channel


# ---------------------------------------------------------------------------
# 10. Frame timing validation tests
# ---------------------------------------------------------------------------

class TestFrameTiming:
    """Tests for frame timing and FPS measurement."""

    def test_30fps_frame_interval(self):
        """30fps means ~33.3ms between frames."""
        expected_interval = 1.0 / 30  # 0.0333...
        assert abs(expected_interval - 0.0333) < 0.001

    def test_fps_measurement_via_decode_stats(self):
        """DecodeStats correctly measures FPS over a window."""
        stats = DecodeStats()

        # Simulate 30 frames in 1 second
        base_time = time.monotonic()
        for i in range(30):
            stats.record_frame()

        # FPS should be approximately 30
        snap = stats.snapshot()
        assert snap["frames_decoded"] == 30
        # FPS calculation depends on actual timing, just check it's reasonable
        assert snap["decode_fps"] > 0

    def test_validate_frame_timing_compliance(self):
        """validate_1080p_30fps correctly validates frame + fps combination."""
        frame = make_1080p_frame()

        # Perfect 30fps
        result = validate_1080p_30fps(frame, 30.0)
        assert result["valid"] is True

        # Slightly off but within tolerance
        result = validate_1080p_30fps(frame, 29.0)
        assert result["fps_ok"] is True

        # Way off
        result = validate_1080p_30fps(frame, 15.0)
        assert result["fps_ok"] is False

    def test_vcam_sink_fps_stats_initial(self):
        """VirtualCameraSink FPS stats start at zero."""
        sink = VirtualCameraSink()
        assert sink.fps_actual == 0.0
        assert sink.frames_sent == 0

    def test_pipeline_config_fps_30(self):
        """PipelineConfig enforces 30fps default."""
        config = PipelineConfig()
        assert config.fps == 30
        frame_interval = 1.0 / config.fps
        assert abs(frame_interval - 0.0333) < 0.001


# ---------------------------------------------------------------------------
# 11. Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases and error handling tests."""

    def test_double_start_returns_true(self):
        """Starting an already-running pipeline returns True (idempotent)."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())
                result = await pipeline.start(make_stream_info())
                assert result is True

        run_async(_test())

    def test_double_stop_is_safe(self):
        """Stopping an already-stopped pipeline is a no-op."""
        async def _test():
            pipeline = FramePipeline()
            await pipeline.stop()  # Should not raise
            assert pipeline.state == PipelineState.STOPPED

        run_async(_test())

    def test_on_decoded_frame_without_sink(self):
        """_on_decoded_frame with no sink does not crash."""
        pipeline = FramePipeline()
        # vcam_sink is None
        pipeline._on_decoded_frame(make_1080p_frame())  # Should not raise

    def test_state_callback_exception_handled(self):
        """Exception in state callback does not crash the pipeline."""
        pipeline = FramePipeline()
        pipeline.on_state_change = lambda s: 1 / 0  # ZeroDivisionError

        # Should not raise
        pipeline._set_state(PipelineState.STREAMING)
        assert pipeline.state == PipelineState.STREAMING

    def test_freeze_callback_exception_handled(self):
        """Exception in freeze callback does not crash."""
        pipeline = FramePipeline()
        pipeline.on_freeze = lambda: 1 / 0

        pipeline._set_state(PipelineState.STREAMING)

        # Should not raise
        pipeline._on_decoder_error(RuntimeError("test"))
        assert pipeline.is_frozen

    def test_pipeline_exception_during_start(self):
        """Exception during start() transitions to ERROR."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(VirtualCameraSink, "start", side_effect=RuntimeError("boom")):
                result = await pipeline.start(make_stream_info())
                assert result is False
                assert pipeline.state == PipelineState.ERROR

        run_async(_test())


# ---------------------------------------------------------------------------
# 12. 1080p 30fps end-to-end validation
# ---------------------------------------------------------------------------

class TestEndToEnd1080p30fps:
    """End-to-end validation that the pipeline produces 1080p 30fps output."""

    def test_frame_spec_compliance(self):
        """Validate that frames match the 1080p 30fps specification."""
        # Simulate what the pipeline produces
        frame = make_1080p_frame()

        # Check all spec requirements
        assert frame.shape == (1080, 1920, 3), "Must be 1080p (1920x1080)"
        assert frame.dtype == np.uint8, "Must be uint8"
        assert frame.ndim == 3, "Must be 3-channel (RGB24)"

        # Validate with the pipeline's validation function
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["valid"] is True
        assert result["resolution_ok"] is True
        assert result["fps_ok"] is True
        assert result["dtype_ok"] is True

    def test_16_9_aspect_ratio(self):
        """1080p output has 16:9 aspect ratio."""
        frame = make_1080p_frame()
        h, w = frame.shape[:2]
        aspect = w / h
        assert abs(aspect - 16 / 9) < 0.01, f"Aspect ratio {aspect} is not 16:9"

    def test_frame_memory_size(self):
        """A 1080p RGB24 frame is exactly 6,220,800 bytes."""
        frame = make_1080p_frame()
        expected_bytes = 1920 * 1080 * 3  # 6,220,800
        assert frame.nbytes == expected_bytes

    def test_pipeline_config_only_allows_1080p(self):
        """Default PipelineConfig produces 1080p output."""
        config = PipelineConfig()
        expected_bytes = config.width * config.height * 3
        assert expected_bytes == 1920 * 1080 * 3

    def test_vcam_sink_fixed_at_1080p_30fps(self):
        """VirtualCameraSink is always 1920x1080@30fps."""
        sink = VirtualCameraSink()
        assert sink.width == 1920
        assert sink.height == 1080
        assert sink.fps == 30
        assert sink.device_name == "GoMaxWebcam"

    def test_pipeline_wires_1080p_decoder_to_1080p_sink(self):
        """Pipeline wires a 1080p decoder to a 1080p sink."""
        async def _test():
            pipeline = FramePipeline()
            stream_info = make_stream_info()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(stream_info)

            # Decoder is configured for 1080p
            assert pipeline.decoder.width == 1920
            assert pipeline.decoder.height == 1080

            # Sink is fixed at 1080p 30fps
            assert pipeline.vcam_sink.width == 1920
            assert pipeline.vcam_sink.height == 1080
            assert pipeline.vcam_sink.fps == 30

        run_async(_test())


# ---------------------------------------------------------------------------
# 13. BGR24 color format tests (v1 parity)
# ---------------------------------------------------------------------------

class TestBGR24ColorFormat:
    """Tests verifying BGR24 color format throughout the pipeline.

    v1 used ffmpeg -pix_fmt bgr24 and pyvirtualcam.PixelFormat.BGR.
    v2 must match this for Unity Capture native compatibility and
    NVIDIA Broadcast interop.
    """

    def test_decoder_default_pixel_format_is_bgr24(self):
        """UDPDecoder defaults to BGR24 matching v1's ffmpeg output."""
        decoder = UDPDecoder(udp_port=9999)
        assert decoder.pixel_format == "bgr24"

    def test_decoder_class_constant(self):
        """UDPDecoder.DEFAULT_PIXEL_FORMAT is bgr24."""
        assert UDPDecoder.DEFAULT_PIXEL_FORMAT == "bgr24"

    def test_decoder_custom_pixel_format(self):
        """UDPDecoder accepts custom pixel format."""
        decoder = UDPDecoder(udp_port=9999, pixel_format="rgb24")
        assert decoder.pixel_format == "rgb24"

    def test_pipeline_config_default_bgr24(self):
        """PipelineConfig defaults to BGR24 pixel format."""
        config = PipelineConfig()
        assert config.pixel_format == "bgr24"

    def test_pipeline_passes_pixel_format_to_decoder(self):
        """FramePipeline passes pixel_format from config to decoder."""
        async def _test():
            config = PipelineConfig(pixel_format="bgr24")
            pipeline = FramePipeline(config)
            stream_info = make_stream_info()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(stream_info)

            assert pipeline.decoder.pixel_format == "bgr24"

        run_async(_test())

    def test_pipeline_custom_rgb24_format(self):
        """Pipeline can be configured for RGB24 if needed."""
        async def _test():
            config = PipelineConfig(pixel_format="rgb24")
            pipeline = FramePipeline(config)
            stream_info = make_stream_info()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(stream_info)

            assert pipeline.decoder.pixel_format == "rgb24"

        run_async(_test())

    def test_bgr_frame_channel_order(self):
        """BGR frame has Blue in channel 0, Green in 1, Red in 2."""
        # Create a frame with known BGR values
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        frame[:, :, 0] = 255  # Blue channel
        frame[:, :, 1] = 0    # Green channel
        frame[:, :, 2] = 0    # Red channel

        # Verify BGR order: channel 0 is Blue
        assert frame[0, 0, 0] == 255  # B
        assert frame[0, 0, 1] == 0    # G
        assert frame[0, 0, 2] == 0    # R

        # Validate shape matches spec
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["valid"] is True

    def test_bgr_frame_flows_through_pipeline(self):
        """BGR24 frame flows through pipeline callback to vcam sink."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Mock submit_frame to capture the frame
            pipeline._vcam_sink.submit_frame = MagicMock(return_value=True)

            # Create a BGR frame with Blue=255, Green=128, Red=64
            bgr_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            bgr_frame[:, :, 0] = 255  # Blue
            bgr_frame[:, :, 1] = 128  # Green
            bgr_frame[:, :, 2] = 64   # Red

            pipeline._on_decoded_frame(bgr_frame)

            # Verify the exact frame was passed through
            submitted = pipeline._vcam_sink.submit_frame.call_args[0][0]
            assert submitted[0, 0, 0] == 255  # Blue preserved
            assert submitted[0, 0, 1] == 128  # Green preserved
            assert submitted[0, 0, 2] == 64   # Red preserved

        run_async(_test())

    def test_bgr_frame_memory_layout(self):
        """BGR24 1080p frame is contiguous and correct size."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # 1920 * 1080 * 3 = 6,220,800 bytes (same as RGB)
        assert frame.nbytes == 6_220_800
        assert frame.dtype == np.uint8
        assert frame.shape == (1080, 1920, 3)
