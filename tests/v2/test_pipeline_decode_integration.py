"""
tests/v2/test_pipeline_decode_integration.py — Frame extraction and decode pipeline tests.

Tests the end-to-end frame extraction chain:
  Transport.start_stream() -> StreamInfo -> PipelineConfig -> UDPDecoder -> BGR24 frames

Focus areas:
  1. PipelineConfig.from_stream_info() factory — codec validation, dimension mapping
  2. BGR24 pixel format matches v1 output (Unity Capture compatible)
  3. Resolution handling (1080p, 720p, 4K → 1080p scaling)
  4. Codec support validation (h264, hevc, mjpeg)
  5. Orchestrator._ensure_pipeline() integration with config factory
  6. Frame delivery chain with proper format and dimensions
  7. Decoder error recovery and freeze-frame transitions

All tests mock PyAV and pyvirtualcam — no real hardware or video drivers needed.
"""

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

import sys
import os

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gomaxwebcam.pipeline.frame_pipeline import (
    FramePipeline,
    PipelineConfig,
    PipelineState,
    PipelineStats,
    SUPPORTED_CODECS,
    validate_1080p_30fps,
)
from gomaxwebcam.pipeline.decode import UDPDecoder, DecodeStats, FrameMetadata
from gomaxwebcam.pipeline.virtual_camera_sink import VirtualCameraSink
from gomaxwebcam.transport.base import Transport, TransportState, StreamInfo

# Mark ALL tests in this module as not needing a real GoPro
pytestmark = pytest.mark.no_gopro_needed


def run_async(coro):
    """Helper to run async tests without pytest-asyncio plugin."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_bgr24_frame(
    width: int = 1920,
    height: int = 1080,
    color: tuple = (128, 64, 32),
) -> np.ndarray:
    """Create a BGR24 numpy frame matching v1 output format.

    The frame is (height, width, 3) uint8 in BGR channel order,
    matching ffmpeg -pix_fmt bgr24 output and Unity Capture native format.
    """
    return np.full((height, width, 3), color, dtype=np.uint8)


def make_stream_info(
    port: int = 8554,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
    codec: str = "h264",
) -> StreamInfo:
    """Create a StreamInfo matching GoPro webcam output."""
    return StreamInfo(
        protocol="udp",
        host="0.0.0.0",
        port=port,
        width=width,
        height=height,
        fps=fps,
        codec=codec,
    )


class MockTransport(Transport):
    """Mock transport for pipeline integration testing."""

    def __init__(self, stream_info: Optional[StreamInfo] = None):
        super().__init__(name="MockTransport")
        self._mock_stream_info = stream_info or make_stream_info()

    async def discover(self, timeout: float = 10.0) -> bool:
        self._set_state(TransportState.CONNECTED)
        return True

    async def connect(self) -> bool:
        self._set_state(TransportState.CONNECTED)
        return True

    async def start_stream(self) -> Optional[StreamInfo]:
        self._stream_info = self._mock_stream_info
        self._set_state(TransportState.STREAMING)
        return self._mock_stream_info

    async def stop_stream(self) -> None:
        self._stream_info = None
        self._set_state(TransportState.CONNECTED)

    async def disconnect(self) -> None:
        self._set_state(TransportState.DISCONNECTED)

    async def keep_alive(self) -> bool:
        return True

    def force_streaming(self) -> None:
        """Helper: set state and stream_info for testing."""
        self._stream_info = self._mock_stream_info
        self._set_state(TransportState.STREAMING)


# ===========================================================================
# 1. PipelineConfig.from_stream_info() tests
# ===========================================================================

class TestPipelineConfigFromStreamInfo:
    """Tests for the PipelineConfig factory method."""

    def test_creates_config_from_h264_stream(self):
        """H.264 StreamInfo produces valid config with matching dimensions."""
        info = make_stream_info(port=9000, codec="h264")
        config = PipelineConfig.from_stream_info(info)

        assert config.width == 1920
        assert config.height == 1080
        assert config.fps == 30
        assert config.udp_port == 9000
        assert config.pixel_format == "bgr24"

    def test_creates_config_from_hevc_stream(self):
        """HEVC/H.265 codec is accepted."""
        info = make_stream_info(codec="hevc")
        config = PipelineConfig.from_stream_info(info)
        assert config.width == 1920

    def test_creates_config_from_h265_stream(self):
        """h265 alias is accepted."""
        info = make_stream_info(codec="h265")
        config = PipelineConfig.from_stream_info(info)
        assert config.width == 1920

    def test_creates_config_from_mjpeg_stream(self):
        """MJPEG codec is accepted for forward-compatibility."""
        info = make_stream_info(codec="mjpeg")
        config = PipelineConfig.from_stream_info(info)
        assert config.width == 1920

    def test_rejects_unsupported_codec(self):
        """Unknown codec raises ValueError."""
        info = make_stream_info(codec="vp9")
        with pytest.raises(ValueError, match="Unsupported codec"):
            PipelineConfig.from_stream_info(info)

    def test_codec_case_insensitive(self):
        """Codec matching is case-insensitive."""
        info = make_stream_info(codec="H264")
        config = PipelineConfig.from_stream_info(info)
        assert config.width == 1920

    def test_maps_720p_dimensions(self):
        """720p StreamInfo maps to 720p config."""
        info = make_stream_info(width=1280, height=720)
        config = PipelineConfig.from_stream_info(info)
        assert config.width == 1280
        assert config.height == 720

    def test_custom_device_name(self):
        """Custom device name is passed through."""
        info = make_stream_info()
        config = PipelineConfig.from_stream_info(info, device_name="TestCam")
        assert config.device_name == "TestCam"

    def test_custom_pixel_format(self):
        """Custom pixel format is passed through."""
        info = make_stream_info()
        config = PipelineConfig.from_stream_info(info, pixel_format="rgb24")
        assert config.pixel_format == "rgb24"

    def test_default_pixel_format_is_bgr24(self):
        """Default pixel format is BGR24 for v1 parity."""
        info = make_stream_info()
        config = PipelineConfig.from_stream_info(info)
        assert config.pixel_format == "bgr24"


# ===========================================================================
# 2. Supported codecs constant
# ===========================================================================

class TestSupportedCodecs:
    """Tests for the SUPPORTED_CODECS set."""

    def test_h264_supported(self):
        assert "h264" in SUPPORTED_CODECS

    def test_hevc_supported(self):
        assert "hevc" in SUPPORTED_CODECS

    def test_h265_supported(self):
        assert "h265" in SUPPORTED_CODECS

    def test_mjpeg_supported(self):
        assert "mjpeg" in SUPPORTED_CODECS

    def test_frozen(self):
        """SUPPORTED_CODECS is immutable."""
        assert isinstance(SUPPORTED_CODECS, frozenset)


# ===========================================================================
# 3. BGR24 color format validation (v1 parity)
# ===========================================================================

class TestBGR24Format:
    """Tests that the pipeline uses BGR24 throughout, matching v1 output."""

    def test_default_config_bgr24(self):
        """Default PipelineConfig uses BGR24 pixel format."""
        config = PipelineConfig()
        assert config.pixel_format == "bgr24"

    def test_decoder_default_bgr24(self):
        """UDPDecoder default pixel format is BGR24."""
        assert UDPDecoder.DEFAULT_PIXEL_FORMAT == "bgr24"

    def test_decoder_created_with_bgr24(self):
        """Pipeline creates decoder with BGR24 format by default."""
        async def _test():
            pipeline = FramePipeline()
            stream_info = make_stream_info()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(stream_info)

            assert pipeline.decoder is not None
            assert pipeline.decoder.pixel_format == "bgr24"

        run_async(_test())

    def test_frame_shape_matches_v1(self):
        """Decoded frames are (1080, 1920, 3) uint8 — same as v1 ffmpeg."""
        frame = make_bgr24_frame()
        assert frame.shape == (1080, 1920, 3)
        assert frame.dtype == np.uint8

    def test_frame_channels_bgr_order(self):
        """BGR24 frame has B=0, G=1, R=2 channel layout."""
        # Create frame with known BGR values
        frame = make_bgr24_frame(width=1, height=1, color=(255, 128, 0))
        assert frame[0, 0, 0] == 255  # Blue
        assert frame[0, 0, 1] == 128  # Green
        assert frame[0, 0, 2] == 0    # Red

    def test_validate_1080p_30fps_accepts_bgr24(self):
        """validate_1080p_30fps accepts BGR24 frames (same shape as RGB24)."""
        frame = make_bgr24_frame()
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["valid"] is True
        assert result["resolution_ok"] is True
        assert result["channels"] == 3


# ===========================================================================
# 4. Pipeline start with config from StreamInfo
# ===========================================================================

class TestPipelineStartFromStreamInfo:
    """Tests pipeline creation using PipelineConfig.from_stream_info()."""

    def test_pipeline_with_stream_config(self):
        """Pipeline created from stream info uses correct port and dimensions."""
        async def _test():
            info = make_stream_info(port=9876, width=1920, height=1080)
            config = PipelineConfig.from_stream_info(info)
            pipeline = FramePipeline(config)

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                result = await pipeline.start(info)

            assert result is True
            assert pipeline.decoder.udp_port == 9876
            assert pipeline.decoder.width == 1920
            assert pipeline.decoder.height == 1080

        run_async(_test())

    def test_pipeline_720p_stream(self):
        """Pipeline handles 720p stream info correctly."""
        async def _test():
            info = make_stream_info(width=1280, height=720)
            config = PipelineConfig.from_stream_info(info)
            pipeline = FramePipeline(config)

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                result = await pipeline.start(info)

            assert result is True
            assert pipeline.decoder.width == 1280
            assert pipeline.decoder.height == 720

        run_async(_test())

    def test_pipeline_hevc_stream(self):
        """Pipeline accepts HEVC codec streams."""
        async def _test():
            info = make_stream_info(codec="hevc")
            config = PipelineConfig.from_stream_info(info)
            pipeline = FramePipeline(config)

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                result = await pipeline.start(info)

            assert result is True
            assert pipeline.state == PipelineState.STREAMING

        run_async(_test())


# ===========================================================================
# 5. Frame delivery with format validation
# ===========================================================================

class TestFrameDeliveryFormat:
    """Tests that decoded frames flow through pipeline with correct format."""

    def test_bgr24_frame_delivered_to_sink(self):
        """BGR24 frame from decoder reaches VirtualCameraSink."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Mock submit_frame to capture the frame
            captured_frames = []
            pipeline._vcam_sink.submit_frame = MagicMock(
                side_effect=lambda f: captured_frames.append(f.copy()) or True,
            )

            # Simulate decoder delivering a BGR24 frame
            frame = make_bgr24_frame(color=(255, 128, 0))
            pipeline._on_decoded_frame(frame)

            assert len(captured_frames) == 1
            delivered = captured_frames[0]
            assert delivered.shape == (1080, 1920, 3)
            assert delivered.dtype == np.uint8
            # Verify BGR channel values preserved
            assert delivered[0, 0, 0] == 255  # B
            assert delivered[0, 0, 1] == 128  # G
            assert delivered[0, 0, 2] == 0    # R

        run_async(_test())

    def test_multiple_frames_delivered_sequentially(self):
        """Multiple frames flow through in order."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            delivered_count = [0]
            pipeline._vcam_sink.submit_frame = MagicMock(
                side_effect=lambda f: (
                    delivered_count.__setitem__(0, delivered_count[0] + 1)
                ) or True,
            )

            # Deliver 10 frames
            for _ in range(10):
                pipeline._on_decoded_frame(make_bgr24_frame())

            assert delivered_count[0] == 10

        run_async(_test())

    def test_no_frame_delivered_when_sink_missing(self):
        """No crash when vcam_sink is None (pipeline stopped)."""
        pipeline = FramePipeline()
        assert pipeline._vcam_sink is None

        # Should not crash
        pipeline._on_decoded_frame(make_bgr24_frame())


# ===========================================================================
# 6. Decoder stats tracking
# ===========================================================================

class TestDecodeStatsTracking:
    """Tests that DecodeStats correctly tracks frame delivery metrics."""

    def test_stats_initial_state(self):
        """Fresh DecodeStats has zero counts."""
        stats = DecodeStats()
        snap = stats.snapshot()
        assert snap["frames_decoded"] == 0
        assert snap["frames_dropped"] == 0
        assert snap["errors"] == 0
        assert snap["decode_fps"] == 0.0
        assert snap["codec"] == ""
        assert snap["last_frame_age"] is None

    def test_stats_record_frame(self):
        """record_frame() increments decoded count."""
        stats = DecodeStats()
        stats.record_frame()
        stats.record_frame()
        snap = stats.snapshot()
        assert snap["frames_decoded"] == 2
        assert snap["last_frame_age"] is not None

    def test_stats_record_error(self):
        """record_error() increments error count."""
        stats = DecodeStats()
        stats.record_error()
        snap = stats.snapshot()
        assert snap["errors"] == 1

    def test_stats_record_drop(self):
        """record_drop() increments dropped count."""
        stats = DecodeStats()
        stats.record_drop()
        snap = stats.snapshot()
        assert snap["frames_dropped"] == 1

    def test_stats_set_codec(self):
        """set_codec() updates codec name."""
        stats = DecodeStats()
        stats.set_codec("h264")
        snap = stats.snapshot()
        assert snap["codec"] == "h264"

    def test_stats_fps_calculation(self):
        """FPS calculated from rolling 1-second window."""
        stats = DecodeStats()
        # Record 30 frames rapidly
        for _ in range(30):
            stats.record_frame()
        snap = stats.snapshot()
        # FPS should be high since all frames in same window
        assert snap["decode_fps"] > 0


# ===========================================================================
# 7. Pipeline stats aggregation
# ===========================================================================

class TestPipelineStatsAggregation:
    """Tests that FramePipeline.get_stats() aggregates decoder + vcam stats."""

    def test_stats_stopped_pipeline(self):
        """Stopped pipeline returns default stats."""
        pipeline = FramePipeline()
        stats = pipeline.get_stats()
        assert stats.state == "STOPPED"
        assert stats.decoder_frames == 0
        assert stats.vcam_frames == 0
        assert stats.is_frozen is False

    def test_stats_running_pipeline(self):
        """Running pipeline returns combined decoder + vcam stats."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Manually record some stats
            pipeline._decoder.stats.record_frame()
            pipeline._decoder.stats.record_frame()

            stats = pipeline.get_stats()
            assert stats.state == "STREAMING"
            assert stats.decoder_frames == 2
            assert stats.uptime_s >= 0.0

        run_async(_test())

    def test_detailed_stats_includes_subsystems(self):
        """get_detailed_stats() includes decoder and vcam subsection."""
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


# ===========================================================================
# 8. FrameMetadata dataclass
# ===========================================================================

class TestFrameMetadata:
    """Tests for the FrameMetadata frozen dataclass."""

    def test_metadata_creation(self):
        """FrameMetadata stores frame provenance correctly."""
        meta = FrameMetadata(
            timestamp=1234.5,
            sequence=42,
            height=1080,
            width=1920,
            channels=3,
            nbytes=1920 * 1080 * 3,
            pixel_format="bgr24",
        )
        assert meta.height == 1080
        assert meta.width == 1920
        assert meta.channels == 3
        assert meta.pixel_format == "bgr24"
        assert meta.nbytes == 6_220_800

    def test_metadata_is_frozen(self):
        """FrameMetadata is immutable."""
        meta = FrameMetadata(
            timestamp=0.0, sequence=0, height=1080, width=1920,
            channels=3, nbytes=0,
        )
        with pytest.raises(AttributeError):
            meta.height = 720  # type: ignore

    def test_metadata_default_pixel_format(self):
        """Default pixel format is BGR24."""
        meta = FrameMetadata(
            timestamp=0.0, sequence=0, height=1080, width=1920,
            channels=3, nbytes=0,
        )
        assert meta.pixel_format == "bgr24"


# ===========================================================================
# 9. Orchestrator pipeline integration
# ===========================================================================

class TestOrchestratorPipelineIntegration:
    """Tests that the orchestrator creates pipelines from stream info correctly."""

    def test_ensure_pipeline_uses_stream_config(self):
        """_ensure_pipeline creates pipeline with config from StreamInfo."""
        async def _test():
            from gomaxwebcam.events import EventBus
            from gomaxwebcam.camera_manager import CameraManager
            from gomaxwebcam.transport_manager import TransportManager
            from gomaxwebcam.orchestrator import AppOrchestrator

            bus = EventBus()
            bus.set_loop(asyncio.get_running_loop())
            cam_mgr = CameraManager(bus)
            tm = TransportManager(bus)
            orch = AppOrchestrator(bus, cam_mgr, tm)

            # Create a mock transport that's streaming
            info = make_stream_info(port=7777, codec="h264")
            transport = MockTransport(stream_info=info)
            transport.force_streaming()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await orch._ensure_pipeline(transport)

            assert orch.pipeline is not None
            assert orch.pipeline.state == PipelineState.STREAMING
            assert orch.pipeline.decoder.udp_port == 7777

        run_async(_test())

    def test_ensure_pipeline_rejects_bad_codec(self):
        """_ensure_pipeline skips start if codec is unsupported."""
        async def _test():
            from gomaxwebcam.events import EventBus
            from gomaxwebcam.camera_manager import CameraManager
            from gomaxwebcam.transport_manager import TransportManager
            from gomaxwebcam.orchestrator import AppOrchestrator

            bus = EventBus()
            bus.set_loop(asyncio.get_running_loop())
            cam_mgr = CameraManager(bus)
            tm = TransportManager(bus)
            orch = AppOrchestrator(bus, cam_mgr, tm)

            info = make_stream_info(codec="av1")  # Not supported
            transport = MockTransport(stream_info=info)
            transport.force_streaming()

            await orch._ensure_pipeline(transport)

            # Pipeline should not be created
            assert orch.pipeline is None

        run_async(_test())

    def test_ensure_pipeline_hot_swaps_decoder_on_switch(self):
        """_ensure_pipeline hot-swaps decoder when pipeline is already running.

        With freeze-frame preservation, _ensure_pipeline uses switch_stream()
        to swap the decoder while keeping the vcam sink alive.  The pipeline
        object stays the same — only the decoder is replaced.
        """
        async def _test():
            from gomaxwebcam.events import EventBus
            from gomaxwebcam.camera_manager import CameraManager
            from gomaxwebcam.transport_manager import TransportManager
            from gomaxwebcam.orchestrator import AppOrchestrator
            from gomaxwebcam.pipeline.frame_pipeline import FramePipeline

            bus = EventBus()
            bus.set_loop(asyncio.get_running_loop())
            cam_mgr = CameraManager(bus)
            tm = TransportManager(bus)
            orch = AppOrchestrator(bus, cam_mgr, tm)

            # Start first pipeline with mocked VCam
            info1 = make_stream_info(port=8001)
            transport1 = MockTransport(stream_info=info1)
            transport1.force_streaming()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "is_running", new_callable=PropertyMock, return_value=True):
                await orch._ensure_pipeline(transport1)

            first_pipeline = orch.pipeline
            assert first_pipeline is not None

            # Ensure vcam sink reports as running for hot-swap path
            mock_sink = MagicMock()
            mock_sink.is_running = True
            first_pipeline._vcam_sink = mock_sink

            # Switch to second transport — pipeline hot-swaps decoder
            info2 = make_stream_info(port=8002)
            transport2 = MockTransport(stream_info=info2)
            transport2.force_streaming()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(UDPDecoder, "stop"):
                await orch._ensure_pipeline(transport2)

            # Same pipeline object (vcam sink preserved)
            assert orch.pipeline is first_pipeline
            # Decoder swapped to new port
            assert orch.pipeline._decoder.udp_port == 8002

        run_async(_test())

    def test_ensure_pipeline_skips_when_not_streaming(self):
        """_ensure_pipeline is a no-op if transport isn't streaming."""
        async def _test():
            from gomaxwebcam.events import EventBus
            from gomaxwebcam.camera_manager import CameraManager
            from gomaxwebcam.transport_manager import TransportManager
            from gomaxwebcam.orchestrator import AppOrchestrator

            bus = EventBus()
            bus.set_loop(asyncio.get_running_loop())
            cam_mgr = CameraManager(bus)
            tm = TransportManager(bus)
            orch = AppOrchestrator(bus, cam_mgr, tm)

            transport = MockTransport()
            # Transport is DISCONNECTED (not streaming)

            await orch._ensure_pipeline(transport)
            assert orch.pipeline is None

        run_async(_test())


# ===========================================================================
# 10. Decoder creation and configuration
# ===========================================================================

class TestDecoderCreation:
    """Tests that UDPDecoder is created with correct parameters."""

    def test_decoder_default_params(self):
        """UDPDecoder defaults match expected values."""
        decoder = UDPDecoder()
        assert decoder.udp_port == 8554
        assert decoder.width == 1920
        assert decoder.height == 1080
        assert decoder.pixel_format == "bgr24"
        assert decoder.is_running is False

    def test_decoder_custom_params(self):
        """UDPDecoder accepts custom port and dimensions."""
        decoder = UDPDecoder(
            udp_port=9999,
            width=1280,
            height=720,
            pixel_format="rgb24",
        )
        assert decoder.udp_port == 9999
        assert decoder.width == 1280
        assert decoder.height == 720
        assert decoder.pixel_format == "rgb24"

    def test_decoder_callbacks_stored(self):
        """Callbacks are stored and accessible."""
        on_frame = MagicMock()
        on_error = MagicMock()
        on_stopped = MagicMock()

        decoder = UDPDecoder(
            on_frame=on_frame,
            on_error=on_error,
            on_stopped=on_stopped,
        )

        assert decoder._on_frame is on_frame
        assert decoder._on_error is on_error
        assert decoder._on_stopped is on_stopped

    def test_decoder_stats_independent(self):
        """Each decoder has its own stats instance."""
        d1 = UDPDecoder()
        d2 = UDPDecoder()
        d1.stats.record_frame()
        assert d1.stats.frames_decoded == 1
        assert d2.stats.frames_decoded == 0


# ===========================================================================
# 11. Freeze-frame during codec transitions
# ===========================================================================

class TestFreezeFrameDuringTransitions:
    """Tests freeze-frame behavior during stream loss and transport switch."""

    def test_decoder_error_triggers_freeze_with_callback(self):
        """Decoder error fires on_freeze callback and holds last frame."""
        async def _test():
            pipeline = FramePipeline()
            freeze_events = []
            pipeline.on_freeze = lambda: freeze_events.append("freeze")

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Simulate decoder error
            pipeline._on_decoder_error(RuntimeError("UDP timeout"))

            assert pipeline.state == PipelineState.FREEZE_FRAME
            assert pipeline.is_frozen
            assert len(freeze_events) == 1

        run_async(_test())

    def test_new_frame_after_freeze_resumes_streaming(self):
        """A new frame after freeze transitions back to STREAMING."""
        async def _test():
            pipeline = FramePipeline()
            unfreeze_events = []
            pipeline.on_unfreeze = lambda: unfreeze_events.append("unfreeze")

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Enter freeze
            pipeline._on_decoder_error(RuntimeError("Timeout"))
            assert pipeline.is_frozen

            # Simulate new frame arriving (e.g., after transport switch)
            pipeline._vcam_sink.submit_frame = MagicMock(return_value=True)
            pipeline._on_decoded_frame(make_bgr24_frame())

            assert pipeline.state == PipelineState.STREAMING
            assert not pipeline.is_frozen
            assert len(unfreeze_events) == 1

        run_async(_test())

    def test_decoder_stopped_during_shutdown_no_freeze(self):
        """Decoder stopped event during STOPPING doesn't trigger freeze."""
        async def _test():
            pipeline = FramePipeline()
            freeze_events = []
            pipeline.on_freeze = lambda: freeze_events.append("freeze")

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            # Set state to STOPPING (simulating stop() in progress)
            pipeline._set_state(PipelineState.STOPPING)

            # Decoder stopped callback should be ignored
            pipeline._on_decoder_stopped()
            assert len(freeze_events) == 0

        run_async(_test())


# ===========================================================================
# 12. End-to-end frame format chain
# ===========================================================================

class TestEndToEndFrameFormat:
    """Integration tests verifying the full frame format chain."""

    def test_full_chain_1080p_bgr24(self):
        """1080p BGR24 frame passes through pipeline and validation."""
        frame = make_bgr24_frame()
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["valid"] is True
        assert result["resolution_ok"] is True
        assert result["fps_ok"] is True
        assert result["dtype_ok"] is True
        assert result["width"] == 1920
        assert result["height"] == 1080
        assert result["channels"] == 3

    def test_720p_frame_fails_1080p_validation(self):
        """720p frame fails 1080p validation (would need scaling)."""
        frame = make_bgr24_frame(width=1280, height=720)
        result = validate_1080p_30fps(frame, fps=30.0)
        assert result["resolution_ok"] is False
        assert result["width"] == 1280
        assert result["height"] == 720

    def test_pipeline_start_stop_cleanup(self):
        """Pipeline start -> deliver frames -> stop cleans up everything."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True):
                await pipeline.start(make_stream_info())

            assert pipeline.state == PipelineState.STREAMING
            assert pipeline.decoder is not None
            assert pipeline.vcam_sink is not None

            # Deliver some frames
            pipeline._vcam_sink.submit_frame = MagicMock(return_value=True)
            for _ in range(5):
                pipeline._on_decoded_frame(make_bgr24_frame())

            with patch.object(UDPDecoder, "stop"), \
                 patch.object(VirtualCameraSink, "stop"):
                await pipeline.stop()

            assert pipeline.state == PipelineState.STOPPED
            assert pipeline.decoder is None
            assert pipeline.vcam_sink is None

        run_async(_test())

    def test_pipeline_restart_with_different_port(self):
        """Pipeline restart changes decoder port while maintaining format."""
        async def _test():
            pipeline = FramePipeline()

            with patch.object(UDPDecoder, "start", return_value=True), \
                 patch.object(VirtualCameraSink, "start", return_value=True), \
                 patch.object(UDPDecoder, "stop"), \
                 patch.object(VirtualCameraSink, "stop"):

                await pipeline.start(make_stream_info(port=8554))
                assert pipeline.decoder.udp_port == 8554

                await pipeline.restart(make_stream_info(port=9000))
                assert pipeline.decoder.udp_port == 9000
                assert pipeline.decoder.pixel_format == "bgr24"

        run_async(_test())
