"""
test_udp_decoder.py — Unit tests for the PyAV-based UDP decode pipeline

All tests mock PyAV to run without a real GoPro or UDP stream.
Tests verify:
  - Frame decoding and RGB24 conversion
  - Callback delivery (on_frame, on_error, on_stopped)
  - Graceful stop via threading.Event
  - Error tolerance (transient errors tolerated, persistent errors fatal)
  - Frame scaling when dimensions don't match
  - Statistics tracking
  - Thread lifecycle (start/stop/restart)
"""

import threading
import time
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# Mark all tests as no_gopro_needed since we mock everything
pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Helpers for mocking PyAV
# ---------------------------------------------------------------------------

class FakeVideoFrame:
    """Mock av.VideoFrame that produces numpy arrays."""

    def __init__(self, width=1920, height=1080):
        self._width = width
        self._height = height

    def to_ndarray(self, format="rgb24"):
        """Return a fake RGB24 frame."""
        return np.zeros((self._height, self._width, 3), dtype=np.uint8)

    def reformat(self, width=None, height=None, format="rgb24"):
        """Return a resized fake frame."""
        w = width or self._width
        h = height or self._height
        return FakeVideoFrame(w, h)


class FakeVideoStream:
    """Mock av video stream."""

    def __init__(self, codec_name="h264", width=1920, height=1080):
        self.type = "video"
        self.thread_type = "AUTO"
        self.thread_count = 0
        self.codec_context = MagicMock()
        self.codec_context.name = codec_name
        self.codec_context.width = width
        self.codec_context.height = height


class FakePacket:
    """Mock av packet that yields frames on decode."""

    def __init__(self, frames=None):
        self._frames = frames or [FakeVideoFrame()]

    def decode(self):
        return self._frames


class FakeContainer:
    """Mock av container that yields packets."""

    def __init__(self, packets=None, streams=None, num_packets=5):
        self._streams = streams or [FakeVideoStream()]
        self.streams = self._streams
        if packets is not None:
            self._packets = packets
        else:
            self._packets = [FakePacket() for _ in range(num_packets)]
        self._closed = False

    def demux(self, stream):
        for pkt in self._packets:
            yield pkt

    def close(self):
        self._closed = True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestUDPDecoderInit:
    """Test decoder initialization."""

    def test_default_values(self):
        from gomaxwebcam.pipeline.decode import UDPDecoder

        decoder = UDPDecoder()
        assert decoder.udp_port == 8554
        assert decoder.width == 1920
        assert decoder.height == 1080
        assert not decoder.is_running

    def test_custom_values(self):
        from gomaxwebcam.pipeline.decode import UDPDecoder

        decoder = UDPDecoder(udp_port=9000, width=1280, height=720)
        assert decoder.udp_port == 9000
        assert decoder.width == 1280
        assert decoder.height == 720


class TestUDPDecoderFrameDelivery:
    """Test that decoded frames are delivered via callback."""

    def test_frames_delivered_to_callback(self):
        """Frames from PyAV should be delivered as numpy arrays via on_frame."""
        from gomaxwebcam.pipeline.decode import UDPDecoder

        received_frames = []
        stopped = threading.Event()

        container = FakeContainer(num_packets=3)

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = container
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(
                on_frame=lambda f: received_frames.append(f.copy()),
                on_stopped=stopped.set,
            )
            decoder.start()

            # Wait for decode loop to complete (finite packets)
            stopped.wait(timeout=5.0)

        assert len(received_frames) == 3
        for frame in received_frames:
            assert isinstance(frame, np.ndarray)
            assert frame.shape == (1080, 1920, 3)
            assert frame.dtype == np.uint8

    def test_frame_callback_receives_rgb24(self):
        """Frames should be in RGB24 format (height, width, 3)."""
        from gomaxwebcam.pipeline.decode import UDPDecoder

        shapes = []
        stopped = threading.Event()

        container = FakeContainer(num_packets=1)

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = container
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(
                on_frame=lambda f: shapes.append(f.shape),
                on_stopped=stopped.set,
            )
            decoder.start()
            stopped.wait(timeout=5.0)

        assert shapes == [(1080, 1920, 3)]


class TestUDPDecoderStats:
    """Test decode statistics tracking."""

    def test_stats_count_frames(self):
        from gomaxwebcam.pipeline.decode import UDPDecoder

        stopped = threading.Event()
        container = FakeContainer(num_packets=5)

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = container
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(on_stopped=stopped.set)
            decoder.start()
            stopped.wait(timeout=5.0)

        stats = decoder.stats.snapshot()
        assert stats["frames_decoded"] == 5
        assert stats["errors"] == 0
        assert stats["codec"] == "h264"

    def test_stats_snapshot_keys(self):
        from gomaxwebcam.pipeline.decode import DecodeStats

        stats = DecodeStats()
        snap = stats.snapshot()
        expected_keys = {
            "frames_decoded", "frames_dropped", "errors",
            "decode_fps", "codec", "last_frame_age",
        }
        assert set(snap.keys()) == expected_keys

    def test_stats_fps_calculation(self):
        from gomaxwebcam.pipeline.decode import DecodeStats

        stats = DecodeStats()
        # Record several frames in quick succession
        for _ in range(10):
            stats.record_frame()
        snap = stats.snapshot()
        assert snap["decode_fps"] > 0
        assert snap["frames_decoded"] == 10


class TestUDPDecoderStopBehavior:
    """Test clean shutdown."""

    def test_stop_terminates_thread(self):
        from gomaxwebcam.pipeline.decode import UDPDecoder

        # Create a container that yields packets slowly
        class SlowContainer(FakeContainer):
            def __init__(self, stop_event):
                super().__init__()
                self._stop = stop_event

            def demux(self, stream):
                while not self._stop.is_set():
                    yield FakePacket()
                    time.sleep(0.01)

        stop_evt = threading.Event()
        container = SlowContainer(stop_evt)

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = container
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder()
            decoder.start()
            time.sleep(0.1)  # Let it decode a few frames
            assert decoder.is_running

            stop_evt.set()  # Also stop the slow container
            decoder.stop(timeout=3.0)
            assert not decoder.is_running

    def test_on_stopped_callback_called(self):
        from gomaxwebcam.pipeline.decode import UDPDecoder

        stopped_called = threading.Event()
        container = FakeContainer(num_packets=1)

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = container
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(on_stopped=stopped_called.set)
            decoder.start()
            assert stopped_called.wait(timeout=5.0)


class TestUDPDecoderErrorHandling:
    """Test error tolerance and fatal error reporting."""

    def test_transient_errors_tolerated(self):
        """A few InvalidDataErrors should not stop decoding."""
        from gomaxwebcam.pipeline.decode import UDPDecoder

        stopped = threading.Event()
        frame_count = []

        class ErrorThenGoodPacket:
            def __init__(self, error_class, fail=False):
                self.fail = fail
                self.error_class = error_class

            def decode(self):
                if self.fail:
                    raise self.error_class("corrupted")
                return [FakeVideoFrame()]

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            InvalidData = type("InvalidDataError", (Exception,), {})
            av_mock.error.InvalidDataError = InvalidData
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            # 2 errors followed by 3 good packets
            packets = [
                ErrorThenGoodPacket(InvalidData, fail=True),
                ErrorThenGoodPacket(InvalidData, fail=True),
                ErrorThenGoodPacket(InvalidData, fail=False),
                ErrorThenGoodPacket(InvalidData, fail=False),
                ErrorThenGoodPacket(InvalidData, fail=False),
            ]
            container = FakeContainer(packets=packets)
            av_mock.open.return_value = container

            decoder = UDPDecoder(
                on_frame=lambda f: frame_count.append(1),
                on_stopped=stopped.set,
            )
            decoder.start()
            stopped.wait(timeout=5.0)

        assert len(frame_count) == 3  # 3 good frames
        assert decoder.stats.errors == 2

    def test_fatal_error_triggers_callback(self):
        """on_error should be called when the stream can't be opened."""
        from gomaxwebcam.pipeline.decode import UDPDecoder

        errors = []
        stopped = threading.Event()

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.side_effect = RuntimeError("Connection refused")
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(
                on_error=lambda e: errors.append(e),
                on_stopped=stopped.set,
            )
            decoder.start()
            stopped.wait(timeout=5.0)

        assert len(errors) == 1
        assert "Connection refused" in str(errors[0])

    def test_no_video_stream_error(self):
        """Should error when container has no video streams."""
        from gomaxwebcam.pipeline.decode import UDPDecoder

        errors = []
        stopped = threading.Event()

        # Container with only audio stream
        audio_stream = MagicMock()
        audio_stream.type = "audio"
        container = FakeContainer(streams=[audio_stream])

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = container
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(
                on_error=lambda e: errors.append(e),
                on_stopped=stopped.set,
            )
            decoder.start()
            stopped.wait(timeout=5.0)

        assert len(errors) == 1
        assert "No video stream" in str(errors[0])

    def test_pyav_import_error(self):
        """Should handle missing PyAV gracefully."""
        from gomaxwebcam.pipeline.decode import UDPDecoder

        errors = []
        stopped = threading.Event()

        # Simulate PyAV not installed
        with patch.dict("sys.modules", {"av": None}):
            decoder = UDPDecoder(
                on_error=lambda e: errors.append(e),
                on_stopped=stopped.set,
            )
            decoder.start()
            stopped.wait(timeout=5.0)

        assert len(errors) == 1
        assert isinstance(errors[0], ImportError)


class TestUDPDecoderFrameScaling:
    """Test frame scaling when source dimensions differ from target."""

    def test_mismatched_frame_gets_scaled(self):
        """Frames with wrong dimensions should be scaled via reformat."""
        from gomaxwebcam.pipeline.decode import UDPDecoder

        received_frames = []
        stopped = threading.Event()

        # Source frame is 1280x720 but decoder expects 1920x1080
        class SmallFrame(FakeVideoFrame):
            def __init__(self):
                super().__init__(width=1280, height=720)

            def reformat(self, width=None, height=None, format="rgb24"):
                return FakeVideoFrame(width, height)

        class SmallPacket:
            def decode(self):
                return [SmallFrame()]

        container = FakeContainer(packets=[SmallPacket()])

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = container
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(
                width=1920,
                height=1080,
                on_frame=lambda f: received_frames.append(f.shape),
                on_stopped=stopped.set,
            )
            decoder.start()
            stopped.wait(timeout=5.0)

        assert len(received_frames) == 1
        assert received_frames[0] == (1080, 1920, 3)


class TestUDPDecoderLifecycle:
    """Test start/stop/restart lifecycle."""

    def test_restart_after_stop(self):
        from gomaxwebcam.pipeline.decode import UDPDecoder

        stopped = threading.Event()

        def make_container():
            return FakeContainer(num_packets=2)

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = make_container()
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(on_stopped=stopped.set)

            # First run
            decoder.start()
            stopped.wait(timeout=5.0)
            decoder.stop()
            first_count = decoder.stats.frames_decoded

            # Second run
            stopped.clear()
            av_mock.open.return_value = make_container()
            decoder.start()
            stopped.wait(timeout=5.0)
            decoder.stop()

        # Stats accumulate across restarts
        assert decoder.stats.frames_decoded == first_count + 2

    def test_double_start_is_safe(self):
        from gomaxwebcam.pipeline.decode import UDPDecoder

        stopped = threading.Event()
        container = FakeContainer(num_packets=1)

        with patch.dict("sys.modules", {"av": MagicMock()}):
            import sys
            av_mock = sys.modules["av"]
            av_mock.open.return_value = container
            av_mock.error.InvalidDataError = type("InvalidDataError", (Exception,), {})
            av_mock.error.EOFError = type("EOFError", (Exception,), {})
            av_mock.AVError = type("AVError", (Exception,), {})

            decoder = UDPDecoder(on_stopped=stopped.set)
            decoder.start()
            result = decoder.start()  # Should return True (already running)
            assert result is True

            stopped.wait(timeout=5.0)
            decoder.stop()

    def test_stop_without_start_is_safe(self):
        from gomaxwebcam.pipeline.decode import UDPDecoder

        decoder = UDPDecoder()
        decoder.stop()  # Should not raise
        assert not decoder.is_running


class TestDecodeStats:
    """Test DecodeStats independently."""

    def test_initial_state(self):
        from gomaxwebcam.pipeline.decode import DecodeStats

        stats = DecodeStats()
        snap = stats.snapshot()
        assert snap["frames_decoded"] == 0
        assert snap["frames_dropped"] == 0
        assert snap["errors"] == 0
        assert snap["decode_fps"] == 0.0
        assert snap["codec"] == ""
        assert snap["last_frame_age"] is None

    def test_record_frame_updates(self):
        from gomaxwebcam.pipeline.decode import DecodeStats

        stats = DecodeStats()
        stats.record_frame()
        stats.record_frame()

        snap = stats.snapshot()
        assert snap["frames_decoded"] == 2
        assert snap["last_frame_age"] is not None
        assert snap["last_frame_age"] < 1.0

    def test_record_error_updates(self):
        from gomaxwebcam.pipeline.decode import DecodeStats

        stats = DecodeStats()
        stats.record_error()
        stats.record_error()
        stats.record_error()

        assert stats.snapshot()["errors"] == 3

    def test_record_drop_updates(self):
        from gomaxwebcam.pipeline.decode import DecodeStats

        stats = DecodeStats()
        stats.record_drop()

        assert stats.snapshot()["frames_dropped"] == 1

    def test_set_codec(self):
        from gomaxwebcam.pipeline.decode import DecodeStats

        stats = DecodeStats()
        stats.set_codec("hevc")
        assert stats.snapshot()["codec"] == "hevc"
