"""
tests/v2/test_pipeline_metrics.py — Frame metrics, health checks, and dashboard reporting.

Tests the enhanced DecodeStats (latency, jitter, drop rate), PipelineHealth
assessment against v1 baselines, and dashboard endpoint integration.

All tests use mocks — no real hardware, PyAV, or pyvirtualcam needed.
"""

import asyncio
import sys
import os
import time
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from gomaxwebcam.pipeline.decode import DecodeStats
from gomaxwebcam.pipeline.frame_pipeline import (
    FramePipeline,
    HealthStatus,
    PipelineConfig,
    PipelineHealth,
    PipelineState,
    PipelineStats,
)
from gomaxwebcam.events import pipeline_metrics_event, EventType


def run_async(coro):
    """Helper to run async tests without pytest-asyncio plugin."""
    return asyncio.run(coro)


# =====================================================================
# DecodeStats latency tracking
# =====================================================================


class TestDecodeStatsLatency:
    """Tests for DecodeStats latency, jitter, and drop rate tracking."""

    def test_begin_decode_then_record_frame_captures_latency(self):
        """begin_decode() + record_frame() should measure decode time."""
        stats = DecodeStats()
        stats.begin_decode()
        # Simulate some decode work
        time.sleep(0.005)  # 5ms
        stats.record_frame()

        snap = stats.snapshot()
        assert snap["last_decode_ms"] > 0
        assert snap["avg_decode_ms"] > 0
        assert snap["last_decode_ms"] >= 4.0  # At least ~4ms (sleep not exact)

    def test_record_frame_without_begin_decode_no_crash(self):
        """record_frame() without prior begin_decode() should not crash."""
        stats = DecodeStats()
        stats.record_frame()

        snap = stats.snapshot()
        assert snap["last_decode_ms"] == 0.0
        assert snap["frames_decoded"] == 1

    def test_latency_rolling_average(self):
        """Multiple frames should produce a rolling average."""
        stats = DecodeStats()
        for _ in range(5):
            stats.begin_decode()
            time.sleep(0.002)
            stats.record_frame()

        snap = stats.snapshot()
        assert snap["avg_decode_ms"] > 0
        assert snap["frames_decoded"] == 5

    def test_max_decode_ms_tracks_worst_case(self):
        """max_decode_ms should track the worst frame."""
        stats = DecodeStats()

        # Fast frame
        stats.begin_decode()
        time.sleep(0.001)
        stats.record_frame()

        # Slow frame
        stats.begin_decode()
        time.sleep(0.010)
        stats.record_frame()

        snap = stats.snapshot()
        assert snap["max_decode_ms"] >= 8.0  # Slow frame should be max

    def test_jitter_tracking(self):
        """Inter-frame jitter should be measured against 33.3ms ideal."""
        stats = DecodeStats()

        # Record frames at ~33ms intervals (close to ideal)
        for _ in range(5):
            stats.begin_decode()
            stats.record_frame()
            time.sleep(0.033)

        snap = stats.snapshot()
        # Jitter should be small since we're close to 33ms intervals
        assert snap["avg_jitter_ms"] >= 0
        assert "avg_jitter_ms" in snap

    def test_drop_rate_calculation(self):
        """Drop rate should be frames_dropped / total."""
        stats = DecodeStats()

        # 8 good frames, 2 drops = 20% drop rate
        for _ in range(8):
            stats.record_frame()
        for _ in range(2):
            stats.record_drop()

        snap = stats.snapshot()
        assert abs(snap["drop_rate"] - 0.2) < 0.01

    def test_drop_rate_zero_when_no_drops(self):
        """Drop rate should be 0 when no frames dropped."""
        stats = DecodeStats()
        for _ in range(10):
            stats.record_frame()

        snap = stats.snapshot()
        assert snap["drop_rate"] == 0.0

    def test_drop_rate_zero_when_no_frames(self):
        """Drop rate should be 0 when no frames at all."""
        stats = DecodeStats()
        snap = stats.snapshot()
        assert snap["drop_rate"] == 0.0

    def test_snapshot_contains_all_new_fields(self):
        """Snapshot should include all new metric fields."""
        stats = DecodeStats()
        stats.begin_decode()
        stats.record_frame()

        snap = stats.snapshot()
        expected_keys = {
            "frames_decoded", "frames_dropped", "errors",
            "decode_fps", "codec", "last_frame_age",
            "last_decode_ms", "avg_decode_ms", "max_decode_ms",
            "avg_jitter_ms", "drop_rate",
        }
        assert expected_keys.issubset(set(snap.keys()))

    def test_baselines_are_defined(self):
        """v1 performance baselines should be defined as class constants."""
        assert DecodeStats.BASELINE_FPS_TARGET == 30.0
        assert DecodeStats.BASELINE_FPS_MIN == 25.0
        assert DecodeStats.BASELINE_LATENCY_WARN_MS == 50.0
        assert DecodeStats.BASELINE_LATENCY_CRIT_MS == 100.0
        assert DecodeStats.BASELINE_DROP_RATE_WARN == 0.01
        assert DecodeStats.BASELINE_DROP_RATE_CRIT == 0.05


# =====================================================================
# PipelineHealth assessment
# =====================================================================


class TestPipelineHealth:
    """Tests for FramePipeline health assessment against v1 baselines."""

    def _make_pipeline_with_stats(
        self,
        decoder_fps=30.0,
        avg_decode_ms=5.0,
        drop_rate=0.0,
        frames_decoded=100,
        frames_dropped=0,
        state=PipelineState.STREAMING,
    ) -> FramePipeline:
        """Create a pipeline with mocked decoder/vcam stats."""
        pipeline = FramePipeline(PipelineConfig())
        pipeline._state = state
        pipeline._start_time = time.monotonic() - 10.0

        # Mock decoder
        mock_decoder = MagicMock()
        mock_decoder.stats = MagicMock()
        mock_decoder.stats.snapshot.return_value = {
            "frames_decoded": frames_decoded,
            "frames_dropped": frames_dropped,
            "errors": 0,
            "decode_fps": decoder_fps,
            "codec": "h264",
            "last_frame_age": 0.01,
            "last_decode_ms": avg_decode_ms,
            "avg_decode_ms": avg_decode_ms,
            "max_decode_ms": avg_decode_ms * 1.5,
            "avg_jitter_ms": 2.0,
            "drop_rate": drop_rate,
        }
        pipeline._decoder = mock_decoder

        # Mock vcam
        mock_vcam = MagicMock()
        mock_vcam.frames_sent = frames_decoded
        mock_vcam.freeze_frames_sent = 0
        mock_vcam.fps_actual = decoder_fps
        mock_vcam.get_stats.return_value = {"queue_size": 0}
        pipeline._vcam_sink = mock_vcam

        return pipeline

    def test_healthy_pipeline(self):
        """Pipeline with good metrics should be HEALTHY."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=30.0, avg_decode_ms=5.0, drop_rate=0.0,
        )
        health = pipeline.get_health()
        assert health.status == "healthy"
        assert health.fps_ok is True
        assert health.latency_ok is True
        assert health.drop_rate_ok is True

    def test_degraded_fps(self):
        """Pipeline with low FPS should be DEGRADED."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=20.0, avg_decode_ms=5.0, drop_rate=0.0,
        )
        health = pipeline.get_health()
        assert health.status == "degraded"
        assert health.fps_ok is False

    def test_unhealthy_fps(self):
        """Pipeline with critically low FPS should be UNHEALTHY."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=10.0, avg_decode_ms=5.0, drop_rate=0.0,
        )
        health = pipeline.get_health()
        assert health.status == "unhealthy"
        assert health.fps_ok is False

    def test_degraded_latency(self):
        """Pipeline with high latency should be DEGRADED."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=30.0, avg_decode_ms=60.0, drop_rate=0.0,
        )
        health = pipeline.get_health()
        assert health.status == "degraded"
        assert health.latency_ok is False

    def test_unhealthy_latency(self):
        """Pipeline with critically high latency should be UNHEALTHY."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=30.0, avg_decode_ms=120.0, drop_rate=0.0,
        )
        health = pipeline.get_health()
        assert health.status == "unhealthy"
        assert health.latency_ok is False

    def test_degraded_drop_rate(self):
        """Pipeline with elevated drop rate should be DEGRADED."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=30.0, avg_decode_ms=5.0, drop_rate=0.02,
        )
        health = pipeline.get_health()
        assert health.status == "degraded"
        assert health.drop_rate_ok is False

    def test_unhealthy_drop_rate(self):
        """Pipeline with critical drop rate should be UNHEALTHY."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=30.0, avg_decode_ms=5.0, drop_rate=0.10,
        )
        health = pipeline.get_health()
        assert health.status == "unhealthy"
        assert health.drop_rate_ok is False

    def test_unknown_when_stopped(self):
        """Stopped pipeline should report UNKNOWN health."""
        pipeline = self._make_pipeline_with_stats(state=PipelineState.STOPPED)
        health = pipeline.get_health()
        assert health.status == "unknown"

    def test_unknown_when_insufficient_data(self):
        """Pipeline with < 10 frames should report UNKNOWN health."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=30.0, frames_decoded=5,
        )
        health = pipeline.get_health()
        assert health.status == "unknown"

    def test_worst_issue_wins(self):
        """Multiple issues should report the worst status."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=20.0,       # degraded
            avg_decode_ms=120.0,    # unhealthy
            drop_rate=0.0,          # ok
        )
        health = pipeline.get_health()
        assert health.status == "unhealthy"  # Worst wins

    def test_health_details_populated(self):
        """Health details should describe the issues found."""
        pipeline = self._make_pipeline_with_stats(
            decoder_fps=20.0, avg_decode_ms=5.0, drop_rate=0.0,
        )
        health = pipeline.get_health()
        assert "FPS" in health.details
        assert len(health.details) > 0

    def test_healthy_details_message(self):
        """Healthy pipeline should have a positive details message."""
        pipeline = self._make_pipeline_with_stats()
        health = pipeline.get_health()
        assert "v1 baselines" in health.details.lower() or "within" in health.details.lower()


# =====================================================================
# PipelineStats enhanced fields
# =====================================================================


class TestPipelineStatsEnhanced:
    """Tests for enhanced PipelineStats with latency and health fields."""

    def test_stats_include_latency_fields(self):
        """PipelineStats should include latency and timing fields."""
        stats = PipelineStats()
        assert hasattr(stats, "last_decode_ms")
        assert hasattr(stats, "avg_decode_ms")
        assert hasattr(stats, "max_decode_ms")
        assert hasattr(stats, "avg_jitter_ms")
        assert hasattr(stats, "drop_rate")
        assert hasattr(stats, "health")
        assert hasattr(stats, "decoder_errors")

    def test_stats_defaults(self):
        """Default PipelineStats should have zero metrics."""
        stats = PipelineStats()
        assert stats.last_decode_ms == 0.0
        assert stats.avg_decode_ms == 0.0
        assert stats.avg_jitter_ms == 0.0
        assert stats.drop_rate == 0.0
        assert stats.health == "unknown"
        assert stats.decoder_errors == 0

    def test_get_stats_populates_latency(self):
        """get_stats() should populate latency fields from decoder."""
        pipeline = FramePipeline(PipelineConfig())
        pipeline._state = PipelineState.STREAMING
        pipeline._start_time = time.monotonic()

        mock_decoder = MagicMock()
        mock_decoder.stats.snapshot.return_value = {
            "frames_decoded": 100,
            "frames_dropped": 2,
            "errors": 1,
            "decode_fps": 29.5,
            "codec": "h264",
            "last_frame_age": 0.01,
            "last_decode_ms": 4.2,
            "avg_decode_ms": 3.8,
            "max_decode_ms": 8.1,
            "avg_jitter_ms": 1.5,
            "drop_rate": 0.02,
        }
        pipeline._decoder = mock_decoder

        mock_vcam = MagicMock()
        mock_vcam.frames_sent = 100
        mock_vcam.freeze_frames_sent = 0
        mock_vcam.fps_actual = 29.5
        mock_vcam.get_stats.return_value = {"queue_size": 0}
        pipeline._vcam_sink = mock_vcam

        stats = pipeline.get_stats()
        assert stats.last_decode_ms == 4.2
        assert stats.avg_decode_ms == 3.8
        assert stats.max_decode_ms == 8.1
        assert stats.avg_jitter_ms == 1.5
        assert stats.drop_rate == 0.02
        assert stats.decoder_errors == 1
        assert stats.health in ("healthy", "degraded", "unhealthy", "unknown")


# =====================================================================
# Detailed stats includes health
# =====================================================================


class TestDetailedStatsHealth:
    """Tests for get_detailed_stats() including health assessment."""

    def test_detailed_stats_includes_health(self):
        """get_detailed_stats() should include a health section."""
        pipeline = FramePipeline(PipelineConfig())
        pipeline._state = PipelineState.STREAMING
        pipeline._start_time = time.monotonic()

        mock_decoder = MagicMock()
        mock_decoder.stats.snapshot.return_value = {
            "frames_decoded": 100,
            "frames_dropped": 0,
            "errors": 0,
            "decode_fps": 30.0,
            "codec": "h264",
            "last_frame_age": 0.01,
            "last_decode_ms": 3.0,
            "avg_decode_ms": 3.0,
            "max_decode_ms": 5.0,
            "avg_jitter_ms": 1.0,
            "drop_rate": 0.0,
        }
        pipeline._decoder = mock_decoder

        mock_vcam = MagicMock()
        mock_vcam.frames_sent = 100
        mock_vcam.freeze_frames_sent = 0
        mock_vcam.fps_actual = 30.0
        mock_vcam.get_stats.return_value = {"queue_size": 0}
        pipeline._vcam_sink = mock_vcam

        detailed = pipeline.get_detailed_stats()
        assert "health" in detailed
        assert "status" in detailed["health"]
        assert detailed["health"]["status"] in ("healthy", "degraded", "unhealthy", "unknown")


# =====================================================================
# CameraStatus enhanced fields
# =====================================================================


class TestCameraStatusMetrics:
    """Tests for CameraStatus with frame metrics fields."""

    def test_camera_status_has_metric_fields(self):
        """CameraStatus should include the new frame metric fields."""
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus()
        assert hasattr(status, "last_decode_ms")
        assert hasattr(status, "avg_decode_ms")
        assert hasattr(status, "avg_jitter_ms")
        assert hasattr(status, "drop_rate")
        assert hasattr(status, "pipeline_health")

    def test_camera_status_defaults(self):
        """Default CameraStatus should have zero metrics."""
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus()
        assert status.last_decode_ms == 0.0
        assert status.avg_decode_ms == 0.0
        assert status.avg_jitter_ms == 0.0
        assert status.drop_rate == 0.0
        assert status.pipeline_health == "unknown"

    def test_camera_status_serializable(self):
        """CameraStatus with new fields should be JSON-serializable."""
        from gomaxwebcam.dashboard.status_tracker import CameraStatus
        import json

        status = CameraStatus(
            last_decode_ms=4.2,
            avg_decode_ms=3.8,
            avg_jitter_ms=1.5,
            drop_rate=0.02,
            pipeline_health="healthy",
        )
        data = json.loads(status.to_sse_data())
        assert data["last_decode_ms"] == 4.2
        assert data["avg_decode_ms"] == 3.8
        assert data["pipeline_health"] == "healthy"


# =====================================================================
# StatusTracker collects new metrics
# =====================================================================


class TestStatusTrackerMetrics:
    """Tests for CameraStatusTracker collecting frame metrics from pipeline."""

    def test_collect_status_includes_metrics(self):
        """_collect_status() should populate frame metrics from pipeline."""
        from gomaxwebcam.dashboard.status_tracker import CameraStatusTracker

        tracker = CameraStatusTracker()

        mock_pipeline = MagicMock()
        mock_pipeline.state = PipelineState.STREAMING
        mock_pipeline.is_frozen = False

        mock_stats = PipelineStats(
            state="STREAMING",
            decoder_fps=29.5,
            vcam_fps=29.5,
            decoder_frames=100,
            vcam_frames=100,
            uptime_s=10.0,
            last_decode_ms=4.2,
            avg_decode_ms=3.8,
            avg_jitter_ms=1.5,
            drop_rate=0.02,
            health="healthy",
        )
        mock_pipeline.get_stats.return_value = mock_stats

        tracker.set_pipeline(mock_pipeline)
        status = tracker._collect_status()

        assert status.last_decode_ms == 4.2
        assert status.avg_decode_ms == 3.8
        assert status.avg_jitter_ms == 1.5
        assert status.drop_rate == 0.02
        assert status.pipeline_health == "healthy"


# =====================================================================
# Pipeline metrics event
# =====================================================================


class TestPipelineMetricsEvent:
    """Tests for the pipeline_metrics_event convenience constructor."""

    def test_creates_pipeline_event(self):
        """pipeline_metrics_event should create a PIPELINE event."""
        event = pipeline_metrics_event(
            fps=29.5,
            decode_ms=4.2,
            drop_rate=0.02,
            jitter_ms=1.5,
            health="healthy",
            frames_decoded=100,
            frames_dropped=2,
        )
        assert event.type == EventType.PIPELINE
        assert event.data["metrics"] is True
        assert event.data["fps"] == 29.5
        assert event.data["decode_ms"] == 4.2
        assert event.data["health"] == "healthy"


# =====================================================================
# HealthStatus enum
# =====================================================================


class TestHealthStatusEnum:
    """Tests for the HealthStatus enum values."""

    def test_health_values(self):
        """HealthStatus should have the expected values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_all_statuses_are_strings(self):
        """All HealthStatus values should be strings for JSON serialization."""
        for status in HealthStatus:
            assert isinstance(status.value, str)


# =====================================================================
# PipelineHealth dataclass
# =====================================================================


class TestPipelineHealthDataclass:
    """Tests for the PipelineHealth dataclass."""

    def test_defaults(self):
        """Default PipelineHealth should be unknown with OK flags."""
        health = PipelineHealth()
        assert health.status == "unknown"
        assert health.fps_ok is True
        assert health.latency_ok is True
        assert health.drop_rate_ok is True

    def test_serializable(self):
        """PipelineHealth should be serializable via asdict."""
        health = PipelineHealth(
            status="healthy",
            decode_fps=30.0,
            avg_decode_ms=3.0,
            details="All metrics within v1 baselines",
        )
        d = asdict(health)
        assert d["status"] == "healthy"
        assert d["decode_fps"] == 30.0
        assert d["details"] == "All metrics within v1 baselines"
