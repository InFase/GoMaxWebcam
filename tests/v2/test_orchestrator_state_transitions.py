"""
test_orchestrator_state_transitions.py — Tests for AppOrchestrator state transitions,
transport failover integration, freeze-frame bridging, and credential failure handling.

Complements test_orchestrator_lifecycle.py (start/stop order) and
test_orchestrator_dashboard.py (FastAPI creation) with deeper coverage of:

  - Initial property/state correctness
  - set_pipeline() manual wiring
  - _ensure_pipeline() edge cases (no stream_info, start failure, codec error)
  - Freeze/unfreeze callback propagation through orchestrator → pipeline
  - Full USB→COHN failover flow through orchestrator callbacks
  - _handle_cohn_credential_failure() invalidation + COHN_REPROVISION event
  - Eager COHN reconnect with invalid credentials triggers cache invalidation
  - Pipeline restart on transport switch during active streaming
  - stop() cleans up pipeline reference
  - Restart cycle preserves clean state
  - Multiple rapid transport switches don't corrupt state

All tests are pure unit tests with mocks — no GoPro or hardware needed.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from gomaxwebcam.orchestrator import AppOrchestrator
from gomaxwebcam.transport.base import StreamInfo

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
    bus.publish = MagicMock()
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
    tm.registered_transports = {}
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
    return AppOrchestrator(
        event_bus=mock_event_bus,
        camera_manager=mock_camera_manager,
        transport_manager=mock_transport_manager,
        status_tracker=None,
    )


def _make_streaming_transport(name: str = "USB") -> MagicMock:
    """Helper: mock transport that is_streaming=True with valid stream_info."""
    t = MagicMock()
    t.name = name
    t.is_streaming = True
    t.stream_info = StreamInfo(
        protocol="udp", host="0.0.0.0", port=8554,
        width=1920, height=1080, fps=30, codec="h264",
    )
    return t


def _make_idle_transport(name: str = "USB") -> MagicMock:
    """Helper: mock transport that is_streaming=False."""
    t = MagicMock()
    t.name = name
    t.is_streaming = False
    return t


# ---------------------------------------------------------------------------
# Tests: Initial state & properties
# ---------------------------------------------------------------------------

class TestInitialState:
    """Verify orchestrator properties before any lifecycle methods are called."""

    def test_is_running_false_initially(self, orchestrator):
        assert orchestrator.is_running is False

    def test_pipeline_none_initially(self, orchestrator):
        assert orchestrator.pipeline is None

    def test_app_none_initially(self, orchestrator):
        assert orchestrator.app is None

    def test_auth_token_none_initially(self, orchestrator):
        assert orchestrator.auth_token is None

    def test_dashboard_url_none_initially(self, orchestrator):
        assert orchestrator.dashboard_url is None

    def test_event_bus_accessible(self, orchestrator, mock_event_bus):
        assert orchestrator.event_bus is mock_event_bus

    def test_camera_manager_accessible(self, orchestrator, mock_camera_manager):
        assert orchestrator.camera_manager is mock_camera_manager

    def test_transport_manager_accessible(self, orchestrator, mock_transport_manager):
        assert orchestrator.transport_manager is mock_transport_manager

    def test_status_tracker_accessible(self, orchestrator, mock_status_tracker):
        assert orchestrator.status_tracker is mock_status_tracker

    def test_status_tracker_none_when_omitted(self, orchestrator_no_tracker):
        assert orchestrator_no_tracker.status_tracker is None


# ---------------------------------------------------------------------------
# Tests: set_pipeline() manual wiring
# ---------------------------------------------------------------------------

class TestSetPipeline:
    """Tests for manually setting the pipeline via set_pipeline()."""

    def test_set_pipeline_stores_reference(self, orchestrator):
        mock_pl = MagicMock()
        orchestrator.set_pipeline(mock_pl)
        assert orchestrator.pipeline is mock_pl

    def test_set_pipeline_wires_camera_manager(self, orchestrator, mock_camera_manager):
        mock_pl = MagicMock()
        orchestrator.set_pipeline(mock_pl)
        mock_camera_manager.set_pipeline.assert_called_once_with(mock_pl)

    def test_set_pipeline_wires_status_tracker(self, orchestrator, mock_status_tracker):
        mock_pl = MagicMock()
        orchestrator.set_pipeline(mock_pl)
        mock_status_tracker.set_pipeline.assert_called_once_with(mock_pl)

    def test_set_pipeline_skips_tracker_when_none(self, orchestrator_no_tracker, mock_camera_manager):
        mock_pl = MagicMock()
        orchestrator_no_tracker.set_pipeline(mock_pl)
        mock_camera_manager.set_pipeline.assert_called_once_with(mock_pl)

    def test_set_pipeline_overwrites_previous(self, orchestrator):
        pl1 = MagicMock()
        pl2 = MagicMock()
        orchestrator.set_pipeline(pl1)
        orchestrator.set_pipeline(pl2)
        assert orchestrator.pipeline is pl2


# ---------------------------------------------------------------------------
# Tests: _ensure_pipeline() edge cases
# ---------------------------------------------------------------------------

class TestEnsurePipeline:
    """Tests for _ensure_pipeline() — the transport-to-pipeline bridge."""

    @pytest.mark.asyncio
    async def test_skips_when_not_streaming(self, orchestrator):
        """No pipeline is created when transport is not streaming."""
        transport = _make_idle_transport()
        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPL:
            await orchestrator._ensure_pipeline(transport)
            MockPL.assert_not_called()
        assert orchestrator.pipeline is None

    @pytest.mark.asyncio
    async def test_skips_when_no_stream_info(self, orchestrator):
        """No pipeline is created when transport has no stream_info attribute."""
        transport = MagicMock()
        transport.is_streaming = True
        # Explicitly remove stream_info
        del transport.stream_info
        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPL:
            await orchestrator._ensure_pipeline(transport)
            MockPL.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_stream_info_is_none(self, orchestrator):
        """No pipeline is created when stream_info is None."""
        transport = MagicMock()
        transport.is_streaming = True
        transport.stream_info = None
        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPL:
            await orchestrator._ensure_pipeline(transport)
            MockPL.assert_not_called()

    @pytest.mark.asyncio
    async def test_creates_pipeline_when_streaming(self, orchestrator, mock_camera_manager, mock_status_tracker):
        """Pipeline is created and wired when transport is streaming with valid info."""
        transport = _make_streaming_transport("USB")
        mock_pl = AsyncMock()
        mock_pl.start = AsyncMock(return_value=True)

        with patch("gomaxwebcam.orchestrator.FramePipeline", return_value=mock_pl) as MockPL:
            await orchestrator._ensure_pipeline(transport)

            MockPL.assert_called_once()
            mock_pl.start.assert_awaited_once_with(transport.stream_info)
            mock_camera_manager.set_pipeline.assert_called_with(mock_pl)
            mock_status_tracker.set_pipeline.assert_called_with(mock_pl)
            assert orchestrator.pipeline is mock_pl

    @pytest.mark.asyncio
    async def test_hot_swaps_running_pipeline(self, orchestrator):
        """A running pipeline is hot-swapped (decoder replaced, vcam kept alive)."""
        old_pl = MagicMock()
        old_pl.is_running = True
        old_pl.switch_stream = AsyncMock(return_value=True)
        old_pl.stop = AsyncMock()
        orchestrator._pipeline = old_pl

        transport = _make_streaming_transport("COHN")

        await orchestrator._ensure_pipeline(transport)

        # Hot-swap used: switch_stream called, stop NOT called
        old_pl.switch_stream.assert_awaited_once()
        old_pl.stop.assert_not_awaited()
        # Pipeline is the same object (not replaced)
        assert orchestrator.pipeline is old_pl

    @pytest.mark.asyncio
    async def test_stops_non_running_pipeline_before_new(self, orchestrator):
        """A non-running pipeline is stopped before a new one is created."""
        old_pl = MagicMock()
        old_pl.is_running = False
        old_pl.stop = AsyncMock()
        orchestrator._pipeline = old_pl

        transport = _make_streaming_transport("COHN")
        new_pl = MagicMock()
        new_pl.start = AsyncMock(return_value=True)

        with patch("gomaxwebcam.orchestrator.FramePipeline", return_value=new_pl):
            await orchestrator._ensure_pipeline(transport)

        old_pl.stop.assert_awaited_once()
        assert orchestrator.pipeline is new_pl

    @pytest.mark.asyncio
    async def test_hot_swap_failure_falls_back_to_full_restart(self, orchestrator):
        """If switch_stream fails, old pipeline is stopped and new one created."""
        old_pl = MagicMock()
        old_pl.is_running = True
        old_pl.switch_stream = AsyncMock(return_value=False)
        old_pl.stop = AsyncMock()
        orchestrator._pipeline = old_pl

        transport = _make_streaming_transport("USB")
        new_pl = MagicMock()
        new_pl.start = AsyncMock(return_value=True)

        with patch("gomaxwebcam.orchestrator.FramePipeline", return_value=new_pl):
            await orchestrator._ensure_pipeline(transport)

        old_pl.stop.assert_awaited()
        assert orchestrator.pipeline is new_pl

    @pytest.mark.asyncio
    async def test_pipeline_start_failure_does_not_store(self, orchestrator):
        """If pipeline.start() returns False, pipeline is NOT stored."""
        transport = _make_streaming_transport("USB")
        mock_pl = AsyncMock()
        mock_pl.start = AsyncMock(return_value=False)

        with patch("gomaxwebcam.orchestrator.FramePipeline", return_value=mock_pl):
            await orchestrator._ensure_pipeline(transport)

        # Pipeline should NOT be stored since start failed
        assert orchestrator.pipeline is None

    @pytest.mark.asyncio
    async def test_invalid_codec_does_not_crash(self, orchestrator):
        """If PipelineConfig.from_stream_info() raises ValueError, no crash."""
        transport = MagicMock()
        transport.is_streaming = True
        transport.name = "USB"
        transport.stream_info = StreamInfo(
            protocol="udp", host="0.0.0.0", port=8554,
            width=1920, height=1080, fps=30, codec="av1_unsupported",
        )

        with patch(
            "gomaxwebcam.orchestrator.PipelineConfig.from_stream_info",
            side_effect=ValueError("Unsupported codec: av1_unsupported"),
        ):
            # Should not raise
            await orchestrator._ensure_pipeline(transport)

        assert orchestrator.pipeline is None


# ---------------------------------------------------------------------------
# Tests: Freeze / unfreeze propagation
# ---------------------------------------------------------------------------

class TestFreezeUnfreeze:
    """Tests that freeze/unfreeze callbacks propagate to the pipeline."""

    def test_on_freeze_calls_pipeline_freeze(self, orchestrator):
        mock_pl = MagicMock()
        mock_pl.freeze = MagicMock()
        orchestrator._pipeline = mock_pl

        orchestrator._on_freeze()
        mock_pl.freeze.assert_called_once()

    def test_on_unfreeze_calls_pipeline_unfreeze(self, orchestrator):
        mock_pl = MagicMock()
        mock_pl.unfreeze = MagicMock()
        orchestrator._pipeline = mock_pl

        orchestrator._on_unfreeze()
        mock_pl.unfreeze.assert_called_once()

    def test_on_freeze_noop_without_pipeline(self, orchestrator):
        """freeze is safe when no pipeline is set."""
        assert orchestrator.pipeline is None
        orchestrator._on_freeze()  # Should not raise

    def test_on_unfreeze_noop_without_pipeline(self, orchestrator):
        """unfreeze is safe when no pipeline is set."""
        assert orchestrator.pipeline is None
        orchestrator._on_unfreeze()  # Should not raise

    @pytest.mark.asyncio
    async def test_wired_freeze_callback_reaches_pipeline(self, orchestrator, mock_transport_manager):
        """After start(), the on_freeze callback wired to TransportManager calls pipeline.freeze()."""
        await orchestrator.start()

        mock_pl = MagicMock()
        mock_pl.freeze = MagicMock()
        mock_pl.unfreeze = MagicMock()
        orchestrator._pipeline = mock_pl

        # Call the wired callback
        mock_transport_manager.on_freeze()
        mock_pl.freeze.assert_called_once()

        mock_transport_manager.on_unfreeze()
        mock_pl.unfreeze.assert_called_once()

        await orchestrator.stop()


# ---------------------------------------------------------------------------
# Tests: Full failover flow through orchestrator callbacks
# ---------------------------------------------------------------------------

class TestFailoverFlow:
    """Simulate a USB→COHN failover through orchestrator callback chain."""

    @pytest.mark.asyncio
    async def test_usb_to_cohn_switch_updates_camera_manager(
        self, orchestrator, mock_camera_manager, mock_transport_manager
    ):
        """Transport switch from USB to COHN updates CameraManager."""
        await orchestrator.start()

        usb_transport = _make_streaming_transport("USB")
        cohn_transport = _make_streaming_transport("COHN")

        # Simulate initial USB connection
        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPL:
            mock_pl = MagicMock()
            mock_pl.start = AsyncMock(return_value=True)
            mock_pl.stop = AsyncMock()
            mock_pl.freeze = MagicMock()
            mock_pl.unfreeze = MagicMock()
            MockPL.return_value = mock_pl

            await orchestrator._on_transport_switch("USB", usb_transport)
            mock_camera_manager.set_transport.assert_called_with(usb_transport)

            # Now freeze (disconnect detected)
            orchestrator._on_freeze()
            mock_pl.freeze.assert_called_once()

            # Failover to COHN
            await orchestrator._on_transport_switch("COHN", cohn_transport)
            mock_camera_manager.set_transport.assert_called_with(cohn_transport)

            # Unfreeze (new stream ready)
            orchestrator._on_unfreeze()
            mock_pl.unfreeze.assert_called_once()

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_transport_switch_hot_swaps_running_pipeline(
        self, orchestrator, mock_camera_manager, mock_status_tracker
    ):
        """Switching transports while streaming hot-swaps the pipeline decoder.

        The vcam sink stays alive for freeze-frame continuity — the same
        pipeline object is reused with switch_stream().
        """
        await orchestrator.start()

        # First transport starts a pipeline
        usb = _make_streaming_transport("USB")
        first_pl = MagicMock()
        first_pl.start = AsyncMock(return_value=True)
        first_pl.stop = AsyncMock()
        first_pl.freeze = MagicMock()
        first_pl.unfreeze = MagicMock()
        first_pl.is_running = False  # Not running yet (just created)

        with patch("gomaxwebcam.orchestrator.FramePipeline", return_value=first_pl):
            await orchestrator._on_transport_switch("USB", usb)
        assert orchestrator.pipeline is first_pl

        # Now mark pipeline as running (simulating it started successfully)
        first_pl.is_running = True
        first_pl.switch_stream = AsyncMock(return_value=True)

        # Second transport causes hot-swap
        cohn = _make_streaming_transport("COHN")

        await orchestrator._on_transport_switch("COHN", cohn)

        # Hot-swap used: switch_stream called, pipeline NOT stopped/replaced
        first_pl.switch_stream.assert_awaited_once()
        first_pl.stop.assert_not_awaited()
        # Same pipeline object still active
        assert orchestrator.pipeline is first_pl

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_transport_switch_to_idle_does_not_create_pipeline(
        self, orchestrator, mock_camera_manager
    ):
        """If the new transport is not streaming, no pipeline is created."""
        await orchestrator.start()

        idle_transport = _make_idle_transport("COHN")

        with patch("gomaxwebcam.orchestrator.FramePipeline") as MockPL:
            await orchestrator._on_transport_switch("COHN", idle_transport)
            MockPL.assert_not_called()

        mock_camera_manager.set_transport.assert_called_with(idle_transport)

        await orchestrator.stop()


# ---------------------------------------------------------------------------
# Tests: COHN credential failure handling
# ---------------------------------------------------------------------------

class TestCOHNCredentialFailure:
    """Tests for _handle_cohn_credential_failure()."""

    def test_invalidates_credentials_by_serial(self, orchestrator, mock_event_bus):
        """Credentials are removed from the store when serial is provided."""
        mock_store = MagicMock()
        mock_store.invalidate_credentials = MagicMock(return_value=True)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=mock_store):
            orchestrator._handle_cohn_credential_failure(
                camera_serial="1234",
                camera_ip="192.168.1.100",
            )

        mock_store.invalidate_credentials.assert_called_once_with("1234")

    def test_invalidates_all_when_no_serial(self, orchestrator, mock_event_bus):
        """When no serial is provided, all cached credentials are invalidated."""
        from gomaxwebcam.transport.cohn_persistence import StoredCOHNCredentials

        creds1 = StoredCOHNCredentials(
            ip_address="192.168.1.100", username="gopro", password="p1",
            certificate="cert", camera_serial="1111",
        )
        creds2 = StoredCOHNCredentials(
            ip_address="192.168.1.101", username="gopro", password="p2",
            certificate="cert", camera_serial="2222",
        )

        mock_store = MagicMock()
        mock_store.list_cameras.return_value = [creds1, creds2]
        mock_store.invalidate_credentials = MagicMock()

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=mock_store):
            orchestrator._handle_cohn_credential_failure(camera_serial="", camera_ip="")

        assert mock_store.invalidate_credentials.call_count == 2
        mock_store.invalidate_credentials.assert_any_call("1111")
        mock_store.invalidate_credentials.assert_any_call("2222")

    def test_publishes_cohn_reprovision_event(self, orchestrator, mock_event_bus):
        """A COHN_REPROVISION event is published to the EventBus."""
        mock_store = MagicMock()
        mock_store.invalidate_credentials = MagicMock()

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=mock_store):
            orchestrator._handle_cohn_credential_failure(
                camera_serial="1234",
                camera_ip="192.168.1.100",
            )

        mock_event_bus.publish.assert_called_once()
        event = mock_event_bus.publish.call_args[0][0]
        assert event.type.name == "COHN_REPROVISION"
        assert "1234" in event.data.get("camera_serial", "")

    def test_store_error_does_not_crash(self, orchestrator, mock_event_bus):
        """If CohnCredentialStore raises, the method still publishes the event."""
        with patch(
            "gomaxwebcam.orchestrator.CohnCredentialStore",
            side_effect=RuntimeError("DB locked"),
        ):
            # Should not raise
            orchestrator._handle_cohn_credential_failure(
                camera_serial="1234",
                camera_ip="192.168.1.100",
            )

        # Event should still be published despite store error
        mock_event_bus.publish.assert_called_once()

    def test_publish_error_does_not_crash(self, orchestrator, mock_event_bus):
        """If EventBus.publish raises, the method does not crash."""
        mock_store = MagicMock()
        mock_store.invalidate_credentials = MagicMock()
        mock_event_bus.publish = MagicMock(side_effect=RuntimeError("bus down"))

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=mock_store):
            # Should not raise
            orchestrator._handle_cohn_credential_failure(
                camera_serial="1234",
                camera_ip="192.168.1.100",
            )

        # Invalidation still happened
        mock_store.invalidate_credentials.assert_called_once_with("1234")


# ---------------------------------------------------------------------------
# Tests: Eager COHN reconnect with invalid credentials
# ---------------------------------------------------------------------------

class TestEagerReconnectInvalidCreds:
    """Eager COHN reconnect detects invalid credentials and handles them."""

    @pytest.mark.asyncio
    async def test_invalid_creds_triggers_cache_invalidation(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager, mock_status_tracker,
    ):
        """When connect() fails with credentials_invalid, cache is invalidated."""
        from gomaxwebcam.transport.cohn_persistence import StoredCOHNCredentials

        creds = StoredCOHNCredentials(
            ip_address="192.168.1.100", username="gopro", password="old-pass",
            certificate="cert", camera_serial="9999",
        )
        mock_store = MagicMock()
        mock_store.list_cameras.return_value = [creds]

        mock_cohn = MagicMock()
        mock_cohn.has_credentials = False
        mock_cohn._inject_credentials = MagicMock()
        mock_cohn._inject_camera_ip = MagicMock()
        mock_cohn.discover = AsyncMock(return_value=True)
        mock_cohn.connect = AsyncMock(return_value=False)
        mock_cohn.credentials_invalid = True  # Signal bad credentials

        mock_transport_manager.registered_transports = {"COHN": mock_cohn}

        orch = AppOrchestrator(
            event_bus=mock_event_bus,
            camera_manager=mock_camera_manager,
            transport_manager=mock_transport_manager,
            status_tracker=mock_status_tracker,
        )

        mock_inv_store = MagicMock()
        mock_inv_store.invalidate_credentials = MagicMock()
        mock_inv_store.list_cameras = MagicMock(return_value=[])

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore") as MockStoreClass:
            # First call: for loading creds in _try_eager_cohn_reconnect
            # Second call: for invalidation in _handle_cohn_credential_failure
            MockStoreClass.side_effect = [mock_store, mock_inv_store]

            await orch.start()
            await asyncio.sleep(0.1)

        # Verify re-provisioning event was published
        pub_calls = mock_event_bus.publish.call_args_list
        reprovision_events = [
            c for c in pub_calls
            if hasattr(c[0][0], 'type') and c[0][0].type.name == "COHN_REPROVISION"
        ]
        assert len(reprovision_events) >= 1

        await orch.stop()


# ---------------------------------------------------------------------------
# Tests: stop() pipeline cleanup
# ---------------------------------------------------------------------------

class TestStopPipelineCleanup:
    """Tests that stop() properly cleans up the pipeline."""

    @pytest.mark.asyncio
    async def test_stop_stops_active_pipeline(self, orchestrator):
        """stop() calls pipeline.stop() if a pipeline is active."""
        mock_pl = AsyncMock()
        mock_pl.stop = AsyncMock()
        orchestrator._pipeline = mock_pl

        await orchestrator.start()
        await orchestrator.stop()

        mock_pl.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_clears_pipeline_reference(self, orchestrator):
        """stop() sets _pipeline to None."""
        mock_pl = AsyncMock()
        mock_pl.stop = AsyncMock()
        orchestrator._pipeline = mock_pl

        await orchestrator.start()
        await orchestrator.stop()

        assert orchestrator.pipeline is None

    @pytest.mark.asyncio
    async def test_stop_without_pipeline_is_safe(self, orchestrator):
        """stop() is safe when no pipeline was ever created."""
        await orchestrator.start()
        assert orchestrator.pipeline is None
        await orchestrator.stop()  # Should not raise

    @pytest.mark.asyncio
    async def test_pipeline_stop_error_continues_shutdown(
        self, orchestrator, mock_camera_manager, mock_event_bus
    ):
        """If pipeline.stop() raises, remaining components still stop."""
        mock_pl = AsyncMock()
        mock_pl.stop = AsyncMock(side_effect=RuntimeError("vcam busy"))
        orchestrator._pipeline = mock_pl

        await orchestrator.start()
        await orchestrator.stop()

        # Shutdown continued despite pipeline error
        mock_camera_manager.stop.assert_awaited_once()
        mock_event_bus.shutdown.assert_awaited_once()
        assert not orchestrator.is_running


# ---------------------------------------------------------------------------
# Tests: Restart cycle
# ---------------------------------------------------------------------------

class TestRestartCycle:
    """Tests for start → stop → start again."""

    @pytest.mark.asyncio
    async def test_restart_resets_callback_wiring(
        self, orchestrator, mock_transport_manager
    ):
        """After restart, callbacks are re-wired fresh."""
        await orchestrator.start()
        await orchestrator.stop()

        # Reset mocks
        mock_transport_manager.on_transport_switch = None
        mock_transport_manager.on_freeze = None
        mock_transport_manager.on_unfreeze = None

        await orchestrator.start()

        assert mock_transport_manager.on_transport_switch is not None
        assert mock_transport_manager.on_freeze is not None
        assert mock_transport_manager.on_unfreeze is not None

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_restart_clears_pipeline_from_previous_cycle(self, orchestrator):
        """Pipeline from a previous cycle does not leak into a new start."""
        # First cycle: create a pipeline
        mock_pl = AsyncMock()
        mock_pl.stop = AsyncMock()
        orchestrator._pipeline = mock_pl

        await orchestrator.start()
        await orchestrator.stop()

        # Pipeline was cleaned up
        assert orchestrator.pipeline is None

        # Second cycle starts clean
        await orchestrator.start()
        assert orchestrator.pipeline is None

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_restart_launches_new_cohn_reconnect_task(
        self, orchestrator, mock_transport_manager
    ):
        """Each start() creates a fresh COHN reconnect background task."""
        await orchestrator.start()
        first_task = orchestrator._cohn_reconnect_task
        assert first_task is not None

        await orchestrator.stop()
        assert orchestrator._cohn_reconnect_task is None

        await orchestrator.start()
        second_task = orchestrator._cohn_reconnect_task
        assert second_task is not None
        assert second_task is not first_task

        await orchestrator.stop()


# ---------------------------------------------------------------------------
# Tests: Multiple rapid transport switches
# ---------------------------------------------------------------------------

class TestRapidSwitches:
    """Tests that rapid transport switches don't corrupt state."""

    @pytest.mark.asyncio
    async def test_rapid_switches_last_transport_wins(
        self, orchestrator, mock_camera_manager
    ):
        """Multiple quick _on_transport_switch calls: last one determines state."""
        await orchestrator.start()

        transports = []
        for name in ["USB", "COHN", "USB", "COHN"]:
            t = _make_idle_transport(name)
            transports.append(t)
            await orchestrator._on_transport_switch(name, t)

        # CameraManager should have been called 4 times, last with COHN
        assert mock_camera_manager.set_transport.call_count == 4
        mock_camera_manager.set_transport.assert_called_with(transports[-1])

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_rapid_switches_with_streaming_hot_swaps_each_time(
        self, orchestrator
    ):
        """Each switch to a streaming transport hot-swaps the pipeline decoder.

        With the hot-swap optimization, the same pipeline object is reused
        across transport switches.  switch_stream() is called for each switch
        after the first (which creates the pipeline).
        """
        await orchestrator.start()

        # First switch creates a new pipeline
        first_transport = _make_streaming_transport("T0")
        first_pl = MagicMock()
        first_pl.start = AsyncMock(return_value=True)
        first_pl.stop = AsyncMock()
        first_pl.is_running = False

        with patch("gomaxwebcam.orchestrator.FramePipeline", return_value=first_pl):
            await orchestrator._on_transport_switch("T0", first_transport)

        assert orchestrator.pipeline is first_pl

        # Mark as running for subsequent hot-swaps
        first_pl.is_running = True
        first_pl.switch_stream = AsyncMock(return_value=True)

        # Subsequent switches hot-swap the decoder
        for i in range(1, 3):
            transport = _make_streaming_transport(f"T{i}")
            await orchestrator._on_transport_switch(f"T{i}", transport)

        # switch_stream called for each subsequent switch
        assert first_pl.switch_stream.await_count == 2
        # Pipeline never stopped (hot-swap preserves vcam)
        first_pl.stop.assert_not_awaited()
        # Same pipeline object still active
        assert orchestrator.pipeline is first_pl

        await orchestrator.stop()


# ---------------------------------------------------------------------------
# Tests: Dashboard URL
# ---------------------------------------------------------------------------

class TestDashboardURL:
    """Tests for dashboard_url property formatting."""

    def test_url_format(self, orchestrator):
        orchestrator.create_dashboard(auth_token="tok123", host="0.0.0.0", port=5000)
        assert orchestrator.dashboard_url == "http://0.0.0.0:5000/?token=tok123"

    def test_url_none_without_port(self, orchestrator):
        """dashboard_url is None if create_dashboard hasn't set all fields."""
        orchestrator._dashboard_host = "127.0.0.1"
        orchestrator._auth_token = "tok"
        # port is still None
        assert orchestrator.dashboard_url is None
