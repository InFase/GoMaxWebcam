"""
test_eager_cohn_reconnect.py — Tests for AppOrchestrator eager COHN background reconnection.

Verifies:
  - On startup with cached COHN credentials, a background task launches
  - The background task injects credentials and calls discover + connect
  - The background task does NOT call start_stream (that's failover's job)
  - No COHN transport registered → task is a no-op
  - No cached credentials → task is a no-op
  - Transport already has credentials → skips store lookup, still pre-warms
  - Background task errors are logged but never crash the orchestrator
  - stop() cancels the background task cleanly
  - The main startup flow is NOT blocked by the background task
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

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
    # Default: no registered transports (empty dict)
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
def mock_cohn_transport():
    """A mock COHNTransport with the required test helper methods."""
    t = MagicMock()
    t.has_credentials = False
    t._inject_credentials = MagicMock()
    t._inject_camera_ip = MagicMock()
    t.discover = AsyncMock(return_value=True)
    t.connect = AsyncMock(return_value=True)
    t.start_stream = AsyncMock()
    t.is_streaming = False
    t.name = "COHN"
    return t


@pytest.fixture
def mock_cred_store():
    """A mock CohnCredentialStore with one valid credential."""
    from gomaxwebcam.transport.cohn_persistence import StoredCOHNCredentials

    creds = StoredCOHNCredentials(
        ip_address="192.168.1.100",
        username="gopro",
        password="s3cret",
        certificate="-----BEGIN CERTIFICATE-----\nMOCK\n-----END CERTIFICATE-----",
        camera_serial="1234",
    )
    store = MagicMock()
    store.list_cameras.return_value = [creds]
    return store, creds


def _make_orchestrator(bus, cm, tm, st=None):
    return AppOrchestrator(
        event_bus=bus,
        camera_manager=cm,
        transport_manager=tm,
        status_tracker=st,
    )


# ---------------------------------------------------------------------------
# Tests: eager COHN reconnect launches on startup
# ---------------------------------------------------------------------------

class TestEagerCohnReconnectLaunch:
    """Tests that start() launches the background reconnect task."""

    @pytest.mark.asyncio
    async def test_start_creates_cohn_reconnect_task(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager
    ):
        """start() creates _cohn_reconnect_task as a background asyncio.Task."""
        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)
        await orch.start()

        assert orch._cohn_reconnect_task is not None
        assert isinstance(orch._cohn_reconnect_task, asyncio.Task)
        # Let the task complete
        await asyncio.sleep(0.01)
        await orch.stop()

    @pytest.mark.asyncio
    async def test_start_does_not_block_on_cohn_reconnect(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager
    ):
        """start() returns immediately — the COHN reconnect runs in background."""
        # Make discover take a long time to prove start() doesn't wait
        slow_transport = MagicMock()
        slow_transport.has_credentials = True
        slow_transport.discover = AsyncMock(side_effect=lambda **kw: asyncio.sleep(10))
        slow_transport.connect = AsyncMock(return_value=True)
        mock_transport_manager.registered_transports = {"COHN": slow_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        # start() should return quickly even though COHN discover would take 10s
        await asyncio.wait_for(orch.start(), timeout=2.0)
        assert orch.is_running

        await orch.stop()


# ---------------------------------------------------------------------------
# Tests: credential lookup and injection
# ---------------------------------------------------------------------------

class TestCredentialLookup:
    """Tests for credential store lookup and injection into transport."""

    @pytest.mark.asyncio
    async def test_injects_cached_credentials_into_transport(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport, mock_cred_store,
    ):
        """When cached credentials exist, they are injected into the COHN transport."""
        store, creds = mock_cred_store
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=store):
            await orch.start()
            # Let the background task run
            await asyncio.sleep(0.05)

        mock_cohn_transport._inject_credentials.assert_called_once_with(
            creds.username, creds.password,
        )
        mock_cohn_transport._inject_camera_ip.assert_called_once_with(creds.ip_address)

        await orch.stop()

    @pytest.mark.asyncio
    async def test_calls_discover_and_connect(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport, mock_cred_store,
    ):
        """Background task calls discover() and connect() on the COHN transport."""
        store, _ = mock_cred_store
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=store):
            await orch.start()
            await asyncio.sleep(0.05)

        mock_cohn_transport.discover.assert_awaited_once()
        mock_cohn_transport.connect.assert_awaited_once()

        await orch.stop()

    @pytest.mark.asyncio
    async def test_does_not_call_start_stream(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport, mock_cred_store,
    ):
        """Background task does NOT call start_stream (that's failover's job)."""
        store, _ = mock_cred_store
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=store):
            await orch.start()
            await asyncio.sleep(0.05)

        mock_cohn_transport.start_stream.assert_not_awaited()

        await orch.stop()


# ---------------------------------------------------------------------------
# Tests: no-op scenarios
# ---------------------------------------------------------------------------

class TestNoOpScenarios:
    """Tests that the background task gracefully handles missing prerequisites."""

    @pytest.mark.asyncio
    async def test_no_cohn_transport_registered(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
    ):
        """When no COHN transport is registered, background task is a no-op."""
        mock_transport_manager.registered_transports = {"USB": MagicMock()}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)
        await orch.start()
        await asyncio.sleep(0.05)

        # Task should complete without error
        assert orch._cohn_reconnect_task.done()
        assert orch._cohn_reconnect_task.exception() is None

        await orch.stop()

    @pytest.mark.asyncio
    async def test_no_cached_credentials(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport,
    ):
        """When no cached credentials exist, background task is a no-op."""
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        empty_store = MagicMock()
        empty_store.list_cameras.return_value = []

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=empty_store):
            await orch.start()
            await asyncio.sleep(0.05)

        # No credentials → no discover/connect calls
        mock_cohn_transport.discover.assert_not_awaited()
        mock_cohn_transport.connect.assert_not_awaited()

        await orch.stop()

    @pytest.mark.asyncio
    async def test_no_valid_credentials(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport,
    ):
        """When credentials exist but are incomplete, background task is a no-op."""
        from gomaxwebcam.transport.cohn_persistence import StoredCOHNCredentials

        invalid_creds = StoredCOHNCredentials(
            ip_address="", username="", password="", certificate="", camera_serial="1234",
        )
        store = MagicMock()
        store.list_cameras.return_value = [invalid_creds]

        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=store):
            await orch.start()
            await asyncio.sleep(0.05)

        mock_cohn_transport.discover.assert_not_awaited()

        await orch.stop()

    @pytest.mark.asyncio
    async def test_transport_already_has_credentials_skips_store(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport,
    ):
        """When transport already has credentials, store lookup is skipped."""
        mock_cohn_transport.has_credentials = True
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore") as MockStore:
            await orch.start()
            await asyncio.sleep(0.05)
            # Store should not be instantiated
            MockStore.assert_not_called()

        # But discover + connect should still be called
        mock_cohn_transport.discover.assert_awaited_once()
        mock_cohn_transport.connect.assert_awaited_once()

        await orch.stop()


# ---------------------------------------------------------------------------
# Tests: error resilience
# ---------------------------------------------------------------------------

class TestErrorResilience:
    """Tests that background task errors never crash the orchestrator."""

    @pytest.mark.asyncio
    async def test_discover_failure_does_not_crash(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport, mock_cred_store,
    ):
        """If discover() returns False, task exits gracefully."""
        store, _ = mock_cred_store
        mock_cohn_transport.discover = AsyncMock(return_value=False)
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=store):
            await orch.start()
            await asyncio.sleep(0.05)

        # connect should not be called after discover failure
        mock_cohn_transport.connect.assert_not_awaited()
        assert orch.is_running

        await orch.stop()

    @pytest.mark.asyncio
    async def test_connect_failure_does_not_crash(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport, mock_cred_store,
    ):
        """If connect() returns False, task exits gracefully."""
        store, _ = mock_cred_store
        mock_cohn_transport.connect = AsyncMock(return_value=False)
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=store):
            await orch.start()
            await asyncio.sleep(0.05)

        assert orch.is_running

        await orch.stop()

    @pytest.mark.asyncio
    async def test_exception_in_discover_does_not_crash(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport, mock_cred_store,
    ):
        """If discover() raises an exception, task catches it gracefully."""
        store, _ = mock_cred_store
        mock_cohn_transport.discover = AsyncMock(side_effect=ConnectionError("network down"))
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch("gomaxwebcam.orchestrator.CohnCredentialStore", return_value=store):
            await orch.start()
            await asyncio.sleep(0.05)

        # Orchestrator should still be running
        assert orch.is_running
        # Task should have completed (not left dangling)
        assert orch._cohn_reconnect_task.done()

        await orch.stop()

    @pytest.mark.asyncio
    async def test_credential_store_error_does_not_crash(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
        mock_cohn_transport,
    ):
        """If CohnCredentialStore raises, task catches it gracefully."""
        mock_transport_manager.registered_transports = {"COHN": mock_cohn_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)

        with patch(
            "gomaxwebcam.orchestrator.CohnCredentialStore",
            side_effect=RuntimeError("DB corrupt"),
        ):
            await orch.start()
            await asyncio.sleep(0.05)

        assert orch.is_running

        await orch.stop()


# ---------------------------------------------------------------------------
# Tests: stop() cancellation
# ---------------------------------------------------------------------------

class TestStopCancellation:
    """Tests that stop() properly cancels the background task."""

    @pytest.mark.asyncio
    async def test_stop_cancels_running_reconnect_task(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
    ):
        """stop() cancels a still-running COHN reconnect task."""
        # Make discover hang forever so the task is still running at stop time
        slow_transport = MagicMock()
        slow_transport.has_credentials = True
        slow_transport.discover = AsyncMock(side_effect=lambda **kw: asyncio.sleep(999))
        slow_transport.connect = AsyncMock()
        mock_transport_manager.registered_transports = {"COHN": slow_transport}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)
        await orch.start()

        # Task should be running (discover is blocking)
        assert orch._cohn_reconnect_task is not None
        assert not orch._cohn_reconnect_task.done()

        # stop() should cancel it
        await orch.stop()
        assert orch._cohn_reconnect_task is None

    @pytest.mark.asyncio
    async def test_stop_handles_already_completed_task(
        self, mock_event_bus, mock_camera_manager, mock_transport_manager,
    ):
        """stop() handles a reconnect task that already completed."""
        # No COHN transport → task completes immediately
        mock_transport_manager.registered_transports = {}

        orch = _make_orchestrator(mock_event_bus, mock_camera_manager, mock_transport_manager)
        await orch.start()
        await asyncio.sleep(0.05)  # Let task complete

        assert orch._cohn_reconnect_task.done()

        # stop() should not raise
        await orch.stop()
        assert orch._cohn_reconnect_task is None
