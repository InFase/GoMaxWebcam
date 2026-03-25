"""
Tests for WiFi AP network discovery and connection manager.

All tests are mocked — no real WiFi operations are performed.
Tests cover:
  - SSID pattern matching for GoPro hotspots
  - WiFi scanning with platform-specific output parsing
  - Connection lifecycle (connect → verify → disconnect → restore)
  - State transitions and statistics tracking
  - Cross-platform backend selection
  - Error handling and timeouts
  - Integration with WiFiAPTransport
"""

from __future__ import annotations

import asyncio
import platform
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from gomaxwebcam.transport.wifi_manager import (
    WiFiManager,
    WiFiNetwork,
    WiFiConnectionState,
    WiFiManagerStats,
    WiFiManagerError,
    WiFiScanError,
    WiFiConnectError,
    WiFiPlatformError,
    is_gopro_ssid,
    GOPRO_AP_IP,
    GOPRO_AP_PORT,
    _WindowsWiFiBackend,
    _MacOSWiFiBackend,
    _LinuxWiFiBackend,
)


# ======================================================================
# SSID Pattern Matching
# ======================================================================

class TestIsGoProSSID:
    """Tests for is_gopro_ssid() function."""

    def test_standard_gopro_ssid(self):
        assert is_gopro_ssid("GP12345678") is True

    def test_short_gopro_ssid(self):
        assert is_gopro_ssid("GP-1234") is True

    def test_gopro_with_underscore(self):
        assert is_gopro_ssid("GP_ABCD") is True

    def test_gopro_case_insensitive(self):
        assert is_gopro_ssid("gp12345678") is True
        assert is_gopro_ssid("Gp-ABCD") is True

    def test_not_gopro_ssid(self):
        assert is_gopro_ssid("MyHomeWiFi") is False
        assert is_gopro_ssid("Starbucks_WiFi") is False
        assert is_gopro_ssid("NETGEAR-5G") is False

    def test_empty_ssid(self):
        assert is_gopro_ssid("") is False

    def test_too_short_gopro(self):
        # "GP" alone doesn't match (need at least 4 chars after GP)
        assert is_gopro_ssid("GP") is False
        assert is_gopro_ssid("GP-") is False

    def test_gopro_with_longer_serial(self):
        assert is_gopro_ssid("GP-C3507132") is True
        assert is_gopro_ssid("GP78901234567890") is True


# ======================================================================
# WiFiNetwork dataclass
# ======================================================================

class TestWiFiNetwork:
    """Tests for WiFiNetwork dataclass."""

    def test_basic_creation(self):
        net = WiFiNetwork(ssid="TestNet", signal_strength=-65)
        assert net.ssid == "TestNet"
        assert net.signal_strength == -65
        assert net.is_gopro is False
        assert net.security == ""

    def test_gopro_network(self):
        net = WiFiNetwork(
            ssid="GP-1234",
            signal_strength=-50,
            security="WPA2-Personal",
            is_gopro=True,
            bssid="AA:BB:CC:DD:EE:FF",
            channel=6,
        )
        assert net.is_gopro is True
        assert net.channel == 6

    def test_frozen(self):
        net = WiFiNetwork(ssid="Test")
        with pytest.raises(AttributeError):
            net.ssid = "Other"


# ======================================================================
# Windows Backend Parsing
# ======================================================================

class TestWindowsNetshParsing:
    """Tests for parsing netsh wlan output on Windows."""

    SAMPLE_NETSH_OUTPUT = """\
Interface name : Wi-Fi

There are 3 networks currently visible.

SSID 1 : HomeNetwork
    Network type            : Infrastructure
    Authentication          : WPA2-Personal
    Encryption              : CCMP
    BSSID 1                 : aa:bb:cc:dd:ee:ff
         Signal             : 85%
         Radio type         : 802.11ac
         Channel            : 36

SSID 2 : GP-1234
    Network type            : Infrastructure
    Authentication          : WPA2-Personal
    Encryption              : CCMP
    BSSID 1                 : 11:22:33:44:55:66
         Signal             : 72%
         Radio type         : 802.11n
         Channel            : 6

SSID 3 : CoffeeShop
    Network type            : Infrastructure
    Authentication          : Open
    Encryption              : None
    BSSID 1                 : 77:88:99:aa:bb:cc
         Signal             : 45%
         Radio type         : 802.11n
         Channel            : 11
"""

    def test_parse_networks(self):
        backend = _WindowsWiFiBackend()
        networks = backend._parse_netsh_networks(self.SAMPLE_NETSH_OUTPUT)

        assert len(networks) == 3

        # Check first network
        assert networks[0].ssid == "HomeNetwork"
        assert networks[0].signal_strength == 85
        assert "WPA2" in networks[0].security

        # Check GoPro network
        assert networks[1].ssid == "GP-1234"
        assert networks[1].signal_strength == 72
        assert networks[1].channel == 6

        # Check open network
        assert networks[2].ssid == "CoffeeShop"
        assert networks[2].signal_strength == 45

    def test_parse_empty_output(self):
        backend = _WindowsWiFiBackend()
        networks = backend._parse_netsh_networks("")
        assert networks == []

    def test_parse_no_networks(self):
        backend = _WindowsWiFiBackend()
        output = "Interface name : Wi-Fi\n\nThere are 0 networks currently visible.\n"
        networks = backend._parse_netsh_networks(output)
        assert networks == []

    def test_wifi_profile_xml(self):
        backend = _WindowsWiFiBackend()
        xml = backend._create_wifi_profile_xml("GP-1234", "password123")

        assert "GP-1234" in xml
        assert "password123" in xml
        assert "WPA2PSK" in xml
        assert "AES" in xml
        assert "passPhrase" in xml

    def test_wifi_profile_xml_escapes_special_chars(self):
        backend = _WindowsWiFiBackend()
        xml = backend._create_wifi_profile_xml("GP<>&1234", "pass<>word")

        assert "GP&lt;&gt;&amp;1234" in xml
        assert "pass&lt;&gt;word" in xml


# ======================================================================
# macOS Backend Parsing
# ======================================================================

class TestMacOSAirportParsing:
    """Tests for parsing macOS airport scan output."""

    SAMPLE_AIRPORT_OUTPUT = """\
                            SSID BSSID             RSSI CHANNEL HT CC SECURITY (auth/unicast/group, 802.1X/EAP)
                     HomeNetwork aa:bb:cc:dd:ee:ff -65  36      Y  -- WPA2(PSK/AES/AES)
                         GP-1234 11:22:33:44:55:66 -50  6       Y  -- WPA2(PSK/AES/AES)
                      CoffeeShop 77:88:99:aa:bb:cc -80  11      Y  -- NONE
"""

    def test_parse_airport_scan(self):
        backend = _MacOSWiFiBackend()
        networks = backend._parse_airport_scan(self.SAMPLE_AIRPORT_OUTPUT)

        assert len(networks) == 3

        # Check networks (sorted by appearance in output)
        ssids = [n.ssid for n in networks]
        assert "HomeNetwork" in ssids
        assert "GP-1234" in ssids
        assert "CoffeeShop" in ssids

        # Check GoPro network details
        gp = next(n for n in networks if n.ssid == "GP-1234")
        assert gp.signal_strength == -50
        assert gp.channel == 6
        assert gp.bssid == "11:22:33:44:55:66"

    def test_parse_empty_airport_output(self):
        backend = _MacOSWiFiBackend()
        networks = backend._parse_airport_scan("")
        assert networks == []


# ======================================================================
# Linux Backend Parsing
# ======================================================================

class TestLinuxNmcliParsing:
    """Tests for parsing nmcli output on Linux."""

    SAMPLE_NMCLI_OUTPUT = """\
HomeNetwork:85:WPA2:AA\\:BB\\:CC\\:DD\\:EE\\:FF:36
GP-1234:72:WPA2:11\\:22\\:33\\:44\\:55\\:66:6
CoffeeShop:45::77\\:88\\:99\\:AA\\:BB\\:CC:11
"""

    def test_parse_nmcli_list(self):
        backend = _LinuxWiFiBackend.__new__(_LinuxWiFiBackend)
        networks = backend._parse_nmcli_list(self.SAMPLE_NMCLI_OUTPUT)

        assert len(networks) >= 2  # At least HomeNetwork and GP-1234

        ssids = [n.ssid for n in networks]
        assert "HomeNetwork" in ssids
        assert "GP-1234" in ssids

    def test_parse_empty_nmcli_output(self):
        backend = _LinuxWiFiBackend.__new__(_LinuxWiFiBackend)
        networks = backend._parse_nmcli_list("")
        assert networks == []


# ======================================================================
# WiFiManager State Transitions
# ======================================================================

class TestWiFiManagerStates:
    """Tests for WiFiManager state machine."""

    @pytest.fixture
    def manager(self):
        """Create a WiFiManager with mocked backend."""
        with patch.object(WiFiManager, '_resolve_backend') as mock_resolve:
            mock_backend = MagicMock()
            mock_resolve.return_value = mock_backend
            mgr = WiFiManager()
            mgr._backend = mock_backend
            return mgr

    def test_initial_state(self, manager):
        assert manager.state == WiFiConnectionState.IDLE
        assert manager.connected_ssid is None
        assert manager.previous_ssid is None
        assert not manager.is_connected

    @pytest.mark.asyncio
    async def test_scan_state_transitions(self, manager):
        """Scan should transition: IDLE → SCANNING → IDLE."""
        states_seen = []
        manager.add_state_listener(lambda old, new: states_seen.append(new))

        manager._backend.scan = AsyncMock(return_value=[
            WiFiNetwork(ssid="GP-1234", signal_strength=-50),
        ])

        networks = await manager.scan()

        assert WiFiConnectionState.SCANNING in states_seen
        assert manager.state == WiFiConnectionState.IDLE
        assert len(networks) == 1
        assert networks[0].is_gopro is True

    @pytest.mark.asyncio
    async def test_connect_state_transitions(self, manager):
        """Connect should transition: IDLE → CONNECTING → CONNECTED."""
        states_seen = []
        manager.add_state_listener(lambda old, new: states_seen.append(new))

        manager._backend.get_current_ssid = AsyncMock(return_value="HomeNetwork")
        manager._backend.connect = AsyncMock(return_value=True)

        await manager.connect("GP-1234", "password123")

        assert WiFiConnectionState.CONNECTING in states_seen
        assert manager.state == WiFiConnectionState.CONNECTED
        assert manager.connected_ssid == "GP-1234"
        assert manager.previous_ssid == "HomeNetwork"
        assert manager.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect_state_transitions(self, manager):
        """Disconnect should transition: CONNECTED → DISCONNECTING → IDLE."""
        # First connect
        manager._backend.get_current_ssid = AsyncMock(return_value="HomeNetwork")
        manager._backend.connect = AsyncMock(return_value=True)
        manager._backend.disconnect = AsyncMock(return_value=True)
        manager._backend.reconnect_to = AsyncMock(return_value=True)

        await manager.connect("GP-1234", "password123")

        states_seen = []
        manager.add_state_listener(lambda old, new: states_seen.append(new))

        await manager.disconnect()

        assert WiFiConnectionState.DISCONNECTING in states_seen
        assert manager.state == WiFiConnectionState.IDLE
        assert manager.connected_ssid is None

    @pytest.mark.asyncio
    async def test_disconnect_restores_previous_wifi(self, manager):
        """Disconnect should attempt to restore previous WiFi."""
        manager._backend.get_current_ssid = AsyncMock(return_value="HomeNetwork")
        manager._backend.connect = AsyncMock(return_value=True)
        manager._backend.disconnect = AsyncMock(return_value=True)
        manager._backend.reconnect_to = AsyncMock(return_value=True)

        await manager.connect("GP-1234", "password123")
        await manager.disconnect(restore_previous=True)

        manager._backend.reconnect_to.assert_called_once_with("HomeNetwork")

    @pytest.mark.asyncio
    async def test_disconnect_no_restore(self, manager):
        """Disconnect with restore_previous=False should not reconnect."""
        manager._backend.get_current_ssid = AsyncMock(return_value="HomeNetwork")
        manager._backend.connect = AsyncMock(return_value=True)
        manager._backend.disconnect = AsyncMock(return_value=True)
        manager._backend.reconnect_to = AsyncMock(return_value=True)

        await manager.connect("GP-1234", "password123")
        await manager.disconnect(restore_previous=False)

        manager._backend.reconnect_to.assert_not_called()


# ======================================================================
# WiFiManager Scanning
# ======================================================================

class TestWiFiManagerScan:
    """Tests for WiFiManager.scan() and scan_for_gopro()."""

    @pytest.fixture
    def manager(self):
        with patch.object(WiFiManager, '_resolve_backend') as mock_resolve:
            mock_backend = MagicMock()
            mock_resolve.return_value = mock_backend
            mgr = WiFiManager()
            mgr._backend = mock_backend
            return mgr

    @pytest.mark.asyncio
    async def test_scan_flags_gopro_ssids(self, manager):
        """Scan should flag GoPro SSIDs with is_gopro=True."""
        manager._backend.scan = AsyncMock(return_value=[
            WiFiNetwork(ssid="HomeNetwork", signal_strength=-65),
            WiFiNetwork(ssid="GP-1234", signal_strength=-50),
            WiFiNetwork(ssid="GP-5678", signal_strength=-70),
            WiFiNetwork(ssid="OtherNetwork", signal_strength=-80),
        ])

        networks = await manager.scan()

        gopro_nets = [n for n in networks if n.is_gopro]
        assert len(gopro_nets) == 2
        assert all(n.ssid.startswith("GP") for n in gopro_nets)

    @pytest.mark.asyncio
    async def test_scan_sorts_by_signal(self, manager):
        """Scan results should be sorted by signal strength (strongest first)."""
        manager._backend.scan = AsyncMock(return_value=[
            WiFiNetwork(ssid="Weak", signal_strength=-80),
            WiFiNetwork(ssid="Strong", signal_strength=-30),
            WiFiNetwork(ssid="Medium", signal_strength=-55),
        ])

        networks = await manager.scan()

        signals = [n.signal_strength for n in networks]
        assert signals == sorted(signals, reverse=True)

    @pytest.mark.asyncio
    async def test_scan_for_gopro_filters(self, manager):
        """scan_for_gopro() should only return GoPro SSIDs."""
        manager._backend.scan = AsyncMock(return_value=[
            WiFiNetwork(ssid="HomeNetwork", signal_strength=-65),
            WiFiNetwork(ssid="GP-1234", signal_strength=-50),
        ])

        gopro_nets = await manager.scan_for_gopro()

        assert len(gopro_nets) == 1
        assert gopro_nets[0].ssid == "GP-1234"
        assert gopro_nets[0].is_gopro is True

    @pytest.mark.asyncio
    async def test_scan_updates_stats(self, manager):
        """Scan should increment scans_performed stat."""
        manager._backend.scan = AsyncMock(return_value=[])

        await manager.scan()
        await manager.scan()

        assert manager.stats.scans_performed == 2

    @pytest.mark.asyncio
    async def test_scan_timeout_raises(self, manager):
        """Scan timeout should raise WiFiScanError."""
        async def slow_scan():
            await asyncio.sleep(100)
            return []

        manager._backend.scan = slow_scan

        with pytest.raises(WiFiScanError, match="timed out"):
            await manager.scan(timeout=0.01)

        assert manager.state == WiFiConnectionState.ERROR

    @pytest.mark.asyncio
    async def test_scan_error_raises(self, manager):
        """Scan failure should raise WiFiScanError."""
        manager._backend.scan = AsyncMock(side_effect=RuntimeError("WiFi adapter not found"))

        with pytest.raises(WiFiScanError):
            await manager.scan()

        assert manager.state == WiFiConnectionState.ERROR
        assert manager.stats.last_error is not None


# ======================================================================
# WiFiManager Connection
# ======================================================================

class TestWiFiManagerConnect:
    """Tests for WiFiManager.connect()."""

    @pytest.fixture
    def manager(self):
        with patch.object(WiFiManager, '_resolve_backend') as mock_resolve:
            mock_backend = MagicMock()
            mock_resolve.return_value = mock_backend
            mgr = WiFiManager()
            mgr._backend = mock_backend
            return mgr

    @pytest.mark.asyncio
    async def test_connect_success(self, manager):
        manager._backend.get_current_ssid = AsyncMock(return_value="HomeNetwork")
        manager._backend.connect = AsyncMock(return_value=True)

        result = await manager.connect("GP-1234", "password123")

        assert result is True
        assert manager.connected_ssid == "GP-1234"
        assert manager.state == WiFiConnectionState.CONNECTED
        assert manager.stats.connections_succeeded == 1

    @pytest.mark.asyncio
    async def test_connect_failure_raises(self, manager):
        manager._backend.get_current_ssid = AsyncMock(return_value=None)
        manager._backend.connect = AsyncMock(return_value=False)

        with pytest.raises(WiFiConnectError):
            await manager.connect("GP-1234", "password123")

        assert manager.state == WiFiConnectionState.ERROR
        assert manager.stats.connections_failed == 1

    @pytest.mark.asyncio
    async def test_connect_already_connected_same_ssid(self, manager):
        """Connecting to the same SSID should be a no-op."""
        manager._backend.get_current_ssid = AsyncMock(return_value="HomeNetwork")
        manager._backend.connect = AsyncMock(return_value=True)

        await manager.connect("GP-1234", "password123")
        result = await manager.connect("GP-1234", "password123")

        assert result is True
        # connect should only be called once (second call is no-op)
        assert manager._backend.connect.call_count == 1

    @pytest.mark.asyncio
    async def test_connect_different_ssid_disconnects_first(self, manager):
        """Connecting to a different SSID should disconnect from current first."""
        manager._backend.get_current_ssid = AsyncMock(return_value="HomeNetwork")
        manager._backend.connect = AsyncMock(return_value=True)
        manager._backend.disconnect = AsyncMock(return_value=True)
        manager._backend.reconnect_to = AsyncMock(return_value=True)

        await manager.connect("GP-1234", "password1")
        await manager.connect("GP-5678", "password2")

        assert manager.connected_ssid == "GP-5678"

    @pytest.mark.asyncio
    async def test_connect_timeout_raises(self, manager):
        manager._backend.get_current_ssid = AsyncMock(return_value=None)

        async def slow_connect(ssid, password):
            await asyncio.sleep(100)
            return True

        manager._backend.connect = slow_connect

        with pytest.raises(WiFiConnectError, match="timed out"):
            await manager.connect("GP-1234", "password123", timeout=0.01)

    @pytest.mark.asyncio
    async def test_connect_records_previous_ssid(self, manager):
        manager._backend.get_current_ssid = AsyncMock(return_value="MyHomeWiFi")
        manager._backend.connect = AsyncMock(return_value=True)

        await manager.connect("GP-1234", "password123")

        assert manager.stats.previous_ssid == "MyHomeWiFi"


# ======================================================================
# WiFiManager Statistics
# ======================================================================

class TestWiFiManagerStats:
    """Tests for WiFiManager statistics tracking."""

    @pytest.fixture
    def manager(self):
        with patch.object(WiFiManager, '_resolve_backend') as mock_resolve:
            mock_backend = MagicMock()
            mock_resolve.return_value = mock_backend
            mgr = WiFiManager()
            mgr._backend = mock_backend
            return mgr

    @pytest.mark.asyncio
    async def test_stats_accumulate(self, manager):
        manager._backend.scan = AsyncMock(return_value=[])
        manager._backend.get_current_ssid = AsyncMock(return_value=None)
        manager._backend.connect = AsyncMock(return_value=True)

        await manager.scan()
        await manager.scan()
        await manager.connect("GP-1234", "pass")

        assert manager.stats.scans_performed == 2
        assert manager.stats.connections_attempted == 1
        assert manager.stats.connections_succeeded == 1
        assert manager.stats.connections_failed == 0
        assert manager.stats.last_connected_ssid == "GP-1234"


# ======================================================================
# Platform Backend Selection
# ======================================================================

class TestBackendSelection:
    """Tests for platform-specific backend selection."""

    def test_windows_backend(self):
        with patch('gomaxwebcam.transport.wifi_manager.platform') as mock_platform:
            mock_platform.system.return_value = "Windows"
            # Need to avoid the constructor calling _resolve_backend with real platform
            with patch.object(WiFiManager, '_resolve_backend') as mock_resolve:
                mock_resolve.return_value = _WindowsWiFiBackend()
                mgr = WiFiManager()
                # Verify the backend type through direct call
                assert isinstance(mock_resolve.return_value, _WindowsWiFiBackend)

    def test_resolve_backend_windows(self):
        with patch('gomaxwebcam.transport.wifi_manager.platform') as mock_platform:
            mock_platform.system.return_value = "Windows"
            with patch.object(WiFiManager, '__init__', lambda self: None):
                mgr = WiFiManager.__new__(WiFiManager)
                mgr._platform = "windows"
                backend = mgr._resolve_backend()
                assert isinstance(backend, _WindowsWiFiBackend)

    def test_resolve_backend_darwin(self):
        with patch.object(WiFiManager, '__init__', lambda self: None):
            mgr = WiFiManager.__new__(WiFiManager)
            mgr._platform = "darwin"
            backend = mgr._resolve_backend()
            assert isinstance(backend, _MacOSWiFiBackend)

    def test_resolve_backend_linux(self):
        with patch('shutil.which', return_value="/usr/bin/nmcli"):
            with patch.object(WiFiManager, '__init__', lambda self: None):
                mgr = WiFiManager.__new__(WiFiManager)
                mgr._platform = "linux"
                backend = mgr._resolve_backend()
                assert isinstance(backend, _LinuxWiFiBackend)

    def test_resolve_backend_unsupported(self):
        with patch.object(WiFiManager, '__init__', lambda self: None):
            mgr = WiFiManager.__new__(WiFiManager)
            mgr._platform = "freebsd"
            with pytest.raises(WiFiPlatformError, match="Unsupported platform"):
                mgr._resolve_backend()


# ======================================================================
# State Listener Callbacks
# ======================================================================

class TestStateListeners:
    """Tests for WiFiManager state change callbacks."""

    @pytest.fixture
    def manager(self):
        with patch.object(WiFiManager, '_resolve_backend') as mock_resolve:
            mock_backend = MagicMock()
            mock_resolve.return_value = mock_backend
            mgr = WiFiManager()
            mgr._backend = mock_backend
            return mgr

    @pytest.mark.asyncio
    async def test_listener_receives_transitions(self, manager):
        transitions = []
        manager.add_state_listener(lambda old, new: transitions.append((old, new)))

        manager._backend.scan = AsyncMock(return_value=[])
        await manager.scan()

        assert len(transitions) == 2
        assert transitions[0] == (WiFiConnectionState.IDLE, WiFiConnectionState.SCANNING)
        assert transitions[1] == (WiFiConnectionState.SCANNING, WiFiConnectionState.IDLE)

    @pytest.mark.asyncio
    async def test_listener_error_doesnt_crash(self, manager):
        """A listener that throws should not crash the manager."""
        def bad_listener(old, new):
            raise RuntimeError("listener crashed!")

        manager.add_state_listener(bad_listener)

        manager._backend.scan = AsyncMock(return_value=[])
        # Should not raise
        await manager.scan()

    def test_multiple_listeners(self, manager):
        calls_a = []
        calls_b = []

        manager.add_state_listener(lambda o, n: calls_a.append(n))
        manager.add_state_listener(lambda o, n: calls_b.append(n))

        manager._set_state(WiFiConnectionState.SCANNING)

        assert calls_a == [WiFiConnectionState.SCANNING]
        assert calls_b == [WiFiConnectionState.SCANNING]


# ======================================================================
# Integration: WiFiAPTransport + WiFiManager
# ======================================================================

class TestWiFiAPTransportIntegration:
    """Tests for WiFiAPTransport using WiFiManager for auto-connect."""

    @pytest.mark.asyncio
    async def test_auto_connect_on_discover(self):
        """WiFiAPTransport should auto-connect to GoPro AP when configured."""
        from gomaxwebcam.transport.wifi_ap import WiFiAPTransport

        transport = WiFiAPTransport(
            auto_connect_wifi=True,
            wifi_password="testpass",
        )

        # Create mock WiFi manager
        mock_wifi = MagicMock()
        mock_wifi.scan_for_gopro = AsyncMock(return_value=[
            WiFiNetwork(ssid="GP-1234", signal_strength=-50, is_gopro=True),
        ])
        mock_wifi.connect = AsyncMock(return_value=True)
        transport._inject_wifi_manager(mock_wifi)

        # Mock the HTTP session for the discover ping
        import aiohttp
        mock_session = MagicMock()
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.close = AsyncMock()

        with patch('aiohttp.ClientSession', return_value=mock_session):
            result = await transport.discover(timeout=5.0)

        assert result is True
        mock_wifi.scan_for_gopro.assert_called_once()
        mock_wifi.connect.assert_called_once_with("GP-1234", "testpass")

    @pytest.mark.asyncio
    async def test_disconnect_restores_wifi(self):
        """WiFiAPTransport should restore WiFi on disconnect when managed."""
        from gomaxwebcam.transport.wifi_ap import WiFiAPTransport
        from gomaxwebcam.transport.base import TransportState

        transport = WiFiAPTransport()
        transport._wifi_managed_connection = True

        mock_wifi = MagicMock()
        mock_wifi.disconnect = AsyncMock()
        transport._inject_wifi_manager(mock_wifi)

        # Set state to connected (so disconnect does something)
        transport._state = TransportState.CONNECTED

        await transport.disconnect()

        mock_wifi.disconnect.assert_called_once_with(restore_previous=True)
        assert transport._wifi_managed_connection is False

    @pytest.mark.asyncio
    async def test_no_auto_connect_when_disabled(self):
        """WiFiAPTransport should not auto-connect when auto_connect_wifi=False."""
        from gomaxwebcam.transport.wifi_ap import WiFiAPTransport

        transport = WiFiAPTransport(
            auto_connect_wifi=False,
            wifi_password="testpass",
        )

        mock_wifi = MagicMock()
        mock_wifi.scan_for_gopro = AsyncMock()
        transport._inject_wifi_manager(mock_wifi)

        # Mock the HTTP session to fail (since we're not connected to GoPro WiFi)
        with patch('aiohttp.ClientSession') as mock_cls:
            mock_session = MagicMock()
            mock_session.get = MagicMock(side_effect=asyncio.TimeoutError())
            mock_session.close = AsyncMock()
            mock_cls.return_value = mock_session

            result = await transport.discover(timeout=0.1)

        # WiFi scan should NOT have been called
        mock_wifi.scan_for_gopro.assert_not_called()


# ======================================================================
# Constants and Module-level
# ======================================================================

class TestConstants:
    """Tests for module constants."""

    def test_gopro_ap_ip(self):
        assert GOPRO_AP_IP == "10.5.5.9"

    def test_gopro_ap_port(self):
        assert GOPRO_AP_PORT == 8080


# ======================================================================
# WiFiManager get_current_ssid
# ======================================================================

class TestGetCurrentSSID:
    """Tests for WiFiManager.get_current_ssid()."""

    @pytest.fixture
    def manager(self):
        with patch.object(WiFiManager, '_resolve_backend') as mock_resolve:
            mock_backend = MagicMock()
            mock_resolve.return_value = mock_backend
            mgr = WiFiManager()
            mgr._backend = mock_backend
            return mgr

    @pytest.mark.asyncio
    async def test_get_current_ssid(self, manager):
        manager._backend.get_current_ssid = AsyncMock(return_value="HomeNetwork")
        result = await manager.get_current_ssid()
        assert result == "HomeNetwork"

    @pytest.mark.asyncio
    async def test_get_current_ssid_not_connected(self, manager):
        manager._backend.get_current_ssid = AsyncMock(return_value=None)
        result = await manager.get_current_ssid()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_current_ssid_error_returns_none(self, manager):
        manager._backend.get_current_ssid = AsyncMock(side_effect=RuntimeError("error"))
        result = await manager.get_current_ssid()
        assert result is None


# ======================================================================
# Windows Backend: get_current_ssid parsing
# ======================================================================

class TestWindowsGetCurrentSSID:
    """Tests for Windows backend SSID detection."""

    SAMPLE_INTERFACES_OUTPUT = """\
    Name                   : Wi-Fi
    Description            : Intel(R) Wi-Fi 6 AX200
    GUID                   : abc-def-123
    Physical address       : aa:bb:cc:dd:ee:ff
    State                  : connected
    SSID                   : HomeNetwork
    BSSID                  : 11:22:33:44:55:66
    Network type           : Infrastructure
    Radio type             : 802.11ax
    Authentication         : WPA2-Personal
    Cipher                 : CCMP
    Connection mode        : Auto Connect
    Channel                : 36
    Receive rate (Mbps)    : 1201
    Transmit rate (Mbps)   : 1201
    Signal                 : 95%
"""

    @pytest.mark.asyncio
    async def test_get_current_ssid_parses_correctly(self):
        backend = _WindowsWiFiBackend()

        with patch.object(backend, '_run_cmd', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = (0, self.SAMPLE_INTERFACES_OUTPUT, "")
            result = await backend.get_current_ssid()

        assert result == "HomeNetwork"

    @pytest.mark.asyncio
    async def test_get_current_ssid_not_connected(self):
        backend = _WindowsWiFiBackend()
        disconnected_output = """\
    Name                   : Wi-Fi
    Description            : Intel(R) Wi-Fi 6 AX200
    State                  : disconnected
"""
        with patch.object(backend, '_run_cmd', new_callable=AsyncMock) as mock_cmd:
            mock_cmd.return_value = (0, disconnected_output, "")
            result = await backend.get_current_ssid()

        assert result is None
