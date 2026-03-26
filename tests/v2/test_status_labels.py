"""
tests/v2/test_status_labels.py — Unit tests for dashboard status label translation.

Verifies that compute_status_info(), friendly_error(), and friendly_camera_model()
produce correct, non-technical, user-friendly output for every meaningful backend
state combination.

All tests are pure-Python — no hardware, no network, no event loop required.
"""

import pytest

from gomaxwebcam.dashboard.status_labels import (
    compute_status_info,
    friendly_camera_model,
    friendly_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _info(conn, pipe="STOPPED", error="", frozen=False):
    """Shorthand wrapper around compute_status_info."""
    return compute_status_info(conn, pipe, error, frozen)


# ---------------------------------------------------------------------------
# compute_status_info — idle / disconnected states
# ---------------------------------------------------------------------------

class TestDisconnectedStates:
    """No camera detected."""

    @pytest.mark.parametrize("state", ["DISCONNECTED", "NONE", "", None])
    def test_disconnected_returns_paused_ui_state(self, state):
        info = _info(state or "")
        assert info["ui_state"] == "paused"

    def test_disconnected_label_is_waiting(self):
        info = _info("DISCONNECTED")
        assert "GoPro" in info["state_label"]
        # Must NOT be a raw enum name
        assert "DISCONNECTED" not in info["state_label"]

    def test_disconnected_hint_mentions_usb_or_wifi(self):
        info = _info("DISCONNECTED")
        hint = info["state_hint"].lower()
        assert "usb" in hint or "wi-fi" in hint or "wifi" in hint


# ---------------------------------------------------------------------------
# compute_status_info — startup / discovery
# ---------------------------------------------------------------------------

class TestStartupStates:

    @pytest.mark.parametrize("state", ["INITIALIZING", "DISCOVERY"])
    def test_startup_ui_state_is_connecting(self, state):
        assert _info(state)["ui_state"] == "connecting"

    @pytest.mark.parametrize("state", ["INITIALIZING", "DISCOVERY"])
    def test_startup_label_is_friendly(self, state):
        label = _info(state)["state_label"]
        assert state not in label  # no raw enum name
        assert label  # not empty


# ---------------------------------------------------------------------------
# compute_status_info — active connection states
# ---------------------------------------------------------------------------

class TestConnectingStates:

    @pytest.mark.parametrize("state", ["CONNECTING", "PROVISIONING"])
    def test_connecting_ui_state(self, state):
        assert _info(state)["ui_state"] == "connecting"

    @pytest.mark.parametrize("state", ["RECONNECTING", "RETRYING"])
    def test_reconnecting_ui_state(self, state):
        assert _info(state)["ui_state"] == "connecting"

    def test_authenticating_ui_state(self):
        assert _info("AUTHENTICATING")["ui_state"] == "connecting"

    @pytest.mark.parametrize("state", [
        "CONNECTING", "PROVISIONING", "RECONNECTING",
        "RETRYING", "AUTHENTICATING",
    ])
    def test_connecting_labels_are_human_readable(self, state):
        info = _info(state)
        # Label must not contain the raw enum name
        assert state not in info["state_label"]
        # Label must be a non-empty string
        assert info["state_label"].strip()


# ---------------------------------------------------------------------------
# compute_status_info — user-pause states
# ---------------------------------------------------------------------------

class TestPauseStates:

    @pytest.mark.parametrize("state", ["PAUSED", "PAUSED_USER"])
    def test_paused_ui_state(self, state):
        assert _info(state)["ui_state"] == "paused"

    @pytest.mark.parametrize("state", ["PAUSED", "PAUSED_USER"])
    def test_paused_label_mentions_paused(self, state):
        assert "pause" in _info(state)["state_label"].lower()

    @pytest.mark.parametrize("state", ["PAUSED", "PAUSED_USER"])
    def test_paused_hint_mentions_resume(self, state):
        assert "resume" in _info(state)["state_hint"].lower() or \
               "p" in _info(state)["state_hint"].lower()


# ---------------------------------------------------------------------------
# compute_status_info — error states
# ---------------------------------------------------------------------------

class TestErrorStates:

    def test_auth_failed_ui_state(self):
        assert _info("AUTH_FAILED")["ui_state"] == "error"

    def test_auth_failed_label_is_friendly(self):
        label = _info("AUTH_FAILED")["state_label"]
        assert "AUTH_FAILED" not in label
        assert label.strip()

    def test_auth_failed_hint_mentions_wizard(self):
        hint = _info("AUTH_FAILED")["state_hint"].lower()
        assert "setup" in hint or "wizard" in hint or "credential" in hint

    def test_no_credentials_ui_state(self):
        assert _info("NO_CREDENTIALS")["ui_state"] == "error"

    def test_no_credentials_hint_mentions_setup(self):
        hint = _info("NO_CREDENTIALS")["state_hint"].lower()
        assert "setup" in hint or "wizard" in hint

    def test_timeout_ui_state(self):
        assert _info("TIMEOUT")["ui_state"] == "error"

    def test_timeout_label_is_friendly(self):
        label = _info("TIMEOUT")["state_label"]
        assert "TIMEOUT" not in label
        assert label.strip()

    @pytest.mark.parametrize("state", ["ERROR", "FAILED"])
    def test_generic_error_ui_state(self, state):
        assert _info(state)["ui_state"] == "error"

    @pytest.mark.parametrize("state", ["ERROR", "FAILED"])
    def test_generic_error_uses_provided_last_error(self, state):
        """When last_error is provided, hint must not be the generic fallback."""
        info = _info(state, error="Connection refused: [WinError 10061]")
        # The hint should come from friendly_error(), not be the raw exception
        assert "WinError 10061" not in info["state_hint"]
        assert info["state_hint"].strip()

    @pytest.mark.parametrize("state", ["ERROR", "FAILED"])
    def test_generic_error_fallback_hint(self, state):
        """When no last_error, hint gives generic reconnect guidance."""
        info = _info(state)
        assert info["state_hint"].strip()


# ---------------------------------------------------------------------------
# compute_status_info — streaming / connected states
# ---------------------------------------------------------------------------

class TestStreamingStates:

    def test_streaming_with_streaming_pipeline(self):
        info = _info("STREAMING", pipe="STREAMING")
        assert info["ui_state"] == "streaming"
        assert info["state_label"]
        assert "STREAMING" not in info["state_label"]

    def test_streaming_label_says_live(self):
        info = _info("STREAMING", pipe="STREAMING")
        assert "live" in info["state_label"].lower() or \
               "stream" in info["state_label"].lower()

    def test_streaming_hint_is_empty_when_live(self):
        info = _info("STREAMING", pipe="STREAMING")
        assert info["state_hint"] == ""

    def test_freeze_frame_via_is_frozen(self):
        info = _info("STREAMING", pipe="STREAMING", frozen=True)
        assert info["ui_state"] == "freeze_frame"
        assert "FREEZE_FRAME" not in info["state_label"]

    def test_freeze_frame_via_pipeline_state(self):
        info = _info("CONNECTED", pipe="FREEZE_FRAME")
        assert info["ui_state"] == "freeze_frame"
        assert info["state_hint"].strip()  # explains what's happening

    def test_freeze_frame_hint_mentions_seconds_or_connection(self):
        info = _info("STREAMING", pipe="FREEZE_FRAME")
        hint = info["state_hint"].lower()
        assert "second" in hint or "connection" in hint or "switch" in hint

    @pytest.mark.parametrize("pipe", ["STARTING", "STOPPED", "INITIALIZING"])
    def test_starting_pipeline_is_connecting(self, pipe):
        info = _info("STREAMING", pipe=pipe)
        assert info["ui_state"] == "connecting"

    def test_paused_pipeline_is_paused(self):
        info = _info("STREAMING", pipe="PAUSED")
        assert info["ui_state"] == "paused"

    def test_errored_pipeline_is_error(self):
        info = _info("STREAMING", pipe="ERROR")
        assert info["ui_state"] == "error"

    def test_unknown_pipeline_state_still_friendly(self):
        info = _info("CONNECTED", pipe="UNKNOWN_FUTURE_STATE")
        # Should be connecting (preparing) or similar — not expose raw name
        assert info["ui_state"] in ("connecting", "streaming", "paused", "error")
        assert "UNKNOWN_FUTURE_STATE" not in info["state_label"]

    def test_connected_state_treated_same_as_streaming(self):
        info = _info("CONNECTED", pipe="STREAMING")
        assert info["ui_state"] == "streaming"


# ---------------------------------------------------------------------------
# compute_status_info — catch-all / unknown states
# ---------------------------------------------------------------------------

class TestFallbackState:

    def test_unknown_conn_state_returns_prettified_label(self):
        info = _info("SOME_NEW_FUTURE_STATE")
        # Should not crash and should not expose raw ALL_CAPS name verbatim
        assert info["state_label"]
        assert info["ui_state"] in ("paused", "connecting", "error", "streaming", "freeze_frame")

    def test_unknown_conn_state_includes_reconnect_hint(self):
        info = _info("SOME_NEW_FUTURE_STATE")
        hint = info["state_hint"].lower()
        assert "reconnect" in hint or "try" in hint


# ---------------------------------------------------------------------------
# friendly_error — error string sanitisation
# ---------------------------------------------------------------------------

class TestFriendlyError:

    def test_empty_string_returns_default_hint(self):
        result = friendly_error("")
        assert result.strip()
        assert "WinError" not in result

    def test_none_input_handled(self):
        # friendly_error expects str; empty string is the safe form
        result = friendly_error("")
        assert isinstance(result, str)

    @pytest.mark.parametrize("raw,expected_fragment", [
        ("[WinError 10061] No connection could be made", "reach"),
        ("Connection refused: ECONNREFUSED", "reach"),
        ("Request timed out: ETIMEDOUT", "too long"),
        ("Network unreachable: ENETUNREACH", "network"),
        ("Connection reset by peer", "interrupted"),
        ("401 Unauthorized: bad credentials", "authentication"),
        ("No such device: errno 19", "detected"),
        ("PermissionError: Access is denied", "permission"),
        ("OSError: [WinError 5]", "system"),
    ])
    def test_known_error_patterns(self, raw, expected_fragment):
        result = friendly_error(raw).lower()
        assert expected_fragment in result

    @pytest.mark.parametrize("raw", [
        "[WinError 10061] No connection",
        "Request timed out",
        "Connection refused",
    ])
    def test_no_raw_exception_class_in_output(self, raw):
        result = friendly_error(raw)
        assert "WinError" not in result
        assert "ETIMEDOUT" not in result
        assert "ECONNREFUSED" not in result

    def test_generic_fallback_for_unknown_error(self):
        result = friendly_error("some totally unexpected error zxqw")
        assert result.strip()
        # Must be plain English, not a Python traceback
        assert "Traceback" not in result


# ---------------------------------------------------------------------------
# friendly_camera_model — display formatting
# ---------------------------------------------------------------------------

class TestFriendlyCameraModel:

    def test_empty_returns_empty(self):
        assert friendly_camera_model("") == ""

    def test_none_equivalent_empty_string(self):
        assert friendly_camera_model("") == ""

    def test_at_sign_formatted_as_parentheses(self):
        result = friendly_camera_model("GoPro@192.168.1.50")
        assert result == "GoPro (192.168.1.50)"

    def test_at_sign_with_spaces_trimmed(self):
        result = friendly_camera_model("GoPro @ 172.20.10.1")
        assert "(172.20.10.1)" in result

    def test_plain_model_name_returned_unchanged(self):
        result = friendly_camera_model("HERO13 Black")
        assert result == "HERO13 Black"

    def test_ip_only_after_at(self):
        result = friendly_camera_model("Camera@10.0.0.1")
        assert result == "Camera (10.0.0.1)"

    def test_name_without_address(self):
        # Edge case: "@" present but nothing after it
        result = friendly_camera_model("GoPro@")
        # Should return just the name part without empty parens
        assert result == "GoPro"

    def test_no_raw_at_sign_in_output(self):
        result = friendly_camera_model("GoPro@192.168.1.1")
        assert "@" not in result


# ---------------------------------------------------------------------------
# Integration: to_sse_data includes friendly labels
# ---------------------------------------------------------------------------

class TestStatusTrackerIntegration:
    """Verify that CameraStatus.to_sse_data() embeds friendly labels."""

    import json as _json

    def test_sse_data_includes_state_label(self):
        import json
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus(connection_state="DISCONNECTED", pipeline_state="STOPPED")
        data = json.loads(status.to_sse_data())

        assert "state_label" in data
        assert data["state_label"]  # non-empty
        assert "DISCONNECTED" not in data["state_label"]

    def test_sse_data_includes_state_hint(self):
        import json
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus(connection_state="DISCONNECTED", pipeline_state="STOPPED")
        data = json.loads(status.to_sse_data())

        assert "state_hint" in data

    def test_sse_data_includes_ui_state(self):
        import json
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus(connection_state="DISCONNECTED")
        data = json.loads(status.to_sse_data())

        assert "ui_state" in data
        assert data["ui_state"] in (
            "streaming", "connecting", "freeze_frame", "error", "paused"
        )

    def test_streaming_status_ui_state(self):
        import json
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus(
            connection_state="STREAMING",
            pipeline_state="STREAMING",
        )
        data = json.loads(status.to_sse_data())
        assert data["ui_state"] == "streaming"
        assert data["state_hint"] == ""

    def test_camera_model_display_field(self):
        import json
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus(
            connection_state="STREAMING",
            pipeline_state="STREAMING",
            camera_model="GoPro@192.168.1.50",
        )
        data = json.loads(status.to_sse_data())
        assert "camera_model_display" in data
        assert data["camera_model_display"] == "GoPro (192.168.1.50)"

    def test_error_state_uses_last_error_for_hint(self):
        import json
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus(
            connection_state="ERROR",
            last_error="Connection refused: [WinError 10061]",
        )
        data = json.loads(status.to_sse_data())
        # The hint must be friendly — not expose the raw WinError text
        assert "WinError 10061" not in data["state_hint"]
        assert data["state_hint"].strip()

    def test_existing_fields_still_present(self):
        """Adding new fields must not break existing transport_type / connection_state."""
        import json
        from gomaxwebcam.dashboard.status_tracker import CameraStatus

        status = CameraStatus(
            transport_type="USB",
            connection_state="STREAMING",
            battery_level=75,
        )
        data = json.loads(status.to_sse_data())
        assert data["transport_type"] == "USB"
        assert data["connection_state"] == "STREAMING"
        assert data["battery_level"] == 75
