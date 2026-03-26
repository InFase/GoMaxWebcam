"""
dashboard/status_labels.py — Translate raw backend state names to user-friendly labels.

This module converts technical internal state strings (like "AUTH_FAILED",
"FREEZE_FRAME", "DISCONNECTED") into plain-English labels and actionable hints
that non-technical users can understand.

The same translations are mirrored in the frontend JavaScript (_computeStateInfo)
for resilience. When the backend provides labels, they take priority in the UI.

Exported functions:
    compute_status_info()   — primary state → label/hint translation
    friendly_error()        — sanitise raw Python exception strings
    friendly_camera_model() — format "GoPro@192.168.1.1" display strings
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Main translation function
# ---------------------------------------------------------------------------

def compute_status_info(
    connection_state: str,
    pipeline_state: str,
    last_error: str = "",
    is_frozen: bool = False,
) -> dict:
    """Compute human-readable label and hint from raw backend states.

    Maps the raw TransportState / PipelineState enum names that the backend
    sends over SSE into plain-English strings for the dashboard UI.

    Args:
        connection_state: Raw connection state name (e.g. "DISCONNECTED", "STREAMING").
        pipeline_state:   Raw pipeline state name (e.g. "STOPPED", "STREAMING").
        last_error:       Optional raw error message from the backend.
        is_frozen:        True when the pipeline is showing a freeze-frame.

    Returns:
        dict with three keys:
            state_label (str): Short status line shown prominently in the UI.
            state_hint  (str): One-sentence actionable hint; empty when all is well.
            ui_state    (str): CSS key for the coloured dot:
                               "streaming" | "connecting" | "freeze_frame" |
                               "error" | "paused"
    """
    conn = (connection_state or "").upper().strip()
    pipe = (pipeline_state or "").upper().strip()

    # ── No camera / idle ────────────────────────────────────────────────────
    if not conn or conn in ("DISCONNECTED", "NONE", ""):
        return _r(
            "Waiting for GoPro\u2026",
            "Turn on your camera and plug in a USB cable, or connect via Wi\u2011Fi.",
            "paused",
        )

    # ── Startup / discovery ─────────────────────────────────────────────────
    if conn in ("INITIALIZING", "DISCOVERY"):
        return _r(
            "Starting up\u2026",
            "Initialising connection to your GoPro \u2014 this only takes a moment.",
            "connecting",
        )

    # ── Connection attempts ─────────────────────────────────────────────────
    if conn in ("CONNECTING", "PROVISIONING"):
        return _r(
            "Connecting to camera\u2026",
            "Make sure the camera is powered on and within range.",
            "connecting",
        )

    if conn in ("RECONNECTING", "RETRYING"):
        return _r(
            "Reconnecting\u2026",
            "Lost connection \u2014 trying again automatically. No action needed.",
            "connecting",
        )

    if conn == "AUTHENTICATING":
        return _r(
            "Authenticating\u2026",
            "Verifying your GoPro credentials \u2014 this only takes a moment.",
            "connecting",
        )

    # ── User-initiated pause ────────────────────────────────────────────────
    if conn in ("PAUSED", "PAUSED_USER"):
        return _r(
            "Stream paused",
            "Press \u00ABP\u00BB or click Resume to start the live stream again.",
            "paused",
        )

    # ── Credential / auth errors ────────────────────────────────────────────
    if conn == "AUTH_FAILED":
        return _r(
            "Authentication failed",
            "Your camera rejected the credentials. "
            "Re-run the Setup wizard to fix this.",
            "error",
        )

    if conn == "NO_CREDENTIALS":
        return _r(
            "Camera credentials not found",
            "Go to the Setup page and run the COHN Setup Wizard to configure your camera.",
            "error",
        )

    if conn == "TIMEOUT":
        return _r(
            "Connection timed out",
            "The camera took too long to respond. "
            "Check it is powered on and press \u00ABR\u00BB to reconnect.",
            "error",
        )

    # ── Generic error / failed ──────────────────────────────────────────────
    if conn in ("ERROR", "FAILED"):
        hint = (
            friendly_error(last_error)
            if last_error
            else "Check that your camera is on and reachable, then press \u00ABR\u00BB to reconnect."
        )
        return _r("Connection error", hint, "error")

    # ── Connected / streaming ───────────────────────────────────────────────
    if conn in ("STREAMING", "CONNECTED"):
        # Freeze-frame: pipeline is holding last frame during transport switch
        if is_frozen or pipe == "FREEZE_FRAME":
            return _r(
                "Switching connection\u2026",
                "Brief pause while the app switches to the best available connection. "
                "This takes only a few seconds.",
                "freeze_frame",
            )

        if pipe == "STREAMING":
            return _r("Live \u2014 Streaming", "", "streaming")

        if pipe in ("STARTING", "STOPPED", "INITIALIZING"):
            return _r(
                "Starting video pipeline\u2026",
                "Connected to camera \u2014 setting up the video stream.",
                "connecting",
            )

        if pipe == "PAUSED":
            return _r(
                "Stream paused",
                "Camera connected. Press \u00ABP\u00BB to resume live streaming.",
                "paused",
            )

        if pipe == "ERROR":
            return _r(
                "Video pipeline error",
                "The video stream stopped unexpectedly. Press \u00ABR\u00BB to reconnect.",
                "error",
            )

        # Connected but unknown pipeline state — probably still starting
        return _r(
            "Camera connected",
            "Preparing video stream\u2026",
            "connecting",
        )

    # ── Catch-all: prettify raw state name ──────────────────────────────────
    pretty = conn.replace("_", " ").capitalize()
    return _r(
        pretty,
        "Unexpected state. Try reconnecting (\u00ABR\u00BB) if this persists.",
        "paused",
    )


# ---------------------------------------------------------------------------
# Error sanitisation
# ---------------------------------------------------------------------------

def friendly_error(raw_error: str) -> str:
    """Convert a technical Python exception string to a user-friendly message.

    Inspects common error patterns (Windows error codes, POSIX errno strings,
    HTTP status codes) and returns a plain-English sentence.

    Args:
        raw_error: Raw exception/error string from the backend.

    Returns:
        Short, non-technical explanation suitable for display in the dashboard.
    """
    if not raw_error:
        return (
            "Check that your camera is on and reachable, "
            "then press \u00ABR\u00BB to reconnect."
        )

    err = raw_error.lower()

    # Network / connectivity
    if any(k in err for k in ("connection refused", "winerror 10061", "econnrefused")):
        return "Unable to reach the camera \u2014 check USB or Wi\u2011Fi connection."

    if any(k in err for k in ("timed out", "timeout", "etimedout")):
        return "Camera took too long to respond \u2014 check it is powered on."

    if any(k in err for k in ("network unreachable", "no route to host", "enetunreach")):
        return "Network unreachable \u2014 check your Wi\u2011Fi or USB cable."

    if any(k in err for k in ("connection reset", "econnreset", "broken pipe")):
        return "Connection was interrupted \u2014 trying to reconnect automatically."

    if any(k in err for k in ("name or service not known", "name resolution")):
        return "Camera address not found \u2014 check your network settings."

    # Auth / credential errors
    if any(k in err for k in ("401", "unauthorized", "credentials", "password")):
        return "Authentication failed \u2014 re-run the Setup wizard to refresh credentials."

    if "403" in err or "forbidden" in err:
        return "Access denied \u2014 check your camera credentials in the Setup page."

    # Hardware / OS errors
    if any(k in err for k in ("no such device", "device not found", "errno 19")):
        return "Camera not detected \u2014 try unplugging and replugging the USB cable."

    if any(k in err for k in ("permission denied", "access is denied", "errno 13")):
        return "Permission denied \u2014 another app may be using the camera."

    if any(k in err for k in ("oserror", "ioerror", "winerror")):
        return "System error communicating with camera \u2014 try reconnecting (\u00ABR\u00BB)."

    # Generic fallback
    return (
        "Connection error \u2014 check your camera is on "
        "and try reconnecting (\u00ABR\u00BB)."
    )


# ---------------------------------------------------------------------------
# Camera model display
# ---------------------------------------------------------------------------

def friendly_camera_model(raw_model: str) -> str:
    """Format a raw camera model/address string for human-friendly display.

    Converts technical strings like "GoPro@192.168.1.50" into
    "GoPro (192.168.1.50)".

    Args:
        raw_model: Raw model string from the backend transport.

    Returns:
        Formatted display string, or empty string if input is empty/None.
    """
    if not raw_model:
        return ""

    # "GoPro@192.168.1.50" → "GoPro (192.168.1.50)"
    if "@" in raw_model:
        name, _, addr = raw_model.partition("@")
        name = name.strip()
        addr = addr.strip()
        if addr:
            return f"{name} ({addr})"
        return name

    return raw_model


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _r(label: str, hint: str, ui_state: str) -> dict:
    """Build a status info dict."""
    return {"state_label": label, "state_hint": hint, "ui_state": ui_state}
