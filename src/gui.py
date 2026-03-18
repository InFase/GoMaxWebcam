"""
gui.py — Main GUI for GoPro Bridge (PyQt6-based)

Provides:
  - DashboardWindow: Main application window with status display and controls
  - RetryButton: Manual retry widget that appears after discovery timeout
  - ManualModeGuide: Fallback step-by-step instructions shown when programmatic
    webcam mode switching fails, guiding the user through manual GoPro setup
  - System tray integration (placeholder for future sub-ACs)

The GUI communicates with the AppController via callbacks and Qt signals.
All AppController callbacks fire from background threads, so we use
Qt signals to safely update the UI from the main thread.

Thread safety:
  - AppController callbacks → Qt signals → slot methods on the main thread
  - RetryButton click → controller.retry_connection() on a background thread
  - GUI never blocks on network operations
"""

import sys
import threading
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont, QColor, QPalette, QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSystemTrayIcon,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app_controller import AppController, AppState
from config import Config
from logger import get_logger

log = get_logger("gui")


# ── Style constants ──────────────────────────────────────────────────────────

# Colors for state indicators
STATE_COLORS = {
    AppState.INITIALIZING: "#888888",
    AppState.CHECKING_PREREQUISITES: "#888888",
    AppState.DISCOVERING: "#2196F3",       # Blue
    AppState.CONNECTING: "#FF9800",        # Orange
    AppState.STREAMING: "#4CAF50",         # Green
    AppState.PAUSED: "#FF9800",            # Orange
    AppState.CHARGE_MODE: "#9C27B0",       # Purple
    AppState.RECONNECTING: "#FF9800",      # Orange
    AppState.DISCONNECTED: "#F44336",      # Red
    AppState.ERROR: "#F44336",             # Red
    AppState.STOPPED: "#888888",           # Gray
}

STATE_LABELS = {
    AppState.INITIALIZING: "Initializing…",
    AppState.CHECKING_PREREQUISITES: "Checking prerequisites…",
    AppState.DISCOVERING: "Searching for GoPro…",
    AppState.CONNECTING: "Connecting…",
    AppState.STREAMING: "Streaming",
    AppState.PAUSED: "Paused",
    AppState.CHARGE_MODE: "Charging",
    AppState.RECONNECTING: "Reconnecting…",
    AppState.DISCONNECTED: "Disconnected",
    AppState.ERROR: "Error",
    AppState.STOPPED: "Stopped",
}

# States where the retry button should be visible
RETRY_VISIBLE_STATES = {AppState.ERROR, AppState.DISCONNECTED, AppState.STOPPED}


# ── RetryButton widget ──────────────────────────────────────────────────────

class RetryButton(QWidget):
    """A prominent retry button that appears after discovery timeout.

    Shows a styled button with a brief explanation, designed to be hidden
    by default and shown only when discovery has failed and user
    intervention may help (e.g., after checking the USB cable).

    Signals:
        retry_requested: Emitted when the user clicks Retry.
    """

    retry_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._build_ui()
        # Hidden by default — shown after discovery timeout
        self.setVisible(False)

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        # Explanation label
        self._label = QLabel(
            "GoPro not found. Check that the camera is connected\n"
            "via USB-C and powered on, then try again."
        )
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setStyleSheet("color: #9E9E9E; font-size: 12px;")

        # The retry button itself
        self._button = QPushButton("Retry Discovery")
        self._button.setMinimumHeight(44)
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 24px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
            QPushButton:disabled {
                background-color: #1e2d4a;
                color: #445566;
            }
        """)
        self._button.clicked.connect(self._on_click)

        layout.addWidget(self._label)
        layout.addSpacing(8)
        layout.addWidget(self._button)

    def _on_click(self):
        """Handle button click — emit signal and disable to prevent spam."""
        self._button.setEnabled(False)
        self._button.setText("Retrying…")
        self.retry_requested.emit()

    def show_retry(self, message: Optional[str] = None):
        """Show the retry button with an optional custom message.

        Args:
            message: Custom explanation text. If None, uses the default.
        """
        if message:
            self._label.setText(message)
        self._button.setEnabled(True)
        self._button.setText("Retry Discovery")
        self.setVisible(True)

    def hide_retry(self):
        """Hide the retry button (e.g., when discovery starts again)."""
        self.setVisible(False)

    def set_retrying(self):
        """Set the button to a 'retrying' state (disabled with spinner text)."""
        self._button.setEnabled(False)
        self._button.setText("Retrying…")


# ── ManualModeGuide widget ─────────────────────────────────────────────────

# Step-by-step manual instructions for switching GoPro to webcam mode.
# Shown as a fallback when programmatic switching via the HTTP API fails.
MANUAL_MODE_STEPS = [
    ("Power cycle the GoPro", "Turn the camera OFF, wait 5 seconds, then turn it back ON."),
    ("Check the USB-C cable", "Ensure the cable is firmly connected at both ends. Try a different USB-C port if available."),
    ("Check USB Preferences", 'On the GoPro: swipe down \u2192 Preferences \u2192 Connections \u2192 USB Connection. If the option exists, select "GoPro Connect". (Some models may not have this setting \u2014 that\'s OK, skip this step.)'),
    ("Wait for USB Connected", 'The GoPro screen should display "USB Connected" or show the webcam icon once the link is established.'),
    ("Click Retry below", "Once the camera shows USB Connected, click Retry to let GoPro Bridge reconnect."),
]


class ManualModeGuide(QWidget):
    """A collapsible panel showing step-by-step instructions for manually
    switching the GoPro to webcam mode.

    Displayed as a fallback when programmatic switching via the HTTP API fails
    (e.g., the camera is in an unexpected state, firmware quirk, or the USB
    connection mode isn't set correctly).

    The guide is hidden by default and shown via show_guide() / hidden via
    hide_guide(). It can be collapsed/expanded by the user.

    Attributes:
        steps: List of (title, detail) tuples for each instruction step.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.steps = MANUAL_MODE_STEPS
        self._expanded = True
        self._build_ui()
        self.setVisible(False)

    def _build_ui(self):
        """Build the guide layout with header and numbered steps."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(6)

        # ── Header with collapse toggle ──
        header_layout = QHBoxLayout()
        header_layout.setSpacing(8)

        self._collapse_btn = QPushButton("▼")
        self._collapse_btn.setFixedSize(24, 24)
        self._collapse_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._collapse_btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: #FFB74D;
                border: none;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { color: #FFA726; }
        """)
        self._collapse_btn.clicked.connect(self._toggle_collapse)

        self._header_label = QLabel("Manual Setup Required")
        self._header_label.setStyleSheet(
            "color: #FFB74D; font-size: 13px; font-weight: bold;"
        )

        header_layout.addWidget(self._collapse_btn)
        header_layout.addWidget(self._header_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # ── Description ──
        self._desc_label = QLabel(
            "Automatic webcam mode switching failed. Follow these steps\n"
            "to manually prepare your GoPro, then retry:"
        )
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet("color: #B0B0B0; font-size: 11px; margin-bottom: 4px;")
        layout.addWidget(self._desc_label)

        # ── Steps container ──
        self._steps_container = QWidget()
        steps_layout = QVBoxLayout(self._steps_container)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(4)

        self._step_labels = []
        for i, (title, detail) in enumerate(self.steps, start=1):
            step_widget = self._create_step_widget(i, title, detail)
            steps_layout.addWidget(step_widget)
            self._step_labels.append(step_widget)

        layout.addWidget(self._steps_container)

        # ── Outer styling ──
        self.setStyleSheet("""
            ManualModeGuide {
                background-color: #1e2818;
                border: 1px solid #3a4a2a;
                border-radius: 8px;
            }
        """)

    def _create_step_widget(self, number: int, title: str, detail: str) -> QWidget:
        """Create a single numbered step with title and detail text."""
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(8)

        # Step number circle
        num_label = QLabel(str(number))
        num_label.setFixedSize(22, 22)
        num_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        num_label.setStyleSheet("""
            background-color: #5C4A2A;
            color: #FFB74D;
            border-radius: 11px;
            font-size: 11px;
            font-weight: bold;
        """)

        # Text content
        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(1)

        title_label = QLabel(f"<b>{title}</b>")
        title_label.setStyleSheet("color: #E0E0E0; font-size: 11px;")

        detail_label = QLabel(detail)
        detail_label.setWordWrap(True)
        detail_label.setStyleSheet("color: #909090; font-size: 10px;")

        text_layout.addWidget(title_label)
        text_layout.addWidget(detail_label)

        layout.addWidget(num_label, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(text_layout, 1)

        return widget

    def _toggle_collapse(self):
        """Toggle the expanded/collapsed state of the steps."""
        self._expanded = not self._expanded
        self._steps_container.setVisible(self._expanded)
        self._desc_label.setVisible(self._expanded)
        self._collapse_btn.setText("▼" if self._expanded else "▶")

    # ── Public API ──

    def show_guide(self, reason: Optional[str] = None):
        """Show the manual mode guide with an optional custom reason.

        Args:
            reason: Custom description of why manual setup is needed.
                    If None, uses the default description.
        """
        if reason:
            self._desc_label.setText(reason)
        else:
            self._desc_label.setText(
                "Automatic webcam mode switching failed. Follow these steps\n"
                "to manually prepare your GoPro, then retry:"
            )
        # Reset to expanded state when shown
        self._expanded = True
        self._steps_container.setVisible(True)
        self._desc_label.setVisible(True)
        self._collapse_btn.setText("▼")
        self.setVisible(True)
        log.info("[EVENT:manual_guide] Manual mode guide shown (reason: %s)",
                 reason[:60] if reason else "default")

    def hide_guide(self):
        """Hide the manual mode guide."""
        self.setVisible(False)

    @property
    def is_expanded(self) -> bool:
        """Whether the steps are currently expanded (visible)."""
        return self._expanded

    def step_count(self) -> int:
        """Return the number of instruction steps."""
        return len(self.steps)


# ── SettingsPanel widget ──────────────────────────────────────────────────

# Resolution options: code → display label
# Resolution codes vary by GoPro model. The app auto-remaps if a code is rejected.
# We show the options the camera actually supports.
_RESOLUTION_OPTIONS = [
    (7,  "720p"),
    (12, "1080p"),
]

# FOV options: code → display label
_FOV_OPTIONS = [
    (0, "Wide"),
    (4, "Linear"),
    (2, "Narrow"),
    (3, "Superview"),
]

# Anti-flicker options: GoPro setting 134
_ANTI_FLICKER_OPTIONS = [
    (0, "60Hz (NTSC)"),
    (1, "50Hz (PAL)"),
]


class SettingsPanel(QWidget):
    """Collapsible settings panel for adjusting GoPro Bridge configuration.

    Changes are saved to config.json immediately. Resolution/FOV changes
    are applied live if the camera is currently streaming.
    """

    def __init__(self, config: Config, controller: AppController,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self._controller = controller
        self._expanded = False
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 12)
        layout.setSpacing(10)

        # ── Section: Camera ──
        layout.addWidget(self._section_label("Camera"))

        # Resolution
        res_row = QHBoxLayout()
        res_row.addWidget(self._field_label("Resolution"))
        self._resolution_combo = QComboBox()
        for code, label in _RESOLUTION_OPTIONS:
            self._resolution_combo.addItem(label, code)
        self._set_combo_value(self._resolution_combo, self._config.resolution)
        self._resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self._style_combo(self._resolution_combo)
        res_row.addWidget(self._resolution_combo, 1)
        layout.addLayout(res_row)

        # FOV
        fov_row = QHBoxLayout()
        fov_row.addWidget(self._field_label("Field of View"))
        self._fov_combo = QComboBox()
        for code, label in _FOV_OPTIONS:
            self._fov_combo.addItem(label, code)
        self._set_combo_value(self._fov_combo, self._config.fov)
        self._fov_combo.currentIndexChanged.connect(self._on_fov_changed)
        self._style_combo(self._fov_combo)
        fov_row.addWidget(self._fov_combo, 1)
        layout.addLayout(fov_row)

        # Anti-Flicker
        flicker_row = QHBoxLayout()
        flicker_row.addWidget(self._field_label("Anti-Flicker"))
        self._flicker_combo = QComboBox()
        for code, label in _ANTI_FLICKER_OPTIONS:
            self._flicker_combo.addItem(label, code)
        self._set_combo_value(self._flicker_combo, self._config.anti_flicker)
        self._flicker_combo.currentIndexChanged.connect(self._on_flicker_changed)
        self._style_combo(self._flicker_combo)
        flicker_row.addWidget(self._flicker_combo, 1)
        layout.addLayout(flicker_row)

        # ── Section: Stream ──
        layout.addSpacing(4)
        layout.addWidget(self._section_label("Stream"))

        # UDP Port
        port_row = QHBoxLayout()
        port_row.addWidget(self._field_label("UDP Port"))
        self._port_spin = QSpinBox()
        self._port_spin.setRange(1024, 65535)
        self._port_spin.setValue(self._config.udp_port)
        self._port_spin.editingFinished.connect(self._on_port_changed)
        self._style_spin(self._port_spin)
        port_row.addWidget(self._port_spin, 1)
        layout.addLayout(port_row)

        # Virtual Camera Name
        vcam_row = QHBoxLayout()
        vcam_row.addWidget(self._field_label("Camera Name"))
        self._vcam_name_edit = QLineEdit(self._config.virtual_camera_name)
        self._vcam_name_edit.editingFinished.connect(self._on_vcam_name_changed)
        self._style_line_edit(self._vcam_name_edit)
        vcam_row.addWidget(self._vcam_name_edit, 1)
        layout.addLayout(vcam_row)

        # ── Section: Startup ──
        layout.addSpacing(4)
        layout.addWidget(self._section_label("Startup"))

        # Auto-connect on launch
        self._auto_connect_cb = QCheckBox("Auto-connect on launch")
        self._auto_connect_cb.setChecked(self._config.auto_connect_on_launch)
        self._auto_connect_cb.stateChanged.connect(self._on_auto_connect_changed)
        self._style_checkbox(self._auto_connect_cb)
        layout.addWidget(self._auto_connect_cb)

        # Auto-start webcam
        self._auto_webcam_cb = QCheckBox("Auto-start webcam mode")
        self._auto_webcam_cb.setChecked(self._config.auto_start_webcam)
        self._auto_webcam_cb.stateChanged.connect(self._on_auto_webcam_changed)
        self._style_checkbox(self._auto_webcam_cb)
        layout.addWidget(self._auto_webcam_cb)

        # ── Section: Advanced ──
        layout.addSpacing(4)
        layout.addWidget(self._section_label("Advanced"))

        # FPS
        fps_row = QHBoxLayout()
        fps_row.addWidget(self._field_label("FPS"))
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 60)
        self._fps_spin.setValue(self._config.stream_fps)
        self._fps_spin.editingFinished.connect(self._on_fps_changed)
        self._style_spin(self._fps_spin)
        fps_row.addWidget(self._fps_spin, 1)
        layout.addLayout(fps_row)

        # Log level
        log_row = QHBoxLayout()
        log_row.addWidget(self._field_label("Log Level"))
        self._log_combo = QComboBox()
        for lvl in ("DEBUG", "INFO", "WARNING", "ERROR"):
            self._log_combo.addItem(lvl, lvl)
        self._set_combo_value(self._log_combo, self._config.log_level)
        self._log_combo.currentIndexChanged.connect(self._on_log_level_changed)
        self._style_combo(self._log_combo)
        log_row.addWidget(self._log_combo, 1)
        layout.addLayout(log_row)

        # ── Save button + restart note ──
        layout.addSpacing(8)

        self._restart_note = QLabel("")
        self._restart_note.setStyleSheet("color: #FF9800; font-size: 10px;")
        self._restart_note.setVisible(False)
        layout.addWidget(self._restart_note)

        self._save_btn = QPushButton("Save Settings")
        self._save_btn.setMinimumHeight(38)
        self._save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_btn.setEnabled(False)
        self._save_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; color: white;
                border: none; border-radius: 6px;
                font-size: 13px; font-weight: bold;
                padding: 8px 20px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:pressed { background-color: #1565C0; }
            QPushButton:disabled {
                background-color: #1e2d4a; color: #445566;
            }
        """)
        self._save_btn.clicked.connect(self._on_save_clicked)
        layout.addWidget(self._save_btn)

        # Track dirty state
        self._dirty = False
        self._needs_restart = False

        # Outer styling — match main window theme
        self.setStyleSheet("""
            SettingsPanel {
                background-color: #16213e;
                border: 1px solid #1e2d4a;
                border-radius: 8px;
            }
        """)

    # ── Helpers ──

    def _section_label(self, text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            "color: #556677; font-size: 10px; font-weight: bold; "
            "letter-spacing: 1px; padding-bottom: 2px;"
        )
        return lbl

    def _field_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setFixedWidth(110)
        lbl.setStyleSheet("color: #9E9E9E; font-size: 12px;")
        return lbl

    def _style_combo(self, combo: QComboBox):
        combo.setStyleSheet("""
            QComboBox {
                background-color: #0d1526; color: #E0E0E0;
                border: 1px solid #1e2d4a; border-radius: 4px;
                padding: 5px 10px; font-size: 12px;
            }
            QComboBox:hover { border-color: #2196F3; }
            QComboBox::drop-down {
                border: none; width: 20px;
            }
            QComboBox QAbstractItemView {
                background-color: #16213e; color: #E0E0E0;
                selection-background-color: #2196F3;
                border: 1px solid #1e2d4a;
            }
        """)

    def _style_spin(self, spin: QSpinBox):
        spin.setStyleSheet("""
            QSpinBox {
                background-color: #0d1526; color: #E0E0E0;
                border: 1px solid #1e2d4a; border-radius: 4px;
                padding: 5px 10px; font-size: 12px;
            }
            QSpinBox:hover { border-color: #2196F3; }
        """)

    def _style_line_edit(self, edit: QLineEdit):
        edit.setStyleSheet("""
            QLineEdit {
                background-color: #0d1526; color: #E0E0E0;
                border: 1px solid #1e2d4a; border-radius: 4px;
                padding: 5px 10px; font-size: 12px;
            }
            QLineEdit:hover { border-color: #2196F3; }
            QLineEdit:focus { border-color: #2196F3; }
        """)

    def _style_checkbox(self, cb: QCheckBox):
        cb.setStyleSheet("""
            QCheckBox { color: #9E9E9E; font-size: 12px; spacing: 8px; }
            QCheckBox::indicator {
                width: 16px; height: 16px; border-radius: 4px;
                border: 1px solid #1e2d4a; background-color: #0d1526;
            }
            QCheckBox::indicator:checked {
                background-color: #2196F3; border-color: #2196F3;
            }
        """)

    def _set_combo_value(self, combo: QComboBox, value):
        for i in range(combo.count()):
            if combo.itemData(i) == value:
                combo.setCurrentIndex(i)
                return

    def _mark_dirty(self):
        """Mark settings as changed — enable the Save button."""
        self._dirty = True
        self._save_btn.setEnabled(True)
        self._save_btn.setText("Save Settings")

    def _mark_restart_needed(self, msg: str):
        """Mark that a restart is required after save."""
        self._needs_restart = True
        self._mark_dirty()
        self._restart_note.setText(msg + " App will restart on save.")
        self._restart_note.setVisible(True)
        self._save_btn.setText("Save & Restart")

    # ── Change handlers ──

    def _on_resolution_changed(self):
        code = self._resolution_combo.currentData()
        if code is None or code == self._config.resolution:
            return
        from gopro_connection import RESOLUTION_MAP
        self._config.resolution = code
        w, h, _ = RESOLUTION_MAP[code]
        self._config.stream_width = w
        self._config.stream_height = h
        self._mark_dirty()

    def _on_fov_changed(self):
        code = self._fov_combo.currentData()
        if code is None or code == self._config.fov:
            return
        self._config.fov = code
        self._mark_dirty()

    def _on_flicker_changed(self):
        code = self._flicker_combo.currentData()
        if code is None or code == self._config.anti_flicker:
            return
        self._config.anti_flicker = code
        self._mark_dirty()

    def _on_port_changed(self):
        val = self._port_spin.value()
        if val == self._config.udp_port:
            return
        self._config.udp_port = val
        self._mark_restart_needed("Port change requires restart.")

    def _on_vcam_name_changed(self):
        val = self._vcam_name_edit.text().strip()
        if not val or val == self._config.virtual_camera_name:
            return
        self._config.virtual_camera_name = val
        self._mark_restart_needed("Camera name change requires restart.")

    def _on_auto_connect_changed(self, state):
        self._config.auto_connect_on_launch = bool(state)
        self._mark_dirty()

    def _on_auto_webcam_changed(self, state):
        self._config.auto_start_webcam = bool(state)
        self._mark_dirty()

    def _on_fps_changed(self):
        val = self._fps_spin.value()
        if val == self._config.stream_fps:
            return
        self._config.stream_fps = val
        self._mark_restart_needed("FPS change requires restart.")

    def _on_log_level_changed(self):
        val = self._log_combo.currentData()
        if val is None or val == self._config.log_level:
            return
        self._config.log_level = val
        self._mark_dirty()

    def _on_save_clicked(self):
        """Save settings to disk and restart if needed."""
        self._config.save()
        log.info("[EVENT:config] Settings saved to disk")

        if self._needs_restart:
            log.info("[EVENT:config] Restart required — restarting app")
            self._save_btn.setText("Restarting...")
            self._save_btn.setEnabled(False)
            # Give the user a moment to see the message, then restart
            QTimer.singleShot(500, self._do_restart)
        else:
            # Apply settings live if streaming
            if self._controller.state == AppState.STREAMING:
                def _apply_live():
                    self._controller._apply_anti_flicker()
                    self._controller.change_resolution(
                        self._config.resolution, self._config.fov,
                    )
                threading.Thread(
                    target=_apply_live,
                    name="ApplySettings",
                    daemon=True,
                ).start()
            self._dirty = False
            self._save_btn.setEnabled(False)
            self._save_btn.setText("Saved")
            self._restart_note.setVisible(False)
            QTimer.singleShot(2000, lambda: self._save_btn.setText("Save Settings"))

    def _do_restart(self):
        """Stop the controller and re-launch the application.

        Uses subprocess.Popen instead of os.execv to avoid corrupting
        PyInstaller's _internal directory when running as a frozen exe.
        The new process starts independently before this one exits.
        """
        import os
        import subprocess
        self._controller.stop()

        if getattr(sys, 'frozen', False):
            # Frozen exe: just re-launch the exe
            exe = sys.executable
            log.info("[EVENT:config] Restarting frozen exe: %s", exe)
            subprocess.Popen(
                [exe],
                creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0),
            )
        else:
            # Running from source: re-launch with python
            python = sys.executable
            script = os.path.abspath(sys.argv[0])
            args = sys.argv[1:]
            log.info("[EVENT:config] Restarting: %s %s %s", python, script, args)
            subprocess.Popen(
                [python, script] + args,
                creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0),
            )

        # Exit this process
        QApplication.instance().quit()

    # ── Public API ──

    def toggle(self):
        """Toggle visibility."""
        self._expanded = not self._expanded
        self.setVisible(self._expanded)

    @property
    def is_expanded(self) -> bool:
        return self._expanded


# ── DashboardWindow ─────────────────────────────────────────────────────────

class DashboardWindow(QMainWindow):
    """Main application window for GoPro Bridge.

    Displays:
      - Connection state indicator (colored dot + label)
      - Status log (scrollable text area)
      - Camera info (battery, etc.) when connected
      - Retry button (appears after discovery timeout)

    Thread-safe: All UI updates are routed through Qt signals so
    AppController callbacks (which fire from background threads) never
    touch widgets directly.
    """

    # Qt signals for thread-safe UI updates from AppController callbacks
    _sig_state_changed = pyqtSignal(object)     # AppState
    _sig_status_message = pyqtSignal(str, str)  # (message, level)
    _sig_camera_info = pyqtSignal(dict)         # camera info dict
    _sig_webcam_mode_failed = pyqtSignal(str)   # reason string
    _sig_active_port = pyqtSignal(int)          # active UDP port

    def __init__(self, controller: AppController, config: Optional[Config] = None,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.controller = controller
        self._config = config

        self._build_ui()
        self._connect_signals()
        self._wire_controller()

        log.info("[EVENT:startup] Dashboard window created")

    # ── UI construction ──────────────────────────────────────────────

    def _build_ui(self):
        """Build the window layout."""
        self.setWindowTitle("GoPro Bridge v1.0.0")
        self.setMinimumSize(480, 420)
        self.resize(540, 600)

        # Dark theme — deep navy-black matching setup wizard
        self.setStyleSheet("""
            QMainWindow { background-color: #1a1a2e; }
            QLabel { color: #E0E0E0; }
            QTextEdit {
                background-color: #16213e;
                color: #D4D4D4;
                border: 1px solid #1e2d4a;
                border-radius: 6px;
                font-family: 'Consolas', 'Cascadia Code', 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
                selection-background-color: #2196F3;
            }
            QScrollBar:vertical {
                background: #16213e;
                width: 8px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical {
                background: #2a3a5c;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #3a4a6c;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 16)
        main_layout.setSpacing(14)

        # ── Header: state indicator ──
        header = QHBoxLayout()
        header.setSpacing(10)

        self._state_dot = QLabel("\u25CF")
        self._state_dot.setFixedWidth(22)
        self._state_dot.setStyleSheet("font-size: 18px; color: #888888;")

        self._state_label = QLabel("Initializing...")
        state_font = QFont("Segoe UI", 14, QFont.Weight.Bold)
        self._state_label.setFont(state_font)

        header.addWidget(self._state_dot)
        header.addWidget(self._state_label)
        header.addStretch()

        # Info badges (battery + port) in a compact row
        self._battery_label = QLabel("")
        self._battery_label.setStyleSheet(
            "color: #9E9E9E; font-size: 11px; background-color: #16213e; "
            "border-radius: 4px; padding: 2px 8px; border: 1px solid #1e2d4a;"
        )
        self._battery_label.setVisible(False)
        header.addWidget(self._battery_label)

        self._port_label = QLabel("")
        self._port_label.setStyleSheet(
            "color: #9E9E9E; font-size: 11px; background-color: #16213e; "
            "border-radius: 4px; padding: 2px 8px; border: 1px solid #1e2d4a;"
        )
        self._port_label.setVisible(False)
        header.addWidget(self._port_label)

        # Settings gear button
        self._settings_btn = QPushButton("\u2699")
        self._settings_btn.setFixedSize(34, 34)
        self._settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._settings_btn.setToolTip("Settings")
        self._settings_btn.setStyleSheet("""
            QPushButton {
                background: transparent; color: #666666;
                border: none; font-size: 20px;
                border-radius: 17px;
            }
            QPushButton:hover { color: #2196F3; background-color: #16213e; }
        """)
        self._settings_btn.clicked.connect(self._on_settings_clicked)
        header.addWidget(self._settings_btn)

        main_layout.addLayout(header)

        # ── Retry button (hidden by default) ──
        self._retry_widget = RetryButton()
        main_layout.addWidget(self._retry_widget)

        # ── Manual mode guide (hidden by default) ──
        self._manual_guide = ManualModeGuide()
        main_layout.addWidget(self._manual_guide)

        # ── Settings panel (hidden by default, toggled by gear button) ──
        if self._config is not None:
            self._settings_panel = SettingsPanel(self._config, self.controller)
            # Wrap in a scroll area so it works at any window size
            self._settings_scroll = QScrollArea()
            self._settings_scroll.setWidget(self._settings_panel)
            self._settings_scroll.setWidgetResizable(True)
            self._settings_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            self._settings_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; } QWidget { background: transparent; }")
            self._settings_scroll.setVisible(False)
            # Don't let settings hog space — cap at 300px, give rest to log
            self._settings_scroll.setMaximumHeight(320)
            main_layout.addWidget(self._settings_scroll, stretch=0)
        else:
            self._settings_panel = None
            self._settings_scroll = None

        # ── Status log ──
        log_label = QLabel("STATUS LOG")
        log_label.setStyleSheet(
            "color: #556677; font-size: 10px; font-weight: bold; "
            "letter-spacing: 1px;"
        )
        main_layout.addWidget(log_label)

        self._log_area = QTextEdit()
        self._log_area.setReadOnly(True)
        self._log_area.setMinimumHeight(100)
        main_layout.addWidget(self._log_area, stretch=1)

        # ── Bottom bar: action buttons ──
        bottom = QHBoxLayout()

        _btn_style = """
            QPushButton {{
                background-color: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 6px;
                padding: 7px 18px;
                font-size: 12px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {hover}; }}
            QPushButton:pressed {{ background-color: {pressed}; }}
            QPushButton:disabled {{ background-color: #16213e; color: #445566; border-color: #1e2d4a; }}
        """

        # Pause button
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setMinimumWidth(90)
        self._pause_btn.setStyleSheet(_btn_style.format(
            bg="#1a3a6a", fg="#90CAF9", border="#2a5a8a",
            hover="#2a4a7a", pressed="#0a2a5a"))
        self._pause_btn.clicked.connect(self._on_pause_clicked)
        self._pause_btn.setVisible(False)
        bottom.addWidget(self._pause_btn)

        # Resume button (shown when paused or in charge mode)
        self._resume_btn = QPushButton("Resume")
        self._resume_btn.setMinimumWidth(90)
        self._resume_btn.setStyleSheet(_btn_style.format(
            bg="#1a4a2a", fg="#A5D6A7", border="#2a6a3a",
            hover="#2a5a3a", pressed="#0a3a1a"))
        self._resume_btn.clicked.connect(self._on_resume_clicked)
        self._resume_btn.setVisible(False)
        bottom.addWidget(self._resume_btn)

        # Charge mode button
        self._charge_btn = QPushButton("Charge")
        self._charge_btn.setMinimumWidth(90)
        self._charge_btn.setToolTip("Stop webcam to maximize USB charging")
        self._charge_btn.setStyleSheet(_btn_style.format(
            bg="#3a1a5a", fg="#CE93D8", border="#5a2a7a",
            hover="#4a2a6a", pressed="#2a0a4a"))
        self._charge_btn.clicked.connect(self._on_charge_clicked)
        self._charge_btn.setVisible(False)
        bottom.addWidget(self._charge_btn)

        bottom.addStretch()

        # Stop button
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setMinimumWidth(80)
        self._stop_btn.setStyleSheet(_btn_style.format(
            bg="#16213e", fg="#9E9E9E", border="#1e2d4a",
            hover="#1e2d4a", pressed="#0d1526"))
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        bottom.addWidget(self._stop_btn)

        main_layout.addLayout(bottom)

        # ── System tray icon ──
        self._setup_system_tray()

    def _setup_system_tray(self):
        """Set up system tray icon with context menu."""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            log.debug("[EVENT:tray] System tray not available")
            self._tray_icon = None
            return

        self._tray_icon = QSystemTrayIcon(self)
        # Use a simple colored circle as tray icon
        from PyQt6.QtGui import QPixmap, QPainter
        pixmap = QPixmap(32, 32)
        pixmap.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pixmap)
        painter.setBrush(QColor("#4CAF50"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(4, 4, 24, 24)
        painter.end()
        self._tray_icon.setIcon(QIcon(pixmap))
        self._tray_icon.setToolTip("GoPro Bridge")

        # Context menu
        tray_menu = QMenu()
        tray_menu.setStyleSheet("""
            QMenu { background-color: #16213e; color: #E0E0E0; border: 1px solid #1e2d4a; border-radius: 4px; }
            QMenu::item { padding: 6px 20px; }
            QMenu::item:selected { background-color: #1e2d4a; }
        """)
        show_action = tray_menu.addAction("Show")
        show_action.triggered.connect(self._tray_show)
        tray_menu.addSeparator()
        quit_action = tray_menu.addAction("Quit")
        quit_action.triggered.connect(self._tray_quit)
        self._tray_icon.setContextMenu(tray_menu)
        self._tray_icon.activated.connect(self._tray_activated)
        self._tray_icon.show()

    def _tray_activated(self, reason):
        """Handle tray icon double-click — restore window."""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self._tray_show()

    def _tray_show(self):
        """Restore window from tray."""
        self.showNormal()
        self.activateWindow()

    def _tray_quit(self):
        """Quit from tray — stop controller and exit."""
        self.controller.stop()
        QApplication.instance().quit()

    def closeEvent(self, event):
        """Minimize to tray on close instead of quitting."""
        if self._tray_icon and self._tray_icon.isVisible():
            self.hide()
            self._tray_icon.showMessage(
                "GoPro Bridge",
                "Minimized to tray. Double-click to restore.",
                QSystemTrayIcon.MessageIcon.Information,
                2000,
            )
            event.ignore()
        else:
            self.controller.stop()
            event.accept()

    # ── Signal wiring ────────────────────────────────────────────────

    def _connect_signals(self):
        """Connect Qt signals to slot methods."""
        self._sig_state_changed.connect(self._on_state_changed)
        self._sig_status_message.connect(self._on_status_message)
        self._sig_camera_info.connect(self._on_camera_info)
        self._sig_webcam_mode_failed.connect(self._on_webcam_mode_failed)
        self._sig_active_port.connect(self._on_active_port)
        self._retry_widget.retry_requested.connect(self._on_retry_requested)

    def _wire_controller(self):
        """Wire AppController callbacks to emit Qt signals.

        AppController callbacks fire from background threads, so we
        emit Qt signals which get dispatched to the main thread.
        """
        self.controller.on_state_change = lambda state: self._sig_state_changed.emit(state)
        self.controller.on_status = lambda msg, lvl: self._sig_status_message.emit(msg, lvl)
        self.controller.on_camera_info = lambda info: self._sig_camera_info.emit(info)
        self.controller.on_webcam_mode_failed = lambda reason: self._sig_webcam_mode_failed.emit(reason)
        self.controller.on_active_port = lambda port: self._sig_active_port.emit(port)

    # ── Slot methods (run on main thread) ────────────────────────────

    @pyqtSlot(object)
    def _on_state_changed(self, state: AppState):
        """Update the UI when the app state changes."""
        color = STATE_COLORS.get(state, "#888888")
        label = STATE_LABELS.get(state, state.name)

        self._state_dot.setStyleSheet(f"font-size: 16px; color: {color};")
        self._state_label.setText(label)
        self._state_label.setStyleSheet(f"color: {color};")

        # Show/hide the retry button based on state
        if state in RETRY_VISIBLE_STATES:
            # Determine the message based on state
            if state == AppState.ERROR:
                self._retry_widget.show_retry(
                    "GoPro not found after all retries.\n"
                    "Check that the camera is connected via USB-C and powered on."
                )
            elif state == AppState.DISCONNECTED:
                self._retry_widget.show_retry(
                    "GoPro disconnected.\n"
                    "Reconnect the USB cable and click Retry."
                )
            else:
                self._retry_widget.show_retry()
        else:
            self._retry_widget.hide_retry()

        # Hide the manual guide when we move to an active/recovering state
        if state in (AppState.DISCOVERING, AppState.CONNECTING,
                     AppState.STREAMING, AppState.RECONNECTING,
                     AppState.PAUSED, AppState.CHARGE_MODE):
            self._manual_guide.hide_guide()

        # Show/hide action buttons based on state
        is_streaming = state == AppState.STREAMING
        is_paused = state == AppState.PAUSED
        is_charging = state == AppState.CHARGE_MODE

        self._pause_btn.setVisible(is_streaming)
        self._charge_btn.setVisible(is_streaming or is_paused)
        self._resume_btn.setVisible(is_paused or is_charging)

        log.debug("[EVENT:state_change] GUI updated for state: %s", state.name)

    @pyqtSlot(str, str)
    def _on_status_message(self, message: str, level: str):
        """Append a status message to the log area."""
        color_map = {
            "info": "#D4D4D4",
            "success": "#4CAF50",
            "warning": "#FF9800",
            "error": "#F44336",
        }
        color = color_map.get(level, "#D4D4D4")
        prefix_map = {
            "info": "ℹ",
            "success": "✓",
            "warning": "⚠",
            "error": "✗",
        }
        prefix = prefix_map.get(level, " ")

        html = f'<span style="color: {color};">{prefix} {message}</span>'
        self._log_area.append(html)

        # Auto-scroll to bottom
        scrollbar = self._log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @pyqtSlot(dict)
    def _on_camera_info(self, info: dict):
        """Update camera info display (battery level, etc.)."""
        battery = info.get("battery_level")
        if battery is not None:
            # Color-coded battery indicator
            if battery <= 10:
                icon, color = "🪫", "#F44336"  # Red
            elif battery <= 25:
                icon, color = "🔋", "#FF9800"  # Orange
            elif battery <= 50:
                icon, color = "🔋", "#FFEB3B"  # Yellow
            else:
                icon, color = "🔋", "#4CAF50"  # Green
            self._battery_label.setText(f"{icon} {battery}%")
            self._battery_label.setStyleSheet(f"color: {color}; font-size: 12px;")
            self._battery_label.setVisible(True)

    @pyqtSlot(str)
    def _on_webcam_mode_failed(self, reason: str):
        """Show the manual mode guide when programmatic webcam switching fails.

        This is triggered by AppController when start_webcam() fails but the
        camera is still reachable over HTTP — indicating the camera is connected
        but won't enter webcam mode programmatically.

        Args:
            reason: Human-readable explanation of why switching failed.
        """
        self._manual_guide.show_guide(reason)
        log.info("[EVENT:manual_guide] Showing manual mode guide: %s", reason[:80])

    @pyqtSlot(int)
    def _on_active_port(self, port: int):
        """Show the active UDP port in the header (read-only indicator).

        This is separate from the port spinner in SettingsPanel — the spinner
        controls the *configured* port, while this label shows the port
        actually in use (which may differ after ephemeral auto-selection).
        """
        self._port_label.setText(f"Port {port}")
        self._port_label.setVisible(True)
        log.debug("[EVENT:active_port] Active port indicator set to %d", port)

    # ── Button handlers ──────────────────────────────────────────────

    @pyqtSlot()
    def _on_retry_requested(self):
        """Handle retry button click — restart discovery in background."""
        log.info("[EVENT:discovery_start] User triggered manual retry")
        self._retry_widget.set_retrying()

        # Run retry on a background thread so the GUI stays responsive
        thread = threading.Thread(
            target=self.controller.retry_connection,
            name="ManualRetry",
            daemon=True,
        )
        thread.start()

    def _on_settings_clicked(self):
        """Toggle the settings panel."""
        if self._settings_scroll is not None:
            expanded = self._settings_scroll.isVisible()
            self._settings_scroll.setVisible(not expanded)
            if self._settings_panel is not None:
                self._settings_panel._expanded = not expanded
            color = "#808080" if expanded else "#2196F3"
            self._settings_btn.setStyleSheet(f"""
                QPushButton {{
                    background: transparent; color: {color};
                    border: none; font-size: 18px;
                }}
                QPushButton:hover {{ color: #2196F3; }}
            """)

    def _on_stop_clicked(self):
        """Handle stop button click."""
        log.info("[EVENT:shutdown] User clicked Stop")
        self.controller.stop()

    def _on_pause_clicked(self):
        """Handle pause button click."""
        log.info("[EVENT:pause] User clicked Pause")
        self.controller.pause_webcam()

    def _on_resume_clicked(self):
        """Handle resume button click."""
        log.info("[EVENT:resume] User clicked Resume")
        self.controller.resume_webcam()

    def _on_charge_clicked(self):
        """Handle charge mode button click."""
        log.info("[EVENT:charge_mode] User clicked Charge")
        self.controller.enter_charge_mode()

    # ── Public API ───────────────────────────────────────────────────

    @property
    def retry_button(self) -> RetryButton:
        """Access the retry button widget (for testing)."""
        return self._retry_widget

    @property
    def manual_guide(self) -> ManualModeGuide:
        """Access the manual mode guide widget (for testing)."""
        return self._manual_guide

    @property
    def port_label(self) -> QLabel:
        """Access the active port label widget (for testing)."""
        return self._port_label

    def closeEvent(self, event):
        """Handle window close — stop the controller gracefully."""
        log.info("[EVENT:shutdown] Window closed by user")
        self.controller.stop()
        event.accept()


# ── Application launcher ─────────────────────────────────────────────────────

def run_gui(controller: AppController, config: Optional[Config] = None) -> int:
    """Create the QApplication, show the dashboard, and run the event loop.

    This replaces the console-based main loop in main.py.

    Args:
        controller: The AppController instance (already configured).
        config: Config instance for the settings panel.

    Returns:
        The Qt application exit code.
    """
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("GoPro Bridge")
    app.setApplicationDisplayName("GoPro Bridge")

    window = DashboardWindow(controller, config=config)
    window.show()

    # Start the controller after the window is visible
    # so the user immediately sees status updates
    QTimer.singleShot(100, controller.start)

    return app.exec()
