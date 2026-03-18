"""
setup_wizard.py -- First-run setup wizard for GoPro Bridge

Displays a polished PyQt6 wizard that guides new users through installing
the three runtime dependencies before the main dashboard appears:

  1. ffmpeg (video decoder) -- downloaded from gyan.dev
  2. Unity Capture (virtual camera driver) -- registered via regsvr32
  3. Windows Firewall rule (UDP 8554) -- created via netsh

The wizard checks each dependency on launch. Already-installed deps show
as complete immediately. The user clicks "Install" / "Set Up" for each
missing dep. Once all are resolved (installed or skipped), the "Continue"
button activates and the wizard signals the main app to proceed.

Install operations run on background threads so the UI stays responsive.
Progress updates (especially for the ~90 MB ffmpeg download) are pushed
to the main thread via Qt signals.

Integration:
    from setup_wizard import SetupWizard
    from dependency_checker import DependencyChecker

    checker = DependencyChecker()
    wizard = SetupWizard(checker)
    wizard.setup_complete.connect(on_complete)
    wizard.show()
"""

import threading
from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QSize
from PyQt6.QtGui import QFont, QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QFrame,
    QGraphicsDropShadowEffect,
)

from logger import get_logger

log = get_logger("setup_wizard")


# ── Style constants ──────────────────────────────────────────────────────────

BG_PRIMARY = "#1a1a2e"
BG_CARD = "#16213e"
BG_CARD_HOVER = "#1a2744"
ACCENT_BLUE = "#2196F3"
COLOR_SUCCESS = "#4CAF50"
COLOR_WARNING = "#FF9800"
COLOR_ERROR = "#F44336"
COLOR_GRAY = "#888888"
TEXT_PRIMARY = "#E0E0E0"
TEXT_SECONDARY = "#9E9E9E"
TEXT_DIM = "#666666"

# Unicode status icons (no emojis)
ICON_PENDING = "\u25CB"      # White circle (outline)
ICON_IN_PROGRESS = "\u25CF"  # Black circle (filled)
ICON_COMPLETE = "\u2713"     # Check mark
ICON_ERROR = "\u2717"        # X mark
ICON_SKIPPED = "\u2014"      # Em dash


# ── Step states ──────────────────────────────────────────────────────────────

class StepState:
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    ERROR = "error"
    SKIPPED = "skipped"


# ── StepCard widget ──────────────────────────────────────────────────────────

class StepCard(QFrame):
    """A single dependency step displayed as a styled card.

    Shows the step name, description, status icon, action button,
    and (for ffmpeg) a download progress bar. Visual state updates
    are driven by set_state() calls from the wizard.

    Signals:
        action_clicked: Emitted when the user clicks Install / Set Up / Retry.
    """

    action_clicked = pyqtSignal()

    def __init__(
        self,
        step_id: str,
        title: str,
        description: str,
        button_label: str = "Install",
        show_progress: bool = False,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.step_id = step_id
        self._title = title
        self._description = description
        self._button_label = button_label
        self._show_progress = show_progress
        self._state = StepState.PENDING
        self._build_ui()
        self._apply_state()

    def _build_ui(self):
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(f"""
            StepCard {{
                background-color: {BG_CARD};
                border-radius: 8px;
                border: 1px solid #1e2d4a;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(16)

        # Status icon
        self._icon_label = QLabel(ICON_PENDING)
        self._icon_label.setFixedSize(32, 32)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_font = QFont()
        icon_font.setPointSize(16)
        icon_font.setBold(True)
        self._icon_label.setFont(icon_font)
        self._icon_label.setStyleSheet(f"color: {COLOR_GRAY}; background: transparent; border: none;")
        layout.addWidget(self._icon_label)

        # Text area (title + description + progress)
        text_layout = QVBoxLayout()
        text_layout.setSpacing(4)

        self._title_label = QLabel(self._title)
        title_font = QFont()
        title_font.setPointSize(11)
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        self._title_label.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent; border: none;")
        text_layout.addWidget(self._title_label)

        self._desc_label = QLabel(self._description)
        desc_font = QFont()
        desc_font.setPointSize(9)
        self._desc_label.setFont(desc_font)
        self._desc_label.setWordWrap(True)
        self._desc_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; border: none;")
        text_layout.addWidget(self._desc_label)

        # Status text (shown for complete/error/skipped)
        self._status_label = QLabel("")
        status_font = QFont()
        status_font.setPointSize(9)
        self._status_label.setFont(status_font)
        self._status_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; border: none;")
        self._status_label.setVisible(False)
        text_layout.addWidget(self._status_label)

        # Progress bar (ffmpeg download)
        if self._show_progress:
            self._progress_bar = QProgressBar()
            self._progress_bar.setFixedHeight(6)
            self._progress_bar.setTextVisible(False)
            self._progress_bar.setRange(0, 100)
            self._progress_bar.setValue(0)
            self._progress_bar.setStyleSheet(f"""
                QProgressBar {{
                    background-color: #0d1526;
                    border-radius: 3px;
                    border: none;
                }}
                QProgressBar::chunk {{
                    background-color: {ACCENT_BLUE};
                    border-radius: 3px;
                }}
            """)
            self._progress_bar.setVisible(False)
            text_layout.addWidget(self._progress_bar)

            self._progress_label = QLabel("")
            progress_font = QFont()
            progress_font.setPointSize(8)
            self._progress_label.setFont(progress_font)
            self._progress_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; border: none;")
            self._progress_label.setVisible(False)
            text_layout.addWidget(self._progress_label)

        layout.addLayout(text_layout, stretch=1)

        # Action button
        self._action_btn = QPushButton(self._button_label)
        self._action_btn.setFixedSize(100, 36)
        self._action_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._action_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {ACCENT_BLUE};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 12px;
                font-weight: bold;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                background-color: #1976D2;
            }}
            QPushButton:pressed {{
                background-color: #1565C0;
            }}
            QPushButton:disabled {{
                background-color: #2a3a5c;
                color: #555555;
            }}
        """)
        self._action_btn.clicked.connect(self.action_clicked.emit)
        layout.addWidget(self._action_btn, alignment=Qt.AlignmentFlag.AlignVCenter)

    def set_state(self, state: str, message: str = ""):
        """Update the visual state of this step card."""
        self._state = state
        if message:
            self._status_label.setText(message)
        self._apply_state()

    def _apply_state(self):
        if self._state == StepState.PENDING:
            self._icon_label.setText(ICON_PENDING)
            self._icon_label.setStyleSheet(f"color: {COLOR_GRAY}; background: transparent; border: none;")
            self._desc_label.setVisible(True)
            self._status_label.setVisible(False)
            self._action_btn.setVisible(True)
            self._action_btn.setEnabled(True)
            self._action_btn.setText(self._button_label)
            if self._show_progress:
                self._progress_bar.setVisible(False)
                self._progress_label.setVisible(False)

        elif self._state == StepState.IN_PROGRESS:
            self._icon_label.setText(ICON_IN_PROGRESS)
            self._icon_label.setStyleSheet(f"color: {ACCENT_BLUE}; background: transparent; border: none;")
            self._desc_label.setVisible(True)
            self._status_label.setVisible(False)
            self._action_btn.setVisible(True)
            self._action_btn.setEnabled(False)
            self._action_btn.setText("Installing...")
            if self._show_progress:
                self._progress_bar.setVisible(True)
                self._progress_label.setVisible(True)

        elif self._state == StepState.COMPLETE:
            self._icon_label.setText(ICON_COMPLETE)
            self._icon_label.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent; border: none;")
            self._desc_label.setVisible(False)
            self._status_label.setText(self._status_label.text() or "Installed")
            self._status_label.setStyleSheet(f"color: {COLOR_SUCCESS}; background: transparent; border: none;")
            self._status_label.setVisible(True)
            self._action_btn.setVisible(False)
            if self._show_progress:
                self._progress_bar.setVisible(False)
                self._progress_label.setVisible(False)

        elif self._state == StepState.ERROR:
            self._icon_label.setText(ICON_ERROR)
            self._icon_label.setStyleSheet(f"color: {COLOR_ERROR}; background: transparent; border: none;")
            self._desc_label.setVisible(False)
            self._status_label.setStyleSheet(f"color: {COLOR_ERROR}; background: transparent; border: none;")
            self._status_label.setVisible(True)
            self._action_btn.setVisible(True)
            self._action_btn.setEnabled(True)
            self._action_btn.setText("Retry")
            if self._show_progress:
                self._progress_bar.setVisible(False)
                self._progress_label.setVisible(False)

        elif self._state == StepState.SKIPPED:
            self._icon_label.setText(ICON_SKIPPED)
            self._icon_label.setStyleSheet(f"color: {COLOR_GRAY}; background: transparent; border: none;")
            self._desc_label.setVisible(False)
            self._status_label.setText("Skipped")
            self._status_label.setStyleSheet(f"color: {COLOR_WARNING}; background: transparent; border: none;")
            self._status_label.setVisible(True)
            self._action_btn.setVisible(False)
            if self._show_progress:
                self._progress_bar.setVisible(False)
                self._progress_label.setVisible(False)

    def set_progress(self, downloaded: int, total: int):
        """Update the download progress bar (ffmpeg only)."""
        if not self._show_progress:
            return
        if total > 0:
            pct = int((downloaded / total) * 100)
            self._progress_bar.setValue(pct)
            dl_mb = downloaded / (1024 * 1024)
            total_mb = total / (1024 * 1024)
            self._progress_label.setText(f"Downloading... {dl_mb:.1f} MB / {total_mb:.1f} MB")
        else:
            self._progress_bar.setRange(0, 0)  # Indeterminate
            dl_mb = downloaded / (1024 * 1024)
            self._progress_label.setText(f"Downloading... {dl_mb:.1f} MB")

    @property
    def state(self) -> str:
        return self._state


# ── SetupWizard main widget ─────────────────────────────────────────────────

class SetupWizard(QWidget):
    """First-run setup wizard for dependency installation.

    Checks for ffmpeg, Unity Capture, and the Windows Firewall rule.
    Guides the user through installing any missing dependencies with
    a polished dark-themed UI.

    Signals:
        setup_complete: Emitted when all deps are resolved. Carries a dict
            with keys like 'ffmpeg_path' for config updates.
        setup_skipped: Emitted if the user skips all remaining steps.
    """

    # Signals for cross-thread UI updates
    setup_complete = pyqtSignal(dict)
    setup_skipped = pyqtSignal()

    _step_state_changed = pyqtSignal(str, str, str)  # step_id, state, message
    _ffmpeg_progress = pyqtSignal(int, int)           # downloaded, total

    def __init__(self, checker, parent: Optional[QWidget] = None):
        """Initialize the setup wizard.

        Args:
            checker: DependencyChecker instance for check/install operations.
            parent: Optional parent widget.
        """
        super().__init__(parent)
        self._checker = checker
        self._results = {}  # step_id -> DependencyStatus
        self._step_cards = {}  # step_id -> StepCard

        self.setWindowTitle("GoPro Bridge - Setup")
        self.setFixedSize(620, 520)
        self.setStyleSheet(f"background-color: {BG_PRIMARY};")

        self._build_ui()
        self._connect_signals()

        # Run initial checks after the widget is shown
        QTimer.singleShot(300, self._run_initial_checks)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 24)
        root.setSpacing(0)

        # ── Header ──
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)

        title = QLabel("GoPro Bridge")
        title_font = QFont()
        title_font.setPointSize(22)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setStyleSheet(f"color: {TEXT_PRIMARY};")
        header_layout.addWidget(title)

        subtitle = QLabel("First-Time Setup")
        sub_font = QFont()
        sub_font.setPointSize(11)
        subtitle.setFont(sub_font)
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY};")
        header_layout.addWidget(subtitle)

        desc = QLabel(
            "The following components are needed to run GoPro Bridge. "
            "Click Install for each one, or skip if you prefer to set them up manually."
        )
        desc_font = QFont()
        desc_font.setPointSize(9)
        desc.setFont(desc_font)
        desc.setWordWrap(True)
        desc.setStyleSheet(f"color: {TEXT_DIM}; margin-top: 8px;")
        header_layout.addWidget(desc)

        root.addLayout(header_layout)
        root.addSpacing(24)

        # ── Step cards ──
        steps_layout = QVBoxLayout()
        steps_layout.setSpacing(12)

        # Step 1: ffmpeg
        ffmpeg_card = StepCard(
            step_id="ffmpeg",
            title="ffmpeg Video Decoder",
            description="Required to decode the GoPro video stream. Will be downloaded from the official gyan.dev builds (~90 MB).",
            button_label="Install",
            show_progress=True,
        )
        self._step_cards["ffmpeg"] = ffmpeg_card
        steps_layout.addWidget(ffmpeg_card)

        # Step 2: Unity Capture
        vcam_card = StepCard(
            step_id="unity_capture",
            title="Virtual Camera Driver",
            description="Registers a virtual webcam device so apps like Zoom and Teams can see your GoPro. Requires administrator permission.",
            button_label="Install",
        )
        self._step_cards["unity_capture"] = vcam_card
        steps_layout.addWidget(vcam_card)

        # Step 3: Firewall
        fw_card = StepCard(
            step_id="firewall",
            title="Firewall Rule",
            description="Opens UDP port 8554 so the GoPro video stream can reach this app. May require administrator permission.",
            button_label="Set Up",
        )
        self._step_cards["firewall"] = fw_card
        steps_layout.addWidget(fw_card)

        root.addLayout(steps_layout)
        root.addStretch(1)

        # ── Footer ──
        footer_layout = QHBoxLayout()
        footer_layout.setSpacing(12)

        self._skip_btn = QPushButton("Skip All")
        self._skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip_btn.setStyleSheet(f"""
            QPushButton {{
                background: transparent;
                color: {TEXT_SECONDARY};
                border: none;
                font-size: 11px;
                padding: 8px 16px;
            }}
            QPushButton:hover {{
                color: {TEXT_PRIMARY};
                text-decoration: underline;
            }}
        """)
        footer_layout.addWidget(self._skip_btn)

        footer_layout.addStretch(1)

        self._continue_btn = QPushButton("Continue")
        self._continue_btn.setFixedSize(120, 40)
        self._continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._continue_btn.setEnabled(False)
        self._continue_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_SUCCESS};
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 13px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: #43A047;
            }}
            QPushButton:pressed {{
                background-color: #388E3C;
            }}
            QPushButton:disabled {{
                background-color: #2a3a5c;
                color: #555555;
            }}
        """)
        footer_layout.addWidget(self._continue_btn)

        root.addLayout(footer_layout)

    def _connect_signals(self):
        # Step action buttons
        self._step_cards["ffmpeg"].action_clicked.connect(
            lambda: self._start_install("ffmpeg")
        )
        self._step_cards["unity_capture"].action_clicked.connect(
            lambda: self._start_install("unity_capture")
        )
        self._step_cards["firewall"].action_clicked.connect(
            lambda: self._start_install("firewall")
        )

        # Footer buttons
        self._skip_btn.clicked.connect(self._on_skip_all)
        self._continue_btn.clicked.connect(self._on_continue)

        # Cross-thread signals
        self._step_state_changed.connect(self._on_step_state_changed)
        self._ffmpeg_progress.connect(self._on_ffmpeg_progress)

    # ── Initial dependency checks ────────────────────────────────────────────

    def _run_initial_checks(self):
        """Check all dependencies on a background thread."""
        log.info("[EVENT:setup] Running initial dependency checks")

        def _check():
            statuses = self._checker.check_all()
            for status in statuses:
                self._results[status.name] = status
                if status.installed:
                    msg = status.path or "Installed"
                    self._step_state_changed.emit(
                        status.name, StepState.COMPLETE, msg
                    )
                    log.info("[EVENT:setup] %s: already installed (%s)", status.name, msg)
                else:
                    log.info("[EVENT:setup] %s: not found", status.name)

        thread = threading.Thread(target=_check, name="SetupCheck", daemon=True)
        thread.start()

    # ── Install handlers ─────────────────────────────────────────────────────

    def _start_install(self, step_id: str):
        """Launch the install for a step on a background thread."""
        log.info("[EVENT:setup] Starting install for %s", step_id)
        self._step_state_changed.emit(step_id, StepState.IN_PROGRESS, "")

        def _run():
            try:
                if step_id == "ffmpeg":
                    result = self._checker.install_ffmpeg(
                        progress_callback=lambda dl, total: self._ffmpeg_progress.emit(dl, total)
                    )
                elif step_id == "unity_capture":
                    result = self._checker.install_unity_capture()
                elif step_id == "firewall":
                    result = self._checker.install_firewall()
                else:
                    return

                self._results[step_id] = result

                if result.installed:
                    msg = result.path or "Installed"
                    self._step_state_changed.emit(step_id, StepState.COMPLETE, msg)
                    log.info("[EVENT:setup] %s installed successfully", step_id)
                else:
                    err = result.error or "Installation failed"
                    self._step_state_changed.emit(step_id, StepState.ERROR, err)
                    log.error("[EVENT:setup] %s install failed: %s", step_id, err)

            except Exception as exc:
                self._step_state_changed.emit(step_id, StepState.ERROR, str(exc))
                log.error("[EVENT:setup] %s install error: %s", step_id, exc, exc_info=True)

        thread = threading.Thread(
            target=_run, name=f"Setup-{step_id}", daemon=True
        )
        thread.start()

    # ── Signal handlers (main thread) ────────────────────────────────────────

    @pyqtSlot(str, str, str)
    def _on_step_state_changed(self, step_id: str, state: str, message: str):
        """Handle step state change on the main thread."""
        card = self._step_cards.get(step_id)
        if card:
            card.set_state(state, message)
        self._update_continue_button()

    @pyqtSlot(int, int)
    def _on_ffmpeg_progress(self, downloaded: int, total: int):
        """Handle ffmpeg download progress on the main thread."""
        card = self._step_cards.get("ffmpeg")
        if card:
            card.set_progress(downloaded, total)

    def _update_continue_button(self):
        """Enable Continue if all steps are complete or skipped."""
        all_resolved = all(
            card.state in (StepState.COMPLETE, StepState.SKIPPED)
            for card in self._step_cards.values()
        )
        self._continue_btn.setEnabled(all_resolved)

    def _on_skip_all(self):
        """Mark all pending steps as skipped and enable Continue."""
        for step_id, card in self._step_cards.items():
            if card.state == StepState.PENDING:
                card.set_state(StepState.SKIPPED)
                log.warning("[EVENT:setup] %s skipped by user", step_id)
        self._update_continue_button()
        self.setup_skipped.emit()

    def _on_continue(self):
        """Emit setup_complete with results and close the wizard."""
        result_dict = {}
        ffmpeg_result = self._results.get("ffmpeg")
        if ffmpeg_result and ffmpeg_result.installed and ffmpeg_result.path:
            result_dict["ffmpeg_path"] = ffmpeg_result.path
        log.info("[EVENT:setup] Setup wizard complete, proceeding to dashboard")
        self.setup_complete.emit(result_dict)
        self.close()


def run_setup_wizard(checker) -> Optional[dict]:
    """Run the setup wizard as a blocking call and return the results.

    This is a convenience function for integration into main.py.
    Creates a QApplication if one doesn't exist, shows the wizard,
    and blocks until the user clicks Continue or closes the window.

    Args:
        checker: DependencyChecker instance.

    Returns:
        dict with setup results (e.g., ffmpeg_path), or None if
        the wizard was closed without completing.
    """
    app = QApplication.instance()
    created_app = False
    if app is None:
        app = QApplication([])
        created_app = True

    result = {"completed": False, "data": {}}

    wizard = SetupWizard(checker)

    def on_complete(data):
        result["completed"] = True
        result["data"] = data

    def on_skipped():
        result["completed"] = True

    wizard.setup_complete.connect(on_complete)
    wizard.setup_skipped.connect(on_skipped)
    wizard.show()

    if created_app:
        app.exec()
    else:
        # If app already exists, run a local event loop
        from PyQt6.QtCore import QEventLoop
        loop = QEventLoop()
        wizard.destroyed.connect(loop.quit)
        wizard.setup_complete.connect(loop.quit)
        loop.exec()

    if result["completed"]:
        return result["data"]
    return None
