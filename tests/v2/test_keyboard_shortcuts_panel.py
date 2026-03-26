"""
tests/v2/test_keyboard_shortcuts_panel.py — Verify the keyboard shortcuts
help panel/overlay in the dashboard HTML.

Tests verify that:
  1. The kbd-overlay and kbd-panel CSS classes exist for the help overlay
  2. All expected keyboard shortcuts are documented in the overlay
  3. Each shortcut section (Navigation, Camera, Interface) is present
  4. The overlay can be opened from the sidebar footer hint
  5. The overlay can be opened from the page header help button
  6. The `L` shortcut (live preview) is documented
  7. Each shortcut has a human-readable description
  8. The `kbd` element CSS gives a visually distinct key appearance
  9. ARIA accessibility attributes are present on the overlay
  10. The overlay panel has a close button

All tests are purely static HTML analysis — no server, no hardware, no network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path to the HTML file under test
# ---------------------------------------------------------------------------

HTML_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "gomaxwebcam" / "dashboard" / "static" / "index.html"
)

pytestmark = pytest.mark.no_gopro_needed


# ---------------------------------------------------------------------------
# Fixture: parsed HTML text
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def html() -> str:
    """Read index.html once for the entire module."""
    assert HTML_PATH.exists(), f"Dashboard HTML not found at {HTML_PATH}"
    return HTML_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def shortcuts_section(html: str) -> str:
    """Extract the keyboard shortcuts overlay section from the full HTML."""
    start_marker = "KEYBOARD SHORTCUTS OVERLAY"
    end_marker = "</template>"
    start = html.find(start_marker)
    assert start != -1, "Could not find KEYBOARD SHORTCUTS OVERLAY section"
    end = html.find(end_marker, start)
    assert end != -1, "Could not find closing </template> for shortcuts overlay"
    return html[start:end + len(end_marker)]


# ---------------------------------------------------------------------------
# 1. CSS infrastructure tests
# ---------------------------------------------------------------------------

class TestShortcutOverlayCSS:
    """Verify that the necessary CSS classes are defined."""

    def test_kbd_overlay_class_defined(self, html: str):
        assert ".kbd-overlay {" in html or ".kbd-overlay{" in html, \
            ".kbd-overlay CSS class is missing"

    def test_kbd_panel_class_defined(self, html: str):
        assert ".kbd-panel {" in html or ".kbd-panel{" in html, \
            ".kbd-panel CSS class is missing"

    def test_kbd_row_class_defined(self, html: str):
        assert ".kbd-row {" in html or ".kbd-row{" in html, \
            ".kbd-row CSS class is missing"

    def test_kbd_element_styled(self, html: str):
        """The <kbd> element should have a distinct visual style."""
        assert "kbd {" in html or "kbd{" in html, \
            "<kbd> element CSS styling is missing"

    def test_kbd_element_has_border(self, html: str):
        """kbd should use border-bottom-width:2px for a 3D key look."""
        # Find the kbd CSS block
        start = html.find("kbd {")
        if start == -1:
            start = html.find("kbd{")
        assert start != -1
        end = html.find("}", start)
        kbd_css = html[start:end]
        assert "border" in kbd_css, "kbd element should have a border for key appearance"

    def test_kbd_fade_in_animation_defined(self, html: str):
        """The shortcuts panel should animate smoothly on open."""
        assert "kbd-fade-in" in html, \
            "Missing kbd-fade-in animation — panel should fade/scale in"

    def test_kbd_section_label_class_defined(self, html: str):
        assert ".kbd-section-label" in html, \
            ".kbd-section-label class is missing — sections need headings"

    def test_help_btn_class_defined(self, html: str):
        """The floating help button style should be defined."""
        assert ".help-btn" in html, \
            ".help-btn CSS is missing — page header help button needs styling"


# ---------------------------------------------------------------------------
# 2. Overlay structure tests
# ---------------------------------------------------------------------------

class TestShortcutOverlayStructure:
    """Verify the overlay HTML structure is correct."""

    def test_overlay_uses_template_x_if(self, html: str):
        """The overlay should use Alpine.js x-if for show/hide."""
        assert 'x-if="showShortcuts"' in html, \
            "Shortcuts overlay should use x-if=\"showShortcuts\""

    def test_overlay_has_aria_dialog(self, shortcuts_section: str):
        """The overlay should have ARIA role=dialog for accessibility."""
        assert 'role="dialog"' in shortcuts_section, \
            "Shortcuts overlay missing role=\"dialog\" for screen reader accessibility"

    def test_overlay_has_aria_modal(self, shortcuts_section: str):
        assert 'aria-modal="true"' in shortcuts_section, \
            "Shortcuts overlay missing aria-modal=\"true\""

    def test_overlay_has_aria_label(self, shortcuts_section: str):
        assert 'aria-label=' in shortcuts_section, \
            "Shortcuts overlay missing aria-label for accessibility"

    def test_overlay_closes_on_backdrop_click(self, shortcuts_section: str):
        """Clicking outside the panel should close the overlay."""
        assert "@click.self" in shortcuts_section, \
            "Overlay backdrop should close on @click.self"
        assert "showShortcuts = false" in shortcuts_section, \
            "Overlay backdrop click should set showShortcuts = false"

    def test_overlay_has_title(self, shortcuts_section: str):
        title_patterns = ["Keyboard Shortcuts", "keyboard shortcuts", "Keyboard shortcuts"]
        found = any(p in shortcuts_section for p in title_patterns)
        assert found, "Shortcuts panel should have a 'Keyboard Shortcuts' title"

    def test_overlay_has_close_button(self, shortcuts_section: str):
        """Panel must have an explicit close button."""
        assert "Close" in shortcuts_section, \
            "Shortcuts panel should have a Close button"
        assert "showShortcuts = false" in shortcuts_section, \
            "Close button should set showShortcuts = false"

    def test_overlay_has_three_sections(self, shortcuts_section: str):
        """Shortcuts should be organised into Navigation, Camera, Interface sections."""
        assert "Navigation" in shortcuts_section, \
            "Missing 'Navigation' section in shortcuts panel"
        assert "Camera" in shortcuts_section, \
            "Missing 'Camera' section in shortcuts panel"
        assert "Interface" in shortcuts_section, \
            "Missing 'Interface' section in shortcuts panel"

    def test_overlay_footer_hint(self, shortcuts_section: str):
        """A footer tip should remind users when shortcuts are disabled."""
        hint_patterns = ["text field", "input", "typing", "INPUT", "Shortcuts work"]
        found = any(p in shortcuts_section for p in hint_patterns)
        assert found, \
            "Shortcuts panel should mention that shortcuts don't work in text fields"


# ---------------------------------------------------------------------------
# 3. Navigation shortcut documentation tests
# ---------------------------------------------------------------------------

class TestNavigationShortcuts:
    """Verify all navigation shortcuts are documented."""

    def test_h_shortcut_documented(self, shortcuts_section: str):
        assert "<kbd>H</kbd>" in shortcuts_section, \
            "H shortcut (Go to Dashboard) not documented in panel"

    def test_s_shortcut_documented(self, shortcuts_section: str):
        assert "<kbd>S</kbd>" in shortcuts_section, \
            "S shortcut (Go to Settings) not documented in panel"

    def test_d_shortcut_documented(self, shortcuts_section: str):
        assert "<kbd>D</kbd>" in shortcuts_section, \
            "D shortcut (Go to Diagnostics) not documented in panel"

    def test_u_shortcut_documented(self, shortcuts_section: str):
        assert "<kbd>U</kbd>" in shortcuts_section, \
            "U shortcut (Go to Setup) not documented in panel"

    def test_nav_shortcuts_have_descriptions(self, shortcuts_section: str):
        """Each nav shortcut should have a human-readable description."""
        assert "Dashboard" in shortcuts_section
        assert "Settings" in shortcuts_section
        assert "Diagnostics" in shortcuts_section
        assert "Setup" in shortcuts_section


# ---------------------------------------------------------------------------
# 4. Camera shortcut documentation tests
# ---------------------------------------------------------------------------

class TestCameraShortcuts:
    """Verify all camera action shortcuts are documented."""

    def test_l_shortcut_documented(self, shortcuts_section: str):
        """L key for Live Preview — added in v2 UX polish iteration."""
        assert "<kbd>L</kbd>" in shortcuts_section, \
            "L shortcut (Open live preview) not documented in panel"

    def test_p_shortcut_documented(self, shortcuts_section: str):
        assert "<kbd>P</kbd>" in shortcuts_section, \
            "P shortcut (Pause/Resume stream) not documented in panel"

    def test_v_shortcut_documented(self, shortcuts_section: str):
        assert "<kbd>V</kbd>" in shortcuts_section, \
            "V shortcut (Toggle webcam visibility) not documented in panel"

    def test_r_shortcut_documented(self, shortcuts_section: str):
        assert "<kbd>R</kbd>" in shortcuts_section, \
            "R shortcut (Reconnect camera) not documented in panel"

    def test_live_preview_description_present(self, shortcuts_section: str):
        preview_hints = ["preview", "Preview", "live preview", "Live preview"]
        found = any(p in shortcuts_section for p in preview_hints)
        assert found, "L shortcut should have a 'live preview' description"

    def test_pause_description_present(self, shortcuts_section: str):
        pause_hints = ["Pause", "pause", "Resume", "resume"]
        found = any(p in shortcuts_section for p in pause_hints)
        assert found, "P shortcut should have a pause/resume description"

    def test_visibility_description_present(self, shortcuts_section: str):
        vis_hints = ["visibility", "Visibility", "webcam", "hide", "Hide"]
        found = any(p in shortcuts_section for p in vis_hints)
        assert found, "V shortcut should have a visibility description"

    def test_reconnect_description_present(self, shortcuts_section: str):
        conn_hints = ["reconnect", "Reconnect", "connect", "Connect"]
        found = any(p in shortcuts_section for p in conn_hints)
        assert found, "R shortcut should have a reconnect description"


# ---------------------------------------------------------------------------
# 5. Interface shortcut documentation tests
# ---------------------------------------------------------------------------

class TestInterfaceShortcuts:
    """Verify interface/UI shortcuts are documented."""

    def test_question_mark_shortcut_documented(self, shortcuts_section: str):
        assert "<kbd>?</kbd>" in shortcuts_section, \
            "? shortcut (Show/hide panel) not documented"

    def test_esc_shortcut_documented(self, shortcuts_section: str):
        esc_patterns = ["<kbd>Esc</kbd>", "<kbd class=\"wide\">Esc</kbd>"]
        found = any(p in shortcuts_section for p in esc_patterns)
        assert found, "Esc shortcut (Close overlay) not documented"

    def test_question_mark_has_description(self, shortcuts_section: str):
        panel_hints = ["this panel", "panel", "shortcuts", "hide"]
        found = any(p in shortcuts_section for p in panel_hints)
        assert found, "? shortcut should describe showing/hiding the panel"

    def test_esc_has_description(self, shortcuts_section: str):
        close_hints = ["Close", "close", "overlay", "Overlay", "modal", "Modal"]
        found = any(p in shortcuts_section for p in close_hints)
        assert found, "Esc shortcut should have a 'close' description"


# ---------------------------------------------------------------------------
# 6. Discoverability tests — sidebar footer hint + header help button
# ---------------------------------------------------------------------------

class TestShortcutDiscoverability:
    """Verify shortcuts can be discovered by users (not just power users)."""

    def test_sidebar_footer_has_shortcuts_hint(self, html: str):
        """Sidebar footer should have a clickable '? keyboard shortcuts' hint."""
        assert "kbd-footer-hint" in html, \
            "Sidebar footer should have .kbd-footer-hint for discoverability"

    def test_sidebar_hint_triggers_show_shortcuts(self, html: str):
        """The sidebar hint must open the shortcuts panel when clicked."""
        # Find the HTML usage (class=".." or class="kbd-footer-hint"), not the CSS definition
        # Look for the @click handler pattern in an element using kbd-footer-hint
        pattern = r'kbd-footer-hint[^>]*@click[^>]*showShortcuts'
        import re as _re
        found = _re.search(pattern, html)
        if not found:
            # Also check for @click before the class attribute
            pattern2 = r'@click[^>]*showShortcuts[^>]*kbd-footer-hint'
            found = _re.search(pattern2, html)
        assert found is not None, \
            "Sidebar shortcuts hint should set showShortcuts = true on click"

    def test_sidebar_hint_mentions_shortcuts(self, html: str):
        """The hint text should mention 'shortcuts' or 'keyboard'."""
        hint_texts = ["keyboard shortcuts", "Keyboard Shortcuts", "shortcuts"]
        found = any(t in html for t in hint_texts)
        assert found, "Sidebar hint should mention keyboard shortcuts"

    def test_page_header_has_help_button(self, html: str):
        """Dashboard page header should have a visible help button."""
        # Look for an element that uses the help-btn class in an HTML tag (not CSS def)
        import re as _re
        # Match class="help-btn" or class="... help-btn ..."
        pattern = r'class=["\'][^"\']*help-btn[^"\']*["\']'
        assert _re.search(pattern, html), \
            "Dashboard page header should have a .help-btn element"

    def test_header_help_button_triggers_panel(self, html: str):
        """The header help button must open the shortcuts panel."""
        # Find elements that have both help-btn class and @click with showShortcuts
        import re as _re
        pattern = r'help-btn[^>]*@click[^>]*showShortcuts'
        found = _re.search(pattern, html)
        if not found:
            pattern2 = r'@click[^>]*showShortcuts[^>]*help-btn'
            found = _re.search(pattern2, html)
        assert found is not None, \
            "Header help button should set showShortcuts = true on click"

    def test_header_help_button_has_title_hint(self, html: str):
        """The help button should have a title attribute as hover hint."""
        import re as _re
        # Find elements that have both help-btn class and a title attribute
        pattern = r'class=["\'][^"\']*help-btn[^"\']*["\'][^>]*title='
        found = _re.search(pattern, html)
        if not found:
            pattern2 = r'title=[^>]*class=["\'][^"\']*help-btn'
            found = _re.search(pattern2, html)
        assert found is not None, \
            "Help button should have a title= attribute as a hover tooltip"


# ---------------------------------------------------------------------------
# 7. JavaScript shortcut handler tests
# ---------------------------------------------------------------------------

class TestShortcutJavaScript:
    """Verify the JavaScript keydown handler implements all documented shortcuts."""

    @pytest.fixture(scope="class")
    def js_section(self, html: str) -> str:
        """Extract the JavaScript section."""
        start = html.find("<script>")
        end = html.rfind("</script>")
        assert start != -1, "No <script> block found"
        return html[start:end]

    def test_l_key_handler_present(self, js_section: str):
        """L key should toggle the live preview."""
        assert "case 'l':" in js_section or "case 'L':" in js_section, \
            "L key case is missing from the keyboard handler"

    def test_l_key_toggles_show_preview(self, js_section: str):
        """L key handler should reference showPreview."""
        assert "showPreview" in js_section, \
            "showPreview should be in the JS section"
        # Find the L case and check it sets showPreview
        l_idx = js_section.find("case 'l':")
        if l_idx == -1:
            l_idx = js_section.find("case 'L':")
        context = js_section[l_idx:l_idx + 200]
        assert "showPreview" in context, \
            "L key handler should toggle showPreview"

    def test_h_key_handler_present(self, js_section: str):
        assert "case 'h':" in js_section or "case 'H':" in js_section, \
            "H key case is missing from keyboard handler"

    def test_p_key_handler_present(self, js_section: str):
        assert "case 'p':" in js_section or "case 'P':" in js_section, \
            "P key case is missing from keyboard handler"

    def test_v_key_handler_present(self, js_section: str):
        assert "case 'v':" in js_section or "case 'V':" in js_section, \
            "V key case is missing from keyboard handler"

    def test_r_key_handler_present(self, js_section: str):
        assert "case 'r':" in js_section or "case 'R':" in js_section, \
            "R key case is missing from keyboard handler"

    def test_question_mark_handler_present(self, js_section: str):
        assert "case '?':" in js_section, \
            "? key case is missing from keyboard handler"

    def test_escape_handler_present(self, js_section: str):
        assert "case 'Escape':" in js_section, \
            "Escape key case is missing from keyboard handler"

    def test_input_guard_present(self, js_section: str):
        """Handler should guard against firing when user is typing in an input."""
        assert "INPUT" in js_section, \
            "Keyboard handler should skip shortcuts when focused on INPUT elements"
        assert "TEXTAREA" in js_section, \
            "Keyboard handler should skip shortcuts when focused on TEXTAREA elements"

    def test_modifier_key_guard_present(self, js_section: str):
        """Handler should not fire when Ctrl/Alt/Meta are held."""
        assert "ctrlKey" in js_section, \
            "Keyboard handler should check ctrlKey to avoid browser shortcut conflicts"

    def test_show_shortcuts_initial_false(self, js_section: str):
        """showShortcuts data property should start as false."""
        assert "showShortcuts: false" in js_section, \
            "showShortcuts state should initialise to false"
