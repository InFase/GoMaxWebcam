"""
tests/v2/test_settings_tooltips.py — Verify settings page has tooltips for all options.

Tests verify that:
  1. The tooltip CSS infrastructure (.tip, .tip-icon, .tip-text) is present and correct
  2. Every setting on the settings page has a tooltip element
  3. Tooltip text is non-empty, non-technical and actionable
  4. Keyboard-accessible (tabindex="0") on all tip-icon elements
  5. Edge-aware tooltip flipping (.tip-left) applied where needed

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
def settings_section(html: str) -> str:
    """Extract just the settings page section from the full HTML."""
    # The settings page starts at SETTINGS PAGE comment and ends at DIAGNOSTICS
    start_marker = "SETTINGS PAGE"
    end_marker = "DIAGNOSTICS PAGE"
    start = html.find(start_marker)
    end = html.find(end_marker)
    assert start != -1, "Could not find SETTINGS PAGE marker in HTML"
    assert end != -1, "Could not find DIAGNOSTICS PAGE marker in HTML"
    assert start < end, "SETTINGS PAGE should come before DIAGNOSTICS PAGE"
    return html[start:end]


# ---------------------------------------------------------------------------
# 1. CSS infrastructure tests
# ---------------------------------------------------------------------------

class TestTooltipCSS:
    """Verify the tooltip CSS classes are defined in the stylesheet."""

    def test_tip_class_defined(self, html: str):
        """The .tip container class must be defined."""
        assert ".tip {" in html or ".tip{" in html, \
            ".tip CSS class not found — tooltip container is missing"

    def test_tip_icon_class_defined(self, html: str):
        """The .tip-icon class must be defined and styled."""
        assert ".tip-icon {" in html or ".tip-icon{" in html, \
            ".tip-icon CSS class not found — info button is missing"

    def test_tip_text_class_defined(self, html: str):
        """The .tip-text class must be defined with positioning."""
        assert ".tip-text {" in html or ".tip-text{" in html, \
            ".tip-text CSS class not found — tooltip popup is missing"

    def test_tip_text_has_visibility_hidden(self, html: str):
        """Tooltips must start hidden (visibility: hidden)."""
        assert "visibility: hidden" in html, \
            "Tooltips must start hidden with visibility:hidden"

    def test_tip_hover_shows_tooltip(self, html: str):
        """Hovering .tip must reveal .tip-text."""
        assert ".tip:hover .tip-text" in html, \
            "Missing hover rule to show tooltip on .tip hover"

    def test_tip_focus_shows_tooltip(self, html: str):
        """Keyboard focus must also reveal tooltip (accessibility)."""
        assert ".tip:focus-within .tip-text" in html, \
            "Missing :focus-within rule — tooltips won't work with keyboard navigation"

    def test_tip_left_variant_defined(self, html: str):
        """.tip-left variant must be defined for right-edge tooltips."""
        assert ".tip.tip-left .tip-text" in html, \
            ".tip-left variant missing — tooltips near right edge will overflow"

    def test_tip_icon_cursor_is_help(self, html: str):
        """The info icon cursor should be 'help' for good UX."""
        assert "cursor: help" in html, \
            "tip-icon cursor should be 'help' to indicate it provides information"

    def test_tip_icon_hover_style(self, html: str):
        """.tip-icon should have a hover highlight."""
        assert ".tip-icon:hover" in html, \
            ".tip-icon:hover style missing — no visual feedback on hover"

    def test_tip_text_has_transition(self, html: str):
        """Tooltip should fade in/out with a CSS transition."""
        assert "transition: opacity" in html, \
            "Tooltip fade transition missing — abrupt show/hide is jarring"

    def test_tip_text_has_arrow(self, html: str):
        """Tooltip should have a ::after arrow pointing at the trigger."""
        assert ".tip-text::after" in html, \
            "Tooltip arrow (::after) missing — tooltip won't visually connect to its trigger"


# ---------------------------------------------------------------------------
# 2. Settings page completeness — every setting has a tooltip
# ---------------------------------------------------------------------------

class TestSettingsTooltipCoverage:
    """Every interactive setting must have a tooltip explaining it."""

    def _has_tooltip_near(self, section: str, label: str) -> bool:
        """Check that a label appears near a .tip-text element."""
        # Find the label in section
        idx = section.find(label)
        if idx == -1:
            return False
        # Look for tip-text within 800 chars around the label
        window = section[max(0, idx - 200):idx + 800]
        return "tip-text" in window

    def test_resolution_has_tooltip(self, settings_section: str):
        """Resolution dropdown must have a tooltip."""
        assert self._has_tooltip_near(settings_section, "Resolution"), \
            "Resolution setting missing tooltip"

    def test_fov_has_tooltip(self, settings_section: str):
        """Field of View dropdown must have a tooltip."""
        assert self._has_tooltip_near(settings_section, "Field of View"), \
            "Field of View setting missing tooltip"

    def test_transport_priority_has_tooltip(self, settings_section: str):
        """Transport Priority section must have a tooltip."""
        assert self._has_tooltip_near(settings_section, "Transport Priority"), \
            "Transport Priority section missing tooltip"

    def test_usb_transport_has_tooltip(self, settings_section: str):
        """USB transport legend item must have a tooltip."""
        assert self._has_tooltip_near(settings_section, ">USB"), \
            "USB transport missing tooltip"

    def test_cohn_transport_has_tooltip(self, settings_section: str):
        """COHN transport legend item must have a tooltip."""
        assert self._has_tooltip_near(settings_section, ">COHN"), \
            "COHN transport missing tooltip"

    def test_wifi_ap_transport_has_tooltip(self, settings_section: str):
        """WiFi AP transport legend item must have a tooltip."""
        assert self._has_tooltip_near(settings_section, "WiFi AP"), \
            "WiFi AP transport missing tooltip"

    def test_manual_switch_has_tooltip(self, settings_section: str):
        """Manual Switch section must have a tooltip."""
        assert self._has_tooltip_near(settings_section, "Manual Switch"), \
            "Manual Switch section missing tooltip"

    def test_auto_start_has_tooltip(self, settings_section: str):
        """Auto-Start on Login toggle must have a tooltip."""
        assert self._has_tooltip_near(settings_section, "Auto-Start"), \
            "Auto-Start on Login toggle missing tooltip"

    def test_ble_wake_has_tooltip(self, settings_section: str):
        """BLE Wake toggle must have a tooltip."""
        assert self._has_tooltip_near(settings_section, "BLE Wake"), \
            "BLE Wake toggle missing tooltip"

    def test_debug_logging_has_tooltip(self, settings_section: str):
        """Debug Logging toggle must have a tooltip."""
        assert self._has_tooltip_near(settings_section, "Debug Logging"), \
            "Debug Logging toggle missing tooltip"


# ---------------------------------------------------------------------------
# 3. Tooltip text quality checks
# ---------------------------------------------------------------------------

class TestTooltipTextQuality:
    """Tooltip text must be clear, non-empty, and actionable."""

    def _extract_tip_texts(self, html: str) -> list[str]:
        """Extract all .tip-text content from the HTML."""
        pattern = re.compile(r'class="tip-text">(.*?)</span>', re.DOTALL)
        return [m.group(1).strip() for m in pattern.finditer(html)]

    def test_all_tooltips_have_content(self, settings_section: str):
        """No tooltip should be empty."""
        tip_texts = self._extract_tip_texts(settings_section)
        assert len(tip_texts) > 0, "No tooltip text found in settings section"
        for text in tip_texts:
            clean = re.sub(r'<[^>]+>', '', text).strip()  # strip HTML tags
            assert len(clean) > 10, \
                f"Tooltip too short (likely empty): {repr(clean)}"

    def test_resolution_tooltip_mentions_quality(self, settings_section: str):
        """Resolution tooltip should explain the quality/performance tradeoff."""
        idx = settings_section.find("Resolution")
        window = settings_section[idx:idx + 600]
        tip_match = re.search(r'class="tip-text">(.*?)</span>', window, re.DOTALL)
        assert tip_match, "Resolution tooltip text not found"
        text = tip_match.group(1).lower()
        # Should mention video quality concepts
        assert any(word in text for word in ["quality", "sharp", "cpu", "performance", "slow"]), \
            "Resolution tooltip should explain quality/performance tradeoff"

    def test_fov_tooltip_mentions_options(self, settings_section: str):
        """FOV tooltip should explain what the different options do."""
        idx = settings_section.find("Field of View")
        window = settings_section[idx:idx + 600]
        tip_match = re.search(r'class="tip-text">(.*?)</span>', window, re.DOTALL)
        assert tip_match, "FOV tooltip text not found"
        text = tip_match.group(1).lower()
        # Should mention at least one FOV type
        assert any(word in text for word in ["wide", "narrow", "linear", "fisheye", "superview"]), \
            "FOV tooltip should explain the FOV options"

    def test_transport_tooltip_explains_concept(self, settings_section: str):
        """Transport Priority tooltip should explain what a transport is."""
        idx = settings_section.find("Transport Priority")
        window = settings_section[idx:idx + 600]
        tip_match = re.search(r'class="tip-text">(.*?)</span>', window, re.DOTALL)
        assert tip_match, "Transport Priority tooltip text not found"
        text = tip_match.group(1).lower()
        assert any(word in text for word in ["connection", "connect", "usb", "cable", "order"]), \
            "Transport Priority tooltip should explain what transports are"

    def test_auto_start_tooltip_explains_benefit(self, settings_section: str):
        """Auto-start tooltip should explain what happens and why it's useful."""
        idx = settings_section.find("Auto-Start")
        window = settings_section[idx:idx + 600]
        tip_match = re.search(r'class="tip-text">(.*?)</span>', window, re.DOTALL)
        assert tip_match, "Auto-Start tooltip text not found"
        text = tip_match.group(1).lower()
        assert any(word in text for word in ["launch", "start", "login", "log in", "automatic"]), \
            "Auto-Start tooltip should explain it launches the app on login"

    def test_ble_wake_tooltip_mentions_battery(self, settings_section: str):
        """BLE Wake tooltip should mention battery impact."""
        idx = settings_section.find("BLE Wake")
        window = settings_section[idx:idx + 600]
        tip_match = re.search(r'class="tip-text">(.*?)</span>', window, re.DOTALL)
        assert tip_match, "BLE Wake tooltip text not found"
        text = tip_match.group(1).lower()
        assert any(word in text for word in ["battery", "bluetooth", "ble", "wake", "power"]), \
            "BLE Wake tooltip should mention battery or Bluetooth"

    def test_debug_logging_tooltip_advises_when_to_use(self, settings_section: str):
        """Debug Logging tooltip should advise users when to enable it."""
        idx = settings_section.find("Debug Logging")
        window = settings_section[idx:idx + 600]
        tip_match = re.search(r'class="tip-text">(.*?)</span>', window, re.DOTALL)
        assert tip_match, "Debug Logging tooltip text not found"
        text = tip_match.group(1).lower()
        assert any(word in text for word in ["troubleshoot", "log", "performance", "normal"]), \
            "Debug Logging tooltip should advise when to enable it"

    def test_no_tooltip_contains_only_placeholder_text(self, settings_section: str):
        """No tooltip should contain generic placeholder text."""
        bad_phrases = ["todo", "tbd", "placeholder", "coming soon", "n/a"]
        tip_texts = self._extract_tip_texts(settings_section)
        for text in tip_texts:
            clean = text.lower()
            for phrase in bad_phrases:
                assert phrase not in clean, \
                    f"Tooltip contains placeholder text '{phrase}': {repr(text[:80])}"


# ---------------------------------------------------------------------------
# 4. Accessibility checks
# ---------------------------------------------------------------------------

class TestTooltipAccessibility:
    """Tooltips must be keyboard-accessible."""

    def test_all_tip_icons_have_tabindex(self, settings_section: str):
        """All tip-icon elements must have tabindex="0" for keyboard access."""
        # Find all tip-icon instances
        pattern = re.compile(r'<i class="tip-icon"([^>]*)>', re.DOTALL)
        icons = pattern.findall(settings_section)
        assert len(icons) > 0, "No tip-icon elements found in settings section"
        for attrs in icons:
            assert 'tabindex="0"' in attrs, \
                f"tip-icon missing tabindex=\"0\" (keyboard inaccessible): <i class=\"tip-icon\"{attrs}>"

    def test_tip_icons_use_i_element(self, settings_section: str):
        """Info icons should use <i> element (not div/span) for correct semantics."""
        # All tip icons in settings should be <i> tags
        assert '<i class="tip-icon"' in settings_section, \
            "tip-icon should use <i> element"

    def test_focus_within_rule_present(self, html: str):
        """The :focus-within rule enables keyboard navigation for tooltips."""
        assert ":focus-within" in html, \
            ":focus-within CSS rule missing — tooltips won't show on keyboard focus"


# ---------------------------------------------------------------------------
# 5. Edge-overflow prevention
# ---------------------------------------------------------------------------

class TestTooltipEdgeHandling:
    """Tooltips near the right edge should use .tip-left variant."""

    def test_tip_left_used_in_settings(self, settings_section: str):
        """At least one tooltip near the right edge should use tip-left."""
        assert "tip-left" in settings_section, \
            "No .tip-left tooltips found — right-edge overflow may occur"

    def test_debug_logging_uses_tip_left(self, settings_section: str):
        """Debug Logging (last toggle, near right) should use tip-left."""
        idx = settings_section.find("Debug Logging")
        # Search a window before the label for tip-left class
        window_start = max(0, idx - 200)
        window = settings_section[window_start:idx + 200]
        assert "tip-left" in window, \
            "Debug Logging tooltip should use .tip-left to avoid right-edge overflow"

    def test_wifi_ap_uses_tip_left(self, settings_section: str):
        """WiFi AP legend (rightmost) should use tip-left.

        The tip-left class is on the outer <span class="tip tip-left"> which
        wraps the "WiFi AP" label, so we need a larger look-behind window.
        """
        idx = settings_section.find("WiFi AP")
        # Use a 400-char look-behind to capture the containing span's class attribute
        window_start = max(0, idx - 400)
        window = settings_section[window_start:idx + 400]
        assert "tip-left" in window, \
            "WiFi AP tooltip should use .tip-left to avoid right-edge overflow"


# ---------------------------------------------------------------------------
# 6. Tooltip count sanity check
# ---------------------------------------------------------------------------

class TestTooltipCount:
    """Ensure we have the expected number of tooltips."""

    def test_settings_section_has_sufficient_tooltips(self, settings_section: str):
        """Settings section should have at least 9 distinct tooltips."""
        count = settings_section.count('class="tip-text"')
        assert count >= 9, \
            f"Expected at least 9 tooltips in settings section, found {count}. " \
            "Some settings may be missing explanatory tooltips."

    def test_settings_section_has_sufficient_tip_icons(self, settings_section: str):
        """Settings section should have at least 9 tip-icon elements."""
        count = settings_section.count('class="tip-icon"')
        assert count >= 9, \
            f"Expected at least 9 tip-icon elements in settings section, found {count}."
