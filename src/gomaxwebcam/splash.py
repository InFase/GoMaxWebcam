"""
splash.py -- Branded splash frame for GoMaxWebcam v2.

Generates a 1920x1080 splash frame with "GoMaxWebcam" text and
"Connecting..." subtitle. Uses PIL/Pillow when available for proper
text rendering, falls back to a plain dark frame otherwise.

Returns numpy arrays in BGR24 format compatible with pyvirtualcam.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger("gomaxwebcam.splash")

# Brand colors
BG_COLOR_BGR = (0x2E, 0x1A, 0x1A)  # #1a1a2e in BGR
TEXT_COLOR_BGR = (0xFF, 0xFF, 0xFF)  # white in BGR
SUBTITLE_COLOR_BGR = (0xAA, 0xAA, 0xAA)  # light grey in BGR

# Default resolution
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080


def generate_splash_frame(
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    title: str = "GoMaxWebcam",
    subtitle: str = "Connecting...",
) -> np.ndarray:
    """Generate a branded splash frame as a numpy array (BGR24).

    Tries PIL/Pillow for proper text rendering. Falls back to a plain
    dark frame if PIL is not available.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        title: Main title text.
        subtitle: Subtitle text below the title.

    Returns:
        numpy array of shape (height, width, 3) in BGR24 format.
    """
    frame = _try_pil_splash(width, height, title, subtitle)
    if frame is not None:
        return frame

    return _plain_splash(width, height)


def _try_pil_splash(
    width: int,
    height: int,
    title: str,
    subtitle: str,
) -> Optional[np.ndarray]:
    """Generate splash frame using PIL/Pillow."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        log.debug("PIL not available, using plain splash")
        return None

    try:
        # Create RGB image with dark background
        bg_rgb = (BG_COLOR_BGR[2], BG_COLOR_BGR[1], BG_COLOR_BGR[0])
        img = Image.new("RGB", (width, height), bg_rgb)
        draw = ImageDraw.Draw(img)

        text_rgb = (TEXT_COLOR_BGR[2], TEXT_COLOR_BGR[1], TEXT_COLOR_BGR[0])
        sub_rgb = (SUBTITLE_COLOR_BGR[2], SUBTITLE_COLOR_BGR[1], SUBTITLE_COLOR_BGR[0])

        # Try to load a reasonable font size
        title_font = _get_font(size=max(48, height // 18))
        subtitle_font = _get_font(size=max(28, height // 30))

        # Draw title centered
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_w = title_bbox[2] - title_bbox[0]
        title_h = title_bbox[3] - title_bbox[1]
        title_x = (width - title_w) // 2
        title_y = (height // 2) - title_h - 20
        draw.text((title_x, title_y), title, fill=text_rgb, font=title_font)

        # Draw subtitle centered below title
        sub_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        sub_w = sub_bbox[2] - sub_bbox[0]
        sub_x = (width - sub_w) // 2
        sub_y = title_y + title_h + 30
        draw.text((sub_x, sub_y), subtitle, fill=sub_rgb, font=subtitle_font)

        # Convert RGB -> BGR for pyvirtualcam
        arr = np.array(img)
        return arr[:, :, ::-1].copy()

    except Exception as e:
        log.debug("PIL splash generation failed: %s", e)
        return None


def _get_font(size: int):
    """Try to load a TrueType font at the given size, fall back to default."""
    from PIL import ImageFont

    # Try common system font paths
    candidates = [
        "arial.ttf",
        "Arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
        "segoeui.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except (OSError, IOError):
            continue

    # Fall back to default bitmap font
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        # Older Pillow versions don't accept size argument
        return ImageFont.load_default()


def _plain_splash(width: int, height: int) -> np.ndarray:
    """Generate a plain dark frame (fallback when PIL is unavailable)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = BG_COLOR_BGR
    return frame
