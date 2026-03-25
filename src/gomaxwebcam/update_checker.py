"""
update_checker.py -- GitHub release checker for GoMaxWebcam v2.

On startup, queries the GitHub Releases API for the latest release.
If a newer version is available, publishes an event to the EventBus
with the download URL. Silently skips on network errors or timeouts.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from gomaxwebcam.events import EventBus, Event, EventType

log = logging.getLogger("gomaxwebcam.update_checker")

# GitHub API endpoint -- replace USER with actual org/user when public
RELEASES_URL = "https://api.github.com/repos/USER/GoMaxWebcam/releases/latest"
REQUEST_TIMEOUT_S = 3.0


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Parse a version string like 'v1.2.3' or '1.2.3' into a tuple of ints."""
    cleaned = version_str.lstrip("vV").strip()
    parts = re.split(r"[.\-]", cleaned)
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            break
    return tuple(result) if result else (0,)


def _get_current_version() -> str:
    """Read the current app version from the package."""
    try:
        from gomaxwebcam import __version__  # type: ignore[attr-defined]
        return __version__
    except (ImportError, AttributeError):
        pass

    # Fallback: try the top-level src/__init__.py
    try:
        import importlib.metadata
        return importlib.metadata.version("gomaxwebcam")
    except Exception:
        pass

    return "0.0.0"


async def check_for_updates(
    event_bus: EventBus,
    current_version: Optional[str] = None,
    releases_url: str = RELEASES_URL,
) -> None:
    """Check GitHub for a newer release and publish event if found.

    Designed to be called once at startup. Silently returns on any
    network error, timeout, or parse failure.

    Args:
        event_bus: EventBus to publish update-available event.
        current_version: Override current version string. None = auto-detect.
        releases_url: GitHub Releases API URL.
    """
    version = current_version or _get_current_version()

    try:
        import aiohttp
    except ImportError:
        log.debug("aiohttp not installed, skipping update check")
        return

    try:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_S)
        headers = {"Accept": "application/vnd.github.v3+json"}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(releases_url, headers=headers) as resp:
                if resp.status != 200:
                    log.debug("Update check returned HTTP %d", resp.status)
                    return

                data = await resp.json(content_type=None)

        tag = data.get("tag_name", "")
        html_url = data.get("html_url", "")

        if not tag:
            return

        current_tuple = _parse_version(version)
        latest_tuple = _parse_version(tag)

        if latest_tuple > current_tuple:
            log.info(
                "Update available: %s -> %s (%s)",
                version, tag, html_url,
            )
            event_bus.publish(Event(
                type=EventType.STATUS,
                data={
                    "update_available": True,
                    "current_version": version,
                    "latest_version": tag,
                    "download_url": html_url,
                },
            ))
        else:
            log.debug("Up to date (%s >= %s)", version, tag)

    except Exception:
        # Silently skip on any error (network, JSON parse, etc.)
        log.debug("Update check failed", exc_info=True)
