"""
utils.py — Helper functions for GoPro Bridge

Delegates logging setup to logger.py (the single source of truth).
Contains general-purpose utility functions used across the app.
"""

from pathlib import Path

# Re-export for backward compatibility — prefer importing from logger.py directly
from logger import setup_logger as setup_logging  # noqa: F401

__all__ = ["setup_logging"]
