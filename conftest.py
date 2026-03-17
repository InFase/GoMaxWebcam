"""
Root conftest.py — Ensures 'src/' is on sys.path so module imports
(e.g. 'from logger import get_logger') work when running pytest from
the project root.
"""

import sys
import os

# Add src/ to path so internal imports like 'from logger import get_logger' resolve
src_dir = os.path.join(os.path.dirname(__file__), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
