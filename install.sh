#!/usr/bin/env bash
# GoMaxWebcam Installer — run ./install.sh to set up
# Finds Python 3.12+ and runs install.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Find a suitable Python
for cmd in python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "0.0")
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 12 ]; then
            exec "$cmd" "$SCRIPT_DIR/install.py" "$@"
        fi
    fi
done

echo ""
echo "  ERROR: Python 3.12+ not found."
echo "  Install Python from https://www.python.org/downloads/"
echo ""
echo "  On macOS:   brew install python@3.13"
echo "  On Ubuntu:  sudo apt install python3.13"
echo ""
exit 1
