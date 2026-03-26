# Contributing to GoMaxWebcam

Thanks for your interest in contributing. This document covers the development setup, testing workflow, code style, and pull request process.

---

## Development Setup

**Prerequisites:** Python 3.12+, Git

```bash
git clone https://github.com/InFase/GoMaxWebcam.git
cd GoMaxWebcam
pip install -e ".[dev]"
```

This installs the project in editable mode along with development dependencies (pytest, pytest-asyncio, pytest-cov, ruff).

---

## Running Tests

### Safe Test Runner (Recommended)

GoMaxWebcam uses a custom test runner that enforces OS-level RAM limits via Windows Job Objects. This exists because a past runaway test consumed 23 GB of memory and crashed the machine.

```bash
python run_tests.py                         # Default: 2 GB limit
python run_tests.py --limit-mb 1500         # Custom limit
python run_tests.py -- -k test_frame        # Pass extra args to pytest
python run_tests.py --force-run             # Skip GoPro hardware check
```

On non-Windows platforms, the runner falls back to a plain `pytest` invocation (no Job Object).

### Direct pytest

You can also run pytest directly, but you lose the memory safety net:

```bash
pytest tests/v2/ -m "not hardware" -v --tb=short
```

### Test Markers

| Marker | Meaning |
|--------|---------|
| `no_gopro_needed` | Runs without any physical hardware |
| `hardware` | Requires a GoPro connected via USB |

Tests that need hardware are automatically skipped when no GoPro is detected (unless `--force-run` or `TEST_SKIP_GOPRO_CHECK=1` is set).

---

## Code Style

GoMaxWebcam uses [ruff](https://docs.astral.sh/ruff/) for linting.

```bash
ruff check src/gomaxwebcam/
```

Configuration (from `pyproject.toml`):

- **Target:** Python 3.12
- **Line length:** 100 characters
- **Rules:** `E` (pycodestyle errors), `F` (pyflakes), `I` (isort), `N` (pep8-naming), `W` (pycodestyle warnings)

Please run `ruff check` before submitting a PR. CI will also run it automatically.

---

## Architecture for New Contributors

GoMaxWebcam v2 is an asyncio application with these main layers:

1. **Entry point** (`__main__.py`) -- CLI parsing, single-instance lock, uvicorn server on a background thread, pystray on a daemon thread.
2. **Orchestrator** (`orchestrator.py`) -- Owns the lifecycle of all components. Creates the FastAPI app with lifespan-managed startup/shutdown.
3. **TransportManager** (`transport_manager.py`) -- Manages USB, COHN, and WiFi AP transports with priority-based failover and fail-back.
4. **FramePipeline** (`pipeline/frame_pipeline.py`) -- Receives MPEG-TS UDP packets, decodes frames via PyAV (with ffmpeg subprocess fallback), and outputs BGR24 frames to the virtual camera via pyvirtualcam.
5. **EventBus** (`events.py`) -- Async pub/sub for decoupled communication between components. Dashboard SSE uses this to push real-time updates to the browser.
6. **Dashboard** (`dashboard/`) -- FastAPI app serving a single-page Alpine.js application with SSE for live status updates.
7. **BLE** (`ble/`) -- Bluetooth Low Energy scanner and GATT client for COHN provisioning.
8. **Config** (`config.py`) -- TOML-based configuration with dataclass sections, validation, and OS keyring integration for secrets.

The general flow:

```
User starts app
  -> __main__.py acquires instance lock
  -> Orchestrator creates all components
  -> TransportManager tries USB, falls back to COHN/WiFi
  -> FramePipeline decodes stream, outputs to virtual camera
  -> Dashboard shows live status via SSE
  -> User's video app sees "Unity Video Capture" as a webcam
```

---

## Pull Request Process

1. **Fork the repo** and create a feature branch from `v2-rewrite`.
2. **Make your changes.** Keep diffs small and focused on one thing.
3. **Add or update tests** for any new behavior. Tests go in `tests/v2/`.
4. **Run the linter and tests locally:**
   ```bash
   ruff check src/gomaxwebcam/
   python run_tests.py --force-run
   ```
5. **Open a PR** against `v2-rewrite` (or `main` for hotfixes).
6. **Describe what you changed and why** in the PR description.

### What Makes a Good PR

- Small, focused changes (one concern per PR)
- Tests included for new behavior
- Linter passes (`ruff check`)
- No secrets or credentials in committed files
- No unnecessary dependency additions

---

## Reporting Bugs

Open an issue on GitHub with:

- What you expected to happen
- What actually happened
- Steps to reproduce
- Your OS, Python version, and GoPro model
- Relevant log output (run with `--debug` to get verbose logs)

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
