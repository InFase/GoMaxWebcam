"""
tests/conftest.py — Global test safeguards

1. RAM LIMIT (OS-level): At session start, a Windows Job Object caps the
   process's committed virtual memory.  The OS kernel itself will refuse
   any allocation that would exceed the cap — the RAM is never consumed,
   so the machine cannot OOM or crash.  On non-Windows platforms, the
   ``resource`` module is used as a best-effort fallback.

2. RAM LIMIT (per-test): An autouse fixture logs per-test memory deltas
   and fails any test whose post-run RSS exceeds the cap.  This is a
   secondary, informational guard (the Job Object is the real protection).

3. GOPRO REQUIRED: ALL tests are auto-skipped when no GoPro camera is
   detected on USB.  Override with TEST_SKIP_GOPRO_CHECK=1.

Fixtures exposed
----------------
ram_limit_info   session-scoped dict with ``limit_mb``, ``limit_bytes``,
                 ``backend`` (``"job_object"``, ``"resource"``, or ``None``),
                 and ``active`` (bool).
"""

import gc
import logging
import os
import sys
import time

import psutil
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAM_LIMIT_MB = int(os.environ.get("TEST_RAM_LIMIT_MB", "2048"))
RAM_LIMIT_BYTES = RAM_LIMIT_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Logging — write a memory log so we can post-mortem any future leak
# ---------------------------------------------------------------------------

_LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

_mem_logger = logging.getLogger("test_memory")
_mem_logger.setLevel(logging.DEBUG)
_mem_handler = logging.FileHandler(
    os.path.join(_LOG_DIR, "test_memory.log"), mode="w", encoding="utf-8"
)
_mem_handler.setFormatter(logging.Formatter("%(asctime)s  %(message)s"))
_mem_logger.addHandler(_mem_handler)

# ---------------------------------------------------------------------------
# OS-level memory cap — implementation
# ---------------------------------------------------------------------------

_JOB_HANDLE = None

# Shared state dict exposed via the ``ram_limit_info`` fixture.
_RAM_LIMIT_STATE: dict = {
    "limit_mb": RAM_LIMIT_MB,
    "limit_bytes": RAM_LIMIT_BYTES,
    "backend": None,      # "job_object" | "resource" | None
    "active": False,
}


def _apply_job_object_memory_limit() -> bool:
    """
    Create a Windows Job Object with a hard per-process memory limit and
    assign the current process to it.

    Returns True if the limit was successfully applied, False otherwise.
    """
    global _JOB_HANDLE
    try:
        import win32api
        import win32job

        hjob = win32job.CreateJobObject(None, "")
        info = win32job.QueryInformationJobObject(
            hjob, win32job.JobObjectExtendedLimitInformation
        )
        info["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        )
        info["ProcessMemoryLimit"] = RAM_LIMIT_BYTES
        win32job.SetInformationJobObject(
            hjob, win32job.JobObjectExtendedLimitInformation, info
        )
        win32job.AssignProcessToJobObject(hjob, win32api.GetCurrentProcess())
        _JOB_HANDLE = hjob
        _mem_logger.info(
            "Job Object memory limit ACTIVE: %d MB (%d bytes)",
            RAM_LIMIT_MB, RAM_LIMIT_BYTES,
        )
        _RAM_LIMIT_STATE["backend"] = "job_object"
        _RAM_LIMIT_STATE["active"] = True
        return True
    except ImportError:
        _mem_logger.warning(
            "pywin32 not installed — OS-level memory limit NOT active. "
            "Install with: pip install pywin32"
        )
        return False
    except OSError as exc:
        _mem_logger.warning("Failed to set Job Object memory limit: %s", exc)
        return False


def _apply_resource_memory_limit() -> bool:
    """
    Best-effort memory limit via the POSIX ``resource`` module.

    Sets RLIMIT_AS (address space) which is the closest equivalent to
    the Windows Job Object per-process memory limit.

    Returns True if the limit was successfully applied, False otherwise.
    """
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        new_limit = RAM_LIMIT_BYTES
        # Don't exceed existing hard limit if one is already set.
        if hard != resource.RLIM_INFINITY:
            new_limit = min(new_limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (new_limit, hard))
        _mem_logger.info(
            "resource.RLIMIT_AS memory limit ACTIVE: %d MB (%d bytes)",
            new_limit // (1024 * 1024), new_limit,
        )
        _RAM_LIMIT_STATE["backend"] = "resource"
        _RAM_LIMIT_STATE["active"] = True
        return True
    except (ImportError, AttributeError):
        # ``resource`` doesn't exist on Windows, and some POSIX systems
        # may not support RLIMIT_AS.
        _mem_logger.warning(
            "resource module not available — no OS-level memory limit"
        )
        return False
    except (ValueError, OSError) as exc:
        _mem_logger.warning("Failed to set resource memory limit: %s", exc)
        return False


def _apply_memory_limit():
    """
    Apply the best available OS-level memory limit.

    Windows → Job Object.  POSIX → resource.RLIMIT_AS.  Fallback → warning.
    """
    if sys.platform == "win32":
        if _apply_job_object_memory_limit():
            return
        # Fall through to resource attempt (unlikely to work on Windows,
        # but costs nothing to try).

    if not _apply_resource_memory_limit():
        _mem_logger.warning(
            "No OS-level memory limit could be applied — "
            "tests will run without a hard RAM cap"
        )


# Apply immediately at import time (before any test is collected).
_apply_memory_limit()

# ---------------------------------------------------------------------------
# GoPro detection
# ---------------------------------------------------------------------------

GOPRO_USB_VID = "2672"  # GoPro vendor ID


def _gopro_connected() -> bool:
    """Return True if a GoPro camera is visible on USB (Windows-only)."""
    try:
        import subprocess

        result = subprocess.run(
            ["pnputil", "/enum-devices", "/connected"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        output_upper = result.stdout.upper()
        return GOPRO_USB_VID.upper() in output_upper or "GOPRO" in output_upper
    except Exception:
        return False


_GOPRO_AVAILABLE: bool | None = None


def gopro_is_available() -> bool:
    global _GOPRO_AVAILABLE
    if _GOPRO_AVAILABLE is None:
        _GOPRO_AVAILABLE = _gopro_connected()
    return _GOPRO_AVAILABLE


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "no_gopro_needed: mark test as not requiring GoPro hardware",
    )


# ---------------------------------------------------------------------------
# Session-scoped RAM limit fixture (for test introspection)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def ram_limit_info() -> dict:
    """
    Expose the active RAM limit configuration to tests.

    Returns a dict with keys:
        limit_mb    (int)  — configured limit in megabytes
        limit_bytes (int)  — configured limit in bytes
        backend     (str | None) — "job_object", "resource", or None
        active      (bool) — True if an OS-level limit is enforced
    """
    return dict(_RAM_LIMIT_STATE)


# ---------------------------------------------------------------------------
# Collection hook — skip ALL tests when no GoPro
# ---------------------------------------------------------------------------

def pytest_collection_modifyitems(config, items):
    """Skip every test when no GoPro is connected."""
    override = os.environ.get("TEST_SKIP_GOPRO_CHECK", "0") == "1"
    gopro = gopro_is_available()

    _mem_logger.info("GoPro detected: %s | override: %s | tests collected: %d",
                     gopro, override, len(items))

    if gopro or override:
        return

    reason = (
        "GoPro not detected on USB — ALL tests skipped. "
        "Set TEST_SKIP_GOPRO_CHECK=1 to force-run."
    )
    skip_marker = pytest.mark.skip(reason=reason)

    for item in items:
        if "no_gopro_needed" not in [m.name for m in item.iter_markers()]:
            item.add_marker(skip_marker)


# ---------------------------------------------------------------------------
# Per-test memory fixture (informational + secondary guard)
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _enforce_ram_limit():
    """Log memory before/after each test; fail if RSS exceeds cap."""
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    test_name = os.environ.get("PYTEST_CURRENT_TEST", "unknown")
    _mem_logger.debug(
        "START  %s  RSS=%.1f MB", test_name, rss_before / (1024 ** 2)
    )
    t0 = time.perf_counter()

    yield

    # Force garbage collection to reclaim numpy arrays, mock frames, etc.
    gc.collect()

    elapsed = time.perf_counter() - t0
    rss_after = proc.memory_info().rss
    delta = rss_after - rss_before
    _mem_logger.debug(
        "END    %s  RSS=%.1f MB  delta=%+.1f MB  elapsed=%.2fs",
        test_name, rss_after / (1024 ** 2), delta / (1024 ** 2), elapsed,
    )

    if rss_after > RAM_LIMIT_BYTES:
        msg = (
            f"RAM limit exceeded! RSS={rss_after / (1024 ** 3):.2f} GB "
            f"(limit={RAM_LIMIT_MB} MB, delta={delta / (1024 ** 2):+.1f} MB)"
        )
        _mem_logger.error(msg)
        pytest.fail(msg)
