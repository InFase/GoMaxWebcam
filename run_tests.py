"""
run_tests.py — Safe test runner with OS-enforced memory limits.

Spawns pytest inside a Windows Job Object so the OS kernel will refuse
any allocation that would push committed memory past the cap.  The limit
is set BEFORE pytest imports a single module, so even import-time leaks
are caught.

Usage:
    python run_tests.py                        # defaults: 2 GB limit
    python run_tests.py --limit-mb 1500        # custom limit
    python run_tests.py -- -k test_frame       # pass args to pytest
    python run_tests.py --force-run             # ignore GoPro check
    python run_tests.py --limit-mb 1024 -- -x  # combine flags
"""

import argparse
import logging
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(LOG_DIR, "test_runner.log"), mode="a", encoding="utf-8"
        ),
    ],
)
log = logging.getLogger("run_tests")

# ---------------------------------------------------------------------------
# GoPro check (duplicated here so we can bail before spawning anything)
# ---------------------------------------------------------------------------

GOPRO_USB_VID = "2672"


def gopro_connected() -> bool:
    try:
        result = subprocess.run(
            ["pnputil", "/enum-devices", "/connected"],
            capture_output=True, text=True, timeout=10,
        )
        upper = result.stdout.upper()
        return GOPRO_USB_VID.upper() in upper or "GOPRO" in upper
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Job Object launcher (Windows only)
# ---------------------------------------------------------------------------

def run_with_job_object(cmd: list[str], limit_bytes: int) -> int:
    """
    Spawn *cmd* as a suspended process, assign it to a memory-limited
    Job Object, then resume it.  Returns the process exit code.
    """
    import win32api
    import win32con
    import win32event
    import win32job
    import win32process

    # 1. Create job with memory limit
    hjob = win32job.CreateJobObject(None, "")
    info = win32job.QueryInformationJobObject(
        hjob, win32job.JobObjectExtendedLimitInformation
    )
    info["BasicLimitInformation"]["LimitFlags"] |= (
        win32job.JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    )
    info["ProcessMemoryLimit"] = limit_bytes
    win32job.SetInformationJobObject(
        hjob, win32job.JobObjectExtendedLimitInformation, info
    )
    log.info("Job Object created — memory limit: %d MB", limit_bytes // (1024 * 1024))

    # 2. Spawn suspended
    si = win32process.STARTUPINFO()
    CREATE_SUSPENDED = 0x00000004
    hproc, hthread, pid, tid = win32process.CreateProcess(
        None,               # appName
        subprocess.list2cmdline(cmd),
        None, None,         # security
        True,               # inherit handles (so pytest output works)
        CREATE_SUSPENDED,
        None,               # env (inherit parent)
        None,               # cwd
        si,
    )
    log.info("pytest spawned (PID %d) — suspended, assigning to job ...", pid)

    # 3. Assign to job BEFORE resuming
    win32job.AssignProcessToJobObject(hjob, hproc)

    # 4. Resume
    win32process.ResumeThread(hthread)
    win32api.CloseHandle(hthread)
    log.info("pytest resumed inside Job Object")

    # 5. Wait
    INFINITE = 0xFFFFFFFF
    win32event.WaitForSingleObject(hproc, INFINITE)
    exit_code = win32process.GetExitCodeProcess(hproc)
    win32api.CloseHandle(hproc)
    win32api.CloseHandle(hjob)
    return exit_code


# ---------------------------------------------------------------------------
# Fallback (non-Windows or pywin32 missing)
# ---------------------------------------------------------------------------

def run_plain(cmd: list[str]) -> int:
    log.warning("No Job Object available — running pytest without OS memory limit!")
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run pytest with OS-enforced memory limits.",
    )
    parser.add_argument(
        "--limit-mb", type=int, default=2048,
        help="Max memory in MB (default: 2048)",
    )
    parser.add_argument(
        "--force-run", action="store_true",
        help="Run tests even if no GoPro is detected",
    )
    args, pytest_args = parser.parse_known_args()

    # Strip leading "--" separator if present
    if pytest_args and pytest_args[0] == "--":
        pytest_args = pytest_args[1:]

    limit_bytes = args.limit_mb * 1024 * 1024

    log.info("=" * 60)
    log.info("Test runner started — limit: %d MB", args.limit_mb)

    # GoPro check
    gopro = gopro_connected()
    log.info("GoPro detected: %s", gopro)
    if not gopro and not args.force_run:
        log.info(
            "No GoPro connected — tests will be skipped by conftest.py. "
            "Use --force-run to override."
        )

    # Build pytest command
    cmd = [sys.executable, "-m", "pytest"] + pytest_args
    if args.force_run and not gopro:
        os.environ["TEST_SKIP_GOPRO_CHECK"] = "1"
    os.environ["TEST_RAM_LIMIT_MB"] = str(args.limit_mb)

    log.info("Command: %s", subprocess.list2cmdline(cmd))
    t0 = time.perf_counter()

    # Run
    if sys.platform == "win32":
        try:
            import win32job  # noqa: F401
            exit_code = run_with_job_object(cmd, limit_bytes)
        except ImportError:
            log.warning("pywin32 not installed — falling back to plain execution")
            exit_code = run_plain(cmd)
    else:
        exit_code = run_plain(cmd)

    elapsed = time.perf_counter() - t0
    log.info("pytest exited with code %d in %.1fs", exit_code, elapsed)
    log.info("=" * 60)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
