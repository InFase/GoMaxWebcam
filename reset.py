"""Reset GoMaxWebcam — wipe all stored credentials, config, and lock files.

Usage: python reset.py
"""

import os
import sys
import shutil
import time
from pathlib import Path


def main():
    print("=" * 50)
    print("  GoMaxWebcam — Full Reset")
    print("=" * 50)

    # Kill running GoMaxWebcam processes
    try:
        import psutil
        my_pid = os.getpid()
        killed = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = " ".join(proc.info["cmdline"] or [])
                if "gomaxwebcam" in cmdline.lower() and "reset" not in cmdline.lower() and proc.pid != my_pid:
                    proc.kill()
                    killed += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        print(f"[+] Killed {killed} process(es)")
    except ImportError:
        print("[*] psutil not available, skipping process kill")

    time.sleep(1)

    # Known paths where GoMaxWebcam stores data
    home = Path.home()
    localappdata = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
    temp = Path(os.environ.get("TEMP", home / "AppData" / "Local" / "Temp"))
    project = Path(__file__).parent

    # Windows Store Python puts config here
    packages = localappdata / "Packages"
    store_python_dirs = list(packages.glob("PythonSoftwareFoundation.Python.*")) if packages.exists() else []

    targets = []

    # Config directories
    for base in [localappdata, temp]:
        targets.append(base / "GoMaxWebcam-v2")

    # Store Python config dirs
    for spd in store_python_dirs:
        targets.append(spd / "LocalCache" / "Local" / "GoMaxWebcam-v2")

    # cohn_db.json in project root
    targets.append(project / "cohn_db.json")

    # Lock files
    targets.append(temp / "GoMaxWebcam-v2" / "GoMaxWebcam-v2" / "gomaxwebcam.lock")

    removed = 0
    for t in targets:
        if t.exists():
            try:
                if t.is_dir():
                    shutil.rmtree(t)
                else:
                    t.unlink()
                print(f"  [-] {t}")
                removed += 1
            except Exception as e:
                print(f"  [!] {t}: {e}")

    if removed == 0:
        print("  (nothing to remove — already clean)")

    print()
    print("=" * 50)
    print(f"  Reset complete — removed {removed} item(s)")
    print("=" * 50)
    print()
    print("Start the app:  python -m gomaxwebcam --debug")


if __name__ == "__main__":
    main()
