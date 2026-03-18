"""
test_live.py -- Live integration test harness for GoProBridge

Requires a GoPro connected via USB. Tests every state transition
with precise timing measurements and verbose logging.

Usage:
    python test_live.py              # Run all tests
    python test_live.py --test 3     # Run specific test
    python test_live.py --list       # List available tests
"""

import sys
import os
import time
import logging
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import Config
from gopro_connection import GoProConnection, WebcamStatus, WebcamError, ConnectionState
from stream_reader import StreamReader
from frame_pipeline import FramePipeline
from frame_buffer import FrameBuffer
from virtual_camera import VirtualCamera

# ── Logging setup ──

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d [%(levelname)-5s] %(name)-20s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_live")

# Suppress noisy loggers
for name in ("urllib3", "charset_normalizer"):
    logging.getLogger(name).setLevel(logging.WARNING)


# ── Timing helper ──

class Timer:
    def __init__(self, label):
        self.label = label
        self.start = None
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        log.info("START: %s", self.label)
        return self

    def __exit__(self, *args):
        self.elapsed = (time.perf_counter() - self.start) * 1000
        log.info("END:   %s -- %.1f ms", self.label, self.elapsed)

    @property
    def ms(self):
        return self.elapsed


# ── Test results ──

class TestResult:
    def __init__(self):
        self.tests = []
        self.passed = 0
        self.failed = 0
        self.errors = []

    def record(self, name, passed, elapsed_ms, detail=""):
        status = "PASS" if passed else "FAIL"
        self.tests.append((name, status, elapsed_ms, detail))
        if passed:
            self.passed += 1
        else:
            self.failed += 1
            self.errors.append((name, detail))

    def summary(self):
        print("\n" + "=" * 70)
        print("TEST RESULTS")
        print("=" * 70)
        for name, status, ms, detail in self.tests:
            marker = "OK" if status == "PASS" else "FAIL"
            line = f"  [{marker:4s}] {name:50s} {ms:8.1f} ms"
            if detail:
                line += f"  ({detail})"
            print(line)
        print("-" * 70)
        print(f"  {self.passed} passed, {self.failed} failed, {len(self.tests)} total")
        if self.errors:
            print("\nFAILURES:")
            for name, detail in self.errors:
                print(f"  - {name}: {detail}")
        print("=" * 70)
        return self.failed == 0


# ── Test fixtures ──

def create_gopro(config=None):
    config = config or Config.load()
    gopro = GoProConnection(config)
    return gopro, config


def discover_and_connect(gopro):
    with Timer("discover") as t:
        ok = gopro.discover()
    if not ok:
        raise RuntimeError("GoPro not found on USB")
    log.info("Discovered: %s at %s", gopro.device_info.description, gopro.device_info.camera_ip)

    with Timer("open_connection") as t:
        ok = gopro.open_connection(gopro.device_info)
    if not ok:
        raise RuntimeError("Failed to open connection")
    return t.ms


def start_webcam(gopro, res=7, fov=4):
    with Timer(f"start_webcam(res={res}, fov={fov})") as t:
        ok = gopro.start_webcam(resolution=res, fov=fov)
    return ok, t.ms


def stop_webcam(gopro):
    with Timer("stop_webcam") as t:
        ok = gopro.stop_webcam()
    return ok, t.ms


def exit_webcam(gopro):
    with Timer("exit_webcam") as t:
        ok = gopro.exit_webcam()
    return ok, t.ms


def query_status(gopro):
    with Timer("webcam_status") as t:
        status = gopro.webcam_status()
    log.info("Camera status: %s (%d) -- %.1f ms", status.name, status.value, t.ms)
    return status, t.ms


def start_ffmpeg(config):
    reader = StreamReader(config)
    with Timer("ffmpeg_start") as t:
        ok = reader.start()
    if not ok:
        raise RuntimeError("Failed to start ffmpeg")
    return reader, t.ms


def wait_first_frame(reader, timeout=10.0):
    with Timer("first_frame_wait") as t:
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            frame = reader.read_frame()
            if frame is not None:
                log.info("First frame: shape=%s, elapsed=%.1f ms", frame.shape, t.ms)
                return frame, (time.perf_counter() - start) * 1000
            time.sleep(0.05)
    log.warning("No frame received within %.1f s", timeout)
    return None, timeout * 1000


# ── Individual tests ──

def test_01_discovery(results):
    """Test GoPro USB discovery speed"""
    gopro, config = create_gopro()
    with Timer("full discovery") as t:
        ok = gopro.discover()
    results.record("01_discovery", ok, t.ms,
                   f"device={gopro.device_info.description if ok else 'none'}")
    return gopro, config


def test_02_connection(results, gopro):
    """Test connection + API verification speed"""
    ms = discover_and_connect(gopro)
    results.record("02_connection", True, ms)


def test_03_webcam_start(results, gopro):
    """Test webcam start speed (should accept status 2 immediately)"""
    ok, ms = start_webcam(gopro, res=7, fov=4)
    target = 3000  # Should be under 3s with preview warmup
    results.record("03_webcam_start", ok and ms < target, ms,
                   f"target<{target}ms")


def test_04_status_check(results, gopro):
    """Test that status 2 (READY) is returned and accepted"""
    status, ms = query_status(gopro)
    ok = status in (WebcamStatus.READY, WebcamStatus.STREAMING)
    results.record("04_status_is_streaming", ok, ms,
                   f"status={status.name}")


def test_05_ffmpeg_first_frame(results, config):
    """Test ffmpeg startup and first frame latency"""
    reader, start_ms = start_ffmpeg(config)
    frame, frame_ms = wait_first_frame(reader, timeout=10.0)
    total = start_ms + frame_ms
    ok = frame is not None
    results.record("05_ffmpeg_first_frame", ok, total,
                   f"ffmpeg_start={start_ms:.0f}ms, first_frame={frame_ms:.0f}ms")
    reader.stop()


def test_06_stop_webcam(results, gopro):
    """Test stop speed and intentional flag"""
    ok, ms = stop_webcam(gopro)
    flag = gopro._last_stop_was_intentional
    results.record("06_stop_webcam", ok and flag, ms,
                   f"intentional_flag={flag}")


def test_07_restart_skips_idle(results, gopro):
    """Test that restart after intentional stop skips IDLE workaround"""
    # Flag should be True from test_06
    log.info("Intentional flag before restart: %s", gopro._last_stop_was_intentional)
    ok, ms = start_webcam(gopro, res=7, fov=4)
    # Should be fast since IDLE workaround is skipped (no 2s delay)
    target = 3000
    results.record("07_restart_skips_idle", ok and ms < target, ms,
                   f"target<{target}ms")


def test_08_resolution_change(results, gopro, config):
    """Test resolution change via stop+start (not exit)"""
    with Timer("resolution_change_720_to_1080") as t:
        stop_webcam(gopro)
        time.sleep(max(config.idle_reset_delay, 1.0))
        ok, _ = start_webcam(gopro, res=12, fov=4)
    target = 5000
    results.record("08_resolution_change", ok and t.ms < target, t.ms,
                   f"target<{target}ms")

    # Change back
    stop_webcam(gopro)
    time.sleep(max(config.idle_reset_delay, 1.0))
    start_webcam(gopro, res=7, fov=4)


def test_09_pipeline_freeze_drain(results, config, gopro):
    """Test that freeze-frame mode drains ffmpeg stdout (no crash)"""
    reader, _ = start_ffmpeg(config)
    frame, _ = wait_first_frame(reader, timeout=10.0)
    if frame is None:
        results.record("09_pipeline_freeze_drain", False, 0, "no first frame")
        reader.stop()
        return

    buf = FrameBuffer(width=config.stream_width, height=config.stream_height)
    vcam = VirtualCamera(config)
    vcam_ok = vcam.start()
    if not vcam_ok:
        results.record("09_pipeline_freeze_drain", False, 0, "vcam failed to start")
        reader.stop()
        return

    pipeline = FramePipeline(config)
    pipeline.start(reader, vcam, buf)
    time.sleep(2.0)  # Let it stream for 2s

    # Enter freeze-frame (drain thread should start)
    with Timer("enter_freeze_frame") as t:
        pipeline.enter_freeze_frame()
    log.info("Freeze entered in %.1f ms", t.ms)

    # Wait 3 seconds in freeze mode -- ffmpeg should NOT crash
    log.info("Holding freeze for 3 seconds (drain thread should keep ffmpeg alive)...")
    time.sleep(3.0)

    ffmpeg_alive = reader.is_running
    log.info("After 3s freeze: ffmpeg alive=%s", ffmpeg_alive)

    # Exit freeze -- should resume immediately
    with Timer("exit_freeze_frame") as t:
        pipeline.exit_freeze_frame()
    log.info("Freeze exited in %.1f ms", t.ms)

    time.sleep(1.0)  # Give it a moment to read fresh frames
    pipeline_running = pipeline.is_running

    results.record("09_pipeline_freeze_drain",
                   ffmpeg_alive and pipeline_running, t.ms,
                   f"ffmpeg_alive={ffmpeg_alive}, pipeline={pipeline_running}")

    pipeline.stop()
    vcam.stop()
    reader.stop()


def test_10_instant_pause_resume(results, config, gopro):
    """Test instant pause/resume (the key spec requirement)"""
    reader, _ = start_ffmpeg(config)
    frame, _ = wait_first_frame(reader, timeout=10.0)
    if frame is None:
        results.record("10_instant_pause_resume", False, 0, "no first frame")
        reader.stop()
        return

    buf = FrameBuffer(width=config.stream_width, height=config.stream_height)
    vcam = VirtualCamera(config)
    vcam.start()

    pipeline = FramePipeline(config)
    pipeline.start(reader, vcam, buf)
    time.sleep(2.0)

    # Pause (enter freeze)
    with Timer("pause (enter_freeze_frame)") as t_pause:
        pipeline.enter_freeze_frame()
    log.info("Pause: %.1f ms", t_pause.ms)

    time.sleep(2.0)  # Hold pause

    # Resume (exit freeze)
    with Timer("resume (exit_freeze_frame)") as t_resume:
        pipeline.exit_freeze_frame()
    log.info("Resume: %.1f ms", t_resume.ms)

    time.sleep(1.0)

    pause_ok = t_pause.ms < 100
    resume_ok = t_resume.ms < 100
    ffmpeg_alive = reader.is_running

    results.record("10_instant_pause_resume",
                   pause_ok and resume_ok and ffmpeg_alive,
                   t_pause.ms + t_resume.ms,
                   f"pause={t_pause.ms:.0f}ms, resume={t_resume.ms:.0f}ms, "
                   f"target<100ms each, ffmpeg={ffmpeg_alive}")

    pipeline.stop()
    vcam.stop()
    reader.stop()


def test_11_error_code_handling(results, gopro):
    """Test that error codes are correctly identified"""
    # Just verify the enum values are correct
    ok = (WebcamError.SHUTTER == 4 and
          WebcamError.UNAVAILABLE == 7 and
          WebcamError.INVALID_PARAM == 6 and
          WebcamError.SUCCESS == 0)
    results.record("11_error_code_enum", ok, 0,
                   f"SHUTTER={WebcamError.SHUTTER}, UNAVAIL={WebcamError.UNAVAILABLE}")


def test_12_full_cycle(results, gopro, config):
    """Test complete cycle: start -> pause -> resume -> charge -> resume -> stop"""
    # Ensure webcam is started
    ok, _ = start_webcam(gopro, res=7, fov=4)
    if not ok:
        results.record("12_full_cycle", False, 0, "webcam start failed")
        return

    reader, _ = start_ffmpeg(config)
    frame, _ = wait_first_frame(reader, timeout=10.0)
    if frame is None:
        results.record("12_full_cycle", False, 0, "no first frame")
        reader.stop()
        return

    buf = FrameBuffer(width=config.stream_width, height=config.stream_height)
    vcam = VirtualCamera(config)
    vcam.start()
    pipeline = FramePipeline(config)
    pipeline.start(reader, vcam, buf)
    time.sleep(2.0)

    total_start = time.perf_counter()

    # Step 1: Pause
    with Timer("cycle: pause"):
        pipeline.enter_freeze_frame()
    time.sleep(1.0)

    # Step 2: Resume from pause (instant)
    with Timer("cycle: resume from pause") as t_resume:
        pipeline.exit_freeze_frame()
    time.sleep(1.0)
    resume_fast = t_resume.ms < 100

    # Step 3: Enter charge mode (stop everything)
    with Timer("cycle: enter charge mode"):
        pipeline.enter_freeze_frame()
        reader.stop()
        gopro.stop_webcam()
        gopro.exit_webcam()
    time.sleep(1.0)

    # Step 4: Resume from charge (full restart)
    with Timer("cycle: resume from charge") as t_charge:
        time.sleep(max(config.idle_reset_delay, 2.0))
        ok_start, _ = start_webcam(gopro, res=7, fov=4)
        if ok_start:
            reader2, _ = start_ffmpeg(config)
            frame2, _ = wait_first_frame(reader2, timeout=10.0)
        else:
            reader2 = None
            frame2 = None

    charge_ok = ok_start and frame2 is not None

    total = (time.perf_counter() - total_start) * 1000

    results.record("12_full_cycle",
                   resume_fast and charge_ok, total,
                   f"pause_resume={t_resume.ms:.0f}ms(<100ms), "
                   f"charge_resume={t_charge.ms:.0f}ms, "
                   f"charge_ok={charge_ok}")

    # Cleanup
    if pipeline.is_running:
        pipeline.stop()
    vcam.stop()
    if reader2:
        reader2.stop()
    gopro.stop_webcam()
    gopro.exit_webcam()


# ── Main ──

def run_all_tests():
    results = TestResult()

    try:
        # Setup
        gopro, config = test_01_discovery(results)
        if not gopro.device_info:
            print("ABORT: No GoPro found")
            return results

        test_02_connection(results, gopro)
        test_03_webcam_start(results, gopro)
        test_04_status_check(results, gopro)
        test_05_ffmpeg_first_frame(results, config)
        test_06_stop_webcam(results, gopro)
        test_07_restart_skips_idle(results, gopro)
        test_08_resolution_change(results, gopro, config)

        # Stop webcam for pipeline tests
        gopro.stop_webcam()
        gopro.exit_webcam()
        time.sleep(1.0)
        start_webcam(gopro, res=7, fov=4)

        test_09_pipeline_freeze_drain(results, config, gopro)

        # Restart for next test
        gopro.stop_webcam()
        gopro.exit_webcam()
        time.sleep(1.0)
        start_webcam(gopro, res=7, fov=4)

        test_10_instant_pause_resume(results, config, gopro)

        test_11_error_code_handling(results, gopro)

        # Restart for full cycle
        gopro.stop_webcam()
        gopro.exit_webcam()
        time.sleep(1.0)

        test_12_full_cycle(results, gopro, config)

    except Exception as e:
        log.exception("Test harness crashed: %s", e)
        results.record("HARNESS_CRASH", False, 0, str(e))

    finally:
        # Final cleanup
        try:
            gopro.stop_webcam()
        except Exception:
            pass
        try:
            gopro.exit_webcam()
        except Exception:
            pass

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GoProBridge live integration tests")
    parser.add_argument("--list", action="store_true", help="List available tests")
    args = parser.parse_args()

    if args.list:
        tests = [f for f in dir() if f.startswith("test_")]
        for t in sorted(tests):
            doc = globals()[t].__doc__ or ""
            print(f"  {t}: {doc.strip()}")
        sys.exit(0)

    print("=" * 70)
    print("GoProBridge Live Integration Tests")
    print("Requires GoPro connected via USB")
    print("=" * 70)
    print()

    results = run_all_tests()
    all_passed = results.summary()
    sys.exit(0 if all_passed else 1)
