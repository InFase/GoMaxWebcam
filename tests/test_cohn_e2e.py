"""
test_cohn_e2e.py — End-to-end user flow tests over COHN WiFi.

Tests all real user scenarios: start, stop, restart, resolution switch,
health checks, rapid cycling, sustained streaming, latency.

Requires: GoPro on WiFi with COHN provisioned, cohn_db.json present.
"""

import asyncio
import json
import time

import httpx

from gomaxwebcam.pipeline.decode import UDPDecoder


# Load COHN credentials
with open("cohn_db.json") as f:
    db = json.load(f)
creds = list(db["_default"].values())[-1]["credentials"]
IP = creds["ip_address"]
PASSWORD = creds["password"]

results = []


def record(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))


def get_client():
    return httpx.AsyncClient(
        base_url=f"https://{IP}",
        auth=("gopro", PASSWORD),
        verify=False,
        timeout=15.0,
    )


async def webcam_status(client):
    r = await client.get("/gopro/webcam/status")
    return r.json()


async def webcam_start(client, res=12):
    r = await client.get(
        "/gopro/webcam/start",
        params={"res": res, "fov": 0, "port": 8554, "protocol": "TS"},
    )
    return r.json()


async def webcam_stop(client):
    try:
        r = await client.get("/gopro/webcam/stop")
        return r.json()
    except Exception:
        return {}


async def webcam_exit(client):
    try:
        r = await client.get("/gopro/webcam/exit")
        return r.json()
    except Exception:
        return {}


def collect_frames(port=8554, duration=5, width=1920, height=1080):
    """Collect frames and return (count, first_shape, fps)."""
    frame_stamps = []
    first_shape = [None]

    def on_frame(frame):
        frame_stamps.append(time.monotonic())
        if first_shape[0] is None:
            first_shape[0] = frame.shape

    decoder = UDPDecoder(udp_port=port, width=width, height=height, on_frame=on_frame)
    decoder.start()
    time.sleep(duration)
    decoder.stop()

    count = len(frame_stamps)
    fps = count / duration if duration > 0 else 0
    return count, first_shape[0], fps


async def run_tests():
    print("=" * 60)
    print("GoMaxWebcam v2 -- End-to-End User Flow Tests (COHN WiFi)")
    print("=" * 60)

    # === TEST 1: COHN HTTPS connectivity ===
    print()
    print("--- Test 1: COHN HTTPS Connectivity ---")
    try:
        async with get_client() as client:
            s = await webcam_status(client)
            record("HTTPS connects to GoPro", True, f"status={s}")
    except Exception as e:
        record("HTTPS connects to GoPro", False, str(e)[:80])
        print("Cannot reach GoPro -- aborting remaining tests")
        return

    # === TEST 2: Start webcam stream 1080p ===
    print()
    print("--- Test 2: Start Webcam Stream (1080p) ---")
    async with get_client() as client:
        try:
            s = await webcam_start(client, res=12)
            started = s.get("status") in [1, 2] and s.get("error") == 0
            record("Webcam start 1080p", started, f"response={s}")
            await asyncio.sleep(3)

            s2 = await webcam_status(client)
            streaming = s2.get("status") == 2
            record("Stream reaches HIGH_POWER_PREVIEW", streaming, f"status={s2}")
        except Exception as e:
            record("Webcam start 1080p", False, str(e)[:80])

    # === TEST 3: Receive and decode frames ===
    print()
    print("--- Test 3: Frame Decode (5s) ---")
    count, shape, fps = collect_frames(duration=5)
    record("Frames received", count > 0, f"{count} frames, {fps:.1f} fps")
    record(
        "Frame shape correct (1080p BGR24)",
        shape == (1080, 1920, 3) if shape else False,
        f"{shape}",
    )
    record("FPS >= 20", fps >= 20, f"{fps:.1f} fps")

    # === TEST 4: Stop stream gracefully ===
    print()
    print("--- Test 4: Stop Stream ---")
    async with get_client() as client:
        try:
            s = await webcam_stop(client)
            record("Webcam stop", True, f"response={s}")
            await asyncio.sleep(2)
            s2 = await webcam_status(client)
            stopped = s2.get("status") in [0, 1, 4]  # 0=OFF, 1=IDLE, 4=READY
            record("Status returns to idle/ready", stopped, f"status={s2}")
        except Exception as e:
            record("Webcam stop", False, str(e)[:80])

    # === TEST 5: Restart stream after stop ===
    print()
    print("--- Test 5: Restart Stream After Stop ---")
    async with get_client() as client:
        try:
            await webcam_exit(client)
            await asyncio.sleep(5)  # GoPro needs time to fully exit webcam mode

            # Retry start — camera may briefly drop HTTPS during mode transition
            started = False
            for attempt in range(3):
                try:
                    s = await webcam_start(client, res=12)
                    started = s.get("error") == 0
                    if started:
                        break
                except Exception:
                    await asyncio.sleep(2)

            record("Restart after stop", started, f"response={s}" if started else "retries exhausted")
            await asyncio.sleep(3)

            count2, _, fps2 = collect_frames(duration=5)
            record("Frames after restart", count2 > 0, f"{count2} frames, {fps2:.1f} fps")

            await webcam_stop(client)
        except Exception as e:
            record("Restart after stop", False, str(e)[:80])

    # === TEST 6: Resolution switch to 720p ===
    print()
    print("--- Test 6: Resolution Switch to 720p ---")
    async with get_client() as client:
        try:
            await webcam_exit(client)
            await asyncio.sleep(2)
            s = await webcam_start(client, res=7)  # 7 = 720p
            record("Webcam start 720p", s.get("error") == 0, f"response={s}")
            await asyncio.sleep(3)

            count3, shape3, fps3 = collect_frames(duration=5, width=1280, height=720)
            record(
                "720p frames received",
                count3 > 0,
                f"{count3} frames, {fps3:.1f} fps, shape={shape3}",
            )

            await webcam_stop(client)
        except Exception as e:
            record("720p switch", False, str(e)[:80])

    # === TEST 7: Webcam exit (full cleanup) ===
    print()
    print("--- Test 7: Webcam Exit ---")
    async with get_client() as client:
        try:
            try:
                await webcam_exit(client)
            except Exception:
                pass  # Camera may drop connection during exit — that IS the exit
            await asyncio.sleep(5)  # Wait for camera to settle

            # Retry status — camera HTTPS may restart after exit
            s = None
            for attempt in range(3):
                try:
                    s = await webcam_status(client)
                    break
                except Exception:
                    await asyncio.sleep(2)

            exited = s is not None and s.get("status") in [0, 1]
            record("Webcam exit cleans up", exited, f"status={s}")
        except Exception as e:
            record("Webcam exit", False, str(e)[:80])

    # === TEST 8: Rapid start/stop cycles (stability) ===
    print()
    print("--- Test 8: Rapid Start/Stop Cycles (3x) ---")
    async with get_client() as client:
        cycle_ok = True
        for i in range(3):
            try:
                await webcam_start(client, res=12)
                await asyncio.sleep(2)
                await webcam_stop(client)
                await asyncio.sleep(1)
            except Exception as e:
                cycle_ok = False
                record(f"Cycle {i+1}", False, str(e)[:60])
                break
        record("3 rapid start/stop cycles", cycle_ok, "no crashes")
        await webcam_exit(client)

    # === TEST 9: Health check while streaming ===
    print()
    print("--- Test 9: Health Check During Stream ---")
    async with get_client() as client:
        try:
            await webcam_start(client, res=12)
            await asyncio.sleep(3)

            health_ok = True
            for i in range(5):
                s = await webcam_status(client)
                if s.get("status") != 2:
                    health_ok = False
                    break
                await asyncio.sleep(0.5)

            record("Health check (5x rapid status)", health_ok, "all returned streaming")
            await webcam_stop(client)
            await webcam_exit(client)
        except Exception as e:
            record("Health check", False, str(e)[:80])

    # === TEST 10: Long stream stability (30s) + latency ===
    print()
    print("--- Test 10: Long Stream Stability + Latency (30s) ---")
    async with get_client() as client:
        try:
            await webcam_start(client, res=12)
            await asyncio.sleep(3)

            # Detailed frame timing for latency analysis
            frame_times = []
            first_shape_10 = [None]

            def on_frame_timed(frame):
                frame_times.append(time.monotonic())
                if first_shape_10[0] is None:
                    first_shape_10[0] = frame.shape

            decoder = UDPDecoder(
                udp_port=8554, width=1920, height=1080, on_frame=on_frame_timed
            )
            decoder.start()
            time.sleep(30)
            decoder.stop()

            count4 = len(frame_times)
            fps4 = count4 / 30.0

            record("30s sustained stream", count4 > 500, f"{count4} frames, {fps4:.1f} fps")
            record("Sustained FPS >= 20", fps4 >= 20, f"{fps4:.1f} fps")

            # Latency analysis
            if len(frame_times) > 10:
                intervals = [
                    (frame_times[i + 1] - frame_times[i]) * 1000
                    for i in range(len(frame_times) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals)
                ideal = 1000.0 / 30.0
                jitter = [abs(iv - ideal) for iv in intervals]
                avg_jitter = sum(jitter) / len(jitter)
                max_interval = max(intervals)
                drops = sum(1 for iv in intervals if iv > ideal * 2)
                drop_pct = drops / len(intervals) * 100

                record("Avg jitter < 20ms", avg_jitter < 20, f"{avg_jitter:.1f}ms")
                record("Max frame gap < 500ms", max_interval < 500, f"{max_interval:.1f}ms")
                record("Drop rate < 5%", drop_pct < 5, f"{drop_pct:.1f}% ({drops} drops)")

            # Camera still healthy?
            s = await webcam_status(client)
            record("Camera healthy after 30s", s.get("status") == 2, f"status={s}")

            await webcam_stop(client)
            await webcam_exit(client)
        except Exception as e:
            record("Long stream", False, str(e)[:80])

    # === SUMMARY ===
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    for name, status, detail in results:
        icon = "+" if status == "PASS" else "X"
        line = f"  [{icon}] {name}"
        if status == "FAIL" and detail:
            line += f" -- {detail}"
        print(line)
    print()
    print(f"  {passed} passed, {failed} failed out of {len(results)} tests")
    if failed == 0:
        print("  ALL TESTS PASSED -- READY FOR WEBCAM USE")
    else:
        print("  SOME TESTS FAILED -- see details above")


if __name__ == "__main__":
    asyncio.run(run_tests())
