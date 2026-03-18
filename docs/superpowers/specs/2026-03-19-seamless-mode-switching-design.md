# Seamless Mode Switching and Latency Reduction

## Goal

Eliminate unnecessary delays and false triggers during mode switches
(pause/resume/charge/resolution change) so the app feels instant and
the webcam never cycles unexpectedly.

## Review History

- **Iteration 1**: 3 BLOCKERS found by code-reviewer (stdout overflow,
  silent dedup, exit_freeze doesn't swap). All addressed.
- **Iteration 2**: GoPro API research revealed incorrect error code
  mappings and status code interpretations. Critical corrections added.
- **Iteration 3**: Second code review found stdout drain must use a
  dedicated background thread (blocking read_frame in freeze loop would
  halve vcam framerate). IDLE skip fallback needs status variable update
  to avoid dead guard. Both addressed in this revision.

## Critical API Corrections (from official GoPro Open API research)

The codebase has several incorrect interpretations of the GoPro API.
These MUST be fixed as part of this work:

### Error codes are wrong

Current code treats error code 4 as "UNAVAILABLE". Per the official
Kotlin SDK (`Webcam.kt`), the actual error codes are:

| Code | Name | Meaning |
|------|------|---------|
| 0 | SUCCESS | No error |
| 1 | SET_PRESET | Failed to set webcam preset |
| 2 | SET_WINDOW_SIZE | Failed to set resolution |
| 3 | EXEC_STREAM | Failed to start stream |
| 4 | SHUTTER | Camera shutter is active (recording) |
| 5 | COM_TIMEOUT | Communication timeout |
| 6 | INVALID_PARAM | Bad resolution/FOV value |
| 7 | UNAVAILABLE | Camera unavailable for webcam |
| 8 | EXIT | Error during exit |

**Impact:** When we get error 4, we should call `set_shutter(OFF)`
(`/gopro/camera/shutter?mode=0`) and retry, NOT run the IDLE workaround.

### Status codes need clarification

| Code | Official Name | Our Current Name | Correct Interpretation |
|------|--------------|-----------------|----------------------|
| 0 | OFF | OFF | Webcam fully off |
| 1 | IDLE | IDLE | Initialized, not streaming |
| 2 | HIGH_POWER_PREVIEW | READY | Actually streaming (misnamed in our code) |
| 3 | LOW_POWER_PREVIEW | STREAMING | Preview mode, low power |
| 4 | STATUS_UNAVAILABLE | UNAVAILABLE | Hero 13 returns this during normal streaming |

**Impact:** Our code waits for status 3 (STREAMING) but the camera
reports status 2 (HIGH_POWER_PREVIEW) when actually streaming. The
3-second "accept READY after 6 polls" workaround exists because we're
waiting for the wrong status code. Status 2 IS streaming.

### Resolution change: stop+start is sufficient (no exit needed)

Per official C# demo and GitHub issues: resolution changes only need
`webcam/stop` then `webcam/start` with new params. Full `webcam/exit`
is overkill and adds unnecessary delay. Our recent change to use
`exit_webcam()` for resolution changes should be reverted to
`stop_webcam()`.

### Preview mode for faster startup

`/gopro/webcam/preview` warms up the sensor without starting the UDP
stream. Calling this during connection setup (after IDLE workaround)
pre-warms the sensor so `webcam/start` produces frames faster.

### Keep-alive interval

Official SDK uses 3 seconds. Our 1.0s minimum floor is fine (more
conservative). The keep-alive endpoint is `/gopro/camera/keep_alive`.

## Changes

### 1. Instant Pause/Resume (no ffmpeg restart)

**Pause:** Freeze the pipeline but keep everything running.
- `pause_webcam()`: set state PAUSED, call `pipeline.enter_freeze_frame()`.
- Do NOT stop ffmpeg, webcam, monitors, or disconnect detector.
- The pipeline continues to drain ffmpeg stdout (read and discard frames)
  while pushing the freeze frame to the virtual camera.

**Frame pipeline change (frame_pipeline.py):**
- When entering freeze-frame mode while ffmpeg is still alive, start a
  background drain thread that calls `reader.read_frame()` in a loop,
  discarding frames. This prevents ffmpeg's stdout pipe (4KB on Windows)
  from filling up and blocking ffmpeg.
- The drain thread is started by `enter_freeze_frame()` and stopped by
  `exit_freeze_frame()` or `stop()`.
- IMPORTANT: `read_frame()` is blocking (reads full frame from stdout),
  so it CANNOT be called inside the freeze-frame push loop — that would
  halve the vcam frame rate. A dedicated thread is required.
- The drain thread checks `_freeze_event.is_set()` and
  `_stop_event.is_set()` to know when to exit.

**Resume from PAUSED:** Unfreeze the pipeline.
- `resume_webcam()` when state is PAUSED: call `pipeline.exit_freeze_frame()`,
  set state STREAMING. No start_webcam, no stream reader swap.
- ffmpeg is still running, GoPro is still streaming — instant resume.

**Resume from CHARGE_MODE:** Keeps current full restart flow.

**Files:** `app_controller.py`, `frame_pipeline.py`

### 2. Fix Error Code Handling

Replace the incorrect error code interpretation throughout
`gopro_connection.py`:

- Error 4 (SHUTTER): call `/gopro/camera/shutter?mode=0` then retry
- Error 6 (INVALID_PARAM): current remap logic is correct
- Error 7 (UNAVAILABLE): current bail-early logic, move from code 4
- Add all error codes as an IntEnum for clarity

Also fix the start_webcam polling loop:
- Accept status 2 (HIGH_POWER_PREVIEW) immediately as "streaming"
- Accept status 3 (LOW_POWER_PREVIEW) as valid preview state
- Handle status 4 from Hero 13 gracefully (treat as streaming)

**Files:** `gopro_connection.py`

### 3. Skip IDLE Workaround on Intentional Resume

Track intentional stops via `_last_stop_was_intentional` flag.
When `start_webcam()` sees IDLE but the last stop was intentional,
skip the IDLE workaround and go straight to webcam/start.

**Fallback:** If webcam/start fails after skipping the workaround,
clear the flag and retry WITH the workaround. IMPORTANT: the existing
fallback guard at line ~1131 checks `status != WebcamStatus.IDLE` —
since we came from IDLE and skipped the reset, `status` is still IDLE,
which would prevent the fallback from firing. Fix: update `status`
variable to the live camera status after the failed start attempt,
before hitting the fallback guard.

**Files:** `gopro_connection.py`

### 4. Fix First Launch False Recovery

Delay the disconnect detector's first health check by
`stream_startup_timeout` seconds instead of blocking on first frame.

- Pass a `startup_grace` parameter to the disconnect detector
- Gives ffmpeg time to connect without blocking the startup thread
- Staleness monitor's `_staleness_was_stale = True` guard prevents
  false triggers during this initial period

**Files:** `app_controller.py`, `disconnect_detector.py`

### 5. Use Preview Mode for Faster Startup

After the IDLE workaround (if needed), call `/gopro/webcam/preview`
to pre-warm the sensor before calling `/gopro/webcam/start`. This
reduces the time between `start` and the first UDP packet arriving.

**Files:** `gopro_connection.py`

### 6. Fix Resolution Change (use stop, not exit)

Revert `_apply_resolution_on_camera()` to use `stop_webcam()` instead
of `exit_webcam()`. Per official API docs, stop+start is sufficient
for resolution changes and is faster (no full exit/re-enter cycle).

The 2.0s minimum delay can be reduced to 1.0s since we're not doing
a full exit.

**Files:** `app_controller.py`

### 7a. Suppress "Stream interrupted" During Intentional Shutdown

Add state guard in `_on_pipeline_stream_lost()` — suppress message
when state is PAUSED or CHARGE_MODE.

**Files:** `app_controller.py`

### 7b. Deduplicate "Live video restored!"

- Instant resume (PAUSED): pipeline callback is the sole source
- Full restart (CHARGE_MODE): keep explicit emit in resume_webcam()

**Files:** `app_controller.py`

### 7c. Fix Camera Name in DirectShow

Write FriendlyName to the correct DirectShow registry keys (HKLM).
Unity Capture CLSID: `{5C2CD55C-92AD-4999-8666-912BD3E70010}`.

Keys to update:
- `HKLM\SOFTWARE\Classes\CLSID\{5C2CD55C-92AD-4999-8666-912BD3E70010}` default
- `HKLM\...\{860BB310-5D01-11D0-BD3B-00A0C911CE86}\Instance\{5C2CD55C-92AD-4999-8666-912BD3E70010}` FriendlyName

Guards: check before writing, marker file to avoid repeated UAC, fail
gracefully if denied.

**Files:** `virtual_camera.py`

## Expected Results

| Operation | Before | After |
|-----------|--------|-------|
| Pause | 1-2s (kill ffmpeg + stop webcam) | <100ms (freeze pipeline) |
| Resume from pause | 4-7s (full restart) | <100ms (unfreeze pipeline) |
| Resume from charge | 6-10s | 3-5s (skip IDLE + correct status) |
| Resolution change | 4-8s (exit + restart) | 2-3s (stop + start) |
| First launch | 15-25s (false recovery) | 5-10s (no false recovery) |
| start_webcam poll | 3-10s (waits for wrong status) | <1s (accept status 2) |
| Camera name in apps | "Unity Video Capture" | Configured name |

## Files Modified

- `app_controller.py` — pause/resume, startup, resolution change, cosmetics
- `frame_pipeline.py` — drain stdout during freeze-frame
- `gopro_connection.py` — error codes, status codes, IDLE skip, preview, shutter fix
- `disconnect_detector.py` — startup grace period
- `virtual_camera.py` — DirectShow FriendlyName update

## Sources

- [GoPro Open API HTTP Spec](https://gopro.github.io/OpenGoPro/http)
- [GoPro Open API FAQ](https://gopro.github.io/OpenGoPro/faq)
- [OpenGoPro GitHub](https://github.com/gopro/OpenGoPro)
- [Hero 13 webcam compatibility issue #603](https://github.com/gopro/OpenGoPro/issues/603)
- [DeepWiki OpenGoPro Commands API](https://deepwiki.com/gopro/OpenGoPro/2.2-commands-and-settings-api)
