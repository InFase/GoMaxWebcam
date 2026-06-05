"""
camera_api.py — Direct HTTPS camera control for the dashboard.

Talks to the GoPro camera directly via its HTTPS REST API using COHN
credentials. Bypasses the open-gopro SDK entirely for reliability.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import httpx

log = logging.getLogger("gomaxwebcam.camera_api")

TIMEOUT = httpx.Timeout(10.0, connect=5.0)

# GoPro status key IDs (from Open GoPro spec)
ST_ENCODING = "8"
ST_BATTERY = "70"
ST_REMAINING_PHOTOS = "34"
ST_REMAINING_VIDEO_SEC = "35"
ST_VIDEO_DURATION = "39"
ST_CURRENT_PRESET_GROUP = "96"
ST_CURRENT_PRESET = "97"
ST_WIFI_SSID = "29"
ST_AP_SSID = "30"
ST_REMAINING_SPACE_KB = "54"

# Preset group numeric IDs
PRESET_GROUP_VIDEO = 1000
PRESET_GROUP_PHOTO = 1001
PRESET_GROUP_TIMELAPSE = 1002


class CameraNotConnected(Exception):
    pass


class CameraAPI:
    def __init__(
        self,
        get_gopro: Callable[[], Any],
        cohn_ip: str = "",
        cohn_user: str = "",
        cohn_password: str = "",
    ):
        self._get_gopro = get_gopro
        self._cohn_ip = cohn_ip
        self._cohn_user = cohn_user
        self._cohn_password = cohn_password

    def set_cohn_credentials(self, ip: str, user: str, password: str) -> None:
        self._cohn_ip = ip
        self._cohn_user = user
        self._cohn_password = password
        log.info("COHN credentials updated: %s@%s", user, ip)

    def _base_url(self) -> str:
        if self._cohn_ip:
            return f"https://{self._cohn_ip}"
        gp = self._get_gopro()
        if gp:
            ip = getattr(gp, "ip_address", None)
            if ip:
                return f"http://{ip}:8080"
        raise CameraNotConnected()

    def _auth(self) -> Optional[httpx.BasicAuth]:
        if self._cohn_user and self._cohn_password:
            return httpx.BasicAuth(self._cohn_user, self._cohn_password)
        return None

    async def _get(self, path: str) -> Any:
        url = self._base_url() + path
        async with httpx.AsyncClient(verify=False, timeout=TIMEOUT, auth=self._auth()) as c:
            r = await c.get(url)
            r.raise_for_status()
            ct = r.headers.get("content-type", "")
            if "json" in ct:
                return r.json()
            return {"raw": r.text}

    async def _get_bytes(self, path: str) -> httpx.Response:
        url = self._base_url() + path
        async with httpx.AsyncClient(verify=False, timeout=TIMEOUT, auth=self._auth()) as c:
            return await c.get(url)

    # -- Recording --

    async def start_recording(self) -> dict:
        await self._get("/gopro/camera/shutter/start")
        return {"ok": True, "recording": True}

    async def stop_recording(self) -> dict:
        await self._get("/gopro/camera/shutter/stop")
        return {"ok": True, "recording": False}

    # -- Presets --

    async def get_presets(self) -> dict:
        data = await self._get("/gopro/camera/presets/get")
        groups = []
        group_name_map = {
            "PRESET_GROUP_ID_VIDEO": PRESET_GROUP_VIDEO,
            "PRESET_GROUP_ID_PHOTO": PRESET_GROUP_PHOTO,
            "PRESET_GROUP_ID_TIMELAPSE": PRESET_GROUP_TIMELAPSE,
        }
        title_map = {
            "PRESET_TITLE_VIDEO": "Video",
            "PRESET_TITLE_PHOTO": "Photo",
            "PRESET_TITLE_BURST": "Burst",
            "PRESET_TITLE_NIGHT": "Night",
            "PRESET_TITLE_TIME_LAPSE": "Time Lapse",
            "PRESET_TITLE_NIGHT_LAPSE": "Night Lapse",
            "PRESET_TITLE_TIME_WARP": "TimeWarp",
            "PRESET_TITLE_STAR_TRAIL": "Star Trail",
            "PRESET_TITLE_LIGHT_PAINTING": "Light Painting",
            "PRESET_TITLE_LIGHT_TRAIL": "Light Trail",
            "PRESET_TITLE_BURST_SLOMO": "Slo-Mo",
            "PRESET_TITLE_BIKE": "Bike",
        }
        for group in data.get("presetGroupArray", []):
            raw_id = group.get("id", 0)
            gid = group_name_map.get(str(raw_id), raw_id)
            if isinstance(gid, str):
                gid = group_name_map.get(gid, 0)
            g = {"id": gid, "presets": []}
            for preset in group.get("presetArray", []):
                raw_title = preset.get("titleId", "")
                title = title_map.get(str(raw_title), preset.get("customName", "") or str(raw_title))
                g["presets"].append({
                    "id": preset.get("id", 0),
                    "title": title,
                    "is_modified": preset.get("isModified", False),
                })
            groups.append(g)
        return {"ok": True, "groups": groups}

    async def load_preset(self, preset_id: int) -> dict:
        await self._get(f"/gopro/camera/presets/load?id={preset_id}")
        return {"ok": True, "preset": preset_id}

    async def load_preset_group(self, group_id: int) -> dict:
        await self._get(f"/gopro/camera/presets/set_group?id={group_id}")
        return {"ok": True, "group": group_id}

    # -- Camera State --

    async def get_state(self) -> dict:
        data = await self._get("/gopro/camera/state")
        raw_status = data.get("status", {})
        raw_settings = data.get("settings", {})
        friendly = {
            "encoding": raw_status.get(ST_ENCODING, 0) == 1,
            "battery": raw_status.get(ST_BATTERY, -1),
            "remaining_photos": raw_status.get(ST_REMAINING_PHOTOS, 0),
            "remaining_video_sec": raw_status.get(ST_REMAINING_VIDEO_SEC, 0),
            "video_duration": raw_status.get(ST_VIDEO_DURATION, 0),
            "current_preset_group": raw_status.get(ST_CURRENT_PRESET_GROUP, 0),
            "current_preset": raw_status.get(ST_CURRENT_PRESET, 0),
            "wifi_ssid": raw_status.get(ST_WIFI_SSID, ""),
            "ap_ssid": raw_status.get(ST_AP_SSID, ""),
            "remaining_space_kb": raw_status.get(ST_REMAINING_SPACE_KB, 0),
        }
        return {"ok": True, "status": friendly, "settings": raw_settings, "raw_status": raw_status}

    # -- Media --

    async def get_media_list(self) -> dict:
        data = await self._get("/gopro/media/list")
        files = []
        for group in data.get("media", []):
            directory = group.get("d", "")
            for f in group.get("fs", []):
                name = f.get("n", "")
                size = f.get("s", "0")
                files.append({
                    "directory": directory,
                    "filename": name,
                    "size": int(size) if str(size).isdigit() else 0,
                    "path": f"{directory}/{name}",
                })
        return {"ok": True, "files": files}

    def get_download_url(self, path: str) -> str:
        return f"{self._base_url()}/videos/DCIM/{path}"

    def get_thumbnail_url(self, path: str) -> str:
        return f"{self._base_url()}/gopro/media/thumbnail?path={path}"

    async def get_media_info(self, path: str) -> dict:
        data = await self._get(f"/gopro/media/info?path={path}")
        return {"ok": True, "metadata": data}

    async def delete_file(self, path: str) -> dict:
        await self._get(f"/gopro/media/delete/file?path={path}")
        return {"ok": True, "deleted": path}

    # -- Camera Info --

    async def get_info(self) -> dict:
        data = await self._get("/gopro/camera/info")
        return {"ok": True, "info": data}

    # -- Power --

    async def reboot(self) -> dict:
        try:
            await self._get("/gopro/camera/reboot")
        except Exception:
            pass
        return {"ok": True}

    # -- Digital Zoom --

    async def set_zoom(self, percent: int) -> dict:
        await self._get(f"/gopro/camera/digital_zoom?percent={percent}")
        return {"ok": True, "zoom": percent}

    # -- Webcam / Preview --

    async def webcam_start(self, resolution: int = 12, fov: int = 0, port: int = 8554) -> dict:
        data = await self._get(f"/gopro/webcam/start?res={resolution}&fov={fov}&port={port}&protocol=TS")
        return {"ok": True, "webcam_status": data.get("status", 0)}

    async def webcam_stop(self) -> dict:
        await self._get("/gopro/webcam/stop")
        await self._get("/gopro/webcam/exit")
        return {"ok": True}

    async def webcam_status(self) -> dict:
        data = await self._get("/gopro/webcam/status")
        return {"ok": True, "status": data.get("status", 0), "error": data.get("error", 0)}

    async def preview_start(self) -> dict:
        await self._get("/gopro/camera/stream/start")
        return {"ok": True}

    async def preview_stop(self) -> dict:
        await self._get("/gopro/camera/stream/stop")
        return {"ok": True}

    # -- Sleep (via HTTP) --

    async def sleep(self) -> dict:
        try:
            await self._get("/gopro/camera/sleep")
        except Exception:
            pass  # Camera drops connection when sleeping
        return {"ok": True}

    # -- Keep Alive --

    async def keep_alive(self) -> dict:
        await self._get("/gopro/camera/keep_alive")
        return {"ok": True}
