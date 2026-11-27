"""Typed configuration loading (device profiles)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator


class DeviceProfile(BaseModel):
    name: str
    serial: str
    local_port: int = Field(ge=1024, le=65535)
    account: str = ""
    scrcpy_server_version: str
    scrcpy_server_jar: str = "scrcpy-server"  # local path of the jar to push
    max_size: int = 1024
    max_fps: int = 60
    video_bit_rate: int = 8_000_000
    video_encoder: str | None = None
    codec_options: str | None = "priority:int=0,latency:int=0"
    crop: str | None = None
    display_power_on_byte: int = 1
    turn_display_off: bool = False
    exec_latency_ms: float = 40.0  # measured p50 from the loopback test; used for lead targeting
    rois: str | None = None

    @model_validator(mode="after")
    def _serial_env_override(self) -> "DeviceProfile":
        """Let the rig's ADB serial move without editing the tracked profile.

        The phone drops off USB between sessions and is picked up over
        adb-over-TCP instead (``192.168.1.190:5555``); every construction path
        goes through this model, so the override lives here.
        """
        override = os.environ.get("DROID_HARNESS_SERIAL")
        if override and self.serial != "sim":
            self.serial = override
        return self


def load_profile(path: str | Path) -> DeviceProfile:
    data = yaml.safe_load(Path(path).read_text())
    data.setdefault("name", Path(path).stem)
    return DeviceProfile(**data)


def check_unique_ports(profiles: list[DeviceProfile]) -> None:
    seen: dict[int, str] = {}
    for p in profiles:
        if p.local_port in seen:
            raise ValueError(f"{p.name} and {seen[p.local_port]} share local_port {p.local_port}")
        seen[p.local_port] = p.name
