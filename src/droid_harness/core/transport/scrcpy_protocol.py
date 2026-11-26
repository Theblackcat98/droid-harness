"""Pure encoders/decoders for the scrcpy client <-> server protocol.

Everything here is side-effect free so it can be unit-tested without a device.

The protocol changes between scrcpy versions and the server refuses a client of a
different version. Struct layouts below match the v2.x-v5.x family as documented in
``doc/develop.md``; re-verify them whenever you change the pinned server version
(``tests/test_scrcpy_protocol.py`` pins the byte layouts so a change is deliberate).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import IntEnum

# --- video stream -----------------------------------------------------------

FLAG_CONFIG = 1 << 63
FLAG_KEY_FRAME = 1 << 62
PTS_MASK = (1 << 62) - 1

FRAME_HEADER = struct.Struct(">QI")  # pts_and_flags (u64), packet size (u32)
CODEC_META = struct.Struct(">III")  # codec id (4 ascii chars as u32), width, height
DEVICE_NAME_LEN = 64


@dataclass(frozen=True)
class FrameHeader:
    pts_us: int
    is_config: bool
    is_key_frame: bool
    size: int
    raw_flags: int


def parse_frame_header(data: bytes) -> FrameHeader:
    flags, size = FRAME_HEADER.unpack(data)
    return FrameHeader(
        pts_us=flags & PTS_MASK,
        is_config=bool(flags & FLAG_CONFIG),
        is_key_frame=bool(flags & FLAG_KEY_FRAME),
        size=size,
        raw_flags=flags,
    )


def encode_frame_header(pts_us: int, size: int, *, config: bool = False, key_frame: bool = False) -> bytes:
    flags = pts_us & PTS_MASK
    if config:
        flags |= FLAG_CONFIG
    if key_frame:
        flags |= FLAG_KEY_FRAME
    return FRAME_HEADER.pack(flags, size)


def parse_codec_meta(data: bytes) -> tuple[str, int, int]:
    codec_id, width, height = CODEC_META.unpack(data)
    return codec_id.to_bytes(4, "big").decode("ascii", "replace"), width, height


def encode_codec_meta(codec: str, width: int, height: int) -> bytes:
    return CODEC_META.pack(int.from_bytes(codec.encode("ascii")[:4].ljust(4, b"\0"), "big"), width, height)


def parse_device_name(data: bytes) -> str:
    return data.split(b"\0", 1)[0].decode("utf-8", "replace")


# --- control messages -------------------------------------------------------


class ControlType(IntEnum):
    INJECT_KEYCODE = 0
    INJECT_TEXT = 1
    INJECT_TOUCH_EVENT = 2
    INJECT_SCROLL_EVENT = 3
    BACK_OR_SCREEN_ON = 4
    EXPAND_NOTIFICATION_PANEL = 5
    EXPAND_SETTINGS_PANEL = 6
    COLLAPSE_PANELS = 7
    GET_CLIPBOARD = 8
    SET_CLIPBOARD = 9
    SET_DISPLAY_POWER = 10  # "SET_SCREEN_POWER_MODE" in older docs


class MotionAction(IntEnum):
    DOWN = 0
    UP = 1
    MOVE = 2


class KeyAction(IntEnum):
    DOWN = 0
    UP = 1


TOUCH = struct.Struct(">BBqiiHHHII")  # 32 bytes
KEYCODE = struct.Struct(">BBIII")  # 14 bytes
POINTER_ID_GENERIC_FINGER = -2
MAX_PRESSURE = 0xFFFF


def encode_touch(
    action: MotionAction | int,
    x: int,
    y: int,
    screen_w: int,
    screen_h: int,
    *,
    pointer_id: int = 0,
    pressure: int | None = None,
    action_button: int = 0,
    buttons: int = 0,
) -> bytes:
    """Touch event in *video-frame* coordinates.

    ``screen_w``/``screen_h`` must equal the size of the frames the client is currently
    receiving; the server drops events whose size doesn't match (protects against a
    rotation race).
    """
    if pressure is None:
        pressure = 0 if int(action) == MotionAction.UP else MAX_PRESSURE
    if not (0 <= x < 1 << 31 and 0 <= y < 1 << 31):
        raise ValueError(f"touch outside frame: {(x, y)}")
    return TOUCH.pack(
        ControlType.INJECT_TOUCH_EVENT, int(action), pointer_id, x, y, screen_w, screen_h,
        pressure, action_button, buttons,
    )


def decode_touch(data: bytes) -> dict:
    t, action, pid, x, y, w, h, pressure, ab, buttons = TOUCH.unpack(data)
    return {"type": t, "action": action, "pointer_id": pid, "x": x, "y": y, "w": w, "h": h,
            "pressure": pressure, "action_button": ab, "buttons": buttons}


def encode_keycode(action: KeyAction | int, keycode: int, repeat: int = 0, metastate: int = 0) -> bytes:
    return KEYCODE.pack(ControlType.INJECT_KEYCODE, int(action), keycode, repeat, metastate)


def encode_inject_text(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack(">BI", ControlType.INJECT_TEXT, len(raw)) + raw


def encode_display_power(on: bool, on_byte: int = 1) -> bytes:
    """Turn the physical panel off/on while capture continues.

    Older servers take a power-mode byte (0 = off, 2 = normal); newer ones a boolean.
    ``0`` means off in both, so only the "on" byte is version-dependent and comes from
    the device profile.
    """
    return struct.pack(">BB", ControlType.SET_DISPLAY_POWER, on_byte if on else 0)


# --- server launch arguments -----------------------------------------------


def build_server_args(
    scid: str,
    *,
    max_size: int = 1024,
    max_fps: int = 60,
    video_bit_rate: int = 8_000_000,
    video_encoder: str | None = None,
    codec_options: str | None = "priority:int=0,latency:int=0",
    crop: str | None = None,
    control: bool = True,
    log_level: str = "warn",
) -> list[str]:
    args = [
        f"scid={scid}", f"log_level={log_level}", "video=true", "audio=false",
        f"control={'true' if control else 'false'}", "video_codec=h264",
        f"max_size={max_size}", f"max_fps={max_fps}", f"video_bit_rate={video_bit_rate}",
        "tunnel_forward=true", "send_frame_meta=true", "send_device_meta=true",
        "send_dummy_byte=true", "send_codec_meta=true", "stay_awake=true",
    ]
    if codec_options:
        args.append(f"video_codec_options={codec_options}")
    if video_encoder:
        args.append(f"video_encoder={video_encoder}")
    if crop:
        args.append(f"crop={crop}")
    return args
