"""A fake scrcpy server for tests and demos (no phone needed).

Speaks the same wire protocol as ``scrcpy-server`` over a local TCP port: dummy byte,
device meta, codec meta, then H.264 packets (SPS/PPS split into a config packet) from
a real libx264 encoder. It parses incoming control messages and can render a "show
taps" indicator after a configurable delay, which makes the loopback latency test
runnable in CI.
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction

import av
import cv2
import numpy as np

from ..core.transport import scrcpy_protocol as proto

FrameSource = Callable[[int, float], np.ndarray]  # (frame_index, t_seconds) -> BGR image


def split_config_nals(packet: bytes) -> tuple[bytes, bytes]:
    """Split an Annex-B packet into (SPS/PPS bytes, remaining bytes)."""
    starts = []
    i = 0
    n = len(packet)
    while i < n - 3:
        if packet[i] == 0 and packet[i + 1] == 0 and packet[i + 2] == 1:
            starts.append(i)
            i += 3
        else:
            i += 1
    if not starts:
        return b"", packet
    units = []
    for k, s in enumerate(starts):
        begin = s - 1 if s > 0 and packet[s - 1] == 0 else s
        end = starts[k + 1] if k + 1 < len(starts) else n
        if k + 1 < len(starts) and starts[k + 1] > 0 and packet[starts[k + 1] - 1] == 0:
            end = starts[k + 1] - 1
        nal_type = packet[s + 3] & 0x1F
        units.append((nal_type, packet[begin:end]))
    config = b"".join(u for t, u in units if t in (7, 8))
    rest = b"".join(u for t, u in units if t not in (7, 8))
    return config, rest


@dataclass
class FakeDeviceState:
    touches: list[dict] = field(default_factory=list)  # decoded touch messages with t_rx
    display_power: list[int] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    keycodes: list[tuple[int, int]] = field(default_factory=list)


class FakeScrcpyServer:
    def __init__(self, port: int, *, width: int = 320, height: int = 640, fps: int = 60,
                 source: FrameSource | None = None, device_name: str = "FakePhone",
                 tap_indicator_delay_s: float | None = None, static_after_frames: int | None = None) -> None:
        self.port = port
        self.width, self.height, self.fps = width, height, fps
        self.source = source or self._default_source
        self.device_name = device_name
        self.tap_indicator_delay_s = tap_indicator_delay_s
        self.static_after_frames = static_after_frames
        self.state = FakeDeviceState()
        self._active: dict[int, dict] = {}
        self._stop = threading.Event()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", port))
        self._srv.listen(4)
        self._threads: list[threading.Thread] = []
        self.frames_sent = 0

    # --- rendering --------------------------------------------------------
    def _default_source(self, i: int, t: float) -> np.ndarray:
        img = np.zeros((self.height, self.width, 3), np.uint8)
        x = int((i * 4) % self.width)
        cv2.rectangle(img, (x, 40), (min(x + 30, self.width - 1), 70), (0, 200, 255), -1)
        return img

    def _render(self, i: int, t: float) -> np.ndarray:
        img = self.source(i, t).copy()
        if self.tap_indicator_delay_s is not None:
            now = time.perf_counter()
            for tch in list(self._active.values()):  # like "Show taps": only fingers still down
                if now - tch["t_rx"] >= self.tap_indicator_delay_s:
                    x = int(tch["x"] * self.width / max(tch["w"], 1))
                    y = int(tch["y"] * self.height / max(tch["h"], 1))
                    cv2.circle(img, (x, y), 12, (255, 255, 255), -1)
        return img

    # --- server -----------------------------------------------------------
    def start(self) -> FakeScrcpyServer:
        t = threading.Thread(target=self._serve, name="fake-scrcpy", daemon=True)
        t.start()
        self._threads.append(t)
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass

    def _serve(self) -> None:
        try:
            video, _ = self._srv.accept()
            video.sendall(b"\0")  # dummy byte
            control, _ = self._srv.accept()
        except OSError:
            return
        video.sendall(self.device_name.encode().ljust(proto.DEVICE_NAME_LEN, b"\0"))
        video.sendall(proto.encode_codec_meta("h264", self.width, self.height))
        ct = threading.Thread(target=self._control_loop, args=(control,), daemon=True)
        ct.start()
        self._threads.append(ct)
        self._video_loop(video)

    def _video_loop(self, sock: socket.socket) -> None:
        enc = av.CodecContext.create("libx264", "w")
        enc.width, enc.height = self.width, self.height
        enc.pix_fmt = "yuv420p"
        enc.time_base = Fraction(1, self.fps)
        enc.options = {"tune": "zerolatency", "preset": "ultrafast", "crf": "18"}
        t0 = time.perf_counter()
        i = 0
        period = 1.0 / self.fps
        try:
            while not self._stop.is_set():
                if self.static_after_frames is not None and i >= self.static_after_frames:
                    time.sleep(0.01)  # real scrcpy sends nothing while the screen is static
                    continue
                t = time.perf_counter() - t0
                img = self._render(i, t)
                frame = av.VideoFrame.from_ndarray(img, format="bgr24")
                frame.pts = i
                for pkt in enc.encode(frame):
                    data = bytes(pkt)
                    pts_us = int(t * 1e6)  # device-clock timestamp, like MediaCodec
                    config, rest = split_config_nals(data)
                    if config:
                        sock.sendall(proto.encode_frame_header(pts_us, len(config), config=True) + config)
                    if rest:
                        sock.sendall(proto.encode_frame_header(pts_us, len(rest), key_frame=pkt.is_keyframe) + rest)
                        self.frames_sent += 1
                i += 1
                sleep = t0 + i * period - time.perf_counter()
                if sleep > 0:
                    time.sleep(sleep)
        except OSError:
            pass
        finally:
            try:
                sock.close()
            except OSError:
                pass

    def _control_loop(self, sock: socket.socket) -> None:
        def recv_exact(n: int) -> bytes:
            buf = b""
            while len(buf) < n:
                chunk = sock.recv(n - len(buf))
                if not chunk:
                    raise ConnectionError
                buf += chunk
            return buf

        try:
            while not self._stop.is_set():
                t = recv_exact(1)[0]
                if t == proto.ControlType.INJECT_TOUCH_EVENT:
                    msg = proto.decode_touch(bytes([t]) + recv_exact(proto.TOUCH.size - 1))
                    msg["t_rx"] = time.perf_counter()
                    self.state.touches.append(msg)
                    if msg["action"] == proto.MotionAction.DOWN:
                        self._active[msg["pointer_id"]] = msg
                    elif msg["action"] == proto.MotionAction.UP:
                        self._active.pop(msg["pointer_id"], None)
                elif t == proto.ControlType.SET_DISPLAY_POWER:
                    self.state.display_power.append(recv_exact(1)[0])
                elif t == proto.ControlType.INJECT_KEYCODE:
                    _, action, keycode, _, _ = proto.KEYCODE.unpack(bytes([t]) + recv_exact(proto.KEYCODE.size - 1))
                    self.state.keycodes.append((action, keycode))
                elif t == proto.ControlType.INJECT_TEXT:
                    (n,) = struct.unpack(">I", recv_exact(4))
                    self.state.texts.append(recv_exact(n).decode())
                else:
                    return  # unknown message: stop parsing (real server would too)
        except (ConnectionError, OSError):
            return


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
