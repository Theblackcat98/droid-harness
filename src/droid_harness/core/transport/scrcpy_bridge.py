"""scrcpy session: launch the server, connect the sockets, decode video, inject input.

One ``ScrcpyClient`` per phone. Ports and session ids are per device so two phones
can run side by side (self-play).
"""

from __future__ import annotations

import logging
import random
import socket
import threading
import time
from collections.abc import Callable

from ..config import DeviceProfile
from . import scrcpy_protocol as proto
from .adb import Adb
from .decoder import H264Decoder
from .frame_slot import Frame, LatestFrame

log = logging.getLogger(__name__)

SERVER_REMOTE_PATH = "/data/local/tmp/scrcpy-server.jar"
PacketTee = Callable[[proto.FrameHeader, bytes], None]


def recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk
    return bytes(buf)


class ScrcpyClient:
    def __init__(self, profile: DeviceProfile, *, adb: Adb | None = None,
                 on_packet: PacketTee | None = None, host: str = "127.0.0.1") -> None:
        self.profile = profile
        self.adb = adb or Adb(profile.serial)
        self.host = host
        self.scid = f"{random.getrandbits(31):08x}"
        self.on_packet = on_packet
        self.slot = LatestFrame()
        self.decoder = H264Decoder()
        self.size: tuple[int, int] = (0, 0)
        self.device_name = ""
        self.codec = ""
        self.running = False
        self.error: BaseException | None = None
        self.last_packet_ns = 0
        self.last_config: bytes = b""
        self.proc = None
        self.video: socket.socket | None = None
        self.control: socket.socket | None = None
        self._control_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # --- lifecycle --------------------------------------------------------
    def start(self, *, launch: bool = True, connect_attempts: int = 50) -> None:
        if launch:
            self._launch_server()
        self.video = self._connect_first(connect_attempts)
        self.control = self._connect()
        # Device meta is sent on the first socket only after all sockets are connected.
        self.device_name = proto.parse_device_name(recv_exact(self.video, proto.DEVICE_NAME_LEN))
        self.codec, w, h = proto.parse_codec_meta(recv_exact(self.video, proto.CODEC_META.size))
        self.size = (w, h)
        self.video.settimeout(None)
        self.running = True
        self._thread = threading.Thread(target=self._video_loop, name=f"T1-video-{self.profile.name}",
                                        daemon=True)
        self._thread.start()
        if self.profile.turn_display_off:
            self.set_display_power(False)

    def stop(self) -> None:
        self.running = False
        for s in (self.video, self.control):
            try:
                if s is not None:
                    s.close()
            except OSError:
                pass
        if self.proc is not None:
            self.proc.terminate()
        try:
            self.adb.forward_remove(self.profile.local_port)
        except Exception:  # pragma: no cover - best effort
            pass

    def _launch_server(self) -> None:
        p = self.profile
        self.adb.push(p.scrcpy_server_jar, SERVER_REMOTE_PATH)
        self.adb.forward(p.local_port, f"localabstract:scrcpy_{self.scid}")
        args = proto.build_server_args(
            self.scid, max_size=p.max_size, max_fps=p.max_fps, video_bit_rate=p.video_bit_rate,
            video_encoder=p.video_encoder, codec_options=p.codec_options, crop=p.crop)
        self.proc = self.adb.popen("shell", f"CLASSPATH={SERVER_REMOTE_PATH}", "app_process", "/",
                                   "com.genymobile.scrcpy.Server", p.scrcpy_server_version, *args)

    def _connect(self) -> socket.socket:
        s = socket.create_connection((self.host, self.profile.local_port), timeout=2)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        return s

    def _connect_first(self, attempts: int) -> socket.socket:
        # adb forward accepts connect() before the server listens; the dummy byte is the
        # real "ready" signal, so retry until it arrives.
        last: BaseException | None = None
        for _ in range(attempts):
            try:
                s = self._connect()
            except OSError as e:
                last = e
                time.sleep(0.1)
                continue
            try:
                recv_exact(s, 1)
                return s
            except (ConnectionError, OSError) as e:
                last = e
                s.close()
                time.sleep(0.1)
        raise TimeoutError(f"scrcpy-server never became ready: {last}")

    # --- video ------------------------------------------------------------
    def _video_loop(self) -> None:
        seq = 0
        assert self.video is not None
        try:
            while self.running:
                hdr = proto.parse_frame_header(recv_exact(self.video, proto.FRAME_HEADER.size))
                data = recv_exact(self.video, hdr.size)
                t_recv = time.perf_counter_ns()
                self.last_packet_ns = t_recv
                if hdr.is_config:
                    self.last_config = data
                if self.on_packet is not None:
                    self.on_packet(hdr, data)  # recorder sees every packet, config included
                for img in self.decoder.feed(data, is_config=hdr.is_config, pts=hdr.pts_us):
                    seq += 1
                    self.size = (img.shape[1], img.shape[0])
                    self.slot.put(Frame(seq, img, hdr.pts_us, t_recv, time.perf_counter_ns()))
        except Exception as e:  # surfaced to the liveness monitor
            if self.running:
                self.error = e
                log.warning("video loop stopped: %s", e)
            self.running = False

    # --- control ----------------------------------------------------------
    def send_control(self, msg: bytes) -> int:
        """Send one control message; returns perf_counter_ns at send time."""
        if self.control is None:
            raise RuntimeError("not connected")
        with self._control_lock:  # whole message at once, never interleaved
            self.control.sendall(msg)
            return time.perf_counter_ns()

    def touch(self, action: int, x: int, y: int, pointer_id: int = 0) -> int:
        w, h = self.size
        return self.send_control(proto.encode_touch(action, int(x), int(y), w, h, pointer_id=pointer_id))

    def set_display_power(self, on: bool) -> int:
        return self.send_control(proto.encode_display_power(on, self.profile.display_power_on_byte))

    def inject_text(self, text: str) -> int:
        return self.send_control(proto.encode_inject_text(text))

    def keycode(self, keycode: int) -> None:
        self.send_control(proto.encode_keycode(proto.KeyAction.DOWN, keycode))
        self.send_control(proto.encode_keycode(proto.KeyAction.UP, keycode))

    # --- status -----------------------------------------------------------
    def ms_since_last_packet(self) -> float:
        if not self.last_packet_ns:
            return float("inf")
        return (time.perf_counter_ns() - self.last_packet_ns) / 1e6

    def server_alive(self) -> bool:
        return self.proc is None or self.proc.poll() is None
