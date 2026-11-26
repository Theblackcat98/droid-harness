"""Low-delay H.264 decoding of scrcpy packets with PyAV.

Rules (plan §3):
* whole packets go straight to the decoder — no bitstream parser, which would hold a
  frame back until the next start code arrives;
* the config packet (SPS/PPS) is prepended to the next packet;
* one decoding thread — frame threading adds (threads - 1) frames of delay.
"""

from __future__ import annotations

from collections.abc import Iterator

import av
import numpy as np


class H264Decoder:
    def __init__(self, output_format: str = "bgr24") -> None:
        self.ctx = av.CodecContext.create("h264", "r")
        self.ctx.thread_count = 1
        try:
            # Option handling differs across PyAV versions; best effort.
            self.ctx.options = {"flags": "low_delay"}
        except Exception:  # pragma: no cover - depends on PyAV build
            pass
        self.output_format = output_format
        self._pending_config = b""
        self.packets_in = 0
        self.frames_out = 0

    def feed(self, data: bytes, *, is_config: bool, pts: int | None = None) -> Iterator[np.ndarray]:
        """Feed one scrcpy packet; yields decoded frames (usually 0 or 1)."""
        if is_config:
            self._pending_config = data
            return
        if self._pending_config:
            data = self._pending_config + data
            self._pending_config = b""
        pkt = av.Packet(data)
        if pts is not None:
            pkt.pts = pts
        self.packets_in += 1
        for frame in self.ctx.decode(pkt):
            self.frames_out += 1
            yield frame.to_ndarray(format=self.output_format)
