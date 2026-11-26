"""Latest-frame slot: writers overwrite, readers wait for something newer.

Only *decoded* frames are dropped here. H.264 packets must all be decoded because later
frames reference earlier ones, so the decoder has to keep up with the stream.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Frame:
    seq: int
    img: np.ndarray  # BGR, HxWx3
    pts_us: int  # device encoder timestamp
    t_recv_ns: int  # host perf_counter_ns when the packet arrived
    t_decoded_ns: int  # host perf_counter_ns when decoding finished
    meta: dict = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        return self.img.shape[1], self.img.shape[0]


class LatestFrame:
    def __init__(self) -> None:
        self._frame: Frame | None = None
        self._cv = threading.Condition()
        self.overwritten = 0  # frames replaced before anyone read them

    def put(self, frame: Frame) -> None:
        with self._cv:
            if self._frame is not None and not self._frame.meta.get("_read"):
                self.overwritten += 1
            self._frame = frame
            self._cv.notify_all()

    def peek(self) -> Frame | None:
        with self._cv:
            return self._frame

    def wait_newer(self, seq: int, timeout: float = 0.5) -> Frame | None:
        """Block until a frame with ``frame.seq > seq`` exists, or ``timeout`` passes."""
        with self._cv:
            self._cv.wait_for(lambda: self._frame is not None and self._frame.seq > seq, timeout)
            f = self._frame
            if f is not None and f.seq > seq:
                f.meta["_read"] = True
                return f
            return None
