"""T4 recorder: raw H.264 + JSONL per match, written off the hot path.

* ``on_packet`` is the T1 tee: every scrcpy packet, config packets included. A match file
  starts with the latest config packet (SPS/PPS) *and* the packets since the last key frame
  (a short pre-roll). Without that pre-roll, a recording that starts mid-stream holds only
  P-frames that reference a key frame it doesn't contain, and nothing decodes until the
  encoder's next key frame (scrcpy's default interval is several seconds).
* ``log`` appends JSON records to per-match streams (decisions, frames, events...).
* The queue is bounded. When it's full, records are dropped and counted; the recorder
  never blocks the brain.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

try:  # orjson is ~5x faster; optional
    import orjson

    def _dumps(obj: Any) -> bytes:
        return orjson.dumps(obj, option=orjson.OPT_SERIALIZE_NUMPY, default=str)
except ImportError:  # pragma: no cover - depends on environment
    def _dumps(obj: Any) -> bytes:
        return json.dumps(obj, default=str, separators=(",", ":")).encode()


class Recorder:
    def __init__(self, root: str | Path, *, max_queue: int = 20000, record_video: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.record_video = record_video
        self._q: queue.Queue = queue.Queue(maxsize=max_queue)
        self._last_config = b""
        self._gop: list[tuple[int, bytes]] = []  # (pts_us, packet) since the last key frame
        self._gop_bytes = 0
        self.max_gop_bytes = 32 * 1024 * 1024
        self.dropped = 0
        self.match_dir: Path | None = None
        self._video = None
        self._pts = None  # stream.pts: one device pts_us per video packet, for exact replay timing
        self._streams: dict[str, Any] = {}
        self._thread = threading.Thread(target=self._loop, name="T4-recorder", daemon=True)
        self._running = True
        self._thread.start()

    # --- producer side (any thread, never blocks) --------------------------
    def _put(self, item: tuple) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def on_packet(self, header, data: bytes) -> None:
        if header.is_config:
            self._last_config = data
            self._gop, self._gop_bytes = [], 0
        elif header.is_key_frame:
            self._gop, self._gop_bytes = [(header.pts_us, data)], len(data)
        elif self._gop and self._gop_bytes < self.max_gop_bytes:
            self._gop.append((header.pts_us, data))
            self._gop_bytes += len(data)
        if self.record_video:
            self._put(("pkt", header.pts_us, data))

    def start_match(self, match_id: str, meta: dict | None = None) -> Path:
        d = self.root / match_id
        self._put(("start", d, self._last_config, list(self._gop), meta or {}))
        return d

    def end_match(self, summary: dict | None = None) -> None:
        self._put(("end", summary or {}))

    def log(self, stream: str, record: dict) -> None:
        self._put(("log", stream, record))

    # --- writer thread ------------------------------------------------------
    def _loop(self) -> None:
        while self._running or not self._q.empty():
            try:
                item = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._handle(item)
            finally:
                self._q.task_done()

    def _handle(self, item: tuple) -> None:
        kind = item[0]
        if kind == "start":
            self._close_match()
            _, d, config, preroll, meta = item
            d.mkdir(parents=True, exist_ok=True)
            self.match_dir = d
            if self.record_video:
                self._video = open(d / "stream.h264", "wb")
                self._pts = open(d / "stream.pts", "w")
                if config:
                    self._video.write(config)
                for pts, pkt in preroll:  # back to the last key frame, so the file decodes from its start
                    self._video.write(pkt)
                    self._pts.write(f"{pts}\n")
                meta = {**meta, "preroll_packets": len(preroll)}
            (d / "meta.json").write_bytes(_dumps({**meta, "started_unix": time.time()}))
        elif kind == "pkt":
            if self._video is not None:
                self._video.write(item[2])
                self._pts.write(f"{item[1]}\n")
        elif kind == "log":
            _, stream, record = item
            if self.match_dir is None:
                return
            f = self._streams.get(stream)
            if f is None:
                f = self._streams[stream] = open(self.match_dir / f"{stream}.jsonl", "ab")
            f.write(_dumps(record) + b"\n")
        elif kind == "end":
            if self.match_dir is not None:
                (self.match_dir / "summary.json").write_bytes(
                    _dumps({**item[1], "recorder_dropped": self.dropped}))
            self._close_match()

    def _close_match(self) -> None:
        if self._video is not None:
            self._video.close()
            self._pts.close()
            self._video, self._pts = None, None
        for f in self._streams.values():
            f.close()
        self._streams.clear()
        self.match_dir = None

    def flush(self, timeout_s: float = 5.0) -> None:
        t0 = time.time()
        while not self._q.empty() and time.time() - t0 < timeout_s:
            time.sleep(0.01)
        self._q.join()

    def close(self) -> None:
        self.flush()
        self._running = False
        self._thread.join(timeout=2)
        self._close_match()
