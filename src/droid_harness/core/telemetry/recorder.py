"""T4 recorder: raw H.264 + JSONL per match, written off the hot path.

* ``on_packet`` is the T1 tee: every scrcpy packet, config packets included. A match file
  starts with the latest config packet (SPS/PPS) *and* the packets since the last key frame
  (a short pre-roll). Without that pre-roll, a recording that starts mid-stream holds only
  P-frames that reference a key frame it doesn't contain, and nothing decodes until the
  encoder's next key frame (scrcpy's default interval is several seconds).
* ``log`` appends JSON records to per-match streams (decisions, frames, events...).
* The queue is bounded. When it's full, records are dropped and counted; the recorder
  never blocks the brain.
* A failed write (disk full, permissions) is counted and logged, and the writer keeps going
  with the next item. Errors that mean nothing more can be written (no space, read-only,
  permission denied, I/O error) set ``fatal_error``, which ``BotRunner.run`` turns into a
  clean stop. ``flush`` and ``close`` never wait longer than their timeouts.
* ``end_match(summary, after=fn)`` runs ``fn(match_dir)`` on a helper thread once the match
  is fully written (summary.json on disk, files closed), e.g. video retention.
"""

from __future__ import annotations

import errno
import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Errors after which further writes can't succeed: stop the run instead of recording nothing.
FATAL_ERRNOS = {errno.ENOSPC, errno.EDQUOT, errno.EROFS, errno.EACCES, errno.EPERM, errno.EIO}

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
        self.write_errors = 0
        self.last_error: str | None = None
        self.fatal_error: str | None = None
        self._error_kinds: set[str] = set()
        self._after: list[threading.Thread] = []
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

    def end_match(self, summary: dict | None = None, *, after: Callable[[Path], Any] | None = None) -> None:
        """Queue the summary; ``after(match_dir)`` runs once the match is fully on disk."""
        self._put(("end", summary or {}, after))

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
            except Exception as e:  # noqa: BLE001 - one bad write must not end recording
                self._note_error(item[0], e)
            finally:
                self._q.task_done()

    def _note_error(self, kind: str, e: Exception) -> None:
        self.write_errors += 1
        self.last_error = f"{kind}: {type(e).__name__}: {e}"
        if isinstance(e, OSError) and e.errno in FATAL_ERRNOS and self.fatal_error is None:
            self.fatal_error = self.last_error
            log.error("recorder: %s; nothing more can be recorded", self.last_error)
        elif kind not in self._error_kinds:
            log.exception("recorder: writing %s failed (later failures of this kind are only counted)", kind)
        self._error_kinds.add(kind)

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
            d, after = self.match_dir, item[2] if len(item) > 2 else None
            try:
                if d is not None:
                    (d / "summary.json").write_bytes(_dumps({
                        **item[1], "recorder_dropped": self.dropped, "recorder_write_errors": self.write_errors}))
            finally:
                self._close_match()
                if after is not None and d is not None:
                    self._run_after(after, d)

    def _close_match(self) -> None:
        if self._video is not None:
            self._video.close()
            self._pts.close()
            self._video, self._pts = None, None
        for f in self._streams.values():
            f.close()
        self._streams.clear()
        self.match_dir = None

    def _run_after(self, fn: Callable[[Path], Any], d: Path) -> None:
        def run() -> None:
            try:
                fn(d)
            except Exception:  # noqa: BLE001
                log.exception("recorder: after-match step for %s failed", d.name)

        t = threading.Thread(target=run, name=f"T4-after-{d.name}", daemon=True)
        self._after = [x for x in self._after if x.is_alive()] + [t]
        t.start()

    def flush(self, timeout_s: float = 5.0) -> bool:
        """Wait until everything queued is written; False if that didn't happen in time."""
        deadline = time.time() + timeout_s
        while self._q.unfinished_tasks and time.time() < deadline:
            if not self._thread.is_alive():
                return False
            time.sleep(0.01)
        return not self._q.unfinished_tasks

    def close(self, timeout_s: float = 10.0, after_timeout_s: float = 120.0) -> None:
        if not self.flush(timeout_s):
            log.warning("recorder: %d items still unwritten after %.0f s; closing anyway",
                        self._q.unfinished_tasks, timeout_s)
        self._running = False
        self._thread.join(timeout=2)
        if not self._thread.is_alive():
            self._close_match()
        deadline = time.time() + after_timeout_s
        for t in self._after:  # e.g. the last match's video retention
            t.join(timeout=max(0.0, deadline - time.time()))
