"""Studio HTTP server (standard library only).

Routes
  GET /                          the page (static/studio.html)
  GET /api/info                  mode, title, and whether live/replay are available
  GET /api/live/config           live overlay geometry (ROIs, tile homography, arena)
  GET /api/live/events           Server-Sent Events: one snapshot JSON per publish
  GET /api/live/snapshot         the latest snapshot (polling fallback)
  GET /api/live/frame.jpg        the frame the latest snapshot describes
  GET /api/matches[?refresh=1]   recorded matches
  GET /api/match?id=…            one match: summary, events, strategy, geometry, timeline
  GET /api/match/frame.jpg?id=…&t=…   nearest decoded video frame at or before t (ms)
  GET /api/match/video?id=…      background decode progress
  GET /api/ledger                latest record per experiment

Match ids are looked up in the library index; paths from the URL never touch the filesystem.
Binds to 127.0.0.1 by default; use an SSH tunnel to watch from another machine.
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .library import MatchLibrary, read_ledger, timeline_bytes
from .telemetry import LivePublisher

log = logging.getLogger(__name__)


def studio_html() -> bytes:
    return resources.files("droid_harness.ui").joinpath("static/studio.html").read_bytes()


class Studio:
    """State shared by request handlers."""

    def __init__(self, *, title: str = "droid-harness Studio", live: LivePublisher | None = None,
                 live_config: dict | None = None, library: MatchLibrary | None = None,
                 ledger: Path | None = Path("ledger/experiments.jsonl"), initial_match: str | None = None) -> None:
        self.title, self.live, self.live_config = title, live, live_config or {}
        self.library, self.ledger, self.initial_match = library, ledger, initial_match
        self._match_cache: dict[str, bytes] = {}
        self._cache_lock = threading.Lock()

    def info(self) -> dict:
        return {"title": self.title, "live": self.live is not None, "replay": self.library is not None,
                "mode": "live" if self.live is not None else "replay", "initial_match": self.initial_match}

    def match_json(self, mid: str) -> bytes | None:
        with self._cache_lock:
            hit = self._match_cache.get(mid)
        if hit is not None:
            return hit
        m = self.library.load(mid) if self.library else None
        if m is None:
            return None
        data = timeline_bytes(m)
        with self._cache_lock:
            if len(self._match_cache) >= 4:
                self._match_cache.pop(next(iter(self._match_cache)))
            self._match_cache[mid] = data
        return data


def _handler(studio: Studio):
    class Handler(BaseHTTPRequestHandler):
        server_version = "droid-harness-studio/1"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:  # keep the bot's log readable
            log.debug("ui %s - " + fmt, self.address_string(), *args)

        # --- helpers --------------------------------------------------------------------
        def _send(self, body: bytes, ctype: str, status: int = 200, headers: dict | None = None) -> None:
            if (len(body) > 32_768 and not ctype.startswith("image/")
                    and "gzip" in (self.headers.get("Accept-Encoding") or "")):
                body = gzip.compress(body, compresslevel=5)  # match timelines shrink ~10x (helps over SSH)
                headers = {**(headers or {}), "Content-Encoding": "gzip"}
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, obj: Any, status: int = 200) -> None:
            self._send(json.dumps(obj, default=str, separators=(",", ":")).encode(), "application/json", status)

        def _err(self, status: int, msg: str) -> None:
            self._json({"error": msg}, status)

        # --- routes ---------------------------------------------------------------------
        def do_HEAD(self) -> None:  # noqa: N802
            self.do_GET()

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            q = {k: v[-1] for k, v in parse_qs(u.query).items()}
            try:
                self._route(u.path, q)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as e:
                log.exception("ui request failed: %s", self.path)
                try:
                    self._err(500, f"{type(e).__name__}: {e}")
                except OSError:
                    pass

        def _route(self, path: str, q: dict[str, str]) -> None:
            if path in ("/", "/index.html"):
                return self._send(studio_html(), "text/html; charset=utf-8")
            if path == "/api/info":
                return self._json(studio.info())
            if path.startswith("/api/live/"):
                return self._live(path, q)
            if path == "/api/matches":
                if studio.library is None:
                    return self._err(404, "replay is off: start with `droid-harness inspect`")
                return self._json(studio.library.refresh())
            if path == "/api/match":
                return self._match(q)
            if path == "/api/match/frame.jpg":
                return self._match_frame(q)
            if path == "/api/match/video":
                vc = studio.library.video(q.get("id", "")) if studio.library else None
                return self._json(vc.status() if vc else {"frames": 0, "done": True, "error": "no video"})
            if path == "/api/ledger":
                return self._json(read_ledger(studio.ledger) if studio.ledger else [])
            return self._err(404, f"no route {path}")

        def _live(self, path: str, q: dict[str, str]) -> None:
            pub = studio.live
            if pub is None:
                return self._err(404, "live view is off: start with `play --ui` or `watch`")
            if path == "/api/live/config":
                return self._json(studio.live_config)
            if path == "/api/live/snapshot":
                return self._send(pub.snapshot_json, "application/json")
            if path == "/api/live/frame.jpg":
                with pub.cv:
                    jpeg, seq = pub.jpeg, pub.jpeg_seq
                if not jpeg:
                    return self._err(404, "no frame yet")
                return self._send(jpeg, "image/jpeg", headers={"X-Frame-Seq": seq})
            if path == "/api/live/events":
                return self._sse(pub)
            return self._err(404, f"no route {path}")

        def _sse(self, pub: LivePublisher) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            self.close_connection = True
            version = -1
            while pub.running:
                v = pub.wait(version, timeout=5.0)
                if v == version:
                    self.wfile.write(b": ping\n\n")
                else:
                    version = v
                    with pub.cv:
                        data = pub.snapshot_json
                    self.wfile.write(b"data: " + data + b"\n\n")
                self.wfile.flush()

        def _match(self, q: dict[str, str]) -> None:
            if studio.library is None:
                return self._err(404, "replay is off: start with `droid-harness inspect`")
            data = studio.match_json(q.get("id", ""))
            if data is None:
                return self._err(404, f"unknown match {q.get('id')!r}")
            return self._send(data, "application/json")

        def _match_frame(self, q: dict[str, str]) -> None:
            vc = studio.library.video(q.get("id", "")) if studio.library else None
            if vc is None:
                return self._err(404, "this match has no stream.h264")
            try:
                t = float(q.get("t", "0"))
            except ValueError:
                return self._err(400, "t must be milliseconds")
            hit = vc.frame_at(t)
            if hit is None:
                return self._err(404, "decoding: no frame yet" if not vc.done else (vc.error or "no frames"))
            t_frame, jpeg = hit
            return self._send(jpeg, "image/jpeg", headers={"X-Frame-T": t_frame})

    return Handler


class StudioServer:
    def __init__(self, studio: Studio, host: str = "127.0.0.1", port: int = 8765) -> None:
        self.studio = studio
        self.httpd = ThreadingHTTPServer((host, port), _handler(studio))
        self.httpd.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        shown = "localhost" if host in ("127.0.0.1", "0.0.0.0", "") else host
        return f"http://{shown}:{port}/"

    def start(self) -> StudioServer:
        self._thread = threading.Thread(target=self.httpd.serve_forever, name="UI-http", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self.studio.live is not None:
            self.studio.live.stop()  # also ends open event streams
        self.httpd.shutdown()
        self.httpd.server_close()
