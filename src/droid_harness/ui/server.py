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
  GET /api/live/frame.png        the same frame, lossless (screenshots)
  GET /api/ledger                latest record per experiment

With the app (``droid-harness`` with no arguments, or ``droid-harness studio``):
  GET  /api/app/state            settings, jobs, phone, guided-flow state
  GET  /api/app/schema           the settings form
  GET  /api/app/strategies       strategy files with names and IDs
  GET  /api/app/datasets         folders of recorded matches
  GET  /api/app/devices          ``adb devices``
  GET  /api/app/jobs/<id>/log?since=N
  POST /api/app/settings, /api/app/settings/reset
  POST /api/app/run              {"kind": ..., "params": {...}} starts a job
  POST /api/app/jobs/<id>/stop   {"now": false}
  POST /api/app/phone/lock|tap|swipe|key
  POST /api/app/screenshot       {"overlay_png"?, "frame_png"?: base64, "label"?}
  POST /api/app/refine/<step>    configure, scan, propose, check, test, promote, next_round, reset
  POST /api/app/quit             stop every job and the Studio
  /api/live/* is then proxied to whichever job is showing a live view.

Match ids are looked up in the library index; paths from the URL never touch the filesystem.
Binds to 127.0.0.1 by default; use an SSH tunnel to watch from another machine. With the
app, every request must name this machine in its Host header (no DNS rebinding) and every
POST must carry the page's per-start token (no cross-site requests).
"""

from __future__ import annotations

import base64
import gzip
import http.client
import ipaddress
import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .library import MatchLibrary, read_ledger, timeline_bytes
from .telemetry import LivePublisher

TOKEN_META = b'<meta name="studio-token" content="">'
MAX_BODY = 64 * 1024 * 1024

log = logging.getLogger(__name__)


def studio_html() -> bytes:
    return resources.files("droid_harness.ui").joinpath("static/studio.html").read_bytes()


class Studio:
    """State shared by request handlers."""

    def __init__(self, *, title: str = "droid-harness Studio", live: LivePublisher | None = None,
                 live_config: dict | None = None, library: MatchLibrary | None = None,
                 ledger: Path | None = Path("ledger/experiments.jsonl"), initial_match: str | None = None,
                 app=None) -> None:
        self.title, self.live, self.live_config = title, live, live_config or {}
        self.library, self.ledger, self.initial_match = library, ledger, initial_match
        self.app = app  # a control.StudioApp: buttons, settings, jobs, guided flow
        self._match_cache: dict[str, bytes] = {}
        self._cache_lock = threading.Lock()
        self.allowed_hosts = {"localhost", socket.gethostname().lower(), f"{socket.gethostname().lower()}.local"}

    def info(self) -> dict:
        live = self.live is not None or self.app is not None
        return {"title": self.title, "live": live, "replay": self.library is not None, "app": self.app is not None,
                "mode": "live" if self.live is not None else "replay", "initial_match": self.initial_match}

    def host_ok(self, host_header: str | None) -> bool:
        """Accept IP literals and this machine's names; a rebinding attack arrives under a domain name."""
        if self.app is None:
            return True
        host = (host_header or "").strip().lower()
        if not host:
            return False
        if host.startswith("["):
            host = host[1:].split("]", 1)[0]
        elif host.count(":") == 1:
            host = host.rsplit(":", 1)[0]
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return host in self.allowed_hosts

    def page(self) -> bytes:
        html = studio_html()
        if self.app is not None:
            html = html.replace(TOKEN_META, f'<meta name="studio-token" content="{self.app.token}">'.encode())
        return html

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
            if not studio.host_ok(self.headers.get("Host")):
                return self._err(403, "unrecognized Host header")
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
                return self._send(studio.page(), "text/html; charset=utf-8")
            if path == "/api/info":
                return self._json(studio.info())
            if path.startswith("/api/live/"):
                if studio.live is None and studio.app is not None:
                    return self._proxy_live(path)
                return self._live(path, q)
            if path.startswith("/api/app/"):
                return self._app_get(path, q)
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
            if path == "/api/live/frame.png":
                _, raw = pub.tap.latest()
                frame = raw[0] if raw else None
                if frame is None or getattr(frame, "img", None) is None:
                    return self._err(404, "no frame yet")
                import cv2

                ok, buf = cv2.imencode(".png", frame.img)
                if not ok:
                    return self._err(500, "couldn't encode the frame")
                return self._send(buf.tobytes(), "image/png", headers={"X-Frame-Seq": frame.seq})
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

        # --- the app ------------------------------------------------------------------------
        def _proxy_live(self, path: str) -> None:
            job = studio.app.jobs.live_job()
            if job is None:
                return self._err(404, "no live view: start Preview, Play or a phone test")
            try:
                c = http.client.HTTPConnection("127.0.0.1", job.live_port, timeout=30)
                c.request("GET", path)
                r = c.getresponse()
            except OSError:
                return self._err(503, "the live view is starting")
            if path == "/api/live/events" and r.status == 200:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                try:
                    while True:
                        chunk = r.read1(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                except (OSError, http.client.HTTPException):
                    pass
                finally:
                    c.close()
                return None
            body = r.read()
            c.close()
            headers = {k: v for k, v in r.getheaders() if k.lower().startswith("x-frame")}
            return self._send(body, r.getheader("Content-Type") or "application/octet-stream", r.status, headers)

        def _app_get(self, path: str, q: dict[str, str]) -> None:
            app = studio.app
            if app is None:
                return self._err(404, "the app is off: start with `droid-harness` (no arguments)")
            if path == "/api/app/state":
                return self._json(app.state())
            if path == "/api/app/schema":
                from .settings import Settings

                return self._json(Settings.schema())
            if path == "/api/app/strategies":
                return self._json(app.strategies(refresh=q.get("refresh") == "1"))
            if path == "/api/app/datasets":
                return self._json(app.datasets(refresh=q.get("refresh") == "1"))
            if path == "/api/app/devices":
                return self._json(app.phone.devices())
            parts = path.strip("/").split("/")
            if len(parts) == 5 and parts[2] == "jobs" and parts[4] == "log":
                job = app.jobs.get(parts[3])
                if job is None:
                    return self._err(404, f"no job {parts[3]}")
                try:
                    since = int(q.get("since", "0"))
                except ValueError:
                    since = 0
                return self._json(job.log_since(since))
            return self._err(404, f"no route {path}")

        def do_POST(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            app = studio.app
            if app is None:
                return self._err(404, "the app is off: start with `droid-harness` (no arguments)")
            if not studio.host_ok(self.headers.get("Host")):
                return self._err(403, "unrecognized Host header")
            if not secrets_equal(self.headers.get("X-Studio-Token") or "", app.token):
                return self._err(403, "missing or wrong X-Studio-Token: reload the page")
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = -1
            if n < 0 or n > MAX_BODY:
                return self._err(413, "request body too large")
            try:
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as e:
                return self._err(400, f"bad JSON: {e}")
            try:
                return self._app_post(u.path, body)
            except (BrokenPipeError, ConnectionResetError):
                return None
            except Exception as e:  # noqa: BLE001 - shown on the page
                from .control import ActionError
                from .jobs import JobError
                from .phone import PhoneError, PhoneLocked
                from .refine import FlowError

                if isinstance(e, PhoneLocked):
                    return self._err(423, str(e))
                if isinstance(e, (ActionError, JobError, PhoneError, FlowError, ValueError, KeyError, TypeError)):
                    return self._err(400, str(e) or type(e).__name__)
                log.exception("ui POST failed: %s", self.path)
                return self._err(500, f"{type(e).__name__}: {e}")

        def _app_post(self, path: str, b: dict) -> None:
            app = studio.app
            if path == "/api/app/settings":
                errors = app.settings.update(b.get("changes") or {})
                return self._json({"settings": app.settings.values, "errors": errors})
            if path == "/api/app/settings/reset":
                app.settings.reset()
                return self._json({"settings": app.settings.values, "errors": {}})
            if path == "/api/app/quit":
                threading.Timer(0.3, app.quit).start()  # answer first, then stop
                return self._json({"quitting": True})
            if path == "/api/app/run":
                return self._json(app.run(str(b.get("kind", "")), b.get("params") or {}).summary())
            parts = path.strip("/").split("/")
            if len(parts) == 5 and parts[2] == "jobs" and parts[4] == "stop":
                return self._json(app.jobs.stop(parts[3], now=bool(b.get("now"))).summary())
            if path == "/api/app/phone/lock":
                return self._json({"locked": app.phone.set_locked(bool(b.get("locked", True)))})
            if path == "/api/app/phone/tap":
                return self._json(app.phone.tap(float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])))
            if path == "/api/app/phone/swipe":
                return self._json(app.phone.swipe(float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"]),
                                                  float(b["w"]), float(b["h"]), int(b.get("ms", 300))))
            if path == "/api/app/phone/key":
                return self._json(app.phone.key(str(b.get("key", ""))))
            if path == "/api/app/screenshot":
                dec = lambda k: base64.b64decode(b[k]) if b.get(k) else None  # noqa: E731
                return self._json(app.screenshot(overlay_png=dec("overlay_png"), frame_png=dec("frame_png"),
                                                 label=b.get("label")))
            if len(parts) == 4 and parts[2] == "refine":
                flow, step = app.refine, parts[3]
                p = b.get("params") or {}
                handlers = {
                    "configure": lambda: flow.configure(str(p.get("champion", "")), str(p.get("data", ""))),
                    "scan": flow.scan, "check": flow.check, "promote": flow.promote,
                    "next_round": flow.next_round, "reset": flow.reset,
                    "propose": lambda: flow.propose(str(p.get("pattern", ""))),
                    "test": lambda: flow.test(**{k: v for k, v in p.items() if k in (
                        "title", "hypothesis", "mode", "test", "max_games", "futility_pairs")}),
                }
                if step not in handlers:
                    return self._err(404, f"no refine step {step}")
                return self._json(handlers[step]())
            return self._err(404, f"no route {path}")

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


def secrets_equal(a: str, b: str) -> bool:
    import hmac

    return hmac.compare_digest(a.encode(), b.encode())


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
        if self.studio.app is not None:
            self.studio.app.jobs.stop_all()
        self.httpd.shutdown()
        self.httpd.server_close()
