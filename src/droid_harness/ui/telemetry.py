"""Live telemetry: a zero-copy tap on the brain loop and a rate-capped publisher.

The brain thread calls ``TelemetryTap.publish`` once per step with references it already
holds (frame, observation, decision, stage timings, detector boxes). That is one lock and one
tuple assignment: no copies, no JSON, no image encoding. A separate ``LivePublisher`` thread
samples the latest tuple at ``fps`` (15 by default), builds the snapshot JSON and a JPEG of the
frame, and wakes the HTTP clients. If nobody is watching, it still only costs the sampling.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from .snapshot import CostBook, snapshot_from_record

log = logging.getLogger(__name__)


class TelemetryTap:
    def __init__(self, *, max_taps: int = 48) -> None:
        self._lock = threading.Lock()
        self._latest: tuple | None = None
        self._seq = 0
        self._taps: deque[tuple[int, int, int, str, int]] = deque(maxlen=max_taps)

    # brain thread ------------------------------------------------------------------------
    def publish(self, item: tuple) -> None:
        with self._lock:
            self._latest = item
            self._seq += 1

    # executor thread ---------------------------------------------------------------------
    def on_sent(self, sent) -> None:
        e = sent.event
        self._taps.append((sent.t_sent_ns, int(e.x), int(e.y), str(e.tag), int(e.action)))

    # publisher thread --------------------------------------------------------------------
    def latest(self) -> tuple[int, tuple | None]:
        with self._lock:
            return self._seq, self._latest

    def recent_taps(self, now_ns: int | None = None, window_ms: float = 1200.0) -> list[dict]:
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        out = []
        for t, x, y, tag, action in list(self._taps):
            age = (now_ns - t) / 1e6
            if 0 <= age <= window_ms and action == 0:  # DOWN events mark where a touch landed
                out.append({"x": x, "y": y, "age_ms": round(age, 1), "tag": tag.split(":")[0]})
        return out


def experiment_state(runner, active_arm: str | None = None) -> dict[str, Any]:
    """What the experiment strip shows, from an ``ExperimentRunner``'s public API."""
    from ..flywheel.eval.sprt import bounds

    t = runner.tallies()
    better, worse, tied = runner.pair_counts()
    lo, hi = bounds(runner.cfg.alpha, runner.cfg.beta)
    d = runner.decision
    return {
        "id": runner.cfg.experiment_id, "title": runner.cfg.title, "status": runner.status,
        "test": runner.cfg.test, "p0": runner.cfg.p0, "p1": runner.cfg.p1,
        "matches": len(runner.matches), "max_games": runner.cfg.max_games, "active_arm": active_arm,
        "arms": {arm: {"wins": a.wins, "losses": a.losses, "draws": a.draws, "aborts": a.aborts}
                 for arm, a in t.items()},
        "pairs": {"candidate_better": better, "champion_better": worse, "tied": tied,
                  "dropped": len(runner.dropped_pairs)},
        "dense": runner.dense_pair_stats(),
        "llr": round(d.llr, 3) if d else 0.0, "lower": round(d.lower if d else lo, 3),
        "upper": round(d.upper if d else hi, 3),
        "futility": runner.futility_reason, "void": runner.void_reason,
    }


def _safe_dict(d: Any) -> dict:
    for _ in range(3):  # the brain thread may grow it while we copy
        try:
            return dict(d or {})
        except RuntimeError:
            continue
    return {}


class LivePublisher:
    """Turns the tap's latest tuple into ``snapshot_json`` + ``jpeg`` at a capped rate."""

    def __init__(self, tap: TelemetryTap, context: Callable[[], dict], *, costs: CostBook | None = None,
                 fps: float = 15.0, jpeg_quality: int = 72, heartbeat_s: float = 2.0) -> None:
        self.tap, self.context, self.costs = tap, context, costs or CostBook(None)
        self.period = 1.0 / max(1.0, fps)
        self.jpeg_quality = jpeg_quality
        self.heartbeat_s = heartbeat_s
        self.cv = threading.Condition()
        self.version = 0
        self.snapshot_json = b"{}"
        self.jpeg: bytes = b""
        self.jpeg_seq = 0
        self.build_ms = 0.0
        self._running = False
        self._thread: threading.Thread | None = None
        self._lat_cache: tuple[float, dict] = (0.0, {})

    def start(self) -> LivePublisher:
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="UI-publisher", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        with self.cv:
            self.cv.notify_all()
        if self._thread:
            self._thread.join(timeout=1)

    @property
    def running(self) -> bool:
        return self._running

    def wait(self, version: int, timeout: float) -> int:
        with self.cv:
            self.cv.wait_for(lambda: self.version != version or not self._running, timeout)
            return self.version

    def _loop(self) -> None:
        last_seq, last_pub = -1, 0.0
        while self._running:
            t0 = time.monotonic()
            seq, raw = self.tap.latest()
            due_heartbeat = t0 - last_pub >= self.heartbeat_s
            if seq != last_seq or due_heartbeat:
                try:
                    self.publish_once(raw)
                except Exception:  # the UI must never take the bot down
                    log.exception("ui snapshot failed")
                last_seq, last_pub = seq, t0
            time.sleep(max(0.0, self.period - (time.monotonic() - t0)))

    def publish_once(self, raw: tuple | None) -> dict:
        t0 = time.perf_counter()
        ctx = self.context() or {}
        snap = self.build(raw, ctx)
        jpeg = None
        frame = raw[0] if raw else None
        if frame is not None and getattr(frame, "img", None) is not None and frame.seq != self.jpeg_seq:
            import cv2

            ok, buf = cv2.imencode(".jpg", frame.img, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok:
                jpeg = buf.tobytes()
        self.build_ms = (time.perf_counter() - t0) * 1000
        snap["ui_build_ms"] = round(self.build_ms, 2)
        data = json.dumps(snap, default=str, separators=(",", ":")).encode()
        with self.cv:
            if jpeg is not None:
                self.jpeg, self.jpeg_seq = jpeg, frame.seq
            self.snapshot_json = data
            self.version += 1
            self.cv.notify_all()
        return snap

    def _latency_stats(self, session) -> dict:
        now = time.monotonic()
        if now - self._lat_cache[0] >= 1.0 and session is not None:
            s = getattr(session, "match_latency", None)
            self._lat_cache = (now, s.summary("total") if s is not None else {})
        return self._lat_cache[1]

    def build(self, raw: tuple | None, ctx: dict) -> dict:
        frame, obs, decision, durations, dets = (raw if raw else (None, None, None, {}, []))
        perception = ctx.get("perception")
        session = ctx.get("session")
        # BotRunner passes a session Observation; the bare simulator passes the GameState itself.
        state = obs if hasattr(obs, "digest") else getattr(obs, "state", None)
        if decision is not None and hasattr(decision, "to_record"):
            record = decision.to_record()
        elif state is not None:
            record = {"t_ms": state.t_ms, "state": state.digest()}
        else:
            record = {"t_ms": 0}
        record["lat"] = durations or {}
        snap = snapshot_from_record(
            record, params=ctx.get("params"), costs=self.costs,
            tower_max=_safe_dict(getattr(perception, "tower_max_hp", None)) if state is not None else None)
        in_battle = state is not None
        if not in_battle:
            snap["decision"]["candidates"] = []
        stats = self._latency_stats(session)
        snap["latency"].update({"p50": round(stats.get("p50", 0.0) or 0.0, 2),
                                "p99": round(stats.get("p99", 0.0) or 0.0, 2)})
        menu_taps = len(getattr(decision, "events", None) or []) if decision is not None and not in_battle else 0
        snap.update({
            "mode": "live",
            "screen": getattr(obs, "screen", None) or ctx.get("screen") or ("IN_BATTLE" if state is not None else None),
            "in_battle": in_battle,
            "frame": {"seq": getattr(frame, "seq", None),
                      "w": int(frame.img.shape[1]) if frame is not None else None,
                      "h": int(frame.img.shape[0]) if frame is not None else None,
                      "has_image": frame is not None},
            "detections": list(dets or []) if in_battle else [],
            "taps": self.tap.recent_taps(),
            "menu_taps": menu_taps,
            "match": ctx.get("match"),
            "strategy": ctx.get("strategy"),
            "runner": ctx.get("runner"),
            "experiment": ctx.get("experiment"),
            "elixir_leaked": round(float(getattr(getattr(perception, "elixir", None), "leaked", 0.0) or 0.0), 2)
            if in_battle else None,
            "warning_counts": _safe_dict(getattr(perception, "warnings", None)) if in_battle else {},
            "observe_only": bool(ctx.get("observe_only")),
        })
        return snap
