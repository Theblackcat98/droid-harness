"""Recorded matches for the replay inspector: index, timelines, and (when kept) video frames.

A match directory is any folder with a ``summary.json`` or ``decisions.jsonl`` (what the
recorder writes). Most committed matches keep only the compact files, so the inspector
works from ``decisions.jsonl`` alone (the tactical map); when ``stream.h264`` survived
retention, frames are decoded once in the background at ~10 fps and served as JPEGs.
"""

from __future__ import annotations

import bisect
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .snapshot import (
    CostBook,
    StrategyIndex,
    device_geometry,
    running_tower_max,
    snapshot_from_record,
    strategy_summary,
)

log = logging.getLogger(__name__)


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def _read_jsonl(p: Path) -> list[dict]:
    out = []
    try:
        lines = p.read_text().splitlines()
    except OSError:
        return out
    for line in lines:
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def is_match_dir(d: Path) -> bool:
    return d.is_dir() and ((d / "summary.json").exists() or (d / "decisions.jsonl").exists())


def find_match_dirs(roots: list[Path], max_depth: int = 3) -> list[tuple[Path, Path]]:
    """(root, match_dir) for every match under the roots, a root may itself be a match."""
    found: list[tuple[Path, Path]] = []
    seen: set[Path] = set()

    def walk(root: Path, d: Path, depth: int) -> None:
        if d.resolve() in seen:
            return
        if is_match_dir(d):
            seen.add(d.resolve())
            found.append((root, d))
            return
        if depth >= max_depth:
            return
        try:
            children = sorted(p for p in d.iterdir() if p.is_dir() and not p.name.startswith("."))
        except OSError:
            return
        for c in children:
            walk(root, c, depth + 1)

    for r in roots:
        r = Path(r)
        if r.exists():
            walk(r.parent if is_match_dir(r) else r, r, 0)
    return found


class VideoCache:
    """Decodes ``stream.h264`` once into JPEG frames sampled every ``stride_ms`` of match time."""

    def __init__(self, match_dir: Path, start_pts_us: int | None, *, stride_ms: int = 100,
                 jpeg_quality: int = 70, max_t_ms: float | None = None) -> None:
        self.match_dir, self.start = match_dir, start_pts_us
        self.stride_ms, self.quality, self.max_t_ms = stride_ms, jpeg_quality, max_t_ms
        self.times: list[int] = []
        self.jpegs: list[bytes] = []
        self.size: tuple[int, int] | None = None
        self.done = False
        self.error: str | None = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._decode, name=f"UI-decode-{match_dir.name}", daemon=True)
        self._thread.start()

    def _decode(self) -> None:
        try:
            import cv2

            from ..domains.clash_royale.replay import iter_frames

            last = None
            start = self.start
            for f in iter_frames(self.match_dir):
                if start is None:
                    start = f.pts_us
                if f.pts_us < start:
                    continue  # key-frame pre-roll before the battle
                t_ms = int((f.pts_us - start) / 1000)
                if self.max_t_ms is not None and t_ms > self.max_t_ms + 2000:
                    break
                if last is not None and t_ms - last < self.stride_ms:
                    continue
                ok, buf = cv2.imencode(".jpg", f.img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if not ok:
                    continue
                with self._lock:
                    self.times.append(t_ms)
                    self.jpegs.append(buf.tobytes())
                    self.size = (int(f.img.shape[1]), int(f.img.shape[0]))
                last = t_ms
        except Exception as e:  # a corrupt or partial recording still shows the map view
            self.error = f"{type(e).__name__}: {e}"
            log.warning("video decode failed for %s: %s", self.match_dir, e)
        finally:
            self.done = True

    def frame_at(self, t_ms: float) -> tuple[int, bytes] | None:
        with self._lock:
            if not self.times:
                return None
            i = max(0, bisect.bisect_right(self.times, t_ms) - 1)
            return self.times[i], self.jpegs[i]

    def status(self) -> dict:
        with self._lock:
            return {"frames": len(self.times), "decoded_to_ms": self.times[-1] if self.times else 0,
                    "done": self.done, "error": self.error, "size": self.size}


class MatchLibrary:
    def __init__(self, roots: list[str | Path], *, strategies: StrategyIndex | None = None,
                 costs: CostBook | None = None, root: Path = Path("."), max_videos: int = 2) -> None:
        self.roots = [Path(r) for r in roots]
        self.strategies = strategies or StrategyIndex(root / "configs/strategies")
        self.costs = costs or CostBook(None)
        self.root = root
        self.max_videos = max_videos
        self._index: dict[str, Path] = {}
        self._videos: OrderedDict[str, VideoCache] = OrderedDict()
        self._lock = threading.Lock()

    # --- index --------------------------------------------------------------------------------
    def refresh(self) -> list[dict]:
        rows = []
        index: dict[str, Path] = {}
        for root, d in find_match_dirs(self.roots):
            try:
                mid = d.relative_to(root).as_posix()
            except ValueError:
                mid = d.name
            if mid == ".":
                mid = d.name
            index[mid] = d
            rows.append(self._row(mid, d))
        with self._lock:
            self._index = index
        rows.sort(key=lambda r: r.get("started_unix") or 0, reverse=True)
        return rows

    def _row(self, mid: str, d: Path) -> dict:
        s = _read_json(d / "summary.json")
        m = _read_json(d / "meta.json")
        info = {**m, **s}
        strat = self.strategies.get(info.get("strategy_id"))
        return {
            "id": mid, "group": mid.rsplit("/", 1)[0] if "/" in mid else "", "match_id": info.get("match_id") or d.name,
            "result": info.get("result"), "status": info.get("status"),
            "my_crowns": info.get("my_crowns"), "opp_crowns": info.get("opp_crowns"),
            "duration_s": info.get("duration_s"), "started_unix": info.get("started_unix"),
            "strategy_id": info.get("strategy_id"), "strategy": strat[0] if strat else None,
            "arm": info.get("arm"), "experiment_id": info.get("experiment_id"), "pair_id": info.get("pair_id"),
            "opponent_kind": info.get("opponent_kind"), "device": info.get("device"),
            "opp_deck_seen": info.get("opp_deck_seen") or [],
            "has_video": (d / "stream.h264").exists(), "has_decisions": (d / "decisions.jsonl").exists(),
        }

    def path(self, mid: str) -> Path | None:
        with self._lock:
            p = self._index.get(mid)
        if p is None:  # new matches since the last listing
            self.refresh()
            with self._lock:
                p = self._index.get(mid)
        return p

    # --- one match -------------------------------------------------------------------------
    def load(self, mid: str) -> dict | None:
        d = self.path(mid)
        if d is None:
            return None
        summary = _read_json(d / "summary.json")
        meta = _read_json(d / "meta.json")
        info = {**meta, **summary}
        records = sorted(_read_jsonl(d / "decisions.jsonl"), key=lambda r: r.get("t_ms") or 0)
        strat = self.strategies.get(info.get("strategy_id"))
        params = strat[1] if strat else None
        maxes = running_tower_max(records)
        timeline = [snapshot_from_record(r, params=params, costs=self.costs, tower_max=mx)
                    for r, mx in zip(records, maxes, strict=True)]
        events = list(summary.get("events") or [])
        summary_out = {k: v for k, v in summary.items() if k != "events"}
        return {
            "id": mid, "meta": meta, "summary": summary_out, "events": events,
            "strategy": strategy_summary(strat[0] if strat else None, info.get("strategy_id"), params,
                                         strat[2] if strat else ""),
            "geometry": device_geometry(info.get("device") or ("sim" if info.get("mode") == "sim" else None),
                                        self.root),
            "has_video": (d / "stream.h264").exists(),
            "timeline": timeline,
        }

    def video(self, mid: str) -> VideoCache | None:
        d = self.path(mid)
        if d is None or not (d / "stream.h264").exists():
            return None
        with self._lock:
            vc = self._videos.get(mid)
            if vc is not None:
                self._videos.move_to_end(mid)
                return vc
            info = {**_read_json(d / "meta.json"), **_read_json(d / "summary.json")}
            dur = info.get("duration_s")
            vc = VideoCache(d, info.get("start_pts_us"),
                            max_t_ms=dur * 1000 if isinstance(dur, (int, float)) else None)
            self._videos[mid] = vc
            while len(self._videos) > self.max_videos:
                self._videos.popitem(last=False)
            return vc


def read_ledger(store: Path) -> list[dict]:
    """Latest record per experiment from ``ledger/experiments.jsonl``, newest first."""
    by_id: dict[str, dict] = {}
    for rec in _read_jsonl(store):
        if rec.get("experiment_id"):
            by_id[rec["experiment_id"]] = rec
    return sorted(by_id.values(), key=lambda r: r.get("experiment_id", ""), reverse=True)


def timeline_bytes(match: dict[str, Any]) -> bytes:
    return json.dumps(match, default=str, separators=(",", ":")).encode()
