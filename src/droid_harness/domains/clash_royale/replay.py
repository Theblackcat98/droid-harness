"""Replay recorded matches through perception (M3 replay regression), no phone needed.

Decodes a match's ``stream.h264`` with its recorded device timestamps (``stream.pts``,
written by the recorder), starts the match at the recorded ``start_pts_us`` and feeds
every frame to a fresh ``CRPerception`` the way the live loop does (minus live frame
drops). The emitted events (my plays, spawn plays, tower destructions) are compared with
a previous run, so a new detector, a recalibration or a threshold change shows what it
gained and lost before it touches the phone.

Older recordings have no ``stream.pts``; they replay at an assumed frame rate, which
gets event *times* only roughly right.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from ...core.transport.frame_slot import Frame

log = logging.getLogger(__name__)


def iter_frames(match_dir: str | Path, fallback_fps: float = 60.0) -> Iterator[Frame]:
    import av

    d = Path(match_dir)
    pts_path = d / "stream.pts"
    pts = [int(x) for x in pts_path.read_text().split()] if pts_path.exists() else None
    if pts is None:
        log.warning("%s has no stream.pts; assuming %.0f fps", d.name, fallback_fps)
    with av.open(str(d / "stream.h264"), format="h264") as container:
        for i, vf in enumerate(container.decode(video=0)):
            p = pts[i] if pts is not None and i < len(pts) else int(i * 1e6 / fallback_fps)
            yield Frame(i + 1, vf.to_ndarray(format="bgr24"), p, 0, 0)


def _read_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def replay_match(match_dir: str | Path, perception) -> dict:
    d = Path(match_dir)
    meta, summary = _read_json(d / "meta.json"), _read_json(d / "summary.json")
    start = summary.get("start_pts_us") or meta.get("start_pts_us")
    dur = summary.get("duration_s")
    end_ms = dur * 1000 if isinstance(dur, (int, float)) else None
    frames, started = 0, False
    for f in iter_frames(d):
        if start is not None and f.pts_us < start:
            continue  # key-frame pre-roll before the battle
        if not started:
            perception.start_match(f)
            started = True
        if end_ms is not None and (f.pts_us - perception.start_pts_us) / 1000 > end_ms:
            break
        perception.process(f)
        frames += 1
    return {"match_id": summary.get("match_id") or meta.get("match_id") or d.name, "frames": frames,
            "result": summary.get("result"), "events": list(perception.events) if started else [],
            "warnings": dict(perception.warnings) if started else {}}


def _same(a: dict, b: dict, tol_ms: int, tol_tiles: float) -> bool:
    if a.get("kind") != b.get("kind") or a.get("card") != b.get("card") or a.get("tower") != b.get("tower"):
        return False
    if abs(a.get("t_ms", 0) - b.get("t_ms", 0)) > tol_ms:
        return False
    ta, tb = a.get("tile"), b.get("tile")
    if ta is not None and tb is not None:
        if ((ta[0] - tb[0]) ** 2 + (ta[1] - tb[1]) ** 2) ** 0.5 > tol_tiles:
            return False
    return True


def diff_events(old: list[dict], new: list[dict], *, tol_ms: int = 600, tol_tiles: float = 2.0) -> dict:
    """Greedy one-to-one matching in time order. ``lost`` = only in old, ``gained`` = only in new."""
    used: set[int] = set()
    lost = []
    for a in sorted(old, key=lambda e: e.get("t_ms", 0)):
        j = next((j for j, b in enumerate(new) if j not in used and _same(a, b, tol_ms, tol_tiles)), None)
        if j is None:
            lost.append(a)
        else:
            used.add(j)
    gained = [b for j, b in enumerate(new) if j not in used]
    kinds = sorted({e.get("kind") for e in old} | {e.get("kind") for e in new})
    by_kind = {k: {"old": sum(e.get("kind") == k for e in old), "new": sum(e.get("kind") == k for e in new),
                   "lost": sum(e.get("kind") == k for e in lost),
                   "gained": sum(e.get("kind") == k for e in gained)} for k in kinds}
    return {"matched": len(used), "lost": lost, "gained": gained, "by_kind": by_kind}


def diff_runs(old: dict, new: dict, **kw) -> dict:
    """Both are ``{match_id: replay_match(...)}``. Totals plus per-match lost/gained."""
    totals: Counter = Counter()
    per = {}
    for mid in sorted(set(old) & set(new)):
        d = diff_events(old[mid]["events"], new[mid]["events"], **kw)
        totals.update({"matched": d["matched"], "lost": len(d["lost"]), "gained": len(d["gained"])})
        if d["lost"] or d["gained"]:
            per[mid] = {"lost": d["lost"][:20], "gained": d["gained"][:20], "by_kind": d["by_kind"]}
    return {"matches_compared": len(set(old) & set(new)), "only_old": sorted(set(old) - set(new)),
            "only_new": sorted(set(new) - set(old)), **dict(totals), "changed_matches": per}
