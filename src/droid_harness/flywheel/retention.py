"""Disk retention & compaction for 50-100+ game campaigns (README §11 "Storage"; run log Fix B).

Keeps compact JSON/JSONL telemetry (~1.5 MB/match) on 100% of matches, prunes raw
~180 MB ``stream.h264`` files on clean completed wins, and extracts diagnostic keyframes
before pruning losses when requested.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _extract_loss_keyframes(match_dir: Path) -> list[str]:
    h264 = match_dir / "stream.h264"
    if not h264.exists():
        return []
    try:
        import av
        import cv2
    except Exception:
        return []
    saved: list[str] = []
    try:
        container = av.open(str(h264))
        ring: list[Any] = []
        for frame in container.decode(video=0):
            ring.append(frame.to_ndarray(format="bgr24"))
            if len(ring) > 90:  # ~1.5s rolling tail buffer
                del ring[0]
        container.close()
        if not ring:
            return []
        kf_dir = match_dir / "keyframes"
        kf_dir.mkdir(parents=True, exist_ok=True)
        picks = [("pre_end.jpg", ring[0]), ("end.jpg", ring[-1])]
        for name, img in picks:
            out = kf_dir / name
            cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            saved.append(str(out.name))
    except Exception:
        return saved
    return saved


def retain_match(match_dir: str | Path, *, prune_loss_video: bool = False,
                 dry_run: bool = False) -> dict[str, Any]:
    """Compact one match directory; returns bytes freed and actions taken."""
    d = Path(match_dir)
    sum_path = d / "summary.json"
    if not sum_path.exists():
        return {"match_id": d.name, "action": "skipped_incomplete", "bytes_freed": 0}
    try:
        summary = json.loads(sum_path.read_text())
    except (OSError, ValueError):
        return {"match_id": d.name, "action": "skipped_corrupt", "bytes_freed": 0}

    status = summary.get("status")
    result = summary.get("result")
    clean_win = status == "completed" and result == "win"
    if not clean_win and not prune_loss_video:
        kfs = [] if dry_run else _extract_loss_keyframes(d)
        return {"match_id": d.name, "action": "kept_video", "keyframes": kfs, "bytes_freed": 0}

    kfs = []
    if not clean_win and not dry_run:
        kfs = _extract_loss_keyframes(d)

    targets = [d / "stream.h264", d / "stream.pts", d / "frames.jsonl"]
    freed = 0
    removed = []
    for p in targets:
        if p.exists():
            freed += p.stat().st_size
            removed.append(p.name)
            if not dry_run:
                p.unlink(missing_ok=True)
    return {
        "match_id": d.name,
        "action": "pruned_win" if clean_win else "pruned_non_win",
        "removed": removed,
        "keyframes": kfs,
        "bytes_freed": freed,
    }


def retain_dir(data_root: str | Path, *, prune_loss_video: bool = False,
               dry_run: bool = False) -> dict[str, Any]:
    root = Path(data_root)
    dirs = [root] if (root / "summary.json").exists() else sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "summary.json").exists()
    )
    items = [retain_match(d, prune_loss_video=prune_loss_video, dry_run=dry_run) for d in dirs]
    total_freed = sum(int(x.get("bytes_freed", 0)) for x in items)
    pruned = sum(1 for x in items if str(x.get("action", "")).startswith("pruned"))
    return {
        "matches_scanned": len(items),
        "matches_pruned": pruned,
        "mb_freed": round(total_freed / (1024 * 1024), 2),
        "dry_run": dry_run,
    }
