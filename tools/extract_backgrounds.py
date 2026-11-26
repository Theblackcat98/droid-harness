#!/usr/bin/env python3
"""Extract empty-arena backgrounds from our recorded matches for synth compositing.

Scans recorded `stream.h264` files, samples frames, and keeps ones whose arena
region is static and unit-free (approximated here: low frame-to-frame difference
inside the arena ROI and no detector available, so we keep calm mid-game frames
for manual review; final curation is a human glance in a gallery).

Writes *arena crops* (layout.arena), the same region the detector sees, because
``learn/synth.py`` letterboxes backgrounds exactly like inference does. ``--full``
keeps whole frames instead (for screen references, not for synth).

Usage:
  python tools/extract_backgrounds.py --data data/matches-livecheck2 --out data/backgrounds
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def arena_rect(layout_path: str, shape) -> tuple[int, int, int, int]:
    from droid_harness.domains.clash_royale.perception.grid import Layout
    lay = Layout.load(layout_path)
    return lay.arena.to_px(shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/matches2")
    ap.add_argument("--layout", default="configs/domains/clash_royale/rois/phone-a.yaml")
    ap.add_argument("--out", default="data/backgrounds")
    ap.add_argument("--every", type=int, default=120, help="sample every Nth frame")
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--full", action="store_true", help="save whole frames instead of arena crops")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    kept = 0
    for h264 in sorted(Path(args.data).glob("*/stream.h264")):
        if kept >= args.max:
            break
        # Probe frames via ffmpeg to PNGs in a temp dir (avoids decoding full video in RAM).
        tmp = out / "_frames"
        tmp.mkdir(exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(h264),
                        "-vf", f"select='not(mod(n\\,{args.every}))'", "-vsync", "0",
                        str(tmp / "f_%04d.png")], check=False)
        prev = None
        first = True
        for png in sorted(tmp.glob("*.png")):
            img = cv2.imread(str(png))
            if img is None:
                continue
            x0, y0, x1, y1 = arena_rect(args.layout, img.shape)
            arena = img[y0:y1, x0:x1]
            keep = img if args.full else arena
            if first:
                # Match start is the emptiest the arena gets; always a candidate.
                cv2.imwrite(str(out / f"{h264.parent.name}_{png.stem}.png"), keep)
                kept += 1
                first = False
                if kept >= args.max:
                    break
            if prev is not None:
                diff = float(np.abs(arena.astype(np.int16) - prev.astype(np.int16)).mean())
                if diff < 2.0:  # static arena: candidate background
                    cv2.imwrite(str(out / f"{h264.parent.name}_{png.stem}.png"), keep)
                    kept += 1
                    if kept >= args.max:
                        break
            prev = arena
        for png in tmp.glob("*.png"):
            png.unlink()
    print(f"kept {kept} candidate backgrounds in {out} (curate by glance before training)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
