#!/usr/bin/env python3
"""Fetch KataCR training assets (MIT-licensed) for the M3 detector.

Downloads (via git, shallow + sparse when possible):
  1. katacr label list  -> models/katacr_label_list.py (class reference only)
  2. detection slices for our closed set only -> data/katacr/segment/<cls>/

Usage:
  python tools/fetch_katacr.py --out data/katacr [--classes knight,princess,minion,...]
  python tools/fetch_katacr.py --label-list-only   # tiny: just the class reference
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/wty-yy/Clash-Royale-Detection-Dataset.git"
LABEL_URL = "https://raw.githubusercontent.com/wty-yy/KataCR/master/katacr/constants/label_list.py"

# KataCR segment dir name -> our detector class (must stay in sync with katacr_alias.json)
SEGMENTS = {
    "knight": "knight", "princess": "princess", "minion": "minion", "giant": "giant",
    "musketeer": "musketeer", "mini-pekka": "mini_pekka", "arrows": "arrows_effect",
    "fireball": "fireball_effect", "queen-tower": "tower_princess",
    "king-tower": "tower_king",
}


def run(*a: str) -> None:
    print("+", " ".join(a), flush=True)
    subprocess.run(a, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/katacr")
    ap.add_argument("--classes", default=",".join(SEGMENTS),
                    help="KataCR segment dirs to fetch (comma-separated)")
    ap.add_argument("--label-list-only", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import urllib.request
    dest = ROOT / "models" / "katacr_label_list.py"
    print(f"fetching label list -> {dest}", flush=True)
    urllib.request.urlretrieve(LABEL_URL, dest)

    if args.label_list_only:
        print("label-list-only: done")
        return 0

    wanted = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in wanted if c not in SEGMENTS]
    if unknown:
        print(f"unknown segment classes: {unknown}", file=sys.stderr)
        return 2
    work = out / "_repo"
    if not (work / ".git").exists():
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(work))
    run("git", "-C", str(work), "sparse-checkout", "set", "--no-cone",
        *[f"images/segment/{c}" for c in wanted], "LICENSE", "README_en.md")
    run("git", "-C", str(work), "checkout")
    seg = out / "segment"
    seg.mkdir(exist_ok=True)
    counts = {}
    for c in wanted:
        src = work / "images" / "segment" / c
        dst = seg / SEGMENTS[c]
        dst.mkdir(exist_ok=True)
        n = 0
        for p in sorted(src.glob("*.png")):
            link = dst / p.name
            if not link.exists():
                try:
                    link.symlink_to(p)
                except OSError:
                    import shutil
                    shutil.copy(p, link)
            n += 1
        counts[c] = n
    print(json.dumps({"out": str(seg), "counts": counts}, indent=1))
    print("upstream license: MIT (see _repo/LICENSE). Dataset snapshots May 2024: newer cards missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
