#!/usr/bin/env python3
"""Fetch KataCR training assets (MIT-licensed) for the M3 detector.

Downloads via git (shallow + sparse + blob-less, so only what's asked for):
  1. label list           -> models/katacr_label_list.py (class reference only)
  2. sprite cut-outs       -> <out>/segment/<our_base_class>/   (classes from models/katacr_alias.json)
     File names keep KataCR's side flag: ``knight_0_*.png`` = yours, ``knight_1_*.png`` = theirs;
     ``learn/synth.py::load_sprites`` turns that into knight_me / knight_opp.
  3. HP-bar sprites        -> <out>/bars/          (``--no-bars`` to skip)
  4. empty arenas          -> <out>/backgrounds/   (``--no-backgrounds`` to skip)
  5. labeled real frames   -> <out>/_repo/images/part2  (``--part2``; large: ~7,400 frames)
     Convert with ``droid-harness convert-katacr`` for a held-out real validation set.

Usage:
  python tools/fetch_katacr.py --out data/katacr [--part2]
  python tools/fetch_katacr.py --label-list-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/wty-yy/Clash-Royale-Detection-Dataset.git"
LABEL_URL = "https://raw.githubusercontent.com/wty-yy/KataCR/master/katacr/constants/label_list.py"


def run(*a: str) -> None:
    print("+", " ".join(a), flush=True)
    subprocess.run(a, check=True)


def link_all(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(src.glob("*.png")):
        target = dst / p.name
        if not target.exists():
            try:
                target.symlink_to(p.resolve())
            except OSError:
                shutil.copy(p, target)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/katacr")
    ap.add_argument("--alias", default=str(ROOT / "models" / "katacr_alias.json"),
                    help="KataCR name -> our base class (the single source for which segments to fetch)")
    ap.add_argument("--label-list-only", action="store_true")
    ap.add_argument("--no-bars", action="store_true")
    ap.add_argument("--no-backgrounds", action="store_true")
    ap.add_argument("--part2", action="store_true", help="also fetch the ~7,400 hand-labeled real frames")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    dest = ROOT / "models" / "katacr_label_list.py"
    print(f"fetching label list -> {dest}", flush=True)
    urllib.request.urlretrieve(LABEL_URL, dest)
    if args.label_list_only:
        print("label-list-only: done")
        return 0

    alias: dict[str, str] = json.loads(Path(args.alias).read_text())
    work = out / "_repo"
    if not (work / ".git").exists():
        run("git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO, str(work))
    paths = [f"images/segment/{k}" for k in alias] + ["LICENSE", "README_en.md"]
    if not args.no_bars:
        paths.append("images/segment/bar")
    if not args.no_backgrounds:
        paths.append("images/segment/backgrounds")
    if args.part2:
        paths.append("images/part2")
    run("git", "-C", str(work), "sparse-checkout", "set", "--no-cone", *paths)
    run("git", "-C", str(work), "checkout")

    counts: dict[str, dict[str, int]] = {}
    for kata, ours in alias.items():
        src = work / "images" / "segment" / kata
        n = link_all(src, out / "segment" / ours)
        mine = len(list(src.glob(f"{kata}_0_*.png")))
        counts[ours] = {"sprites": n, "yours": mine, "theirs": n - mine}
    report = {"segment": counts}
    if not args.no_bars:
        report["bars"] = link_all(work / "images" / "segment" / "bar", out / "bars")
    if not args.no_backgrounds:
        report["backgrounds"] = link_all(work / "images" / "segment" / "backgrounds", out / "backgrounds")
    if args.part2:
        report["part2"] = str(work / "images" / "part2")
    print(json.dumps(report, indent=1))
    thin = [c for c, v in counts.items() if not v["yours"] or not v["theirs"]]
    if thin:
        print(f"no sprites for one side of: {thin} — cut those from your own recordings", file=sys.stderr)
    print("upstream license: MIT (see _repo/LICENSE). Dataset snapshot May 2024: newer cards missing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
