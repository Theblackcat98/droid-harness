#!/usr/bin/env python3
"""Rolling Trophy Road win-rate window over recorded matches.

Reads ``summary.json`` files under one or more data roots, keeps completed ladder matches,
de-duplicates by match id, orders them by start time and reports the win rate of the last
``--window`` matches (default 40) with a Wilson 95 % interval.

    python tools/ladder_window.py data/ladder-v9-40 --window 40
    python tools/ladder_window.py data --window 40 --detail

Exit code 0 when the window meets ``--target`` (default 0.80), 1 otherwise.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys


def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def load(roots: list[str]) -> list[dict]:
    rows: dict[str, dict] = {}
    for root in roots:
        for path in glob.glob(os.path.join(root, "**", "summary.json"), recursive=True):
            try:
                d = json.load(open(path))
            except (OSError, ValueError):
                continue
            if d.get("opponent_kind") != "ladder":
                continue
            if d.get("status") != "completed":
                continue
            if d.get("result") not in ("win", "loss", "draw"):
                continue
            key = d.get("match_id") or os.path.dirname(path)
            prev = rows.get(key)
            if prev is None or d.get("started_unix", 0) < prev.get("started_unix", 0):
                d["_path"] = path
                rows[key] = d
    return sorted(rows.values(), key=lambda d: d.get("started_unix", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data", nargs="+", help="data root(s) searched recursively for summary.json")
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()

    matches = load(a.data)
    if not matches:
        print("no completed ladder matches found")
        return 1
    for d in matches:
        if a.detail:
            print(f"{d.get('started_unix', 0):.0f} {d.get('match_id')} {d.get('strategy_name')} "
                  f"{d.get('result')} {d.get('my_crowns')}-{d.get('opp_crowns')} "
                  f"lvl {d.get('my_king_level_seen')}v{d.get('opp_king_level_seen')}")
    window = matches[-a.window:]
    wins = sum(1 for d in window if d.get("result") == "win")
    losses = sum(1 for d in window if d.get("result") == "loss")
    draws = len(window) - wins - losses
    lo, hi = wilson(wins, len(window))
    rate = wins / len(window)
    print(f"matches recorded: {len(matches)}   window: last {len(window)}")
    print(f"record: {wins}W-{losses}L-{draws}D   win rate {rate:.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    over = matches[-a.window:]
    print("window timeline:", "".join("W" if d["result"] == "win" else ("L" if d["result"] == "loss" else "D")
                                      for d in over))
    ok = len(window) >= a.window and rate >= a.target
    print(f"target {a.target:.0%} over {a.window}: {'MET' if ok else 'NOT MET'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
