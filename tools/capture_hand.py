#!/usr/bin/env python3
"""Capture hand-card crops from a live battle, for cards the matcher can't read yet.

A new card in the deck can't be played until it has templates, and the in-run harvester can't
label a card that has none (it learns a card's identity from the "next" slot, which it must
already read). This script takes the crops and you name them:

1. Play a battle (Training Camp is free) with the bot or by hand.
2. While it runs, look at the phone and note which hand slot (0-3, left to right; ``next`` for
   the small next-card slot) shows the unreadable card. It stays there: the bot never plays a
   card it can't read.

       python tools/capture_hand.py --out data/hand-2026-11-28 --port 43739 \\
           --label 0=skeletons --label 1=cannon --label 2=mega_minion

   ``--port`` is the live Studio port of the running job (its own stream frames). Without it,
   frames come from ``adb exec-out screencap``.
3. Look at ``<out>/sheet_<card>.png`` (the crops, enlarged), then:

       droid-harness promote-templates --from data/hand-2026-11-28

Crops the current matcher already reads are filed under that card too (more templates for the
cards you have), and every crop is kept under ``<out>/raw/`` with ``index.jsonl``.
"""

from __future__ import annotations

import argparse
import http.client
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from droid_harness.app import resolve_device_bundle, screen_classifier
from droid_harness.core.config import load_profile
from droid_harness.domains.clash_royale.perception.cards import CardMatcher
from droid_harness.domains.clash_royale.perception.grid import Layout
from droid_harness.domains.clash_royale.perception.hud import is_greyed

SLOTS = ("0", "1", "2", "3", "next")
THUMB = (124, 150)
PER_ROW = 10


def live_frame(port: int, host: str = "127.0.0.1") -> np.ndarray | None:
    """The running job's latest stream frame: exactly what the bot sees."""
    c = http.client.HTTPConnection(host, port, timeout=4.0)
    try:
        c.request("GET", "/api/live/frame.png")
        r = c.getresponse()
        body = r.read() if r.status == 200 else b""
    except OSError:
        return None
    finally:
        c.close()
    return cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) if body else None


def adb_frame(adb: list[str]) -> np.ndarray | None:
    try:
        body = subprocess.run([*adb, "exec-out", "screencap", "-p"], capture_output=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR) if body else None


def parse_labels(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items:
        slot, _, card = item.partition("=")
        if not card or slot not in SLOTS:
            raise SystemExit(f"--label expects SLOT=CARD with SLOT one of {', '.join(SLOTS)}, got {item!r}")
        out[slot] = card
    return out


def which_card(slot: str, label: str | None, read: str | None) -> str | None:
    """The card a crop is filed under: your label, unless the matcher reads a different known
    card there (then the label is wrong and the crop stays in ``raw/`` only)."""
    if label is not None:
        return None if read is not None and read != label else label
    return read


def write_sheets(out: Path) -> list[str]:
    cards = []
    for d in sorted(p for p in out.iterdir() if p.is_dir() and p.name != "raw"):
        files = sorted(d.glob("*_*.png"))
        if not files:
            continue
        thumbs = [cv2.resize(cv2.imread(str(f)), THUMB, interpolation=cv2.INTER_CUBIC) for f in files]
        rows, pad = [], np.zeros_like(thumbs[0])
        for i in range(0, len(thumbs), PER_ROW):
            chunk = thumbs[i:i + PER_ROW]
            rows.append(np.hstack(chunk + [pad] * (PER_ROW - len(chunk))))
        cv2.imwrite(str(out / f"sheet_{d.name}.png"), np.vstack(rows))
        cards.append(d.name)
    return cards


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Capture hand-card crops from a live battle (see the module docstring).")
    ap.add_argument("--out", required=True, help="folder for the crops")
    ap.add_argument("--port", type=int, default=None, help="live Studio port of the running job")
    ap.add_argument("--device", default="phone-a")
    ap.add_argument("--seconds", type=float, default=300.0, help="stop after this long")
    ap.add_argument("--every", type=float, default=0.6, help="seconds between frames")
    ap.add_argument("--min-gap", type=float, default=1.5,
                    help="seconds between two crops filed for the same slot and state")
    ap.add_argument("--label", dest="labels", action="append", default=[], help="SLOT=CARD (repeatable)")
    a = ap.parse_args(argv)

    labels = parse_labels(a.labels)
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    bundle = resolve_device_bundle(a.device, with_detector=False)
    layout = Layout.load(Path(bundle["layout"]))
    matcher = CardMatcher.load(Path(bundle["templates"]))
    screens = screen_classifier(a.device)
    adb = ["adb", "-s", str(load_profile(Path(bundle["profile"])).serial)]
    regions = [(str(i), r) for i, r in enumerate(layout.hand)] + [("next", layout.next_card)]
    source = f"live port {a.port}" if a.port else "adb screencap"
    print(f"capturing to {out} from {source}; labels {labels or 'none'}; Ctrl+C to stop", flush=True)

    filed: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    n = 0
    t_end = time.monotonic() + a.seconds
    with open(out / "index.jsonl", "a") as index:
        try:
            while time.monotonic() < t_end:
                t0 = time.monotonic()
                img = live_frame(a.port) if a.port else adb_frame(adb)
                if img is not None and screens.battle_evidence(img) >= screens.battle_min_frac:
                    for slot, roi in regions:
                        crop = roi.crop(img)
                        if crop.size == 0:
                            continue
                        r = matcher.match(crop)
                        read = r.card if r.accepted else None
                        index.write(json.dumps({"n": n, "t": round(time.time(), 2), "slot": slot, "read": read,
                                                "score": round(r.score, 3)}) + "\n")
                        cv2.imwrite(str(out / "raw" / f"{n:05d}_{slot}.png"), crop)
                        card = which_card(slot, labels.get(slot), read)
                        if card is None:
                            continue
                        state = "greyed" if is_greyed(crop) else "affordable"
                        if t0 - filed.get((slot, state), -1e9) < a.min_gap:
                            continue
                        (out / card).mkdir(exist_ok=True)
                        cv2.imwrite(str(out / card / f"{state}_{int(time.time())}_{n:05d}_{slot}.png"), crop)
                        filed[(slot, state)] = t0
                        counts[(card, state)] = counts.get((card, state), 0) + 1
                    n += 1
                time.sleep(max(0.0, a.every - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass

    cards = write_sheets(out)
    for (card, state), k in sorted(counts.items()):
        print(f"  {card:<14} {state:<10} {k}")
    print(f"{n} frames. Look at {', '.join(str(out / f'sheet_{c}.png') for c in cards) or 'the raw crops'}, then: "
          f"droid-harness promote-templates --from {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
