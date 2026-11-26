#!/usr/bin/env python3
"""Aggressive Training Camp spam bot.

Uses scrcpy only for frames; input goes through `adb shell input tap` because we
already verified that path on this phone.  It reads the elixir bar from the frame,
plays every affordable card at the bridges, and stops when the battle screen is gone.

This is deliberately not the full CRPolicy: it exists to prove the capture + input
loop end-to-end and to win the first bot match before HUD/detector calibration.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from droid_harness.core.config import load_profile  # noqa: E402
from droid_harness.core.transport.scrcpy_bridge import ScrcpyClient  # noqa: E402

DEV_W, DEV_H = 1080, 2400
OUT = Path("data/bootstrap/screens")

# normalized positions on the phone (from 1080x2400 screenshots)
HAND_X = [0.307, 0.490, 0.671, 0.858]
HAND_Y = 0.908
RIGHT = (0.77, 0.525)
LEFT = (0.23, 0.525)
RIGHT_BACK = (0.77, 0.61)
CENTER = (0.50, 0.60)
MENU = (0.913, 0.122)
TRAINING = (0.583, 0.297)
CONFIRM = (0.696, 0.571)


def adb_tap(serial: str, nx: float, ny: float, hold_ms: int = 35) -> None:
    x, y = int(nx * DEV_W), int(ny * DEV_H)
    subprocess.run(["adb", "-s", serial, "shell", "input", "tap", str(x), str(y)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=4)
    time.sleep(hold_ms / 1000)


def elixir_fraction(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    roi = img[int(0.965 * h):int(0.99 * h), int(0.2 * w):int(0.98 * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([135, 80, 120]), np.array([175, 255, 255])) > 0
    cols = mask.any(axis=0)
    return float(cols.mean()) if len(cols) else 0.0


def card_saturations(img: np.ndarray) -> list[float]:
    h, w = img.shape[:2]
    out = []
    for nx in HAND_X:
        x0, x1 = int((nx - 0.085) * w), int((nx + 0.085) * w)
        y0, y1 = int(0.865 * h), int(0.952 * h)
        crop = img[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        out.append(float(hsv[:, :, 1].mean()))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="configs/device_profiles/phone-a.yaml")
    ap.add_argument("--start", action="store_true", help="navigate HOME -> Training Camp -> OK first")
    ap.add_argument("--max-seconds", type=float, default=330)
    ap.add_argument("--log", default="data/bootstrap/spam_win.log")
    args = ap.parse_args()

    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(args.log, "a", buffering=1)
    def log(*a):
        msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
        print(msg, flush=True)
        logf.write(msg + "\n")

    profile = load_profile(args.profile)
    client = ScrcpyClient(profile)
    client.start()
    log("connected", client.device_name, client.size, client.codec)

    if args.start:
        log("navigating: menu -> training camp -> confirm")
        adb_tap(profile.serial, *MENU)
        time.sleep(0.8)
        adb_tap(profile.serial, *TRAINING)
        time.sleep(1.2)
        adb_tap(profile.serial, *CONFIRM)
        time.sleep(1.0)

    f = client.slot.peek()
    if f is None:
        f = client.slot.wait_newer(0, timeout=5)
    if f is None:
        log("no frames")
        client.stop()
        return 2

    # wait for the battle HUD
    t_wait = time.time()
    battle = False
    while time.time() - t_wait < 25:
        f = client.slot.peek()
        if f is not None:
            ef = elixir_fraction(f.img)
            if ef > 0.15:
                battle = True
                log("battle detected; elixir frac", round(ef, 3))
                break
        time.sleep(0.15)
    if not battle:
        log("battle never detected")
        cv2.imwrite(str(OUT / "spam_nobattle.png"), f.img if f else np.zeros((10, 10, 3), np.uint8))
        client.stop()
        return 3

    t0 = time.time()
    last_battle = time.time()
    last_play = 0.0
    slot_i = 0
    plays = 0
    targets = [RIGHT, RIGHT, LEFT, RIGHT, CENTER, RIGHT, LEFT, RIGHT_BACK]

    while time.time() - t0 < args.max_seconds:
        f = client.slot.peek()
        if f is None:
            time.sleep(0.05)
            continue
        img = f.img
        ef = elixir_fraction(img)
        if ef < 0.15:
            if time.time() - last_battle > 2.0:
                log("battle screen gone after", round(time.time() - t0, 1), "s; plays", plays)
                cv2.imwrite(str(OUT / "spam_result.png"), img)
                break
            time.sleep(0.08)
            continue
        last_battle = time.time()

        elapsed = time.time() - t0
        double = elapsed > 120
        threshold = 0.33 if double else 0.41  # ~3 elixir / ~4 elixir
        # don't hammer faster than the card-selection round trip
        if time.time() - last_play < (0.28 if double else 0.45):
            time.sleep(0.03)
            continue
        if ef < threshold:
            time.sleep(0.03)
            continue

        sats = card_saturations(img)
        affordable = [i for i, s in enumerate(sats) if s > 55.0]
        if not affordable:
            time.sleep(0.05)
            continue
        # rotate through affordable slots so all cards get used
        slot = next((i for i in range(4) if (slot_i + i) % 4 in affordable), affordable[0])
        slot_i = (slot + 1) % 4
        target = targets[plays % len(targets)]
        adb_tap(profile.serial, HAND_X[slot], HAND_Y)
        time.sleep(0.10)
        adb_tap(profile.serial, *target)
        plays += 1
        last_play = time.time()
        if plays % 5 == 1:
            log(f"play {plays}: slot {slot} elixir {ef*10:.1f} sat {[round(s) for s in sats]} -> {target}")

    else:
        log("max seconds reached; stopping")
        f = client.slot.peek()
        if f is not None:
            cv2.imwrite(str(OUT / "spam_timeout.png"), f.img)

    # let the result screen settle so we can read it from the saved frame
    time.sleep(1.5)
    f = client.slot.peek()
    if f is not None:
        cv2.imwrite(str(OUT / "spam_final.png"), f.img)
    log("done, plays", plays)
    client.stop()
    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
