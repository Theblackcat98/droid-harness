"""Automatic Battle Deck card & Tower Princess upgrader (gold-only, never spends gems).

Runs safely between matches (from HOME, never mid-battle and never during paired
SPRT experiments). Walks:

    HOME -> Collection tab -> Battle Deck (8 cards + Tower Princess)
      -> tap upgradeable green-bar slot (Tower Princess prioritized first!)
      -> verify bright-green Upgrade button with yellow GOLD coin & zero red digits
      -> Card Info Upgrade button (gold-verified) -> Confirm button (gold-verified)
      -> tap through upgrade / King Level-Up celebration -> repeat -> Battle tab -> HOME

Safety invariants:
- NEVER taps a red button (such as ``Remove``).
- NEVER taps a button without a verified yellow/gold coin icon (blocks Gem prompts).
- NEVER taps a button containing red digits (insufficient gold).
- Always finishes back on HOME via ``walk_to_home`` so farming resumes cleanly.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...core.transport.scrcpy_protocol import MotionAction
from .account import AccountLevels
from .perception.cards import CardMatcher
from .perception.grid import Layout, Roi
from .recovery import KEYCODE_BACK, walk_to_home

# Default normalized ROIs calibrated from phone-a (1080x2400 / 464x1024).
DEFAULT_COLLECTION_TAB = Roi(0.180, 0.920, 0.330, 0.990)
DEFAULT_BATTLE_TAB = Roi(0.400, 0.920, 0.600, 0.990)
DEFAULT_DECKS_SUBTAB = Roi(0.090, 0.100, 0.480, 0.150)
DEFAULT_DECK_OK = Roi(0.380, 0.945, 0.620, 0.990)
DEFAULT_KING_JOURNEY = Roi(0.015, 0.045, 0.375, 0.088)
DEFAULT_COLLECTION_BADGE = Roi(0.240, 0.915, 0.330, 0.955)

_BANNER_ROI = Roi(0.350, 0.155, 0.800, 0.210)

# Preferred upgrade priority when multiple cards are ready (Tower Princess always first).
DEFAULT_PRIORITY = (
    "tower_princess",
    "musketeer",
    "knight",
    "fireball",
    "giant",
    "mini_pekka",
    "archers",
    "minions",
    "arrows",
)


@dataclass(frozen=True)
class UpgradeSlot:
    slot_id: str
    card_roi: Roi
    bar_roi: Roi


def default_upgrade_slots() -> list[UpgradeSlot]:
    """Tower Princess (bottom-right) plus the 8 Battle Deck slots (2x4 grid)."""
    cols = ((0.030, 0.235), (0.268, 0.473), (0.515, 0.720), (0.758, 0.963))
    rows = ((0.248, 0.360, 0.360, 0.390), (0.422, 0.534, 0.534, 0.564))
    slots = [
        UpgradeSlot(
            "tower_princess",
            Roi(0.758, 0.665, 0.963, 0.785),
            Roi(0.758, 0.785, 0.963, 0.818),
        )
    ]
    for r_i, (py0, py1, by0, by1) in enumerate(rows):
        for c_i, (x0, x1) in enumerate(cols):
            slots.append(UpgradeSlot(f"deck_{r_i * 4 + c_i}", Roi(x0, py0, x1, py1), Roi(x0, by0, x1, by1)))
    return slots


DEFAULT_UPGRADE_SLOTS = tuple(default_upgrade_slots())


@dataclass(frozen=True)
class UpgradeSlotHit:
    slot_id: str
    card: str | None
    tap_xy: tuple[int, int]
    green_frac: float


@dataclass(frozen=True)
class ButtonHit:
    x: int
    y: int
    w: int
    h: int
    area: int
    gold_frac: float
    red_frac: float

    @property
    def center(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)


@dataclass
class UpgradeReport:
    upgraded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    king_journey_claimed: bool = False
    reached_deck: bool = False
    returned_home: bool = False


# --- visual detectors (pure functions) ---------------------------------------

def _green_fraction(img: np.ndarray, roi: Roi) -> float:
    crop = roi.crop(img)
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([42, 140, 150], np.uint8), np.array([78, 255, 255], np.uint8))
    return float((mask > 0).mean())


def _red_fraction(img: np.ndarray, roi: Roi) -> float:
    crop = roi.crop(img)
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    m = ((hsv[:, :, 0] <= 8) | (hsv[:, :, 0] >= 170)) & (hsv[:, :, 1] >= 140) & (hsv[:, :, 2] >= 150)
    return float(m.mean())


def has_collection_badge(img: np.ndarray, badge_roi: Roi = DEFAULT_COLLECTION_BADGE) -> bool:
    """True when the red upgrade notification badge is visible on the Collection tab."""
    return _red_fraction(img, badge_roi) >= 0.05


def has_king_journey_badge(img: np.ndarray, roi: Roi = DEFAULT_KING_JOURNEY) -> bool:
    """True when the top-left King's Journey bar shows a red/green claimable badge."""
    return _red_fraction(img, roi) >= 0.04 or _green_fraction(img, roi) >= 0.12


def is_deck_screen(img: np.ndarray) -> bool:
    """True when the Battle Deck screen (blue 'Battle Deck' header banner) is visible."""
    crop = _BANNER_ROI.crop(img)
    if crop.size == 0:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    blue = ((hsv[:, :, 0] >= 95) & (hsv[:, :, 0] <= 122)
            & (hsv[:, :, 1] >= 140) & (hsv[:, :, 2] >= 140))
    return float(blue.mean()) >= 0.62


def find_upgradeable_slots(
    img: np.ndarray,
    *,
    matcher: CardMatcher | None = None,
    slots: tuple[UpgradeSlot, ...] | list[UpgradeSlot] = DEFAULT_UPGRADE_SLOTS,
    min_green: float = 0.12,
    priority: tuple[str, ...] = DEFAULT_PRIORITY,
) -> list[UpgradeSlotHit]:
    """Return Battle Deck & Tower Princess slots whose progress bar is bright green."""
    hits: list[UpgradeSlotHit] = []
    for slot in slots:
        gf = _green_fraction(img, slot.bar_roi)
        if gf < min_green:
            continue
        card: str | None = "tower_princess" if slot.slot_id == "tower_princess" else None
        if card is None and matcher is not None:
            m = matcher.match(slot.card_roi.crop(img))
            if m.card is not None and m.score >= 0.12:
                card = m.card
        hits.append(UpgradeSlotHit(slot.slot_id, card, slot.card_roi.center_px(img.shape), round(gf, 3)))

    rank = {name: i for i, name in enumerate(priority)}
    hits.sort(key=lambda h: (rank.get(h.card or h.slot_id, 999), -h.green_frac))
    return hits


def find_gold_upgrade_button(
    img: np.ndarray,
    *,
    x_min: float = 0.04,
    x_max: float = 0.96,
    y_min: float = 0.22,
    y_max: float = 0.90,
    min_gold: float = 0.015,
    max_red: float = 0.008,
) -> ButtonHit | None:
    """Find a bright-green Upgrade/Confirm button backed by a GOLD coin and zero red digits.

    Guarantees we never tap ``Remove`` (red), unaffordable buttons (red cost text), or
    "Buy Gold with Gems" prompts (green gem icon, ``gold_frac == 0``).
    """
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([42, 140, 150], np.uint8), np.array([78, 255, 255], np.uint8))
    green[: int(y_min * h), :] = 0
    green[int(y_max * h):, :] = 0
    green[:, : int(x_min * w)] = 0
    green[:, int(x_max * w):] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, w // 80), max(3, h // 200)))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)
    n, _labels, stats, _ = cv2.connectedComponentsWithStats(green, 8)
    best: ButtonHit | None = None
    for x, y, bw, bh, area in stats[1:]:
        if not (0.14 * w <= bw <= 0.45 * w):
            continue
        if not (0.030 * h <= bh <= 0.095 * h):
            continue
        if not (1.6 <= bw / max(bh, 1) <= 6.2):
            continue
        if area < 0.45 * bw * bh:
            continue  # reject hollow/irregular green grass patches inside preview videos
        crop_hsv = hsv[y:y + bh, x:x + bw]
        gold = float((cv2.inRange(crop_hsv, np.array([15, 140, 160], np.uint8),
                                  np.array([32, 255, 255], np.uint8)) > 0).mean())
        red = float((((crop_hsv[:, :, 0] <= 8) | (crop_hsv[:, :, 0] >= 170))
                     & (crop_hsv[:, :, 1] >= 140) & (crop_hsv[:, :, 2] >= 140)).mean())
        if gold < min_gold or red > max_red:
            continue
        cand = ButtonHit(int(x), int(y), int(bw), int(bh), int(area), round(gold, 4), round(red, 4))
        if best is None or cand.area > best.area:
            best = cand
    return best


def find_info_upgrade_button(img: np.ndarray) -> ButtonHit | None:
    """Green+gold Upgrade button at the bottom of the Card Info modal."""
    return find_gold_upgrade_button(img, x_min=0.30, x_max=0.70, y_min=0.75, y_max=0.88)


def find_confirm_button(img: np.ndarray) -> ButtonHit | None:
    """Green+gold Confirm button in the Upgrade confirmation popover."""
    return find_gold_upgrade_button(img, x_min=0.30, x_max=0.70, y_min=0.64, y_max=0.78)


# --- interactive state machine -----------------------------------------------

def run_auto_upgrade(
    client,
    screens,
    layout: Layout,
    *,
    matcher: CardMatcher | None = None,
    account: AccountLevels | None = None,
    max_upgrades: int = 10,
    dry_run: bool = False,
    settle_s: float = 0.85,
    log: Callable[..., None] | None = None,
) -> UpgradeReport:
    """Navigate HOME -> Collection -> upgrade ready Tower Princess & Battle Deck cards -> HOME."""
    say = log or (lambda *_a, **_k: None)
    rep = UpgradeReport()

    def latest_frame(wait_s: float = 0.5) -> np.ndarray | None:
        f = client.slot.peek()
        last_seq = f.seq if f is not None else 0
        nf = client.slot.wait_newer(last_seq, timeout=wait_s)
        f = nf or client.slot.peek()
        return f.img if f is not None else None

    def tap(x: int, y: int, label: str, pause_s: float | None = None) -> np.ndarray | None:
        say("upgrade: tap %s (%d,%d)", label, x, y)
        if not dry_run:
            client.touch(MotionAction.DOWN, x, y)
            time.sleep(0.04)
            client.touch(MotionAction.UP, x, y)
            time.sleep(settle_s if pause_s is None else pause_s)
        return latest_frame(wait_s=0.4)

    def back(label: str) -> np.ndarray | None:
        say("upgrade: key BACK (%s)", label)
        if not dry_run:
            client.keycode(KEYCODE_BACK)
            time.sleep(settle_s)
        return latest_frame(wait_s=0.4)

    # 1. Start cleanly on HOME.
    if not walk_to_home(client, screens, layout, timeout_s=45.0, stable_s=1.0,
                        stuck_after=2.5, restart_after=30.0, dry_run=dry_run, log=say):
        say("upgrade: could not reach HOME; aborting pass")
        return rep

    img = latest_frame()
    if img is None:
        return rep

    # 2. If King's Journey has a claimable chest badge at top-left, claim it first so
    #    Tower Princess cards enter our inventory before we scan the Battle Deck.
    if has_king_journey_badge(img):
        kx, ky = DEFAULT_KING_JOURNEY.center_px(img.shape)
        say("upgrade: King's Journey badge detected; claiming level-up rewards")
        img = tap(kx, ky, "king_journey", pause_s=1.2)
        for step in range(8):
            if img is None:
                break
            name, _ = screens.classify(img)
            if name == "HOME":
                break
            img = tap(img.shape[1] // 2, int(img.shape[0] * 0.65), f"king_chest:{step}", pause_s=0.7)
        rep.king_journey_claimed = True
        walk_to_home(client, screens, layout, timeout_s=20.0, stable_s=0.8,
                     stuck_after=2.0, dry_run=dry_run, log=say)
        img = latest_frame()
        if img is None:
            return rep

    # 3. Open the Collection / Battle Deck tab.
    coll_roi = layout.buttons.get("collection_tab", DEFAULT_COLLECTION_TAB)
    cx, cy = coll_roi.center_px(img.shape)
    img = tap(cx, cy, "collection_tab", pause_s=1.0)
    if img is not None and not is_deck_screen(img):
        dx, dy = DEFAULT_DECKS_SUBTAB.center_px(img.shape)
        img = tap(dx, dy, "decks_subtab", pause_s=0.9)
    if img is None or not is_deck_screen(img):
        say("upgrade: Battle Deck screen not detected; returning to HOME")
        rep.returned_home = walk_to_home(client, screens, layout, timeout_s=20.0, stable_s=0.8,
                                         stuck_after=2.0, dry_run=dry_run, log=say)
        return rep
    rep.reached_deck = True

    # 4. Upgrade loop: Tower Princess first, then Battle Deck cards.
    skipped: set[str] = set()
    for _ in range(max_upgrades):
        if img is None:
            break
        if not is_deck_screen(img):
            # Dismiss any stray edit footer / overlay back to Battle Deck.
            ok_x, ok_y = DEFAULT_DECK_OK.center_px(img.shape)
            img = tap(ok_x, ok_y, "deck_ok", pause_s=0.8)
            if img is None or not is_deck_screen(img):
                break
        hits = [h for h in find_upgradeable_slots(img, matcher=matcher) if h.slot_id not in skipped]
        if not hits:
            say("upgrade: no more upgradeable Battle Deck / Tower Princess slots")
            break
        target = hits[0]
        label = target.card or target.slot_id
        say("upgrade: checking %s (green=%.2f)", label, target.green_frac)
        img = tap(*target.tap_xy, f"slot:{label}", pause_s=0.8)
        if img is None:
            break

        # Card popup Upgrade button (or Card Info modal if tapping jumped straight to info).
        btn = find_gold_upgrade_button(img)
        if btn is None:
            say("upgrade: %s has no affordable gold Upgrade button; skipping", label)
            skipped.add(target.slot_id)
            rep.skipped.append(label)
            img = back(f"close_popup:{label}")
            if img is not None and not is_deck_screen(img):
                img = tap(img.shape[1] // 2, int(img.shape[0] * 0.12), "dismiss_popup", pause_s=0.6)
            continue

        if dry_run:
            say("upgrade: [dry-run] would upgrade %s", label)
            rep.upgraded.append(label)
            skipped.add(target.slot_id)
            continue

        img = tap(*btn.center, f"upgrade_popup:{label}", pause_s=0.9)
        if img is None:
            break

        # Check whether we're on the Card Info modal (Upgrade at bottom) or already on Confirm.
        conf = find_confirm_button(img)
        if conf is None:
            info_btn = find_info_upgrade_button(img)
            if info_btn is None:
                say("upgrade: %s info Upgrade not affordable (gold-only guard); backing out", label)
                skipped.add(target.slot_id)
                rep.skipped.append(label)
                img = back(f"close_info:{label}")
                continue
            img = tap(*info_btn.center, f"upgrade_info:{label}", pause_s=0.85)
            if img is None:
                break
            conf = find_confirm_button(img)

        if conf is None:
            say("upgrade: %s Confirm button missing or not gold-backed; backing out", label)
            skipped.add(target.slot_id)
            rep.skipped.append(label)
            img = back(f"cancel_confirm:{label}")
            if img is not None and not is_deck_screen(img):
                img = back(f"cancel_info:{label}")
            continue

        # Confirmed gold-backed upgrade!
        img = tap(*conf.center, f"confirm:{label}", pause_s=1.4)
        rep.upgraded.append(label)
        say("upgrade: upgraded %s!", label)
        if account is not None and account.record_upgrade(label):
            account.save()

        # Dismiss upgrade celebration + any King Level-Up popups until Battle Deck returns.
        for step in range(8):
            if img is None or is_deck_screen(img):
                break
            img = tap(img.shape[1] // 2, int(img.shape[0] * 0.88), f"done:{label}:{step}", pause_s=0.8)

        # If Battle Deck shows the blue 'OK' edit footer, tap OK so we see Tower Princess again.
        if img is not None and is_deck_screen(img):
            # Tapping OK only if the bottom nav bar is replaced by the 'OK' button (no Collection tab).
            if not has_collection_badge(img):
                ok_crop = DEFAULT_DECK_OK.crop(img)
                if ok_crop.size:
                    ok_hsv = cv2.cvtColor(ok_crop, cv2.COLOR_BGR2HSV)
                    ok_blue = float(((ok_hsv[:, :, 0] >= 95) & (ok_hsv[:, :, 0] <= 122)
                                     & (ok_hsv[:, :, 1] >= 120) & (ok_hsv[:, :, 2] >= 140)).mean())
                    if ok_blue >= 0.35:
                        ox, oy = DEFAULT_DECK_OK.center_px(img.shape)
                        img = tap(ox, oy, "deck_edit_ok", pause_s=0.8)

    # 5. Return to Battle tab -> HOME cleanly.
    if img is not None:
        battle_roi = layout.buttons.get("battle_tab", DEFAULT_BATTLE_TAB)
        bx, by = battle_roi.center_px(img.shape)
        tap(bx, by, "battle_tab", pause_s=0.9)
    rep.returned_home = walk_to_home(client, screens, layout, timeout_s=30.0, stable_s=1.0,
                                     stuck_after=2.5, restart_after=25.0, dry_run=dry_run, log=say)
    return rep
