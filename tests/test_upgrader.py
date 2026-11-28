"""Tests for automatic Battle Deck card & Tower Princess upgrader."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from droid_harness.domains.clash_royale.account import AccountLevels
from droid_harness.domains.clash_royale.perception.cards import CardMatcher
from droid_harness.domains.clash_royale.perception.grid import Layout, Roi
from droid_harness.domains.clash_royale.upgrader import (
    DEFAULT_UPGRADE_SLOTS,
    find_confirm_button,
    find_gold_upgrade_button,
    find_info_upgrade_button,
    find_upgradeable_slots,
    has_collection_badge,
    is_deck_screen,
    run_auto_upgrade,
)

ROOT = Path(__file__).resolve().parents[1]
CALIB = ROOT / "data/calib"
SCREENS = ROOT / "configs/domains/clash_royale/screens/phone-a"
TEMPLATES = ROOT / "configs/domains/clash_royale/templates/phone-a"


def _bgr_from_hsv(h: int, s: int, v: int) -> tuple[int, int, int]:
    px = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(px[0]), int(px[1]), int(px[2])


GREEN_BGR = _bgr_from_hsv(58, 220, 220)
GOLD_BGR = _bgr_from_hsv(24, 230, 240)
RED_BGR = _bgr_from_hsv(2, 230, 230)
BLUE_BGR = _bgr_from_hsv(108, 220, 220)


def _make_deck_frame(*, green_slots: tuple[str, ...] = ()) -> np.ndarray:
    img = np.full((1024, 464, 3), (40, 30, 20), np.uint8)
    # Battle Deck blue banner
    img[int(0.155 * 1024):int(0.210 * 1024), int(0.35 * 464):int(0.80 * 464)] = BLUE_BGR
    by_id = {s.slot_id: s for s in DEFAULT_UPGRADE_SLOTS}
    for sid in green_slots:
        x0, y0, x1, y1 = by_id[sid].bar_roi.to_px(img.shape)
        img[y0:y1, x0:x1] = GREEN_BGR
    return img


def _make_button_frame(
    x0: float, y0: float, x1: float, y1: float,
    *, bg=GREEN_BGR, with_gold: bool = True, with_red: bool = False,
) -> np.ndarray:
    img = np.full((1024, 464, 3), (30, 25, 20), np.uint8)
    px0, py0, px1, py1 = int(x0 * 464), int(y0 * 1024), int(x1 * 464), int(y1 * 1024)
    img[py0:py1, px0:px1] = bg
    if with_gold:
        gx0, gy0 = px0 + int(0.65 * (px1 - px0)), py0 + int(0.35 * (py1 - py0))
        img[gy0:gy0 + 14, gx0:gx0 + 14] = GOLD_BGR
    if with_red:
        rx0, ry0 = px0 + int(0.15 * (px1 - px0)), py0 + int(0.35 * (py1 - py0))
        img[ry0:ry0 + 14, rx0:rx0 + 20] = RED_BGR
    return img


def test_gold_upgrade_button_blocks_gems_red_digits_and_remove():
    # Valid green + gold coin + no red -> accepted
    good = _make_button_frame(0.38, 0.69, 0.62, 0.745, bg=GREEN_BGR, with_gold=True, with_red=False)
    hit = find_confirm_button(good)
    assert hit is not None and hit.gold_frac >= 0.015 and hit.red_frac == 0.0

    # Gem prompt (green button, NO gold coin) -> rejected!
    gem_only = _make_button_frame(0.38, 0.69, 0.62, 0.745, bg=GREEN_BGR, with_gold=False, with_red=False)
    assert find_confirm_button(gem_only) is None

    # Unaffordable (green button with gold coin + RED cost text) -> rejected!
    no_gold = _make_button_frame(0.38, 0.69, 0.62, 0.745, bg=GREEN_BGR, with_gold=True, with_red=True)
    assert find_confirm_button(no_gold) is None

    # Red Remove button -> rejected!
    remove = _make_button_frame(0.26, 0.43, 0.49, 0.48, bg=RED_BGR, with_gold=False, with_red=True)
    assert find_gold_upgrade_button(remove) is None


def test_tower_princess_prioritized_first_over_deck_cards():
    img = _make_deck_frame(green_slots=("deck_0", "deck_5", "tower_princess"))
    assert is_deck_screen(img)
    hits = find_upgradeable_slots(img)
    assert [h.slot_id for h in hits] == ["tower_princess", "deck_0", "deck_5"]
    assert hits[0].card == "tower_princess"


def test_run_auto_upgrade_upgrades_tower_princess_and_updates_account(tmp_path):
    home = np.full((1024, 464, 3), (60, 40, 20), np.uint8)
    deck_ready = _make_deck_frame(green_slots=("tower_princess", "deck_1"))
    deck_after_tp = _make_deck_frame(green_slots=("deck_1",))
    deck_done = _make_deck_frame(green_slots=())

    tp_popup = _make_button_frame(0.72, 0.62, 0.95, 0.675, with_gold=True)
    tp_info = _make_button_frame(0.38, 0.785, 0.62, 0.838, with_gold=True)
    tp_confirm = _make_button_frame(0.38, 0.692, 0.62, 0.744, with_gold=True)
    # Deck_1 is unaffordable (no gold button on popup) -> safely skipped via BACK!
    unaffordable_popup = _make_button_frame(0.26, 0.375, 0.49, 0.420, with_gold=False)

    stage = {"name": "HOME", "img": home}

    class _Slot:
        def __init__(self):
            self.seq = 0

        def peek(self):
            return SimpleNamespace(seq=self.seq, img=stage["img"])

        def wait_newer(self, _seq, timeout=0.2):
            self.seq += 1
            return self.peek()

    class _Client:
        def __init__(self):
            self.slot = _Slot()
            self.taps = []
            self.keys = []
            self.adb = SimpleNamespace(force_stop=lambda *_: None, launch=lambda *_: None)

        def touch(self, action, x, y, pointer_id=0):
            if action != 0:  # act on DOWN once
                return
            self.taps.append((x, y))
            st = stage["name"]
            if st == "HOME":
                stage.update(name="DECK1", img=deck_ready)
            elif st == "DECK1":
                stage.update(name="TP_POPUP", img=tp_popup)
            elif st == "TP_POPUP":
                stage.update(name="TP_INFO", img=tp_info)
            elif st == "TP_INFO":
                stage.update(name="TP_CONFIRM", img=tp_confirm)
            elif st == "TP_CONFIRM":
                stage.update(name="DECK2", img=deck_after_tp)
            elif st == "DECK2":
                stage.update(name="DECK1_UNAF", img=unaffordable_popup)
            elif st == "DECK3":
                stage.update(name="HOME", img=home)

        def keycode(self, code):
            self.keys.append(code)
            if stage["name"] == "DECK1_UNAF":
                stage.update(name="DECK3", img=deck_done)

    class _Screens:
        def classify(self, _img):
            return ("HOME", 0) if stage["name"] == "HOME" else ("UNKNOWN", 99)

    acct_path = tmp_path / "acct.yaml"
    acct = AccountLevels(account="test", king_level=2, cards={"knight": 2, "archers": 3},
                         spell_tower_damage={"fireball": 75, "arrows": 30}, path=str(acct_path))
    layout = Layout(hand=[], next_card=Roi(0, 0, 1, 1), elixir_bar=Roi(0, 0.9, 1, 0.95),
                    clock=Roi(0.8, 0, 1, 0.1), towers={}, arena=Roi(0, 0, 1, 1), buttons={})
    client = _Client()
    rep = run_auto_upgrade(client, _Screens(), layout, account=acct, settle_s=0.0)
    assert rep.reached_deck is True and rep.returned_home is True
    assert rep.upgraded == ["tower_princess"]
    assert rep.skipped == ["deck_1"]
    assert client.keys == [4]  # backed out cleanly from unaffordable card
    reloaded = AccountLevels.load(acct_path)
    assert reloaded.king_level == 3
    assert reloaded.spell_tower_damage == {"fireball": 75, "arrows": 30}


@pytest.mark.skipif(not (CALIB / "deck.png").exists() or not TEMPLATES.exists(),
                    reason="local phone-a calibration screenshots not present")
def test_upgrader_detectors_on_real_phone_a_screenshots():
    matcher = CardMatcher.load(TEMPLATES)
    deck = cv2.imread(str(CALIB / "deck.png"))
    tap = cv2.imread(str(CALIB / "archers_tap.png"))
    upg1 = cv2.imread(str(CALIB / "archers_upg1.png"))
    upg2 = cv2.imread(str(CALIB / "archers_upg2.png"))
    done1 = cv2.imread(str(CALIB / "archers_done1.png"))
    home = cv2.imread(str(SCREENS / "HOME__0.png"))

    assert has_collection_badge(home) is True
    assert is_deck_screen(deck) is True
    assert is_deck_screen(home) is False
    assert is_deck_screen(upg1) is False
    assert is_deck_screen(upg2) is False
    assert is_deck_screen(done1) is False

    hits = find_upgradeable_slots(deck, matcher=matcher)
    assert len(hits) == 1 and hits[0].card == "archers" and hits[0].green_frac >= 0.25

    assert find_gold_upgrade_button(tap) is not None
    assert find_info_upgrade_button(upg1) is not None
    assert find_confirm_button(upg2) is not None
    assert find_confirm_button(done1) is None
