"""Arena-independent battle detection, and the ladder menu flow.

The dhash references are captured in one arena with one opponent name in the top HUD.
A ladder match (another arena theme, a real nameplate) hashes far from every reference,
so the classifier used to answer UNKNOWN while the bot sat in a live match: no match
started, no policy ran, elixir sat at 10 until the result screen. These tests pin the
fallback that fixes it (elixir bar present -> in battle) and the ladder flow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from droid_harness.domains.clash_royale.menus import DEFAULT_FLOWS, MenuNavigator
from droid_harness.domains.clash_royale.perception.grid import Roi
from droid_harness.domains.clash_royale.perception.screens import ScreenClassifier

BAR = Roi(0.2, 0.962, 0.98, 0.988)
TOP = Roi(0.0, 0.0, 1.0, 0.18)


def frame(*, elixir_px: int = 0, top_value: int = 40, shape=(240, 108)):
    """Synthetic portrait frame: dark blue body, optional magenta elixir fill bottom-right."""
    img = np.full((*shape, 3), (90, 40, 20), np.uint8)
    img[: int(0.18 * shape[0]), :] = (top_value, top_value, top_value)
    if elixir_px:
        x0, y0, x1, y1 = BAR.to_px(img.shape)
        img[y0:y1, x0:x0 + int((x1 - x0) * elixir_px / 10)] = (230, 60, 190)
    return img


def classifier(refs=(), battle=True) -> ScreenClassifier:
    sc = ScreenClassifier([TOP], battle_roi=BAR if battle else None)
    for name, img in refs:
        sc.register(name, img)
    return sc


def test_unknown_battle_frame_with_elixir_reads_as_in_battle():
    # No references at all: exactly the ladder case (new arena, new opponent name).
    sc = classifier()
    assert sc.classify(frame(elixir_px=5))[0] == "IN_BATTLE"


def test_empty_bar_stays_unknown():
    sc = classifier()
    assert sc.classify(frame())[0] == "UNKNOWN"


def test_reference_match_beats_the_bar():
    home = frame(top_value=120)
    sc = classifier(refs=[("HOME", home)])
    assert sc.classify(home)[0] == "HOME"           # a registered screen wins on distance
    assert sc.battle_evidence(home) < sc.battle_min_frac


def test_result_banner_is_not_a_battle():
    # A result screen has no elixir bar, so the fallback must not fire on it.
    sc = classifier()
    assert sc.classify(frame(elixir_px=0))[0] == "UNKNOWN"


def test_elixir_bar_detected_at_low_elixir():
    sc = classifier()
    assert sc.battle_evidence(frame(elixir_px=2)) > sc.battle_min_frac


def test_battle_check_can_be_disabled():
    sc = classifier(battle=False)
    assert sc.classify(frame(elixir_px=8))[0] == "UNKNOWN"


# --- ladder flow -------------------------------------------------------------

def test_ladder_flow_queues_from_home():
    flow = DEFAULT_FLOWS["ladder"]
    assert flow["HOME"] == ["battle"]
    for screen in ("RESULT_WIN", "RESULT_LOSS", "RESULT_DRAW"):
        assert flow[screen] == ["ok"]
    # The asset-loading screen between tapping Battle and the battle itself must not be
    # tapped: a stray tap there can cancel the queue.
    assert flow["LOADING"] == []


def test_ladder_navigator_taps_the_battle_button():
    buttons = {"battle": Roi(0.36, 0.79, 0.66, 0.875), "ok": Roi(0.37, 0.815, 0.63, 0.88)}
    nav = MenuNavigator(buttons, dict(DEFAULT_FLOWS["ladder"]))
    nav.cooldown_s = 0.0
    events = nav.events_for("HOME", (2400, 1080), 0)
    assert events and events[0].tag.startswith("menu")
    x, y = events[0].x, events[0].y
    assert 0.36 * 1080 < x < 0.66 * 1080 and 0.79 * 2400 < y < 0.875 * 2400


def test_ladder_loading_screen_produces_no_taps():
    buttons = {"battle": Roi(0.36, 0.79, 0.66, 0.875)}
    nav = MenuNavigator(buttons, dict(DEFAULT_FLOWS["ladder"]))
    nav.cooldown_s = 0.0
    assert nav.events_for("LOADING", (2400, 1080), 0) == []


def test_ladder_flow_needs_a_battle_button():
    # A layout calibrated for Training Camp has no `battle` button; the flow must not
    # fall back to some other tap (events_for returns nothing when the name is missing).
    nav = MenuNavigator({"menu": Roi(0.87, 0.105, 0.96, 0.14)}, dict(DEFAULT_FLOWS["ladder"]))
    nav.cooldown_s = 0.0
    assert nav.events_for("HOME", (2400, 1080), 0) == []


def test_run_accepts_ladder_mode():
    from droid_harness.app import RunConfig

    kw = {k: Path(v) for k, v in dict(profile="p", layout="l", tiles="t", templates="tp",
                                      screens="s", strategy="st", cards="c").items()}
    assert RunConfig(**kw, mode="ladder").mode == "ladder"
