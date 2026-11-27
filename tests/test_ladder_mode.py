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


def test_ladder_result_screen_classifying_unknown_still_ends_match(cards, arena, strategy, tmp_path):
    """Regression for 38-min ladder hang (0178cd440935): a ladder result screen has a
    different opponent name and no elixir bar, so ScreenClassifier returns UNKNOWN.
    CRSession must still verify banners/crowns and end the match."""
    import time

    from droid_harness.core.telemetry.recorder import Recorder
    from droid_harness.core.transport.frame_slot import Frame
    from droid_harness.domains.clash_royale.actions import CRActionMapper
    from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
    from droid_harness.domains.clash_royale.policy.utility import CRPolicy
    from droid_harness.domains.clash_royale.session import CRSession, MatchMeta
    from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene

    _, params, sid = strategy
    deck = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]
    scene = CRScene(cards, deck, draw_tower_hp=False)
    battle = scene.render()
    # Build a result image whose top 18% hashes to all-1s (monotonic ramp) so
    # ScreenClassifier returns UNKNOWN, while the middle banners are valid 3-1 win.
    result = np.zeros_like(battle)
    h, w = result.shape[:2]
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    result[: int(0.18 * h), :, :] = ramp[None, :, None]
    result[int(0.25 * h):int(0.35 * h), int(0.05 * w):int(0.95 * w)] = (147, 20, 220)
    result[int(0.46 * h):int(0.56 * h), int(0.05 * w):int(0.95 * w)] = (220, 120, 20)
    result[int(0.20 * h):int(0.30 * h), int(0.42 * w):int(0.58 * w)] = (0, 215, 255)
    for x in (0.25, 0.50, 0.75):
        result[int(0.42 * h):int(0.52 * h), int((x - 0.08) * w):int((x + 0.08) * w)] = (0, 215, 255)

    screens = ScreenClassifier([TOP], battle_roi=scene.layout.elixir_bar,
                               battle_hsv_lo=scene.layout.elixir_hsv_lo,
                               battle_hsv_hi=scene.layout.elixir_hsv_hi)
    screens.register("IN_BATTLE", battle)
    assert screens.classify(result)[0] == "UNKNOWN"

    ended = []
    rec = Recorder(tmp_path)
    sess = CRSession(
        screens=screens, menus=MenuNavigator(scene.layout.buttons, dict(DEFAULT_FLOWS["ladder"])),
        perception=CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                                matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene)),
        policy=CRPolicy(cards, arena, params), mapper=CRActionMapper(scene.layout, scene.tiles),
        recorder=rec, meta_factory=lambda: MatchMeta("", "phone-a", "acct", sid, "9.x", "2026-11", "ladder"),
        on_match_end=ended.append, screen_every=1, result_settle_s=0, unknown_battle_timeout_s=0.05,
    )
    for i in range(4):
        obs = sess.process(Frame(i + 1, battle, i * 33_000, 0, 0))
        assert obs.screen == "IN_BATTLE"
    for i in range(3):
        obs = sess.process(Frame(10 + i, result, 200_000 + i * 33_000, 0, 0))
    assert obs.screen == "RESULT_WIN" and len(ended) == 1
    assert ended[0]["result"] == "win" and (ended[0]["my_crowns"], ended[0]["opp_crowns"]) == (3, 1)

    # Second test: persistent UNKNOWN with unreadable hand eventually aborts battle so recovery runs.
    sess2 = CRSession(
        screens=screens, menus=MenuNavigator(scene.layout.buttons, dict(DEFAULT_FLOWS["ladder"])),
        perception=CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                                matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene)),
        policy=CRPolicy(cards, arena, params), mapper=CRActionMapper(scene.layout, scene.tiles),
        recorder=rec, meta_factory=lambda: MatchMeta("", "phone-a", "acct", sid, "9.x", "2026-11", "ladder"),
        on_match_end=ended.append, screen_every=1, result_settle_s=0, unknown_battle_timeout_s=0.04,
    )
    for i in range(3):
        assert sess2.process(Frame(i + 1, battle, i * 33_000, 0, 0)).screen == "IN_BATTLE"
    blank = np.zeros_like(battle)
    blank[: int(0.18 * h), :, :] = ramp[None, :, None]
    assert sess2.process(Frame(10, blank, 200_000, 0, 0)).screen == "IN_BATTLE"
    time.sleep(0.06)
    obs2 = sess2.process(Frame(11, blank, 260_000, 0, 0))
    assert obs2.screen == "UNKNOWN" and len(ended) == 2 and ended[1]["status"] == "infra_abort"
    rec.close()


def test_split_spotlight_and_transient_bar_do_not_fool_session(cards, arena, strategy, tmp_path):
    """Regression for 1s phantom ladder win (cebe436715e9):
    1. HOME/MATCHMAKING stadium spotlights (pink on right half, blue on left half) must NOT
       pass read_result_detail.
    2. A tiny (<12%) magenta transition flash in the elixir bar must NOT trigger IN_BATTLE.
    3. When self.match is None, an unrecognized ladder result screen still maps to RESULT_*.
    """
    import cv2

    from droid_harness.core.telemetry.recorder import Recorder
    from droid_harness.core.transport.frame_slot import Frame
    from droid_harness.domains.clash_royale.actions import CRActionMapper
    from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
    from droid_harness.domains.clash_royale.perception.screens import read_result_detail
    from droid_harness.domains.clash_royale.policy.utility import CRPolicy
    from droid_harness.domains.clash_royale.session import CRSession, MatchMeta
    from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene

    spotlight = np.full((240, 108, 3), (90, 40, 20), np.uint8)
    # Pink/magenta on RIGHT top half only, blue on LEFT bottom half only, plus a gold icon
    spotlight[int(0.24 * 240):int(0.36 * 240), int(0.55 * 108):] = (147, 20, 220)
    spotlight[int(0.44 * 240):int(0.56 * 240), :int(0.45 * 108)] = (220, 120, 20)
    spotlight[int(0.20 * 240):int(0.30 * 240), int(0.42 * 108):int(0.58 * 108)] = (0, 215, 255)
    assert read_result_detail(spotlight) == (None, 0, 0)

    sc = classifier()
    # 6% filled elixir bar flash must not trigger IN_BATTLE (battle_min_frac == 0.12)
    assert sc.classify(frame(elixir_px=0.6))[0] == "UNKNOWN"

    root = Path(__file__).resolve().parents[1]
    screens_dir = root / "configs/domains/clash_royale/screens/phone-a"
    for name in ("HOME__0.png", "MATCHMAKING__0.png"):
        p = screens_dir / name
        if p.exists():
            assert read_result_detail(cv2.imread(str(p))) == (None, 0, 0), name
    ladder_end = root / "data/calib/phone_ladder_end.png"
    if ladder_end.exists():
        assert read_result_detail(cv2.imread(str(ladder_end))) == ("loss", 3, 0)

    # Pre-match unrecognized result screen still resolves to RESULT_WIN so MenuNavigator clicks OK
    _, params, sid = strategy
    deck = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]
    scene = CRScene(cards, deck, draw_tower_hp=False)
    battle = scene.render()
    h, w = battle.shape[:2]
    result = np.zeros_like(battle)
    ramp = np.linspace(0, 255, w, dtype=np.uint8)
    result[: int(0.18 * h), :, :] = ramp[None, :, None]
    result[int(0.25 * h):int(0.35 * h), int(0.05 * w):int(0.95 * w)] = (147, 20, 220)
    result[int(0.46 * h):int(0.56 * h), int(0.05 * w):int(0.95 * w)] = (220, 120, 20)
    result[int(0.42 * h):int(0.52 * h), int(0.40 * w):int(0.60 * w)] = (0, 215, 255)

    screens2 = ScreenClassifier([TOP], battle_roi=scene.layout.elixir_bar,
                                battle_hsv_lo=scene.layout.elixir_hsv_lo,
                                battle_hsv_hi=scene.layout.elixir_hsv_hi)
    screens2.register("IN_BATTLE", battle)
    ended = []
    rec = Recorder(tmp_path)
    sess = CRSession(
        screens=screens2, menus=MenuNavigator(scene.layout.buttons, dict(DEFAULT_FLOWS["ladder"])),
        perception=CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                                matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene)),
        policy=CRPolicy(cards, arena, params), mapper=CRActionMapper(scene.layout, scene.tiles),
        recorder=rec, meta_factory=lambda: MatchMeta("", "phone-a", "acct", sid, "9.x", "2026-11", "ladder"),
        on_match_end=ended.append, screen_every=1, result_settle_s=0,
    )
    obs = sess.process(Frame(1, result, 0, 0, 0))
    assert obs.screen == "RESULT_WIN" and ended == []
    rec.close()


def test_ladder_navigator_dismisses_menu_and_waits_to_cycle_unknown():
    import time

    buttons = {
        "battle": Roi(0.34, 0.775, 0.67, 0.855),
        "dismiss": Roi(0.08, 0.45, 0.25, 0.55),
        "ok": Roi(0.37, 0.815, 0.63, 0.88),
        "close": Roi(0.85, 0.1, 0.95, 0.15),
    }
    nav = MenuNavigator(buttons, dict(DEFAULT_FLOWS["ladder"]), cooldown_s=0.02, unknown_settle_s=0.05)
    # MENU immediately taps dismiss (never close/menu)
    ev = nav.events_for("MENU", (2400, 1080), 0)
    assert len(ev) == 2 and "dismiss" in ev[0].action_id

    # UNKNOWN waits unknown_settle_s before tapping, then cycles through available buttons
    time.sleep(0.03)
    assert nav.events_for("UNKNOWN", (2400, 1080), 1) == []
    time.sleep(0.06)
    e1 = nav.events_for("UNKNOWN", (2400, 1080), 2)[0]
    time.sleep(0.03)
    e2 = nav.events_for("UNKNOWN", (2400, 1080), 3)[0]
    time.sleep(0.03)
    e3 = nav.events_for("UNKNOWN", (2400, 1080), 4)[0]
    assert ("dismiss" in e1.action_id and "ok" in e2.action_id and "close" in e3.action_id)


def test_transient_non_battle_screen_does_not_orphan_match(cards, arena, strategy, tmp_path):
    """Regression for orphaned dirs without summary.json (ce465cc86960/9d5cfaadd863):
    a brief HOME/LOADING flash mid-battle (or with elixir visible) stays IN_BATTLE; a sustained
    non-battle screen past left_battle_timeout_s cleanly aborts and writes summary.json."""
    import time

    from droid_harness.core.telemetry.recorder import Recorder
    from droid_harness.core.transport.frame_slot import Frame
    from droid_harness.domains.clash_royale.actions import CRActionMapper
    from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
    from droid_harness.domains.clash_royale.policy.utility import CRPolicy
    from droid_harness.domains.clash_royale.session import CRSession, MatchMeta
    from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene

    _, params, sid = strategy
    deck = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]
    scene = CRScene(cards, deck, draw_tower_hp=False)
    battle = scene.render()
    h, w = battle.shape[:2]
    home = np.full_like(battle, 180)
    home[: int(0.18 * h), :, :] = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None]
    loading_with_bar = np.full_like(battle, 120)
    loading_with_bar[: int(0.18 * h), :, :] = np.linspace(255, 0, w, dtype=np.uint8)[None, :, None]
    x0, y0, x1, y1 = scene.layout.elixir_bar.to_px(battle.shape)
    loading_with_bar[y0:y1, x0:x1] = battle[y0:y1, x0:x1]

    screens = ScreenClassifier([TOP], battle_roi=scene.layout.elixir_bar,
                               battle_hsv_lo=scene.layout.elixir_hsv_lo,
                               battle_hsv_hi=scene.layout.elixir_hsv_hi)
    screens.register("IN_BATTLE", battle)
    screens.register("HOME", home)
    screens.register("LOADING", loading_with_bar)
    assert screens.classify(home)[0] == "HOME"
    assert screens.classify(loading_with_bar)[0] == "IN_BATTLE"  # elixir present beats LOADING ref

    ended = []
    rec = Recorder(tmp_path)
    sess = CRSession(
        screens=screens, menus=MenuNavigator(scene.layout.buttons, dict(DEFAULT_FLOWS["ladder"])),
        perception=CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                                matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene)),
        policy=CRPolicy(cards, arena, params), mapper=CRActionMapper(scene.layout, scene.tiles),
        recorder=rec, meta_factory=lambda: MatchMeta("", "phone-a", "acct", sid, "9.x", "2026-11", "ladder"),
        on_match_end=ended.append, screen_every=1, result_settle_s=0,
        unknown_battle_timeout_s=0.05, left_battle_timeout_s=0.05,
    )
    for i in range(3):
        assert sess.process(Frame(i + 1, battle, i * 33_000, 0, 0)).screen == "IN_BATTLE"
    mid = sess.match.match_id
    # Even if top dhash matches LOADING, visible elixir keeps us in battle indefinitely
    assert sess.process(Frame(10, loading_with_bar, 200_000, 0, 0)).screen == "IN_BATTLE"
    # A single brief HOME frame (< left_battle_timeout_s) stays IN_BATTLE
    assert sess.process(Frame(11, home, 233_000, 0, 0)).screen == "IN_BATTLE"
    assert ended == [] and sess.match is not None and sess.match.match_id == mid
    # Sustained HOME past left_battle_timeout_s aborts cleanly (never orphans)
    time.sleep(0.06)
    obs = sess.process(Frame(12, home, 300_000, 0, 0))
    assert obs.screen == "HOME" and len(ended) == 1 and ended[0]["result"] == "aborted:left_battle:HOME"
    rec.close()
    assert (tmp_path / mid / "summary.json").exists()
