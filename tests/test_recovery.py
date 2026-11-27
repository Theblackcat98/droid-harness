"""Stuck-screen recovery: find buttons visually and walk any screen back to HOME.

The regression that motivated this: a ladder loss screen with a reward chest adds a
"Play Again" button and shifts "OK" right, so the calibrated `ok` centre lands in the
gap between the two buttons. Every tap misses, the screen never changes, and the farm
stalls until a human intervenes.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from droid_harness.core.telemetry.recorder import Recorder
from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.actions import CRActionMapper
from droid_harness.domains.clash_royale.menus import DEFAULT_FLOWS, MenuNavigator
from droid_harness.domains.clash_royale.perception.grid import Layout, Roi
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.perception.screens import ScreenClassifier
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.recovery import HomeRecovery, find_buttons, home_visible, walk_to_home
from droid_harness.domains.clash_royale.session import CRSession, MatchMeta
from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene

DECK = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]


def _frame_with_buttons() -> np.ndarray:
    """Dark portrait frame with a gold left button and a blue right button."""
    img = np.zeros((480, 216, 3), np.uint8)
    img[400:440, 40:100] = (0, 200, 255)  # gold, left
    img[400:440, 130:200] = (230, 120, 20)  # blue, right
    return img


def test_find_buttons_returns_rightmost_first():
    buttons = find_buttons(_frame_with_buttons())
    assert len(buttons) == 2
    assert buttons[0].center[0] > buttons[1].center[0]
    # The right button is the dismiss/positive action (OK), the left is Play Again.
    assert buttons[0].center == (165, 420)


def test_find_buttons_ignores_the_upper_screen():
    img = _frame_with_buttons()
    img[100:140, 40:200] = (230, 120, 20)
    assert len(find_buttons(img)) == 2


def test_battle_stuck_on_a_post_match_chest_aborts_and_recovers():
    """A purple chest screen reads as a full elixir bar; no-progress must end the match."""
    import time

    from droid_harness.domains.clash_royale.state import GameState

    class Perception:
        def __init__(self):
            self.elixir = SimpleNamespace(leaked=0.0)
            self.warnings = {}
            self.events = []
            self.opp = SimpleNamespace(cycle=SimpleNamespace(deck_seen=[]))

        def start_match(self, _frame):
            pass

        def process(self, _frame):
            return GameState(t_ms=0, phase="ot_x3", my_elixir=10.0, opp_elixir=(10.0, 10.0),
                             hand=[None] * 4, hand_fresh=[False] * 4)

    class Screens:
        battle_min_frac = 0.12

        def classify(self, _img):
            return "IN_BATTLE", 0

        def battle_evidence(self, _img):
            return 1.0

    class Policy:
        def reset_match(self):
            pass

    ended = []
    sess = CRSession(
        screens=Screens(), menus=MenuNavigator({"ok": Roi(0.4, 0.85, 0.6, 0.9)},
                                               {"RESULT_LOSS": ["ok"]}),
        perception=Perception(), policy=Policy(), mapper=None, recorder=None,
        meta_factory=lambda: MatchMeta("", "phone-a", "acct", "sid", "9.x", "2026-11", "ladder"),
        on_match_end=ended.append, screen_every=1, battle_no_progress_s=0.05,
    )
    img = np.zeros((60, 40, 3), np.uint8)
    assert sess.process(Frame(1, img, 0, 0, 0)).screen == "IN_BATTLE"
    assert sess.match is not None
    time.sleep(0.06)
    obs = sess.process(Frame(2, img, 33_000, 0, 0))
    assert obs.screen == "UNKNOWN" and obs.state is None
    assert len(ended) == 1 and ended[0]["status"] == "infra_abort"
    assert ended[0]["result"] == "aborted:battle_no_progress"


def test_walk_to_home_taps_a_stuck_result_then_stops_on_home():
    """The shared routine used by `recover` and the batch-end settle."""
    class Slot:
        def __init__(self):
            self.seq, self.frame = 0, None

        def wait_newer(self, _seq, timeout=0.5):
            self.seq += 1
            self.frame = SimpleNamespace(seq=self.seq, img=np.zeros((60, 40, 3), np.uint8))
            return self.frame

        def peek(self):
            return self.frame

    class Screens:
        def __init__(self):
            self.names = iter(["RESULT_LOSS", "RESULT_LOSS", "HOME"])

        def classify(self, _img):
            return next(self.names, "HOME"), 0

    class Client:
        def __init__(self):
            self.slot, self.taps, self.keys = Slot(), [], []
            self.adb = SimpleNamespace(force_stop=lambda *_a: None, launch=lambda *_a: None)

        def touch(self, action, x, y, pointer_id=0):
            self.taps.append((action, x, y))

        def keycode(self, code):
            self.keys.append(code)

    buttons = {"ok": Roi(0.4, 0.85, 0.6, 0.9)}
    layout = Layout(hand=[], next_card=Roi(0, 0, 1, 1), elixir_bar=Roi(0, 0, 1, 0.1),
                    clock=Roi(0, 0, 1, 1), towers={}, arena=Roi(0, 0, 1, 1), buttons=buttons)
    client = Client()
    assert walk_to_home(client, Screens(), layout, timeout_s=2.0, stable_s=0.0,
                        stuck_after=0.0, restart_after=10.0) is True
    # A tap is a DOWN + UP pair at the calibrated `ok` centre.
    assert len(client.taps) == 2 and client.taps[0][1:] == (20, 52)


def test_walk_to_home_treats_a_frozen_battle_as_stuck():
    """The purple chest reads as IN_BATTLE; a frozen clock+elixir must still recover."""
    class Slot:
        def __init__(self):
            self.seq, self.frame = 0, None

        def wait_newer(self, _seq, timeout=0.5):
            self.seq += 1
            self.frame = SimpleNamespace(seq=self.seq, img=np.zeros((60, 40, 3), np.uint8))
            return self.frame

        def peek(self):
            return self.frame

    class Client:
        def __init__(self):
            self.slot, self.taps, self.keys = Slot(), [], []
            self.adb = SimpleNamespace(force_stop=lambda *_a: None, launch=lambda *_a: None)

        def touch(self, action, x, y, pointer_id=0):
            self.taps.append((action, x, y))

        def keycode(self, code):
            self.keys.append(code)

    client = Client()

    class Screens:
        def classify(self, _img):
            return ("HOME", 0) if client.taps else ("IN_BATTLE", 0)

    buttons = {"ok": Roi(0.4, 0.85, 0.6, 0.9)}
    layout = Layout(hand=[], next_card=Roi(0, 0, 1, 1), elixir_bar=Roi(0, 0.9, 1, 0.95),
                    clock=Roi(0.8, 0, 1, 0.1), towers={}, arena=Roi(0, 0, 1, 1), buttons=buttons)
    assert walk_to_home(client, Screens(), layout, timeout_s=2.0, stable_s=0.0,
                        stuck_after=0.0, battle_stuck_after=-1.0) is True
    assert client.taps, "the frozen battle must be tapped through"
    assert client.keys == []


def test_home_requires_the_battle_button():
    """Trophy Road shares HOME's blue HUD; its reward track has no Battle button."""
    root = Path(__file__).resolve().parents[1]
    shots = root / "configs/domains/clash_royale/screens/phone-a"
    home = cv2.imread(str(shots / "HOME__0.png"))
    road = cv2.imread(str(shots / "TROPHY_ROAD__0.png"))
    assert home is not None and road is not None
    buttons = {"battle": Roi(0.34, 0.775, 0.67, 0.855)}
    assert home_visible(home, buttons) is True
    assert home_visible(road, buttons) is False
    assert home_visible(home, {}) is True  # no calibrated Battle button: trust the name


def test_wait_and_home_screens_are_never_stuck():
    rec = HomeRecovery({}, result_stuck_s=1.0, screen_stuck_s=1.0)
    for screen in ("LOADING", "MATCHMAKING", "IN_BATTLE", "HOME"):
        rec.observe(screen, 0.0)
        assert not rec.is_stuck(screen, 999.0), screen


def test_result_screen_arms_after_its_own_timeout():
    buttons = {"ok": Roi(0.4, 0.85, 0.6, 0.9)}
    rec = HomeRecovery(buttons, result_stuck_s=8.0, screen_stuck_s=15.0)
    rec.observe("RESULT_LOSS", 0.0)
    assert not rec.is_stuck("RESULT_LOSS", 7.9)
    assert rec.is_stuck("RESULT_LOSS", 8.0)


def test_plan_tries_calibrated_then_visual_then_back_then_restart():
    buttons = {"ok": Roi(0.0, 0.0, 0.05, 0.05)}  # deliberately far from any real button
    rec = HomeRecovery(buttons, result_stuck_s=0.0, step_interval_s=0.0, max_steps=10)
    rec.observe("RESULT_LOSS", 0.0)
    img = _frame_with_buttons()
    labels = []
    for i in range(4):
        action = rec.next_action("RESULT_LOSS", img, 0.0 + i)
        assert action is not None
        labels.append(action.label)
    assert labels[0] == "ok"
    assert labels[1] == "visual:0" and labels[2] == "visual:1"
    assert labels[3] == "back"
    # Exhausted: the next call asks BotRunner for a fresh app instead of tapping forever.
    assert rec.next_action("RESULT_LOSS", img, 10.0) is None
    assert rec.take_restart_request() is True
    assert rec.take_restart_request() is False


def test_episode_acts_fast_on_followup_screens():
    rec = HomeRecovery({}, result_stuck_s=0.0, screen_stuck_s=15.0, episode_settle_s=1.5,
                       step_interval_s=0.0)
    rec.observe("RESULT_LOSS", 0.0)
    assert rec.next_action("RESULT_LOSS", np.zeros((20, 20, 3), np.uint8), 0.0) is not None
    rec.observe("POPUP", 10.0)
    # A fresh screen during an episode is not given the full 15 s grace period.
    assert not rec.is_stuck("POPUP", 11.0)
    assert rec.is_stuck("POPUP", 11.6)


def test_home_clears_the_episode():
    rec = HomeRecovery({}, result_stuck_s=0.0, episode_settle_s=0.0)
    rec.observe("RESULT_LOSS", 0.0)
    rec.next_action("RESULT_LOSS", np.zeros((20, 20, 3), np.uint8), 0.0)
    rec.observe("HOME", 1.0)
    assert not rec.is_stuck("HOME", 99.0)
    rec.observe("POPUP", 100.0)
    assert not rec.is_stuck("POPUP", 100.1)


def _result_frame(battle: np.ndarray) -> np.ndarray:
    """Loss result screen: pink top banner, blue bottom banner, 0 vs 3 gold crowns."""
    img = np.zeros_like(battle)
    h, w = img.shape[:2]
    # A monotonic top ramp hashes far from the battle HUD, so the classifier answers
    # UNKNOWN exactly like a real ladder result screen with a different nameplate.
    img[: int(0.18 * h), :, :] = np.linspace(0, 255, w, dtype=np.uint8)[None, :, None]
    img[int(0.25 * h):int(0.35 * h), int(0.05 * w):int(0.95 * w)] = (147, 20, 220)
    img[int(0.46 * h):int(0.56 * h), int(0.05 * w):int(0.95 * w)] = (220, 120, 20)
    for x in (0.25, 0.50, 0.75):
        img[int(0.20 * h):int(0.30 * h), int((x - 0.08) * w):int((x + 0.08) * w)] = (0, 215, 255)
    return img


def test_session_recovers_a_stuck_loss_screen(cards, arena, strategy, tmp_path):
    """A result screen that never leaves emits recovery taps, then BACK, then restart."""
    _, params, sid = strategy
    scene = CRScene(cards, DECK, draw_tower_hp=False)
    battle = scene.render()
    result = _result_frame(battle)

    screens = ScreenClassifier([Roi(0, 0, 1, 0.18)], battle_roi=scene.layout.elixir_bar,
                               battle_hsv_lo=scene.layout.elixir_hsv_lo,
                               battle_hsv_hi=scene.layout.elixir_hsv_hi)
    screens.register("IN_BATTLE", battle)
    assert screens.classify(result)[0] == "UNKNOWN"  # ladder result hashes UNKNOWN

    keys: list[int] = []
    recovery = HomeRecovery(scene.layout.buttons, result_stuck_s=0.0, step_interval_s=0.0,
                            episode_settle_s=0.0, max_steps=10)
    rec = Recorder(tmp_path)
    sess = CRSession(
        screens=screens, menus=MenuNavigator(scene.layout.buttons, dict(DEFAULT_FLOWS["ladder"])),
        perception=CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                                matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene)),
        policy=CRPolicy(cards, arena, params), mapper=CRActionMapper(scene.layout, scene.tiles),
        recorder=rec, meta_factory=lambda: MatchMeta("", "phone-a", "acct", sid, "9.x", "2026-11", "ladder"),
        on_match_end=lambda s: None, screen_every=1, result_settle_s=0,
        recovery=recovery, send_key=keys.append,
    )
    for i in range(3):
        sess.process(Frame(i + 1, battle, i * 33_000, 0, 0))
    for i in range(3):
        obs = sess.process(Frame(10 + i, result, 200_000 + i * 33_000, 0, 0))
    assert obs.screen == "RESULT_LOSS"

    # Batch-end settle drives the UI directly and must not let HOME queue a battle.
    sess.suspend_menu = True
    assert sess.decide(obs, None).events == []
    sess.suspend_menu = False

    first = sess.decide(obs, None)
    assert first.events and first.events[0].tag.startswith("recover:")
    # The calibrated `ok` ROI in the sim layout is real, so it is tried first.
    assert "ok" in first.events[0].action_id

    sess.decide(obs, None)  # visual candidates (none on a black frame) are skipped
    assert keys == [4], "BACK is sent through the scrcpy key channel"
    assert sess.decide(obs, None) is not None
    assert sess.take_restart_request() is True
    rec.close()
