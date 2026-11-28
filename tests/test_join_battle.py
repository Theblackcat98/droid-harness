"""Start in a battle: the Studio joins the battle already on the phone.

The player queued a battle and presses Start. It must be played with the selected strategy,
with no mode to choose, and nothing else may happen to the phone before the battle is
recognised: no menu flow, upgrade pass, recovery tap, BACK or app restart.
"""

from __future__ import annotations

import shutil
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

import cv2
import pytest

from droid_harness.app import JoinWait, RunAborted
from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.actions import CRActionMapper
from droid_harness.domains.clash_royale.menus import MenuNavigator
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.perception.screens import ScreenClassifier
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.session import CRSession, MatchMeta
from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene
from droid_harness.ui.control import StudioApp
from droid_harness.ui.jobs import JOB_ENV, JobManager
from droid_harness.ui.phone import PhoneControl
from droid_harness.ui.settings import Settings

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests/fixtures"
DECK = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]


class _Phone:
    """Just enough of a session for the join logic: what runs, what screen, and recovery's resets."""

    def __init__(self) -> None:
        self.match = None
        self.screen = "UNKNOWN"
        self.joining = False
        self.had_battle_hud = False
        self.resets = 0
        self.recovery = SimpleNamespace(reset=lambda: setattr(self, "resets", self.resets + 1))


# --- Ctrl+C must reach a Studio job ---------------------------------------------------------
def test_ctrl_c_is_rearmed_in_a_process_that_inherited_it_as_ignored():
    from droid_harness.cli import arm_ctrl_c

    before = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # what a non-interactive shell leaves on `&`
        arm_ctrl_c()
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler

        def mine(*_a):
            pass

        signal.signal(signal.SIGINT, mine)  # a handler of our own is left alone
        arm_ctrl_c()
        assert signal.getsignal(signal.SIGINT) is mine
    finally:
        signal.signal(signal.SIGINT, before)


def test_studio_jobs_are_marked_so_the_cli_rearms_ctrl_c(tmp_path):
    env = JobManager(tmp_path).env()
    assert env[JOB_ENV] == "1" and env["PYTHONUNBUFFERED"] == "1"


# --- the run loop: wait, play, stop -----------------------------------------------------------
def test_join_wait_keeps_the_menus_untouched_until_a_real_battle_runs():
    phone = _Phone()
    join = JoinWait(wait_s=60.0, strategy="v13")
    assert join.tick(phone, []) == "waiting" and phone.joining  # nothing on the phone yet

    phone.screen = "IN_BATTLE"  # the brain thread is starting the match this instant
    assert join.tick(phone, []) == "waiting"

    phone.match = object()  # the battle is running
    assert join.tick(phone, []) == "playing" and phone.resets == 1

    phone.match, phone.screen, phone.had_battle_hud = None, "RESULT_WIN", True  # it ended
    assert join.tick(phone, [{"status": "completed"}]) == "over" and not phone.joining
    assert join.tick(phone, []) == "over"  # stays over: the rest of the run is the normal flow


def test_join_wait_resumes_after_a_chest_screen_that_flashed_as_a_battle():
    phone = _Phone()
    join = JoinWait(wait_s=60.0, strategy="v13")
    join.tick(phone, [])
    phone.match = object()
    assert join.tick(phone, []) == "playing"
    # No clock, hand or elixir ever appeared: the session aborted it. That was not the battle.
    phone.match, phone.screen = None, "UNKNOWN"
    assert join.tick(phone, [{"status": "infra_abort", "result": "aborted:battle_no_progress"}]) == "waiting"
    assert phone.joining  # the menu flow stays off


def test_join_wait_gives_up_cleanly_when_no_battle_appears():
    phone = _Phone()
    join = JoinWait(wait_s=0.0, strategy="v13")
    phone.joining = True  # as run() sets it before the loop
    with pytest.raises(RunAborted, match="no battle appeared on the phone"):
        join.tick(phone, [])
    assert phone.joining  # and still nothing was tapped


# --- the session: not one tap while joining, then the battle is marked as joined ----------------
def _joining_session(cards, arena, strategy, sid):
    from droid_harness.domains.clash_royale.perception.grid import Roi

    _, params, _ = strategy
    scene = CRScene(cards, DECK, draw_tower_hp=False)
    battle = scene.render()
    home = battle.copy()
    home[:] = (60, 120, 200)  # a flat, unmistakable menu screen
    screens = ScreenClassifier([Roi(0, 0, 1, 0.5), Roi(0, 0.5, 1, 1)])
    screens.register("IN_BATTLE", battle)
    screens.register("HOME", home)
    flow = {"HOME": [next(iter(scene.layout.buttons))]}  # the menu flow taps something on HOME
    return (
        scene,
        battle,
        home,
        CRSession(
            screens=screens,
            menus=MenuNavigator(scene.layout.buttons, flow=flow),
            perception=CRPerception(
                layout=scene.layout,
                tiles=scene.tiles,
                cards=cards,
                arena=arena,
                matcher=build_matcher_from_scene(scene),
                detector=SceneDetector(scene),
            ),
            policy=CRPolicy(cards, arena, params),
            mapper=CRActionMapper(scene.layout, scene.tiles),
            meta_factory=lambda: MatchMeta("", "phone-a", "acct-1", sid, "9.x", "2026-11", "ladder"),
            screen_every=1,
        ),
    )


def test_session_taps_nothing_while_joining_then_plays_the_battle(cards, arena, strategy):
    _, _, sid = strategy
    scene, battle, home, sess = _joining_session(cards, arena, strategy, sid)
    # Control: with the menu flow on, a menu screen gets its button tapped.
    obs = sess.process(Frame(1, home, 0, 0, 0))
    assert obs.screen == "HOME" and sess.decide(obs, None).events

    sess = _joining_session(cards, arena, strategy, sid)[3]
    sess.joining = True
    obs = sess.process(Frame(1, home, 0, 0, 0))
    assert obs.screen == "HOME"
    sess.recovery.force()  # even a liveness finding that armed the HOME walk changes nothing
    assert sess.decide(obs, None).events == []
    assert sess.match is None

    obs = sess.process(Frame(2, battle, 33_000, 0, 0))
    assert obs.screen == "IN_BATTLE" and obs.state is not None
    assert sess.match is not None and sess.match.joined and not sess.joining
    assert sess.decide(obs, None) is not None


# --- the Studio: Start joins a battle, and is a preview anywhere else ---------------------------
class FakeAdb:
    """Records adb command lines; ``screencap`` returns the given image as a PNG (as adb does),
    the rest answers like a phone."""

    def __init__(self, image_file: Path) -> None:
        self.calls: list[list[str]] = []
        ok, buf = cv2.imencode(".png", cv2.imread(str(image_file)))
        assert ok
        self.png = buf.tobytes()

    def __call__(self, cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        args = cmd[3:] if len(cmd) > 2 and cmd[1] == "-s" else cmd[1:]
        out = b""
        if args[:2] == ["exec-out", "screencap"]:
            out = self.png
        elif args[:3] == ["shell", "wm", "size"]:
            out = b"Physical size: 1080x2400\n"
        elif args[:1] == ["devices"]:
            out = b"List of devices attached\nSER1  device usb:1-1 product:x model:Pixel_7 device:y\n\n"
        return subprocess.CompletedProcess(cmd, 0, out, b"")


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A minimal checkout: two strategies, a CHAMPION pointer and a device profile."""
    strat = tmp_path / "configs/strategies"
    (strat / "counters").mkdir(parents=True)
    for v in ("v4", "v6"):
        shutil.copy(ROOT / f"configs/strategies/ladder-vision-{v}.yaml", strat)
        shutil.copy(ROOT / f"configs/strategies/counters/ladder-vision-{v}.csv", strat / "counters")
    (strat / "CHAMPION").write_text("ladder-vision-v4.yaml\n")
    prof = tmp_path / "configs/device_profiles"
    prof.mkdir(parents=True)
    (prof / "phone-a.yaml").write_text(
        'serial: "SER1"\nlocal_port: 27183\nscrcpy_server_version: "2.4"\nmax_size: 1024\ncrop: null\n'
    )
    return tmp_path


def _calibrated_root(tmp_path: Path) -> Path:
    """The test checkout, plus phone-a's screen references and button regions."""
    cr = tmp_path / "configs/domains/clash_royale"
    (cr / "rois").mkdir(parents=True)
    shutil.copy(ROOT / "configs/domains/clash_royale/rois/phone-a.yaml", cr / "rois/phone-a.yaml")
    shutil.copytree(ROOT / "configs/domains/clash_royale/screens/phone-a", cr / "screens/phone-a")
    return tmp_path


def _app(tmp_path: Path, image: Path) -> StudioApp:
    st = Settings.load(tmp_path / ".studio/settings.json")
    return StudioApp(tmp_path, phone=PhoneControl(st, tmp_path, runner=FakeAdb(image)))


def test_start_plays_the_battle_on_the_phone_and_previews_any_other_screen(root):
    _calibrated_root(root)
    battle, vs = FIX / "battle_phone_a.jpg", FIX / "vs_screen_phone_a.jpg"
    app = _app(root, battle)
    app.settings.update({"auto_upgrade": True, "play_matches": 10, "play_mode": "training_camp"})
    started = app.build("start", {})
    a = started["args"]
    assert started["kind"] == "play" and a[0] == "play"
    assert "--join-battle" in a and a[a.index("--matches") + 1] == "1"
    assert a[a.index("--strategy") + 1] == "configs/strategies/ladder-vision-v4.yaml"  # the selected strategy
    assert "--auto-upgrade" not in a  # never navigate menus around a battle in progress
    assert started["foreground"] and started["live"]

    app = _app(root, vs)  # matchmaking / VS: not a battle yet
    prev = app.build("start", {})
    assert prev["kind"] == "preview" and prev["args"][0] == "watch"

    assert _app(root, FIX / "home_phone_a.png").build("start", {})["kind"] == "preview"


def test_play_joins_a_battle_unless_told_not_to(root):
    _calibrated_root(root)
    battle = FIX / "battle_phone_a.jpg"
    app = _app(root, battle)
    play = app.build("play", {"matches": 5, "mode": "ladder"})
    assert "--join-battle" in play["args"] and play["args"][play["args"].index("--matches") + 1] == "1"
    assert "battle on the phone" in play["title"]

    adb = FakeAdb(battle)
    st = Settings.load(root / ".studio/settings.json")
    app = StudioApp(root, phone=PhoneControl(st, root, runner=adb))
    batch = app.build("play", {"matches": 5, "mode": "ladder", "join": False})  # a deliberate batch
    assert "--join-battle" not in batch["args"] and batch["args"][batch["args"].index("--matches") + 1] == "5"
    assert not any("screencap" in " ".join(c) for c in adb.calls)  # the phone wasn't even asked


def test_start_without_calibration_never_guesses_a_battle(root):
    app = StudioApp(
        root,
        phone=PhoneControl(
            Settings.load(root / ".studio/settings.json"), root, runner=FakeAdb(FIX / "battle_phone_a.jpg")
        ),
    )
    assert app.build("start", {})["kind"] == "preview"
