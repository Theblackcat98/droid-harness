"""Regression tests for the repo-sweep bugs (GitHub issues #7-#20), one section per issue."""

from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from droid_harness.core.telemetry.recorder import Recorder
from droid_harness.core.transport.device_lock import DeviceBusy, DeviceLock, holder
from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.account import AccountLevels, check_strategy, update_yaml_text
from droid_harness.domains.clash_royale.actions import CRActionMapper
from droid_harness.domains.clash_royale.arena import Arena
from droid_harness.domains.clash_royale.cards import CardDB
from droid_harness.domains.clash_royale.menus import MenuNavigator
from droid_harness.domains.clash_royale.perception.grid import Roi
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.perception.screens import ScreenClassifier
from droid_harness.domains.clash_royale.policy.levels import Levels, level_mult
from droid_harness.domains.clash_royale.policy.strategy import load_strategy
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.session import CRSession, MatchMeta
from droid_harness.flywheel.eval.inclusion import tally
from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner
from droid_harness.flywheel.ledger import Ledger
from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene

ROOT = Path(__file__).resolve().parents[1]
STRATS = ROOT / "configs/strategies"
STARTER = CardDB.load(ROOT / "tests/fixtures/cards_starter.json")
DECK = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]


def _session(cards, arena, strategy, tmp_path):
    """A CRSession on the synthetic scene (as in test_cr_session), plus its recorder and the summaries."""
    _, params, sid = strategy
    scene = CRScene(cards, DECK, draw_tower_hp=False)
    battle = scene.render()
    screens = ScreenClassifier([Roi(0, 0, 1, 0.5), Roi(0, 0.5, 1, 1)])
    screens.register("IN_BATTLE", battle)
    perception = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                              matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene))
    ended: list[dict] = []
    rec = Recorder(tmp_path / "rec")
    sess = CRSession(screens=screens, menus=MenuNavigator(scene.layout.buttons), perception=perception,
                     policy=CRPolicy(cards, arena, params), mapper=CRActionMapper(scene.layout, scene.tiles),
                     recorder=rec, on_match_end=ended.append, screen_every=1, result_settle_s=0,
                     meta_factory=lambda: MatchMeta("", "phone-a", "acct-1", sid, "9.x", "2026-11", "training_camp"))
    for i in range(3):
        sess.process(Frame(i + 1, scene.render(), i * 33_000, 0, 0))
    assert sess.match is not None
    return sess, rec, ended


# --- #7: one failed write must not end recording; a full disk stops the run -------------------
def test_recorder_survives_a_failed_write_and_keeps_writing(tmp_path):
    rec = Recorder(tmp_path, record_video=False)
    d = rec.start_match("m1")
    assert rec.flush(2)
    (d / "bad.jsonl").mkdir()  # opening the stream fails (EISDIR): not fatal
    rec.log("bad", {"x": 1})
    rec.log("good", {"x": 2})
    rec.end_match({"result": "win"})
    assert rec.flush(2)
    assert rec.write_errors == 1 and rec.fatal_error is None and "log" in rec.last_error
    assert (d / "good.jsonl").read_text().strip() == '{"x":2}'
    assert json.loads((d / "summary.json").read_text())["recorder_write_errors"] == 1
    rec.close()


def test_recorder_flags_a_full_disk_as_fatal(tmp_path, monkeypatch):
    rec = Recorder(tmp_path, record_video=False)
    real = rec._handle

    def handle(item):
        if item[0] == "log":
            raise OSError(errno.ENOSPC, "No space left on device")
        real(item)

    monkeypatch.setattr(rec, "_handle", handle)
    rec.start_match("m1")
    rec.log("decisions", {"x": 1})
    assert rec.flush(2)
    assert rec.fatal_error and "No space" in rec.fatal_error
    rec.close()


def test_recorder_close_never_hangs_on_a_stuck_writer(tmp_path, monkeypatch):
    rec = Recorder(tmp_path, record_video=False)
    gate = threading.Event()
    monkeypatch.setattr(rec, "_handle", lambda item: gate.wait(10))
    rec.log("x", {})
    t0 = time.monotonic()
    assert rec.flush(0.2) is False
    rec.close(timeout_s=0.2)
    assert time.monotonic() - t0 < 4.0
    gate.set()


# --- #8: video retention runs once the match is fully written ----------------------------------
def test_after_hook_sees_the_finished_match(tmp_path):
    rec = Recorder(tmp_path)
    seen: list[tuple] = []

    def after(d: Path) -> None:
        summary = json.loads((d / "summary.json").read_text())
        seen.append((d.name, summary["result"], (d / "decisions.jsonl").read_text().count("\n"),
                     rec.match_dir))

    rec.start_match("m1")
    for i in range(50):
        rec.log("decisions", {"i": i})
    rec.end_match({"result": "loss"}, after=after)
    rec.close()
    assert seen == [("m1", "loss", 50, None)]  # summary on disk, every record written, files closed


def test_session_calls_after_written_with_the_summary(cards, arena, strategy, tmp_path):
    sess, rec, ended = _session(cards, arena, strategy, tmp_path)
    calls: list[tuple] = []
    sess.after_written = lambda d, summary: calls.append((d.name, summary["status"],
                                                         (d / "summary.json").exists()))
    sess.abort_match("test")
    rec.close()
    assert calls == [(ended[0]["match_id"], "infra_abort", True)]


# --- #9: account mismatches are doctor warnings, not problems ----------------------------------
def _doctor_root(tmp_path: Path) -> Path:
    cr = tmp_path / "configs/domains/clash_royale"
    cr.mkdir(parents=True)
    for p in (ROOT / "configs/domains/clash_royale").iterdir():
        (cr / p.name).symlink_to(p)
    shutil.copy(ROOT / "tests/fixtures/cards_starter.json", cr / "cards.json")
    for sub in ("accounts", "device_profiles", "strategies"):
        (tmp_path / "configs" / sub).symlink_to(ROOT / "configs" / sub)
    return tmp_path


def test_doctor_reports_account_mismatches_as_warnings(tmp_path, monkeypatch):
    import droid_harness.app as app

    root = _doctor_root(tmp_path)
    monkeypatch.setattr(app, "check_strategy", lambda params, cards, acct: {
        "problems": ["fireball: strategy says 75, the account measured 97"], "card_levels_missing": ["knight"]})
    doc = app.run_doctor(device="phone-a", strategy="v6", root=root, live=False)
    assert not any(p.startswith("account") for p in doc["problems"])
    assert any("fireball" in w for w in doc["warnings"]) and any("knight" in w for w in doc["warnings"])
    if not doc["problems"]:  # the checkout has everything else: warnings alone don't fail it
        assert doc["ok"] is True
        assert app.run_doctor(device="phone-a", strategy="v6", root=root, live=False, strict=True)["ok"] is False


def test_systemd_unit_checks_the_champion():
    unit = (ROOT / "tools/systemd/droid-autopilot.service").read_text()
    pre = next(ln for ln in unit.splitlines() if ln.startswith("ExecStartPre=") and "doctor" in ln)
    assert "--strategy champion" in pre


# --- #10: the brain and the runner can't both end the same match --------------------------------
def test_concurrent_end_and_abort_record_one_summary(cards, arena, strategy, tmp_path):
    sess, rec, ended = _session(cards, arena, strategy, tmp_path)
    start = threading.Barrier(8)

    def abort(i: int) -> None:
        start.wait()
        if i % 2:
            sess.abort_match(f"t{i}")
        else:
            sess._end_match("win", status="completed")

    threads = [threading.Thread(target=abort, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rec.close()
    assert len(ended) == 1 and sess.match is None
    assert len([p for p in (tmp_path / "rec").iterdir() if (p / "summary.json").exists()]) == 1
    assert sess.abort_match("again") is None


# --- #11: an unread game version is unknown, not a change ---------------------------------------
def _match(arm, pid, version, result="win", **kw):
    return {"match_id": f"{arm}-{pid}", "arm": arm, "pair_id": pid, "game_version": version,
            "status": "completed", "result": result, **kw}


def test_missing_game_version_does_not_void_and_is_adopted():
    r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="", max_games=50, seed=1))
    arm, pid = r.next_assignment()
    r.record(_match(arm, pid, None))
    assert r.void_reason is None and r.cfg.game_version == ""
    arm, pid = r.next_assignment()
    r.record(_match(arm, pid, "9.1"))
    assert r.void_reason is None and r.cfg.game_version == "9.1"
    arm, pid = r.next_assignment()
    r.record(_match(arm, pid, ""))
    assert r.void_reason is None
    arm, pid = r.next_assignment()
    r.record(_match(arm, pid, "9.2"))
    assert r.void_reason and "9.2 != 9.1" in r.void_reason


def test_tally_keeps_matches_whose_version_is_unknown():
    ms = [_match("candidate", "p1", None), _match("candidate", "p2", "9.1"), _match("candidate", "p3", "9.0")]
    t = tally(ms, "candidate", "9.1")
    assert t.excluded_other_version == 1 and t.wins == 2
    assert tally(ms, "candidate", "").excluded_other_version == 0


# --- #12: King Level needs both towers to agree, and two matches in a row ----------------------
def _king(tower_max_hp: dict) -> int | None:
    return CRPerception._king_level(SimpleNamespace(tower_max_hp=tower_max_hp), "me")


def test_king_level_needs_agreeing_towers():
    assert _king({"me_left": 1512, "me_right": 1512}) == 2
    assert _king({"me_left": 1512, "me_right": 1890}) is None  # one of them is misread
    assert _king({"me_left": 1512, "me_right": 1300}) == 2  # a damaged first read maps to no level
    assert _king({}) is None


def test_sync_king_level_needs_two_matching_matches():
    acct = AccountLevels("a", 5, {})
    assert acct.sync_king_level(7) is False and acct.king_level == 5
    assert acct.sync_king_level(None) is False  # no reading keeps the streak
    assert acct.sync_king_level(7) is True and acct.king_level == 7
    assert acct.sync_king_level(9) is False
    assert acct.sync_king_level(7) is False  # back at the stored level: streak reset
    assert acct.sync_king_level(9) is False and acct.king_level == 7
    assert acct.sync_king_level(8) is False and acct.sync_king_level(8) is True and acct.king_level == 8


# --- #13: upgrading a card whose level isn't recorded doesn't invent one -----------------------
def test_upgrade_of_an_unknown_level_stays_unknown():
    acct = AccountLevels("a", 5, {"fireball": None, "arrows": 4}, spell_tower_damage={"arrows": 44})
    assert acct.record_upgrade("fireball") is False and acct.cards["fireball"] is None
    assert acct.record_upgrade("arrows") is True and acct.cards["arrows"] == 5
    want = round(44 * level_mult("arrows", 2) / level_mult("arrows", 1))
    assert acct.spell_tower_damage["arrows"] == want and want > 44
    assert acct.record_upgrade("not_in_deck") is False


# --- #14: saving the account file keeps its comments --------------------------------------------
ACCOUNT_TEXT = """\
# My account. Edit by hand after upgrades.
account: me   # in-game name
king_level: 5

# Card levels
cards:
  arrows: 4     # common
  fireball: 3
  knight:       # unknown
# measured in game
spell_tower_damage:
  arrows: 44
"""


def test_account_save_keeps_comments_and_layout(tmp_path):
    p = tmp_path / "acct.yaml"
    p.write_text(ACCOUNT_TEXT)
    acct = AccountLevels.load(p)
    acct.record_upgrade("arrows")
    acct.cards["giant"] = 3
    acct.save(p)
    text = p.read_text()
    for comment in ("# My account. Edit by hand after upgrades.", "# in-game name", "# Card levels",
                    "# common", "# unknown", "# measured in game"):
        assert comment in text
    doc = yaml.safe_load(text)
    assert doc["cards"]["arrows"] == 5 and doc["cards"]["giant"] == 3 and doc["cards"]["knight"] is None
    assert doc["spell_tower_damage"]["arrows"] == 48
    assert text.index("giant") < text.index("# measured in game")  # added inside its own section
    assert AccountLevels.load(p).levels_id == acct.levels_id


def test_update_yaml_text_falls_back_without_losing_data():
    odd = "# head\n{account: me, king_level: 5}\n"  # flow style: not understood line by line
    doc = {"account": "me", "king_level": 6, "cards": {"arrows": 4}}
    out = update_yaml_text(odd, doc)
    assert out.startswith("# head\n") and yaml.safe_load(out) == doc


# --- #15: level-aware strategies use the account's measured tower damage ------------------------
def test_level_aware_tier0_uses_measured_tower_damage():
    _, v6, _ = load_strategy(STRATS / "ladder-vision-v6.yaml")
    _, v7, _ = load_strategy(STRATS / "ladder-vision-v7.yaml")
    assert not v6.level_aware and v7.level_aware
    lv = Levels(king_level=5, cards={"fireball": 5}, tower_damage={"fireball": 123})
    assert CRPolicy(STARTER, Arena(), v7, levels=lv).tier0.spell_tower_damage("fireball", v7) == 123
    fixed = v6.tier0.spell_tower_damage["fireball"]
    assert CRPolicy(STARTER, Arena(), v6, levels=lv).tier0.spell_tower_damage("fireball", v6) == fixed
    acct = AccountLevels("a", 5, {"fireball": 5}, spell_tower_damage={"fireball": 123})
    assert acct.policy_levels().tower_damage == {"fireball": 123}


def test_check_strategy_skips_measured_damage_for_level_aware(cards):
    _, v6, _ = load_strategy(STRATS / "ladder-vision-v6.yaml")
    _, v7, _ = load_strategy(STRATS / "ladder-vision-v7.yaml")
    acct = AccountLevels("a", 5, {c: 5 for c in v7.deck}, spell_tower_damage={"fireball": 999})
    assert any("fireball" in p for p in check_strategy(v6, STARTER, acct)["problems"])
    assert not any("fireball" in p for p in check_strategy(v7, STARTER, acct)["problems"])


# --- #16: commands default to the champion; autopilot needs one ---------------------------------
def test_cli_defaults_to_the_champion(monkeypatch):
    import droid_harness.cli as cli

    seen = {}
    monkeypatch.setattr(cli, "cmd_doctor", lambda a: seen.setdefault("doctor", a.strategy))
    cli.main(["doctor"])
    assert seen["doctor"] == "champion"
    from droid_harness.app import resolve_strategy_path

    champ = (STRATS / "CHAMPION").read_text().strip()
    assert resolve_strategy_path("champion", root=ROOT).name == champ


def test_autopilot_without_a_champion_says_so(tmp_path):
    from droid_harness.flywheel.autopilot import AutopilotConfig, read_champion

    strat = tmp_path / "configs/strategies"
    strat.mkdir(parents=True)
    cfg = AutopilotConfig(champion=None, strategies_dir=strat, data_root=tmp_path / "data",
                          store=tmp_path / "ledger.jsonl", reports_dir=tmp_path / "reports", repo=tmp_path)
    with pytest.raises(FileNotFoundError, match="CHAMPION"):
        read_champion(cfg)
    shutil.copy(STRATS / "ladder-vision-v6.yaml", strat)
    (strat / "CHAMPION").write_text("ladder-vision-v6.yaml\n")
    assert read_champion(cfg).name == "ladder-vision-v6.yaml"


# --- #17: reconnect retries with backoff, then gives up with a reason --------------------------
def _fake_runner(connect_results, deadline_s=300.0):
    from droid_harness.app import BotRunner

    calls = {"connect": 0, "disconnect": 0, "adb": 0}
    results = list(connect_results)

    def connect():
        calls["connect"] += 1
        r = results.pop(0) if results else None
        if isinstance(r, BaseException):
            raise r

    fake = SimpleNamespace(cfg=SimpleNamespace(reconnect_deadline_s=deadline_s), reconnects=0,
                           RECONNECT_DELAYS_S=BotRunner.RECONNECT_DELAYS_S, _connect=connect,
                           _disconnect=lambda: calls.__setitem__("disconnect", calls["disconnect"] + 1),
                           client=SimpleNamespace(adb=SimpleNamespace(
                               run=lambda *a, **k: calls.__setitem__("adb", calls["adb"] + 1))))
    return BotRunner, fake, calls


def test_reconnect_retries_until_the_phone_is_back(monkeypatch):
    import droid_harness.app as app

    sleeps: list[float] = []
    monkeypatch.setattr(app.time, "sleep", sleeps.append)
    BotRunner, fake, calls = _fake_runner([ConnectionError("no device"), ConnectionError("no device"), None])
    BotRunner._reconnect(fake, "stall")
    assert calls["connect"] == 3 and fake.reconnects == 1 and calls["adb"] == 2  # adb reconnect on retries
    assert sleeps == [1.0, 2.0, 5.0]


def test_reconnect_gives_up_after_the_deadline(monkeypatch):
    import droid_harness.app as app

    monkeypatch.setattr(app.time, "sleep", lambda s: None)
    BotRunner, fake, _ = _fake_runner([ConnectionError("gone")] * 10, deadline_s=0.0)
    with pytest.raises(app.RunAborted, match="unreachable"):
        BotRunner._reconnect(fake, "stall")
    BotRunner, fake, calls = _fake_runner([DeviceBusy("taken")])
    with pytest.raises(DeviceBusy):
        BotRunner._reconnect(fake, "stall")
    assert calls["connect"] == 1


def test_cli_turns_a_run_abort_into_exit_2(monkeypatch, capsys):
    import droid_harness.cli as cli
    from droid_harness.app import RunAborted

    def boom(a):
        raise RunAborted("the phone stayed unreachable")

    monkeypatch.setattr(cli, "cmd_doctor", boom)
    with pytest.raises(SystemExit) as e:
        cli.main(["doctor"])
    assert e.value.code == 2 and "unreachable" in capsys.readouterr().err


# --- #18: one process per phone; experiment IDs can't collide -----------------------------------
HOLD = """
import sys, time
from droid_harness.core.transport.device_lock import DeviceLock
lk = DeviceLock(sys.argv[1]).acquire()
print("ready", flush=True)
sys.stdin.read()
"""


@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    monkeypatch.setenv("DROID_HARNESS_LOCK_DIR", str(tmp_path / "locks"))
    return tmp_path / "locks"


@pytest.fixture
def other_process(lock_env):
    procs = []

    def hold(serial: str) -> subprocess.Popen:
        p = subprocess.Popen([sys.executable, "-c", HOLD, serial], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, env={**os.environ, "DROID_HARNESS_LOCK_DIR": str(lock_env)})
        assert p.stdout.readline().strip() == "ready"
        procs.append(p)
        return p

    yield hold
    for p in procs:
        p.stdin.close()
        p.wait(5)


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_a_second_process_cannot_take_the_phone(other_process):
    p = other_process("SER1")
    with pytest.raises(DeviceBusy, match=f"PID {p.pid}"):
        DeviceLock("SER1").acquire()
    assert holder("SER1")["pid"] == p.pid
    DeviceLock("SER2").acquire().release()  # other phones are free
    p.stdin.close()
    p.wait(5)
    assert holder("SER1") is None
    DeviceLock("SER1").acquire().release()  # free once the holder exits


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_the_lock_is_reentrant_within_a_process(lock_env):
    a = DeviceLock("SER1").acquire()
    b = DeviceLock("SER1").acquire()  # a reconnect inside the same run
    a.release()
    assert holder("SER1")["pid"] == os.getpid()
    b.release()
    assert holder("SER1") is None


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_a_busy_phone_leaves_the_other_process_forward_alone(other_process):
    from droid_harness.core.config import DeviceProfile
    from droid_harness.core.transport.scrcpy_bridge import ScrcpyClient

    other_process("SER1")
    removed: list[int] = []
    adb = SimpleNamespace(forward_remove=removed.append)
    client = ScrcpyClient(DeviceProfile(name="a", serial="SER1", local_port=27183, scrcpy_server_version="2.4"),
                          adb=adb)
    with pytest.raises(DeviceBusy):
        client.start()
    client.stop()
    assert removed == []


@pytest.mark.skipif(sys.platform == "win32", reason="flock")
def test_studio_refuses_phone_jobs_while_another_process_has_the_phone(tmp_path, other_process):
    from droid_harness.ui.control import ActionError, StudioApp

    prof = tmp_path / "configs/device_profiles"
    prof.mkdir(parents=True)
    (prof / "phone-a.yaml").write_text('serial: "SER1"\nlocal_port: 27183\nscrcpy_server_version: "2.4"\n')
    p = other_process("SER1")
    app = StudioApp(tmp_path)
    assert app.phone_holder()["pid"] == p.pid
    with pytest.raises(ActionError, match="in use by"):
        app.run("recover", {})
    assert app.phone.status()["holder"]["pid"] == p.pid


def test_reserved_experiment_ids_are_unique(tmp_path):
    store = Ledger(tmp_path / "exp.jsonl")
    ids: list[str] = []
    lock = threading.Lock()

    def reserve(i: int) -> None:
        exp_id = Ledger(tmp_path / "exp.jsonl").reserve_id(title=f"t{i}", champion_id="c", candidate_id="d")
        with lock:
            ids.append(exp_id)

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == 6 and store.next_id() == "EXP-0007"
    assert {e.status for e in store.all()} == {"running"}
    assert store.void_reservation(ids[0], "never started: adb failed") is True
    e = next(e for e in store.all() if e.experiment_id == ids[0])
    assert e.status == "void" and "adb failed" in e.notes
    assert store.void_reservation(ids[0], "again") is False  # only a running, empty one


# --- #19: the warehouse row says where it was played and counts the candidate ------------------
def test_warehouse_row_has_mode_and_candidate_counts(tmp_path):
    from droid_harness.flywheel import warehouse

    r = ExperimentRunner(ExperimentConfig(experiment_id="EXP-0005", game_version="9.x", max_games=4, seed=2,
                                          mode="ladder"))
    for _ in range(4):
        arm, pid = r.next_assignment()
        r.record(_match(arm, pid, "9.x", "win" if arm == "candidate" else "loss"))
    con = warehouse.connect(tmp_path / "w.duckdb")
    r.write_warehouse(con)
    row = con.execute("select opponent_pool, wins, losses, draws from experiments "
                      "where experiment_id='EXP-0005'").fetchone()
    con.close()
    assert row == ("ladder", 2, 0, 0)


# --- #20: experiments stop at a pair boundary ---------------------------------------------------
def test_experiment_stops_only_at_a_pair_boundary():
    r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="9.x", max_games=3, seed=4))
    for _ in range(3):
        arm, pid = r.next_assignment()
        r.record(_match(arm, pid, "9.x"))
    assert r.mid_pair and not r.done  # three games in: the second pair is half played
    arm, pid = r.next_assignment()
    r.record(_match(arm, pid, "9.x"))
    assert not r.mid_pair and r.done and len(r.matches) == 4


def test_endless_aborts_still_end_the_experiment():
    r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="9.x", max_games=3, seed=4))
    n = 0
    while not r.done and n < 50:
        arm, pid = r.next_assignment()
        r.record({**_match(arm, pid, "9.x"), "status": "infra_abort", "result": "aborted:x"})
        n += 1
    assert r.done and n == 6
