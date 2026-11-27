import json
import time

import av

from droid_harness.core.actuate import gestures
from droid_harness.core.actuate.executor import Executor
from droid_harness.core.config import DeviceProfile, check_unique_ports
from droid_harness.core.runtime.intent_bus import Intent, IntentBus
from droid_harness.core.runtime.liveness import LivenessSnapshot, assess
from droid_harness.core.telemetry.latency import StageClock, percentiles
from droid_harness.core.telemetry.loopback import loopback_rtt
from droid_harness.core.telemetry.recorder import Recorder
from droid_harness.core.transport import scrcpy_protocol as proto
from droid_harness.core.transport.scrcpy_bridge import ScrcpyClient
from droid_harness.sim.fake_scrcpy import FakeScrcpyServer, free_port

MS = 1_000_000


def test_tap_tap_orders_select_before_place():
    ev = gestures.tap_tap((10, 900), (200, 400), at_ns=1000 * MS, hold_ms=20, gap_ms=15, action_id="a1")
    assert [e.tag for e in ev] == ["select:down", "select:up", "place:down", "place:up"]
    assert ev[1].at_ns - ev[0].at_ns == 20 * MS and ev[2].at_ns - ev[1].at_ns == 15 * MS
    assert gestures.tap_tap((1, 1), (2, 2), 0, preselected=True)[0].tag == "place:down"


def test_drag_ends_with_up_at_target():
    ev = gestures.drag((0, 0), (100, 50), 0, duration_ms=100, steps=4)
    assert ev[0].action == proto.MotionAction.DOWN and ev[-1].action == proto.MotionAction.UP
    assert (ev[-2].x, ev[-2].y) == (100, 50)


def test_executor_sends_in_time_order_and_cancels():
    sent = []
    ex = Executor(lambda a, x, y, p: sent.append((a, x, y)) or 0, clock=lambda: 0)
    ex.schedule(gestures.tap(5, 5, 30 * MS, hold_ms=5, action_id="late")
                + gestures.tap(1, 1, 10 * MS, hold_ms=5, action_id="early"))
    ex.schedule(gestures.tap(9, 9, 50 * MS, action_id="x"))
    assert ex.cancel("x") == 2
    ex.run_due(25 * MS)
    assert sent == [(0, 1, 1), (1, 1, 1)]
    ex.run_due(10**12)
    assert [s[1] for s in sent] == [1, 1, 5, 5]


def test_executor_thread_timing_precision():
    times = []
    ex = Executor(lambda a, x, y, p: times.append(time.perf_counter_ns()) or times[-1]).start()
    t0 = time.perf_counter_ns()
    ex.schedule(gestures.tap(1, 1, t0 + 30 * MS, hold_ms=10))
    time.sleep(0.1)
    ex.stop()
    assert len(ex.sent) == 2
    assert max(abs(s.lateness_ms) for s in ex.sent) < 3.0


def test_liveness_static_screen_is_not_a_failure():
    quiet_menu = LivenessSnapshot(False, True, 0.5, in_battle=False, ms_since_packet=5000, screen_id="HOME",
                                  seconds_on_screen=3)
    assert assess(quiet_menu) == []
    battle_gap = LivenessSnapshot(False, True, 0.5, in_battle=True, ms_since_packet=400)
    assert [f.kind for f in assess(battle_gap)] == ["battle_frame_gap"]
    stuck = LivenessSnapshot(False, True, 0.5, in_battle=False, ms_since_packet=60000, screen_id="POPUP",
                             seconds_on_screen=45)
    assert [f.action for f in assess(stuck)] == ["recover_ui"]
    broken = LivenessSnapshot(True, False, 9, in_battle=False, ms_since_packet=1)
    assert {f.kind for f in assess(broken)} == {"session_broken", "server_exited", "ping_failed"}


def test_intent_bus_expires():
    bus = IntentBus(max_age_s=0.05)
    bus.publish(Intent(aggression=0.9))
    assert bus.latest().aggression == 0.9
    time.sleep(0.07)
    assert bus.latest() is None


def test_ports_must_be_unique():
    a = DeviceProfile(name="a", serial="1", local_port=27183, scrcpy_server_version="5.0")
    b = DeviceProfile(name="b", serial="2", local_port=27183, scrcpy_server_version="5.0")
    try:
        check_unique_ports([a, b])
        raise AssertionError
    except ValueError:
        pass


def test_stage_clock_and_percentiles():
    c = StageClock(t0_ns=0)
    c.marks = {"a": 2 * MS, "b": 5 * MS}
    assert c.durations_ms() == {"a": 2.0, "b": 3.0, "total": 5.0}
    assert percentiles([1, 2, 3, 4])["p50"] == 2.5


def test_recorder_writes_config_first_and_decodable_stream(tmp_path):
    port = free_port()
    srv = FakeScrcpyServer(port, width=96, height=160).start()
    rec = Recorder(tmp_path)
    client = ScrcpyClient(DeviceProfile(name="f", serial="f", local_port=port, scrcpy_server_version="5.0"),
                          on_packet=rec.on_packet)
    try:
        client.start(launch=False)
        client.slot.wait_newer(0, timeout=3)
        time.sleep(0.2)  # mid-session: the config packet arrived long ago
        d = rec.start_match("m1", {"strategy_id": "abc"})
        rec.log("decisions", {"seq": 1, "rule": "defend.lane_threat"})
        time.sleep(0.3)
        rec.end_match({"result": "win"})
        rec.flush()
    finally:
        client.stop()
        srv.stop()
        rec.close()
    raw = (d / "stream.h264").read_bytes()
    assert raw[:5] in (b"\x00\x00\x00\x01\x67", b"\x00\x00\x00\x01\x27")  # SPS first
    with av.open(str(d / "stream.h264"), format="h264") as container:  # decodes on its own
        n_frames = sum(1 for _ in container.decode(video=0))
    assert n_frames >= 5
    pts = [int(x) for x in (d / "stream.pts").read_text().split()]
    assert len(pts) == n_frames and pts == sorted(pts)  # one device timestamp per frame, for replay
    assert json.loads((d / "decisions.jsonl").read_text().splitlines()[0])["rule"] == "defend.lane_threat"
    assert json.loads((d / "summary.json").read_text())["result"] == "win"


def test_loopback_measures_injected_delay():
    port = free_port()
    srv = FakeScrcpyServer(port, width=160, height=320, fps=60, tap_indicator_delay_s=0.05).start()
    client = ScrcpyClient(DeviceProfile(name="f", serial="f", local_port=port, scrcpy_server_version="5.0"))
    try:
        client.start(launch=False)
        client.slot.wait_newer(0, timeout=3)
        r = loopback_rtt(client, (80, 200), samples=5, settle_s=0.1)
    finally:
        client.stop()
        srv.stop()
    assert r["timeouts"] == 0 and r["n"] == 5
    assert 50 <= r["p50"] < 50 + 60  # injected delay + at most ~3 frames at 60 fps


def test_soak_static_stream_has_no_reconnects_and_broken_session_reconnects():
    import threading

    from droid_harness.core.config import DeviceProfile
    from droid_harness.core.runtime.liveness import LivenessPolicy
    from droid_harness.core.telemetry.soak import soak
    from droid_harness.core.transport.scrcpy_bridge import ScrcpyClient
    from droid_harness.sim.fake_scrcpy import FakeScrcpyServer, free_port

    servers = []

    def make_client():
        port = free_port()
        servers.append(FakeScrcpyServer(port, width=64, height=128, fps=30, static_after_frames=5).start())
        return ScrcpyClient(DeviceProfile(name="sim", serial="sim", local_port=port, scrcpy_server_version="5.0"))

    quiet = LivenessPolicy(ping_fail_s=600)  # no adb in CI; liveness pings are not under test
    try:
        r = soak(make_client, 1.5, start_kw={"launch": False}, policy=quiet, ping_every_s=1e9)
        assert r["reconnects"] == 0 and r["frames"] >= 1 and r["verdict"] == "ok"  # static menu != broken

        servers.clear()
        threading.Timer(0.6, lambda: servers[0].stop()).start()  # kill the first session mid-soak
        r = soak(make_client, 2.5, start_kw={"launch": False}, policy=quiet, ping_every_s=1e9, backoff_s=0.2)
        assert r["reconnects"] >= 1 and r["findings"].get("session_broken", 0) >= 1 and len(servers) >= 2
    finally:
        for s in servers:
            s.stop()


def test_app_watchdog_needs_a_live_sighting_before_acting():
    from droid_harness.core.runtime.watchdog import AppWatchdog

    class FakeAdb:
        def __init__(self, seq):
            self.seq = list(seq)

        def pidof(self, package):
            return self.seq.pop(0)

    t = [0.0]
    w = AppWatchdog(FakeAdb(["", "", "123", "123", "", "", "456"]), "pkg", every_s=10, clock=lambda: t[0])
    results = []
    for _ in range(7):
        assert w.due()
        results.append(w.died())
        assert not w.due()
        t[0] += 10
    # unknown, unknown (pidof never seen working), alive, alive, DIED, still down (not re-reported), back
    assert results == [False, False, False, False, True, False, False] and w.deaths == 1


def test_replay_diff_cli_gates_on_logged_states(tmp_path, capsys):
    import json
    from pathlib import Path

    from droid_harness.cli import main
    from droid_harness.domains.clash_royale.state import GameState

    root = Path(__file__).resolve().parents[1]
    m = tmp_path / "m1"
    m.mkdir()
    states = [GameState(t_ms=t, my_elixir=e, hand=["giant", "knight", "musketeer", "fireball"],
                        opp_elixir=(4.0, 10.0), frame_size=(464, 1024)) for t, e in ((40_000, 8.0), (50_000, 4.0))]
    (m / "decisions.jsonl").write_text("".join(json.dumps({"state": s.digest()}) + "\n" for s in states))
    args = ["replay-diff", "--champion", str(root / "configs/strategies/training-camp-v2.yaml"),
            "--candidate", str(root / "configs/strategies/training-camp-blind-v1.yaml"),
            "--data", str(tmp_path), "--cards", str(root / "tests/fixtures/cards_starter.json")]
    with __import__("pytest").raises(SystemExit):
        main([*args, "--expected-pct", "1"])  # blind changes the 8-elixir state: far more than 1%
    out = json.loads(capsys.readouterr().out)
    assert out["states"] == 2 and out["changed"] >= 1 and out["gate"] == "fail"
    main([*args, "--expected-pct", "50"])
    assert json.loads(capsys.readouterr().out)["gate"] == "pass"


def test_profile_serial_env_override(monkeypatch):
    base = dict(local_port=27183, scrcpy_server_version="2.4")
    # tracked profile value wins when no override is set
    assert DeviceProfile(name="phone-a", serial="741be240", **base).serial == "741be240"
    monkeypatch.setenv("DROID_HARNESS_SERIAL", "192.168.1.190:5555")
    # every construction path (yaml load, app bundle, cli) lands here
    assert DeviceProfile(name="phone-a", serial="741be240", **base).serial == "192.168.1.190:5555"
    # the simulator must never be pointed at a real phone
    assert DeviceProfile(name="sim", serial="sim", **base).serial == "sim"
