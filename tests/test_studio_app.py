"""The Studio app: settings, phone input, jobs, actions, the guided flow and the server."""

from __future__ import annotations

import http.client
import json
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import pytest

from droid_harness.ui.control import ActionError, StudioApp
from droid_harness.ui.jobs import Job, JobError, JobManager, parse_json_output
from droid_harness.ui.phone import (
    PhoneControl,
    PhoneError,
    PhoneLocked,
    frame_to_device,
    parse_crop,
    parse_wm_size,
)
from droid_harness.ui.refine import FlowError
from droid_harness.ui.server import Studio, StudioServer
from droid_harness.ui.settings import Settings, find_repo_root

ROOT = Path(__file__).resolve().parents[1]


def _png(w: int = 8, h: int = 16) -> bytes:
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 90, np.uint8))
    assert ok
    return buf.tobytes()


@pytest.fixture
def root(tmp_path: Path) -> Path:
    """A minimal checkout: two strategies, a CHAMPION pointer, a device profile, two matches."""
    strat = tmp_path / "configs/strategies"
    (strat / "counters").mkdir(parents=True)
    for v in ("v4", "v6"):
        shutil.copy(ROOT / f"configs/strategies/ladder-vision-{v}.yaml", strat)
        shutil.copy(ROOT / f"configs/strategies/counters/ladder-vision-{v}.csv", strat / "counters")
    (strat / "CHAMPION").write_text("ladder-vision-v4.yaml\n")
    prof = tmp_path / "configs/device_profiles"
    prof.mkdir(parents=True)
    (prof / "phone-a.yaml").write_text('serial: "SER1"\nlocal_port: 27183\nscrcpy_server_version: "2.4"\n'
                                       'max_size: 1024\ncrop: null\n')
    for i, res in enumerate(("win", "loss")):
        d = tmp_path / f"data/matches/m{i}"
        d.mkdir(parents=True)
        (d / "decisions.jsonl").write_text("{}\n")
        (d / "summary.json").write_text(json.dumps({"result": res, "strategy_id": "abc"}))
    return tmp_path


class FakeAdb:
    """Records adb command lines and answers like a 1080x2400 phone."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], timeout: float) -> subprocess.CompletedProcess:
        self.calls.append(cmd)
        args = cmd[3:] if len(cmd) > 2 and cmd[1] == "-s" else cmd[1:]
        out = b""
        if args[:3] == ["shell", "wm", "size"]:
            out = b"Physical size: 1080x2400\n"
        elif args[:1] == ["devices"]:
            out = b"List of devices attached\nSER1  device usb:1-1 product:x model:Pixel_7 device:y\n\n"
        elif args[:2] == ["exec-out", "screencap"]:
            out = _png(1080, 2400)
        return subprocess.CompletedProcess(cmd, 0, out, b"")


# --- settings ------------------------------------------------------------------------------
def test_settings_defaults_coercion_and_persistence(tmp_path):
    p = tmp_path / ".studio/settings.json"
    p.parent.mkdir()
    p.write_text(json.dumps({"play_matches": "12", "test": "nonsense", "port": 99, "unknown": 1, "prune_win_video": "no"}))
    s = Settings.load(p)
    assert s["play_matches"] == 12 and s["test"] == "dense" and s["port"] == 8765 and "unknown" not in s.values
    assert s["prune_win_video"] is False
    errors = s.update({"max_games": 0, "serial": " 10.0.0.5:5555 ", "nope": 1})
    assert set(errors) == {"max_games", "nope"} and s["serial"] == "10.0.0.5:5555"
    assert Settings.load(p)["serial"] == "10.0.0.5:5555"
    assert s.job_env()["DROID_HARNESS_SERIAL"] == "10.0.0.5:5555"
    s.reset()
    assert Settings.load(p)["serial"] == "" and {f["key"] for f in Settings.schema()} >= {"screenshot_dir", "champion"}


def test_find_repo_root(root):
    assert find_repo_root(root / "data/matches") == root
    assert (find_repo_root(Path("/")) / "configs/strategies").is_dir()  # falls back to this checkout


# --- phone ---------------------------------------------------------------------------------
def test_frame_to_device_mapping():
    assert parse_wm_size("Physical size: 1080x2400\nOverride size: 720x1600") == (720, 1600)
    assert parse_wm_size("Physical size: 1080x2400") == (1080, 2400)
    assert parse_crop("1080:1800:0:300") == (1080, 1800, 0, 300) and parse_crop("bad") is None
    assert frame_to_device(232, 512, 464, 1024, (1080, 2400)) == (540, 1200)
    assert frame_to_device(464, 1024, 464, 1024, (1080, 2400)) == (1079, 2399)  # clamped inside the screen
    assert frame_to_device(0, 0, 464, 768, (1080, 2400), (1080, 1800, 0, 300)) == (0, 300)
    assert frame_to_device(512, 232, 1024, 464, (1080, 2400)) == (1200, 540)  # rotated to landscape


def test_phone_lock_tap_swipe_key_and_screencap(root):
    st = Settings.load(root / ".studio/settings.json")
    adb = FakeAdb()
    ph = PhoneControl(st, root, runner=adb)
    with pytest.raises(PhoneLocked):
        ph.tap(10, 10, 464, 1024)
    assert ph.set_locked(False) is False
    assert ph.tap(232, 512, 464, 1024)["device_xy"] == [540, 1200]
    assert adb.calls[-1][1:] == ["-s", "SER1", "shell", "input", "tap", "540", "1200"]
    ph.swipe(0, 0, 464, 1024, 464, 1024, duration_ms=10_000)
    assert adb.calls[-1][-5:] == ["0", "0", "1079", "2399", "3000"]
    ph.key("back")
    assert adb.calls[-1][-2:] == ["keyevent", "4"]
    with pytest.raises(ValueError):
        ph.key("power")
    assert ph.screencap().startswith(b"\x89PNG")
    assert ph.devices() == [{"serial": "SER1", "state": "device", "model": "Pixel_7"}]
    st.update({"serial": "10.0.0.5:5555"})
    assert ph.serial() == "10.0.0.5:5555"
    ph.set_locked(True)
    with pytest.raises(PhoneLocked):
        ph.key("home")


def test_phone_errors_are_readable(root):
    st = Settings.load(root / ".studio/settings.json")

    def missing(cmd, timeout):
        raise FileNotFoundError(cmd[0])

    ph = PhoneControl(st, root, runner=missing)
    ph.set_locked(False)
    with pytest.raises(PhoneError, match="adb isn't installed"):
        ph.tap(1, 1, 10, 10)
    ph = PhoneControl(st, root, runner=lambda c, t: subprocess.CompletedProcess(c, 0, b"not a png", b""))
    with pytest.raises(PhoneError):
        ph.screencap()


# --- jobs ----------------------------------------------------------------------------------
def test_parse_json_output():
    assert parse_json_output('{"a": 1}') == {"a": 1}
    assert parse_json_output('log line\n{\n "a": [1,\n  2]\n}\n') == {"a": [1, 2]}
    assert parse_json_output('noise\n{"x": 1}\nmore noise') == {"x": 1}
    assert parse_json_output("no json here") is None and parse_json_output("") is None


def _wait(job: Job, timeout: float = 20.0) -> Job:
    t0 = time.time()
    while job.active and time.time() - t0 < timeout:
        time.sleep(0.05)
    return job


def test_jobs_run_capture_result_and_fail(tmp_path):
    jm = JobManager(tmp_path, base_cmd=[sys.executable, "-c"])
    done = []
    j = jm.start("t", "Test", ["import sys; print('hello'); print('warn', file=sys.stderr); print('{\"ok\": 1}')"],
                 on_done=done.append)
    _wait(j)
    assert j.status == "done" and j.result == {"ok": 1} and done == [j]
    lines = [r["t"] for r in j.log_since(0)["lines"]]
    assert "hello" in lines and "warn" in lines and j.log_path.read_text().count("\nhello\n") == 1
    assert j.log_since(j.seq)["lines"] == []
    bad = _wait(jm.start("t", "Bad", ["import sys; print('boom', file=sys.stderr); sys.exit(3)"]))
    assert bad.status == "failed" and bad.returncode == 3 and bad.error == "boom"
    assert [x["id"] for x in jm.list()][:2] == [bad.id, j.id]


def test_jobs_foreground_is_exclusive_and_stops(tmp_path):
    jm = JobManager(tmp_path, base_cmd=[sys.executable, "-c"])
    a = jm.start("p", "Phone job", ["import time; time.sleep(30)"], foreground=True)
    with pytest.raises(JobError, match="Phone job is still running"):
        jm.start("p", "Another", ["pass"], foreground=True)
    offline = _wait(jm.start("o", "Offline", ["print(1)"]))  # offline jobs may run alongside
    assert offline.status == "done" and jm.foreground() is a
    jm.stop(a.id, grace_s=5)
    assert _wait(a).status == "stopped" and jm.foreground() is None
    stop_file = tmp_path / "STOP"
    g = jm.start("auto", "Graceful", [f"import time, pathlib\nwhile not pathlib.Path({str(stop_file)!r}).exists(): "
                                      "time.sleep(0.05)"], foreground=True, stop_file=stop_file)
    assert jm.stop(g.id).status == "stopping" and stop_file.exists()
    assert _wait(g).status == "stopped" and g.returncode == 0


def test_live_jobs_get_a_private_ui_port(tmp_path):
    jm = JobManager(tmp_path, base_cmd=[sys.executable, "-c"])
    j = jm.start("demo", "Live", ["import sys; print(sys.argv[1:])"], foreground=True, live=True, raw=False)
    _wait(j)
    assert j.argv[-5:] == ["--ui", "--ui-host", "127.0.0.1", "--ui-port", str(j.live_port)]
    w = jm.start("preview", "Watch", ["pass"], foreground=True, live=True, ui_flag=False)
    _wait(w)
    assert "--ui" not in w.argv and "--ui-port" in w.argv


# --- actions -------------------------------------------------------------------------------
def test_actions_build_the_commands_you_would_type(root):
    app = StudioApp(root, phone=PhoneControl(Settings.load(root / ".studio/settings.json"), root, runner=FakeAdb()))
    app.settings.update({"cards": "tests/fixtures/cards_starter.json", "prune_win_video": True})
    play = app.build("play", {"matches": 3, "mode": "training_camp"})
    assert play["args"][:2] == ["play", "--device"] and play["foreground"] and play["live"]
    a = play["args"]
    assert a[a.index("--strategy") + 1] == "configs/strategies/ladder-vision-v4.yaml"  # 'champion' resolved
    assert a[a.index("--matches") + 1] == "3" and "--prune-win-video" in a and "--cards" in a
    prev = app.build("preview", {})
    assert prev["args"][0] == "watch" and prev["ui_flag"] is False
    app.settings.update({"preview_detector": False})
    assert "--no-detector" in app.build("preview", {})["args"]
    scan = app.build("scan", {"data": "data/matches"})["args"]
    assert scan[0] == "tune-targeted" and "--scan" in scan and not app.build("scan", {}).get("foreground")
    exp = app.build("experiment", {"candidate": "v6", "title": "t", "max_games": 10})["args"]
    assert exp[exp.index("--candidate") + 1] == "configs/strategies/ladder-vision-v6.yaml"
    assert exp[exp.index("--max-games") + 1] == "10" and exp[exp.index("--test") + 1] == "dense"
    rd = app.build("replay_diff", {"candidate": "v6", "expected_pct": 2.5})["args"]
    assert rd[0] == "replay-diff" and rd[rd.index("--expected-pct") + 1] == "2.5"
    with pytest.raises(ActionError):
        app.build("propose", {})  # needs an output file
    with pytest.raises(ActionError):
        app.build("format_disk", {})
    with pytest.raises(ActionError):
        app.resolve_strategy("v99")


def test_strategies_and_datasets(root):
    app = StudioApp(root)
    rows = app.strategies()
    assert [r["file"] for r in rows] == ["ladder-vision-v4.yaml", "ladder-vision-v6.yaml"]
    assert rows[0]["champion"] and rows[0]["id"] and rows[0]["name"] == "ladder-vision-v4"
    ds = app.datasets()
    assert ds == [{**ds[0], "path": "data/matches", "matches": 2, "win": 1, "loss": 1, "strategies": {"abc": 2}}]


class _FakeChild(BaseHTTPRequestHandler):
    """Stands in for a job's own Studio: a frame, a snapshot and a short event stream."""

    def do_GET(self):  # noqa: N802
        if self.path == "/api/live/frame.png":
            body, ctype = _png(), "image/png"
        elif self.path == "/api/live/snapshot":
            body, ctype = b'{"screen": "IN_BATTLE"}', "application/json"
        elif self.path == "/api/live/config":
            body, ctype = b'{"rois": {}}', "application/json"
        elif self.path == "/api/live/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(2):
                self.wfile.write(f"data: {{\"n\": {i}}}\n\n".encode())
                self.wfile.flush()
            return
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Frame-Seq", "7")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def fake_live(root):
    """An app whose foreground job is 'streaming' from a fake child Studio."""
    child = ThreadingHTTPServer(("127.0.0.1", 0), _FakeChild)
    threading.Thread(target=child.serve_forever, daemon=True).start()
    app = StudioApp(root, phone=PhoneControl(Settings.load(root / ".studio/settings.json"), root, runner=FakeAdb()))
    job = Job(id="live1", kind="preview", title="Preview", argv=[], foreground=True, live=True, status="running",
              live_port=child.server_address[1])
    app.jobs.jobs[job.id] = job
    yield app
    child.shutdown()


def test_screenshots_from_the_stream_the_page_and_the_phone(fake_live, root):
    app = fake_live
    shot = app.screenshot(overlay_png=_png(4, 4))
    assert shot["source"] == "stream" and (shot["w"], shot["h"]) == (8, 16)
    assert shot["files"][0].endswith("-in-battle.png") and shot["files"][1].endswith("-in-battle-overlay.png")
    assert all((root / f).read_bytes().startswith(b"\x89PNG") for f in shot["files"])
    rep = app.screenshot(frame_png=_png(3, 5), label="replay 42/7s")
    assert rep["source"] == "replay" and rep["files"][0].endswith("-replay-42-7s.png")
    app.jobs.jobs.clear()
    ph = app.screenshot()
    assert ph["source"] == "phone" and (ph["w"], ph["h"]) == (1080, 2400)
    with pytest.raises(ActionError):
        app.screenshot(frame_png=b"GIF89a")
    app.settings.update({"screenshot_dir": "elsewhere/shots"})
    assert app.screenshot(frame_png=_png())["files"][0].startswith("elsewhere/shots/")
    assert len(app.recent_shots) == 4


# --- the guided flow -----------------------------------------------------------------------
class ScriptedApp(StudioApp):
    """Jobs finish at once with scripted results instead of running commands."""

    def __init__(self, root, results):
        super().__init__(root, phone=PhoneControl(Settings.load(root / ".studio/settings.json"), root,
                                                  runner=FakeAdb()))
        self.results, self.ran = results, []

    def run(self, kind, params=None, *, on_done=None):
        self.ran.append((kind, dict(params or {})))
        res = self.results[kind](params) if callable(self.results[kind]) else self.results[kind]
        job = Job(id=f"j{len(self.ran)}", kind=kind, title=kind, argv=[], status="running")
        threading.Timer(0.01, lambda: (setattr(job, "status", "done"), setattr(job, "result", res),
                                       on_done and on_done(job))).start()
        return job


def _settle(flow, step, timeout=5.0):
    t0 = time.time()
    while flow.state[step]["status"] == "running" and time.time() - t0 < timeout:
        time.sleep(0.02)
    return flow.state[step]


def test_refine_flow_end_to_end(root):
    def propose(p):
        out = root / p["out"]
        shutil.copy(root / "configs/strategies/ladder-vision-v4.yaml", out)
        return {"states": 1000, "champion_hits": {"over_defend_bleed": 34}, "candidate_yaml": str(out),
                "best": {"knobs": {"params.defend_trigger_tiles": 9.25}, "removed": 25, "removed_frac": 0.735,
                         "collateral_pct": 0.17}}

    app = ScriptedApp(root, {
        "scan": {"over_defend_bleed": 34, "opening_throw": 9, "spell_inefficiency": 0},
        "propose": propose,
        "shadow": {"states": 1000, "bad_engagements_flipped": 7, "good_engagements_changed": 3},
        "replay_diff": {"changed_pct": 2.4, "gate": "pass"},
        "experiment": {"experiment_id": "EXP-0007", "status": "ACCEPTED", "matches": 24},
    })
    flow = app.refine
    assert flow.snapshot()["current"] == "data"
    with pytest.raises(FlowError):
        flow.scan()
    with pytest.raises(FlowError):
        flow.configure("champion", "data/nowhere")
    flow.configure("champion", "data/matches")
    assert flow.state["champion"] == "configs/strategies/ladder-vision-v4.yaml"
    flow.scan()
    assert _settle(flow, "scan")["status"] == "done" and flow.state["pattern"] == "over_defend_bleed"
    with pytest.raises(FlowError):
        flow.propose("made_up")
    flow.propose("over_defend_bleed")
    assert _settle(flow, "propose")["status"] == "done"
    assert flow.state["candidate"].startswith("configs/strategies/ladder-vision-v4-over-defend-bleed-")
    assert flow.snapshot()["current"] == "check" and flow.expected_pct() == 2.67
    with pytest.raises(FlowError):
        flow.test()  # the offline check comes first
    flow.check()
    _settle(flow, "shadow")
    _settle(flow, "replay_diff")
    assert app.ran[-1] == ("replay_diff", {**app.ran[-2][1], "expected_pct": 2.67})
    assert flow.snapshot()["current"] == "test"
    flow.test(mode="practice", max_games=20)
    assert _settle(flow, "test")["status"] == "done" and flow.snapshot()["current"] == "decide"
    kind, params = app.ran[-1]
    assert kind == "experiment" and params["mode"] == "practice" and params["max_games"] == 20
    assert params["title"] == "studio: fewer over defend bleed" and "defend_trigger_tiles 9.25" in params["hypothesis"]
    flow.promote()
    assert (root / "configs/strategies/CHAMPION").read_text().strip() == Path(flow.state["candidate"]).name
    flow.next_round()
    st = flow.snapshot()
    assert st["round"] == 2 and st["current"] == "scan" and st["champion"].endswith(st["history"][0]["candidate"].split("/")[-1])
    assert st["history"][0]["test"] == "ACCEPTED" and st["history"][0]["promoted"]
    # it survives a restart; a step that was running comes back as interrupted
    flow.state["scan"]["status"] = "running"
    flow._save()
    again = StudioApp(root).refine
    assert again.state["round"] == 2 and again.state["scan"]["status"] == "interrupted"


def test_refine_refuses_to_promote_a_rejected_candidate(root):
    def propose(p):
        out = root / p["out"]
        shutil.copy(root / "configs/strategies/ladder-vision-v6.yaml", out)
        return {"states": 10, "candidate_yaml": str(out), "best": {"knobs": {}, "removed": 1, "collateral_pct": 0}}

    app = ScriptedApp(root, {"scan": {"opening_throw": 3}, "propose": propose,
                             "shadow": {"bad_engagements_flipped": 0, "good_engagements_changed": 1},
                             "replay_diff": {"gate": "fail"}, "experiment": {"status": "REJECTED"}})
    f = app.refine
    f.configure("v4", "data/matches")
    f.scan()
    _settle(f, "scan")
    f.propose("opening_throw")
    _settle(f, "propose")
    f.check()
    _settle(f, "shadow")
    _settle(f, "replay_diff")
    f.test()
    _settle(f, "test")
    with pytest.raises(FlowError, match="accepted"):
        f.promote()
    assert (root / "configs/strategies/CHAMPION").read_text().strip() == "ladder-vision-v4.yaml"
    f.next_round()
    assert f.state["champion"] == "configs/strategies/ladder-vision-v4.yaml" and f.state["history"][0]["test"] == "REJECTED"


# --- the server ----------------------------------------------------------------------------
def _req(srv, method, path, body=None, *, token=None, host=None):
    port = srv.httpd.server_address[1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": host or f"127.0.0.1:{port}"}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-Studio-Token"] = token
    c.request(method, path, body=data, headers=headers)
    r = c.getresponse()
    out = r.read()
    c.close()
    return r.status, out, r


def test_server_guards_and_routes(fake_live):
    app = fake_live
    srv = StudioServer(Studio(app=app), "127.0.0.1", 0).start()
    try:
        status, page, _ = _req(srv, "GET", "/")
        assert status == 200 and f'content="{app.token}"'.encode() in page
        assert _req(srv, "GET", "/", host="evil.example:8765")[0] == 403  # DNS rebinding
        assert _req(srv, "GET", "/api/app/state", host="localhost:1")[0] == 200
        assert _req(srv, "POST", "/api/app/phone/lock", {"locked": False})[0] == 403  # no token
        assert _req(srv, "POST", "/api/app/phone/lock", {"locked": False}, token="wrong")[0] == 403
        st, body, _ = _req(srv, "POST", "/api/app/phone/lock", {"locked": False}, token=app.token)
        assert st == 200 and json.loads(body) == {"locked": False}
        st, body, _ = _req(srv, "POST", "/api/app/phone/tap", {"x": 232, "y": 512, "w": 464, "h": 1024},
                           token=app.token)
        assert st == 200 and json.loads(body)["device_xy"] == [540, 1200]
        _req(srv, "POST", "/api/app/phone/lock", {"locked": True}, token=app.token)
        assert _req(srv, "POST", "/api/app/phone/key", {"key": "home"}, token=app.token)[0] == 423
        st, body, _ = _req(srv, "POST", "/api/app/settings", {"changes": {"play_matches": 3, "port": 1}},
                           token=app.token)
        assert st == 200 and json.loads(body)["errors"].keys() == {"port"} and app.settings["play_matches"] == 3
        st, body, _ = _req(srv, "POST", "/api/app/run", {"kind": "nope"}, token=app.token)
        assert st == 400 and "unknown action" in json.loads(body)["error"]
        st, body, _ = _req(srv, "POST", "/api/app/refine/scan", {}, token=app.token)
        assert st == 400 and "champion" in json.loads(body)["error"]
        assert json.loads(_req(srv, "GET", "/api/app/state")[1])["live"] is True
        # the live view is the running job's, proxied
        st, body, r = _req(srv, "GET", "/api/live/frame.png")
        assert st == 200 and body.startswith(b"\x89PNG") and r.getheader("X-Frame-Seq") == "7"
        assert json.loads(_req(srv, "GET", "/api/live/config")[1]) == {"rois": {}}
        st, body, r = _req(srv, "GET", "/api/live/events")
        assert st == 200 and body.count(b"data: ") == 2 and r.getheader("Content-Type") == "text/event-stream"
        st, body, _ = _req(srv, "POST", "/api/app/screenshot", {}, token=app.token)
        assert st == 200 and json.loads(body)["source"] == "stream"
        app.jobs.jobs.clear()
        assert _req(srv, "GET", "/api/live/snapshot")[0] == 404
        assert json.loads(_req(srv, "GET", "/api/app/strategies")[1])[0]["champion"] is True
    finally:
        srv.stop()


def test_server_without_the_app_is_unchanged(tmp_path):
    srv = StudioServer(Studio(), "127.0.0.1", 0).start()
    try:
        st, page, _ = _req(srv, "GET", "/", host="anything.example")
        assert st == 200 and b'name="studio-token" content=""' in page
        assert _req(srv, "POST", "/api/app/run", {"kind": "play"})[0] == 404
        assert json.loads(_req(srv, "GET", "/api/info")[1])["app"] is False
    finally:
        srv.stop()


def test_bare_command_opens_the_studio(monkeypatch):
    from droid_harness import cli
    from droid_harness.ui import launch

    calls = []
    monkeypatch.setattr(launch, "run_app", lambda **kw: calls.append(kw))
    cli.main([])
    cli.main(["studio", "--port", "9000", "--no-open"])
    assert calls == [{"host": None, "port": None, "open_browser": None, "native": None},
                     {"host": None, "port": 9000, "open_browser": False, "native": None}]


def test_quit_stops_jobs_and_the_app(root):
    app = StudioApp(root)
    quit_called = threading.Event()
    app.on_quit = quit_called.set
    srv = StudioServer(Studio(app=app), "127.0.0.1", 0).start()
    try:
        assert _req(srv, "POST", "/api/app/quit", {})[0] == 403
        st, body, _ = _req(srv, "POST", "/api/app/quit", {}, token=app.token)
        assert st == 200 and json.loads(body) == {"quitting": True}
        assert quit_called.wait(5)
    finally:
        srv.stop()
