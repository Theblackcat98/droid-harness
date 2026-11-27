"""Autopilot cycle with a fake phone, notifications, publishing, and the value model."""

from __future__ import annotations

import datetime as dt
import json
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from droid_harness.flywheel.autopilot import Autopilot, AutopilotConfig, notify, publish, read_champion
from droid_harness.learn.value_model import ValueModel, swings, train

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "data/matches-exp-v4-v6"
pytestmark = pytest.mark.skipif(not EXP.exists(), reason="committed EXP-0001 data not present")


class FakePlayer:
    """Copies committed compact match files instead of playing; experiments return a scripted verdict."""

    def __init__(self, verdict: str = "accepted") -> None:
        self.verdict = verdict
        self.played: list[tuple] = []
        self.tested: list[tuple] = []

    def play(self, strategy, matches, data, mode):
        self.played.append((Path(strategy).stem, matches, mode))
        data = Path(data)
        data.mkdir(parents=True, exist_ok=True)
        out = []
        for d in sorted(EXP.iterdir())[:matches]:
            shutil.copytree(d, data / d.name)
            out.append(json.loads((d / "summary.json").read_text()))
        return out

    def experiment(self, champion, candidate, data, mode, *, title, hypothesis):
        self.tested.append((Path(champion).stem, Path(candidate).stem, mode, title))
        return {"experiment_id": "EXP-0099", "status": self.verdict, "matches": 24, "notes": "fake"}


def _workspace(tmp_path: Path) -> AutopilotConfig:
    strat = tmp_path / "configs/strategies"
    (strat / "counters").mkdir(parents=True)
    for v in ("v4", "v6"):
        shutil.copy(ROOT / f"configs/strategies/ladder-vision-{v}.yaml", strat)
        shutil.copy(ROOT / f"configs/strategies/counters/ladder-vision-{v}.csv", strat / "counters")
    return AutopilotConfig(champion="ladder-vision-v4", farm_matches=12, testbed="training_camp",
                           data_root=tmp_path / "data/autopilot", strategies_dir=strat,
                           store=tmp_path / "ledger/experiments.jsonl", reports_dir=tmp_path / "reports",
                           value_model=tmp_path / "models/value.json", min_matches_for_value=6, trials=12,
                           min_removed=0.1, max_collateral_pct=5.0, cards=ROOT / "tests/fixtures/cards_starter.json",
                           repo=tmp_path)


def _now():
    return dt.datetime(2026, 11, 28, 2, 30, tzinfo=dt.UTC)


def test_cycle_farms_proposes_tests_promotes_and_reports(tmp_path):
    cfg = _workspace(tmp_path)
    player = FakePlayer("accepted")
    rep = Autopilot(cfg, player, now=_now).cycle()
    assert player.played == [("ladder-vision-v4", 12, "ladder")]
    assert rep["farm"]["record"]["win"] + rep["farm"]["record"]["loss"] >= 10
    assert rep["value_model"]["matches"]["train"] > 0 and cfg.value_model.exists()
    assert rep["proposals"] and rep["proposals"][-1]["best"] is not None
    champ, cand, mode, title = player.tested[0]
    assert champ == "ladder-vision-v4" and cand.startswith("ladder-vision-v4-") and mode == "training_camp"
    assert title.startswith("auto: ")
    assert rep["promoted"] == cand and read_champion(cfg).stem == cand
    assert (cfg.strategies_dir / "CHAMPION").read_text().strip().startswith("auto/")
    out = cfg.reports_dir / rep["stamp"]
    assert (out / "report.md").exists() and (out / "digest.md").exists() and (cfg.reports_dir / "latest.md").exists()
    assert "Promoted" in (cfg.reports_dir / "latest.md").read_text()


def test_rejected_candidate_keeps_the_champion_and_stop_file_ends_the_run(tmp_path):
    cfg = _workspace(tmp_path)
    cfg.cycles = 0
    player = FakePlayer("rejected")
    pilot = Autopilot(cfg, player, now=_now)
    rep = pilot.cycle()
    assert rep["experiment"]["status"] == "rejected" and "promoted" not in rep
    assert read_champion(cfg).stem == "ladder-vision-v4"
    (cfg.data_root / "STOP").touch()
    assert pilot.run() == []  # STOP exists: no cycle starts


def test_notify_posts_a_summary():
    got = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            got["title"] = self.headers.get("Title")
            got["body"] = self.rfile.read(int(self.headers["Content-Length"])).decode()
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    assert notify(f"http://127.0.0.1:{srv.server_address[1]}/topic", "Champion v4: farmed 9W-3L.", title="t")
    srv.server_close()
    assert got == {"title": "t", "body": "Champion v4: farmed 9W-3L."}
    assert notify("http://127.0.0.1:9/nothing-listens", "x") is False


def test_publish_commits_reports_to_a_branch(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True, text=True)  # noqa: E731
    git("init", "-q", "-b", "main")
    (repo / ".gitignore").write_text("reports/\n.autopilot-publish/\n")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    cfg = AutopilotConfig(reports_dir=Path("reports"), store=Path("ledger/experiments.jsonl"),
                          strategies_dir=Path("configs/strategies"))
    (repo / "reports/2026-11-28_0230").mkdir(parents=True)
    (repo / "reports/2026-11-28_0230/report.md").write_text("# r\n")
    (repo / "reports/latest.md").write_text("# r\n")
    res = publish(cfg, "2026-11-28_0230", repo=repo, push=False)
    assert res["ok"] and res["commit"], res
    files = git("ls-tree", "-r", "--name-only", "autopilot/reports").stdout.split()
    assert "reports/latest.md" in files and "reports/2026-11-28_0230/report.md" in files
    assert git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "main"  # your checkout is untouched
    assert publish(cfg, "2026-11-28_0230", repo=repo, push=False)["commit"] is None  # nothing new


def test_value_model_trains_predicts_and_finds_swings(tmp_path):
    m = train([EXP])
    assert m["matches"]["train"] >= 8 and 0 <= m["base_rate"] <= 1
    (tmp_path / "v.json").write_text(json.dumps(m))
    vm = ValueModel.load(tmp_path / "v.json")
    d = next(p for p in sorted(EXP.iterdir()) if json.loads((p / "summary.json").read_text()).get("result") == "loss")
    states = [json.loads(ln)["state"] for ln in (d / "decisions.jsonl").read_text().splitlines()[::10]]
    probs = vm.predict_many(states)
    assert all(0.0 <= p <= 1.0 for p in probs)
    sw = swings(states, probs, top=3)
    assert all(r["drop"] > 0.05 and r["from_t_ms"] <= r["t_ms"] for r in sw)
    bad = dict(m, features=["x"])
    with pytest.raises(ValueError):
        ValueModel(bad)
