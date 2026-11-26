"""Offline tests for the M9 L2 advisor. No network: urllib is monkeypatched/stubbed."""

import json
import urllib.request

from droid_harness.cli import main as cli_main
from droid_harness.core.runtime.advisor import POSTURES, Advisor, build_prompt, parse_reply
from droid_harness.core.runtime.intent_bus import IntentBus


def _state(**kw):
    d = {"t_ms": 61000, "phase": "x1", "my_elixir": 6.5, "opp_elixir": [3.0, 10.0],
         "hand": ["knight", "musketeer", "fireball", "skeletons"], "hand_fresh": [True] * 4,
         "next": "hog_rider", "units": [[1, "knight", "opp", 5.5, 20.0, 0.0, -1.0, 0.8]],
         "towers": {"me_left": 1400}, "crowns": [0, 0], "opp_last_four": ["knight"],
         "opp_deck_seen": [], "pocket_open": {"left": False, "right": False},
         "posture": "BALANCED", "last_action_ms": -10000, "warnings": []}
    d.update(kw)
    return d


def test_build_prompt_contains_state_facts_and_schema():
    p = build_prompt(_state())
    assert isinstance(p, str)
    for fact in ("61000", "x1", "6.5", "knight", "hog_rider", "BALANCED"):
        assert fact in p, fact
    for schema_bit in ('"posture"', '"note"', "OPENING", "SPELL_CYCLE"):
        assert schema_bit in p, schema_bit


def test_parse_reply_accepts_valid_json_within_allowed():
    intent = parse_reply('{"posture": "PRESS", "note": "opp broke, punish"}', {"PRESS", "DEFEND_LEAD"})
    assert intent is not None and intent.posture == "PRESS" and intent.note == "opp broke, punish"
    fenced = parse_reply('```json\n{"posture": "DEFEND_LEAD", "note": "sit"}\n```', {"DEFEND_LEAD"})
    assert fenced is not None and fenced.posture == "DEFEND_LEAD"
    noop = parse_reply('{"posture": null, "note": "no read"}', {"PRESS"})
    assert noop is not None and noop.posture is None


def test_parse_reply_rejects_posture_outside_allowed():
    assert parse_reply('{"posture": "PRESS", "note": "x"}', {"DEFEND_LEAD"}) is None
    assert parse_reply('{"posture": "PRESS", "note": "x"}', set()) is None


def test_parse_reply_rejects_malformed():
    assert parse_reply("not json at all", set(POSTURES)) is None
    assert parse_reply('{"note": "missing posture"}', set(POSTURES)) is None
    assert parse_reply('{"posture": "YOLO", "note": "x"}', set(POSTURES)) is None
    assert parse_reply("", set(POSTURES)) is None


class _FakeResp:
    def __init__(self, payload):
        self._raw = json.dumps(payload).encode()

    def read(self):
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_advisor_publishes_with_ttl_roughly_2x_cadence(monkeypatch):
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["timeout"] = timeout
        seen["url"] = req.full_url
        return _FakeResp({"choices": [{"message": {"content": '{"posture": "PRESS", "note": "go"}'}}]})

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    bus = IntentBus()
    adv = Advisor(bus, "http://localhost:11434/v1", "k", "m", cadence_s=4.0, allowed={"PRESS"})
    intent = adv.poll(_state())
    assert intent is not None and intent.posture == "PRESS"
    assert bus.latest() is not None and bus.latest().posture == "PRESS"
    assert bus.max_age_s == 8.0  # TTL = 2x cadence
    assert seen["timeout"] == 8.0 and seen["url"].endswith("/chat/completions")


def test_advisor_skips_on_timeout(monkeypatch):
    def boom(req, timeout=None):
        raise TimeoutError("boom")

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    bus = IntentBus()
    adv = Advisor(bus, "http://localhost:11434/v1", "k", "m", allowed=set(POSTURES))
    assert adv.poll(_state()) is None
    assert bus.latest() is None


def test_cli_advise_dry_run_prints_prompts(tmp_path, capsys):
    m = tmp_path / "m1"
    m.mkdir()
    recs = [{"seq": i, "state": _state(t_ms=i * 1000, my_elixir=5.0 + i)} for i in range(3)]
    (m / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    cli_main(["advise", "--data", str(m), "--dry-run"])
    out = capsys.readouterr().out
    assert "t=0ms" in out and "t=2000ms" in out and "elixir=you 7.0" in out
    assert out.count("Respond with JSON ONLY") == 3  # all three prompts printed, no network
