import random
import subprocess

import pytest
from matchgen import write_match

from droid_harness.flywheel import warehouse
from droid_harness.flywheel.coach.coach import context_pack, propose
from droid_harness.flywheel.coach.proposal import Proposal, apply_patch, check_guardrails, diff_paths
from droid_harness.flywheel.engagements import segment
from droid_harness.flywheel.eval import sprt as S
from droid_harness.flywheel.eval.inclusion import abort_imbalance, tally
from droid_harness.flywheel.eval.selfplay import Seat, balance_report, schedule
from droid_harness.flywheel.eval.stats import Arm, prob_best, two_proportion_z, wilson
from droid_harness.flywheel.ledger import Experiment, Ledger
from droid_harness.flywheel.loss_stories import loss_story
from droid_harness.flywheel.report import batch_report
from droid_harness.flywheel.timeline import load_match
from droid_harness.flywheel.triage import triage


def test_engagement_value_and_credit(tmp_path, arena, cards):
    tl = load_match(write_match(tmp_path, "m1"))
    engs = segment(tl, arena, cards)
    assert len(engs) == 1
    e = engs[0]
    assert e.lane == "left" and e.t_start == 10_000 and 15_000 <= e.t_end <= 17_500
    assert (e.spent_me, e.spent_opp) == (3, 4)
    assert e.board_me_end == pytest.approx(3 * 0.6) and e.board_opp_end == 0
    assert e.tower_dmg_taken == 200 and e.tower_dmg_dealt == 0
    assert e.value == pytest.approx((1.8 - 3) - (0 - 4) - 0.01 * 200)
    [(seq, v)] = e.credit.items()
    assert v == pytest.approx(e.value)


@pytest.mark.parametrize("kw,tag", [
    ({}, "OK"),
    ({"warnings_at": (11_000, 11_500)}, "PERCEPTION"),
    ({"confirm_play": False}, "EXECUTION"),
    ({"frame_gap_at": 12_000}, "LATENCY"),
    ({"respond": False, "result": "loss"}, "STRATEGY"),
])
def test_triage_tags(tmp_path, arena, cards, kw, tag):
    tl = load_match(write_match(tmp_path, "m", **kw))
    [e] = [triage(e, tl, arena) for e in segment(tl, arena, cards)]
    assert e.triage == tag, e.evidence


def test_loss_story_only_strategy(tmp_path, arena, cards):
    tl = load_match(write_match(tmp_path, "lost", respond=False, result="loss"))
    engs = [triage(e, tl, arena) for e in segment(tl, arena, cards)]
    story = loss_story(tl, engs)
    assert "STRATEGY" in story and "hog_rider" in story and "loss 0-1" in story


def test_report_and_warehouse(tmp_path, arena, cards):
    root = tmp_path / "matches"
    write_match(root, "a", result="win")
    write_match(root, "b", respond=False, result="loss")
    write_match(root, "c", status="infra_abort", result="aborted:disconnect")
    analyzed = [warehouse.analyze_match(root / m, arena, cards) for m in "abc"]
    rep = batch_report(analyzed)
    assert rep["results"] == {"win": 1, "loss": 1} and rep["win_rate"] == 0.5
    assert "defend.lane_threat" in rep["play_value_by_rule"]
    con = warehouse.connect(tmp_path / "w.duckdb")
    assert warehouse.ingest_dir(con, root, arena, cards) == 3
    [row] = warehouse.win_rates(con)
    assert (row["wins"], row["losses"]) == (1, 1)  # the infra abort is not a loss
    rules = warehouse.play_value_by_rule(con)
    assert rules and rules[0][0] == "defend.lane_threat"
    assert con.execute("select count(*) from engagements where triage='STRATEGY'").fetchone()[0] == 1


def test_sprt_matches_plan_numbers():
    assert S.expected_games(0.5, 0.58) == pytest.approx(205.8, abs=1)
    assert S.expected_games(0.5, 0.60) == pytest.approx(131.7, abs=1)
    assert S.expected_games(0.5, 0.55) == pytest.approx(529, abs=2)
    assert S.sprt(0, 0).decision == "continue"
    assert S.sprt(130, 70).decision == "accept_H1"
    assert S.sprt(90, 110).decision == "accept_H0"
    assert S.elo_from_p(0.58) == pytest.approx(56.07, abs=0.05)
    assert S.p_from_elo(S.elo_from_p(0.6)) == pytest.approx(0.6)
    assert S.paired_sprt(40, 10).decision == "accept_H1"


def test_sprt_error_rates_by_simulation():
    rng = random.Random(1)

    def run(p):
        w = lo = 0
        while True:
            if rng.random() < p:
                w += 1
            else:
                lo += 1
            r = S.sprt(w, lo)
            if r.decision != "continue":
                return r.decision

    false_accept = sum(run(0.5) == "accept_H1" for _ in range(400)) / 400
    power = sum(run(0.58) == "accept_H1" for _ in range(400)) / 400
    assert false_accept < 0.09 and power > 0.9


def test_stats_helpers():
    assert wilson(50, 100) == pytest.approx((0.404, 0.596), abs=0.002)
    assert wilson(0, 0) == (0.0, 1.0)
    pb = prob_best([Arm("a", 30, 50), Arm("b", 20, 50)])
    assert pb["a"] > 0.9
    z, p = two_proportion_z(10, 100, 2, 100)
    assert z > 0 and p < 0.05


def test_selfplay_schedule_is_balanced():
    a, b = Seat("phone-a", "acct-1"), Seat("phone-b", "acct-2")
    games = schedule(200, a, b)
    rep = balance_report(games)
    assert rep["candidate_phone"] == {"phone-a": 100, "phone-b": 100}
    assert rep["candidate_account"] == {"acct-1": 100, "acct-2": 100}
    assert rep["host"] == {"phone-a": 100, "phone-b": 100}
    assert all(g.candidate_seat != g.champion_seat for g in games)


def test_inclusion_rule_counts_completed_and_flags_abort_imbalance():
    ms = ([{"arm": "cand", "game_version": "9.x", "status": "completed", "result": "win"}] * 30
          + [{"arm": "cand", "game_version": "9.x", "status": "infra_abort"}] * 12
          + [{"arm": "cand", "game_version": "8.x", "status": "completed", "result": "loss"}] * 5
          + [{"arm": "champ", "game_version": "9.x", "status": "completed", "result": "loss"}] * 30
          + [{"arm": "champ", "game_version": "9.x", "status": "infra_abort"}] * 1)
    c, ch = tally(ms, "cand", "9.x"), tally(ms, "champ", "9.x")
    assert (c.wins, c.aborts, c.excluded_other_version) == (30, 12, 5)
    flagged, p = abort_imbalance(c, ch)
    assert flagged and p < 0.05


def test_ledger_roundtrip_and_dedupe(tmp_path):
    led = Ledger(tmp_path / "exp.jsonl")
    assert led.next_id() == "EXP-0001"
    e = Experiment("EXP-0001", "Earlier cannon", "Cannon placed late vs golem pushes; move defend trigger 7 to 8 tiles",
                   "coach", "aaa", "bbb", "sprt_h2h", 0.5, 0.58, "9.x", "2026-11")
    led.upsert(e)
    e.status, e.wins, e.losses = "rejected", 112, 119
    led.upsert(e)
    assert led.all()[0].status == "rejected" and led.next_id() == "EXP-0002"
    dup = led.similar_failures("Cannon placed late against golem pushes: move defend trigger to 8 tiles", "9.x")
    assert [d.experiment_id for d in dup] == ["EXP-0001"]
    assert not led.similar_failures("Cannon placed late against golem pushes: move defend trigger to 8 tiles", "10.x")
    md = led.render_markdown(tmp_path / "L.md")
    assert "EXP-0001 · REJECTED" in md and "112–119" in md


PATCH = """--- a/configs/strategies/hog-cycle-v1.yaml
+++ b/configs/strategies/hog-cycle-v1.yaml
@@ -1,1 +1,1 @@
-strategy: hog-cycle-v1
+strategy: hog-cycle-v2
"""


def _proposal(patch=PATCH, change_type="param"):
    return Proposal(title="t", hypothesis="h", change_type=change_type, patch=patch, target_metric="m",
                    expected_direction="up", expected_decision_change_pct=2.0)


def test_proposal_guardrails():
    assert diff_paths(PATCH) == ["configs/strategies/hog-cycle-v1.yaml"]
    assert check_guardrails(_proposal()) == []
    bad = PATCH.replace("configs/strategies/hog-cycle-v1.yaml",
                        "src/droid_harness/domains/clash_royale/policy/constraints.py")
    assert any("not allowed" in p for p in check_guardrails(_proposal(bad)))
    assert any("unit test" in p for p in check_guardrails(_proposal(change_type="rule_edit")))


def test_apply_patch_in_git_repo(tmp_path):
    repo = tmp_path / "r"
    (repo / "configs/strategies").mkdir(parents=True)
    (repo / "configs/strategies/hog-cycle-v1.yaml").write_text("strategy: hog-cycle-v1\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    apply_patch(repo, PATCH)
    assert (repo / "configs/strategies/hog-cycle-v1.yaml").read_text() == "strategy: hog-cycle-v2\n"
    with pytest.raises(subprocess.CalledProcessError):
        apply_patch(repo, PATCH)  # no longer applies


class FakeLLM:
    def __init__(self, payload):
        self.payload, self.seen = payload, None

    def complete_json(self, system, user):
        self.seen = (system, user)
        return self.payload


def test_coach_propose_flags_duplicates(tmp_path):
    led = Ledger(tmp_path / "exp.jsonl")
    led.upsert(Experiment("EXP-0001", "x", "raise leak threshold to cycle faster", "coach", "a", "b", "sprt_h2h",
                          0.5, 0.58, "9.x", "e", status="rejected"))
    ctx = context_pack(strategy_path="configs/strategies/hog-cycle-v1.yaml", report={"win_rate": 0.4},
                       loss_stories=["[match m] ..."], ledger=led, excluded_share=0.12)
    assert "Counter table" in ctx and "12%" in ctx and "EXP-0001" in ctx
    payload = _proposal().model_dump()
    payload["hypothesis"] = "raise leak threshold to cycle faster"
    client = FakeLLM(payload)
    p, problems = propose(client, ctx, led, "9.x")
    assert p.title == "t" and any("EXP-0001" in x for x in problems)
    assert "Proposal" in client.seen[0] or "properties" in client.seen[0]


def test_replay_diff_gate(arena, cards, strategy):
    from droid_harness.domains.clash_royale.policy.strategy import Params
    from droid_harness.domains.clash_royale.policy.utility import CRPolicy
    from droid_harness.domains.clash_royale.state import GameState, Unit
    from droid_harness.flywheel.eval.replay_diff import gate, replay_diff

    spec, params, _ = strategy
    towers = {"me_left": 3000, "me_right": 3000, "me_king": 4800, "opp_left": 3000, "opp_right": 3000,
              "opp_king": 4800}
    states = [GameState(t_ms=60_000 + i * 500, my_elixir=6.0, hand=["hog_rider", "cannon", "fireball", "skeletons"],
                        towers=towers, units=[Unit(1, "hog_rider", "opp", (3.5, 18.0 - 0.5 * i), vel=(0, -2.0))])
              for i in range(6)]
    champion = CRPolicy(cards, arena, params)
    same = replay_diff(champion, CRPolicy(cards, arena, params), states)
    assert same.changed == 0 and gate(same, 1.0)[0]
    timid = Params(spec.model_copy(update={"params": {**spec.params, "act_threshold": 99.0}}), params.counters)
    diff = replay_diff(champion, CRPolicy(cards, arena, timid), states)
    assert diff.changed == len(states) and diff.by_rule["defend.lane_threat"] == len(states)
    ok, why = gate(diff, expected_change_pct=2.0)
    assert not ok and "claimed" in why

    class Broken(CRPolicy):
        def decide(self, s, intent=None):
            raise RuntimeError("boom")

    crashed = replay_diff(champion, Broken(cards, arena, params), states)
    assert crashed.errors == len(states) and not gate(crashed, 100)[0]


def _play(runner, p_champ, p_cand, rng, max_n=2000, draws=0.0):
    n = 0
    while not runner.done and n < max_n:
        arm, pid = runner.next_assignment()
        p = p_cand if arm == "candidate" else p_champ
        u = rng.random()
        result = "draw" if u < draws else ("win" if rng.random() < p else "loss")
        runner.record({"match_id": f"m{n}", "arm": arm, "pair_id": pid, "game_version": "9.x",
                       "opponent_kind": "training_camp", "status": "completed", "result": result})
        n += 1
    return n


def test_experiment_runner_pairs_randomized_blocks_and_accepts_a_better_candidate(tmp_path):
    import random

    from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner

    cfg = ExperimentConfig(experiment_id="EXP-0001", champion_id="champ", candidate_id="cand",
                           game_version="9.x", balance_epoch="2026-11", max_games=2000, seed=1)
    r = ExperimentRunner(cfg)
    assert cfg.test == "pairs"
    # Every block of two has one match per arm, in a random order.
    blocks = []
    for i in range(20):
        arm, pid = r.next_assignment()
        r.record({"match_id": f"b{i}", "arm": arm, "pair_id": pid, "game_version": "9.x",
                  "status": "completed", "result": "win"})
        blocks.append((pid, arm))
    by_pair = {}
    for pid, arm in blocks:
        by_pair.setdefault(pid, []).append(arm)
    assert all(sorted(v) == ["candidate", "champion"] for v in by_pair.values())
    assert len({tuple(v) for v in by_pair.values()}) == 2  # both orders occur

    r = ExperimentRunner(cfg)
    n = _play(r, 0.45, 0.75, random.Random(3))
    assert r.status == "accepted" and n < 2000
    better, worse, _ = r.pair_counts()
    assert better > worse
    tmp_ledger = tmp_path / "exp.jsonl"
    r.write_ledger(str(tmp_ledger))
    assert "pairs: candidate better" in tmp_ledger.read_text()
    from droid_harness.flywheel import warehouse
    con = warehouse.connect(tmp_path / "w.duckdb")
    r.write_warehouse(con)
    row = con.execute("select status from experiments where experiment_id='EXP-0001'").fetchone()
    assert row[0] == "accepted"
    con.close()


def test_an_exhausted_undecided_experiment_is_inconclusive(tmp_path):
    """A run that hits max_games without an SPRT boundary must not look 'running' forever."""
    from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner

    cfg = ExperimentConfig(experiment_id="EXP-0002", game_version="9.x", max_games=2, seed=1)
    r = ExperimentRunner(cfg)
    for i in range(2):
        arm, pid = r.next_assignment()
        r.record({"match_id": f"m{i}", "arm": arm, "pair_id": pid, "game_version": "9.x",
                  "status": "completed", "result": "win" if arm == "champion" else "loss"})
    assert r.done and r.status == "inconclusive"
    kw = r.to_ledger_kwargs()
    assert kw["status"] == "inconclusive" and kw["matches"] == 2
    led = Ledger(tmp_path / "exp.jsonl")
    led.upsert(Experiment(**kw))
    md = led.render_markdown(tmp_path / "L.md")
    assert "INCONCLUSIVE" in md and "2 matches · candidate 0–1–0" in md


def test_a_worse_candidate_is_never_promoted_even_above_fifty_percent():
    """The review's counterexample: champion 80%, candidate 65%. The old candidate-vs-50%
    test accepted it; comparing against the champion must not."""
    import random

    from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner

    accepted_pairs = accepted_fixed = 0
    for seed in range(20):
        r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="9.x", max_games=1500, seed=seed))
        _play(r, 0.80, 0.65, random.Random(seed))
        accepted_pairs += r.status == "accepted"
        f = ExperimentRunner(ExperimentConfig(experiment_id="F", game_version="9.x", test="fixed",
                                              max_games=1500, seed=seed))
        _play(f, 0.80, 0.65, random.Random(seed))
        accepted_fixed += f.status == "accepted"
    assert accepted_pairs == 0
    assert accepted_fixed >= 15  # why "fixed" must never gate a promotion


def test_aborted_slots_are_replayed_and_abort_imbalance_is_flagged():
    from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner

    r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="9.x", seed=0))
    arm, pid = r.next_assignment()
    r.record({"match_id": "a", "arm": arm, "pair_id": pid, "game_version": "9.x",
              "status": "infra_abort", "result": None})
    assert r.next_assignment() == (arm, pid)  # same slot again
    for i, res in enumerate(("win", "loss")):
        arm, pid2 = r.next_assignment()
        assert pid2 == pid
        r.record({"match_id": f"x{i}", "arm": arm, "pair_id": pid, "game_version": "9.x",
                  "status": "completed", "result": res})
    assert sum(r.pair_counts()) == 1
    # One arm keeps aborting: flagged, and its pairs are dropped after the retry cap.
    r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="9.x", seed=0, max_slot_retries=1))
    for i in range(80):
        arm, pid = r.next_assignment()
        aborted = arm == "candidate" and i % 3 != 0
        r.record({"match_id": f"m{i}", "arm": arm, "pair_id": pid, "game_version": "9.x",
                  "status": "infra_abort" if aborted else "completed", "result": None if aborted else "win"})
    assert r.dropped_pairs and r.abort_warning() and "abort rates differ" in r.notes()


def test_h2h_refuses_training_camp_games_and_counts_draws():
    from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner

    r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="9.x", test="h2h"))
    r.record({"match_id": "m", "arm": "candidate", "game_version": "9.x", "opponent_kind": "training_camp",
              "status": "completed", "result": "win"})
    assert r.status == "void" and "pairs" in r.void_reason
    r = ExperimentRunner(ExperimentConfig(experiment_id="E", game_version="9.x", test="sprt_paired"))
    assert r.cfg.test == "h2h"  # legacy name
    for i in range(4):
        r.record({"match_id": f"m{i}", "arm": "candidate", "game_version": "9.x", "opponent_kind": "friendly",
                  "status": "completed", "result": "draw"})
    # Draws are half a win, not dropped: four draws are mild evidence against "candidate is better".
    assert r.decision.wins == 2.0 and r.decision.losses == 2.0 and r.decision.llr < 0


def test_experiment_runner_voids_on_game_update():
    from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner

    r = ExperimentRunner(ExperimentConfig(experiment_id="EXP-0002", game_version="9.x"))
    r.record({"match_id": "m1", "arm": "candidate", "game_version": "9.x",
              "status": "completed", "result": "win"})
    assert r.status == "running" and not r.done
    r.record({"match_id": "m2", "arm": "champion", "game_version": "10.x",
              "status": "completed", "result": "win"})
    assert r.status == "void" and r.done and "10.x" in (r.void_reason or "")
