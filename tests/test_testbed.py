"""R4 (practice gauntlet / self-play flows) and R8 (dense paired test)."""

from __future__ import annotations

import random

from droid_harness.domains.clash_royale.menus import DEFAULT_FLOWS, MenuNavigator, required_for
from droid_harness.domains.clash_royale.perception.grid import Roi
from droid_harness.flywheel.eval.runner import ExperimentConfig, ExperimentRunner


def test_practice_flow_picks_the_pair_opponent():
    slots = {f"practice_slot_{i}": Roi(0.1, 0.2 + 0.1 * i, 0.9, 0.25 + 0.1 * i) for i in range(4)}
    nav = MenuNavigator({**slots, "menu": Roi(0, 0, 0.1, 0.1)}, dict(DEFAULT_FLOWS["practice"]), cooldown_s=0)
    nav.slot_index = 1
    ev = nav.events_for("BATTLE_LOG", (1024, 464, 3), 0)
    assert ev and ev[0].action_id == "menu:BATTLE_LOG:practice_slot_1"
    nav.slot_index = 4  # wraps over the calibrated entries (two fit on screen)
    assert nav.events_for("BATTLE_LOG", (1024, 464, 3), 0)[0].action_id.endswith("practice_slot_0")
    buttons, screens = required_for("practice")
    # The Practice tap starts the match directly (VS screen), so no confirm screen is required.
    assert {"battle_log", "practice_slot_0"} <= buttons and "BATTLE_LOG" in screens
    assert "confirm" not in buttons and "PRACTICE_CONFIRM" not in screens
    assert "RESULT_WIN" not in screens and "UNKNOWN" not in screens
    assert {"friendly_host", "friendly_guest"} <= DEFAULT_FLOWS.keys()


def _run_dense(diff: float, noise: float, seed: int = 0, max_games: int = 200) -> ExperimentRunner:
    rng = random.Random(seed)
    r = ExperimentRunner(ExperimentConfig(experiment_id="D", game_version="v", test="dense", max_games=max_games,
                                          seed=seed, dense_tau=0.5))
    while not r.done:
        arm, pid = r.next_assignment()
        base = rng.gauss(0, noise)
        margin = base + (diff if arm == "candidate" else 0.0)
        res = "win" if margin > 0 else "loss"
        r.record({"arm": arm, "pair_id": pid, "game_version": "v", "status": "completed", "result": res,
                  "my_crowns": 1 if res == "win" else 0, "opp_crowns": 0 if res == "win" else 1,
                  "tower_hp_diff": int(margin * 1000), "duration_s": 180})
    return r


def test_dense_test_accepts_clear_gains_and_rejects_clear_losses():
    good = _run_dense(diff=1.5, noise=0.8)
    assert good.status == "accepted" and len(good.matches) < 60
    bad = _run_dense(diff=-1.5, noise=0.8, seed=1)
    assert bad.status == "rejected" and len(bad.matches) < 60
    flat = _run_dense(diff=0.0, noise=0.8, seed=2, max_games=40)
    assert flat.status in ("inconclusive", "running", "rejected", "accepted")
    log_l, mean, n = flat.msprt()
    assert n >= 1 and isinstance(log_l, float)


def test_dense_test_waits_for_enough_pairs():
    r = ExperimentRunner(ExperimentConfig(experiment_id="D", game_version="v", test="dense", min_dense_pairs=6))
    for _ in range(4):  # two pairs only
        arm, pid = r.next_assignment()
        r.record({"arm": arm, "pair_id": pid, "game_version": "v", "status": "completed",
                  "result": "win" if arm == "candidate" else "loss", "my_crowns": 3, "opp_crowns": 0})
    assert r.decision.decision == "continue"
