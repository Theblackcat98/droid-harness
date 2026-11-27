"""Drive a paired experiment on a live ``BotRunner`` (shared by ``experiment`` and ``autopilot``).

One persistent runner (detector weights and scrcpy loaded once); the strategy is hot-swapped
before every match according to ``ExperimentRunner.next_assignment``; every finished match is
recorded, written to the ledger and the warehouse, and the loop stops when the test decides.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from .runner import ExperimentRunner


def run_paired(bot, runner: ExperimentRunner, champion: str | Path, candidate: str | Path, *,
               store: str | Path = "ledger/experiments.jsonl", db: str | Path | None = "data/warehouse.duckdb",
               stop: Callable[[], bool] = lambda: False, log: Callable[[str], None] = print) -> ExperimentRunner:
    exp_id = runner.cfg.experiment_id

    def assign() -> None:
        arm, pid = runner.next_assignment()
        bot.set_strategy(champion if arm == "champion" else candidate, experiment_id=exp_id, arm=arm, pair_id=pid)

    def on_done(s: dict) -> None:
        if s.get("experiment_id") != exp_id:
            return
        r = runner.record(s)
        log(json.dumps({"match": s.get("match_id"), "arm": s.get("arm"), "pair": s.get("pair_id"),
                        "result": s.get("result"), "sprt": r.decision, "llr": round(r.llr, 3),
                        "notes": runner.notes()}))
        runner.write_ledger(str(store))
        if db:
            try:
                from .. import warehouse

                con = warehouse.connect(str(db))
                runner.write_warehouse(con)
                con.close()
            except Exception as e:  # the warehouse must never stop the loop
                log(f"warehouse write failed: {e}")
        if not runner.done:
            assign()

    assign()
    bot.run(stop_when=lambda: stop() or runner.done or len(runner.matches) >= runner.cfg.max_games * 2,
            on_summary=on_done)
    return runner
