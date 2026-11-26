"""Parameter tuning with Optuna. Tuning *finds candidates*; it never promotes them.

Each trial writes a child strategy file and plays a fixed block of gauntlet matches via a
callback you provide (the match runner). The objective is dense (crown + tower-HP
differential), not win/loss.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path

import yaml

SearchSpace = dict[str, tuple[str, float, float]]  # dotted.path -> ("float"|"int", low, high)

DEFAULT_SPACE: SearchSpace = {
    "params.act_threshold": ("float", 0.05, 0.4),
    "params.elixir_weight": ("float", 0.1, 1.0),
    "params.defend_trigger_tiles": ("float", 6.0, 14.0),
    "params.min_elixir_to_attack": ("float", 4.0, 10.0),
    "tier0.leak_threshold": ("float", 8.5, 9.9),
    "posture_multipliers.PRESS.offense": ("float", 0.8, 2.0),
}


def _set(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def write_child(base_path: Path, out_dir: Path, values: dict, trial_no: int) -> Path:
    spec = yaml.safe_load(base_path.read_text())
    child = copy.deepcopy(spec)
    for k, v in values.items():
        _set(child, k, v)
    child["parent"] = spec["strategy"]
    child["strategy"] = f"{spec['strategy']}-opt{trial_no:03d}"
    out = out_dir / f"{child['strategy']}.yaml"
    if child.get("counters"):
        child["counters"] = str((base_path.parent / child["counters"]).resolve())
    out.write_text(yaml.safe_dump(child, sort_keys=False))
    return out


def run_study(base_strategy: str | Path, play_block: Callable[[Path], list[dict]], *, n_trials: int = 30,
              space: SearchSpace | None = None, out_dir: str | Path = "data/tuning", seed: int = 0):
    import optuna  # optional dependency

    base, out = Path(base_strategy), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    space = space or DEFAULT_SPACE

    def objective(trial):
        values = {}
        for k, (kind, lo, hi) in space.items():
            values[k] = trial.suggest_int(k, int(lo), int(hi)) if kind == "int" else trial.suggest_float(k, lo, hi)
        path = write_child(base, out, values, trial.number)
        results = play_block(path)  # list of match summaries
        score = 0.0
        for i, m in enumerate(results):
            score += (m.get("my_crowns", 0) - m.get("opp_crowns", 0)) + m.get("tower_hp_diff", 0) / 1000
            trial.report(score / (i + 1), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return score / max(1, len(results))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials)
    return study
