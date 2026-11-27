"""Speed-1 Offline Counterfactual Search (0 phone games, 0 LLM calls).

Screens hundreds of YAML/threshold parameter candidates in seconds on CPU against
recorded real matches by pairing ``ShadowRunner`` / ``CRPolicy`` counterfactual choices
with each tick's lane ``Engagement.value`` and enforcing a ``replay-diff`` change budget.
Only the single top-scoring candidate advances to live phone testing.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import yaml

from ...domains.clash_royale.arena import Arena
from ...domains.clash_royale.cards import CardDB
from ...domains.clash_royale.policy.strategy import Params, StrategySpec, load_counters, load_strategy
from ...domains.clash_royale.policy.utility import CRPolicy
from ...domains.clash_royale.state import GameState
from ..engagements import segment
from ..timeline import load_match
from ..triage import triage

OFFLINE_SPACE: dict[str, tuple[str, float, float]] = {
    "params.act_threshold": ("float", 0.05, 0.22),
    "params.elixir_weight": ("float", 0.20, 0.65),
    "params.defend_trigger_tiles": ("float", 9.5, 13.5),
    "params.defend_horizon_s": ("float", 6.0, 10.5),
    "params.min_elixir_to_attack": ("float", 4.5, 7.5),
    "params.max_opp_elixir_to_bridge": ("float", 4.5, 7.5),
    "params.min_elixir_advantage_to_bridge": ("float", 0.5, 3.0),
    "params.spell_value_threshold": ("float", 3.0, 5.5),
    "params.counterpush_min_value": ("float", 1.8, 3.5),
    "params.giant_safe_build_elixir": ("float", 8.2, 9.6),
    "params.defense_reserve": ("float", 1.5, 3.5),
    "params.reserve_penalty": ("float", 0.35, 0.70),
    "params.counter_in_hand_mult": ("float", 0.35, 0.70),
    "params.counter_out_mult": ("float", 1.25, 1.70),
    "params.ignore_chip_max_threat_value": ("float", 0.0, 2.0),
    "posture_multipliers.PRESS.offense": ("float", 1.1, 1.7),
    "posture_multipliers.DEFEND_LEAD.defend": ("float", 1.1, 1.6),
}


def _choice_key(decision: Any) -> tuple[str | None, str | None, tuple[float, float] | None]:
    c = getattr(decision, "chosen", None)
    if c is None:
        return (None, None, None)
    return (c.rule_id, c.card, tuple(c.tile) if c.tile is not None else None)


def load_valued_states(match_dirs: list[Path], cards: CardDB, arena: Arena, *,
                       max_states: int = 6000) -> list[tuple[GameState, float]]:
    """Load ``(GameState, engagement_value)`` pairs from recorded matches."""
    raw_items: list[tuple[GameState, float]] = []
    for d in match_dirs:
        dec_path = d / "decisions.jsonl"
        if not dec_path.exists():
            continue
        tl = load_match(d)
        engs = [triage(e, tl, arena) for e in segment(tl, arena, cards)]
        # Map (t_ms) -> worst/active non-perception engagement value
        clean_engs = [e for e in engs if e.triage in ("STRATEGY", "OK")]
        for line in dec_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            st_dict = rec.get("state")
            if not isinstance(st_dict, dict) or "t_ms" not in st_dict:
                continue
            t_ms = int(st_dict["t_ms"])
            overlapping = [e.value for e in clean_engs if e.t_start <= t_ms <= e.t_end]
            val = min(overlapping) if overlapping else 0.0
            raw_items.append((GameState.from_digest(st_dict), float(val)))
    if len(raw_items) > max_states:
        step = max(1, len(raw_items) // max_states)
        raw_items = raw_items[::step][:max_states]
    return raw_items


def _set_dotted(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = round(float(value), 4) if isinstance(value, float) else value


def evaluate_counterfactual(champ_choices: list[tuple[Any, ...]],
                            cand_policy: CRPolicy,
                            valued_states: list[tuple[GameState, float]], *,
                            min_diff_pct: float = 1.0,
                            max_diff_pct: float = 18.0) -> dict[str, Any]:
    changed = 0
    bad_flipped = 0
    good_broken = 0
    bad_value_fixed = 0.0
    good_value_lost = 0.0
    for (st, val), champ_c in zip(valued_states, champ_choices, strict=True):
        cand_policy.reset_match()
        cand_c = _choice_key(cand_policy.decide(copy.copy(st)))
        if cand_c != champ_c:
            changed += 1
            if val < -0.5:
                bad_flipped += 1
                bad_value_fixed += abs(val)
            elif val > 0.5:
                good_broken += 1
                good_value_lost += val
    n = max(1, len(valued_states))
    diff_pct = 100.0 * changed / n
    # Penalize candidates that change nothing (< min_diff_pct) or rewrite the whole policy (> max_diff_pct).
    guardrail_pen = 0.0
    if diff_pct < min_diff_pct:
        guardrail_pen = 50.0 * (min_diff_pct - diff_pct)
    elif diff_pct > max_diff_pct:
        guardrail_pen = 20.0 * (diff_pct - max_diff_pct)
    score = bad_value_fixed - 1.4 * good_value_lost - guardrail_pen
    return {
        "states": len(valued_states),
        "changed": changed,
        "changed_pct": round(diff_pct, 2),
        "bad_flipped": bad_flipped,
        "good_broken": good_broken,
        "bad_value_fixed": round(bad_value_fixed, 2),
        "good_value_lost": round(good_value_lost, 2),
        "objective": round(score, 3),
        "within_guardrail": min_diff_pct <= diff_pct <= max_diff_pct,
    }


def search_offline(champion_path: str | Path, match_dirs: list[Path], cards: CardDB, arena: Arena, *,
                   n_trials: int = 80, space: dict[str, tuple[str, float, float]] | None = None,
                   out_yaml: str | Path | None = None, seed: int = 0,
                   min_diff_pct: float = 1.0, max_diff_pct: float = 18.0) -> dict[str, Any]:
    """Sample ``n_trials`` parameter sets around ``champion_path`` and emit the single best candidate."""
    champ_path = Path(champion_path)
    base_raw = yaml.safe_load(champ_path.read_text())
    _, champ_params, champ_sid = load_strategy(champ_path)
    counters_path = (champ_path.parent / base_raw["counters"]) if base_raw.get("counters") else None
    counters = load_counters(counters_path)

    valued_states = load_valued_states(match_dirs, cards, arena)
    if not valued_states:
        raise ValueError("no recorded decision states found to run offline counterfactual search")

    champ_pol = CRPolicy(cards, arena, champ_params)
    champ_choices = []
    for st, _ in valued_states:
        champ_pol.reset_match()
        champ_choices.append(_choice_key(champ_pol.decide(copy.copy(st))))

    search_space = space or OFFLINE_SPACE
    rng = random.Random(seed)
    best_trial: dict[str, Any] | None = None
    best_spec_dict: dict[str, Any] | None = None

    for trial_i in range(n_trials):
        cand_dict = copy.deepcopy(base_raw)
        # Perturb 2..5 parameters per trial (local coordinate + multi-knob exploration) so changes stay targeted.
        k_knobs = min(len(search_space), rng.randint(2, 5))
        chosen_keys = rng.sample(sorted(search_space.keys()), k_knobs)
        sampled: dict[str, float | int] = {}
        for key in chosen_keys:
            kind, lo, hi = search_space[key]
            val = rng.randint(int(lo), int(hi)) if kind == "int" else rng.uniform(lo, hi)
            _set_dotted(cand_dict, key, val)
            sampled[key] = round(val, 4) if isinstance(val, float) else val
        spec = StrategySpec(**cand_dict)
        cand_pol = CRPolicy(cards, arena, Params(spec, counters))
        metrics = evaluate_counterfactual(
            champ_choices, cand_pol, valued_states,
            min_diff_pct=min_diff_pct, max_diff_pct=max_diff_pct,
        )
        rec = {"trial": trial_i, "knobs": sampled, **metrics}
        if best_trial is None or rec["objective"] > best_trial["objective"]:
            best_trial = rec
            best_spec_dict = cand_dict

    assert best_trial is not None and best_spec_dict is not None
    written_path = None
    if out_yaml is not None:
        out_p = Path(out_yaml)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        best_spec_dict["parent"] = base_raw["strategy"]
        best_spec_dict["strategy"] = out_p.stem
        out_p.write_text(yaml.safe_dump(best_spec_dict, sort_keys=False))
        written_path = str(out_p)
    return {
        "champion": base_raw["strategy"],
        "champion_id": champ_sid,
        "trials": n_trials,
        "states_evaluated": len(valued_states),
        "best": best_trial,
        "candidate_yaml": written_path,
    }
