"""Speed 1, targeted (todo R5): remove one named failure pattern, change little else.

The value-weighted search (``offline_study.py``) rewards a candidate for changing decisions
inside losing engagements, without knowing whether the new decision is any better. This
search asks a question the logs can answer:

1. A **pattern** is a predicate over one decision on one logged state, mirroring the digest's
   failure clusters (``opening_throw``, ``bad_counter_timing``, ``over_defend_bleed``,
   ``wincon_starved``, ``spell_inefficiency``).
2. The champion is replayed on every logged state; its pattern hits are the target.
3. Each trial changes **one or two** parameters from that pattern's knob list (one hypothesis),
   replays the same states, and counts the hits removed and the **collateral**: decisions that
   changed on states where the champion's decision was not a hit.
4. A candidate passes when it removes at least ``min_removed`` of the hits with collateral at or
   below ``max_collateral_pct``. The best passing candidate is written as a new strategy file
   (never an edit) for the live paired test.

No phone, no LLM, CPU seconds to minutes.
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ...domains.clash_royale.arena import Arena
from ...domains.clash_royale.cards import CardDB
from ...domains.clash_royale.policy.strategy import Params, StrategySpec, load_counters, load_strategy
from ...domains.clash_royale.policy.utility import CRPolicy
from ...domains.clash_royale.state import GameState

HARD_COUNTERS = ("mini_pekka", "goblin_cage", "tombstone", "cannon", "minion_horde", "skeleton_army",
                 "barbarians", "prince", "pekka", "inferno_tower")


@dataclass
class Choice:
    rule: str | None
    card: str | None
    tile: tuple[float, float] | None
    score: float | None

    @property
    def key(self):
        return (self.rule, self.card, self.tile)


def _choice(decision) -> Choice:
    c = getattr(decision, "chosen", None)
    if c is None:
        return Choice(None, None, None, None)
    return Choice(c.rule_id, c.card, tuple(round(v, 2) for v in c.tile) if c.tile else None, c.score)


# --- patterns: (state, choice, params, cards) -> bool ---------------------------------------

def _cost(cards: CardDB, key: str | None) -> float:
    return float(cards.cost(key)) if key and key in cards else 3.0


def opening_throw(st: GameState, ch: Choice, p, cards: CardDB) -> bool:
    """Early play that leaves < 3.2 elixir while they may hold ≥ 6.5."""
    if ch.card is None or st.t_ms >= 15_000:
        return False
    return st.my_elixir - _cost(cards, ch.card) < 3.2 and st.opp_elixir[1] >= 6.5


def bad_counter_timing(st: GameState, ch: Choice, p, cards: CardDB) -> bool:
    """Win condition played while a known hard counter is in their hand and affordable."""
    if ch.card is None or ch.card not in p.win_conditions:
        return False
    counters = set(p.wc_counters or HARD_COUNTERS)
    last4 = set(st.opp_last_four[-4:])
    return any(c in st.opp_deck_seen and c not in last4 and c in cards and st.opp_elixir[1] >= cards.cost(c) - 0.5
               for c in counters)


def over_defend_bleed(st: GameState, ch: Choice, p, cards: CardDB) -> bool:
    """Defending a small push down to < 2 elixir."""
    if ch.rule is None or not ch.rule.startswith(("defend.", "t0.emergency")):
        return False
    if st.my_elixir - _cost(cards, ch.card) >= 2.0:
        return False
    threat_value = sum(cards.unit_value(u.kind) * u.hp_pct for u in st.enemy_units())
    return threat_value < 3.0


def wincon_starved(st: GameState, ch: Choice, p, cards: CardDB) -> bool:
    """Near-full elixir, a win condition in hand, a quiet board, and no win-condition play."""
    if st.my_elixir < 8.5 or st.enemy_units():
        return False
    ready = [c for i, c in enumerate(st.hand) if c in p.win_conditions and st.slot_fresh(i)]
    return bool(ready) and ch.card not in p.win_conditions


def spell_inefficiency(st: GameState, ch: Choice, p, cards: CardDB) -> bool:
    """A spell cast whose own value estimate is below its cost."""
    return bool(ch.rule and ch.rule.startswith("spell.") and ch.score is not None and ch.score < 0.0)


PATTERNS: dict[str, Callable[..., bool]] = {
    "opening_throw": opening_throw, "bad_counter_timing": bad_counter_timing,
    "over_defend_bleed": over_defend_bleed, "wincon_starved": wincon_starved,
    "spell_inefficiency": spell_inefficiency,
}

# Knobs that plausibly move each pattern (dotted path, low, high).
KNOBS: dict[str, list[tuple[str, float, float]]] = {
    "opening_throw": [("params.opening_ms", 10_000, 25_000), ("params.giant_safe_build_elixir", 8.0, 9.8),
                      ("params.giant_x1_min_elixir", 7.5, 9.8), ("params.defense_reserve", 2.0, 4.0),
                      ("params.reserve_penalty", 0.4, 0.9), ("posture_multipliers.OPENING.offense", 0.2, 0.7),
                      ("params.act_threshold", 0.06, 0.2)],
    "bad_counter_timing": [("params.counter_in_hand_mult", 0.25, 0.7), ("params.counter_out_mult", 1.2, 1.8),
                           ("params.max_opp_elixir_to_bridge", 4.0, 6.5),
                           ("params.min_elixir_advantage_to_bridge", 1.0, 3.5)],
    "over_defend_bleed": [("params.ignore_chip_tower_hp_frac", 0.4, 0.8), ("params.ignore_chip_max_threat_value", 1.0, 3.0),
                          ("params.defend_trigger_tiles", 9.0, 13.0), ("params.defend_horizon_s", 5.5, 9.5),
                          ("params.elixir_weight", 0.3, 0.8)],
    "wincon_starved": [("params.min_elixir_to_attack", 4.5, 7.0), ("params.giant_back_build_score", 0.5, 0.9),
                       ("params.giant_safe_build_elixir", 8.0, 9.2), ("params.max_opp_elixir_to_bridge", 5.0, 7.5),
                       ("posture_multipliers.PRESS.offense", 1.2, 1.8), ("params.act_threshold", 0.05, 0.15)],
    "spell_inefficiency": [("params.spell_value_threshold", 3.0, 6.0), ("weights.spell.value", 0.5, 1.0),
                           ("weights.spell.tower_chip", 0.3, 0.9)],
}


def load_states(match_dirs: list[Path], max_states: int = 6000) -> list[GameState]:
    out: list[GameState] = []
    for d in match_dirs:
        p = d / "decisions.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                st = json.loads(line).get("state")
            except ValueError:
                continue
            if isinstance(st, dict) and "t_ms" in st:
                out.append(GameState.from_digest(st))
    if len(out) > max_states:
        step = len(out) / max_states
        out = [out[int(i * step)] for i in range(max_states)]
    return out


def choices(policy: CRPolicy, states: list[GameState]) -> list[Choice]:
    res = []
    for st in states:
        policy.reset_match()
        res.append(_choice(policy.decide(copy.copy(st))))
    return res


def pattern_hits(states, chs, pattern: str, params, cards) -> list[bool]:
    f = PATTERNS[pattern]
    return [f(st, ch, params, cards) for st, ch in zip(states, chs, strict=True)]


def _set(d: dict, dotted: str, v: Any) -> None:
    keys = dotted.split(".")
    if keys[0] == "weights" and len(keys) > 2:  # rule ids contain dots: weights.<family>.<name>
        keys = ["weights", ".".join(keys[1:])]
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = round(v, 4) if isinstance(v, float) else v


def scan_patterns(champion: str | Path, match_dirs: list[Path], cards: CardDB, arena: Arena, *,
                  max_states: int = 6000) -> dict[str, int]:
    """How often the champion's own decisions match each pattern on the logged states."""
    _, params, _ = load_strategy(champion)
    states = load_states(match_dirs, max_states)
    chs = choices(CRPolicy(cards, arena, params), states)
    return {name: sum(pattern_hits(states, chs, name, params, cards)) for name in PATTERNS}


def search_targeted(champion: str | Path, match_dirs: list[Path], cards: CardDB, arena: Arena, *,
                    pattern: str = "auto", trials: int = 40, max_knobs: int = 2, seed: int = 0,
                    min_removed: float = 0.5, max_collateral_pct: float = 3.0, max_states: int = 6000,
                    out_yaml: str | Path | None = None) -> dict[str, Any]:
    champ_path = Path(champion)
    raw = yaml.safe_load(champ_path.read_text())
    _, champ_params, champ_sid = load_strategy(champ_path)
    counters = load_counters((champ_path.parent / raw["counters"]) if raw.get("counters") else None)
    states = load_states(match_dirs, max_states)
    if not states:
        raise ValueError("no logged decision states under the given match directories")
    champ_ch = choices(CRPolicy(cards, arena, champ_params), states)
    counts = {name: sum(pattern_hits(states, champ_ch, name, champ_params, cards)) for name in PATTERNS}
    if pattern == "auto":
        pattern = max(counts, key=counts.get)
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern {pattern!r}; one of {sorted(PATTERNS)}")
    hits0 = pattern_hits(states, champ_ch, pattern, champ_params, cards)
    n_hits0 = sum(hits0)
    result: dict[str, Any] = {"champion": raw.get("strategy"), "champion_id": champ_sid, "states": len(states),
                              "pattern": pattern, "champion_hits": counts, "trials": [], "best": None,
                              "candidate_yaml": None}
    if n_hits0 == 0:
        result["note"] = f"the champion never matches {pattern} on these states; nothing to fix"
        return result
    rng = random.Random(seed)
    knobs = KNOBS[pattern]
    best = None
    for t in range(trials):
        cand = copy.deepcopy(raw)
        chosen = rng.sample(knobs, min(len(knobs), rng.randint(1, max_knobs)))
        sampled = {}
        for path, lo, hi in chosen:
            v = rng.uniform(lo, hi)
            v = int(round(v)) if path.endswith("_ms") else v
            _set(cand, path, v)
            sampled[path] = round(v, 4) if isinstance(v, float) else v
        try:
            params = Params(StrategySpec(**cand), counters)
        except Exception as e:  # a knob that doesn't apply to this strategy
            result["trials"].append({"trial": t, "knobs": sampled, "error": str(e)})
            continue
        ch = choices(CRPolicy(cards, arena, params), states)
        hits = pattern_hits(states, ch, pattern, params, cards)
        removed = sum(1 for h0, h in zip(hits0, hits, strict=True) if h0 and not h)
        added = sum(1 for h0, h in zip(hits0, hits, strict=True) if h and not h0)
        collateral = sum(1 for h0, a, b in zip(hits0, champ_ch, ch, strict=True) if not h0 and a.key != b.key)
        rec = {"trial": t, "knobs": sampled, "hits": sum(hits), "removed": removed, "added": added,
               "removed_frac": round(removed / n_hits0, 3),
               "collateral_pct": round(100.0 * collateral / len(states), 2)}
        rec["passes"] = rec["removed_frac"] >= min_removed and rec["collateral_pct"] <= max_collateral_pct
        rec["objective"] = round((removed - added) / n_hits0 - rec["collateral_pct"] / max(max_collateral_pct, 1e-6), 4)
        result["trials"].append(rec)
        if rec["passes"] and (best is None or rec["objective"] > best[0]["objective"]):
            best = (rec, cand)
    if best is not None:
        rec, cand = best
        result["best"] = rec
        if out_yaml is not None:
            out = Path(out_yaml)
            out.parent.mkdir(parents=True, exist_ok=True)
            cand["parent"] = raw.get("strategy")
            cand["strategy"] = out.stem
            if raw.get("counters"):  # keep pointing at the champion's CSV, relative to the new file
                cand["counters"] = str(Path(_rel(champ_path.parent / raw["counters"], out.parent)))
            header = (f"# Generated by `droid-harness tune-targeted` from {raw.get('strategy')} ({champ_sid}).\n"
                      f"# Pattern: {pattern}. Change: {rec['knobs']}. Removed {rec['removed']}/{n_hits0} "
                      f"champion hits, collateral {rec['collateral_pct']}% of {len(states)} states.\n")
            out.write_text(header + yaml.safe_dump(cand, sort_keys=False))
            result["candidate_yaml"] = str(out)
    result["trials"] = sorted(result["trials"], key=lambda r: r.get("objective", -9), reverse=True)[:10]
    return result


def _rel(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target.resolve(), start.resolve())
