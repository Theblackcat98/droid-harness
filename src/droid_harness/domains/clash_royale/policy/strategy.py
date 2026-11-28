"""Strategy files: versioned YAML params + counter table + rule code, content-hashed.

``strategy_id`` = sha256(policy source code + YAML + counters + model files)[:12]. A
strategy is immutable once it has played a logged match; edits make a child with
``parent`` set.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

POLICY_DIR = Path(__file__).parent


class Tier0Params(BaseModel):
    lethal_margin_hp: int = 30
    # A tower's HP value must have been stable this long before a spell may claim lethal. A
    # single misread (e.g. an undamaged King read as "5") used to fire a 4-elixir spell at 11 s.
    lethal_min_stable_ms: int = 1200
    spell_tower_damage: dict[str, int] = Field(default_factory=dict)  # your levels; overrides the DB
    leak_threshold: float = 9.3
    leak_ms: int = 800
    leak_card_preference: list[str] = Field(default_factory=list)
    leak_tiles: dict[str, list[float]] = Field(default_factory=lambda: {"left": [5.5, 1.5], "right": [12.5, 1.5]})
    emergency_eta_s: float = 1.5
    emergency_hp_frac: float = 0.35


class PlacementParams(BaseModel):
    building_pull: dict[str, list[float]] = Field(default_factory=lambda: {"left": [7.5, 10.5],
                                                                             "right": [10.5, 10.5]})
    ranged_standoff_tiles: float = 3.0
    bridge_tiles: dict[str, list[float]] = Field(default_factory=lambda: {"left": [3.5, 14.0],
                                                                            "right": [14.5, 14.0]})
    back_tiles: dict[str, list[float]] = Field(default_factory=lambda: {"left": [4.5, 1.5],
                                                                          "right": [13.5, 1.5]})
    anti_spell_tower_offset: float = 0.0  # >0 shifts ranged defenders away from our tower vs spells
    melee_kite_center: bool = False       # pull melee threats slightly cross-center into dual-tower fire


class StrategySpec(BaseModel):
    strategy: str
    parent: str | None = None
    deck: list[str]
    win_conditions: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    tier0: Tier0Params = Field(default_factory=Tier0Params)
    placement: PlacementParams = Field(default_factory=PlacementParams)
    weights: dict[str, float] = Field(default_factory=dict)
    posture_multipliers: dict[str, dict[str, float]] = Field(default_factory=dict)
    counters: str | None = None
    learned_ranker: str | None = None
    intent_advisor: bool = False
    advisor_may_set: list[str] = Field(default_factory=list)


DEFAULT_PARAMS: dict[str, Any] = {
    "act_threshold": 0.15,
    "elixir_weight": 0.4,
    "defend_trigger_tiles": 12.0,
    "defend_horizon_s": 8.0,
    "min_elixir_to_attack": 7.0,
    "press_opp_elixir_max": 3.0,
    "spell_value_threshold": 4.0,
    "spell_cycle_max_casts": 3,
    "opening_ms": 30_000,
    "min_action_gap_ms": 350,
    "counter_push_bonus": 0.4,
    "attack_blocked_by_threat_eta_s": 8.0,
    # offense.blind_push (no-detector play; see rules/blind.py). Off unless blind: true.
    "blind": False,
    "blind_lane": "right",
    "blind_switch_every": 2,
    "blind_tanks": [],  # empty = win_conditions
    "blind_push_elixir": 8.0,
    "blind_push_elixir_x2": 6.0,
    "blind_min_elixir": 7.0,
    "blind_min_elixir_x2": 5.0,
    "blind_hold_for_tank_elixir": 9.0,
    "blind_support_ms": 7000,
    "blind_support_back_tiles": 2.0,
    "blind_spell_after_ms": 11000,
    "blind_spell_window_ms": 9000,
    "blind_spell_elixir": 9.5,
    # Giant-beatdown v3 experiments (rules/offense.py). All off by default: one per experiment.
    "counterpush_min_value": 3.0,      # surviving elixir value needed before supporting a push
    "counterpush_tank_ahead": False,   # E3
    "counterpush_tank_bonus": 0.2,
    "counter_gate": False,             # E4
    "wc_counters": [],                 # opponent cards that hard-counter our win condition
    "counter_in_hand_mult": 0.5,
    "counter_out_mult": 1.3,
    "defense_reserve": 0.0,            # E5: elixir to keep after an offensive play (0 = off)
    "reserve_penalty": 0.5,
    "giant_back_build": False,         # E5
    "giant_safe_build_elixir": 9.0,
    "giant_back_build_score": 0.4,
    "giant_build_block_tiles": None,   # None preserves defend_trigger_tiles for legacy strategies
    # Pro card/cost/cycle awareness (opt-in; v1-v3 keep legacy behavior)
    "max_opp_elixir_to_bridge": 10.0,
    "min_elixir_advantage_to_bridge": -10.0,
    "cycle_punish_partial_deck": False,
    "counter_unaffordable_punish_mult": 1.0,
    "back_build_counter_in_hand_mult": None,
    "spell_aware": False,
    "kill_aware_spells": False,
    "save_counters_for_in_hand_threats": False,
    # Pro macro/micro extensions (docs/pro-strategy-research.md; opt-in so v1-v5 stay unchanged)
    "giant_back_build_allow_in_opening": True,
    "giant_x1_min_elixir": 0.0,
    "use_min_cycle_elixir": False,
    "ignore_chip_tower_hp_frac": 1.0,
    "ignore_chip_max_threat_value": 0.0,
    # Cheap-probe discipline: skip answering a <= N-elixir card (source card cost) that is still
    # seconds from a healthy tower while the lane total stays small. 0 = off.
    "ignore_cheap_threat_cost": 0,
    "ignore_cheap_tower_hp_frac": 0.6,
    "ignore_cheap_lane_value": 3.0,
    "opposite_lane_punish": False,
    "punish_min_enemy_cost": 5,
    "punish_enemy_back_y": 23.5,
    "punish_min_my_elixir": 5.0,
    "punish_score": 0.68,
    "punish_cards": [],  # empty = win_conditions + fast melee bridge threats
    # M8 learned ranker blending ("residual" keeps rule safety gates intact; "replace" is legacy)
    "ranker_mode": "residual",
    "ranker_residual_weight": 0.25,
    # Levels (README §7, todo R3; opt-in). Uses configs/accounts/<account>.yaml and the opponent's
    # King Level read from their tower HP: kill-aware spell math and derived spell tower damage
    # scale ~10 %/level; level_gap_respect leans toward defense when under-leveled.
    "level_aware": False,
    "level_gap_respect": 0.0,  # per level of gap: defend ×(1+r·gap), offense ÷(1+r·gap)
    # Counter prior (todo R7; opt-in): a missing counters-CSV row falls back to card attributes.
    "counter_prior": False,
    "counter_prior_scale": 1.0,
    # Confidence & starvation / anti-clumping guards (31-match video audit).
    "spell_min_conf": 0.48,
    "emergency_min_conf": 0.48,
    "bank_for_wc_max_threat": 0.0,      # >0 defers low-threat defense while banking for win condition in hand
    "counterpush_min_elixir": 0.0,      # min elixir required to reinforce a counter-push with support
    "counterpush_bank_for_wc": False,   # defer cheap counterpush support when win condition is in hand & unaffordable
    "avoid_clump_radius": 1.8,          # spread non-tank troops away from already-placed friendly troops
}


class Params:
    """Attribute access over params with defaults; unknown names raise."""

    def __init__(self, spec: StrategySpec, counters: dict[tuple[str, str], float]) -> None:
        self.spec = spec
        self._p = {**DEFAULT_PARAMS, **spec.params}
        self.counters = counters
        self.tier0 = spec.tier0
        self.placement = spec.placement
        self.weights = spec.weights
        self.posture_multipliers = spec.posture_multipliers
        self.advisor_may_set = set(spec.advisor_may_set)
        self.deck = spec.deck
        self.win_conditions = spec.win_conditions

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__["_p"][name]
        except KeyError as e:
            raise AttributeError(name) from e

    def weight(self, rule_id: str) -> float:
        return self.weights.get(rule_id, 1.0)


def load_counters(path: str | Path | None) -> dict[tuple[str, str], float]:
    """CSV with header ``card,threat,effectiveness`` (0..1)."""
    if not path:
        return {}
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["card"].strip(), row["threat"].strip())] = float(row["effectiveness"])
    return out


def policy_source_hash() -> str:
    h = hashlib.sha256()
    for p in sorted(POLICY_DIR.rglob("*.py")):
        h.update(p.relative_to(POLICY_DIR).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()


def strategy_id(yaml_bytes: bytes, counters_bytes: bytes = b"", model_bytes: bytes = b"") -> str:
    h = hashlib.sha256()
    for part in (policy_source_hash().encode(), yaml_bytes, counters_bytes, model_bytes):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:12]


def load_strategy(path: str | Path) -> tuple[StrategySpec, Params, str]:
    path = Path(path)
    raw = path.read_bytes()
    spec = StrategySpec(**yaml.safe_load(raw))
    counters_path = (path.parent / spec.counters) if spec.counters else None
    counters_bytes = counters_path.read_bytes() if counters_path else b""
    model_bytes = (path.parent / spec.learned_ranker).read_bytes() if spec.learned_ranker else b""
    params = Params(spec, load_counters(counters_path))
    return spec, params, strategy_id(raw, counters_bytes, model_bytes)
