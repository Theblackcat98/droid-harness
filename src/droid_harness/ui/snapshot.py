"""One snapshot schema for the live view and the replay inspector.

A snapshot is a JSON-ready dict describing one moment of a match: HUD numbers, opponent
cycle, units, the ranked candidate list with its score breakdown, latency, and (live only)
raw detector boxes and recent touches. ``snapshot_from_record`` builds it from a
``decisions.jsonl`` record; the live publisher turns the running policy's ``Decision`` into
the same record first (``Decision.to_record``), so both paths go through one function.

Nothing here runs on the brain thread.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..domains.clash_royale.arena import Arena
from ..domains.clash_royale.perception.grid import Layout, Roi, TileMapper
from ..domains.clash_royale.state import GameState

SCHEMA = "droid-harness-studio/1"
MAX_CANDIDATES = 12

# Display-only fallback when the local card DB (configs/domains/clash_royale/cards.json,
# gitignored) is missing: elixir costs of cards seen on the low ladder. Never used by the policy.
FALLBACK_COSTS: dict[str, int] = {
    "knight": 3, "archers": 3, "minions": 3, "arrows": 3, "fireball": 4, "giant": 5, "musketeer": 4,
    "mini_pekka": 4, "goblins": 2, "spear_goblins": 2, "skeletons": 1, "skeleton_army": 3,
    "tombstone": 3, "goblin_cage": 4, "goblin_hut": 5, "bomber": 2, "valkyrie": 4, "baby_dragon": 4,
    "barbarians": 5, "cannon": 3, "prince": 5, "hog_rider": 4, "minion_horde": 5, "the_log": 2,
    "zap": 2, "ice_spirit": 1, "fire_spirit": 1, "electro_spirit": 1, "battle_ram": 4, "wizard": 5,
    "witch": 5, "pekka": 7, "balloon": 5, "inferno_tower": 5, "mega_minion": 3, "bats": 2,
    "ice_golem": 2, "royal_giant": 6, "goblin_barrel": 3, "princess": 3, "rocket": 6,
    "earthquake": 3, "barbarian_barrel": 2, "firecracker": 3, "tesla": 4, "bomb_tower": 4,
    "mortar": 4, "x_bow": 6, "lightning": 6, "poison": 4, "golem": 8, "lava_hound": 7, "miner": 3,
    "graveyard": 5, "electro_wizard": 4, "dark_prince": 4, "lumberjack": 4, "bandit": 3,
    "royal_recruits": 7, "wall_breakers": 2, "elixir_collector": 6, "furnace": 4, "guards": 3,
    "dart_goblin": 3, "hunter": 4, "executioner": 5, "fisherman": 3, "goblin_gang": 3,
}


class CostBook:
    """``card in book`` / ``book.cost(card)`` over the real CardDB, else the fallback table."""

    def __init__(self, cards=None) -> None:
        self.cards = cards

    def __contains__(self, key: str) -> bool:
        return (self.cards is not None and key in self.cards) or key in FALLBACK_COSTS

    def cost(self, key: str) -> float:
        if self.cards is not None and key in self.cards:
            return float(self.cards.cost(key))
        return float(FALLBACK_COSTS.get(key, 3))

    def get(self, key: str) -> float | None:
        return self.cost(key) if key in self else None

    @classmethod
    def load(cls, path: str | Path | None) -> CostBook:
        if path and Path(path).exists():
            from ..domains.clash_royale.cards import CardDB

            try:
                return cls(CardDB.load(path))
            except Exception:  # a broken local DB must not stop the inspector
                return cls(None)
        return cls(None)


# --- static geometry ------------------------------------------------------------------------

def _roi(r: Roi | None) -> list[float] | None:
    return None if r is None else [round(r.x0, 4), round(r.y0, 4), round(r.x1, 4), round(r.y1, 4)]


def geometry(layout: Layout | None, tiles: TileMapper | None, arena: Arena | None) -> dict[str, Any]:
    """Everything the page needs to draw overlays: ROIs (fractions of the frame), the
    tile -> normalized-screen homography, and the arena model in tiles."""
    arena = arena or Arena()
    out: dict[str, Any] = {
        "arena": {"width": arena.width, "height": arena.height, "river_y": list(arena.river_y),
                  "bridges_x": list(arena.bridges_x), "tower_radius": arena.tower_radius,
                  "towers": {k: list(v) for k, v in arena.towers.items()}},
        "tile_to_screen": tiles.Hinv.tolist() if tiles is not None else None,
        "rois": None,
    }
    if layout is not None:
        out["rois"] = {
            "hand": [_roi(r) for r in layout.hand], "next_card": _roi(layout.next_card),
            "elixir_bar": _roi(layout.elixir_bar), "clock": _roi(layout.clock), "arena": _roi(layout.arena),
            "towers": {k: _roi(v) for k, v in layout.towers.items()},
            "tower_hp": {k: _roi(v) for k, v in (layout.tower_hp or {}).items()},
            "buttons": {k: _roi(v) for k, v in layout.buttons.items()},
        }
    return out


def device_geometry(device: str | None, root: Path = Path(".")) -> dict[str, Any]:
    """Geometry for a recorded match's device, from the checked-in calibration if present."""
    cr = root / "configs/domains/clash_royale"
    layout = tiles = arena = None
    if device == "sim":  # `droid-harness sim` recordings use the synthetic scene's geometry
        from ..sim.cr_scene import sim_layout, sim_tiles

        return geometry(sim_layout(), sim_tiles(), Arena())
    if device:
        try:
            p = cr / "rois" / f"{device}.yaml"
            layout = Layout.load(p) if p.exists() else None
        except Exception:
            layout = None
        try:
            p = cr / "tiles" / f"{device}.yaml"
            tiles = TileMapper.load(p) if p.exists() else None
        except Exception:
            tiles = None
        try:
            p = cr / "arena" / f"{device}.yaml"
            arena = Arena.load(p) if p.exists() else None
        except Exception:
            arena = None
    return geometry(layout, tiles, arena)


# --- strategies -----------------------------------------------------------------------------

class StrategyIndex:
    """strategy_id -> (name, Params) for the strategy files on disk.

    IDs hash the policy source too, so matches recorded under older policy code don't resolve;
    the scoreboard then derives ``weight × posture`` from ``total / score`` instead.
    """

    def __init__(self, directory: str | Path = "configs/strategies") -> None:
        self.directory = Path(directory)
        self._by_id: dict[str, tuple[str, Any, str]] | None = None

    def _build(self) -> dict[str, tuple[str, Any, str]]:
        from ..domains.clash_royale.policy.strategy import load_strategy

        out: dict[str, tuple[str, Any, str]] = {}
        for p in sorted([*self.directory.glob("*.yaml"), *self.directory.glob("auto/*.yaml")]):
            try:
                spec, params, sid = load_strategy(p)
            except Exception:
                continue
            out[sid] = (spec.strategy, params, str(p))
        return out

    def get(self, sid: str | None, name: str | None = None) -> tuple[str, Any, str] | None:
        """By current ID; else by the recorded ``strategy_name``; else via ``legacy_ids.yaml``."""
        if self._by_id is None:
            self._by_id = self._build()
        hit = self._by_id.get(sid) if sid else None
        if hit is None and not name and sid:
            name = self._legacy().get(sid)
        if hit is None and name:
            hit = next((v for v in self._by_id.values() if v[0] == name), None)
        return hit

    def _legacy(self) -> dict[str, str]:
        if not hasattr(self, "_legacy_map"):
            import yaml

            p = self.directory / "legacy_ids.yaml"
            try:
                self._legacy_map = {str(k): str(v) for k, v in (yaml.safe_load(p.read_text()) or {}).items()}
            except (OSError, ValueError, AttributeError):
                self._legacy_map = {}
        return self._legacy_map

    def add(self, sid: str, name: str, params: Any, path: str = "") -> None:
        if self._by_id is None:
            self._by_id = self._build()
        self._by_id[sid] = (name, params, path)


def strategy_summary(name: str | None, sid: str | None, params: Any | None, path: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {"id": sid, "name": name, "path": path, "known": params is not None}
    if params is not None:
        out.update({
            "deck": list(params.deck), "win_conditions": list(params.win_conditions),
            "act_threshold": params.act_threshold, "weights": dict(params.weights),
            "posture_multipliers": {k: dict(v) for k, v in params.posture_multipliers.items()},
        })
    return out


# --- per-moment pieces ------------------------------------------------------------------------

def _round(v: Any, n: int = 2) -> Any:
    if isinstance(v, float):
        return round(v, n)
    if isinstance(v, (list, tuple)):
        return [_round(x, n) for x in v]
    return v


def opponent_cycle(st: GameState, costs: CostBook) -> dict[str, Any]:
    """The 8-card widget: which seen cards are (possibly) in hand, which are queued and how many
    plays / how much elixir until each returns, and which could be played right now."""
    last = list(st.opp_last_four[-4:])
    hi = st.opp_elixir[1] if st.opp_elixir else 0.0
    cards = []
    for card in st.opp_deck_seen:
        until = st.opp_cards_until_back(card)
        cost = costs.get(card)
        in_hand = until == 0
        cards.append({
            "card": card, "cost": cost, "in_hand": in_hand, "until_back": until,
            "queue_pos": last.index(card) if card in last else None,
            "min_elixir_to_return": round(st.opp_min_cycle_elixir_to_return(card, costs), 1) if not in_hand else 0.0,
            "can_play_now": bool(in_hand and cost is not None and hi >= cost - 0.5),
        })
    return {"deck_known": len(st.opp_deck_seen) >= 8, "unknown_slots": max(0, 8 - len(st.opp_deck_seen)),
            "last_four": last, "cards": cards}


def candidate_rows(record: dict, params: Any | None, *, limit: int = MAX_CANDIDATES) -> dict[str, Any]:
    """Ranked candidates with ``score × weight × posture = total`` and the chosen one marked."""
    posture = record.get("posture") or (record.get("state") or {}).get("posture")
    chosen_key = (record.get("chosen_rule"), record.get("chosen_card"), _tile_key(record.get("tile")))
    rows = []
    for c in record.get("candidates") or []:
        rule = c.get("rule_id") or ""
        family = rule.split(".", 1)[0]
        score, total = c.get("score"), c.get("total")
        weight = mult = None
        if params is not None:
            weight = params.weight(rule)
            mult = params.posture_multipliers.get(posture, {}).get(family, 1.0)
            if total is None and score is not None:
                total = score * weight * mult
        rows.append({
            "rule_id": rule, "family": family, "card": c.get("card"), "slot": c.get("slot"),
            "tile": _round(c.get("tile")), "predicted_tile": _round(c.get("predicted_tile")),
            "score": _round(score, 3), "weight": weight, "posture_mult": mult,
            "wm": _round(weight * mult, 3) if weight is not None and mult is not None
            else (_round(total / score, 3) if score and total is not None else None),
            "total": _round(total, 3), "reason": c.get("reason", ""), "execute_at_ms": c.get("execute_at_ms"),
            "preselect": bool(c.get("preselect")),
            "chosen": (rule, c.get("card"), _tile_key(c.get("tile"))) == chosen_key and chosen_key[0] is not None,
        })
    rows.sort(key=lambda r: (r["total"] if r["total"] is not None else float("-inf")), reverse=True)
    if limit and len(rows) > limit:
        keep = rows[:limit]
        keep += [r for r in rows[limit:] if r["chosen"]]
        rows = keep
    return {
        "posture": posture, "acted": chosen_key[0] is not None,
        "chosen_rule": record.get("chosen_rule"), "chosen_card": record.get("chosen_card"),
        "tile": _round(record.get("tile")), "predicted_tile": _round(record.get("predicted_tile")),
        "execute_at_ms": record.get("execute_at_ms"),
        "threshold": params.act_threshold if params is not None else None,
        "n_candidates": len(record.get("candidates") or []), "candidates": rows,
        "intent": record.get("intent"),
    }


def _tile_key(t: Any) -> tuple | None:
    if t is None:
        return None
    try:
        return (round(float(t[0]), 2), round(float(t[1]), 2))
    except (TypeError, IndexError, ValueError):
        return None


def snapshot_from_record(record: dict, *, params: Any | None = None, costs: CostBook | None = None,
                         tower_max: dict[str, int] | None = None, extra: dict | None = None) -> dict[str, Any]:
    """A ``decisions.jsonl`` record (or a live ``Decision.to_record()``) -> snapshot."""
    costs = costs or CostBook(None)
    sd = record.get("state") or {}
    st = GameState.from_digest(sd) if "t_ms" in sd else GameState(t_ms=int(record.get("t_ms") or 0))
    towers = dict(st.towers)
    hand = []
    fresh = list(sd.get("hand_fresh") or [])
    for i, card in enumerate(st.hand):
        cost = costs.get(card) if card else None
        hand.append({"slot": i, "card": card, "cost": cost,
                     "fresh": (fresh[i] if i < len(fresh) else True) if card else False,
                     "affordable": bool(card and cost is not None and cost <= st.my_elixir + 1e-6)})
    warnings = list(sd.get("warnings") or [])
    snap = {
        "schema": SCHEMA,
        "t_ms": st.t_ms,
        "phase": st.phase,
        "posture": st.posture,
        "elixir": {"me": _round(st.my_elixir), "opp": _round(list(st.opp_elixir))},
        "towers": towers,
        "tower_max": dict(tower_max or {}),
        "crowns": list(st.crowns),
        "pocket_open": dict(st.pocket_open),
        "hand": hand,
        "next": st.next_card,
        "units": [{"id": u.track_id, "kind": u.kind, "side": u.side, "tile": _round(list(u.tile)),
                   "vel": _round(list(u.vel)), "hp": _round(u.hp_pct), "conf": _round(u.conf, 2)}
                  for u in st.units],
        "opp": opponent_cycle(st, costs),
        "decision": candidate_rows(record, params),
        "latency": {k: _round(v, 2) for k, v in (record.get("lat") or {}).items()},
        "warnings": sorted(set(warnings)),
        "seq": record.get("seq"),
    }
    if extra:
        snap.update(extra)
    return snap


def running_tower_max(records: Iterable[dict]) -> list[dict[str, int]]:
    """Per record, the highest HP seen so far for each tower (the HP bar's denominator)."""
    cur: dict[str, int] = {}
    out = []
    for r in records:
        for k, v in ((r.get("state") or {}).get("towers") or {}).items():
            if isinstance(v, (int, float)) and v > cur.get(k, 0):
                cur[k] = int(v)
        out.append(dict(cur))
    return out
