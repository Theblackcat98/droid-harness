"""Label recorded Clash Royale matches for the learned ranker (M8).

For each ``data/matches/*/decisions.jsonl`` row that actually produced a play, emit the
compact state vector the live ranker will see, the demonstrated card+tile, and the match
outcome. A my_play event is the proof a decision became an action (a tap that missed is
not a demonstration); old recordings without events fall back to ``chosen_card``.

Outcome weighting: wins 1.0, losses 0.3 (learn mostly from wins), draws 0.5. Ticks where
the hand is missing or all ``None`` are skipped: the live policy never acts on them, so
they are not demonstrations.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from .train_ranker import encode_state

PLAY_MATCH_WINDOW_MS = 2000
WEIGHTS = {"win": 1.0, "loss": 0.3, "draw": 0.5}


def _advantage_weight(base_weight: float, credit: float | None, *, min_credit: float | None = None) -> float | None:
    """Scale demonstration weight by engagement credit (Advantage-Weighted / Filtered BC)."""
    if credit is None:
        return base_weight
    if min_credit is not None and credit < min_credit:
        return None
    # Smoothly map engagement credit (-6..+6) into [0.15, 1.85] multiplier around 1.0.
    adv = 1.0 / (1.0 + math.exp(-max(-10.0, min(10.0, credit / 3.0))))
    return round(base_weight * (0.15 + 1.70 * adv), 6)


def resolve_deck(deck: Sequence[str] | None = None, strategy: str | Path | None = None) -> list[str]:
    if deck:
        return [str(c) for c in deck]
    if strategy:
        spec = yaml.safe_load(Path(strategy).read_text()) or {}
        cards = spec.get("deck")
        if not cards:
            raise ValueError(f"{strategy} has no deck")
        return [str(c) for c in cards]
    raise ValueError("pass deck=[...] or strategy=<strategy.yaml> to fix the card order")


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _match_dirs(root: str | Path) -> list[Path]:
    root = Path(root)
    if (root / "summary.json").exists():
        return [root]
    return sorted(p for p in root.iterdir() if (p / "summary.json").exists())


def _same_tile(a, b) -> bool:
    try:
        return abs(float(a[0]) - float(b[0])) < 1e-3 and abs(float(a[1]) - float(b[1])) < 1e-3
    except (TypeError, ValueError, IndexError):
        return False


def _chosen_index(candidates: list[dict], card: str, tile) -> int | None:
    for i, c in enumerate(candidates):
        if c.get("card") == card and _same_tile(c.get("tile"), tile):
            return i
    same_card = [i for i, c in enumerate(candidates) if c.get("card") == card]
    return same_card[0] if len(same_card) == 1 else None


def _compact_candidates(raw: object) -> list[dict]:
    out = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict) or not c.get("card"):
            continue
        out.append({"card": c["card"], "tile": c.get("tile"), "rule_id": c.get("rule_id"),
                    "score": c.get("score")})
    return out


def _hand_ok(state: dict) -> bool:
    hand = state.get("hand")
    return isinstance(hand, (list, tuple)) and any(c is not None and c != "" for c in hand)


def _match_credits(match_dir: Path, cards: Any | None, arena: Any | None) -> dict[int, float]:
    if cards is None:
        return {}
    from ..domains.clash_royale.arena import Arena as CRArena
    from ..domains.clash_royale.cards import CardDB
    from ..flywheel.engagements import segment
    from ..flywheel.timeline import load_match

    cdb = cards if isinstance(cards, CardDB) else CardDB.load(cards)
    ar = arena if arena is not None else CRArena()
    tl = load_match(match_dir)
    credits: dict[int, float] = {}
    for eng in segment(tl, ar, cdb):
        for seq, val in eng.credit.items():
            credits[seq] = credits.get(seq, 0.0) + val
    return credits


def label_match(match_dir: str | Path, deck: Sequence[str], *, version: int = 1,
                cards: Any | None = None, arena: Any | None = None,
                min_credit: float | None = None) -> tuple[list[dict], dict]:
    d = Path(match_dir)
    try:
        summary = json.loads((d / "summary.json").read_text())
    except (OSError, ValueError):
        return [], {"matches": 0, "skipped_matches": 1}
    result = summary.get("result")
    weight = WEIGHTS.get(str(result))
    if summary.get("status") != "completed" or weight is None:
        return [], {"matches": 0, "skipped_matches": 1}

    deck = list(deck)
    credits = _match_credits(d, cards, arena)
    my_c, opp_c = summary.get("my_crowns"), summary.get("opp_crowns")
    numeric = (int, float)
    delta = (my_c - opp_c) if isinstance(my_c, numeric) and isinstance(opp_c, numeric) else None
    events = [e for e in summary.get("events") or [] if isinstance(e, dict)]
    plays = [(i, e) for i, e in enumerate(events)
             if e.get("kind") == "my_play" and e.get("card")]
    used_plays: set[int] = set()
    rows: list[dict] = []
    skipped = {"no_hand": 0, "no_action": 0, "unplayed": 0, "no_candidates": 0}

    for dec in _jsonl(d / "decisions.jsonl"):
        state = dec.get("state") or {}
        if not _hand_ok(state):
            skipped["no_hand"] += 1
            continue
        card = dec.get("chosen_card")
        if not card:
            skipped["no_action"] += 1
            continue
        if plays:
            played = None
            for i, e in plays:
                if i in used_plays or e.get("card") != card:
                    continue
                dt = float(e.get("t_ms", 0)) - float(dec.get("t_ms", 0))
                if 0 <= dt <= PLAY_MATCH_WINDOW_MS:
                    played = i
                    break
            if played is None:
                skipped["unplayed"] += 1
                continue
            used_plays.add(played)

        seq = dec.get("seq")
        credit = credits.get(int(seq)) if (credits and isinstance(seq, int)) else None
        row_weight = _advantage_weight(weight, credit, min_credit=min_credit)
        if row_weight is None:
            skipped.setdefault("low_credit", 0)
            skipped["low_credit"] += 1
            continue

        candidates = _compact_candidates(dec.get("candidates"))
        chosen = _chosen_index(candidates, card, dec.get("tile"))
        if chosen is None:
            # A candidate list without an unambiguous positive would teach the model
            # that every candidate is a negative.
            candidates = []
            skipped["no_candidates"] += 1

        row = {
            "match_id": summary.get("match_id", d.name),
            "t_ms": dec.get("t_ms"),
            "result": result,
            "my_crowns": my_c,
            "opp_crowns": opp_c,
            "crowns_delta": delta,
            "weight": row_weight,
            "deck": deck,
            "state": [round(float(v), 6) for v in encode_state(state, deck, version=version)],
            "action": {"card": card, "tile": dec.get("tile"), "rule_id": dec.get("chosen_rule")},
            "candidates": candidates,
            "chosen_index": chosen if chosen is not None else -1,
        }
        if version >= 2:
            row["version"] = int(version)
        if credit is not None:
            row["engagement_credit"] = round(float(credit), 4)
        rows.append(row)
    return rows, {"matches": 1, "skipped_matches": 0, "skipped": skipped}


def label_from_video(matches: str | Path, out: str | Path, *,
                     deck: Sequence[str] | None = None,
                     strategy: str | Path | None = None,
                     version: int = 1,
                     cards: Any | None = None,
                     arena: Any | None = None,
                     min_credit: float | None = None) -> dict:
    """Write labeled JSONL from one match dir or a root of ``match_id/`` dirs."""
    deck = resolve_deck(deck, strategy)
    all_rows: list[dict] = []
    stats = {"matches": 0, "skipped_matches": 0,
             "skipped": {"no_hand": 0, "no_action": 0, "unplayed": 0, "no_candidates": 0}}
    for d in _match_dirs(matches):
        rows, st = label_match(d, deck, version=version, cards=cards, arena=arena, min_credit=min_credit)
        all_rows.extend(rows)
        stats["matches"] += st.get("matches", 0)
        stats["skipped_matches"] += st.get("skipped_matches", 0)
        for k, v in st.get("skipped", {}).items():
            stats["skipped"][k] = stats["skipped"].get(k, 0) + v

    all_rows.sort(key=lambda r: (str(r["match_id"]), float(r.get("t_ms") or 0)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in all_rows))
    stats.update({"rows": len(all_rows), "out": str(out), "deck": deck,
                  "weights": {str(k): v for k, v in WEIGHTS.items()}})
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="label_from_video")
    ap.add_argument("--matches", default="data/matches", help="match dir or root of match dirs")
    ap.add_argument("--out", default="data/ranker/labels.jsonl")
    ap.add_argument("--strategy", default=None, help="strategy YAML that fixes the deck order")
    ap.add_argument("--deck", nargs="+", default=None, help="explicit card order")
    ap.add_argument("--version", type=int, default=2, choices=[1, 2],
                    help="state schema version (2 includes board threat & opponent cycle features)")
    ap.add_argument("--cards", default=None, help="cards.json for engagement-credit advantage weighting")
    ap.add_argument("--min-credit", dest="min_credit", type=float, default=None,
                    help="drop demonstrations whose engagement credit is below this threshold (Filtered BC)")
    a = ap.parse_args(argv)
    stats = label_from_video(a.matches, a.out, deck=a.deck, strategy=a.strategy,
                             version=a.version, cards=a.cards, min_credit=a.min_credit)
    print(f"wrote {stats['rows']} rows from {stats['matches']} matches to {stats['out']} "
          f"(skipped: {stats['skipped']})")


if __name__ == "__main__":
    main()
