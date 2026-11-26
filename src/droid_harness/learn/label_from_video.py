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
from collections.abc import Sequence
from pathlib import Path

import yaml

from .train_ranker import encode_state

PLAY_MATCH_WINDOW_MS = 2000
WEIGHTS = {"win": 1.0, "loss": 0.3, "draw": 0.5}


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


def label_match(match_dir: str | Path, deck: Sequence[str]) -> tuple[list[dict], dict]:
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

        candidates = _compact_candidates(dec.get("candidates"))
        chosen = _chosen_index(candidates, card, dec.get("tile"))
        if chosen is None:
            # A candidate list without an unambiguous positive would teach the model
            # that every candidate is a negative.
            candidates = []
            skipped["no_candidates"] += 1

        rows.append({
            "match_id": summary.get("match_id", d.name),
            "t_ms": dec.get("t_ms"),
            "result": result,
            "my_crowns": my_c,
            "opp_crowns": opp_c,
            "crowns_delta": delta,
            "weight": weight,
            "deck": deck,
            "state": [round(float(v), 6) for v in encode_state(state, deck)],
            "action": {"card": card, "tile": dec.get("tile"), "rule_id": dec.get("chosen_rule")},
            "candidates": candidates,
            "chosen_index": chosen if chosen is not None else -1,
        })
    return rows, {"matches": 1, "skipped_matches": 0, "skipped": skipped}


def label_from_video(matches: str | Path, out: str | Path, *,
                     deck: Sequence[str] | None = None,
                     strategy: str | Path | None = None) -> dict:
    """Write labeled JSONL from one match dir or a root of ``match_id/`` dirs."""
    deck = resolve_deck(deck, strategy)
    all_rows: list[dict] = []
    stats = {"matches": 0, "skipped_matches": 0,
             "skipped": {"no_hand": 0, "no_action": 0, "unplayed": 0, "no_candidates": 0}}
    for d in _match_dirs(matches):
        rows, st = label_match(d, deck)
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
    a = ap.parse_args(argv)
    stats = label_from_video(a.matches, a.out, deck=a.deck, strategy=a.strategy)
    print(f"wrote {stats['rows']} rows from {stats['matches']} matches to {stats['out']} "
          f"(skipped: {stats['skipped']})")


if __name__ == "__main__":
    main()
