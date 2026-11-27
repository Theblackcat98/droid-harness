"""Speed-3 Deterministic Agent Handoff Digest (README §5, "What the digest contains").

Compresses 10-100+ recorded matches into a single ``< 30 KB`` Markdown + JSON dossier
designed to be read by a coding-agent harness (OpenCode / Antigravity) in one turn when
structural strategy changes are needed. Zero LLM calls are made during digest generation.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..domains.clash_royale.arena import Arena
from ..domains.clash_royale.cards import CardDB
from .engagements import Engagement, segment
from .eval.stats import wilson
from .ledger import Ledger
from .loss_stories import loss_story
from .report import batch_report
from .timeline import MatchTimeline, load_match
from .triage import triage

HARD_COUNTERS_DEFAULT = (
    "mini_pekka", "goblin_cage", "tombstone", "cannon", "minion_horde",
    "skeleton_army", "barbarians", "prince", "pekka", "inferno_tower",
)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _quartiles(summaries: list[dict]) -> list[dict]:
    if not summaries:
        return []
    ordered = sorted(summaries, key=lambda s: float(s.get("started_unix") or 0.0))
    n = len(ordered)
    k = min(4, n)
    slices = []
    for qi in range(k):
        lo_i = (qi * n) // k
        hi_i = ((qi + 1) * n) // k
        chunk = ordered[lo_i:hi_i]
        w = sum(1 for s in chunk if s.get("result") == "win")
        losses = sum(1 for s in chunk if s.get("result") == "loss")
        dr = sum(1 for s in chunk if s.get("result") == "draw")
        dec = w + losses
        ci = wilson(w, dec)
        slices.append({
            "slice": f"Q{qi + 1}",
            "matches": len(chunk),
            "w_l_d": f"{w}W-{losses}L-{dr}D",
            "win_rate": round(w / dec, 3) if dec else None,
            "ci95": [round(ci[0], 3), round(ci[1], 3)],
        })
    return slices


def _matchup_matrix(summaries: list[dict], min_seen: int = 2) -> list[dict]:
    by_card: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        if s.get("status") != "completed":
            continue
        for c in set(s.get("opp_deck_seen") or []):
            by_card[str(c)].append(s)
    rows = []
    for card, ms in by_card.items():
        if len(ms) < min_seen:
            continue
        w = sum(1 for m in ms if m.get("result") == "win")
        losses = sum(1 for m in ms if m.get("result") == "loss")
        dec = w + losses
        c_diff = sum(float(m.get("my_crowns") or 0) - float(m.get("opp_crowns") or 0) for m in ms) / len(ms)
        rows.append({
            "opp_card": card,
            "matches": len(ms),
            "wins": w,
            "losses": losses,
            "win_rate": round(w / dec, 3) if dec else None,
            "mean_crown_diff": round(c_diff, 2),
        })
    rows.sort(key=lambda r: (r["win_rate"] if r["win_rate"] is not None else 1.0, -r["matches"]))
    return rows[:18]


def _failure_clusters(match_dirs: list[Path], analyzed: list[tuple[MatchTimeline, list[Engagement]]],
                      cards: CardDB, win_conditions: tuple[str, ...] = ("giant", "hog_rider")) -> dict[str, Any]:
    opening_throws: list[str] = []
    wincon_starved: list[str] = []
    over_defend_bleed: list[str] = []
    spell_inefficiency: list[str] = []
    bad_counter_timing: list[str] = []
    card_counts: Counter = Counter()
    leak_plays = 0
    total_plays = 0

    by_id = {tl.match_id: (tl, engs) for tl, engs in analyzed}
    for d in match_dirs:
        tl_pair = by_id.get(d.name)
        if tl_pair is None:
            for mid, pair in by_id.items():
                if mid == (json.loads((d / "summary.json").read_text()).get("match_id") if (d / "summary.json").exists() else ""):
                    tl_pair = pair
                    break
        if tl_pair is None:
            continue
        tl, engs = tl_pair
        s = tl.summary
        if s.get("status") != "completed":
            continue
        decs = _jsonl(d / "decisions.jsonl")
        low_elix_frames = 0
        total_state_frames = 0
        wc_plays = 0
        threw_opening = False

        for rec in decs:
            st = rec.get("state") or {}
            if st:
                total_state_frames += 1
                if float(st.get("my_elixir") or 0.0) < 5.0:
                    low_elix_frames += 1
            c_card = rec.get("chosen_card")
            c_rule = rec.get("chosen_rule") or ""
            if not c_card or c_rule.startswith("menu"):
                continue
            total_plays += 1
            card_counts[str(c_card)] += 1
            if c_rule == "t0.leak_guard":
                leak_plays += 1
            if c_card in win_conditions:
                wc_plays += 1
                # Check if hard counter was in opponent hand & affordable
                seen = set(st.get("opp_deck_seen") or [])
                last4 = list(st.get("opp_last_four") or [])[-4:]
                opp_hi = float((st.get("opp_elixir") or [5.0, 5.0])[1])
                ready = [
                    hc for hc in HARD_COUNTERS_DEFAULT
                    if hc in seen and hc not in last4 and hc in cards and opp_hi >= cards.cost(hc) - 0.5
                ]
                if ready and len(bad_counter_timing) < 10:
                    bad_counter_timing.append(
                        f"{tl.match_id}@{int(rec.get('t_ms', 0)) // 1000}s:{c_card}_into_{'/'.join(ready[:2])}"
                    )
            t_ms = int(rec.get("t_ms") or 0)
            if t_ms < 15_000 and s.get("result") == "loss" and not threw_opening:
                cost = cards.cost(c_card) if c_card in cards else 3
                my_after = float(st.get("my_elixir") or 5.0) - cost
                opp_hi = float((st.get("opp_elixir") or [5.0, 5.0])[1])
                if my_after < 3.2 and opp_hi >= 6.5:
                    threw_opening = True
                    opening_throws.append(f"{tl.match_id}@{t_ms / 1000:.1f}s({c_card}->my={my_after:.1f},opp={opp_hi:.1f})")

        dur = float(s.get("duration_s") or 0.0)
        if dur >= 120.0 and wc_plays <= 1:
            wincon_starved.append(f"{tl.match_id}(wc_plays={wc_plays},dur={dur:.0f}s,{s.get('result')})")
        if total_state_frames >= 20 and (low_elix_frames / total_state_frames) > 0.75:
            over_defend_bleed.append(
                f"{tl.match_id}({100 * low_elix_frames / total_state_frames:.0f}%_below_5e,{s.get('result')})"
            )
        for e in engs:
            if e.triage != "STRATEGY":
                continue
            for pl in e.plays:
                if pl.side == "me" and pl.card in cards and cards[pl.card].is_spell:
                    cr = e.credit.get(pl.decision_seq, 0.0)
                    if cr < -1.2 and len(spell_inefficiency) < 10:
                        spell_inefficiency.append(f"{tl.match_id}@{pl.t_ms // 1000}s:{pl.card}(val={cr:+.1f})")

    return {
        "opening_throw": {"count": len(opening_throws), "examples": opening_throws[:8]},
        "wincon_starved": {"count": len(wincon_starved), "examples": wincon_starved[:8]},
        "over_defend_bleed": {"count": len(over_defend_bleed), "examples": over_defend_bleed[:8]},
        "spell_inefficiency": {"count": len(spell_inefficiency), "examples": spell_inefficiency[:8]},
        "bad_counter_timing": {"count": len(bad_counter_timing), "examples": bad_counter_timing[:8]},
        "card_utilization": {
            "total_plays": total_plays,
            "leak_guard_share": round(leak_plays / max(1, total_plays), 3),
            "plays_by_card": dict(card_counts.most_common()),
        },
    }


def build_digest(data_root: str | Path, cards: CardDB, arena: Arena, *,
                 ledger_path: str | Path = "ledger/experiments.jsonl",
                 max_loss_stories: int = 8) -> dict[str, Any]:
    root = Path(data_root)
    dirs = [root] if (root / "summary.json").exists() else sorted(
        d for d in root.iterdir() if d.is_dir() and (d / "summary.json").exists()
    )
    analyzed: list[tuple[MatchTimeline, list[Engagement]]] = []
    summaries: list[dict] = []
    for d in dirs:
        tl = load_match(d)
        engs = [triage(e, tl, arena) for e in segment(tl, arena, cards)]
        analyzed.append((tl, engs))
        if tl.summary.get("status") == "completed":
            summaries.append(tl.summary)

    rep = batch_report(analyzed)
    deck8_locked = sum(1 for s in summaries if len(s.get("opp_deck_seen") or []) >= 8)
    crowns_dist = Counter(f"{s.get('my_crowns', '?')}-{s.get('opp_crowns', '?')}" for s in summaries)

    losses_ranked = sorted(
        ((tl, engs) for tl, engs in analyzed
         if tl.summary.get("result") == "loss" and any(e.triage == "STRATEGY" for e in engs)),
        key=lambda pair: min((e.value for e in pair[1] if e.triage == "STRATEGY"), default=0.0),
    )[:max_loss_stories]
    stories = [loss_story(tl, engs, max_engagements=3) for tl, engs in losses_ranked]

    led = Ledger(ledger_path)
    recent_exps = [
        {"id": e.experiment_id, "status": e.status, "title": e.title,
         "hypothesis": e.hypothesis, "w_l_d": f"{e.wins}-{e.losses}-{e.draws}", "notes": e.notes}
        for e in sorted(led.all(), key=lambda x: x.experiment_id)[-15:]
    ]

    return {
        "data_root": str(root),
        "completed_matches": len(summaries),
        "total_dirs": len(dirs),
        "opp_8card_lock_rate": round(deck8_locked / max(1, len(summaries)), 3),
        "crown_distribution": dict(crowns_dist.most_common()),
        "quartiles": _quartiles(summaries),
        "batch_report": rep,
        "opponent_matchups": _matchup_matrix(summaries),
        "failure_clusters": _failure_clusters(dirs, analyzed, cards),
        "loss_stories": stories,
        "recent_ledger": recent_exps,
    }


def render_agent_markdown(digest: dict[str, Any]) -> str:
    rep = digest["batch_report"]
    lines = [
        "# droid-harness — Agent Strategy Refinement Digest",
        "",
        "> **Agent Instructions (Speed-3 Structural Coach):**",
        "> 1. Read the failure clusters, opponent matchup matrix, rule scorecard, and loss stories below.",
        "> 2. Check `Recent Experiment Ledger` so you NEVER re-propose a rejected change.",
        "> 3. Do NOT hand-guess continuous YAML floats blindly — use `droid-harness tune-offline` for continuous weights, and reserve Python edits in `policy/rules/` (+ unit tests in `tests/`) for structural gaps.",
        "> 4. Gate every candidate before live play:",
        ">    - `.venv/bin/pytest -q && .venv/bin/ruff check src/ tests/`",
        ">    - `.venv/bin/droid-harness replay-diff --champion <champ.yaml> --candidate <cand.yaml> --data <dir> --expected-pct <pct>`",
        ">    - `.venv/bin/droid-harness shadow --champion <champ.yaml> --candidate <cand.yaml> --data <dir>`",
        ">    - `.venv/bin/droid-harness experiment --champion <champ.yaml> --candidate <cand.yaml> --mode ladder`",
        "",
        "## 1. Batch Health & Drift Slices",
        f"- **Completed matches:** `{digest['completed_matches']}` (total dirs `{digest['total_dirs']}`)",
        f"- **Overall record:** `{rep.get('results')}` · Win rate `{rep.get('win_rate')}` · 95% CI `{rep.get('win_rate_ci95')}`",
        f"- **Opponent 8-card deck lock rate:** `{digest['opp_8card_lock_rate']:.1%}`",
        f"- **Crown distribution:** `{digest['crown_distribution']}`",
        f"- **Triage mix:** `{rep.get('triage_mix')}`",
        "",
        "| Quartile | Matches | Record | Win Rate | 95% CI |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for q in digest["quartiles"]:
        wr = "n/a" if q["win_rate"] is None else f"{q['win_rate']:.1%}"
        lines.append(f"| {q['slice']} | {q['matches']} | {q['w_l_d']} | {wr} | {q['ci95']} |")

    lines += ["", "## 2. Opponent Card Matchup Matrix (Worst First)", "",
              "| Enemy Card | Seen | W-L | Win Rate | Mean Crown Diff |",
              "| --- | ---: | --- | ---: | ---: |"]
    for m in digest["opponent_matchups"]:
        wr = "n/a" if m["win_rate"] is None else f"{m['win_rate']:.1%}"
        lines.append(f"| `{m['opp_card']}` | {m['matches']} | {m['wins']}-{m['losses']} | {wr} | {m['mean_crown_diff']:+.2f} |")

    lines += ["", "## 3. Rule & Posture Engagement Scorecard", ""]
    for rule, vals in sorted((rep.get("play_value_by_rule") or {}).items(), key=lambda kv: kv[1].get("mean", 0.0)):
        lines.append(f"- `{rule}`: n={vals['n']}, mean_value=`{vals['mean']:+.2f}`, total=`{vals['total']:+.1f}`")

    lines += ["", "## 4. Automated Failure Clusters", ""]
    fc = digest["failure_clusters"]
    for cname in ("opening_throw", "wincon_starved", "over_defend_bleed", "spell_inefficiency", "bad_counter_timing"):
        info = fc[cname]
        lines.append(f"- **`{cname}` ({info['count']}):** {', '.join(info['examples']) if info['examples'] else 'none'}")
    cu = fc["card_utilization"]
    lines.append(f"- **Card utilization:** total_plays={cu['total_plays']}, leak_guard_share=`{cu['leak_guard_share']:.1%}`, plays=`{cu['plays_by_card']}`")

    lines += ["", "## 5. Richest `STRATEGY` Loss Stories", ""]
    if digest["loss_stories"]:
        for st in digest["loss_stories"]:
            lines += ["```text", st, "```"]
    else:
        lines.append("_(No `STRATEGY`-triaged losses in this batch.)_")

    lines += ["", "## 6. Recent Experiment Ledger", ""]
    if digest["recent_ledger"]:
        for e in digest["recent_ledger"]:
            lines.append(f"- `{e['id']}` **{e['status'].upper()}** ({e['w_l_d']}): {e['title']} — {e['hypothesis']} [{e['notes']}]")
    else:
        lines.append("_(No experiments recorded in ledger yet.)_")
    return "\n".join(lines) + "\n"
