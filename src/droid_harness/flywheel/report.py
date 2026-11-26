"""Batch report for the coach and the dashboard."""

from __future__ import annotations

from collections import Counter, defaultdict

from .engagements import Engagement
from .eval.stats import wilson
from .timeline import MatchTimeline


def batch_report(matches: list[tuple[MatchTimeline, list[Engagement]]]) -> dict:
    results = Counter()
    rule_values: dict[str, list[float]] = defaultdict(list)
    posture_values: dict[str, list[float]] = defaultdict(list)
    triage = Counter()
    for tl, engs in matches:
        if tl.summary.get("status") == "completed":
            results[tl.summary.get("result", "?")] += 1
        chosen = {c.seq: c for c in tl.chosen}
        for e in engs:
            triage[e.triage] += 1
            if e.triage != "STRATEGY" and e.triage != "OK":
                continue
            for seq, v in e.credit.items():
                c = chosen.get(seq)
                if c is not None:
                    rule_values[c.rule].append(v)
                    posture_values[c.posture].append(v)
    n = results["win"] + results["loss"]
    lo, hi = wilson(results["win"], n)

    def agg(d):
        return {k: {"n": len(v), "mean": round(sum(v) / len(v), 3), "total": round(sum(v), 2)}
                for k, v in sorted(d.items(), key=lambda kv: sum(kv[1]))}

    total = sum(triage.values()) or 1
    return {
        "matches": sum(results.values()), "results": dict(results),
        "win_rate": round(results["win"] / n, 3) if n else None, "win_rate_95ci": [round(lo, 3), round(hi, 3)],
        "play_value_by_rule": agg(rule_values), "play_value_by_posture": agg(posture_values),
        "triage_mix": {k: round(v / total, 3) for k, v in triage.items()},
    }
