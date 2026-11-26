"""Offline flywheel dashboard (plain Python, marimo-compatible).

Run from the repository root::

    python notebooks/dashboard.py [data/matches]

The module deliberately has no marimo import at import time and no side effects beyond
``main()``: marimo can import its public functions as notebook cells, while the fallback
``__main__`` path prints a plain-text report.

Every input is read defensively (``.get`` throughout) because real match directories are
written by a live recorder: some are interrupted, some lack ``summary.json``, and early
decisions have no chosen action or score.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

try:  # reuse the flywheel's own Wilson implementation when the package is installed
    from droid_harness.flywheel.eval.stats import wilson
except Exception:  # pragma: no cover - notebook can run from a bare checkout
    def wilson(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
        if n == 0:
            return (0.0, 1.0)
        p = wins / n
        d = 1 + z * z / n
        c = (p + z * z / (2 * n)) / d
        h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
        return (max(0.0, c - h), min(1.0, c + h))


try:  # keep the dashboard's perception categories in sync with triage
    from droid_harness.flywheel.triage import PERCEPTION_WARNINGS
except Exception:  # pragma: no cover
    PERCEPTION_WARNINGS = {
        "elixir_discrepancy", "hand_unreadable", "affordability_mismatch", "tower_hp_increase",
    }

DEFAULT_MATCHES = Path("data/matches")
LATENCY_P99_MS = 100.0


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}
    return data if isinstance(data, dict) else {}


def _summary_paths(root: str | Path):
    return sorted(Path(root).glob("*/summary.json"))


def _decision_paths(root: str | Path):
    return sorted(Path(root).glob("*/decisions.jsonl"))


def _decision_records(root: str | Path):
    for path in _decision_paths(root):
        try:
            lines = path.read_text().splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            if isinstance(rec, dict):
                yield rec


def _day(summary: dict) -> str:
    started = summary.get("started_unix")
    if isinstance(started, (int, float)) and not isinstance(started, bool):
        try:
            return _dt.datetime.fromtimestamp(started, tz=_dt.UTC).date().isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    at = summary.get("started_at")
    return at[:10] if isinstance(at, str) and len(at) >= 10 else "unknown"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def win_rates(root: str | Path = DEFAULT_MATCHES) -> list[dict]:
    """Win rate with Wilson 95% CI by ``strategy_id x opponent_kind x game_version``."""
    groups: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "draws": 0})
    for path in _summary_paths(root):
        summary = _read_json(path)
        status = summary.get("status")
        if status is not None and status != "completed":
            continue
        key = (str(summary.get("strategy_id") or "?"),
               str(summary.get("opponent_kind") or "?"),
               str(summary.get("game_version") or "?"))
        result = summary.get("result")
        if result == "win":
            groups[key]["wins"] += 1
        elif result == "loss":
            groups[key]["losses"] += 1
        elif result == "draw":
            groups[key]["draws"] += 1
    rows = []
    for (strategy, opponent, version), g in groups.items():
        n = g["wins"] + g["losses"]
        lo, hi = wilson(g["wins"], n)
        rows.append({
            "strategy_id": strategy, "opponent_kind": opponent, "game_version": version,
            "wins": g["wins"], "losses": g["losses"], "draws": g["draws"], "n": n,
            "win_rate": round(g["wins"] / n, 3) if n else None,
            "ci95": (round(lo, 3), round(hi, 3)),
        })
    rows.sort(key=lambda r: (r["strategy_id"], r["opponent_kind"], r["game_version"]))
    return rows


def play_value(root: str | Path = DEFAULT_MATCHES) -> list[dict]:
    """Mean score of the chosen action by ``rule x posture`` from ``decisions.jsonl``."""
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for rec in _decision_records(root):
        rule = rec.get("chosen_rule")
        score = _number(rec.get("score"))
        if not isinstance(rule, str) or not rule or score is None:
            continue
        posture = str(rec.get("posture") or "?")
        groups[(rule, posture)].append(score)
    rows = []
    for (rule, posture), values in groups.items():
        rows.append({
            "rule": rule, "posture": posture, "n": len(values),
            "mean": round(sum(values) / len(values), 3), "total": round(sum(values), 3),
        })
    rows.sort(key=lambda r: (r["rule"], r["posture"]))
    return rows


def triage_mix(root: str | Path = DEFAULT_MATCHES) -> dict[str, float]:
    """Decision-level triage mix inferred from warnings and loop latency.

    Engagement triage needs a warehouse; match directories on their own do not carry
    ``eng_id``/``triage``.  This mirrors the same priority order as ``triage.py``
    (perception first, then latency) over the recorded decision stream.
    """
    counts: dict[str, int] = defaultdict(int)
    for rec in _decision_records(root):
        state = rec.get("state")
        warnings = state.get("warnings") if isinstance(state, dict) else []
        if not isinstance(warnings, (list, tuple, set)):
            warnings = []
        latency = rec.get("lat")
        total = latency.get("total") if isinstance(latency, dict) else None
        latency = _number(total)
        if any(w in PERCEPTION_WARNINGS for w in warnings):
            counts["PERCEPTION"] += 1
        elif latency is not None and latency > LATENCY_P99_MS:
            counts["LATENCY"] += 1
        else:
            counts["OK"] += 1
    total = sum(counts.values()) or 1
    return {k: round(v / total, 3) for k, v in sorted(counts.items())}


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[index]


def loop_p99_by_day(root: str | Path = DEFAULT_MATCHES) -> dict[str, float]:
    """Daily loop p99 from the per-match ``loop_p99_ms`` summaries."""
    groups: dict[str, list[float]] = defaultdict(list)
    for path in _summary_paths(root):
        summary = _read_json(path)
        value = _number(summary.get("loop_p99_ms"))
        if value is not None:
            groups[_day(summary)].append(value)
    out = {}
    for day, values in sorted(groups.items()):
        p99 = _percentile(values, 99.0)
        if p99 is not None:
            out[day] = round(p99, 3)
    return out


def summarize(root: str | Path = DEFAULT_MATCHES) -> dict:
    return {
        "win_rate": win_rates(root),
        "play_value": play_value(root),
        "triage_mix": triage_mix(root),
        "loop_p99_by_day": loop_p99_by_day(root),
    }


# Friendly aliases for notebook cells written against the report vocabulary.
win_rate_by_strategy = win_rates
play_value_by_rule_posture = play_value
build_report = summarize


def render(summary: dict) -> str:
    lines = ["Win rate by strategy x opponent_kind x game_version"]
    for row in summary.get("win_rate", []):
        rate = "n/a" if row.get("win_rate") is None else f"{row['win_rate']:.3f}"
        lo, hi = row.get("ci95", (0.0, 1.0))
        lines.append(
            f"  {row.get('strategy_id')} | {row.get('opponent_kind')} | {row.get('game_version')}: "
            f"{row.get('wins', 0)}W-{row.get('losses', 0)}L-{row.get('draws', 0)}D "
            f"win_rate={rate} ci95=[{lo:.3f}, {hi:.3f}]")

    lines.append("Play value by rule x posture (mean score of chosen)")
    for row in summary.get("play_value", []):
        lines.append(f"  {row.get('rule')} | {row.get('posture')}: "
                     f"n={row.get('n', 0)} mean={row.get('mean', 0.0):+.3f} "
                     f"total={row.get('total', 0.0):+.3f}")

    lines.append("Triage mix")
    mix = summary.get("triage_mix", {})
    if mix:
        for tag, share in mix.items():
            lines.append(f"  {tag}: {share:.1%}")
    else:
        lines.append("  (no decisions)")

    lines.append("Loop p99 per day")
    daily = summary.get("loop_p99_by_day", {})
    if daily:
        for day, value in daily.items():
            lines.append(f"  {day}: {value:.1f} ms")
    else:
        lines.append("  (no summaries)")
    return "\n".join(lines)


def main(argv: list[str] | None = None, out: Any = None) -> str:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else DEFAULT_MATCHES
    text = render(summarize(root))
    print(text, file=out or sys.stdout)
    return text


if __name__ == "__main__":
    main()
