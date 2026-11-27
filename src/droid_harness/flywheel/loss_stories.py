"""Text timelines of STRATEGY engagements — the main input to the LLM coach."""

from __future__ import annotations

from .engagements import Engagement
from .timeline import MatchTimeline


def _clock(t_ms: int) -> str:
    s = max(0, t_ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


def loss_story(tl: MatchTimeline, engagements: list[Engagement], max_engagements: int = 4) -> str:
    s = tl.summary
    head = (f"[match {tl.match_id} | {s.get('strategy_id', '?')} vs {s.get('opponent_kind', '?')}:"
            f"{s.get('opponent_id', '')} | {s.get('result', '?')} {s.get('my_crowns', '?')}-{s.get('opp_crowns', '?')}"
            f" | game {s.get('game_version', '?')} | epoch {s.get('balance_epoch', '?')}]")
    lines = [head]
    chosen_by_seq = {c.seq: c for c in tl.chosen}
    strategy = sorted((e for e in engagements if e.triage == "STRATEGY"), key=lambda e: e.value)[:max_engagements]
    for e in sorted(strategy, key=lambda e: e.t_start):
        lines.append(f"eng {e.eng_id} · {e.lane} · {_clock(e.t_start)}–{_clock(e.t_end)} · value {e.value:+.1f} · STRATEGY")
        lines.append(f"  spent me {e.spent_me:.0f} / opp {e.spent_opp:.0f} · tower dmg dealt {e.tower_dmg_dealt}"
                     f" / taken {e.tower_dmg_taken}")
        for pl in e.plays:
            who = "opp" if pl.side == "opp" else "we"
            rule = f" · rule {pl.rule}" if pl.rule else ""
            tile = f" @ ({pl.tile[0]:.1f},{pl.tile[1]:.1f})" if pl.tile else ""
            credit = f" · play value {e.credit[pl.decision_seq]:+.1f}" if pl.decision_seq in e.credit else ""
            extra = ""
            if pl.side == "me" and pl.decision_seq in chosen_by_seq:
                ch = chosen_by_seq[pl.decision_seq]
                pt = next((p for p in reversed(tl.points) if p.t_ms <= ch.t_ms), None)
                if pt is not None:
                    extra += f" · elix {pt.my_elixir:.1f} vs [{pt.opp_elixir[0]:.1f},{pt.opp_elixir[1]:.1f}]"
                if ch.runners_up:
                    alts = ", ".join(f"{r}:{c}={sc:+.2f}" for r, c, sc in ch.runners_up)
                    extra += f" · alts [{alts}]"
            lines.append(f"  {_clock(pl.t_ms)} {who}: {pl.card}{tile}{rule}{credit}{extra}")
    excluded = [e for e in engagements if e.triage in ("PERCEPTION", "EXECUTION", "LATENCY")]
    for e in excluded:
        lines.append(f"  excluded: eng {e.eng_id} ({e.triage}: {e.evidence})")
    return "\n".join(lines)
