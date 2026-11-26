from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Candidate:
    rule_id: str
    card: str
    slot: int
    tile: tuple[float, float]
    score: float
    execute_at_ms: int | None = None  # match time; None = now
    reason: str = ""
    predicted_tile: tuple[float, float] | None = None  # where the target is expected at effect time
    preselect: bool = False  # select the card now, place at execute_at_ms

    @property
    def family(self) -> str:
        return family(self.rule_id)


def family(rule_id: str) -> str:
    return rule_id.split(".", 1)[0]


@dataclass
class Decision:
    t_ms: int
    host_ns: int
    posture: str
    chosen: Candidate | None
    candidates: list[Candidate] = field(default_factory=list)
    totals: dict[int, float] = field(default_factory=dict)
    state_digest: dict | None = None
    intent: dict | None = None

    def to_record(self) -> dict:
        c = self.chosen
        return {
            "t_ms": self.t_ms, "posture": self.posture,
            "chosen_rule": c.rule_id if c else None, "chosen_card": c.card if c else None,
            "tile": c.tile if c else None, "execute_at_ms": c.execute_at_ms if c else None,
            "predicted_tile": c.predicted_tile if c else None, "score": c.score if c else None,
            "candidates": [{**asdict(x), "total": self.totals.get(i)} for i, x in enumerate(self.candidates)],
            "state": self.state_digest, "intent": self.intent,
        }
