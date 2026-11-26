"""Shadow mode: run a candidate policy beside the champion on logged states.

This is gate 3 of the promotion funnel (plan §10.4).  The champion is evaluated
normally; the candidate is asked for its choice on the *same* snapshot.  Nothing is
actuated: policies are pure decision functions here, so a disagreement costs nothing
but tells us which rule changed and whether the engagement it belonged to was losing.

The module intentionally reuses :class:`~droid_harness.domains.clash_royale.state.GameState`
rather than defining a parallel state shape.  ``ShadowRunner`` accepts either
``GameState`` instances or state digests (the dict produced by ``GameState.digest()``),
which is what replay/decisions logs contain.
"""

from __future__ import annotations

import copy
import importlib
import types
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ..domains.clash_royale.state import GameState

__all__ = ["ShadowRecord", "ShadowRunner", "load_policy", "report"]


@dataclass(frozen=True)
class ShadowRecord:
    """One tick where the champion and candidate were both asked for a choice.

    ``champion_choice``/``candidate_choice`` are rule ids (or whatever the policy
    returns when it has no ``rule_id``), so the record is cheap to log and rank.
    """

    tick: int
    champion_choice: str | None
    candidate_choice: str | None
    agreement: bool
    value: float = 0.0

    @property
    def rule(self) -> str:
        """The rule to attribute a disagreement to (candidate first, then champion)."""
        return self.candidate_choice or self.champion_choice or "none"

    # Records are also tuple-like, matching the ``(tick, champion, candidate, agreement,
    # value)`` shape used in logs and reports.
    def __iter__(self):
        return iter((self.tick, self.champion_choice, self.candidate_choice, self.agreement, self.value))

    def __getitem__(self, index):
        return tuple(self)[index]

    def __len__(self) -> int:
        return 5


def _import_qualified(name: str) -> Any:
    """Import ``pkg.mod`` or resolve ``pkg.mod:attr`` / ``pkg.mod.attr``."""
    if ":" in name:
        module_name, _, attr_path = name.partition(":")
        obj: Any = importlib.import_module(module_name)
        for attr in attr_path.split("."):
            if attr:
                obj = getattr(obj, attr)
        return obj
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        # Allow a dotted attribute path (e.g. the example in the shadow CLI docs).
        parts = name.split(".")
        for i in range(len(parts) - 1, 0, -1):
            try:
                obj = importlib.import_module(".".join(parts[:i]))
            except ModuleNotFoundError:
                continue
            try:
                for attr in parts[i:]:
                    obj = getattr(obj, attr)
            except AttributeError:
                continue
            return obj
        raise


def load_policy(policy: Any) -> Any:
    """Return a usable policy from an object, module, or qualified name.

    Modules may expose the policy as ``policy``/``POLICY``, or directly as module-level
    ``choose``/``decide`` functions (convenient for test doubles).
    """
    if not isinstance(policy, str):
        return policy
    obj = _import_qualified(policy)
    if isinstance(obj, types.ModuleType):
        for attr in ("policy", "POLICY"):
            candidate = getattr(obj, attr, None)
            if candidate is not None:
                return candidate
    return obj


def _choice_rule(result: Any) -> str | None:
    """Normalize a policy result to a rule id (``Decision`` -> chosen ``Candidate``)."""
    if result is None:
        return None
    chosen = getattr(result, "chosen", None)
    if chosen is not None or hasattr(result, "chosen"):
        result = chosen
    if result is None:
        return None
    if isinstance(result, str):
        return result
    for attr in ("rule_id", "rule", "id", "name"):
        value = getattr(result, attr, None)
        if isinstance(value, str):
            return value
    return str(result)


def _call_policy(policy: Any, state: GameState) -> str | None:
    """Ask one policy for a choice without ever actuating anything."""
    fn = getattr(policy, "choose", None) or getattr(policy, "decide", None)
    if fn is None and callable(policy):
        fn = policy
    if fn is None:
        raise TypeError(
            f"policy {policy!r} has no choose()/decide() method and is not callable; "
            "pass a policy object, module, or qualified name"
        )
    return _choice_rule(fn(state))


def _to_state(item: Any, fallback_tick: int) -> tuple[GameState, float]:
    """Accept a ``GameState``, a digest dict, or ``{"state": digest, "value": ...}``."""
    value = 0.0
    raw = item
    if isinstance(item, tuple) and len(item) == 2:  # convenient (state, value) pair
        raw, value = item
    if isinstance(raw, dict):
        outer = raw
        if isinstance(raw.get("state"), dict):
            raw = raw["state"]
            value = outer.get("engagement_value", outer.get("play_value", outer.get("value", value)))
        else:
            value = outer.get("engagement_value", outer.get("play_value", outer.get("value", value)))
        try:
            state = GameState.from_digest(raw)
        except (KeyError, TypeError, ValueError):
            state = GameState(t_ms=int(raw.get("t_ms", raw.get("tick", fallback_tick))))
    elif isinstance(raw, GameState):
        state = raw
    else:
        raise TypeError(f"expected GameState or state digest, got {type(raw).__name__}")
    try:
        value = float(value or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return state, value


def _lookup_value(values: Any, index: int, tick: int) -> float | None:
    if values is None:
        return None
    if callable(values):
        return float(values(tick, index))
    if isinstance(values, dict):
        return float(values.get(tick, values.get(str(tick), 0.0)))
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if index < len(values):
            return float(values[index])
    return None


class ShadowRunner:
    """Evaluate a candidate next to a champion, without touching the phone.

    ``champion`` and ``candidate`` may be live policy objects, policy modules, or
    qualified names such as ``"my_pkg.my_policy"``.  A policy needs ``choose(state)``
    or ``decide(state)``; the real :class:`CRPolicy` uses ``decide`` and is supported.
    """

    def __init__(self, champion: Any, candidate: Any) -> None:
        self.champion = load_policy(champion)
        self.candidate = load_policy(candidate)
        self.records: list[ShadowRecord] = []

    def compare(self, state: Any, value: float | None = None, *, tick: int | None = None,
                reset: bool = False) -> ShadowRecord:
        """Ask both policies about one state snapshot."""
        snapshot, embedded_value = _to_state(state, 0)
        if reset:
            for policy in (self.champion, self.candidate):
                reset_fn = getattr(policy, "reset_match", None)
                if reset_fn is not None:
                    reset_fn()
        # Copies keep the champion's mutations (posture, last_action_ms, ...) away from
        # the candidate, and vice versa: both see the same snapshot, neither is actuated.
        champion_choice = _call_policy(self.champion, copy.deepcopy(snapshot))
        candidate_choice = _call_policy(self.candidate, copy.deepcopy(snapshot))
        record = ShadowRecord(
            tick=int(snapshot.t_ms if tick is None else tick),
            champion_choice=champion_choice,
            candidate_choice=candidate_choice,
            agreement=champion_choice == candidate_choice,
            value=float(embedded_value if value is None else value),
        )
        self.records.append(record)
        return record

    def run(self, states: Iterable[Any], values: Any = None) -> list[ShadowRecord]:
        """Evaluate both policies over ``states`` in order.

        ``states`` items may carry their own engagement value (``value``/
        ``engagement_value``/``play_value`` key); ``values`` can override it with a
        sequence, a ``{tick: value}`` mapping, or ``callable(tick, index)``.
        """
        records: list[ShadowRecord] = []
        for i, item in enumerate(states):
            snapshot, embedded_value = _to_state(item, i)
            override = _lookup_value(values, i, snapshot.t_ms)
            value = embedded_value if override is None else override
            records.append(self.compare(snapshot, value=value))
        return records

    # ``evaluate`` is a friendly alias used by callers that think in replay terms.
    evaluate = run

    def report(self, disagreements: Iterable[Any]) -> list[tuple[str, int, float]]:
        return report(disagreements)


def report(disagreements: Iterable[Any]) -> list[tuple[str, int, float]]:
    """Rank disagreements as ``(rule, count, mean_value)``.

    The rule is the candidate's choice when it made one, else the champion's.  Ranking
    is by frequency (most common first), then by mean engagement value (worst first),
    which surfaces the changes that touch the most losing engagements.
    """
    counts: dict[str, int] = defaultdict(int)
    sums: dict[str, float] = defaultdict(float)
    for rec in disagreements:
        if isinstance(rec, dict):
            agreement = rec.get("agreement", False)
            rule = rec.get("candidate_choice") or rec.get("champion_choice") or "none"
            value = rec.get("value", rec.get("engagement_value", 0.0))
        elif isinstance(rec, (tuple, list)):
            if len(rec) < 5:
                continue
            _tick, champion, candidate, agreement, value = rec[:5]
            rule = candidate or champion or "none"
        else:
            agreement = getattr(rec, "agreement", False)
            rule = getattr(rec, "candidate_choice", None) or getattr(rec, "champion_choice", None) or "none"
            value = getattr(rec, "value", 0.0)
        if agreement:
            continue
        try:
            value = float(value or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        counts[rule] += 1
        sums[rule] += value
    rows = [(rule, counts[rule], sums[rule] / counts[rule]) for rule in counts]
    rows.sort(key=lambda row: (-row[1], row[2], row[0]))
    return rows
