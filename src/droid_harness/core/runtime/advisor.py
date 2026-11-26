"""M9 L2 advisor: a separate process that polls an LLM and publishes posture Intents.

The advisor never shares a thread with the brain. It reads a compact TEXT state every
``cadence_s`` seconds, asks an OpenAI-compatible API for a posture opinion, and
publishes the result to an :class:`IntentBus`. The brain only ever reads
``bus.latest()`` (latest-only) and the bus expires intents after ``max_age_s``, so a
slow or dead advisor can never stall it: the bot simply plays on.

TTL is expressed through the bus itself (``Intent``/``IntentBus`` have no per-intent
TTL field): the advisor sets ``bus.max_age_s = 2 * cadence_s`` so each published
intent survives roughly one missed poll, then expires.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Collection

from .intent_bus import Intent

log = logging.getLogger(__name__)

# Mirrors domains.clash_royale.policy.posture.POSTURES. Kept local (not imported) so
# core/runtime stays domain-free; replies are validated against the strategy's
# advisor_may_set allowlist at runtime anyway.
POSTURES = ("OPENING", "BALANCED", "DEFEND_LEAD", "PRESS", "SPELL_CYCLE")

SCHEMA = '{"posture": null|"OPENING"|"BALANCED"|"DEFEND_LEAD"|"PRESS"|"SPELL_CYCLE", "note": str}'

SYSTEM_PROMPT = """You advise a Clash Royale bot. Reply with JSON ONLY, never prose.
OPENING: first 30s, no enemy seen — play safe. BALANCED: default, no strong read.
DEFEND_LEAD: ahead on crowns past x1 — protect the lead, don't donate.
PRESS: opponent low/broke on elixir — attack now.
SPELL_CYCLE: overtime and a tower falls to spells within the cast limit — cycle spells.
You may set ONLY "posture" (one of the five, or null for no opinion) plus a short "note".
You may NOT set cards, tiles, elixir floors, lanes, or any other knob.
The brain plays on without you: on any doubt answer {"posture": null, "note": "..."}.
Schema ONLY: {"posture": null|"OPENING"|"BALANCED"|"DEFEND_LEAD"|"PRESS"|"SPELL_CYCLE", "note": str}"""

API_TIMEOUT_S = 8.0


def build_prompt(state: dict) -> str:
    """Compact TEXT state for the advisor. Pure function: no network, no side effects."""
    try:
        my_el = f"{float(state.get('my_elixir', 0.0)):.1f}"
    except (TypeError, ValueError):
        my_el = "?"
    opp = state.get("opp_elixir") or (0.0, 0.0)
    try:
        opp_el = f"{float(opp[0]):.1f}"
    except (TypeError, ValueError, IndexError):
        opp_el = "?"
    units = state.get("units") or []
    bits = []
    for u in units[:8]:
        try:
            _, kind, side, x, y, *_ = u  # GameState.digest() tuple form
            bits.append(f"{side} {kind}@({x},{y})")
        except (TypeError, ValueError):
            bits.append(str(u))
    return "\n".join(
        [
            f"t={state.get('t_ms', 0)}ms phase={state.get('phase', '?')} elixir=you {my_el} "
            f"opp~{opp_el} hand={state.get('hand', [])} next={state.get('next')}",
            f"towers={state.get('towers', {})} crowns={state.get('crowns', (0, 0))} "
            f"current_posture={state.get('posture', '?')}",
            f"units({len(units)}): " + (", ".join(bits) if bits else "none"),
            f"opp_last4={state.get('opp_last_four', [])} warnings={state.get('warnings', [])}",
            f"Respond with JSON ONLY: {SCHEMA}",
        ]
    )


def _extract_json(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        s = text.strip()
        if s.startswith("```"):  # tolerate markdown-fenced replies
            s = s.strip("`")
            nl = s.find("\n")
            s = s[nl + 1 :] if nl >= 0 else s
        start, end = s.find("{"), s.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            obj = json.loads(s[start : end + 1])
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def parse_reply(text: str | None, allowed: Collection[str] | None) -> Intent | None:
    """Parse an advisor reply into an Intent. Returns None for anything not allowed.

    ``allowed`` is the strategy's ``advisor_may_set`` (None means all postures).
    A ``null`` posture is valid (no opinion) and yields ``Intent(posture=None)``.
    """
    allow = set(POSTURES) if allowed is None else set(allowed)
    if not isinstance(text, str) or not text.strip():
        log.warning("advisor: empty reply, skipping")
        return None
    obj = _extract_json(text)
    if obj is None or "posture" not in obj:
        log.warning("advisor: malformed reply (not a JSON object with 'posture'), skipping")
        return None
    posture, note = obj["posture"], obj.get("note", "")
    if posture is not None and (posture not in POSTURES or posture not in allow):
        log.warning("advisor: posture %r not in allowed %s, skipping", posture, sorted(allow))
        return None
    if not isinstance(note, str):
        note = str(note)
    return Intent(posture=posture, note=note)


def _chat_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base if base.endswith("/chat/completions") else base + "/chat/completions"


class Advisor:
    """Polls an OpenAI-compatible API and publishes posture Intents to ``bus``."""

    def __init__(
        self,
        bus,
        api_base: str,
        api_key: str,
        model: str,
        cadence_s: float = 4.0,
        allowed: set[str] | None = None,
    ) -> None:
        self.bus = bus
        self.api_base = api_base
        self.api_key = api_key
        self.model = model
        self.cadence_s = cadence_s
        self.allowed = set(POSTURES) if allowed is None else set(allowed)
        self.ttl_s = 2.0 * cadence_s
        try:
            bus.max_age_s = self.ttl_s  # IntentBus has no per-intent TTL; the bus expiry is the TTL
        except (AttributeError, TypeError):
            pass

    def _call(self, system: str, user: str) -> str:
        """One chat-completions round trip (stdlib urllib only). Raises on any failure."""
        body = json.dumps(
            {"model": self.model, "messages": [{"role": "system", "content": system},
                                               {"role": "user", "content": user}],
             "temperature": 0, "max_tokens": 128}
        ).encode()
        req = urllib.request.Request(
            _chat_url(self.api_base), data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=API_TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
            raise RuntimeError(f"advisor API call failed: {e!r}") from e
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"advisor API returned an unexpected body: {e!r}") from e
        if not isinstance(content, str):
            raise RuntimeError("advisor API returned a non-string completion")
        return content

    def poll(self, state: dict | str) -> Intent | None:
        """Single advisory step: ask the API about ``state`` and maybe publish.

        ``state`` is a GameState digest dict or pre-rendered prompt text. Returns the
        Intent, or None when there is nothing to publish (network failure, rejected
        reply, or a null-posture no-opinion — a no-op must not clobber a still-fresh
        intent under latest-only semantics).
        """
        user = build_prompt(state) if isinstance(state, dict) else state
        try:
            content = self._call(SYSTEM_PROMPT, user)
        except Exception as e:  # advisory only: the bot plays on
            log.warning("advisor: %s; skipping", e)
            return None
        intent = parse_reply(content, self.allowed)
        if intent is None or intent.posture is None:
            return intent  # rejected (already logged) or no opinion: publish nothing
        try:
            self.bus.max_age_s = self.ttl_s
        except (AttributeError, TypeError):
            pass
        try:
            self.bus.publish(intent)
        except Exception as e:
            log.warning("advisor: publish failed (%r); skipping", e)
            return None
        return intent
