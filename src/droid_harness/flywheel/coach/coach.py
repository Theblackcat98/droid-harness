"""Offline coach: builds the context pack, asks an LLM for one Proposal, checks guardrails.

The LLM never plays. It reads STRATEGY-tagged loss stories, the batch report, the current
strategy and the ledger, and proposes one testable change at a time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

from ..ledger import Ledger
from .proposal import Proposal, check_guardrails

SYSTEM = """You are the coach for a Clash Royale bot. You improve its *strategy* only: parameters,
Tier 0 thresholds, posture multipliers, the counter table, and rule code. You never change
perception, the executor, constraints, triage or evaluation code.

Return ONE proposal as JSON matching this schema:
{schema}

Rules:
- One falsifiable hypothesis: cause -> change -> expected movement of a named metric.
- Cite evidence (match ids, engagement ids, report numbers) you actually saw below.
- Keep the change small; estimate the % of decisions it changes (it will be checked by replay).
- Do not re-propose an idea listed as rejected in the ledger unless you explain what changed.
- The patch is a unified diff against the repo root, touching only configs/strategies/**,
  src/droid_harness/domains/clash_royale/policy/rules/** and (for rule changes, required)
  a unit test under tests/coach/**.
"""


class LLMClient(Protocol):
    def complete_json(self, system: str, user: str) -> dict: ...


class AnthropicClient:
    """Optional: ``pip install droid-harness[coach]`` and set ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 4000) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete_json(self, system: str, user: str) -> dict:
        msg = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                          messages=[{"role": "user", "content": user}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        start, end = text.find("{"), text.rfind("}")
        return json.loads(text[start:end + 1])


def context_pack(*, strategy_path: str | Path, report: dict, loss_stories: list[str], ledger: Ledger,
                 rules_dir: str | Path | None = None, last_ledger: int = 30, excluded_share: float | None = None) -> str:
    sp = Path(strategy_path)
    parts = ["## Current strategy", "```yaml", sp.read_text(), "```"]
    import yaml

    spec = yaml.safe_load(sp.read_text())
    if spec.get("counters"):
        parts += ["## Counter table", "```csv", (sp.parent / spec["counters"]).read_text(), "```"]
    if rules_dir:
        for f in sorted(Path(rules_dir).glob("*.py")):
            parts += [f"## Rule source: {f.name}", "```python", f.read_text(), "```"]
    parts += ["## Batch report", "```json", json.dumps(report, indent=1), "```"]
    if excluded_share is not None:
        parts.append(f"Share of engagements excluded by triage (perception/execution/latency): "
                     f"{excluded_share:.0%}. Treat conclusions accordingly.")
    parts.append("## Loss stories (STRATEGY engagements only)")
    parts += [f"```\n{s}\n```" for s in loss_stories]
    exps = sorted(ledger.all(), key=lambda e: e.experiment_id)[-last_ledger:]
    parts.append("## Ledger (recent experiments)")
    parts += [f"- {e.experiment_id} {e.status}: {e.title} — {e.hypothesis} ({e.wins}-{e.losses})" for e in exps]
    return "\n".join(parts)


def propose(client: LLMClient, user_context: str, ledger: Ledger, game_version: str) -> tuple[Proposal, list[str]]:
    system = SYSTEM.format(schema=json.dumps(Proposal.model_json_schema()))
    p = Proposal(**client.complete_json(system, user_context))
    problems = check_guardrails(p)
    for dup in ledger.similar_failures(p.hypothesis, game_version):
        problems.append(f"similar to rejected {dup.experiment_id}: {dup.title}")
    return p, problems
