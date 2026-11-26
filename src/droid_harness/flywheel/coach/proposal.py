"""The coach's output contract, and the guardrails every proposal must pass."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# The coach may only touch strategy content. Never constraints, perception, executor, triage or eval.
ALLOWED_PATHS = (
    "configs/strategies/",
    "src/droid_harness/domains/clash_royale/policy/rules/",
    "tests/coach/",  # coach-written tests for coach-written rules
)
FORBIDDEN_FILES = ("src/droid_harness/domains/clash_royale/policy/constraints.py",)


class Proposal(BaseModel):
    title: str
    hypothesis: str = Field(description="cause -> change -> expected metric move; falsifiable")
    evidence: list[str] = Field(default_factory=list)
    change_type: Literal["param", "tier0", "posture", "counter_table", "rule_new", "rule_edit", "rule_weight"]
    patch: str = Field(description="unified diff against the repo root")
    target_metric: str
    expected_direction: Literal["up", "down"]
    expected_decision_change_pct: float = Field(ge=0, le=100)
    risks: str = ""


def diff_paths(patch: str) -> list[str]:
    paths = []
    for line in patch.splitlines():
        m = re.match(r"^\+\+\+ (?:b/)?(\S+)", line) or re.match(r"^--- (?:a/)?(\S+)", line)
        if m and m.group(1) != "/dev/null":
            paths.append(m.group(1))
    return sorted(set(paths))


def check_guardrails(p: Proposal) -> list[str]:
    problems = []
    paths = diff_paths(p.patch)
    if not paths:
        problems.append("patch touches no files")
    for path in paths:
        if path in FORBIDDEN_FILES or not path.startswith(ALLOWED_PATHS):
            problems.append(f"path not allowed: {path}")
    if p.change_type in ("rule_new", "rule_edit") and not any(x.startswith("tests/coach/") for x in paths):
        problems.append("rule changes must ship with a unit test under tests/coach/")
    return problems


def apply_patch(repo: str | Path, patch: str) -> None:
    """``git apply`` after a dry-run check. Raises CalledProcessError if it doesn't apply."""
    subprocess.run(["git", "-C", str(repo), "apply", "--check", "-"], input=patch, text=True, check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "apply", "-"], input=patch, text=True, check=True, capture_output=True)
