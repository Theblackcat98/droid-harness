"""Compiled skills: replay action sequences that worked, call a model only on a miss.

Key = (package, screen signature, goal). A trace is promoted to "compiled" after it has
succeeded ``promote_after`` times; compiled skills replay step by step with a pre-condition
check before each step and a post-condition check at the end. Any failed check falls back
to the model, and the failure is recorded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Step:
    action: str  # tap | type | swipe | back | home | launch
    args: dict
    expect_screen: str | None = None  # signature expected before this step (pre-condition)


@dataclass
class Skill:
    key: str
    package: str
    goal: str
    steps: list[Step]
    successes: int = 0
    failures: int = 0
    compiled: bool = False
    final_screen: str | None = None  # post-condition

    @classmethod
    def from_dict(cls, d: dict) -> Skill:
        d = dict(d)
        d["steps"] = [Step(**s) for s in d["steps"]]
        return cls(**d)


def skill_key(package: str, screen_signature: str, goal: str) -> str:
    norm_goal = " ".join(goal.lower().split())
    return hashlib.sha256(f"{package}|{screen_signature}|{norm_goal}".encode()).hexdigest()[:16]


@dataclass
class SkillsCache:
    path: Path
    promote_after: int = 2
    demote_after_failures: int = 2
    skills: dict[str, Skill] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path, **kw) -> SkillsCache:
        p = Path(path)
        c = cls(p, **kw)
        if p.exists():
            c.skills = {k: Skill.from_dict(v) for k, v in json.loads(p.read_text()).items()}
        return c

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: asdict(v) for k, v in self.skills.items()}, indent=1))

    def lookup(self, package: str, screen_signature: str, goal: str) -> Skill | None:
        s = self.skills.get(skill_key(package, screen_signature, goal))
        return s if s is not None and s.compiled else None

    def record(self, package: str, screen_signature: str, goal: str, steps: list[Step], success: bool,
               final_screen: str | None = None) -> Skill:
        key = skill_key(package, screen_signature, goal)
        s = self.skills.get(key)
        if s is None or (not s.compiled and [asdict(x) for x in s.steps] != [asdict(x) for x in steps]):
            s = Skill(key, package, goal, steps, final_screen=final_screen)
            self.skills[key] = s
        if success:
            s.successes += 1
            s.final_screen = final_screen or s.final_screen
            if s.successes >= self.promote_after:
                s.compiled = True
        else:
            s.failures += 1
            if s.failures >= self.demote_after_failures:
                s.compiled = False
                s.successes = 0
                s.failures = 0
        self.save()
        return s


def replay(skill: Skill, execute: Callable[[Step], None], screen: Callable[[], str],
           settle: Callable[[], bool]) -> tuple[bool, str]:
    """Run a compiled skill. Returns (ok, reason). On False, hand control back to the model."""
    for i, step in enumerate(skill.steps):
        if step.expect_screen is not None and screen() != step.expect_screen:
            return False, f"pre-condition failed at step {i}"
        execute(step)
        settle()
    if skill.final_screen is not None and screen() != skill.final_screen:
        return False, "post-condition failed"
    return True, "ok"
