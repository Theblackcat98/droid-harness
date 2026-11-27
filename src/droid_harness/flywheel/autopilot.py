"""Autopilot: farm → diagnose → propose → gate → test → promote → report, unattended (plan step 4).

One cycle:

1. **Farm** ``farm_matches`` with the current champion (``configs/strategies/CHAMPION``), then
   prune clean-win video (``retain``).
2. **Diagnose**: the digest over everything farmed (with win-probability swings when a value
   model exists) and, every cycle, a retrained value model once enough matches exist.
3. **Propose**: targeted Speed 1 (``tune-targeted``) on the champion's most frequent failure
   pattern; if nothing passes, the next pattern (up to ``patterns_to_try``). One or two knobs:
   one hypothesis.
4. **Gate**: the search itself is the replay-diff gate (hits removed, collateral budget).
5. **Test** the candidate against the champion with a paired experiment on the testbed
   (``testbed``: ``training_camp``/``practice`` for fixed opponents, ``ladder`` to confirm).
6. **Promote** on acceptance: the ``CHAMPION`` pointer moves to the candidate file (strategy
   files are never edited). Rejections stay in the ledger so they aren't retried.
7. **Report** to ``reports/<date>/`` (+ ``reports/latest.md``), optionally **notify** a phone
   (ntfy.sh or any webhook) and **publish** the reports to a git branch for the weekly
   coding-agent job.

Stops cleanly when ``<data_root>/STOP`` exists or after ``cycles`` cycles (0 = forever).
Code changes (new rules) are never made here: they arrive as pull requests that you merge.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

log = logging.getLogger(__name__)


@dataclass
class AutopilotConfig:
    device: str = "phone-a"
    champion: str = "v6"  # used when the CHAMPION pointer doesn't exist yet
    farm_mode: str = "ladder"
    farm_matches: int = 20
    testbed: str = "ladder"  # mode for experiments: training_camp | practice | ladder
    test: str = "pairs"  # pairs | dense
    max_games: int = 60
    futility_pairs: int = 8
    data_root: Path = Path("data/autopilot")
    strategies_dir: Path = Path("configs/strategies")
    store: Path = Path("ledger/experiments.jsonl")
    reports_dir: Path = Path("reports")
    value_model: Path = Path("models/value.json")
    min_matches_for_value: int = 40
    trials: int = 40
    min_removed: float = 0.5
    max_collateral_pct: float = 3.0
    patterns_to_try: int = 3
    cycles: int = 1  # 0 = forever
    rest_s: float = 60.0  # pause between cycles
    notify_url: str | None = None  # e.g. https://ntfy.sh/<your-topic>
    publish_git: bool = False
    publish_branch: str = "autopilot/reports"
    cards: Path | None = None
    arena: Path | None = None
    extra: dict = field(default_factory=dict)

    @property
    def champion_file(self) -> Path:
        return self.strategies_dir / "CHAMPION"

    @property
    def auto_dir(self) -> Path:
        return self.strategies_dir / "auto"


class Player(Protocol):
    """Plays on the phone. ``BotPlayer`` is the real one; tests use a fake."""

    def play(self, strategy: Path, matches: int, data: Path, mode: str) -> list[dict]: ...

    def experiment(self, champion: Path, candidate: Path, data: Path, mode: str, *, title: str,
                   hypothesis: str) -> dict: ...


def read_champion(cfg: AutopilotConfig) -> Path:
    from ..app import resolve_strategy_path

    if cfg.champion_file.exists():
        name = cfg.champion_file.read_text().strip()
        p = resolve_strategy_path(name, root=cfg.strategies_dir.parent.parent)
        if not p.exists():
            p = cfg.strategies_dir / name
        if p.exists():
            return p
    return resolve_strategy_path(cfg.champion, root=cfg.strategies_dir.parent.parent)


def write_champion(cfg: AutopilotConfig, strategy: Path) -> None:
    try:
        rel = strategy.resolve().relative_to(cfg.strategies_dir.resolve())
    except ValueError:
        rel = strategy
    cfg.champion_file.write_text(f"{rel.as_posix()}\n")


def _wld(summaries: list[dict]) -> dict[str, int]:
    out = {"win": 0, "loss": 0, "draw": 0, "aborted": 0}
    for s in summaries:
        r = str(s.get("result") or "")
        key = "aborted" if (r.startswith("aborted") or s.get("status") == "infra_abort") else r
        out[key] = out.get(key, 0) + 1
    return out


def _summaries(root: Path) -> list[dict]:
    out = []
    for p in sorted(root.rglob("summary.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            pass
    return out


class Autopilot:
    def __init__(self, cfg: AutopilotConfig, player: Player, *, now: Callable[[], dt.datetime] | None = None,
                 log_fn: Callable[[str], None] | None = None) -> None:
        self.cfg, self.player = cfg, player
        self.now = now or (lambda: dt.datetime.now().astimezone())
        self.say = log_fn or (lambda m: log.info("autopilot: %s", m))

    # --- helpers --------------------------------------------------------------------------------
    def _cards_arena(self):
        from ..domains.clash_royale.arena import Arena
        from ..domains.clash_royale.cards import CardDB

        cards = CardDB.load(self.cfg.cards) if self.cfg.cards and Path(self.cfg.cards).exists() else None
        if cards is None:
            raise FileNotFoundError(f"card DB not found: {self.cfg.cards} (build it with tools/build_card_db.py)")
        arena = Arena.load(self.cfg.arena) if self.cfg.arena and Path(self.cfg.arena).exists() else Arena()
        return cards, arena

    def stop_requested(self) -> bool:
        return (self.cfg.data_root / "STOP").exists()

    # --- one cycle --------------------------------------------------------------------------------
    def cycle(self) -> dict[str, Any]:
        cfg = self.cfg
        stamp = self.now().strftime("%Y-%m-%d_%H%M")
        rep: dict[str, Any] = {"started": self.now().isoformat(timespec="seconds"), "stamp": stamp}
        champion = read_champion(cfg)
        rep["champion"] = champion.stem
        cards, arena = self._cards_arena()

        # 1. farm
        farm_dir = cfg.data_root / "farm" / stamp
        if cfg.farm_matches > 0 and not self.stop_requested():
            self.say(f"farming {cfg.farm_matches} {cfg.farm_mode} matches with {champion.stem}")
            sums = self.player.play(champion, cfg.farm_matches, farm_dir, cfg.farm_mode)
            rep["farm"] = {"dir": str(farm_dir), "record": _wld(sums)}
            try:
                from .retention import retain_dir

                retain_dir(farm_dir)
            except Exception as e:  # pruning is housekeeping, never fatal
                rep["farm"]["retain_error"] = str(e)
        all_farm = cfg.data_root / "farm"
        all_sums = [s for s in _summaries(all_farm) if s.get("status") == "completed"] if all_farm.exists() else []
        rep["farm_total"] = {"matches": len(all_sums), "record": _wld(all_sums)}

        # 2. diagnose: value model, digest
        vm = None
        if len(all_sums) >= cfg.min_matches_for_value:
            try:
                from ..learn.value_model import train

                m = train([all_farm])
                cfg.value_model.parent.mkdir(parents=True, exist_ok=True)
                cfg.value_model.write_text(json.dumps(m, indent=1))
                rep["value_model"] = {"matches": m["matches"], "val": {k: v for k, v in m["metrics"]["val"].items()
                                                                       if k != "calibration"}}
            except Exception as e:
                rep["value_model"] = {"error": str(e)}
        if cfg.value_model.exists():
            try:
                from ..learn.value_model import ValueModel

                vm = ValueModel.load(cfg.value_model)
            except ValueError:
                vm = None
        out_dir = cfg.reports_dir / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        if all_sums:
            from .digest import build_digest, render_agent_markdown

            # farm/<stamp>/<match>: digest this cycle's batch (or the newest one that has matches)
            stamps = sorted(d for d in all_farm.iterdir() if d.is_dir() and any(d.glob("*/summary.json")))
            root = farm_dir if farm_dir in stamps else (stamps[-1] if stamps else None)
            dig = build_digest(root, cards, arena, ledger_path=cfg.store, value_model=vm) if root else None
            if dig is not None:
                (out_dir / "digest.md").write_text(render_agent_markdown(dig))
                (out_dir / "digest.json").write_text(json.dumps(dig, indent=1, default=str))
                rep["failure_clusters"] = {k: v["count"] for k, v in dig["failure_clusters"].items()
                                           if isinstance(v, dict) and "count" in v}

        # 3-4. propose + gate
        from .tuning.targeted import scan_patterns, search_targeted

        match_dirs = sorted({p.parent for p in cfg.data_root.rglob("decisions.jsonl")})
        candidate = None
        if match_dirs and not self.stop_requested():
            counts = scan_patterns(champion, match_dirs, cards, arena)
            rep["pattern_hits"] = counts
            tried = []
            for pattern in [p for p, n in sorted(counts.items(), key=lambda kv: -kv[1]) if n > 0][: cfg.patterns_to_try]:
                out = cfg.auto_dir / f"{champion.stem}-{pattern.replace('_', '-')}-{stamp}.yaml"
                res = search_targeted(champion, match_dirs, cards, arena, pattern=pattern, trials=cfg.trials,
                                      min_removed=cfg.min_removed, max_collateral_pct=cfg.max_collateral_pct,
                                      out_yaml=out)
                tried.append({"pattern": pattern, "best": res.get("best")})
                if res.get("candidate_yaml"):
                    candidate = (Path(res["candidate_yaml"]), pattern, res["best"])
                    break
            rep["proposals"] = tried
        if candidate is None:
            rep["experiment"] = {"status": "skipped", "why": "no candidate passed the targeted gate"}
        elif not self.stop_requested():
            cand_path, pattern, best = candidate
            title = f"auto: {pattern} via {', '.join(best['knobs'])}"
            hyp = (f"Removes {best['removed']} of the champion's {pattern} decisions "
                   f"({best['removed_frac']:.0%}) with {best['collateral_pct']}% collateral")
            self.say(f"testing {cand_path.stem} vs {champion.stem} on {cfg.testbed}")
            res = self.player.experiment(champion, cand_path, cfg.data_root / "exp" / stamp, cfg.testbed,
                                         title=title, hypothesis=hyp)
            rep["experiment"] = {**res, "candidate": cand_path.stem, "pattern": pattern, "knobs": best["knobs"]}
            if res.get("status") == "accepted":
                write_champion(cfg, cand_path)
                rep["promoted"] = cand_path.stem
        rep["champion_after"] = read_champion(cfg).stem
        rep["finished"] = self.now().isoformat(timespec="seconds")

        # 7. report, notify, publish
        md = render_report(rep)
        (out_dir / "report.md").write_text(md)
        (out_dir / "report.json").write_text(json.dumps(rep, indent=1, default=str))
        (cfg.reports_dir / "latest.md").write_text(md)
        if (out_dir / "digest.md").exists():
            shutil.copy(out_dir / "digest.md", cfg.reports_dir / "latest-digest.md")
        if cfg.notify_url:
            rep["notified"] = notify(cfg.notify_url, short_summary(rep), title="droid-harness autopilot")
        if cfg.publish_git:
            rep["published"] = publish(cfg, stamp)
        return rep

    def run(self) -> list[dict]:
        reports = []
        n = 0
        while not self.stop_requested():
            reports.append(self.cycle())
            n += 1
            if self.cfg.cycles and n >= self.cfg.cycles:
                break
            time.sleep(self.cfg.rest_s)
        return reports


# --- outputs ----------------------------------------------------------------------------------------

def short_summary(rep: dict) -> str:
    farm = rep.get("farm", {}).get("record", {})
    exp = rep.get("experiment", {})
    parts = [f"Champion {rep.get('champion')}: farmed {farm.get('win', 0)}W-{farm.get('loss', 0)}L"
             if farm else f"Champion {rep.get('champion')}"]
    if exp.get("status") == "skipped":
        parts.append("no candidate passed the gate")
    elif exp:
        parts.append(f"{exp.get('candidate')} {exp.get('status')} ({exp.get('experiment_id', '')})")
    if rep.get("promoted"):
        parts.append(f"new champion {rep['promoted']}")
    return ". ".join(parts) + "."


def render_report(rep: dict) -> str:
    lines = [f"# Autopilot report {rep['stamp']}", "",
             f"- Champion at start: **{rep.get('champion')}**; after: **{rep.get('champion_after')}**"]
    if "farm" in rep:
        f = rep["farm"]["record"]
        lines.append(f"- Farmed this cycle: {f.get('win', 0)}W–{f.get('loss', 0)}L–{f.get('draw', 0)}D"
                     f" ({f.get('aborted', 0)} aborted) in `{rep['farm']['dir']}`")
    t = rep.get("farm_total", {})
    if t:
        r = t["record"]
        lines.append(f"- All farmed: {t['matches']} matches, {r.get('win', 0)}W–{r.get('loss', 0)}L–{r.get('draw', 0)}D")
    if rep.get("value_model"):
        lines.append(f"- Value model: `{json.dumps(rep['value_model'])}`")
    if rep.get("failure_clusters"):
        lines.append("- Failure clusters (digest): " + ", ".join(f"{k} {v}" for k, v in rep["failure_clusters"].items()))
    if rep.get("pattern_hits"):
        lines.append("- Champion pattern hits on logged states: "
                     + ", ".join(f"{k} {v}" for k, v in rep["pattern_hits"].items()))
    for p in rep.get("proposals", []):
        b = p.get("best")
        lines.append(f"- Proposal for `{p['pattern']}`: " + (
            f"passed with {b['knobs']} (removed {b['removed_frac']:.0%}, collateral {b['collateral_pct']}%)"
            if b else "nothing passed the gate (needs a structural rule: see the weekly agent job)"))
    exp = rep.get("experiment", {})
    if exp.get("status") == "skipped":
        lines.append(f"- Experiment: skipped ({exp.get('why')})")
    elif exp:
        lines.append(f"- Experiment {exp.get('experiment_id', '')}: **{exp.get('status')}** — {exp.get('candidate')} "
                     f"({exp.get('matches', '?')} matches; {exp.get('notes', '')})")
    if rep.get("promoted"):
        lines.append(f"- **Promoted** `{rep['promoted']}` to champion.")
    lines += ["", f"Started {rep.get('started')}, finished {rep.get('finished')}.", ""]
    return "\n".join(lines)


def notify(url: str, text: str, *, title: str = "droid-harness") -> bool:
    """POST a plain-text message (ntfy.sh style: the URL is the topic). Never raises."""
    try:
        req = urllib.request.Request(url, data=text.encode(), method="POST",
                                     headers={"Title": title, "Content-Type": "text/plain; charset=utf-8"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return 200 <= r.status < 300
    except Exception as e:
        log.warning("notify failed: %s", e)
        return False


def publish(cfg: AutopilotConfig, stamp: str, *, repo: Path = Path("."), push: bool = True) -> dict:
    """Commit the reports (and the files the agent needs) to ``publish_branch`` via a worktree."""
    wt = repo / ".autopilot-publish"

    def git(*args, cwd=repo, check=True):
        return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)

    try:
        if not wt.exists():
            git("fetch", "origin", cfg.publish_branch, check=False)
            base = f"origin/{cfg.publish_branch}" if git("rev-parse", "--verify", f"origin/{cfg.publish_branch}",
                                                         check=False).returncode == 0 else "HEAD"
            git("worktree", "add", "-B", cfg.publish_branch, str(wt), base)
        for rel in [cfg.reports_dir / stamp, cfg.reports_dir / "latest.md", cfg.reports_dir / "latest-digest.md",
                    cfg.store, cfg.store.with_name("STRATEGY_LEDGER.md"), cfg.champion_file, cfg.auto_dir]:
            src = repo / rel
            if not src.exists():
                continue
            dst = wt / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy(src, dst)
        git("add", "-A", cwd=wt)
        if git("diff", "--cached", "--quiet", cwd=wt, check=False).returncode == 0:
            return {"ok": True, "commit": None}
        git("-c", "user.name=droid-harness autopilot", "-c", "user.email=autopilot@localhost",
            "commit", "-m", f"autopilot report {stamp}", cwd=wt)
        if push:
            git("push", "-u", "origin", cfg.publish_branch, cwd=wt)
        return {"ok": True, "commit": git("rev-parse", "HEAD", cwd=wt).stdout.strip()}
    except subprocess.CalledProcessError as e:
        return {"ok": False, "error": (e.stderr or str(e))[-400:]}


# --- the real player ------------------------------------------------------------------------------

class BotPlayer:
    """Plays through ``BotRunner`` (one runner per phase; detector weights stay cached)."""

    def __init__(self, base_args: dict[str, Any], *, store: Path, db: Path | None = Path("data/warehouse.duckdb"),
                 test: str = "pairs", max_games: int = 60, futility_pairs: int = 8, ui=None) -> None:
        self.base, self.store, self.db = base_args, store, db
        self.test, self.max_games, self.futility_pairs = test, max_games, futility_pairs
        self.ui = ui
        self.current = None  # the BotRunner playing right now (the Studio follows it)

    def _bot(self, strategy: Path, data: Path, mode: str, matches: int):
        from ..app import BotRunner, RunConfig

        fields = RunConfig.__dataclass_fields__
        kw = {k: (Path(v) if isinstance(v, str) and k not in _NON_PATHS else v)
              for k, v in self.base.items() if k in fields and v is not None}
        kw.update(strategy=Path(strategy), data=Path(data), mode=mode, matches=matches)
        self.current = BotRunner(RunConfig(**kw), ui=self.ui)
        return self.current

    def ui_context(self) -> dict:
        return self.current.ui_context() if self.current is not None else {"screen": "STARTING"}

    def play(self, strategy: Path, matches: int, data: Path, mode: str) -> list[dict]:
        bot = self._bot(strategy, data, mode, matches)
        bot.game_version = bot.client_version()
        bot.run()
        return list(bot._match_summaries)

    def experiment(self, champion: Path, candidate: Path, data: Path, mode: str, *, title: str,
                   hypothesis: str) -> dict:
        from ..domains.clash_royale.policy.strategy import load_strategy
        from ..flywheel.ledger import Ledger
        from .eval.live import run_paired
        from .eval.runner import ExperimentConfig, ExperimentRunner

        bot = self._bot(champion, data, mode, self.max_games * 2)
        bot.game_version = bot.client_version()
        exp_id = Ledger(self.store).next_id()
        runner = ExperimentRunner(ExperimentConfig(
            experiment_id=exp_id, title=title, hypothesis=hypothesis, proposer="autopilot",
            champion_id=load_strategy(champion)[2], candidate_id=load_strategy(candidate)[2], test=self.test,
            game_version=bot.game_version or "", balance_epoch=str(self.base.get("balance_epoch") or ""),
            max_games=self.max_games, futility_min_pairs=self.futility_pairs,
            account_levels_id=bot.account.levels_id if bot.account else ""))
        if self.ui is not None:
            from ..ui.launch import attach_experiment

            attach_experiment(bot, runner)
        run_paired(bot, runner, champion, candidate, store=self.store, db=self.db)
        return {"experiment_id": exp_id, "status": runner.status, "matches": len(runner.matches),
                "notes": runner.notes(), "void": runner.void_reason}


_NON_PATHS = {"mode", "opponent_id", "balance_epoch", "experiment_id", "arm", "pair_id"}


def config_dict(cfg: AutopilotConfig) -> dict:
    d = asdict(cfg)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}
