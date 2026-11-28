"""The guided refine flow: Speed 1 (offline) into Speed 2 (on the phone), one step at a time.

Steps, each a real command run as a job:

1. data     pick the champion and a folder of recorded matches (or play more)
2. scan     ``tune-targeted --scan``: how often the champion hits each failure pattern
3. propose  ``tune-targeted --pattern P --out <new file>``: a one- or two-knob fix
4. check    ``shadow`` and ``replay-diff``: what the fix changes on logged states
5. test     ``experiment``: champion vs candidate in paired games on the phone
6. decide   promote the candidate (CHAMPION) or start another round

State is saved to ``.studio/refine.json`` after every change, so the flow survives a restart.
Changing an earlier step clears the ones after it.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .control import StudioApp

STEPS = ["data", "scan", "propose", "check", "test", "decide"]
PATTERNS = {
    "opening_throw": "An early play in the first 15 s that leaves you under 3.2 elixir while they may hold 6.5+.",
    "bad_counter_timing": "Your win condition goes in while their shown hard counter is in hand and affordable.",
    "over_defend_bleed": "Defending a push worth under 3 elixir down to under 2 elixir.",
    "wincon_starved": "8.5+ elixir, win condition ready, empty board, and something else played.",
    "spell_inefficiency": "A spell whose own value estimate is negative.",
}
MIN_MATCHES = 20


class FlowError(ValueError):
    pass


def _step() -> dict[str, Any]:
    return {"status": "idle", "job": None, "result": None, "error": None, "at": None}


class RefineFlow:
    def __init__(self, path: Path, app: StudioApp) -> None:
        self.path, self.app = Path(path), app
        self._lock = threading.RLock()
        self.state = self._load()

    # --- persistence --------------------------------------------------------------------
    def _blank(self, *, champion: str | None = None, data: str | None = None, rnd: int = 1,
               history: list | None = None) -> dict[str, Any]:
        return {"round": rnd, "champion": champion, "data": data, "pattern": None, "candidate": None,
                "scan": _step(), "propose": _step(), "shadow": _step(), "replay_diff": _step(), "test": _step(),
                "promoted": None, "history": history or [], "updated": time.time()}

    def _load(self) -> dict[str, Any]:
        try:
            st = json.loads(self.path.read_text())
            if not isinstance(st, dict) or "scan" not in st:
                raise ValueError("not a refine state")
        except (OSError, ValueError):
            return self._blank()
        for k in ("scan", "propose", "shadow", "replay_diff", "test"):
            st.setdefault(k, _step())
            if st[k].get("status") == "running":  # the app restarted while it ran
                st[k]["status"] = "interrupted"
        return st

    def _save(self) -> None:
        self.state["updated"] = time.time()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(self.path)

    # --- view ---------------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            st = json.loads(json.dumps(self.state, default=str))
        st["current"] = self.current_step(st)
        st["patterns"] = PATTERNS
        st["min_matches"] = MIN_MATCHES
        return st

    @staticmethod
    def current_step(st: dict[str, Any]) -> str:
        if not st.get("champion") or not st.get("data"):
            return "data"
        if st["scan"]["status"] != "done":
            return "scan"
        if not st.get("candidate"):
            return "propose"
        if st["shadow"]["status"] != "done" or st["replay_diff"]["status"] not in ("done", "failed"):
            return "check"
        if st["test"]["status"] != "done":
            return "test"
        return "decide"

    # --- steps --------------------------------------------------------------------------
    def configure(self, champion: str, data: str) -> dict[str, Any]:
        champ = self.app.resolve_strategy(champion)
        if not data or not (self.app.root / data).is_dir():
            raise FlowError(f"no match folder {data!r}")
        with self._lock:
            if (champ, data) != (self.state.get("champion"), self.state.get("data")):
                self.state = self._blank(champion=champ, data=data, rnd=self.state.get("round", 1),
                                         history=self.state.get("history"))
                self._save()
        return self.snapshot()

    def _require(self, *keys: str) -> None:
        for k in keys:
            if not self.state.get(k):
                raise FlowError(f"choose the {k} first")

    def _busy(self, *steps: str) -> None:
        for s in steps:
            if self.state[s]["status"] == "running":
                raise FlowError("that step is still running")

    def _launch(self, step: str, kind: str, params: dict[str, Any], after=None) -> dict[str, Any]:
        def done(job) -> None:
            with self._lock:
                rec = self.state.get(step) or {}
                if rec.get("job") != job.id:
                    return  # a newer run replaced this one
                ok = job.status == "done" or (kind == "replay_diff" and isinstance(job.result, dict))
                rec.update(status="done" if ok else ("stopped" if job.status == "stopped" else "failed"),
                           result=job.result, error=None if ok else (job.error or job.status), at=time.time())
                if ok and after is not None:
                    after(job.result)
                self._save()

        with self._lock:  # held until the step records the job, so a fast job's callback waits for it
            job = self.app.run(kind, params, on_done=done)
            self.state[step] = {"status": "running" if job.active else "failed", "job": job.id, "result": None,
                                "error": job.error, "at": time.time()}
            self._save()
            if not job.active:  # failed to start: its callback ran before the step knew the job id
                done(job)
        return self.snapshot()

    def scan(self) -> dict[str, Any]:
        with self._lock:
            self._require("champion", "data")
            self._busy("scan", "propose")
            for k in ("propose", "shadow", "replay_diff", "test"):
                self.state[k] = _step()
            self.state.update(pattern=None, candidate=None, promoted=None)

        def after(res: Any) -> None:
            if isinstance(res, dict) and res:
                self.state["pattern"] = max(res, key=lambda k: res[k]) if any(res.values()) else None

        return self._launch("scan", "scan", {"champion": self.state["champion"], "data": self.state["data"]}, after)

    def propose(self, pattern: str) -> dict[str, Any]:
        if pattern not in PATTERNS:
            raise FlowError(f"unknown pattern {pattern!r}")
        with self._lock:
            if self.state["scan"]["status"] != "done":
                raise FlowError("count the failure patterns first")
            self._busy("propose", "test")
            champ = Path(self.state["champion"]).stem
            out = f"configs/strategies/{champ}-{pattern.replace('_', '-')}-{time.strftime('%Y%m%d-%H%M%S')}.yaml"
            for k in ("shadow", "replay_diff", "test"):
                self.state[k] = _step()
            self.state.update(pattern=pattern, candidate=None, promoted=None)

        def after(res: Any) -> None:
            cand = res.get("candidate_yaml") if isinstance(res, dict) else None
            self.state["candidate"] = self.app.rel(cand) if cand else None

        return self._launch("propose", "propose", {"champion": self.state["champion"], "data": self.state["data"],
                                                   "pattern": pattern, "out": out}, after)

    def expected_pct(self) -> float:
        res = self.state["propose"].get("result") or {}
        best, states = res.get("best") or {}, res.get("states") or 0
        if not best or not states:
            return 5.0
        return round(float(best.get("collateral_pct") or 0) + 100.0 * float(best.get("removed") or 0) / states, 2)

    def check(self) -> dict[str, Any]:
        with self._lock:
            self._require("candidate")
            self._busy("shadow", "replay_diff", "test")
            self.state["test"] = _step()
            params = {"champion": self.state["champion"], "candidate": self.state["candidate"],
                      "data": self.state["data"]}
        self._launch("shadow", "shadow", params)
        return self._launch("replay_diff", "replay_diff", {**params, "expected_pct": max(0.5, self.expected_pct())})

    def test(self, *, title: str = "", hypothesis: str = "", **overrides: Any) -> dict[str, Any]:
        with self._lock:
            self._require("candidate")
            if self.state["shadow"]["status"] != "done":
                raise FlowError("run the offline check first")
            self._busy("test")
            pat = (self.state.get("pattern") or "").replace("_", " ")
            res = self.state["propose"].get("result") or {}
            knobs = ", ".join(f"{k} {v}" for k, v in ((res.get("best") or {}).get("knobs") or {}).items())
            params = {"champion": self.state["champion"], "candidate": self.state["candidate"],
                      "data": self.state["data"],
                      "title": title or f"studio: fewer {pat}",
                      "hypothesis": hypothesis or (f"{knobs} removes {pat} without losing more" if knobs else ""),
                      **{k: v for k, v in overrides.items() if k in ("mode", "test", "max_games", "futility_pairs")}}
        return self._launch("test", "experiment", params)

    def promote(self) -> dict[str, Any]:
        with self._lock:
            res = self.state["test"].get("result") or {}
            if self.state["test"]["status"] != "done" or not self.state.get("candidate"):
                raise FlowError("finish the phone test first")
            if str(res.get("status", "")).lower() != "accepted":
                raise FlowError("only an accepted candidate can be promoted from the guided flow")
            self.state["promoted"] = self.app.promote(self.state["candidate"])
            self._save()
        return self.snapshot()

    def next_round(self) -> dict[str, Any]:
        """Archive this round and start the next one from the current champion."""
        with self._lock:
            for k in ("scan", "propose", "shadow", "replay_diff", "test"):
                self._busy(k)
            st = self.state
            if st.get("candidate") or st["scan"]["status"] == "done":
                test = st["test"].get("result") or {}
                st["history"] = [{"round": st.get("round", 1), "champion": st.get("champion"),
                                  "pattern": st.get("pattern"), "candidate": st.get("candidate"),
                                  "test": test.get("status"), "experiment": test.get("experiment_id"),
                                  "promoted": st.get("promoted"), "at": time.time()}, *st.get("history", [])][:20]
            champion = self.app.resolve_strategy(st["candidate"]) if st.get("promoted") else st.get("champion")
            self.state = self._blank(champion=champion, data=st.get("data"), rnd=st.get("round", 1) + 1,
                                     history=st.get("history"))
            self._save()
        return self.snapshot()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self.state = self._blank(history=self.state.get("history"))
            self._save()
        return self.snapshot()
