"""The Studio app: settings, buttons, screenshots and the guided refine flow, in one object.

Each action turns the current settings into a ``droid-harness`` command line and runs it as
a job (``jobs.py``). Nothing here changes how the bot plays: the app is a front panel for
commands that already exist.
"""

from __future__ import annotations

import http.client
import json
import logging
import os
import secrets
import socket
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .jobs import Job, JobError, JobManager
from .phone import PhoneControl, PhoneError
from .refine import RefineFlow
from .settings import Settings

log = logging.getLogger(__name__)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class ActionError(ValueError):
    pass


class StudioApp:
    def __init__(self, root: Path = Path("."), *, settings: Settings | None = None, jobs: JobManager | None = None,
                 phone: PhoneControl | None = None) -> None:
        self.root = Path(root)
        self.state_dir = self.root / ".studio"
        self.settings = settings or Settings.load(self.state_dir / "settings.json")
        self.jobs = jobs or JobManager(self.root, logs_dir=self.state_dir / "jobs", env=self.settings.job_env)
        self.phone = phone or PhoneControl(self.settings, self.root)
        self.refine = RefineFlow(self.state_dir / "refine.json", self)
        self.token = secrets.token_urlsafe(24)
        self.recent_shots: list[dict[str, Any]] = []
        self._strat_cache: tuple[float, list[dict]] | None = None
        self._data_cache: tuple[float, list[dict]] | None = None
        self._screens_cache: tuple[str, Any] = ("", None)
        self._lock = threading.Lock()
        self.on_quit = None  # set by tests; by default quitting interrupts the main thread

    def quit(self) -> None:
        """Stop every job, then end the app the way Ctrl+C would (``run_app`` cleans up)."""
        import signal

        self.jobs.stop_all()
        if self.on_quit is not None:
            self.on_quit()
        else:
            os.kill(os.getpid(), signal.SIGINT)

    # --- state for the page ---------------------------------------------------------------
    def state(self) -> dict[str, Any]:
        fg = self.jobs.foreground()
        return {
            "settings": dict(self.settings.values),
            "jobs": self.jobs.list(),
            "foreground": fg.summary() if fg else None,
            "live": self.live_ready(),
            "phone": self.phone.status(),
            "refine": self.refine.snapshot(),
            "screenshots": self.recent_shots[:6],
            "champion": self.champion_name(),
        }

    def live_ready(self) -> bool:
        """A job is streaming and its Studio already answers (so the page never hits a dead port)."""
        job = self.jobs.live_job()
        if job is None:
            return False
        try:
            with socket.create_connection(("127.0.0.1", job.live_port), timeout=0.2):
                return True
        except OSError:
            return False

    # --- strategies and data ------------------------------------------------------------
    @property
    def strategies_dir(self) -> Path:
        return self.root / "configs/strategies"

    def rel(self, p: Path | str) -> str:
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(p)

    def resolve_strategy(self, s: str) -> str:
        from ..app import resolve_strategy_path

        p = resolve_strategy_path(s or "champion", root=self.root)
        if not p.is_absolute():
            p = self.root / p if not p.exists() else p
        if not Path(p).exists():
            raise ActionError(f"no strategy {s!r} (looked in configs/strategies)")
        return self.rel(p)

    def champion_name(self) -> str | None:
        f = self.strategies_dir / "CHAMPION"
        return f.read_text().strip() if f.exists() else None

    def strategies(self, refresh: bool = False) -> list[dict[str, Any]]:
        files = sorted(f for f in [*self.strategies_dir.glob("*.yaml"), *self.strategies_dir.glob("auto/*.yaml")]
                       if f.name != "legacy_ids.yaml")
        stamp = max((f.stat().st_mtime for f in files), default=0.0) + len(files)
        if self._strat_cache and self._strat_cache[0] == stamp and not refresh:
            return self._strat_cache[1]
        from ..domains.clash_royale.policy.strategy import load_strategy

        champ = self.champion_name()
        out = []
        for f in files:
            rel = f.relative_to(self.strategies_dir).as_posix()
            row: dict[str, Any] = {"path": self.rel(f), "file": rel, "champion": rel == champ}
            try:
                spec, _, sid = load_strategy(f)
                row.update(name=spec.strategy, id=sid, parent=getattr(spec, "parent", None))
            except Exception as e:  # noqa: BLE001 - a broken file is shown, not fatal
                row.update(name=f.stem, id=None, error=str(e)[:200])
            out.append(row)
        self._strat_cache = (stamp, out)
        return out

    def datasets(self, refresh: bool = False) -> list[dict[str, Any]]:
        """Folders under data/ that directly hold recorded matches, newest first."""
        now = time.monotonic()
        if self._data_cache and now - self._data_cache[0] < 10 and not refresh:
            return self._data_cache[1]
        base = self.root / "data"
        groups: dict[Path, dict[str, Any]] = {}
        if base.is_dir():
            for dirpath, dirnames, filenames in os.walk(base):
                depth = len(Path(dirpath).relative_to(base).parts)
                if depth >= 4:
                    dirnames[:] = []
                if "decisions.jsonl" not in filenames:
                    continue
                d = Path(dirpath)
                g = groups.setdefault(d.parent, {"path": self.rel(d.parent), "matches": 0, "win": 0, "loss": 0,
                                                  "newest": 0.0, "strategies": {}})
                g["matches"] += 1
                g["newest"] = max(g["newest"], (d / "decisions.jsonl").stat().st_mtime)
                try:
                    sm = json.loads((d / "summary.json").read_text())
                except (OSError, ValueError):
                    sm = {}
                r = sm.get("result")
                if r in ("win", "loss"):
                    g[r] += 1
                sid = sm.get("strategy_id")
                if sid:
                    g["strategies"][sid] = g["strategies"].get(sid, 0) + 1
                dirnames[:] = []  # a match folder holds no further matches
        out = sorted(groups.values(), key=lambda g: g["newest"], reverse=True)
        self._data_cache = (now, out)
        return out

    # --- actions ------------------------------------------------------------------------
    def _common(self) -> list[str]:
        return ["--device", str(self.settings["device"])]

    def _cards(self) -> list[str]:
        c = str(self.settings.get("cards") or "")
        return ["--cards", c] if c else []

    def build(self, kind: str, p: dict[str, Any]) -> dict[str, Any]:
        """Command line and job options for an action, from settings plus per-call ``p``."""
        st = self.settings
        fps = ["--ui-fps", f"{float(st['ui_fps']):g}"]
        strat = lambda key="strategy": self.resolve_strategy(str(p.get(key) or st["champion"]))  # noqa: E731
        data = str(p.get("data") or st["data_dir"])
        if kind == "start":
            # One Start button. A battle on the phone is played with the selected strategy, with
            # no mode to choose; on any other screen Start is a preview, as before.
            if self.phone_in_battle():
                return {**self.build("play", {**p, "join": True}), "kind": "play"}
            return {**self.build("preview", p), "kind": "preview"}
        if kind == "preview":
            args = ["watch", *self._common(), "--strategy", strat(), "--data", "data/watch", *fps, *self._cards()]
            if not st["preview_detector"]:
                args.append("--no-detector")
            return {"title": "Preview (watching, no taps from the bot)", "args": args, "foreground": True,
                    "live": True, "ui_flag": False}
        if kind == "demo":
            return {"title": "Demo without a phone", "args": ["sim", "--seconds", "1800", "--out", "data/sim", *fps],
                    "foreground": True, "live": True}
        if kind == "play":
            n = int(p.get("matches") or st["play_matches"])
            mode = str(p.get("mode") or st["play_mode"])
            # A battle already on the phone is played as it is (``join``: True forces it, False
            # skips the check, unset asks the phone). The job then ends on HOME when it is over.
            join = True if p.get("join") else (False if p.get("join") is False else self.phone_in_battle())
            if join:
                n = 1
            args = ["play", *self._common(), "--strategy", strat(), "--mode", mode, "--matches", str(n),
                    "--data", data, *fps, *self._cards()]
            if join:
                args.append("--join-battle")
            if st["prune_win_video"]:
                args.append("--prune-win-video")
            if st["auto_upgrade"] and not join:
                args.append("--auto-upgrade")  # never navigate menus around a battle in progress
            title = (f"Play the battle on the phone with {Path(strat()).stem}" if join else
                     f"Play {n} {'match' if n == 1 else 'matches'} ({mode.replace('_', ' ')})")
            return {"title": title, "args": args, "foreground": True, "live": True}
        if kind == "upgrade":
            args = ["upgrade", *self._common()] + (["--dry-run"] if p.get("dry_run") else [])
            return {"title": "Check upgrades" if p.get("dry_run") else "Upgrade cards (gold only)", "args": args,
                    "foreground": True}
        if kind == "recover":
            return {"title": "Recover to the home screen", "args": ["recover", *self._common()], "foreground": True}
        if kind == "doctor":
            args = ["doctor", *self._common(), "--strategy", strat(), "--live"]
            if p.get("mode"):
                args += ["--mode", str(p["mode"])]
            return {"title": "Check setup", "args": args}
        if kind == "scan":
            return {"title": "Speed 1: count failure patterns",
                    "args": ["tune-targeted", *self._common(), "--champion", strat("champion"), "--data", data,
                             "--scan", *self._cards()]}
        if kind == "propose":
            out = str(p.get("out") or "")
            if not out:
                raise ActionError("propose needs an output file")
            return {"title": f"Speed 1: search a fix for {str(p.get('pattern', 'auto')).replace('_', ' ')}",
                    "args": ["tune-targeted", *self._common(), "--champion", strat("champion"), "--data", data,
                             "--pattern", str(p.get("pattern") or "auto"), "--trials", str(int(st["trials"])),
                             "--min-removed", f"{float(st['min_removed']):g}",
                             "--max-collateral", f"{float(st['max_collateral']):g}", "--out", out, *self._cards()]}
        if kind in ("shadow", "replay_diff"):
            args = ["shadow" if kind == "shadow" else "replay-diff", *self._common(), "--champion", strat("champion"),
                    "--candidate", strat("candidate"), "--data", data, *self._cards()]
            if kind == "replay_diff":
                args += ["--expected-pct", f"{float(p.get('expected_pct') or 5.0):g}"]
            return {"title": "Speed 1: " + ("shadow comparison" if kind == "shadow" else "replay diff"),
                    "args": args}
        if kind == "experiment":
            mode = str(p.get("mode") or st["test_mode"])
            args = ["experiment", *self._common(), "--champion", strat("champion"), "--candidate", strat("candidate"),
                    "--mode", mode, "--test", str(p.get("test") or st["test"]),
                    "--max-games", str(int(p.get("max_games") or st["max_games"])),
                    "--futility-pairs", str(int(p.get("futility_pairs", st["futility_pairs"]))),
                    "--data", data, "--proposer", "studio", *fps, *self._cards()]
            if p.get("title"):
                args += ["--title", str(p["title"])]
            if p.get("hypothesis"):
                args += ["--hypothesis", str(p["hypothesis"])]
            if st["prune_win_video"]:
                args.append("--prune-win-video")
            return {"title": f"Speed 2: test on the phone ({mode.replace('_', ' ')})", "args": args,
                    "foreground": True, "live": True}
        raise ActionError(f"unknown action {kind!r}")

    PHONE_KINDS = {"start", "preview", "play", "upgrade", "recover", "experiment"}

    def phone_holder(self) -> dict | None:
        """Another process using the phone (the autopilot service, a terminal run), or None."""
        from ..core.transport.device_lock import holder

        try:
            serial = self.phone.serial()
            info = holder(serial) if serial else None
        except (PhoneError, OSError):
            return None
        mine = {j.proc.pid for j in self.jobs.jobs.values() if j.active and j.proc is not None}
        return None if not info or info.get("pid") in mine or info.get("pid") == os.getpid() else info

    def run(self, kind: str, params: dict[str, Any] | None = None, *, on_done=None) -> Job:
        spec = self.build(kind, dict(params or {}))
        kind = spec.get("kind", kind)  # Start resolves to play (a battle is on) or preview
        if kind in self.PHONE_KINDS:
            other = self.phone_holder()
            if other:
                from ..core.transport.device_lock import describe

                raise ActionError(f"the phone is in use by {describe(other)}. Stop that first "
                                  "(for the autopilot service: systemctl --user stop droid-autopilot).")
        try:
            return self.jobs.start(kind, spec["title"], spec["args"], foreground=spec.get("foreground", False),
                                   live=spec.get("live", False), ui_flag=spec.get("ui_flag", True),
                                   params=dict(params or {}), on_done=on_done)
        except JobError as e:
            raise ActionError(str(e)) from e

    # --- what is on the phone --------------------------------------------------------------
    def _screen_classifier(self):
        """The device's screen classifier, as the bot builds it (built once per device; None when
        the calibration is missing, in which case Start can't tell and plays the normal flow)."""
        device = str(self.settings["device"])
        if self._screens_cache[0] != device:
            sc = None
            try:
                from ..app import screen_classifier

                sc = screen_classifier(device, root=self.root)
            except Exception as e:  # noqa: BLE001 - the job itself reports what calibration is missing
                log.warning("can't classify the phone's screen for Start (%s): %s", device, e)
            self._screens_cache = (device, sc)
        return self._screens_cache[1]

    def phone_in_battle(self) -> bool:
        """True when the phone shows a battle right now. Read from the running live view when
        there is one, else one adb screenshot is classified (the elixir bar recognises a battle
        in any arena). Nothing is tapped; an unreachable phone reads as "no battle"."""
        sc = self._screen_classifier()
        if sc is None:
            return False
        if self.jobs.live_job() is not None:
            live = self._live_screen()
            if live:
                return live.upper() == sc.battle_label
        try:
            png = self.phone.screencap()
        except (PhoneError, OSError) as e:
            log.warning("can't read the phone's screen for Start: %s", e)
            return False
        img = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return False
        return sc.classify(img)[0] == sc.battle_label

    # --- screenshots ----------------------------------------------------------------------
    def _live_get(self, path: str, timeout: float = 5.0) -> tuple[int, bytes]:
        job = self.jobs.live_job()
        if job is None:
            raise PhoneError("no live view is running")
        c = http.client.HTTPConnection("127.0.0.1", job.live_port, timeout=timeout)
        try:
            c.request("GET", path)
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def screenshot(self, *, overlay_png: bytes | None = None, frame_png: bytes | None = None,
                   label: str | None = None) -> dict[str, Any]:
        """Save the preview's current frame as a PNG (and a copy with overlays when given).

        The clean frame comes from, in order: ``frame_png`` (the page sends it for recorded
        matches), the running job's lossless stream frame, or ``adb screencap``.
        """
        for name, blob in (("overlay", overlay_png), ("frame", frame_png)):
            if blob is not None and (not blob.startswith(PNG_MAGIC) or len(blob) > 40_000_000):
                raise ActionError(f"the {name} image isn't a PNG")
        source, tag, png = "replay", (label or "replay"), frame_png
        if png is None and self.jobs.live_job() is not None:
            try:
                status, body = self._live_get("/api/live/frame.png")
                if status == 200 and body.startswith(PNG_MAGIC):
                    png, source = body, "stream"
                    tag = label or self._live_screen() or "phone"
            except OSError:
                png = None
        if png is None:
            png, source, tag = self.phone.screencap(), "phone", label or "phone"
        folder = self.root / str(self.settings["screenshot_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        t = time.time()
        stem = time.strftime("%Y%m%d-%H%M%S", time.localtime(t)) + f"-{int(t * 1000) % 1000:03d}-{_slug(tag)}"
        files = []
        with self._lock:
            path = folder / f"{stem}.png"
            path.write_bytes(png)
            files.append(self.rel(path))
            if overlay_png is not None:
                op = folder / f"{stem}-overlay.png"
                op.write_bytes(overlay_png)
                files.append(self.rel(op))
        w, h = _png_size(png)
        shot = {"files": files, "source": source, "w": w, "h": h, "t": t, "folder": self.rel(folder)}
        self.recent_shots.insert(0, shot)
        del self.recent_shots[12:]
        return shot

    def _live_screen(self) -> str | None:
        try:
            status, body = self._live_get("/api/live/snapshot", timeout=2.0)
            if status == 200:
                snap = json.loads(body)
                return str(snap.get("screen") or "").lower() or None
        except (OSError, ValueError):
            pass
        return None

    # --- promotion ------------------------------------------------------------------------
    def promote(self, candidate: str) -> str:
        from ..flywheel.autopilot import AutopilotConfig, write_champion

        p = self.root / self.resolve_strategy(candidate)
        write_champion(AutopilotConfig(strategies_dir=self.strategies_dir), p)
        self._strat_cache = None
        return self.champion_name() or ""


def _slug(s: str) -> str:
    keep = "".join(c if c.isalnum() else "-" for c in str(s).lower()).strip("-")
    return "-".join(p for p in keep.split("-") if p)[:40] or "frame"


def _png_size(png: bytes) -> tuple[int | None, int | None]:
    if len(png) >= 24 and png.startswith(PNG_MAGIC):
        return int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    return None, None
