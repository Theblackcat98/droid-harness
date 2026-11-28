"""Wiring: build a running Clash Royale bot (or the synthetic demo) from config files."""

from __future__ import annotations

import logging
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

from .core.actuate.executor import Executor
from .core.config import DeviceProfile
from .core.runtime.liveness import Finding, LivenessMonitor
from .core.runtime.orchestrator import BrainLoop
from .core.runtime.watchdog import AppWatchdog
from .core.telemetry.recorder import Recorder
from .core.transport.device_lock import DeviceBusy, DeviceLock
from .core.transport.scrcpy_bridge import ScrcpyClient
from .domains.clash_royale.account import check_strategy, load_account
from .domains.clash_royale.actions import CRActionMapper
from .domains.clash_royale.arena import Arena
from .domains.clash_royale.cards import CardDB
from .domains.clash_royale.menus import BATTLE_SCREENS, DEFAULT_FLOWS, MenuNavigator
from .domains.clash_royale.perception.cards import CardMatcher
from .domains.clash_royale.perception.classes import missing_unit_classes
from .domains.clash_royale.perception.detector import load_detector, warmup_once
from .domains.clash_royale.perception.digits import DigitReader
from .domains.clash_royale.perception.grid import Layout, Roi, TileMapper
from .domains.clash_royale.perception.harvest import TemplateHarvester
from .domains.clash_royale.perception.pipeline import CRPerception
from .domains.clash_royale.perception.screens import ScreenClassifier
from .domains.clash_royale.perception.towers import TowerStateReader
from .domains.clash_royale.policy.strategy import load_strategy
from .domains.clash_royale.policy.utility import CRPolicy
from .domains.clash_royale.recovery import walk_to_home
from .domains.clash_royale.session import CRSession, MatchMeta
from .domains.clash_royale.upgrader import run_auto_upgrade
from .flywheel.retention import retain_match
from .learn.train_ranker import load_ranker

log = logging.getLogger(__name__)
PACKAGE = "com.supercell.clashroyale"


@dataclass
class RunConfig:
    profile: Path
    layout: Path
    tiles: Path
    templates: Path
    screens: Path
    strategy: Path
    cards: Path
    arena: Path | None = None
    data: Path = Path("data/matches")
    detector: Path | None = None
    classes: Path | None = None
    digits: Path | None = None
    mode: str = "training_camp"
    opponent_id: str | None = None
    matches: int = 10
    balance_epoch: str | None = None
    experiment_id: str | None = None
    arm: str | None = None
    pair_id: str | None = None
    detect_every: int = 1
    det_conf: float = 0.35
    det_iou: float = 0.5
    async_detect: bool = False
    towers: Path | None = None  # tower-state templates dir (droid-harness tower-template)
    prune_win_video: bool = False
    # `droid-harness watch`: perceive and decide, but never touch the phone (no taps, no BACK,
    # no app restarts). You play; the Studio shows what the bot sees and would do.
    observe_only: bool = False
    harvest_templates: Path | None = None  # save crops of unreadable new cards here (promote-templates)
    auto_upgrade: bool = False  # upgrade ready Battle Deck cards & Tower Princess between farm matches
    auto_upgrade_every: int = 4  # matches between mid-batch upgrade passes (0 = start of batch only)
    auto_upgrade_max: int = 10  # max upgrades per pass
    reconnect_deadline_s: float = 300.0  # keep retrying a dropped phone this long before giving up (#17)


def resolve_strategy_path(strategy: str | Path, root: Path = Path(".")) -> Path:
    """Resolve short aliases ('v6', 'ladder-vision-v6') or direct paths to a strategy YAML."""
    p = Path(strategy)
    if p.exists():
        return p
    s = str(strategy).strip()
    strat_dir = root / "configs/strategies"
    if s == "champion" and (strat_dir / "CHAMPION").exists():  # the autopilot's current best
        s = (strat_dir / "CHAMPION").read_text().strip()
        if (strat_dir / s).exists():
            return strat_dir / s
    candidates = [
        strat_dir / f"{s}.yaml",
        strat_dir / f"ladder-vision-{s}.yaml",
        strat_dir / s,
    ]
    for c in candidates:
        if c.exists():
            return c
    return p


def resolve_device_bundle(device: str = "phone-a", root: Path = Path("."), *,
                          with_detector: bool = True) -> dict[str, Path | None]:
    """Auto-discover standard per-device calibration paths under ``configs/`` and ``models/``."""
    cr = root / "configs/domains/clash_royale"
    arena = cr / "arena" / f"{device}.yaml"
    towers = cr / "towers" / device
    digits = cr / "digits" / device
    det_pt = root / "models/yolo26n-cr.pt"
    det_onnx = root / "models/yolo26n-cr.onnx"
    classes = root / "models/classes.txt"
    detector: Path | None = None
    if with_detector:
        if det_pt.exists():
            detector = det_pt
        elif det_onnx.exists():
            detector = det_onnx
    return {
        "profile": root / "configs/device_profiles" / f"{device}.yaml",
        "layout": cr / "rois" / f"{device}.yaml",
        "tiles": cr / "tiles" / f"{device}.yaml",
        "arena": arena if arena.exists() else None,
        "templates": cr / "templates" / device,
        "screens": cr / "screens" / device,
        "towers": towers if towers.exists() else None,
        "digits": digits if digits.exists() else None,
        "cards": cr / "cards.json",
        "detector": detector,
        "classes": classes if (detector is not None and classes.exists()) else None,
    }


def run_doctor(*, device: str = "phone-a", strategy: str | Path = "champion",
               root: Path = Path("."), live: bool = False, mode: str | None = None,
               strict: bool = False) -> dict[str, Any]:
    """Pre-flight verification of calibration, templates, digits, detector, disk, and ADB.

    ``problems`` stop a run from working (no adb, missing templates or detector, no disk);
    ``warnings`` are worth fixing but don't (a strategy's tower damage or card levels that
    disagree with the account file). ``ok`` ignores warnings unless ``strict`` (issue #9)."""
    bundle = resolve_device_bundle(device, root=root, with_detector=True)
    strat_path = resolve_strategy_path(strategy, root=root)
    checks: dict[str, Any] = {}
    problems: list[str] = []
    warnings: list[str] = []

    for key in ("profile", "layout", "tiles", "templates", "screens", "cards"):
        p = bundle[key]
        ok = p is not None and p.exists()
        checks[key] = str(p) if ok else f"MISSING ({p})"
        if not ok:
            problems.append(f"missing required {key}: {p}")

    if mode:
        from .domains.clash_royale.menus import required_for

        layout_p = bundle["layout"]
        raw = yaml.safe_load(layout_p.read_text()) if layout_p and layout_p.exists() else {}
        flows = {mode: {**DEFAULT_FLOWS.get(mode, {}), **((raw.get("flows") or {}).get(mode) or {})}}
        need_buttons, need_screens = required_for(mode, flows)
        have_buttons = set((raw.get("buttons") or {}).keys())
        sdir = bundle["screens"]
        have_screens = {p.stem.split("__")[0] for p in sdir.glob("*.png")} if sdir and sdir.exists() else set()
        slots = {b for b in need_buttons if b.startswith("practice_slot_")}
        miss_b = sorted((need_buttons - slots) - have_buttons)
        if slots and not slots & have_buttons:
            miss_b.append("practice_slot_0 (at least one battle-log entry)")
        miss_s = sorted(need_screens - have_screens)
        checks["mode"] = {"mode": mode, "missing_buttons": miss_b, "missing_screens": miss_s}
        if mode not in DEFAULT_FLOWS:
            problems.append(f"unknown mode {mode!r}")
        elif miss_b or miss_s:
            problems.append(f"mode {mode}: calibrate buttons {miss_b} and screens {miss_s}")

    for opt_key in ("arena", "towers", "digits"):
        p = bundle[opt_key]
        checks[opt_key] = str(p) if (p and p.exists()) else "not configured"

    if bundle["towers"] is not None and bundle["layout"] is not None and bundle["layout"].exists():
        try:
            TowerStateReader.load(bundle["towers"], rois=Layout.load(bundle["layout"]).towers)
        except Exception as e:
            checks["towers"] = f"ERROR: {e}"
            problems.append(f"tower templates invalid: {e}")

    if bundle["digits"] is not None:
        dr = load_digits(bundle["digits"])
        checks["digit_templates"] = 10 if dr is not None else "incomplete (<10 digits)"
        if dr is None:
            problems.append(f"digits directory {bundle['digits']} is missing one or more of 0.png..9.png")
    else:
        problems.append(f"missing tower-HP digits directory configs/domains/clash_royale/digits/{device}")

    if strat_path.exists():
        spec, params, sid = load_strategy(strat_path)
        checks["strategy"] = {"name": spec.strategy, "id": sid, "path": str(strat_path), "deck": params.deck}
        if bundle["profile"] and bundle["profile"].exists():
            account = yaml.safe_load(bundle["profile"].read_text()).get("account")
            cards_db = CardDB.load(bundle["cards"]) if bundle["cards"] and bundle["cards"].exists() else None
            acct = load_account(account, root)
            checks["account"] = check_strategy(params, cards_db, acct)
            warnings += [f"account: {p}" for p in checks["account"]["problems"]]
            if checks["account"].get("card_levels_missing"):
                warnings.append(f"account: no level recorded for {checks['account']['card_levels_missing']} "
                                "(level-aware strategies fall back to your King Level)")
        tdir = bundle["templates"]
        if tdir and tdir.exists():
            matcher = CardMatcher.load(tdir)
            have = set(matcher.templates.keys())
            missing_cards = [c for c in params.deck if c not in have]
            checks["deck_templates_missing"] = missing_cards
            if missing_cards:
                problems.append(f"hand templates missing for deck cards: {missing_cards}")
    else:
        checks["strategy"] = f"MISSING ({strat_path})"
        problems.append(f"strategy file not found: {strat_path}")

    if bundle["detector"] is not None:
        try:
            det = load_detector(bundle["detector"], bundle["classes"])
            checks["detector"] = {"path": str(bundle["detector"]), "provider": det.provider,
                                  "classes": len(det.class_names), "imgsz": det.imgsz}
        except Exception as e:
            checks["detector"] = f"ERROR: {e}"
            problems.append(f"detector failed to load: {e}")
    else:
        checks["detector"] = "none (blind play only)"

    free_gb = round(shutil.disk_usage(root).free / (1024 ** 3), 2)
    checks["disk_free_gb"] = free_gb
    if free_gb < 2.0:
        problems.append(f"low disk space ({free_gb} GB free); run `droid-harness retain` before farming")

    if live and bundle["profile"] and bundle["profile"].exists():
        from .core.transport.adb import Adb

        prof = DeviceProfile(**{**yaml.safe_load(bundle["profile"].read_text()), "name": device})
        try:
            out = Adb(prof.serial).run("get-state", check=False).strip()
            checks["adb_state"] = out or "offline"
            if out != "device":
                problems.append(f"adb serial {prof.serial} not ready (got {out!r})")
        except Exception as e:
            checks["adb_state"] = f"error: {e}"
            problems.append(f"adb check failed: {e}")

    ok = not problems and not (strict and warnings)
    return {"ok": ok, "device": device, "checks": checks, "problems": problems, "warnings": warnings}


def load_screens(directory: Path, regions: list[Roi], *, battle_roi: Roi | None = None,
                 battle_hsv: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None) -> ScreenClassifier:
    kw = {}
    if battle_roi is not None:
        kw["battle_roi"] = battle_roi
    if battle_hsv is not None:
        kw["battle_hsv_lo"], kw["battle_hsv_hi"] = battle_hsv
    sc = ScreenClassifier(regions, **kw)
    for p in sorted(Path(directory).glob("*.png")):
        sc.register(p.stem.split("__")[0], cv2.imread(str(p)))
    return sc


def load_digits(directory: Path | None) -> DigitReader | None:
    if directory is None:
        return None
    r = DigitReader()
    for d in "0123456789":
        p = Path(directory) / f"{d}.png"
        if p.exists():
            r.templates[d] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return r if len(r.templates) == 10 else None


class RunAborted(RuntimeError):
    """The run can't continue (recording failed, the phone stayed unreachable, the phone is busy).

    The CLI prints the message and exits with status 2 instead of a traceback."""


class BotRunner:
    """One phone, N matches, with liveness-driven recovery."""

    def __init__(self, cfg: RunConfig, *, ui=None) -> None:
        self.cfg = cfg
        self.ui = ui  # droid_harness.ui.telemetry.TelemetryTap (Studio), or None
        self.ui_experiment: Callable[[], dict | None] | None = None
        self.profile = DeviceProfile(**{**yaml.safe_load(cfg.profile.read_text()), "name": cfg.profile.stem})
        self.cards = CardDB.load(cfg.cards)
        # Your card/King levels (configs/accounts/<account>.yaml): recorded on every match and used by
        # strategies that set level_aware. Missing file = levels unknown (nothing changes).
        self.account = load_account(self.profile.account)
        self.arena = Arena.load(cfg.arena) if cfg.arena else Arena()
        self.layout = Layout.load(cfg.layout)
        self.tiles = TileMapper.load(cfg.tiles)
        self._policy_cache: dict[Path, tuple[CRPolicy, str, Any]] = {}
        regions = [Roi(*r) for r in yaml.safe_load(cfg.layout.read_text()).get("screen_regions",
                                                                                 [[0, 0, 1, 0.5], [0, 0.5, 1, 1]])]
        self.screens = load_screens(cfg.screens, regions, battle_roi=self.layout.elixir_bar,
                                    battle_hsv=(self.layout.elixir_hsv_lo, self.layout.elixir_hsv_hi))
        self.detector = None
        if cfg.detector:
            # Names come from model metadata; --classes cross-checks and load_detector caches weights in RAM/VRAM.
            self.detector = load_detector(cfg.detector, cfg.classes, conf=cfg.det_conf, iou=cfg.det_iou)
            if warmup_once(self.detector):
                log.info("detector warmed up on %s, input %s, %s head, %d classes", self.detector.provider,
                         self.detector.imgsz, self.detector.mode, len(self.detector.class_names))
        self.recorder = Recorder(cfg.data)
        self.policy, self.strategy_id, self.params = self._build_policy(cfg.strategy)
        self.perception = CRPerception(layout=self.layout, tiles=self.tiles, cards=self.cards, arena=self.arena,
                                       matcher=CardMatcher.load(cfg.templates), digits=load_digits(cfg.digits),
                                       detector=self.detector, on_my_play=self._on_my_play,
                                       detect_every=cfg.detect_every, async_detect=cfg.async_detect,
                                       tower_states=TowerStateReader.load(cfg.towers, rois=self.layout.towers) if cfg.towers else None,
                                       harvest=TemplateHarvester(cfg.harvest_templates)
                                       if cfg.harvest_templates else None,
                                       my_deck=self.params.deck)
        self.completed = 0
        self.reconnects = 0
        self._upgrade_requested = False
        self._match_summaries: list[dict] = []
        self._on_summary_hook: Callable[[dict], None] | None = None
        self.findings: queue.Queue[Finding] = queue.Queue()
        self.game_version = None
        self.session = CRSession(
            screens=self.screens, menus=MenuNavigator(self.layout.buttons, self._flow(cfg)),
            perception=self.perception, policy=self.policy, mapper=CRActionMapper(self.layout, self.tiles),
            recorder=self.recorder, meta_factory=self._meta, on_match_end=self._on_end)
        if cfg.prune_win_video:
            # After the recorder has written summary.json and closed the video (issue #8).
            self.session.after_written = lambda d, summary: retain_match(d)

    @staticmethod
    def _flow(cfg: RunConfig) -> dict[str, list[str]]:
        """The mode's default flow, overridden per screen by a ``flows:`` block in the layout YAML."""
        flow = dict(DEFAULT_FLOWS[cfg.mode])
        raw = yaml.safe_load(Path(cfg.layout).read_text()) or {}
        flow.update((raw.get("flows") or {}).get(cfg.mode) or {})
        return flow

    def _on_my_play(self, card: str, t_ms: int) -> None:
        self.session.policy.confirm_played(card, t_ms)

    def _build_policy(self, strategy_path: Path) -> tuple[CRPolicy, str, Any]:
        sp = Path(strategy_path).resolve()
        cached = self._policy_cache.get(sp)
        if cached is not None:
            pol, sid, params = cached
            pol.reset_match()
            return pol, sid, params
        _, params, sid = load_strategy(sp)
        ranker = None
        if params.spec.learned_ranker:
            model_path = sp.parent / params.spec.learned_ranker
            ranker = load_ranker(model_path, deck=params.deck)
            log.info("ranker on %s (v%d): %d state + %d candidate features",
                     model_path, ranker.version, ranker.state_dim, ranker.candidate_dim)
        if self.detector is not None:
            deck_classes = {u for k in params.deck if k in self.cards for u in self.cards[k].unit_classes}
            missing = missing_unit_classes(self.detector.class_names, deck_classes)
            if missing:
                log.warning("detector cannot see these deck unit classes: %s", missing)
        elif not getattr(params, "blind", False):
            log.warning("no detector and strategy %s is not blind: the policy can't see units and will "
                        "rarely attack; use configs/strategies/training-camp-blind-v1.yaml", sp.name)
        pol = CRPolicy(self.cards, self.arena, params,
                       exec_latency_s=self.profile.exec_latency_ms / 1000, ranker=ranker,
                       levels=self.account.policy_levels() if self.account else None)
        self._policy_cache[sp] = (pol, sid, params)
        return pol, sid, params

    def set_strategy(self, strategy_path: str | Path, *, experiment_id: str | None = None,
                     arm: str | None = None, pair_id: str | None = None) -> str:
        """Hot-swap the active strategy between matches without restarting scrcpy or GPU weights."""
        pol, sid, params = self._build_policy(Path(strategy_path))
        self.cfg.strategy = Path(strategy_path)
        self.cfg.experiment_id = experiment_id
        self.cfg.arm = arm
        self.cfg.pair_id = pair_id
        if pair_id:
            # Practice gauntlet: both arms of a pair face the same battle-log opponent.
            digits = "".join(ch for ch in pair_id.rsplit("-p", 1)[-1] if ch.isdigit())
            self.session.menus.slot_index = int(digits) if digits else 0
            if self.cfg.mode == "practice":
                self.cfg.opponent_id = f"practice:slot{self.session.menus.slot_index}"
        self.policy, self.strategy_id, self.params = pol, sid, params
        self.session.policy = pol
        self.perception.set_my_deck(params.deck)
        return sid

    def _meta(self) -> MatchMeta:
        return MatchMeta(uuid.uuid4().hex[:12], self.profile.name, self.profile.account, self.strategy_id,
                         self.game_version, self.cfg.balance_epoch, self.cfg.mode, self.cfg.experiment_id,
                         self.cfg.arm, self.cfg.opponent_id, self.cfg.pair_id,
                         account_levels=self.account.to_meta() if self.account else None,
                         strategy_name=self.params.spec.strategy)

    def _on_end(self, summary: dict) -> None:
        if summary.get("status") == "completed":
            self.completed += 1
        log.info("match %s: %s (%d/%d)", summary["match_id"], summary["result"], self.completed, self.cfg.matches)
        if self.account is not None and self.cfg.experiment_id is None and not self.cfg.observe_only:
            if self.account.sync_king_level(summary.get("my_king_level_seen")):
                self.account.save()
                self._sync_policy_levels()
        if (self.cfg.auto_upgrade and self.cfg.experiment_id is None and not self.cfg.observe_only
                and self.cfg.auto_upgrade_every > 0 and self.completed > 0
                and self.completed % self.cfg.auto_upgrade_every == 0
                and self.completed < self.cfg.matches):
            self._upgrade_requested = True
            self.session.suspend_menu = True  # hold menu taps until the upgrade pass runs
        self._match_summaries.append(summary)
        if self._on_summary_hook is not None:
            self._on_summary_hook(summary)

    def _sync_policy_levels(self) -> None:
        if self.account is None:
            return
        levels = self.account.policy_levels()
        self.policy.levels = levels
        self.policy.ctx.levels = levels
        self.policy.tier0.levels = levels
        for pol, _, _ in self._policy_cache.values():
            pol.levels = levels
            pol.ctx.levels = levels
            pol.tier0.levels = levels

    def _run_auto_upgrade(self) -> None:
        """Upgrade ready Battle Deck cards & Tower Princess between farm matches."""
        self.session.suspend_menu = True
        try:
            rep = run_auto_upgrade(
                self.client, self.screens, self.layout,
                matcher=self.perception.matcher, account=self.account,
                max_upgrades=self.cfg.auto_upgrade_max,
                log=lambda fmt, *args: log.info("auto-upgrade: " + fmt, *args),
            )
            if rep.upgraded:
                log.info("auto-upgrade completed %d upgrades: %s", len(rep.upgraded), rep.upgraded)
                self._sync_policy_levels()
            else:
                log.info("auto-upgrade: no affordable upgrades ready (deck=%s, home=%s)",
                         rep.reached_deck, rep.returned_home)
        except Exception:
            log.exception("auto-upgrade pass failed; continuing farming")
        finally:
            self._upgrade_requested = False
            self.session.suspend_menu = False

    def _connect(self) -> None:
        self.client = ScrcpyClient(self.profile, on_packet=self.recorder.on_packet)
        self.client.start()
        observe = self.cfg.observe_only
        # Stuck-screen recovery sends BACK through scrcpy's control channel, not ADB.
        self.session.send_key = None if observe else self.client.keycode
        self.session.recovery.reset()
        sink = self._observe_sink if observe else self.client.touch
        self.executor = Executor(sink, on_sent=self.ui.on_sent if self.ui is not None else None).start()
        on_step = None
        if self.ui is not None:
            tap, perception = self.ui, self.perception

            def on_step(frame, obs, decision, durations):
                # References only; the Studio's publisher thread does the work.
                tap.publish((frame, obs, decision, durations, perception.last_detections))
        self.brain = BrainLoop(self.client.slot, self.session, self.session, self.session, self.executor,
                               logger=self.recorder, on_timing=lambda d: self.session.match_latency.add_many(d),
                               on_step=on_step)
        self.brain.start()
        self.liveness = LivenessMonitor(self.client, in_battle=lambda: self.session.screen in BATTLE_SCREENS,
                                        screen=self.session.screen_age_s, on_finding=self.findings.put).start()

    @staticmethod
    def _observe_sink(action: int, x: int, y: int, pointer_id: int = 0) -> int:
        """Observe-only touch sink: the executor runs as usual, nothing reaches the phone."""
        return time.perf_counter_ns()

    def ui_context(self) -> dict:
        """Read by the Studio publisher thread (never by the brain)."""
        m = self.session.match
        tally: dict[str, int] = {}
        for s in list(self._match_summaries):
            r = str(s.get("result") or "")
            key = "aborted" if r.startswith("aborted") else r
            tally[key] = tally.get(key, 0) + 1
        return {
            "perception": self.perception, "session": self.session, "params": self.params,
            "screen": self.session.screen, "observe_only": self.cfg.observe_only,
            "match": None if m is None else {
                "id": m.match_id, "arm": m.arm, "experiment_id": m.experiment_id, "pair_id": m.pair_id,
                "opponent_kind": m.opponent_kind, "device": m.device},
            "strategy": {"id": self.strategy_id, "name": self.params.spec.strategy,
                         "act_threshold": self.params.act_threshold, "deck": list(self.params.deck)},
            "runner": {"completed": self.completed, "target": self.cfg.matches, "mode": self.cfg.mode,
                       "device": self.profile.name, "data": str(self.cfg.data), "results": tally},
            "experiment": self.ui_experiment() if self.ui_experiment is not None else None,
        }

    def _disconnect(self) -> None:
        for part in ("liveness", "brain", "executor"):
            obj = getattr(self, part, None)
            if obj is not None:
                obj.stop()
                setattr(self, part, None)  # a failed reconnect must not stop them twice
        self.session.send_key = None
        client = getattr(self, "client", None)
        if client is not None:
            client.stop()

    def _settle_home(self, timeout_s: float = 25.0) -> bool:
        """Walk a finished batch back to HOME so the next run can queue immediately."""
        if self.session.screen == "HOME":
            return True
        log.info("settle-home: leaving %s for HOME", self.session.screen)
        # The brain keeps running; suspend its menu decisions so the HOME flow cannot
        # queue a fresh battle while we are trying to stop.
        self.session.suspend_menu = True
        try:
            ok = walk_to_home(
                self.client, self.screens, self.layout,
                timeout_s=timeout_s, stable_s=1.5, stuck_after=1.5, battle_stuck_after=5.0,
                restart_after=15.0, log=lambda fmt, *args: log.info("settle-home: " + fmt, *args))
        finally:
            self.session.suspend_menu = False
        return ok

    def run(self, *, stop_when: Callable[[], bool] | None = None,
            on_summary: Callable[[dict], None] | None = None) -> None:
        self._on_summary_hook = on_summary
        if self.game_version is None:
            self.game_version = self.client_version()
        run_start_upgrade = bool(self.cfg.auto_upgrade and self.cfg.experiment_id is None and not self.cfg.observe_only)
        if run_start_upgrade:
            self.session.suspend_menu = True
        # Held for the whole run, so another process can't take the phone between reconnects
        # (issue #18). Raises DeviceBusy, naming the holder, before anything touches the phone.
        run_lock = DeviceLock(self.profile.serial).acquire() if self.profile.serial != "sim" else None
        try:
            self._connect()
        except BaseException:
            if run_lock is not None:
                run_lock.release()
            raise
        stuck_since = None
        last_screen = self.session.screen
        watchdog = AppWatchdog(self.client.adb, PACKAGE)
        try:
            if run_start_upgrade:
                self._run_auto_upgrade()
            while self.completed < self.cfg.matches and not (stop_when and stop_when()):
                if self.recorder.fatal_error:  # disk full, read-only, no permission (issue #7)
                    raise RunAborted(f"recording failed: {self.recorder.fatal_error}. Free disk space or fix "
                                     f"the permissions on {self.cfg.data}, then run again.")
                if self._upgrade_requested and self.session.match is None:
                    self._run_auto_upgrade()
                    self.session.screen_since = time.monotonic()
                    stuck_since = None
                # Checked every pass, not only when liveness is quiet: a dead game shows a
                # stuck launcher, whose findings would otherwise starve this check.
                if not self.cfg.observe_only and watchdog.due() and watchdog.died():
                    log.warning("watchdog: %s is not running; relaunching", PACKAGE)
                    self.session.abort_match("app_died")
                    self.client.adb.launch(PACKAGE)
                    self.session.screen_since = time.monotonic()
                    stuck_since = None
                # Recovery already walked every tap and BACK; a fresh app is the next rung.
                restart = self.session.take_restart_request()
                if restart and self.cfg.observe_only:
                    self.session.recovery.reset()  # watching: never restart the app under the player
                elif restart:
                    log.warning("stuck-screen recovery exhausted taps; restarting %s", PACKAGE)
                    self.session.abort_match("stuck_screen")
                    self.client.adb.force_stop(PACKAGE)
                    self.client.adb.launch(PACKAGE)
                    self.session.recovery.reset()
                    self.session.screen_since = time.monotonic()
                    stuck_since = None
                if self.session.screen != last_screen:
                    last_screen, stuck_since = self.session.screen, None
                try:
                    f = self.findings.get(timeout=0.5)
                except queue.Empty:
                    continue
                log.warning("liveness: %s (%s)", f.kind, f.detail)
                if f.action in ("reconnect", "restart_server", "adb_reconnect"):
                    self.session.abort_match(f.kind)
                    self._reconnect(f.kind, adb_reconnect=f.action == "adb_reconnect")
                elif f.action == "recover_ui" and self.cfg.observe_only:
                    self.session.screen_since = time.monotonic()  # the player is in charge of menus
                elif f.action == "recover_ui":
                    # Arm the deterministic HOME walk: calibrated taps, then buttons
                    # found visually (a mis-calibrated `ok` is the common stall), then
                    # BACK. The session's menu decisions carry it out; the 60 s cap below
                    # is the last resort.
                    self.session.recovery.force()
                    stuck_since = stuck_since or time.monotonic()
                    if time.monotonic() - stuck_since > 60:
                        log.warning("still stuck after 60s; restarting %s", PACKAGE)
                        self.session.abort_match("stuck_screen")
                        self.client.adb.force_stop(PACKAGE)
                        self.client.adb.launch(PACKAGE)
                        self.session.recovery.reset()
                        stuck_since = None
                    self.session.screen_since = time.monotonic()
                # battle frame gaps are logged by the recorder via the brain's frame log
            # Batch finished (or a graceful stop_when): leave the phone on HOME, not on
            # a result screen, so the next run can queue without a manual tap.
            if self.client.running and self.client.error is None and not self.cfg.observe_only:
                self._settle_home()
        finally:
            self._disconnect()
            self.recorder.close()
            if run_lock is not None:
                run_lock.release()

    RECONNECT_DELAYS_S = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

    def _reconnect(self, reason: str, *, adb_reconnect: bool = False) -> None:
        """Reconnect after a liveness finding, retrying with backoff (issue #17).

        Gives up with RunAborted once ``reconnect_deadline_s`` has passed, so a phone that
        dropped off Wi-Fi for a minute doesn't end an overnight batch, but a phone that's
        gone for good still ends it with a clear reason."""
        deadline = time.monotonic() + self.cfg.reconnect_deadline_s
        attempt, last = 0, None
        while True:
            self._disconnect()
            if adb_reconnect or attempt > 0:
                try:
                    self.client.adb.run("reconnect", check=False)
                except Exception:  # noqa: BLE001 - adb itself may be what's broken
                    pass
            time.sleep(self.RECONNECT_DELAYS_S[min(attempt, len(self.RECONNECT_DELAYS_S) - 1)])
            try:
                self._connect()
            except DeviceBusy:
                raise  # another process took the phone: don't fight over it
            except Exception as e:  # noqa: BLE001
                attempt, last = attempt + 1, e
                log.warning("reconnect attempt %d after %s failed: %s", attempt, reason, e)
                if time.monotonic() >= deadline:
                    raise RunAborted(
                        f"the phone stayed unreachable for {self.cfg.reconnect_deadline_s:.0f} s after {reason} "
                        f"({attempt} attempts; last error: {last}). Check the cable or adb over Wi-Fi.") from e
                continue
            self.reconnects += 1
            if attempt:
                log.info("reconnected after %s on attempt %d", reason, attempt + 1)
            return

    def client_version(self, retries: int = 1, delay_s: float = 2.0) -> str | None:
        """The game's version via adb, or None if it couldn't be read (``retries`` attempts)."""
        from .core.transport.adb import Adb

        for i in range(max(1, retries)):
            try:
                v = Adb(self.profile.serial).package_version(PACKAGE)
                if v:
                    return v
            except Exception:  # noqa: BLE001 - phone booting, locked, adb hiccup
                pass
            if i + 1 < retries:
                time.sleep(delay_s)
        return None


def run_sim(seconds: float = 10.0, out: Path = Path("data/sim"), *, ui: dict | None = None) -> dict:
    """The closed loop with no phone: synthetic scene -> H.264 -> bot -> touches -> scene.

    ``ui={"host":..., "port":..., "fps":...}`` serves the live Studio on the synthetic match (a
    no-phone way to try ``--ui``); enemy waves then keep coming until ``seconds`` elapse.
    """
    from .core.runtime.orchestrator import BrainLoop
    from .sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene
    from .sim.fake_scrcpy import FakeScrcpyServer, free_port

    root = Path(__file__).resolve().parents[2]
    cards = CardDB.load(root / "tests/fixtures/cards_min.json")
    _, params, sid = load_strategy(root / "configs/strategies/hog-cycle-v1.yaml")
    deck = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]
    scene = CRScene(cards, deck, elixir=6.0, draw_tower_hp=False)
    lock = threading.Lock()

    def source(i, t):
        with lock:
            scene.step(max(0.0, t - scene.t_s))
            return scene.render()

    port = free_port()
    srv = FakeScrcpyServer(port, width=scene.width, height=scene.height, fps=30, source=source).start()
    rec = Recorder(out)
    client = ScrcpyClient(DeviceProfile(name="sim", serial="sim", local_port=port, scrcpy_server_version="sim"),
                          on_packet=rec.on_packet)
    arena = Arena()
    policy = CRPolicy(cards, arena, params, exec_latency_s=0.03)
    perception = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                              matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene),
                              on_my_play=policy.confirm_played)
    client.start(launch=False)
    tap = server = None
    if ui is not None:
        from .ui.launch import serve_live
        from .ui.telemetry import TelemetryTap

        tap = TelemetryTap()
        context = {"perception": perception, "params": params, "screen": "IN_BATTLE",
                   "strategy": {"id": sid, "name": params.spec.strategy, "act_threshold": params.act_threshold,
                                "deck": list(params.deck)},
                   "runner": {"completed": 0, "target": 1, "mode": "sim", "device": "sim", "data": str(out)}}
        server = serve_live(tap, lambda: context, cards=cards, layout=scene.layout, tiles=scene.tiles,
                            arena=arena, replay_roots=[out], title="droid-harness Studio: simulator",
                            host=ui.get("host", "127.0.0.1"), port=ui.get("port", 8765), fps=ui.get("fps", 15.0))
        print(f"Studio: {server.url}", flush=True)
    executor = Executor(client.touch, on_sent=tap.on_sent if tap else None).start()
    brain = BrainLoop(client.slot, perception, policy, CRActionMapper(scene.layout, scene.tiles), executor,
                      logger=rec,
                      on_step=(lambda f, o, d, lat: tap.publish((f, o, d, lat, perception.last_detections)))
                      if tap else None).start()
    match_id = f"sim-{uuid.uuid4().hex[:6]}"
    rec.start_match(match_id, {"strategy_id": sid, "mode": "sim"})
    stop = threading.Event()

    def pump():
        seen = 0
        while not stop.is_set():
            while seen < len(srv.state.touches):
                with lock:
                    scene.on_touch(srv.state.touches[seen])
                seen += 1
            time.sleep(0.002)

    threading.Thread(target=pump, daemon=True).start()
    t0 = time.time()
    waves = [(1.0, "hog_rider", (3.5, 24.0), (0.0, -2.0)), (6.0, "knight", (14.5, 24.0), (0.0, -1.0))]
    if ui is not None:  # a longer show: alternate lanes every 7 s
        kinds = ["hog_rider", "knight", "minion", "musketeer", "baby_dragon"]
        waves += [(11.0 + 7 * i, kinds[i % len(kinds)], (3.5 if i % 2 else 14.5, 24.0),
                   (0.0, -2.0 if kinds[i % len(kinds)] == "hog_rider" else -1.0)) for i in range(int(seconds // 7))]
    try:
        while time.time() - t0 < seconds:
            el = time.time() - t0
            while waves and waves[0][0] <= el:
                _, cls, tile, vel = waves.pop(0)
                with lock:
                    scene.spawn_enemy(cls, tile, vel)
            time.sleep(0.05)
    finally:
        stop.set()
        brain.stop()
        executor.stop()
        rec.end_match({"match_id": match_id, "result": "sim", "status": "completed", "strategy_id": sid,
                       "events": perception.events, "start_pts_us": perception.start_pts_us})
        rec.close()
        client.stop()
        srv.stop()
        if server is not None:
            server.stop()
    return {"match_dir": str(out / match_id), "plays": scene.plays, "decisions": brain.decisions,
            "latency_ms": brain.stats.summary("total"), "errors": brain.errors}
