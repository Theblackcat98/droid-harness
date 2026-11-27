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
from .core.transport.scrcpy_bridge import ScrcpyClient
from .domains.clash_royale.actions import CRActionMapper
from .domains.clash_royale.arena import Arena
from .domains.clash_royale.cards import CardDB
from .domains.clash_royale.menus import BATTLE_SCREENS, DEFAULT_FLOWS, MenuNavigator
from .domains.clash_royale.perception.cards import CardMatcher
from .domains.clash_royale.perception.classes import missing_unit_classes
from .domains.clash_royale.perception.detector import load_detector, warmup_once
from .domains.clash_royale.perception.digits import DigitReader
from .domains.clash_royale.perception.grid import Layout, Roi, TileMapper
from .domains.clash_royale.perception.pipeline import CRPerception
from .domains.clash_royale.perception.screens import ScreenClassifier
from .domains.clash_royale.perception.towers import TowerStateReader
from .domains.clash_royale.policy.strategy import load_strategy
from .domains.clash_royale.policy.utility import CRPolicy
from .domains.clash_royale.recovery import walk_to_home
from .domains.clash_royale.session import CRSession, MatchMeta
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


def resolve_strategy_path(strategy: str | Path, root: Path = Path(".")) -> Path:
    """Resolve short aliases ('v6', 'ladder-vision-v6') or direct paths to a strategy YAML."""
    p = Path(strategy)
    if p.exists():
        return p
    s = str(strategy).strip()
    strat_dir = root / "configs/strategies"
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


def run_doctor(*, device: str = "phone-a", strategy: str | Path = "v6",
               root: Path = Path("."), live: bool = False) -> dict[str, Any]:
    """Pre-flight verification of calibration, templates, digits, detector, disk, and ADB."""
    bundle = resolve_device_bundle(device, root=root, with_detector=True)
    strat_path = resolve_strategy_path(strategy, root=root)
    checks: dict[str, Any] = {}
    problems: list[str] = []

    for key in ("profile", "layout", "tiles", "templates", "screens", "cards"):
        p = bundle[key]
        ok = p is not None and p.exists()
        checks[key] = str(p) if ok else f"MISSING ({p})"
        if not ok:
            problems.append(f"missing required {key}: {p}")

    for opt_key in ("arena", "towers", "digits"):
        p = bundle[opt_key]
        checks[opt_key] = str(p) if (p and p.exists()) else "not configured"

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

    return {"ok": len(problems) == 0, "device": device, "checks": checks, "problems": problems}


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


class BotRunner:
    """One phone, N matches, with liveness-driven recovery."""

    def __init__(self, cfg: RunConfig) -> None:
        self.cfg = cfg
        self.profile = DeviceProfile(**{**yaml.safe_load(cfg.profile.read_text()), "name": cfg.profile.stem})
        self.cards = CardDB.load(cfg.cards)
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
                                       tower_states=TowerStateReader.load(cfg.towers) if cfg.towers else None)
        self.completed = 0
        self._match_summaries: list[dict] = []
        self._on_summary_hook: Callable[[dict], None] | None = None
        self.findings: queue.Queue[Finding] = queue.Queue()
        self.game_version = None
        self.session = CRSession(
            screens=self.screens, menus=MenuNavigator(self.layout.buttons, dict(DEFAULT_FLOWS[cfg.mode])),
            perception=self.perception, policy=self.policy, mapper=CRActionMapper(self.layout, self.tiles),
            recorder=self.recorder, meta_factory=self._meta, on_match_end=self._on_end)

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
                       exec_latency_s=self.profile.exec_latency_ms / 1000, ranker=ranker)
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
        self.policy, self.strategy_id, self.params = pol, sid, params
        self.session.policy = pol
        return sid

    def _meta(self) -> MatchMeta:
        return MatchMeta(uuid.uuid4().hex[:12], self.profile.name, self.profile.account, self.strategy_id,
                         self.game_version, self.cfg.balance_epoch, self.cfg.mode, self.cfg.experiment_id,
                         self.cfg.arm, self.cfg.opponent_id, self.cfg.pair_id)

    def _on_end(self, summary: dict) -> None:
        if summary.get("status") == "completed":
            self.completed += 1
        log.info("match %s: %s (%d/%d)", summary["match_id"], summary["result"], self.completed, self.cfg.matches)
        if self.cfg.prune_win_video and summary.get("match_id"):
            retain_match(self.cfg.data / str(summary["match_id"]))
        self._match_summaries.append(summary)
        if self._on_summary_hook is not None:
            self._on_summary_hook(summary)

    def _connect(self) -> None:
        self.client = ScrcpyClient(self.profile, on_packet=self.recorder.on_packet)
        self.client.start()
        # Stuck-screen recovery sends BACK through scrcpy's control channel, not ADB.
        self.session.send_key = self.client.keycode
        self.session.recovery.reset()
        self.executor = Executor(self.client.touch).start()
        self.brain = BrainLoop(self.client.slot, self.session, self.session, self.session, self.executor,
                               logger=self.recorder, on_timing=lambda d: self.session.match_latency.add_many(d))
        self.brain.start()
        self.liveness = LivenessMonitor(self.client, in_battle=lambda: self.session.screen in BATTLE_SCREENS,
                                        screen=self.session.screen_age_s, on_finding=self.findings.put).start()

    def _disconnect(self) -> None:
        for part in ("liveness", "brain", "executor"):
            obj = getattr(self, part, None)
            if obj is not None:
                obj.stop()
        self.session.send_key = None
        self.client.stop()

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
        self._connect()
        stuck_since = None
        last_screen = self.session.screen
        watchdog = AppWatchdog(self.client.adb, PACKAGE)
        try:
            while self.completed < self.cfg.matches and not (stop_when and stop_when()):
                # Checked every pass, not only when liveness is quiet: a dead game shows a
                # stuck launcher, whose findings would otherwise starve this check.
                if watchdog.due() and watchdog.died():
                    log.warning("watchdog: %s is not running; relaunching", PACKAGE)
                    self.session.abort_match("app_died")
                    self.client.adb.launch(PACKAGE)
                    self.session.screen_since = time.monotonic()
                    stuck_since = None
                # Recovery already walked every tap and BACK; a fresh app is the next rung.
                if self.session.take_restart_request():
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
                    self._disconnect()
                    if f.action == "adb_reconnect":
                        self.client.adb.run("reconnect", check=False)
                    time.sleep(1.0)
                    self._connect()
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
            if self.client.running and self.client.error is None:
                self._settle_home()
        finally:
            self._disconnect()
            self.recorder.close()

    def client_version(self) -> str | None:
        from .core.transport.adb import Adb

        try:
            return Adb(self.profile.serial).package_version(PACKAGE)
        except Exception:
            return None


def run_sim(seconds: float = 10.0, out: Path = Path("data/sim")) -> dict:
    """The closed loop with no phone: synthetic scene -> H.264 -> bot -> touches -> scene."""
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
    executor = Executor(client.touch).start()
    brain = BrainLoop(client.slot, perception, policy, CRActionMapper(scene.layout, scene.tiles), executor,
                      logger=rec).start()
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
    return {"match_dir": str(out / match_id), "plays": scene.plays, "decisions": brain.decisions,
            "latency_ms": brain.stats.summary("total"), "errors": brain.errors}
