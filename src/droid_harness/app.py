"""Wiring: build a running Clash Royale bot (or the synthetic demo) from config files."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from .core.actuate.executor import Executor
from .core.config import DeviceProfile
from .core.runtime.liveness import Finding, LivenessMonitor
from .core.runtime.orchestrator import BrainLoop
from .core.telemetry.recorder import Recorder
from .core.transport.scrcpy_bridge import ScrcpyClient
from .domains.clash_royale.actions import CRActionMapper
from .domains.clash_royale.arena import Arena
from .domains.clash_royale.cards import CardDB
from .domains.clash_royale.menus import BATTLE_SCREENS, DEFAULT_FLOWS, MenuNavigator
from .domains.clash_royale.perception.cards import CardMatcher
from .domains.clash_royale.perception.detector import OnnxDetector
from .domains.clash_royale.perception.digits import DigitReader
from .domains.clash_royale.perception.grid import Layout, Roi, TileMapper
from .domains.clash_royale.perception.pipeline import CRPerception
from .domains.clash_royale.perception.screens import ScreenClassifier
from .domains.clash_royale.policy.strategy import load_strategy
from .domains.clash_royale.policy.utility import CRPolicy
from .domains.clash_royale.session import CRSession, MatchMeta

log = logging.getLogger(__name__)
PACKAGE = "com.supercell.clashroyale"
KEYCODE_BACK = 4


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


def load_screens(directory: Path, regions: list[Roi]) -> ScreenClassifier:
    sc = ScreenClassifier(regions)
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
        _, self.params, self.strategy_id = load_strategy(cfg.strategy)
        regions = [Roi(*r) for r in yaml.safe_load(cfg.layout.read_text()).get("screen_regions",
                                                                                 [[0, 0, 1, 0.5], [0, 0.5, 1, 1]])]
        self.screens = load_screens(cfg.screens, regions)
        detector = None
        if cfg.detector:
            names = Path(cfg.classes).read_text().split() if cfg.classes else []
            detector = OnnxDetector(cfg.detector, names)
            detector.warmup()
            log.info("detector on %s", detector.provider)
        self.recorder = Recorder(cfg.data)
        self.policy = CRPolicy(self.cards, self.arena, self.params,
                               exec_latency_s=self.profile.exec_latency_ms / 1000)
        self.perception = CRPerception(layout=self.layout, tiles=self.tiles, cards=self.cards, arena=self.arena,
                                       matcher=CardMatcher.load(cfg.templates), digits=load_digits(cfg.digits),
                                       detector=detector, on_my_play=self.policy.confirm_played)
        self.completed = 0
        self.findings: queue.Queue[Finding] = queue.Queue()
        self.game_version = None
        self.session = CRSession(
            screens=self.screens, menus=MenuNavigator(self.layout.buttons, dict(DEFAULT_FLOWS[cfg.mode])),
            perception=self.perception, policy=self.policy, mapper=CRActionMapper(self.layout, self.tiles),
            recorder=self.recorder, meta_factory=self._meta, on_match_end=self._on_end)

    def _meta(self) -> MatchMeta:
        return MatchMeta(uuid.uuid4().hex[:12], self.profile.name, self.profile.account, self.strategy_id,
                         self.game_version, self.cfg.balance_epoch, self.cfg.mode, self.cfg.experiment_id,
                         self.cfg.arm, self.cfg.opponent_id)

    def _on_end(self, summary: dict) -> None:
        if summary.get("status") == "completed":
            self.completed += 1
        log.info("match %s: %s (%d/%d)", summary["match_id"], summary["result"], self.completed, self.cfg.matches)

    def _connect(self) -> None:
        self.client = ScrcpyClient(self.profile, on_packet=self.recorder.on_packet)
        self.client.start()
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
        self.client.stop()

    def run(self) -> None:
        self.game_version = self.client_version()
        self._connect()
        stuck_since = None
        try:
            while self.completed < self.cfg.matches:
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
                    stuck_since = stuck_since or time.monotonic()
                    if time.monotonic() - stuck_since > 60:
                        self.client.adb.force_stop(PACKAGE)
                        self.client.adb.launch(PACKAGE)
                        stuck_since = None
                    else:
                        self.client.keycode(KEYCODE_BACK)
                    self.session.screen_since = time.monotonic()
                # battle frame gaps are logged by the recorder via the brain's frame log
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
