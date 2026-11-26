"""Whole loop without a phone: synthetic scene -> real H.264 over the scrcpy protocol ->
perception -> policy -> executor -> touches -> the scene reacts."""

import threading
import time

import pytest

from droid_harness.core.actuate.executor import Executor
from droid_harness.core.config import DeviceProfile
from droid_harness.core.runtime.orchestrator import BrainLoop
from droid_harness.core.telemetry.recorder import Recorder
from droid_harness.core.transport.scrcpy_bridge import ScrcpyClient
from droid_harness.domains.clash_royale.actions import CRActionMapper
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene
from droid_harness.sim.fake_scrcpy import FakeScrcpyServer, free_port

DECK = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]


def test_bot_defends_hog_with_cannon_end_to_end(cards, arena, strategy, tmp_path):
    _, params, sid = strategy
    scene = CRScene(cards, DECK, elixir=6.0, draw_tower_hp=False)
    lock = threading.Lock()

    def source(i, t):
        with lock:
            scene.step(max(0.0, t - scene.t_s))
            return scene.render()

    port = free_port()
    srv = FakeScrcpyServer(port, width=scene.width, height=scene.height, fps=30, source=source).start()
    _orig = srv._control_loop

    rec = Recorder(tmp_path)
    client = ScrcpyClient(DeviceProfile(name="sim", serial="sim", local_port=port, scrcpy_server_version="5.0"),
                          on_packet=rec.on_packet)
    policy = CRPolicy(cards, arena, params, exec_latency_s=0.03)
    perception = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                              matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene),
                              on_my_play=policy.confirm_played)
    executor = Executor(client.touch).start()
    brain = BrainLoop(client.slot, perception, policy, CRActionMapper(scene.layout, scene.tiles), executor,
                      logger=rec)

    def pump_touches():  # deliver touches the fake server received to the scene
        seen = 0
        while not stop.is_set():
            while seen < len(srv.state.touches):
                with lock:
                    scene.on_touch(srv.state.touches[seen])
                seen += 1
            time.sleep(0.002)

    stop = threading.Event()
    try:
        client.start(launch=False)
        assert client.slot.wait_newer(0, timeout=3) is not None
        rec.start_match("sim-1", {"strategy_id": sid})
        threading.Thread(target=pump_touches, daemon=True).start()
        brain.start()
        time.sleep(0.3)
        with lock:
            scene.spawn_enemy("hog_rider", (3.5, 24.0), vel=(0.0, -2.0))
        t0 = time.time()
        while time.time() - t0 < 6 and not scene.plays:
            time.sleep(0.02)
    finally:
        stop.set()
        brain.stop()
        executor.stop()
        rec.end_match({"result": "n/a"})
        rec.close()
        client.stop()
        srv.stop()

    assert brain.errors == 0
    assert scene.plays, f"no play; decisions={brain.decisions} warnings={perception.warnings}"
    _, card, tile = scene.plays[0]
    assert card == "cannon"
    assert tile == pytest.approx(tuple(params.placement.building_pull["left"]), abs=0.6)
    # the decision was logged with its candidates, posture, and per-stage latency
    lines = (tmp_path / "sim-1" / "decisions.jsonl").read_text().splitlines()
    assert any('"defend.lane_threat"' in line for line in lines)
    lat = brain.stats.summary("total")
    assert lat["n"] > 10 and lat["p50"] < 100


def test_blind_strategy_attacks_end_to_end_without_a_detector(arena, tmp_path):
    """No detector at all (today's phone setup): the blind strategy still pushes Giant at the bridge."""
    from pathlib import Path

    from droid_harness.domains.clash_royale.cards import CardDB
    from droid_harness.domains.clash_royale.policy.strategy import load_strategy

    root = Path(__file__).resolve().parents[1]
    starter = CardDB.load(root / "tests/fixtures/cards_starter.json")
    _, params, _ = load_strategy(root / "configs/strategies/training-camp-blind-v1.yaml")
    deck = ["giant", "knight", "musketeer", "fireball", "princess", "minions", "arrows", "mini_pekka"]
    scene = CRScene(starter, deck, elixir=8.0, draw_tower_hp=False)
    lock = threading.Lock()

    def source(i, t):
        with lock:
            scene.step(max(0.0, t - scene.t_s))
            return scene.render()

    port = free_port()
    srv = FakeScrcpyServer(port, width=scene.width, height=scene.height, fps=30, source=source).start()
    client = ScrcpyClient(DeviceProfile(name="sim", serial="sim", local_port=port, scrcpy_server_version="5.0"))
    policy = CRPolicy(starter, arena, params, exec_latency_s=0.03)
    perception = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=starter, arena=arena,
                              matcher=build_matcher_from_scene(scene), detector=None,
                              on_my_play=policy.confirm_played)
    executor = Executor(client.touch).start()
    brain = BrainLoop(client.slot, perception, policy, CRActionMapper(scene.layout, scene.tiles), executor)
    stop = threading.Event()

    def pump_touches():
        seen = 0
        while not stop.is_set():
            while seen < len(srv.state.touches):
                with lock:
                    scene.on_touch(srv.state.touches[seen])
                seen += 1
            time.sleep(0.002)

    try:
        client.start(launch=False)
        assert client.slot.wait_newer(0, timeout=3) is not None
        threading.Thread(target=pump_touches, daemon=True).start()
        brain.start()
        t0 = time.time()
        while time.time() - t0 < 6 and not scene.plays:
            time.sleep(0.02)
    finally:
        stop.set()
        brain.stop()
        executor.stop()
        client.stop()
        srv.stop()

    assert brain.errors == 0
    assert scene.plays, f"no play; decisions={brain.decisions} warnings={perception.warnings}"
    _, card, tile = scene.plays[0]
    assert card == "giant"
    assert tile == pytest.approx(tuple(params.placement.bridge_tiles["right"]), abs=0.6)
