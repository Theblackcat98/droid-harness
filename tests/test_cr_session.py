from droid_harness.core.telemetry.recorder import Recorder
from droid_harness.core.transport.frame_slot import Frame
from droid_harness.domains.clash_royale.actions import CRActionMapper
from droid_harness.domains.clash_royale.menus import MenuNavigator
from droid_harness.domains.clash_royale.perception.grid import Roi
from droid_harness.domains.clash_royale.perception.pipeline import CRPerception
from droid_harness.domains.clash_royale.perception.screens import ScreenClassifier
from droid_harness.domains.clash_royale.policy.utility import CRPolicy
from droid_harness.domains.clash_royale.session import CRSession, MatchMeta, MenuDecision
from droid_harness.sim.cr_scene import CRScene, SceneDetector, build_matcher_from_scene

DECK = ["cannon", "musketeer", "ice_golem", "hog_rider", "fireball", "the_log", "ice_spirit", "skeletons"]


def test_session_lifecycle(cards, arena, strategy, tmp_path):
    _, params, sid = strategy
    scene = CRScene(cards, DECK, draw_tower_hp=False)
    battle = scene.render()
    victory = battle.copy()
    h, w = victory.shape[:2]
    # Real result layout: pink opponent banner on top, blue player banner below.
    victory[int(0.25 * h):int(0.35 * h), int(0.05 * w):int(0.95 * w)] = (147, 20, 220)
    victory[int(0.46 * h):int(0.56 * h), int(0.05 * w):int(0.95 * w)] = (220, 120, 20)
    # One gold crown above, three below => player win.
    victory[int(0.20 * h):int(0.30 * h), int(0.42 * w):int(0.58 * w)] = (0, 215, 255)
    for x in (0.25, 0.50, 0.75):
        victory[int(0.42 * h):int(0.52 * h), int((x - 0.08) * w):int((x + 0.08) * w)] = (0, 215, 255)
    screens = ScreenClassifier([Roi(0, 0, 1, 0.5), Roi(0, 0.5, 1, 1)])
    screens.register("IN_BATTLE", battle)
    screens.register("RESULT_WIN", victory)
    policy = CRPolicy(cards, arena, params)
    perception = CRPerception(layout=scene.layout, tiles=scene.tiles, cards=cards, arena=arena,
                              matcher=build_matcher_from_scene(scene), detector=SceneDetector(scene))
    ended = []
    rec = Recorder(tmp_path)
    sess = CRSession(screens=screens, menus=MenuNavigator(scene.layout.buttons), perception=perception,
                     policy=policy, mapper=CRActionMapper(scene.layout, scene.tiles), recorder=rec,
                     meta_factory=lambda: MatchMeta("", "phone-a", "acct-1", sid, "9.x", "2026-11",
                                                    "training_camp"),
                     on_match_end=ended.append, screen_every=1)
    for i in range(5):
        obs = sess.process(Frame(i + 1, scene.render(), i * 33_000, 0, 0))
        assert obs.screen == "IN_BATTLE" and obs.state is not None
    for i in range(3):  # result must persist for result_stable_frames observations
        obs = sess.process(Frame(9 + i, victory, 200_000 + i * 33_000, 0, 0))
    assert obs.screen == "RESULT_WIN" and obs.state is None
    # A battle-looking frame right after the result must not start a new match.
    obs = sess.process(Frame(20, battle, 300_000, 0, 0))
    assert obs.screen == "RESULT_WIN" and obs.state is None
    d = sess.decide(obs, None)
    assert isinstance(d, MenuDecision) and d.events and d.events[0].tag == "menu:down"
    rec.close()
    assert len(ended) == 1 and ended[0]["result"] == "win" and ended[0]["status"] == "completed"
    assert ended[0]["strategy_id"] == sid and ended[0]["game_version"] == "9.x"
    assert (tmp_path / ended[0]["match_id"] / "summary.json").exists()
