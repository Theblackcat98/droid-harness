import time

import numpy as np

from droid_harness.core.config import DeviceProfile
from droid_harness.core.transport.scrcpy_bridge import ScrcpyClient
from droid_harness.domains.android_os.a11y import parse_portal_output, parse_uiautomator_xml
from droid_harness.domains.android_os.settle import wait_settled
from droid_harness.domains.android_os.skills_cache import SkillsCache, Step, replay
from droid_harness.domains.android_os.som import Element, legend, overlay
from droid_harness.sim.fake_scrcpy import FakeScrcpyServer, free_port


def _client(port):
    return ScrcpyClient(DeviceProfile(name="f", serial="f", local_port=port, scrcpy_server_version="5.0"))


def test_settle_uses_frame_quiescence():
    port = free_port()
    srv = FakeScrcpyServer(port, width=96, height=160, static_after_frames=10).start()
    c = _client(port)
    try:
        c.start(launch=False)
        t0 = time.monotonic()
        assert wait_settled(c, quiet_ms=200, timeout_s=3)
        assert time.monotonic() - t0 < 1.5
    finally:
        c.stop()
        srv.stop()


def test_settle_times_out_on_constant_motion():
    port = free_port()
    srv = FakeScrcpyServer(port, width=96, height=160).start()  # moving box forever
    c = _client(port)
    try:
        c.start(launch=False)
        assert not wait_settled(c, quiet_ms=200, timeout_s=0.6)
    finally:
        c.stop()
        srv.stop()


def test_settle_requires_stable_tree():
    port = free_port()
    srv = FakeScrcpyServer(port, width=96, height=160, static_after_frames=3).start()
    c = _client(port)
    hashes = iter(["a", "b", "c", "c", "c", "c", "c"])
    try:
        c.start(launch=False)
        assert wait_settled(c, tree_hash=lambda: next(hashes), quiet_ms=100, timeout_s=3)
    finally:
        c.stop()
        srv.stop()


def test_skills_cache_promotes_and_demotes(tmp_path):
    cache = SkillsCache.load(tmp_path / "skills.json", promote_after=2, demote_after_failures=1)
    steps = [Step("tap", {"x": 10, "y": 20}, expect_screen="home"), Step("type", {"text": "hi"})]
    assert cache.lookup("com.app", "home", "send hi") is None
    cache.record("com.app", "home", "send hi", steps, True, final_screen="sent")
    assert cache.lookup("com.app", "home", "send hi") is None
    cache.record("com.app", "home", "Send  HI", steps, True, final_screen="sent")
    skill = SkillsCache.load(tmp_path / "skills.json").lookup("com.app", "home", "send hi")
    assert skill is not None and skill.compiled
    screens = iter(["home", "sent"])
    done = []
    ok, why = replay(skill, done.append, lambda: next(screens), lambda: True)
    assert ok and len(done) == 2
    ok, why = replay(skill, done.append, lambda: "elsewhere", lambda: True)
    assert not ok and "pre-condition" in why
    cache.record("com.app", "home", "send hi", steps, False)
    assert cache.lookup("com.app", "home", "send hi") is None


def test_som_overlay_and_legend():
    img = np.zeros((200, 100, 3), np.uint8)
    els = [Element((10, 10, 90, 40), "Send"), Element((10, 50, 12, 52), "tiny"),
           Element((10, 100, 90, 140), "label", clickable=False)]
    out, marks = overlay(img, els)
    assert list(marks) == [1] and marks[1].center == (50, 25)
    assert out[12, 12].tolist() != [0, 0, 0] and legend(marks) == "[1] Send"


def test_tree_parsers():
    portal = 'Row: 0 result={"elements":[{"bounds":"[0,0][100,50]","text":"OK","resourceId":"ok_btn",' \
             '"clickable":true,"children":[{"bounds":[5,5,20,20],"text":"icon"}]}]}'
    els = parse_portal_output(portal)
    assert [e.text for e in els] == ["OK", "icon"] and els[0].bounds == (0, 0, 100, 50)
    xml = ('UI hierchary dumped<?xml version="1.0"?><hierarchy><node bounds="[0,0][10,10]" text="A" '
           'resource-id="x" clickable="true"><node bounds="[1,1][2,2]" text="B" clickable="false"/></node>'
           '</hierarchy>')
    els = parse_uiautomator_xml(xml)
    assert [(e.text, e.clickable) for e in els] == [("A", True), ("B", False)]
