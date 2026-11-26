import time

import numpy as np

from droid_harness.core.config import DeviceProfile
from droid_harness.core.transport import scrcpy_protocol as proto
from droid_harness.core.transport.scrcpy_bridge import ScrcpyClient
from droid_harness.sim.fake_scrcpy import FakeScrcpyServer, free_port, split_config_nals


def _profile(port, **kw):
    return DeviceProfile(name="fake", serial="fake", local_port=port, scrcpy_server_version="5.0", **kw)


def _wait(pred, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return True
        time.sleep(0.01)
    return False


def test_split_config_nals():
    sps = b"\x00\x00\x00\x01\x67abc"
    pps = b"\x00\x00\x00\x01\x68de"
    idr = b"\x00\x00\x01\x65frame"
    config, rest = split_config_nals(sps + pps + idr)
    assert config == sps + pps and rest == idr


def test_end_to_end_stream_and_touch():
    port = free_port()
    packets = []
    srv = FakeScrcpyServer(port, width=160, height=320, fps=60).start()
    client = ScrcpyClient(_profile(port), on_packet=lambda h, d: packets.append(h))
    try:
        client.start(launch=False)
        assert client.device_name == "FakePhone" and client.codec == "h264"
        assert client.size == (160, 320)
        f = client.slot.wait_newer(0, timeout=3)
        assert f is not None and f.img.shape == (320, 160, 3)
        assert _wait(lambda: client.slot.peek().seq >= 10)
        # config packet seen by the recorder tee before any frame
        assert packets[0].is_config and not packets[1].is_config
        # moving box is decoded (frames differ)
        a = client.slot.wait_newer(0).img.copy()
        b = client.slot.wait_newer(client.slot.peek().seq, timeout=1).img
        assert np.abs(a.astype(int) - b.astype(int)).sum() > 0
        client.touch(proto.MotionAction.DOWN, 80, 160, pointer_id=3)
        client.touch(proto.MotionAction.UP, 80, 160, pointer_id=3)
        client.set_display_power(False)
        assert _wait(lambda: len(srv.state.touches) == 2 and srv.state.display_power == [0])
        t = srv.state.touches[0]
        assert (t["x"], t["y"], t["w"], t["h"], t["pointer_id"]) == (80, 160, 160, 320, 3)
    finally:
        client.stop()
        srv.stop()


def test_static_screen_is_not_an_error():
    port = free_port()
    srv = FakeScrcpyServer(port, width=96, height=192, static_after_frames=5).start()
    client = ScrcpyClient(_profile(port))
    try:
        client.start(launch=False)
        assert _wait(lambda: client.slot.peek() is not None and client.slot.peek().seq >= 5)
        time.sleep(0.3)
        assert client.error is None and client.running
        assert client.ms_since_last_packet() >= 250
    finally:
        client.stop()
        srv.stop()
