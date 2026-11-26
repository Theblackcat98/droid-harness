import pytest

from droid_harness.core.transport import scrcpy_protocol as proto


def test_touch_layout_is_32_bytes_and_roundtrips():
    msg = proto.encode_touch(proto.MotionAction.DOWN, 100, 200, 460, 1024, pointer_id=7)
    assert len(msg) == 32
    d = proto.decode_touch(msg)
    assert d == {"type": 2, "action": 0, "pointer_id": 7, "x": 100, "y": 200, "w": 460, "h": 1024,
                 "pressure": 0xFFFF, "action_button": 0, "buttons": 0}


def test_touch_up_has_zero_pressure():
    assert proto.decode_touch(proto.encode_touch(proto.MotionAction.UP, 1, 1, 10, 10))["pressure"] == 0


def test_touch_rejects_negative_coordinates():
    with pytest.raises(ValueError):
        proto.encode_touch(proto.MotionAction.DOWN, -1, 0, 10, 10)


def test_frame_header_flags():
    raw = proto.encode_frame_header(123456, 99, config=True)
    h = proto.parse_frame_header(raw)
    assert (h.pts_us, h.is_config, h.is_key_frame, h.size) == (123456, True, False, 99)
    h2 = proto.parse_frame_header(proto.encode_frame_header(5, 1, key_frame=True))
    assert h2.is_key_frame and not h2.is_config


def test_codec_meta_and_device_name():
    assert proto.parse_codec_meta(proto.encode_codec_meta("h264", 460, 1024)) == ("h264", 460, 1024)
    assert proto.parse_device_name(b"Pixel 8".ljust(64, b"\0")) == "Pixel 8"


def test_display_power_payloads():
    assert proto.encode_display_power(False, on_byte=2) == bytes([10, 0])
    assert proto.encode_display_power(True, on_byte=2) == bytes([10, 2])
    assert proto.encode_display_power(True, on_byte=1) == bytes([10, 1])


def test_text_and_keycode():
    assert proto.encode_inject_text("hé") == bytes([1, 0, 0, 0, 3]) + "hé".encode()
    assert len(proto.encode_keycode(proto.KeyAction.DOWN, 4)) == 14


def test_server_args_are_per_session():
    args = proto.build_server_args("0badcafe", max_size=1024, video_encoder="c2.qti.avc.encoder")
    assert "scid=0badcafe" in args and "max_size=1024" in args and "tunnel_forward=true" in args
    assert "video_encoder=c2.qti.avc.encoder" in args and "audio=false" in args
