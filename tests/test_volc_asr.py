import gzip
import json
import struct

import pytest

from dududu.interfaces import ASRResult
from dududu.providers.volc_asr import AUDIO_ONLY, VolcASR, decode, encode


def server_frame(payload: dict, compress: bool) -> bytes:
    body = json.dumps(payload).encode()
    if compress:
        body = gzip.compress(body)
    header = bytes([0x11, 0x91, 0x11 if compress else 0x10, 0x00])  # 响应,带 sequence
    return header + struct.pack(">i", 1) + struct.pack(">I", len(body)) + body


def test_encode_audio_frame():
    assert encode(AUDIO_ONLY, b"\x01\x02") == b"\x11\x20\x00\x00" + b"\x00\x00\x00\x02" + b"\x01\x02"


@pytest.mark.parametrize("compress", [True, False])
def test_decode_server_response(compress):
    payload = {"result": {"text": "你好"}}
    assert decode(server_frame(payload, compress)) == payload


def test_decode_error_frame_raises():
    msg = b"bad request"
    frame = b"\x11\xf0\x10\x00" + struct.pack(">II", 45000001, len(msg)) + msg
    with pytest.raises(RuntimeError, match="45000001"):
        decode(frame)


def test_definite_utterance_is_emitted_once():
    asr = VolcASR(api_key="x")
    done = {"text": "你好。", "end_time": 900, "definite": True}
    doing = {"text": "我想", "end_time": 1500, "definite": False}
    assert asr._to_results({"result": {"utterances": [doing]}}) == [ASRResult("我想", False)]
    assert asr._to_results({"result": {"utterances": [done]}}) == [ASRResult("你好。", True)]
    # 全量模式下,已定稿的分句会随后续响应重复出现,不应重复发出
    assert asr._to_results({"result": {"utterances": [done, doing]}}) == [ASRResult("我想", False)]
