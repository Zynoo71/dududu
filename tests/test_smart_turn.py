import wave
from pathlib import Path

from dududu.interfaces import BYTES_PER_MS
from dududu.providers.smart_turn import SmartTurn

HELLO = Path(__file__).parent / "data" / "hello.wav"
# "你好,我想查一下我的订单 order status,谢谢",语音从 1000ms 开始


def test_mid_sentence_is_incomplete_and_full_sentence_is_complete():
    with wave.open(str(HELLO)) as f:
        pcm = f.readframes(f.getnframes())
    turn = SmartTurn()
    start = 1000 * BYTES_PER_MS
    mid_sentence = pcm[start : 3000 * BYTES_PER_MS]  # 截到"…我的订单"中间
    assert turn.end_of_turn_prob(mid_sentence) < 0.5
    assert turn.end_of_turn_prob(pcm[start:]) > 0.5
