import wave
from pathlib import Path

from dududu.providers.silero_vad import SileroVAD

HELLO = Path(__file__).parent / "data" / "hello.wav"  # 1s 静音 + 一句话 + 1s 静音
FRAME_BYTES = 1024


def test_detects_speech_between_silence():
    with wave.open(str(HELLO)) as f:
        pcm = f.readframes(f.getnframes())
    vad = SileroVAD()
    speech = [
        vad.speech_prob(pcm[i : i + FRAME_BYTES]) > 0.5
        for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES)
    ]
    assert not any(speech[:25])  # 开头 800ms 是静音
    assert not any(speech[-25:])  # 结尾 800ms 是静音
    assert sum(speech) > 30  # 中间至少 1s 人声
