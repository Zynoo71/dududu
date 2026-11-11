"""整条链路的离线回放: 真实的 Silero VAD + Smart Turn,ASR 用按音频进度出结果的假实现。"""

import asyncio
from pathlib import Path

from dududu.events import CommittedFinal, InterimTranscript, SpeculativeFinal, SpeechStarted
from dududu.interfaces import BYTES_PER_MS, ASRResult
from dududu.providers.silero_vad import SileroVAD
from dududu.providers.smart_turn import SmartTurn
from dududu.providers.wav_source import WavSource
from dududu.voice_input import VoiceInput

HELLO = Path(__file__).parent / "data" / "hello.wav"  # 共 6528ms,语音在 1000~5500ms
# ASR 会话从开口前约 300ms 开始,所以脚本里的时间比文件时间早约 700ms


class FakeASR:
    """收到的音频达到 at_ms 时吐出对应结果。"""

    sessions = []  # 每个会话实际收到的音频毫秒数

    def __init__(self, script: list[tuple[int, ASRResult]]):
        self._script = list(script)
        self._received_ms = 0
        self._queue = asyncio.Queue()

    async def start(self):
        pass

    async def send(self, pcm):
        self._received_ms += len(pcm) // BYTES_PER_MS
        while self._script and self._script[0][0] <= self._received_ms:
            self._queue.put_nowait(self._script.pop(0)[1])

    async def results(self):
        while (item := await self._queue.get()) is not None:
            yield item

    async def close(self):
        FakeASR.sessions.append(self._received_ms)
        self._queue.put_nowait(None)


async def replay():
    script = [
        (1500, ASRResult("你好", False)),
        (3800, ASRResult("你好,我想查一下我的订单", False)),
        (5100, ASRResult("你好,我想查一下我的订单 order status,谢谢", False)),
        (5500, ASRResult("你好,我想查一下我的订单 order status,谢谢。", True)),
    ]
    voice = VoiceInput(
        source=WavSource(str(HELLO), realtime=True, chunk_ms=20),
        vad=SileroVAD(),
        turn_detector=SmartTurn(),
        new_asr=lambda: FakeASR(script),
    )
    return [event async for event in voice.events()]


def test_replay_produces_one_committed_turn():
    events = asyncio.run(replay())
    kinds = [type(e) for e in events]
    assert kinds[0] is SpeechStarted
    assert kinds[-1] is CommittedFinal
    assert kinds.count(SpeechStarted) == 1  # 句内停顿没有把一句话切成两个 turn
    assert SpeculativeFinal in kinds and InterimTranscript in kinds
    assert events[-1].text == "你好,我想查一下我的订单 order status,谢谢。"
    assert [e.seq for e in events] == sorted(e.seq for e in events)


def test_asr_only_receives_audio_during_the_turn():
    FakeASR.sessions.clear()
    asyncio.run(replay())
    (received_ms,) = FakeASR.sessions  # 只开了一个会话
    assert 5000 < received_ms < 6000  # 文件开头约 700ms 的静音没有发给 ASR
