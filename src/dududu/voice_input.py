"""VoiceInput: 把音频源、VAD、TurnDetector、ASR 和状态机接起来,对外只暴露事件流。

    voice = VoiceInput(source=..., vad=..., turn_detector=..., new_asr=lambda: VolcASR(...))
    async for event in voice.events():
        ...

ASR 由 VAD 门控: 没人说话时不连接、不发送音频。每个 turn 开一个 ASR 会话,
SpeechStarted 时建立(先补发开口前的预留音频),turn 结束时关闭。
"""

import asyncio
from collections.abc import AsyncIterator, Callable

from dududu.events import CommittedFinal, Event, SpeechStarted, TurnDropped
from dududu.interfaces import ASR, BYTES_PER_MS, VAD, AudioSource, TurnDetector
from dududu.machine import TurnConfig, TurnStateMachine


class VoiceInput:
    def __init__(
        self,
        *,
        source: AudioSource,
        vad: VAD,
        turn_detector: TurnDetector,
        new_asr: Callable[[], ASR],  # 每个 turn 调一次,返回一个新的 ASR 会话
        config: TurnConfig | None = None,
    ):
        self._source = source
        self._vad = vad
        self._turn_detector = turn_detector
        self._new_asr = new_asr
        self._machine = TurnStateMachine(config)
        self._queue: asyncio.Queue[Event | Exception | None] = asyncio.Queue()
        self._asr_audio: asyncio.Queue[bytes | None] | None = None  # 当前 turn 待发给 ASR 的音频
        self._tasks: set[asyncio.Task] = set()

    async def events(self) -> AsyncIterator[Event]:
        """音频源结束时迭代结束。只能调用一次。"""
        task = asyncio.create_task(self._run())
        try:
            while (item := await self._queue.get()) is not None:
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            task.cancel()
            for t in self._tasks:
                t.cancel()

    async def _run(self) -> None:
        try:
            await self._audio_loop()
            self._close_asr()
            await asyncio.gather(*self._tasks)
        except Exception as e:
            self._queue.put_nowait(e)
        self._queue.put_nowait(None)

    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _emit(self, events: list[Event]) -> None:
        for event in events:
            self._queue.put_nowait(event)
            if isinstance(event, SpeechStarted):
                self._asr_audio = asyncio.Queue()
                self._asr_audio.put_nowait(self._machine.turn_audio)  # 含开口前的预留音频
                self._spawn(self._asr_session(self._asr_audio))
            elif isinstance(event, (CommittedFinal, TurnDropped)):
                self._close_asr()

    def _close_asr(self) -> None:
        if self._asr_audio is not None:
            self._asr_audio.put_nowait(None)
            self._asr_audio = None

    async def _audio_loop(self) -> None:
        frame_bytes = self._vad.frame_ms * BYTES_PER_MS
        buffer = b""
        async for chunk in self._source.frames():
            buffer += chunk
            while len(buffer) >= frame_bytes:
                frame, buffer = buffer[:frame_bytes], buffer[frame_bytes:]
                asr_audio = self._asr_audio
                self._emit(self._machine.on_frame(frame, self._vad.speech_prob(frame)))
                if asr_audio is not None and asr_audio is self._asr_audio:
                    asr_audio.put_nowait(frame)
                if self._machine.wants_turn_check:
                    self._machine.wants_turn_check = False
                    self._spawn(self._turn_check(self._machine.turn_audio))

    async def _turn_check(self, audio: bytes) -> None:
        prob = await asyncio.to_thread(self._turn_detector.end_of_turn_prob, audio)
        self._emit(self._machine.on_turn_check(prob))

    async def _asr_session(self, audio: asyncio.Queue[bytes | None]) -> None:
        asr = self._new_asr()

        async def send() -> None:
            while (pcm := await audio.get()) is not None:
                await asr.send(pcm)
            await asr.close()

        try:
            await asr.start()
            sender = asyncio.create_task(send())
            try:
                async for result in asr.results():
                    if audio is self._asr_audio:  # 已结束的 turn 迟到的结果直接丢弃
                        self._emit(self._machine.on_asr(result))
                await sender
            finally:
                sender.cancel()
        except Exception as e:
            self._queue.put_nowait(e)
