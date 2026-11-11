"""TurnStateMachine: 把 VAD 概率、ASR 结果、"是否说完"的判断合成事件流。

纯同步、无 IO,三个输入各对应一个方法,返回要发出的事件列表:

    on_frame(frame, speech_prob)   每个 VAD 帧
    on_asr(result)                 每条 ASR 结果
    on_turn_check(prob)            TurnDetector 的判断结果

状态机不自己调 TurnDetector(那要几十毫秒)。需要判断时它把 wants_turn_check 置为 True,
由外层拿 turn_audio 去跑,再把结果通过 on_turn_check 送回来。
"""

from dataclasses import dataclass

from dududu.events import (
    CommittedFinal,
    Event,
    InterimTranscript,
    SpeculativeFinal,
    SpeechStarted,
    TurnDropped,
    TurnRetracted,
)
from dududu.interfaces import BYTES_PER_MS, ASRResult


@dataclass
class TurnConfig:
    speech_threshold: float = 0.5  # VAD 概率超过它算人声
    start_ms: int = 96  # 连续这么久人声才算开口(3 个 Silero 帧)
    turn_check_ms: int = 300  # 静音每满这么久,问一次"说完了吗"
    turn_threshold: float = 0.5  # "已说完"概率超过它算说完
    commit_ms: int = 600  # 判定说完后,静音至少这么久才提交
    no_text_timeout_ms: int = 1500  # 静音这么久还没有任何文本,丢弃该 turn
    max_silence_ms: int = 3000  # 静音这么久强制结束,不管是否判定说完
    preroll_ms: int = 300  # turn 音频里保留开口前这么久
    max_turn_audio_ms: int = 10_000  # turn 音频只保留最后这么久(Smart Turn 只看最后 8 秒)


class TurnStateMachine:
    def __init__(self, config: TurnConfig | None = None):
        self.config = config or TurnConfig()
        self.wants_turn_check = False
        self._audio = bytearray()
        self._audio_ms = 0
        self._seq = 0
        self._turn_id = 0
        self._in_turn = False
        self._speech_run_ms = 0
        self._silence_ms = 0
        self._reset_turn()

    def _reset_turn(self) -> None:
        self._in_turn = False
        self._finals: list[str] = []
        self._interim = ""
        self._last_text = ""
        self._complete = False  # TurnDetector 判定说完了
        self._speculated_text: str | None = None  # 已发出的 SpeculativeFinal 的文本
        self._check_pending = False
        self._next_check_ms = self.config.turn_check_ms
        self.wants_turn_check = False

    @property
    def turn_audio(self) -> bytes:
        return bytes(self._audio)

    # ---- 三个输入 ----

    def on_frame(self, frame: bytes, speech_prob: float) -> list[Event]:
        cfg = self.config
        ms = len(frame) // BYTES_PER_MS
        self._audio_ms += ms
        self._audio += frame
        keep_ms = cfg.max_turn_audio_ms if self._in_turn else cfg.preroll_ms + cfg.start_ms
        del self._audio[: -keep_ms * BYTES_PER_MS]

        if speech_prob > cfg.speech_threshold:
            self._speech_run_ms += ms
        else:
            self._speech_run_ms = 0
            self._silence_ms += ms

        if self._speech_run_ms >= cfg.start_ms:  # 用户正在说话
            self._silence_ms = 0
            self._next_check_ms = cfg.turn_check_ms
            self._check_pending = False
            self.wants_turn_check = False
            self._complete = False
            if not self._in_turn:
                self._in_turn = True
                self._turn_id += 1
                return [self._event(SpeechStarted)]
            if self._speculated_text is not None:
                self._speculated_text = None
                return [self._event(TurnRetracted)]
            return []

        if not self._in_turn:
            return []
        if not self._complete and not self._check_pending and self._silence_ms >= self._next_check_ms:
            self._next_check_ms += cfg.turn_check_ms
            self._check_pending = True
            self.wants_turn_check = True
        return self._end_or_speculate()

    def on_asr(self, result: ASRResult) -> list[Event]:
        if not self._in_turn:
            return []
        if result.is_final:
            self._finals.append(result.text)
            self._interim = ""
        else:
            self._interim = result.text
        events: list[Event] = []
        if self._text() != self._last_text:
            self._last_text = self._text()
            events.append(self._event(InterimTranscript, text=self._last_text))
        return events + self._end_or_speculate()

    def on_turn_check(self, prob: float) -> list[Event]:
        self.wants_turn_check = False
        if not self._check_pending:  # 判断期间用户又开口了,结果作废
            return []
        self._check_pending = False
        self._complete = prob > self.config.turn_threshold
        return self._end_or_speculate()

    # ---- 内部 ----

    def _text(self) -> str:
        return " ".join([*self._finals, self._interim]).strip()

    def _event(self, cls, **fields) -> Event:
        self._seq += 1
        return cls(turn_id=self._turn_id, seq=self._seq, audio_ms=self._audio_ms, **fields)

    def _end_or_speculate(self) -> list[Event]:
        cfg = self.config
        text = self._text()
        settled = bool(self._finals) and not self._interim  # ASR 已定稿
        if (
            self._silence_ms >= cfg.max_silence_ms
            or (self._complete and settled and self._silence_ms >= cfg.commit_ms)
            or (not text and self._silence_ms >= cfg.no_text_timeout_ms)
        ):
            event = self._event(CommittedFinal, text=text) if text else self._event(TurnDropped)
            self._reset_turn()
            return [event]
        if self._complete and text and text != self._speculated_text:
            self._speculated_text = text
            return [self._event(SpeculativeFinal, text=text)]
        return []
