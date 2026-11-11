"""输入端对外的事件契约。

一个 turn 的生命周期:

    SpeechStarted
      → (InterimTranscript | SpeculativeFinal | TurnRetracted)*
      → CommittedFinal | TurnDropped        # 二选一,之后该 turn 不再有事件

所有事件只陈述事实,不判断"是否打断"——那由下游结合自己的播放状态决定。
"""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class Event:
    turn_id: int  # 从 1 递增,SpeechStarted 时开新 turn
    seq: int  # 全局单调递增,下游据此丢弃过期事件
    audio_ms: int  # 事件对应的音频时间(自流开始的毫秒数),离线回放时可复现


@dataclass(frozen=True, kw_only=True)
class SpeechStarted(Event):
    """VAD 检测到用户开口。"""


@dataclass(frozen=True, kw_only=True)
class InterimTranscript(Event):
    """本 turn 到目前为止的全量文本(覆盖上一条,不是增量)。"""

    text: str


@dataclass(frozen=True, kw_only=True)
class SpeculativeFinal(Event):
    """判定用户大概率说完了。可撤销,下游可据此投机预生成。"""

    text: str


@dataclass(frozen=True, kw_only=True)
class TurnRetracted(Event):
    """用户在 SpeculativeFinal 之后又接着说了。上一条 SpeculativeFinal 作废,turn 继续。"""


@dataclass(frozen=True, kw_only=True)
class CommittedFinal(Event):
    """turn 结束,文本不可撤销。"""

    text: str


@dataclass(frozen=True, kw_only=True)
class TurnDropped(Event):
    """turn 结束但没有识别出文本(咳嗽、噪声等)。"""
