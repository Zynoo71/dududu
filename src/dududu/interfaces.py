"""核心依赖的四个接口。providers/ 下是具体实现,核心代码只认这里的 Protocol。

音频格式全局约定: 16kHz / 单声道 / 16-bit 小端 PCM (bytes)。
格式转换(重采样、解码)是 AudioSource 实现自己的事。
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

SAMPLE_RATE = 16000
BYTES_PER_MS = SAMPLE_RATE * 2 // 1000  # 32


class AudioSource(Protocol):
    """音频从哪来: WebSocket、本地声卡、LiveKit、wav 文件……"""

    def frames(self) -> AsyncIterator[bytes]:
        """持续产出 PCM 块,块大小任意;流结束时迭代结束。"""
        ...


class VAD(Protocol):
    """逐帧判断是否有人声。只给概率,起止的时间判定由状态机做。"""

    frame_ms: int  # 该 VAD 要求的固定帧长

    def speech_prob(self, frame: bytes) -> float:
        """frame 恰好 frame_ms 长,返回 0~1 的人声概率。"""
        ...


class TurnDetector(Protocol):
    """根据一段 turn 的音频,判断用户是否说完了。"""

    def end_of_turn_prob(self, turn_audio: bytes) -> float:
        """turn_audio 是本 turn 至今的全部音频,返回 0~1 的"已说完"概率。同步调用,可能耗时几十毫秒。"""
        ...


@dataclass(frozen=True)
class ASRResult:
    text: str  # 当前分句的全量文本
    is_final: bool  # True 表示该分句不再变化,之后的结果属于下一个分句


class ASR(Protocol):
    """流式识别会话: 一边送音频,一边出结果。"""

    async def start(self) -> None: ...

    async def send(self, pcm: bytes) -> None: ...

    def results(self) -> AsyncIterator[ASRResult]: ...

    async def close(self) -> None: ...
