"""从 wav 文件读音频,用于离线回放测试。"""

import asyncio
import wave
from collections.abc import AsyncIterator

from dududu.interfaces import BYTES_PER_MS, SAMPLE_RATE


class WavSource:
    def __init__(self, path: str, chunk_ms: int = 20, realtime: bool = False):
        self.path = path
        self.chunk_bytes = chunk_ms * BYTES_PER_MS
        self.delay = chunk_ms / 1000 if realtime else 0  # realtime=True 时按真实语速播放

    async def frames(self) -> AsyncIterator[bytes]:
        with wave.open(self.path, "rb") as f:
            fmt = (f.getframerate(), f.getnchannels(), f.getsampwidth())
            if fmt != (SAMPLE_RATE, 1, 2):
                raise ValueError(f"需要 16kHz 单声道 16-bit wav,实际是 {fmt}")
            pcm = f.readframes(f.getnframes())
        for i in range(0, len(pcm), self.chunk_bytes):
            yield pcm[i : i + self.chunk_bytes]
            await asyncio.sleep(self.delay)
