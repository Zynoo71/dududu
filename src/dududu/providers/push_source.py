"""由外部主动推送音频的 AudioSource,用于对接任意传输层(WebSocket、LiveKit 回调、声卡回调……)。

    source = PushSource()
    source.push(pcm)   # 每收到一块 16kHz 单声道 16-bit PCM 就推进来
    source.close()     # 音频结束
"""

import asyncio
from collections.abc import AsyncIterator


class PushSource:
    def __init__(self):
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    def push(self, pcm: bytes) -> None:
        self._queue.put_nowait(pcm)

    def close(self) -> None:
        self._queue.put_nowait(None)

    async def frames(self) -> AsyncIterator[bytes]:
        while (pcm := await self._queue.get()) is not None:
            yield pcm
