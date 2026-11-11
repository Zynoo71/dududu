"""火山双向流式语音合成: 文本可以一小段一小段地送进去,音频边合成边返回。

协议文档: https://docs.volcengine.com/docs/6561/1329505

    tts = TTSStream(api_key, speaker)
    await tts.start()                 # 可以和 LLM 请求并行,提前建好
    await tts.send_text("你好,")      # LLM 每出一段就送一段
    await tts.finish()                # 文本送完
    async for pcm in tts.audio(): ... # 24kHz 单声道 16-bit PCM,合成结束时迭代结束
    await tts.close()                 # 任何时候都可以调,用于打断

一个实例对应一次回复(一条连接、一个会话)。
"""

import json
import struct
import uuid
from collections.abc import AsyncIterator

from websockets.asyncio.client import connect

URL = "wss://openspeech.bytedance.com/api/v3/plan/tts/bidirection"

START_CONNECTION, CONNECTION_STARTED = 1, 50
START_SESSION, FINISH_SESSION, SESSION_STARTED = 100, 102, 150
SESSION_FINISHED, SESSION_FAILED = 152, 153
TASK_REQUEST, AUDIO = 200, 352


def encode(event: int, payload: dict, session_id: str | None = None) -> bytes:
    frame = bytes([0x11, 0x14, 0x10, 0x00]) + struct.pack(">i", event)  # JSON,带 event,不压缩
    if session_id:
        frame += struct.pack(">I", len(session_id)) + session_id.encode()
    body = json.dumps(payload, ensure_ascii=False).encode()
    return frame + struct.pack(">I", len(body)) + body


def decode(frame: bytes) -> tuple[int, bytes]:
    """返回 (event, payload)。错误帧抛异常。"""
    if frame[1] >> 4 == 0xF:
        (code,) = struct.unpack(">I", frame[4:8])
        raise RuntimeError(f"TTS 错误 {code}: {frame[8:].decode(errors='replace')}")
    (event,) = struct.unpack(">i", frame[4:8])
    (id_len,) = struct.unpack(">I", frame[8:12])  # connect id 或 session id
    pos = 12 + id_len
    (size,) = struct.unpack(">I", frame[pos : pos + 4])
    return event, frame[pos + 4 : pos + 4 + size]


class TTSStream:
    def __init__(self, api_key: str, speaker: str, sample_rate: int = 24000):
        self._headers = {
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": "seed-tts-2.0",
            "X-Api-Connect-Id": str(uuid.uuid4()),
        }
        self._session_id = uuid.uuid4().hex
        self._params = {
            "speaker": speaker,
            "audio_params": {"format": "pcm", "sample_rate": sample_rate},
        }
        self._ws = None

    def _request(self, event: int, **req_params) -> bytes:
        payload = {
            "user": {"uid": "dududu"},
            "event": event,
            "namespace": "BidirectionalTTS",
            "req_params": {**self._params, **req_params},
        }
        return encode(event, payload, self._session_id)

    async def _expect(self, expected: int) -> None:
        event, payload = decode(await self._ws.recv())
        if event != expected:
            raise RuntimeError(f"TTS: 期望事件 {expected},收到 {event}: {payload[:200]!r}")

    async def start(self) -> None:
        self._ws = await connect(URL, additional_headers=self._headers, proxy=None)
        await self._ws.send(encode(START_CONNECTION, {}))
        await self._expect(CONNECTION_STARTED)
        await self._ws.send(self._request(START_SESSION))
        await self._expect(SESSION_STARTED)

    async def send_text(self, text: str) -> None:
        await self._ws.send(self._request(TASK_REQUEST, text=text))

    async def finish(self) -> None:
        await self._ws.send(encode(FINISH_SESSION, {}, self._session_id))

    async def audio(self) -> AsyncIterator[bytes]:
        async for frame in self._ws:
            event, payload = decode(frame)
            if event == AUDIO:
                yield payload
            elif event == SESSION_FINISHED:
                return
            elif event == SESSION_FAILED:
                raise RuntimeError(f"TTS 会话失败: {payload.decode(errors='replace')}")

    async def close(self) -> None:
        if self._ws:
            await self._ws.close()
