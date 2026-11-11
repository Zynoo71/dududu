"""火山引擎大模型流式语音识别 (双向流式优化版 bigmodel_async)。

协议文档: https://docs.volcengine.com/docs/6561/1354869
"""

import asyncio
import gzip
import json
import struct
import uuid
from collections.abc import AsyncIterator

from websockets.asyncio.client import connect

from dududu.interfaces import BYTES_PER_MS, ASRResult

DEFAULT_URL = "wss://openspeech.bytedance.com/api/v3/plan/sauc/bigmodel_async"
PACKET_BYTES = 100 * BYTES_PER_MS  # 文档建议每包 100~200ms

# 帧头第 2 字节: 高 4 位是消息类型,低 4 位是 flags
FULL_CLIENT_REQUEST = 0x10
AUDIO_ONLY = 0x20
AUDIO_ONLY_LAST = 0x22
SERVER_RESPONSE = 0x9
SERVER_ERROR = 0xF


def encode(type_and_flags: int, payload: bytes, is_json: bool = False) -> bytes:
    header = bytes([0x11, type_and_flags, 0x10 if is_json else 0x00, 0x00])  # 不压缩
    return header + struct.pack(">I", len(payload)) + payload


def decode(frame: bytes) -> dict:
    """解析服务端帧,返回 JSON payload;错误帧抛异常。"""
    msg_type, flags, compression = frame[1] >> 4, frame[1] & 0x0F, frame[2] & 0x0F
    body = frame[4:]
    if msg_type == SERVER_ERROR:
        code, size = struct.unpack(">II", body[:8])
        raise RuntimeError(f"火山 ASR 错误 {code}: {body[8 : 8 + size].decode()}")
    if msg_type != SERVER_RESPONSE:
        raise RuntimeError(f"未知的消息类型 {msg_type:#x}")
    if flags & 0x01:  # 带 sequence
        body = body[4:]
    (size,) = struct.unpack(">I", body[:4])
    payload = body[4 : 4 + size]
    if compression == 0x01:
        payload = gzip.decompress(payload)
    return json.loads(payload) if payload else {}


class VolcASR:
    def __init__(
        self,
        *,
        api_key: str | None = None,  # Agent Plan 专属 Key 或新版控制台 Key: 只需这一个
        app_key: str | None = None,  # 旧版控制台(需同时把 url 换成不带 /plan/ 的地址)
        access_key: str | None = None,
        resource_id: str = "volc.seedasr.sauc.duration",
        url: str = DEFAULT_URL,
        hotwords: list[str] | None = None,  # 直传热词,上限约 100 tokens
        end_window_ms: int = 800,  # 服务端分句的静音时长
        two_pass: bool = True,  # 二遍识别: 分句定稿时用非流式模型修正
    ):
        self._url = url
        self._headers = {
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "X-Api-Connect-Id": str(uuid.uuid4()),
            "X-Api-Sequence": "-1",
        }
        if api_key:
            self._headers["X-Api-Key"] = api_key
        else:
            self._headers |= {"X-Api-App-Key": app_key, "X-Api-Access-Key": access_key}
        self._request = {
            "user": {"uid": "dududu"},
            "audio": {"format": "pcm", "codec": "raw", "rate": 16000, "bits": 16, "channel": 1},
            "request": {
                "model_name": "bigmodel",
                "enable_nonstream": two_pass,
                "enable_itn": True,
                "enable_punc": True,
                "show_utterances": True,
                "end_window_size": end_window_ms,
            },
        }
        if hotwords:
            context = {"hotwords": [{"word": w} for w in hotwords]}
            self._request["request"]["corpus"] = {"context": json.dumps(context, ensure_ascii=False)}
        self._buffer = b""
        self._queue: asyncio.Queue[ASRResult | Exception | None] = asyncio.Queue()
        self._last_final_end = -1  # 已定稿分句的结束时间,用于去重

    async def start(self) -> None:
        # proxy=None: 不走系统代理,直连(否则系统配了 SOCKS 代理时会要求额外依赖)
        self._ws = await connect(self._url, additional_headers=self._headers, proxy=None)
        payload = json.dumps(self._request).encode()
        await self._ws.send(encode(FULL_CLIENT_REQUEST, payload, is_json=True))
        self._reader = asyncio.create_task(self._read())

    async def send(self, pcm: bytes) -> None:
        self._buffer += pcm
        while len(self._buffer) >= PACKET_BYTES:
            packet, self._buffer = self._buffer[:PACKET_BYTES], self._buffer[PACKET_BYTES:]
            await self._ws.send(encode(AUDIO_ONLY, packet))

    async def results(self) -> AsyncIterator[ASRResult]:
        while (item := await self._queue.get()) is not None:
            if isinstance(item, Exception):
                raise item
            yield item

    async def close(self) -> None:
        """发送最后一包,等服务端返回完剩余结果后关闭。"""
        await self._ws.send(encode(AUDIO_ONLY_LAST, self._buffer))
        self._buffer = b""
        await self._reader
        await self._ws.close()

    async def _read(self) -> None:
        try:
            async for frame in self._ws:
                for result in self._to_results(decode(frame)):
                    self._queue.put_nowait(result)
                if frame[1] & 0x02:  # 服务端的最后一包
                    break
        except Exception as e:
            self._queue.put_nowait(e)
        self._queue.put_nowait(None)

    def _to_results(self, response: dict) -> list[ASRResult]:
        results = []
        for u in response.get("result", {}).get("utterances", []):
            if not u.get("definite"):
                results.append(ASRResult(u["text"], is_final=False))
            elif u["end_time"] > self._last_final_end:
                self._last_final_end = u["end_time"]
                results.append(ASRResult(u["text"], is_final=True))
        return results
