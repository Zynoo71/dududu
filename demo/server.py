"""测试用的对话 demo: 浏览器麦克风 → VoiceInput → LLM → TTS → 浏览器播放。

    uv run --all-extras --group demo uvicorn demo.server:app --port 8000

这个文件就是"下游"的参考实现: 它只消费 VoiceInput 的事件流,
打断判定(Session.on_event)也写在这里,而不是在 dududu 包里。

和浏览器之间的 WebSocket 协议:
    浏览器 → 后端   二进制: 16kHz 单声道 16-bit PCM
                    文本:   {"playing": bool}          浏览器是否正在播放 agent 的声音
    后端 → 浏览器   二进制: 24kHz 单声道 16-bit PCM    agent 的语音
                    文本:   {"type": "event" | "reply_text" | "reply_end" | "latency"
                                     | "pause" | "resume" | "clear" | "error", ...}
"""

import asyncio
import base64
import json
import os
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from openai import AsyncOpenAI

from dududu.events import CommittedFinal, Event, InterimTranscript, SpeechStarted, TurnDropped
from dududu.providers.push_source import PushSource
from dududu.providers.silero_vad import SileroVAD
from dududu.providers.smart_turn import SmartTurn
from dududu.providers.volc_asr import VolcASR
from dududu.voice_input import VoiceInput

if Path(".env").exists():
    for line in Path(".env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            os.environ.setdefault(*line.split("=", 1))

API_KEY = os.environ["VOLC_API_KEY"]
# LLM 可换成任意 OpenAI 兼容接口;不配置则用 Agent Plan 的豆包
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
LLM_API_KEY = os.environ.get("LLM_API_KEY", API_KEY)
LLM_MODEL = os.environ.get("LLM_MODEL", "doubao-seed-2.0-mini")
# 豆包默认开着思考,首字会慢 2 秒以上;这个参数是方舟专有的,其他厂商不发
LLM_EXTRA_BODY = {"thinking": {"type": "disabled"}} if "volces.com" in LLM_BASE_URL else None
TTS_SPEAKER = os.environ.get("TTS_SPEAKER", "zh_female_vv_uranus_bigtts")
TTS_URL = "https://openspeech.bytedance.com/api/v3/plan/tts/unidirectional"
SYSTEM_PROMPT = "你是一个语音助手。回答要口语化、简短,一般不超过三句话。不要使用 markdown、列表或表情符号。"
SENTENCE_END = re.compile(r"[。!?!?;;\n]")

llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
http = httpx.AsyncClient(timeout=30)
turn_detector = SmartTurn()  # 无状态,所有连接共用
app = FastAPI()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    await Session(ws).run()


async def synthesize(text: str) -> AsyncIterator[bytes]:
    """流式合成,产出 24kHz 单声道 16-bit PCM。"""
    headers = {"X-Api-Key": API_KEY, "X-Api-Resource-Id": "seed-tts-2.0"}
    audio_params = {"format": "pcm", "sample_rate": 24000}
    body = {"req_params": {"text": text, "speaker": TTS_SPEAKER, "audio_params": audio_params}}
    async with http.stream("POST", TTS_URL, headers=headers, json=body) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line:
                continue
            msg = json.loads(line)
            if msg.get("data"):
                yield base64.b64decode(msg["data"])
            elif msg["code"] not in (0, 20000000):  # 20000000 是正常结束
                raise RuntimeError(f"TTS 错误: {msg}")


def is_meaningful(text: str) -> bool:
    """去掉标点后至少两个字才算"用户真的在说话"。"嗯"、"啊"这类附和不算。"""
    return len(re.sub(r"[\W_]", "", text)) >= 2


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.source = PushSource()
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.reply: asyncio.Task | None = None
        self.playing = False  # 浏览器上报: 是否正在播放
        self.paused = False  # 用户在 agent 说话时开口了,播放已暂停,尚未判定是否真打断
        self.committed_at = 0.0
        self.awaiting_first_audio = False

    async def run(self) -> None:
        events = asyncio.create_task(self.consume_events())
        try:
            while True:
                msg = await self.ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    self.source.push(msg["bytes"])
                elif msg.get("text"):
                    self.playing = json.loads(msg["text"])["playing"]
        finally:
            events.cancel()
            if self.reply:
                self.reply.cancel()

    async def send(self, type: str, **fields) -> None:
        await self.ws.send_text(json.dumps({"type": type, **fields}, ensure_ascii=False))

    async def consume_events(self) -> None:
        voice = VoiceInput(
            source=self.source,
            vad=SileroVAD(),
            turn_detector=turn_detector,
            new_asr=lambda: VolcASR(api_key=API_KEY, end_window_ms=400),
        )
        try:
            async for event in voice.events():
                await self.send(
                    "event",
                    name=type(event).__name__,
                    turn_id=event.turn_id,
                    audio_ms=event.audio_ms,
                    text=getattr(event, "text", None),
                )
                await self.on_event(event)
        except Exception as e:
            await self.send("error", message=f"{type(e).__name__}: {e}")

    # ---- 下游逻辑: 打断判定 + 何时回复 ----

    def agent_busy(self) -> bool:
        return self.playing or (self.reply is not None and not self.reply.done())

    async def on_event(self, event: Event) -> None:
        if isinstance(event, SpeechStarted):
            if self.agent_busy():  # 疑似打断: 先暂停,等文本出来再决定
                self.paused = True
                await self.send("pause")
        elif isinstance(event, InterimTranscript):
            if self.paused and is_meaningful(event.text):  # 确认打断
                self.paused = False
                self.playing = False
                self.reply.cancel()
                await self.send("clear")
        elif isinstance(event, TurnDropped):
            if self.paused:  # 假打断(咳嗽、噪声): 恢复播放
                self.paused = False
                await self.send("resume")
        elif isinstance(event, CommittedFinal):
            if self.paused:  # 只是附和("嗯"): 恢复播放,不当作新一轮输入
                self.paused = False
                await self.send("resume")
                return
            if self.reply:
                self.reply.cancel()
            self.committed_at = time.perf_counter()
            self.awaiting_first_audio = True
            self.reply = asyncio.create_task(self.respond(event.text))

    async def respond(self, user_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        spoken = ""
        try:
            stream = await llm.chat.completions.create(
                model=LLM_MODEL,
                messages=self.history,
                stream=True,
                extra_body=LLM_EXTRA_BODY,
            )
            pending = ""
            async for chunk in stream:
                if not chunk.choices:
                    continue
                pending += chunk.choices[0].delta.content or ""
                while match := SENTENCE_END.search(pending):  # 攒够一句就送去合成
                    sentence, pending = pending[: match.end()], pending[match.end() :]
                    await self.speak(sentence)
                    spoken += sentence
            await self.speak(pending)
            spoken += pending
            await self.send("reply_end")
        except Exception as e:
            await self.send("error", message=f"{type(e).__name__}: {e}")
        finally:
            if spoken:  # 被打断时只记录已经送去播放的部分
                self.history.append({"role": "assistant", "content": spoken})

    async def speak(self, sentence: str) -> None:
        if not sentence.strip():
            return
        await self.send("reply_text", text=sentence)
        leftover = b""  # 保证每次发出的字节数是偶数(16-bit 采样不能被拆开)
        async for pcm in synthesize(sentence):
            pcm = leftover + pcm
            cut = len(pcm) & ~1
            pcm, leftover = pcm[:cut], pcm[cut:]
            if self.awaiting_first_audio:
                self.awaiting_first_audio = False
                await self.send("latency", ms=round((time.perf_counter() - self.committed_at) * 1000))
            await self.ws.send_bytes(pcm)
