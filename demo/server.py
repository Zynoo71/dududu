"""测试用的对话 demo: 浏览器麦克风 → VoiceInput → LLM → TTS → 浏览器播放。

    uv run --all-extras --group demo uvicorn demo.server:app --port 8000

这个文件是"下游"的参考实现: VoiceInput 的事件交给 Coordinator,
Coordinator 判断该继续、排队、打断还是停下,再通过 Session 实现的 AgentControl 指挥播放和回复。

和浏览器之间的 WebSocket 协议:
    浏览器 → 后端   二进制: 16kHz 单声道 16-bit PCM
                    文本:   {"playing": bool}          浏览器是否正在播放 agent 的声音
    后端 → 浏览器   二进制: 24kHz 单声道 16-bit PCM    agent 的语音
                    文本:   {"type": "event" | "judgment" | "reply_text" | "reply_end" | "latency"
                                     | "duck" | "pause" | "resume" | "clear" | "error", ...}
"""

import asyncio
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from openai import AsyncOpenAI

from demo.tts import TTSStream
from dududu.coordinator import Coordinator
from dududu.interjection import InterjectionJudge
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
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "你叫 Agnes,是一个语音助手。被问到身份时只说自己是 Agnes,不要提豆包、字节跳动或底层模型。"
    "回答要口语化、简短,一般不超过三句话。不要使用 markdown、列表或表情符号。",
)

llm = AsyncOpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
turn_detector = SmartTurn()  # 无状态,所有连接共用
app = FastAPI()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    await Session(ws).run()


def make_judge() -> InterjectionJudge:
    """有 TypeSafe 的 key 就用 Jev,否则用 LLM。可用 JUDGE=llm / JUDGE=jev 强制指定。"""
    use_jev = os.environ.get("JUDGE", "jev" if os.environ.get("TYPESAFE_API_KEY") else "llm") == "jev"
    if use_jev:
        from dududu.providers.jev_judge import JevJudge

        return JevJudge(api_key=os.environ["TYPESAFE_API_KEY"])
    from dududu.providers.llm_judge import LLMJudge

    return LLMJudge(LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_EXTRA_BODY)


judge = make_judge()


class Session:
    """一条浏览器连接。同时是 Coordinator 的 AgentControl 实现(下面标了的那几个方法)。"""

    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.source = PushSource()
        self.coordinator = Coordinator(agent=self, judge=judge)
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.reply: asyncio.Task | None = None
        self.reply_text = ""  # 当前这轮回复已生成的文本
        self.generating = False
        self.playing = False  # 浏览器上报: 是否正在播放
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
                    await self.check_idle()
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
                before, start = self.coordinator.last_judgment, time.perf_counter()
                await self.coordinator.on_event(event)
                judgment = self.coordinator.last_judgment
                if judgment is not before:  # 这次事件触发了一次插话判断,发给页面展示
                    await self.send(
                        "judgment",
                        decision=judgment.decision.value,
                        confidence=round(judgment.confidence, 2),
                        ms=round((time.perf_counter() - start) * 1000),
                    )
        except Exception as e:
            await self.send("error", message=f"{type(e).__name__}: {e}")

    async def check_idle(self) -> None:
        if not self.is_busy():
            await self.coordinator.on_agent_idle()

    # ---- AgentControl: Coordinator 通过这些方法指挥 agent ----

    def is_busy(self) -> bool:
        return self.generating or self.playing

    def speaking_text(self) -> str:
        return self.reply_text

    async def duck(self) -> None:
        await self.send("duck")

    async def pause(self) -> None:
        await self.send("pause")

    async def resume(self) -> None:
        await self.send("resume")

    async def cancel(self) -> None:
        if self.reply:
            self.reply.cancel()
        self.generating = False
        self.playing = False
        await self.send("clear")

    async def respond(self, user_text: str) -> None:
        if self.reply:
            self.reply.cancel()
        self.committed_at = time.perf_counter()
        self.awaiting_first_audio = True
        self.generating = True
        self.reply_text = ""
        self.reply = asyncio.create_task(self.generate_reply(user_text))

    # ---- 生成回复: LLM → TTS → 浏览器 ----

    async def generate_reply(self, user_text: str) -> None:
        self.history.append({"role": "user", "content": user_text})
        spoken = ""
        tts = TTSStream(API_KEY, TTS_SPEAKER)
        try:
            # LLM 请求和 TTS 建连并行,TTS 建连的耗时被 LLM 的首字等待盖住
            stream, _ = await asyncio.gather(
                llm.chat.completions.create(
                    model=LLM_MODEL, messages=self.history, stream=True, extra_body=LLM_EXTRA_BODY
                ),
                tts.start(),
            )

            async def feed_text() -> None:  # LLM 出一段就送一段,断句交给 TTS 服务端
                nonlocal spoken
                async for chunk in stream:
                    if chunk.choices and (delta := chunk.choices[0].delta.content):
                        await tts.send_text(delta)
                        spoken += delta
                        self.reply_text = spoken
                        await self.send("reply_text", text=delta)
                await tts.finish()

            feeder = asyncio.create_task(feed_text())
            try:
                async for pcm in tts.audio():
                    if self.awaiting_first_audio:
                        self.awaiting_first_audio = False
                        elapsed = time.perf_counter() - self.committed_at
                        await self.send("latency", ms=round(elapsed * 1000))
                    await self.ws.send_bytes(pcm)
                await feeder
            finally:
                feeder.cancel()
            await self.send("reply_end")
            self.generating = False
            await self.check_idle()
        except Exception as e:
            self.generating = False
            await self.send("error", message=f"{type(e).__name__}: {e}")
        finally:
            await tts.close()
            if spoken:  # 被打断时记录的是已送去合成的文本,可能比用户实际听到的多
                self.history.append({"role": "assistant", "content": spoken})
