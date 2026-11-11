"""Coordinator: 消费 VoiceInput 的事件,决定 agent 该继续、排队、被打断还是停下。

它是 VoiceInput 之上的可选一层。VoiceInput 只陈述"用户说了什么",
Coordinator 结合 agent 的状态把它变成对 agent 的指令。它只管时序和状态:
判断交给 InterjectionJudge,执行交给宿主实现的 AgentControl。

    coordinator = Coordinator(agent=my_agent, judge=JevJudge(...))
    async for event in voice.events():
        await coordinator.on_event(event)
    # 另外,agent 说完一轮(生成结束且播放完)时调用 coordinator.on_agent_idle()
"""

from collections import deque
from typing import Protocol

from dududu.events import CommittedFinal, Event, InterimTranscript, SpeechStarted, TurnDropped
from dududu.interjection import (
    Decision,
    Interjection,
    InterjectionJudge,
    Judgment,
    is_filler,
    is_short_ack,
)


class AgentControl(Protocol):
    """宿主(demo 或你的 agent)需要实现的全部能力。"""

    def is_busy(self) -> bool:
        """agent 是否正在生成或播放回复。"""
        ...

    def speaking_text(self) -> str:
        """agent 这一轮正在说的内容,给判断器做上下文。"""
        ...

    async def duck(self) -> None:
        """压低播放音量,但不停。"""
        ...

    async def pause(self) -> None:
        """暂停播放,保留进度。"""
        ...

    async def resume(self) -> None:
        """恢复正常音量并继续播放。"""
        ...

    async def cancel(self) -> None:
        """丢弃当前回复,清空播放。"""
        ...

    async def respond(self, user_text: str) -> None:
        """开始回复一句用户输入。应立即返回,不要等回复说完。"""
        ...


class Coordinator:
    def __init__(
        self,
        agent: AgentControl,
        judge: InterjectionJudge,
        min_confidence: float = 0.6,  # 置信度低于它时一律按打断处理
    ):
        self._agent = agent
        self._judge = judge
        self._min_confidence = min_confidence
        self._hold: str | None = None  # 本次插话对播放做了什么: None / "duck" / "pause"
        self._queue: deque[str] = deque()  # 等 agent 说完再处理的用户输入
        self.last_judgment: Judgment | None = None  # 最近一次判断,供宿主展示或记录

    async def on_event(self, event: Event) -> None:
        if isinstance(event, SpeechStarted):
            if self._agent.is_busy():
                self._hold = "duck"
                await self._agent.duck()
        elif isinstance(event, InterimTranscript):
            if self._hold == "duck" and not is_short_ack(event.text):
                self._hold = "pause"  # 用户在说正经话,先停下来听
                await self._agent.pause()
        elif isinstance(event, TurnDropped):
            await self._release()
        elif isinstance(event, CommittedFinal):
            await self._on_final(event.text)

    async def on_agent_idle(self) -> None:
        """agent 说完了一轮。有排队的输入就接着处理。"""
        if self._queue and self._hold is None and not self._agent.is_busy():
            await self._agent.respond(self._queue.popleft())

    async def _on_final(self, text: str) -> None:
        if self._hold is None or not self._agent.is_busy():
            # 用户开口时 agent 是空闲的,或者在用户说话期间 agent 已经说完: 普通的一轮
            self._hold = None
            await self._agent.respond(text)
            return
        if is_filler(text):
            await self._release()
            return

        decision = await self._decide(Interjection(self._agent.speaking_text(), text))
        if decision is Decision.CONTINUE:
            await self._release()
        elif decision is Decision.QUEUE:
            self._queue.append(text)
            await self._release()
        elif decision is Decision.INTERRUPT:
            self._hold = None
            await self._agent.cancel()
            await self._agent.respond(text)
        elif decision is Decision.STOP:
            self._hold = None
            self._queue.clear()
            await self._agent.cancel()

    async def _decide(self, interjection: Interjection) -> Decision:
        """拿不准或判断器出错时倒向打断: 无视用户比误打断严重得多。"""
        try:
            judgment = await self._judge.judge(interjection)
        except Exception:
            judgment = Judgment(Decision.INTERRUPT, 0.0)
        self.last_judgment = judgment
        if judgment.confidence < self._min_confidence:
            return Decision.INTERRUPT
        return judgment.decision

    async def _release(self) -> None:
        if self._hold is not None:
            self._hold = None
            await self._agent.resume()
