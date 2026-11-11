import asyncio

from dududu.coordinator import Coordinator
from dududu.events import CommittedFinal, InterimTranscript, SpeechStarted, TurnDropped
from dududu.interjection import Decision, Judgment, is_filler, is_short_ack


class FakeAgent:
    def __init__(self, busy):
        self.busy = busy
        self.calls = []

    def is_busy(self):
        return self.busy

    def speaking_text(self):
        return "您的订单预计明天送达"

    async def duck(self):
        self.calls.append("duck")

    async def pause(self):
        self.calls.append("pause")

    async def resume(self):
        self.calls.append("resume")

    async def cancel(self):
        self.calls.append("cancel")
        self.busy = False

    async def respond(self, user_text):
        self.calls.append(("respond", user_text))
        self.busy = True


class FakeJudge:
    def __init__(self, decision=None, confidence=1.0, error=False):
        self.judgment = Judgment(decision, confidence)
        self.error = error
        self.asked = []

    async def judge(self, interjection):
        self.asked.append(interjection.user_text)
        if self.error:
            raise RuntimeError("judge down")
        return self.judgment


def event(cls, text=None):
    fields = {"turn_id": 1, "seq": 1, "audio_ms": 0}
    return cls(**fields) if text is None else cls(**fields, text=text)


def run(agent, judge, events):
    coordinator = Coordinator(agent, judge)

    async def go():
        for e in events:
            await coordinator.on_event(e)

    asyncio.run(go())
    return coordinator


def interjection(text):
    return [event(SpeechStarted), event(InterimTranscript, text), event(CommittedFinal, text)]


def test_rules():
    assert is_filler("嗯嗯,") and is_filler("") and not is_filler("好的")
    assert is_short_ack("嗯,好的好的。") and is_short_ack("OK") and not is_short_ack("好的那明天呢")


def test_idle_agent_just_responds():
    agent, judge = FakeAgent(busy=False), FakeJudge()
    run(agent, judge, interjection("帮我查订单"))
    assert agent.calls == [("respond", "帮我查订单")]
    assert judge.asked == []


def test_noise_ducks_then_resumes():
    agent = FakeAgent(busy=True)
    run(agent, FakeJudge(), [event(SpeechStarted), event(TurnDropped)])
    assert agent.calls == ["duck", "resume"]


def test_filler_resumes_without_asking_the_judge():
    agent, judge = FakeAgent(busy=True), FakeJudge()
    run(agent, judge, interjection("嗯嗯"))
    assert agent.calls == ["duck", "resume"]
    assert judge.asked == []


def test_short_ack_stays_ducked_and_is_judged():
    agent, judge = FakeAgent(busy=True), FakeJudge(Decision.CONTINUE)
    run(agent, judge, interjection("好的"))
    assert agent.calls == ["duck", "resume"]  # 没有暂停
    assert judge.asked == ["好的"]


def test_continue_pauses_then_resumes():
    agent = FakeAgent(busy=True)
    run(agent, FakeJudge(Decision.CONTINUE), interjection("对,就是这个订单"))
    assert agent.calls == ["duck", "pause", "resume"]


def test_interrupt_cancels_and_responds():
    agent = FakeAgent(busy=True)
    run(agent, FakeJudge(Decision.INTERRUPT), interjection("等一下,改成明天"))
    assert agent.calls == ["duck", "pause", "cancel", ("respond", "等一下,改成明天")]


def test_stop_cancels_without_responding():
    agent = FakeAgent(busy=True)
    run(agent, FakeJudge(Decision.STOP), interjection("别说了"))
    assert agent.calls == ["duck", "pause", "cancel"]


def test_queue_resumes_and_responds_after_agent_finishes():
    agent = FakeAgent(busy=True)
    coordinator = run(agent, FakeJudge(Decision.QUEUE), interjection("你说完再帮我订个闹钟"))
    assert agent.calls == ["duck", "pause", "resume"]

    asyncio.run(coordinator.on_agent_idle())  # agent 还在说: 不动
    assert len(agent.calls) == 3
    agent.busy = False
    asyncio.run(coordinator.on_agent_idle())
    assert agent.calls[-1] == ("respond", "你说完再帮我订个闹钟")


def test_low_confidence_falls_back_to_interrupt():
    agent = FakeAgent(busy=True)
    run(agent, FakeJudge(Decision.CONTINUE, confidence=0.4), interjection("那个不太对吧"))
    assert agent.calls[-2:] == ["cancel", ("respond", "那个不太对吧")]


def test_judge_failure_falls_back_to_interrupt():
    agent = FakeAgent(busy=True)
    run(agent, FakeJudge(error=True), interjection("那个不太对吧"))
    assert agent.calls[-2:] == ["cancel", ("respond", "那个不太对吧")]


def test_agent_finished_while_user_was_speaking():
    agent, judge = FakeAgent(busy=True), FakeJudge(Decision.CONTINUE)
    events = interjection("那明天呢")
    coordinator = Coordinator(agent, judge)

    async def go():
        await coordinator.on_event(events[0])
        agent.busy = False  # 用户还在说,agent 已经说完了
        await coordinator.on_event(events[2])

    asyncio.run(go())
    assert agent.calls == ["duck", ("respond", "那明天呢")]
    assert judge.asked == []
