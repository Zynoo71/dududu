from dududu.events import (
    CommittedFinal,
    InterimTranscript,
    SpeculativeFinal,
    SpeechStarted,
    TurnDropped,
    TurnRetracted,
)
from dududu.interfaces import ASRResult
from dududu.machine import TurnStateMachine

FRAME = b"\x00" * 1024  # 32ms


def feed(machine, ms, speech):
    """喂 ms 毫秒的人声或静音,返回期间发出的事件。"""
    events = []
    for _ in range(ms // 32):
        events += machine.on_frame(FRAME, 0.9 if speech else 0.1)
    return events


def kinds(events):
    return [type(e) for e in events]


def test_normal_turn():
    m = TurnStateMachine()
    assert kinds(feed(m, 1000, speech=True)) == [SpeechStarted]
    assert kinds(m.on_asr(ASRResult("你好", False))) == [InterimTranscript]

    assert feed(m, 320, speech=False) == []
    assert m.wants_turn_check
    (spec,) = m.on_turn_check(0.9)
    assert isinstance(spec, SpeculativeFinal) and spec.text == "你好"

    # ASR 定稿先于 600ms 静音到达: 文本变了,更新投机,但还不提交
    assert kinds(m.on_asr(ASRResult("你好。", True))) == [InterimTranscript, SpeculativeFinal]
    (final,) = feed(m, 320, speech=False)
    assert isinstance(final, CommittedFinal) and final.text == "你好。"
    assert final.turn_id == 1


def test_commit_waits_for_asr_final():
    m = TurnStateMachine()
    feed(m, 1000, speech=True)
    m.on_asr(ASRResult("你好", False))
    feed(m, 320, speech=False)
    m.on_turn_check(0.9)
    assert feed(m, 640, speech=False) == []  # 静音够了,但 ASR 还没定稿
    assert kinds(m.on_asr(ASRResult("你好", True))) == [CommittedFinal]


def test_user_continues_after_speculative_final():
    m = TurnStateMachine()
    feed(m, 500, speech=True)
    m.on_asr(ASRResult("你好", False))
    feed(m, 320, speech=False)
    m.on_turn_check(0.9)

    assert kinds(feed(m, 1000, speech=True)) == [TurnRetracted]
    m.on_asr(ASRResult("你好", True))
    m.on_asr(ASRResult("我想查订单", False))
    feed(m, 320, speech=False)
    m.on_turn_check(0.9)
    m.on_asr(ASRResult("我想查订单", True))
    (final,) = feed(m, 320, speech=False)
    assert isinstance(final, CommittedFinal) and final.text == "你好 我想查订单"
    assert final.turn_id == 1  # 仍是同一个 turn


def test_turn_check_result_is_ignored_if_user_resumed():
    m = TurnStateMachine()
    feed(m, 500, speech=True)
    m.on_asr(ASRResult("你好", False))
    feed(m, 320, speech=False)
    feed(m, 200, speech=True)  # 判断还没回来,用户又开口了
    assert m.on_turn_check(0.9) == []


def test_noise_without_text_is_dropped():
    m = TurnStateMachine()
    feed(m, 300, speech=True)
    events = feed(m, 1600, speech=False)
    assert kinds(events) == [TurnDropped]


def test_forced_end_when_turn_never_judged_complete():
    m = TurnStateMachine()
    feed(m, 1000, speech=True)
    m.on_asr(ASRResult("我想查一下我的", False))
    events = []
    for _ in range(100):  # 3.2s 静音,每次判断都说"没说完"
        events += m.on_frame(FRAME, 0.1)
        if m.wants_turn_check:
            events += m.on_turn_check(0.1)
    (final,) = events
    assert isinstance(final, CommittedFinal) and final.text == "我想查一下我的"


def test_asr_results_outside_a_turn_are_ignored():
    m = TurnStateMachine()
    assert m.on_asr(ASRResult("迟到的结果", True)) == []
