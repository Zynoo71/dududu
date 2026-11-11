"""插话判断: agent 正在说话时用户说了一句,agent 该怎么办?

这里只有类型、接口和不需要模型的规则。具体的判断器实现在 providers/ 下
(jev_judge.py、llm_judge.py),换判断引擎时加一个文件即可。
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class Decision(Enum):
    CONTINUE = "continue"  # 继续说,这句插话不需要处理
    QUEUE = "queue"  # 继续说,说完之后再处理这句
    INTERRUPT = "interrupt"  # 停下当前的话,转而回应这句
    STOP = "stop"  # 停下,不需要回应


# 给判断器看的定义。LLM 和 Jev 共用同一份,保证两者判断的是同一件事。
QUESTION = "语音助手正在说话时,用户插了一句话。助手接下来应该怎么做?"
DEFINITIONS = {
    Decision.CONTINUE: "用户只是在附和、确认、表示在听,或者说的内容和助手正在讲的一致。助手不需要停下,也不需要改变内容。",
    Decision.QUEUE: "用户明确要求助手先把当前的话说完,之后再去做另一件事。例如'你说完以后再帮我订个闹钟'。",
    Decision.INTERRUPT: "用户提出了新的问题或要求、纠正或反对助手、或者回答了助手的提问。助手需要立刻停下,转而回应这句话。",
    Decision.STOP: "用户只是要求助手停下或安静,没有提出别的请求。例如'好了好了'、'别说了'。",
}


@dataclass(frozen=True)
class Interjection:
    agent_text: str  # 助手这一轮正在说的内容
    user_text: str  # 用户插的那句话


@dataclass(frozen=True)
class Judgment:
    decision: Decision
    confidence: float  # 0~1。实现给不出置信度时填 1.0


class InterjectionJudge(Protocol):
    async def judge(self, interjection: Interjection) -> Judgment: ...


# ---- 不需要模型的规则 ----

_FILLER = r"嗯|恩|唔|哦|噢|喔|啊|呃|额|诶|欸|mm+|hm+|uh|um|oh|ah"
_ACK = r"好的?|好吧|行|可以|对的?|是的?|没错|明白了?|了解|知道了|懂了|ok|okay|yes|yeah|yep|right|sure"


def _normalize(text: str) -> str:
    return re.sub(r"[\W_]", "", text).lower()


def is_filler(text: str) -> bool:
    """只有语气词("嗯嗯"、"哦")或空文本。可以直接当作 CONTINUE,不用问模型。"""
    return re.fullmatch(f"({_FILLER})*", _normalize(text)) is not None


def is_short_ack(text: str) -> bool:
    """只有语气词和简短应答("好的"、"对对对")。

    它多半是附和,但也可能是在回答助手的提问,所以仍要交给判断器;
    只是在判断出来之前不必暂停播放。
    """
    return re.fullmatch(f"({_FILLER}|{_ACK})*", _normalize(text)) is not None
