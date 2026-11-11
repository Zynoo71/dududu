"""用 TypeSafe 的 Jev 做插话判断: 一个四选一的 Choice,直接返回每个选项的概率。

文档: https://docs.typesafe.ai/primitives/choice.md
"""

from typesafe_sdk import AsyncTypeSafeClient, Choice

from dududu.interjection import DEFINITIONS, QUESTION, Decision, Interjection, Judgment


class JevJudge:
    def __init__(self, api_key: str, model: str = "jev-latest", timeout: float = 2.0):
        self._client = AsyncTypeSafeClient(api_key=api_key, model=model, timeout=timeout)
        self._question = Choice(
            instructions=QUESTION,
            criteria={decision.value: text for decision, text in DEFINITIONS.items()},
        )

    async def judge(self, interjection: Interjection) -> Judgment:
        response = await self._client.system_one(
            state={
                "assistant_is_saying": interjection.agent_text,
                "user_interjected": interjection.user_text,
            },
            questions={"decision": self._question},
        )
        answer = response.choices["decision"]
        # 用被选中选项的概率作为置信度,含义直接: "这个决策有多大把握"
        return Judgment(Decision(answer.choice), answer.probabilities[answer.choice])
