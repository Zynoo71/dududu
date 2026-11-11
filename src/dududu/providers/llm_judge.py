"""用任意 OpenAI 兼容的 LLM 做插话判断。给不出概率,置信度固定为 1.0。"""

import json

from openai import AsyncOpenAI

from dududu.interjection import DEFINITIONS, QUESTION, Decision, Interjection, Judgment

_PROMPT = (
    QUESTION
    + "\n只输出下面四个标签之一,不要输出任何别的内容:\n"
    + "\n".join(f"- {decision.value}: {text}" for decision, text in DEFINITIONS.items())
)


class LLMJudge:
    def __init__(self, api_key: str, base_url: str, model: str, extra_body: dict | None = None):
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._extra_body = extra_body

    async def judge(self, interjection: Interjection) -> Judgment:
        state = {
            "assistant_is_saying": interjection.agent_text,
            "user_interjected": interjection.user_text,
        }
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False)},
            ],
            max_tokens=8,
            temperature=0,
            extra_body=self._extra_body,
        )
        answer = (response.choices[0].message.content or "").strip().lower()
        for decision in Decision:
            if decision.value in answer:
                return Judgment(decision, 1.0)
        return Judgment(Decision.INTERRUPT, 0.0)  # 输出不合规: 让上层按打断处理
