"""用带标注的例句集评估插话判断器的准确率和延迟。

    uv run --all-extras python scripts/eval_judge.py llm
    uv run --all-extras python scripts/eval_judge.py jev

llm 用 .env 里的 LLM_* 配置(不填则用 Agent Plan 的豆包);jev 需要 TYPESAFE_API_KEY。
"""

import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

from dududu.interjection import Decision, Interjection

for line in Path(".env").read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        os.environ.setdefault(*line.split("=", 1))


def make_judge(name: str):
    if name == "jev":
        from dududu.providers.jev_judge import JevJudge

        return JevJudge(api_key=os.environ["TYPESAFE_API_KEY"])
    from dududu.providers.llm_judge import LLMJudge

    base_url = os.environ.get("LLM_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3")
    return LLMJudge(
        api_key=os.environ.get("LLM_API_KEY", os.environ["VOLC_API_KEY"]),
        base_url=base_url,
        model=os.environ.get("LLM_MODEL", "doubao-seed-2.0-mini"),
        extra_body={"thinking": {"type": "disabled"}} if "volces.com" in base_url else None,
    )


async def main(name: str) -> None:
    judge = make_judge(name)
    cases = json.loads(Path("tests/data/interjections.json").read_text())
    latencies, wrong = [], []
    per_label = {d.value: [0, 0] for d in Decision}  # 标签 -> [正确数, 总数]
    for case in cases:
        start = time.perf_counter()
        judgment = await judge.judge(Interjection(case["agent"], case["user"]))
        latencies.append((time.perf_counter() - start) * 1000)
        per_label[case["label"]][1] += 1
        if judgment.decision.value == case["label"]:
            per_label[case["label"]][0] += 1
        else:
            wrong.append((case, judgment))

    total = len(cases)
    print(f"判断器: {name}   准确率: {total - len(wrong)}/{total}")
    for label, (ok, n) in per_label.items():
        print(f"  {label:10s} {ok}/{n}")
    latencies.sort()
    p90 = latencies[int(len(latencies) * 0.9)]
    print(f"延迟: 中位数 {statistics.median(latencies):.0f} ms, p90 {p90:.0f} ms, 最大 {latencies[-1]:.0f} ms")
    for case, judgment in wrong:
        print(
            f"  ✗ 应为 {case['label']:9s} 判成 {judgment.decision.value:9s}"
            f" (置信度 {judgment.confidence:.2f})  用户: {case['user']}"
        )


asyncio.run(main(sys.argv[1]))
