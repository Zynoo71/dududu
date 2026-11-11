# dududu

Voice agent 的语音输入端:把麦克风音频变成"用户说了什么"的事件流,和下游的 agent 完全解耦。

```
音频 → Silero VAD → Smart Turn(判断是否说完) → 流式 ASR → 事件流
```

## 目录

| 路径 | 内容 |
|---|---|
| `src/dududu/` | 可移植的包。迁移到别的项目时只需要它 |
| `src/dududu/interfaces.py` | 四个接口:`AudioSource` / `VAD` / `TurnDetector` / `ASR` |
| `src/dududu/providers/` | 接口的具体实现,换引擎时在这里加文件 |
| `src/dududu/coordinator.py` | 可选的协调层:agent 说话时用户插话,判断该继续、排队、打断还是停下 |
| `src/dududu/interjection.py` | 插话判断的接口 `InterjectionJudge`,实现在 `providers/`(Jev、LLM) |
| `demo/` | 测试用的对话 demo(LLM + TTS + 网页),也是"下游"的参考实现 |
| `scripts/replay.py` | 把 wav 文件回放给整条链路,打印事件流 |

## 用法

```python
from dududu.events import CommittedFinal
from dududu.providers.push_source import PushSource
from dududu.providers.silero_vad import SileroVAD
from dududu.providers.smart_turn import SmartTurn
from dududu.providers.volc_asr import VolcASR
from dududu.voice_input import VoiceInput

source = PushSource()  # 从任意传输层 source.push(pcm),格式为 16kHz 单声道 16-bit PCM
voice = VoiceInput(
    source=source,
    vad=SileroVAD(),
    turn_detector=SmartTurn(),
    new_asr=lambda: VolcASR(api_key=KEY),
)
async for event in voice.events():
    if isinstance(event, CommittedFinal):
        agent.send_user_message(event.text)  # 替代原来的文本输入
```

事件的含义见 `src/dududu/events.py`。`VoiceInput` 只陈述用户说了什么,不判断是否打断。

需要处理插话时,在它上面加一层 `Coordinator`:宿主实现 `AgentControl`(压低音量、暂停、恢复、取消、回复),`Coordinator` 根据判断器的结果调用它们。参考 `demo/server.py` 的 `Session`。

## 运行

把 `.env.example` 复制为 `.env`,填入方舟 Agent Plan 的专属 API Key。

```bash
uv run --all-extras pytest
```

```bash
uv run --all-extras --group demo uvicorn demo.server:app --port 8000
```

然后用 Chrome 打开 http://localhost:8000 。

```bash
uv run --all-extras python scripts/replay.py tests/data/hello.wav
```

```bash
uv run --all-extras python scripts/eval_judge.py llm
```

用 `tests/data/interjections.json` 里带标注的例句评估插话判断器的准确率和延迟,`llm` 可换成 `jev`。
