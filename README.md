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

事件的含义见 `src/dududu/events.py`。打断判定不在包内,由下游结合自己的播放状态决定,参考 `demo/server.py` 的 `Session.on_event`。

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
