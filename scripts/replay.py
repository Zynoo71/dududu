"""把一个 wav 文件按真实语速回放给整条链路(真实火山 ASR),打印事件流。

    uv run --all-extras python scripts/replay.py tests/data/hello.wav

wav 必须是 16kHz 单声道 16-bit。需要 .env 里有 VOLC_API_KEY。
"""

import asyncio
import sys
import time
from pathlib import Path

from dududu.providers.silero_vad import SileroVAD
from dududu.providers.smart_turn import SmartTurn
from dududu.providers.volc_asr import VolcASR
from dududu.providers.wav_source import WavSource
from dududu.voice_input import VoiceInput


def load_env() -> dict[str, str]:
    lines = Path(".env").read_text().splitlines()
    return dict(l.split("=", 1) for l in lines if "=" in l and not l.startswith("#"))


async def main(path: str) -> None:
    key = load_env()["VOLC_API_KEY"]
    voice = VoiceInput(
        source=WavSource(path, realtime=True),
        vad=SileroVAD(),
        turn_detector=SmartTurn(),
        new_asr=lambda: VolcASR(api_key=key, end_window_ms=400),
    )
    start = time.perf_counter()
    async for event in voice.events():
        text = getattr(event, "text", "")
        elapsed = time.perf_counter() - start
        print(f"[{elapsed:5.2f}s] turn={event.turn_id} {type(event).__name__:17s} {text}")


asyncio.run(main(sys.argv[1]))
