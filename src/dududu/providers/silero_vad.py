"""Silero VAD (v5),用 onnxruntime 直接推理,不依赖 torch。

模型文件 silero_vad.onnx 来自 https://github.com/snakers4/silero-vad (MIT)。
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort

from dududu.interfaces import SAMPLE_RATE

MODEL_PATH = Path(__file__).with_name("silero_vad.onnx")
FRAME_SAMPLES = 512  # 模型在 16kHz 下要求的固定帧长
CONTEXT_SAMPLES = 64  # 每帧前面要拼上一帧的末尾


class SileroVAD:
    frame_ms = FRAME_SAMPLES * 1000 // SAMPLE_RATE  # 32

    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(MODEL_PATH), opts)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)

    def speech_prob(self, frame: bytes) -> float:
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768
        x = np.concatenate([self._context, samples])[np.newaxis, :]
        out, self._state = self._session.run(
            None, {"input": x, "state": self._state, "sr": self._sr}
        )
        self._context = samples[-CONTEXT_SAMPLES:]
        return float(out[0, 0])
