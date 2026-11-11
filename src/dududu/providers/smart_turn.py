"""Smart Turn v3.2 (CPU 版),根据音频判断用户是否说完。

模型文件来自 https://huggingface.co/pipecat-ai/smart-turn-v3 (BSD-2-Clause)。
预处理与官方 inference.py 一致: 取最后 8 秒,不足则在前面补零。
"""

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

from dududu.interfaces import SAMPLE_RATE

MODEL_PATH = Path(__file__).with_name("smart_turn_v3.2_cpu.onnx")
WINDOW_SAMPLES = 8 * SAMPLE_RATE


class SmartTurn:
    def __init__(self):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(str(MODEL_PATH), opts)
        self._extractor = WhisperFeatureExtractor(chunk_length=8)
        self.end_of_turn_prob(b"")  # 预热,首次推理要多花约 200ms

    def end_of_turn_prob(self, turn_audio: bytes) -> float:
        samples = np.frombuffer(turn_audio, dtype=np.int16).astype(np.float32) / 32768
        samples = samples[-WINDOW_SAMPLES:]
        samples = np.pad(samples, (WINDOW_SAMPLES - len(samples), 0))
        features = self._extractor(
            samples,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        logits = self._session.run(None, {"input_features": features})[0]
        return float(logits[0, 0])  # 模型输出已过 sigmoid
