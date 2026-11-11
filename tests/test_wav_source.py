import asyncio
import wave

import pytest

from dududu.providers.wav_source import WavSource


def write_wav(path, rate, seconds=1):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(rate)
        f.writeframes(b"\x00\x00" * rate * seconds)


async def collect(source):
    return [chunk async for chunk in source.frames()]


def test_yields_all_audio_in_chunks(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, 16000)
    chunks = asyncio.run(collect(WavSource(str(path), chunk_ms=20)))
    assert len(chunks) == 50
    assert all(len(c) == 640 for c in chunks)


def test_rejects_wrong_sample_rate(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, 48000)
    with pytest.raises(ValueError):
        asyncio.run(collect(WavSource(str(path))))
