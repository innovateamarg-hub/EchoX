import math
import wave
from pathlib import Path

import numpy as np

from whisperx_diarize.audio_utils import export_speaker_wavs
from whisperx_diarize.types import Segment


def _write_wav(path: Path, sample_rate: int = 16000, duration_s: float = 1.0):
    t = np.arange(int(sample_rate * duration_s))
    tone = (0.2 * np.sin(2 * math.pi * 440 * t / sample_rate)).astype(np.float32)
    data = (tone * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(data.tobytes())


def test_export_speaker_wavs(tmp_path: Path):
    source = tmp_path / "source.wav"
    _write_wav(source)

    segments = [
        Segment(id=0, start=0.0, end=0.5, text="hola", speaker="SPEAKER_00"),
        Segment(id=1, start=0.5, end=1.0, text="mundo", speaker="SPEAKER_01"),
    ]

    outputs = export_speaker_wavs(segments, source, tmp_path / "speakers")
    assert "SPEAKER_00" in outputs
    assert "SPEAKER_01" in outputs

    for path in outputs.values():
        assert path.exists()
        with wave.open(str(path), "rb") as wav:
            assert wav.getnframes() == 16000
            assert wav.getnchannels() == 1