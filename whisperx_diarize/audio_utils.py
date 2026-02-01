"""Audio helpers for EchoX."""

from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Iterable

import numpy as np

from .types import Segment

LOGGER = logging.getLogger(__name__)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def format_audio_to_wav(input_path: Path, output_path: Path) -> None:
    """Convert input audio to 16kHz mono WAV (s16le) using ffmpeg."""
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg or disable --format_audio."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_path),
    ]
    LOGGER.debug("Running ffmpeg: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to convert audio. stderr: " + result.stderr.strip()
        )


def ensure_wav_16k_mono(input_path: Path, cache_dir: Path) -> Path:
    """Ensure audio is WAV 16kHz mono. Convert if needed."""
    input_path = input_path.resolve()
    if input_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(input_path), "rb") as wav:
                channels = wav.getnchannels()
                sample_rate = wav.getframerate()
                sample_width = wav.getsampwidth()
            if channels == 1 and sample_rate == 16000 and sample_width == 2:
                return input_path
        except wave.Error:
            pass

    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{input_path.stem}_16k_mono.wav"
    format_audio_to_wav(input_path, output_path)
    return output_path


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """Load WAV audio as int16 mono array."""
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_rate = wav.getframerate()
        sample_width = wav.getsampwidth()
        if sample_width != 2:
            raise RuntimeError("Only 16-bit WAV is supported for speaker export.")
        frames = wav.readframes(wav.getnframes())

    data = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return data, sample_rate


def write_wav_mono(path: Path, data: np.ndarray, sample_rate: int) -> None:
    """Write int16 mono WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(data, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(clipped.tobytes())


def sanitize_speaker_label(label: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    return safe.strip("_") or "unknown"


def export_speaker_wavs(
    segments: Iterable[Segment],
    source_wav: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Export aligned per-speaker WAVs with silences.

    The output tracks have the same total duration as the source audio.
    """
    data, sample_rate = load_wav_mono(source_wav)
    total_samples = len(data)

    speakers: dict[str, np.ndarray] = {}
    for segment in segments:
        label = segment.speaker or "unknown"
        if label not in speakers:
            speakers[label] = np.zeros(total_samples, dtype=np.int32)

        start_idx = max(int(segment.start * sample_rate), 0)
        end_idx = min(int(segment.end * sample_rate), total_samples)
        if end_idx <= start_idx:
            continue
        speakers[label][start_idx:end_idx] += data[start_idx:end_idx].astype(np.int32)

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    for label, buffer in speakers.items():
        safe_label = sanitize_speaker_label(label)
        output_path = output_dir / f"speaker_{safe_label}.wav"
        write_wav_mono(output_path, buffer, sample_rate)
        outputs[label] = output_path

    return outputs