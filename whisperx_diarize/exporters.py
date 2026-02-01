"""Export utilities for EchoX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional

from .types import Segment, Transcript, Word


def _format_timestamp_srt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int(round((secs - int(secs)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    millis = int(round((secs - int(secs)) * 1000))
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d}.{millis:03d}"


def export_txt(segments: Iterable[Segment], path: Path) -> None:
    lines = []
    for segment in segments:
        speaker = segment.speaker or "UNKNOWN"
        text = segment.text.strip()
        lines.append(f"[{speaker}] {text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_srt(segments: Iterable[Segment], path: Path) -> None:
    entries = []
    for idx, segment in enumerate(segments, start=1):
        speaker = segment.speaker or "UNKNOWN"
        start = _format_timestamp_srt(segment.start)
        end = _format_timestamp_srt(segment.end)
        text = segment.text.strip()
        entries.append(
            f"{idx}\n{start} --> {end}\n{speaker}: {text}\n"
        )
    path.write_text("\n".join(entries).strip() + "\n", encoding="utf-8")


def export_vtt(segments: Iterable[Segment], path: Path) -> None:
    entries = ["WEBVTT", ""]
    for segment in segments:
        speaker = segment.speaker or "UNKNOWN"
        start = _format_timestamp_vtt(segment.start)
        end = _format_timestamp_vtt(segment.end)
        text = segment.text.strip()
        entries.append(f"{start} --> {end}\n{speaker}: {text}\n")
    path.write_text("\n".join(entries).strip() + "\n", encoding="utf-8")


def _serialize_word(word: Word) -> dict:
    payload: dict[str, object] = {
        "start": word.start,
        "end": word.end,
        "text": word.text,
    }
    if word.speaker:
        payload["speaker"] = word.speaker
    if word.score is not None:
        payload["score"] = word.score
    return payload


def _serialize_segment(segment: Segment) -> dict:
    payload: dict[str, object] = {
        "id": segment.id,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }
    if segment.speaker:
        payload["speaker"] = segment.speaker
    if segment.words:
        payload["words"] = [_serialize_word(word) for word in segment.words]
    return payload


def export_json(transcript: Transcript, path: Path) -> None:
    payload = {
        "language": transcript.language,
        "metadata": transcript.metadata,
        "segments": [_serialize_segment(seg) for seg in transcript.segments],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_rttm(rttm_text: str, path: Path) -> None:
    path.write_text(rttm_text, encoding="utf-8")