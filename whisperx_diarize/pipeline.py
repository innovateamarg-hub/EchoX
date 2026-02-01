"""WhisperX + pyannote pipeline for EchoX."""

from __future__ import annotations

import inspect
import io
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .audio_utils import ensure_wav_16k_mono, ffmpeg_available
from .types import Segment, Transcript, Word

LOGGER = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    input_path: Path
    output_dir: Path
    language: str
    model: str
    device: str
    compute_type: str
    diarize: bool
    hf_token: Optional[str]
    num_speakers: Optional[int]
    min_speakers: Optional[int]
    max_speakers: Optional[int]
    format_audio: bool


@dataclass
class PipelineResult:
    transcript: Transcript
    diarization: Any | None
    rttm: Optional[str]
    source_audio: Path
    formatted_audio: Optional[Path]
    aligned: bool


def _call_with_supported_args(func, **kwargs):
    signature = inspect.signature(func)
    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return func(**kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in signature.parameters}
    return func(**filtered)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _resolve_compute_type(device: str, compute_type: str) -> str:
    if compute_type != "auto":
        return compute_type
    if device == "cuda":
        return "float16"
    return "int8"


def _load_whisperx():
    try:
        import whisperx

        return whisperx
    except ImportError as exc:
        raise RuntimeError(
            "whisperx is not installed. Install dependencies before running."
        ) from exc


def _resolve_diarization_pipeline(whisperx):
    if hasattr(whisperx, "DiarizationPipeline"):
        return whisperx.DiarizationPipeline
    try:
        from whisperx.diarize import DiarizationPipeline

        return DiarizationPipeline
    except Exception:
        return None


def _resolve_assign_word_speakers(whisperx):
    if hasattr(whisperx, "assign_word_speakers"):
        return whisperx.assign_word_speakers
    try:
        from whisperx.diarize import assign_word_speakers

        return assign_word_speakers
    except Exception:
        pass
    try:
        from whisperx.utils import assign_word_speakers

        return assign_word_speakers
    except Exception:
        return None


def _load_audio(whisperx, path: Path):
    if hasattr(whisperx, "load_audio"):
        return whisperx.load_audio(str(path))
    try:
        import soundfile as sf

        audio, sample_rate = sf.read(str(path))
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise RuntimeError("Audio must be 16kHz when whisperx.load_audio is missing.")
        return audio
    except Exception as exc:
        raise RuntimeError(
            "Failed to load audio. Install soundfile or enable --format_audio."
        ) from exc


def _load_model(whisperx, model_name: str, device: str, compute_type: str, language: str):
    kwargs = {"device": device, "compute_type": compute_type}
    if language != "auto":
        kwargs["language"] = language
    try:
        return whisperx.load_model(model_name, **kwargs)
    except TypeError:
        kwargs.pop("language", None)
        return whisperx.load_model(model_name, **kwargs)


def _load_align_model(whisperx, language: str, device: str):
    if not hasattr(whisperx, "load_align_model"):
        return None, None
    func = whisperx.load_align_model
    signature = inspect.signature(func)
    kwargs = {"device": device}
    if "language_code" in signature.parameters:
        kwargs["language_code"] = language
    elif "language" in signature.parameters:
        kwargs["language"] = language
    return func(**kwargs)


def _align_segments(whisperx, result: dict, audio, language: str, device: str) -> tuple[dict, bool]:
    if not hasattr(whisperx, "align"):
        return result, False
    if not result.get("segments"):
        return result, False
    if language == "auto":
        language = result.get("language", "auto")
    if language == "auto":
        return result, False

    try:
        align_model, metadata = _load_align_model(whisperx, language, device)
        if align_model is None:
            return result, False
        try:
            aligned = whisperx.align(result["segments"], align_model, metadata, audio, device)
        except TypeError:
            aligned = _call_with_supported_args(
                whisperx.align,
                segments=result["segments"],
                align_model=align_model,
                metadata=metadata,
                audio=audio,
                device=device,
            )
        if isinstance(aligned, dict) and aligned.get("segments"):
            result["segments"] = aligned["segments"]
        return result, True
    except Exception as exc:
        LOGGER.warning("Alignment failed: %s", exc)
        return result, False


def _run_diarization(
    whisperx,
    audio_path: Path,
    audio,
    device: str,
    hf_token: str,
    num_speakers: Optional[int],
    min_speakers: Optional[int],
    max_speakers: Optional[int],
):
    pipeline_cls = _resolve_diarization_pipeline(whisperx)
    if pipeline_cls is None:
        raise RuntimeError("DiarizationPipeline not available in this whisperx version.")

    pipeline_kwargs = {}
    if hf_token:
        pipeline_kwargs["use_auth_token"] = hf_token
    diarize_pipeline = pipeline_cls(**pipeline_kwargs)
    if hasattr(diarize_pipeline, "to"):
        diarize_pipeline.to(device)

    call_kwargs: dict[str, object] = {}
    if num_speakers is not None:
        call_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            call_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            call_kwargs["max_speakers"] = max_speakers

    try:
        return diarize_pipeline(str(audio_path), **call_kwargs)
    except Exception:
        try:
            import torch

            waveform = torch.tensor(audio).unsqueeze(0)
            return diarize_pipeline(
                {"waveform": waveform, "sample_rate": 16000}, **call_kwargs
            )
        except Exception as exc:
            raise RuntimeError("Diarization failed.") from exc


def _assign_speakers(whisperx, diarization, result: dict) -> dict:
    assign_fn = _resolve_assign_word_speakers(whisperx)
    if assign_fn is None:
        LOGGER.warning("assign_word_speakers not available; skipping speaker assignment.")
        return result
    try:
        return assign_fn(diarization, result)
    except TypeError:
        return _call_with_supported_args(assign_fn, diarization=diarization, result=result)


def _build_segments(result: dict) -> list[Segment]:
    segments: list[Segment] = []
    for idx, seg in enumerate(result.get("segments", [])):
        words = []
        for word in seg.get("words", []) or []:
            text = word.get("word") or word.get("text") or ""
            words.append(
                Word(
                    start=float(word.get("start", 0.0)),
                    end=float(word.get("end", 0.0)),
                    text=text,
                    speaker=word.get("speaker"),
                    score=word.get("score"),
                )
            )
        segments.append(
            Segment(
                id=int(seg.get("id", idx)),
                start=float(seg.get("start", 0.0)),
                end=float(seg.get("end", 0.0)),
                text=str(seg.get("text", "")).strip(),
                speaker=seg.get("speaker"),
                words=words,
            )
        )
    return segments


def diarization_to_rttm(diarization, uri: str) -> Optional[str]:
    if diarization is None:
        return None
    if hasattr(diarization, "write_rttm"):
        buffer = io.StringIO()
        diarization.write_rttm(buffer)
        return buffer.getvalue()
    if hasattr(diarization, "itertracks"):
        lines = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = float(turn.start)
            duration = float(turn.end - turn.start)
            lines.append(
                f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>"
            )
        return "\n".join(lines) + "\n"
    return None


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    start_time = time.perf_counter()

    whisperx = _load_whisperx()

    resolved_device = _resolve_device(config.device)
    resolved_compute_type = _resolve_compute_type(resolved_device, config.compute_type)

    formatted_audio: Optional[Path] = None
    audio_path = config.input_path
    if config.format_audio:
        cache_dir = config.output_dir / "cache"
        formatted_audio = ensure_wav_16k_mono(config.input_path, cache_dir)
        audio_path = formatted_audio

    LOGGER.info("Loading audio: %s", audio_path)
    audio = _load_audio(whisperx, audio_path)

    LOGGER.info(
        "Loading WhisperX model: %s (device=%s, compute_type=%s)",
        config.model,
        resolved_device,
        resolved_compute_type,
    )
    model = _load_model(whisperx, config.model, resolved_device, resolved_compute_type, config.language)

    LOGGER.info("Transcribing...")
    transcribe_kwargs = {}
    if config.language != "auto":
        transcribe_kwargs["language"] = config.language
    result = _call_with_supported_args(model.transcribe, audio=audio, **transcribe_kwargs)

    language = result.get("language", config.language)

    LOGGER.info("Aligning words (if available)...")
    result, aligned = _align_segments(whisperx, result, audio, language, resolved_device)

    diarization = None
    rttm_text = None
    if config.diarize:
        LOGGER.info("Running diarization...")
        diarization = _run_diarization(
            whisperx,
            audio_path,
            audio,
            resolved_device,
            config.hf_token or "",
            config.num_speakers,
            config.min_speakers,
            config.max_speakers,
        )
        result = _assign_speakers(whisperx, diarization, result)
        rttm_text = diarization_to_rttm(diarization, uri=audio_path.stem)

    segments = _build_segments(result)
    transcript = Transcript(
        language=language,
        segments=segments,
        metadata={
            "model": config.model,
            "device": resolved_device,
            "compute_type": resolved_compute_type,
            "aligned": aligned,
            "diarize": config.diarize,
        },
    )

    elapsed = time.perf_counter() - start_time
    LOGGER.info("Pipeline finished in %.2f seconds", elapsed)

    return PipelineResult(
        transcript=transcript,
        diarization=diarization,
        rttm=rttm_text,
        source_audio=config.input_path,
        formatted_audio=formatted_audio,
        aligned=aligned,
    )