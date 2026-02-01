"""CLI for EchoX."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import typer

from .audio_utils import ensure_wav_16k_mono, export_speaker_wavs, ffmpeg_available
from .exporters import export_json, export_rttm, export_srt, export_txt, export_vtt
from .pipeline import PipelineConfig, run_pipeline

app = typer.Typer(add_completion=False, help="EchoX: WhisperX + pyannote diarization")

LOGGER = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    try:
        from rich.logging import RichHandler

        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    except Exception:
        logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _parse_export_list(raw: str, diarize: bool) -> list[str]:
    allowed = {"txt", "srt", "vtt", "json", "rttm"}
    if raw.lower() == "all":
        exports = ["txt", "srt", "vtt", "json"]
        if diarize:
            exports.append("rttm")
        return exports

    exports = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in exports if item not in allowed]
    if invalid:
        raise typer.BadParameter(f"Unsupported export format(s): {', '.join(invalid)}")
    if not diarize and "rttm" in exports:
        raise typer.BadParameter("rttm export requires --diarize")
    return exports


@app.command()
def main(
    input_audio: Path = typer.Argument(..., exists=True, readable=True, help="Audio/video file path"),
    output_dir: Path = typer.Option(Path("./out"), "--output_dir", help="Output directory"),
    language: str = typer.Option("auto", "--language", help="Language code or auto"),
    model: str = typer.Option("medium", "--model", help="WhisperX model name"),
    device: str = typer.Option("auto", "--device", help="auto/cpu/cuda"),
    compute_type: str = typer.Option("auto", "--compute_type", help="auto/int8/float16/float32"),
    diarize: bool = typer.Option(False, "--diarize", help="Enable diarization"),
    hf_token: Optional[str] = typer.Option(None, "--hf_token", help="Hugging Face token"),
    num_speakers: Optional[int] = typer.Option(None, "--num_speakers", help="Fixed number of speakers"),
    min_speakers: Optional[int] = typer.Option(None, "--min_speakers", help="Min speakers"),
    max_speakers: Optional[int] = typer.Option(None, "--max_speakers", help="Max speakers"),
    export: str = typer.Option("all", "--export", help="Comma separated list: txt,srt,vtt,json,rttm"),
    export_speaker_wavs: bool = typer.Option(False, "--export_speaker_wavs", help="Write per-speaker WAVs"),
    format_audio: bool = typer.Option(False, "--format_audio", help="Convert to 16kHz mono WAV"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose logging"),
) -> None:
    _setup_logging(verbose)

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_devices = {"auto", "cpu", "cuda"}
    if device not in valid_devices:
        raise typer.BadParameter(
            f"Invalid --device. Use one of: {', '.join(sorted(valid_devices))}"
        )

    valid_compute = {"auto", "int8", "float16", "float32"}
    if compute_type not in valid_compute:
        raise typer.BadParameter(
            f"Invalid --compute_type. Use one of: {', '.join(sorted(valid_compute))}"
        )

    if min_speakers is not None and max_speakers is not None:
        if min_speakers > max_speakers:
            raise typer.BadParameter("--min_speakers cannot be greater than --max_speakers")

    if num_speakers is not None and (min_speakers is not None or max_speakers is not None):
        LOGGER.info("--num_speakers provided; ignoring --min_speakers/--max_speakers")

    if diarize:
        token = hf_token or os.environ.get("HF_TOKEN")
        if not token:
            raise typer.BadParameter(
                "--diarize requires a Hugging Face token. Set HF_TOKEN or pass --hf_token."
            )
        hf_token = token

    if export_speaker_wavs and not diarize:
        raise typer.BadParameter("--export_speaker_wavs requires --diarize")

    if format_audio and not ffmpeg_available():
        raise typer.BadParameter(
            "ffmpeg not found. Install ffmpeg or remove --format_audio."
        )

    exports = _parse_export_list(export, diarize)

    config = PipelineConfig(
        input_path=input_audio,
        output_dir=output_dir,
        language=language,
        model=model,
        device=device,
        compute_type=compute_type,
        diarize=diarize,
        hf_token=hf_token,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        format_audio=format_audio,
    )

    result = run_pipeline(config)

    transcript = result.transcript

    if "txt" in exports:
        path = output_dir / "transcript.txt"
        export_txt(transcript.segments, path)

    if "srt" in exports:
        path = output_dir / "transcript.srt"
        export_srt(transcript.segments, path)

    if "vtt" in exports:
        path = output_dir / "transcript.vtt"
        export_vtt(transcript.segments, path)

    if "json" in exports:
        path = output_dir / "transcript.json"
        export_json(transcript, path)

    if "rttm" in exports and result.rttm:
        path = output_dir / "diarization.rttm"
        export_rttm(result.rttm, path)

    if export_speaker_wavs:
        cache_dir = output_dir / "cache"
        wav_source = result.formatted_audio
        if wav_source is None:
            if not ffmpeg_available():
                raise typer.BadParameter(
                    "ffmpeg not found. Install ffmpeg or enable --format_audio."
                )
        wav_source = ensure_wav_16k_mono(result.source_audio, cache_dir)
        speakers_dir = output_dir / "speakers"
        export_speaker_wavs(transcript.segments, wav_source, speakers_dir)

    LOGGER.info("Outputs written to: %s", output_dir)


if __name__ == "__main__":
    app()
