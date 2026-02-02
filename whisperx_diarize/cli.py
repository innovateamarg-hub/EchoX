"""CLI for EchoX."""

import inspect
import logging
import os
from pathlib import Path
from typing import Optional

import typer
import click
from typer.main import TyperArgument

from .utils import ensure_wav_16k_mono, export_speaker_wavs as export_speaker_wavs_fn, ffmpeg_available
from .exporters import export_json, export_md, export_rttm, export_srt, export_txt, export_vtt
from .pipeline import PipelineConfig, run_pipeline

def _patch_typer_click_compat() -> None:
    signature = inspect.signature(TyperArgument.make_metavar)
    if "ctx" in signature.parameters:
        return
    original = TyperArgument.make_metavar

    def _make_metavar(self, ctx=None):  # type: ignore[override]
        return original(self)

    TyperArgument.make_metavar = _make_metavar  # type: ignore[assignment]


_patch_typer_click_compat()


def _patch_click_metavar_compat() -> None:
    original = click.ParamType.get_metavar

    def _get_metavar(self, param, ctx=None):  # type: ignore[override]
        try:
            return original(self, param, ctx)
        except TypeError:
            return original(self, param)

    click.ParamType.get_metavar = _get_metavar  # type: ignore[assignment]

    param_original = click.core.Parameter.make_metavar

    def _param_make_metavar(self, ctx=None):  # type: ignore[override]
        try:
            return param_original(self, ctx)
        except TypeError:
            return param_original(self)

    click.core.Parameter.make_metavar = _param_make_metavar  # type: ignore[assignment]


_patch_click_metavar_compat()


def _fix_click_option_flags(command: click.Command) -> None:
    for param in command.params:
        if isinstance(param, click.Option) and param.is_flag and not param.is_bool_flag:
            param.is_flag = False
            param.flag_value = None
            if hasattr(param, "_flag_needs_value"):
                param._flag_needs_value = False
    if isinstance(command, click.Group):
        for sub in command.commands.values():
            _fix_click_option_flags(sub)


def _patch_typer_command_builder() -> None:
    original_get_command = typer.main.get_command

    def _get_command(app: typer.Typer) -> click.Command:
        cmd = original_get_command(app)
        _fix_click_option_flags(cmd)
        return cmd

    typer.main.get_command = _get_command  # type: ignore[assignment]


_patch_typer_command_builder()

app = typer.Typer(add_completion=False, help="EchoX: WhisperX + pyannote diarization")

LOGGER = logging.getLogger(__name__)

ALLOWED_EXPORTS = {"txt", "srt", "vtt", "json", "rttm", "md"}
DEFAULT_EXPORTS = ["txt", "srt", "vtt", "json", "md"]
VALID_DEVICES = {"auto", "cpu", "cuda"}
VALID_COMPUTE = {"auto", "int8", "float16", "float32"}


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
    raw = raw.strip().lower()
    if raw == "all":
        return DEFAULT_EXPORTS + (["rttm"] if diarize else [])

    exports = [item.strip().lower() for item in raw.split(",") if item.strip()]
    invalid = [item for item in exports if item not in ALLOWED_EXPORTS]
    if invalid:
        raise typer.BadParameter(f"Unsupported export format(s): {', '.join(invalid)}")
    if not diarize and "rttm" in exports:
        raise typer.BadParameter("rttm export requires --diarize")
    return exports


def _prompt_optional_int(label: str, default: Optional[int] = None) -> Optional[int]:
    raw = typer.prompt(label, default="" if default is None else str(default))
    value = str(raw).strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise typer.BadParameter(f"{label} debe ser un numero entero.") from exc


@app.command()
def main(
    input_audio: Optional[Path] = typer.Argument(None, help="Audio/video file path"),
    output_dir: Path = typer.Option(
        Path("./out"), "--output_dir", help="Output directory", is_flag=False
    ),
    language: str = typer.Option(
        "auto", "--language", help="Language code or auto", is_flag=False
    ),
    model: str = typer.Option(
        "medium", "--model", help="WhisperX model name", is_flag=False
    ),
    device: str = typer.Option(
        "auto", "--device", help="auto/cpu/cuda", is_flag=False
    ),
    compute_type: str = typer.Option(
        "auto", "--compute_type", help="auto/int8/float16/float32", is_flag=False
    ),
    diarize: bool = typer.Option(
        False, "--diarize", help="Enable diarization", is_flag=True, flag_value=True
    ),
    hf_token: Optional[str] = typer.Option(
        None, "--hf_token", help="Hugging Face token", is_flag=False
    ),
    num_speakers: Optional[int] = typer.Option(
        None, "--num_speakers", help="Fixed number of speakers", is_flag=False
    ),
    min_speakers: Optional[int] = typer.Option(
        None, "--min_speakers", help="Min speakers", is_flag=False
    ),
    max_speakers: Optional[int] = typer.Option(
        None, "--max_speakers", help="Max speakers", is_flag=False
    ),
    export: str = typer.Option(
        "all", "--export", help="Comma separated list: txt,srt,vtt,json,md,rttm", is_flag=False
    ),
    export_speaker_wavs_flag: bool = typer.Option(
        False,
        "--export_speaker_wavs",
        help="Write per-speaker WAVs",
        is_flag=True,
        flag_value=True,
    ),
    format_audio: bool = typer.Option(
        False,
        "--format_audio",
        help="Convert to 16kHz mono WAV",
        is_flag=True,
        flag_value=True,
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Verbose logging", is_flag=True, flag_value=True
    ),
    wizard: bool = typer.Option(
        False, "--wizard", help="Interactive setup", is_flag=True, flag_value=True
    ),
) -> None:
    if wizard:
        input_prompt = typer.prompt("Ruta del audio")
        input_audio = Path(input_prompt)
        output_dir = Path(typer.prompt("Directorio de salida", default=str(output_dir)))
        language = typer.prompt("Idioma (auto/es)", default=language)
        model = typer.prompt("Modelo WhisperX", default=model)
        device = typer.prompt("Device (auto/cpu/cuda)", default=device)
        compute_type = typer.prompt("Compute type (auto/int8/float16/float32)", default=compute_type)
        diarize = typer.confirm("Habilitar diarizacion?", default=diarize)
        if diarize:
            token_input = typer.prompt("HF token (enter para usar HF_TOKEN)", default="")
            hf_token = token_input.strip() or hf_token
            num_speakers = _prompt_optional_int("Numero fijo de speakers (opcional)")
            if num_speakers is None:
                min_speakers = _prompt_optional_int("Min speakers (opcional)", min_speakers)
                max_speakers = _prompt_optional_int("Max speakers (opcional)", max_speakers)
        export = typer.prompt(
            "Formatos export (txt,srt,vtt,json,md,rttm o all)", default=export
        )
        export_speaker_wavs_flag = typer.confirm(
            "Exportar WAV por hablante?", default=export_speaker_wavs_flag
        )
        format_audio = typer.confirm("Normalizar audio a WAV 16kHz?", default=format_audio)
        verbose = typer.confirm("Verbose?", default=verbose)

    _setup_logging(verbose)

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_audio is None:
        raise typer.BadParameter("input_audio es obligatorio (o usar --wizard).")
    if not input_audio.exists() or not input_audio.is_file():
        raise typer.BadParameter(f"No existe el archivo: {input_audio}")

    if device not in VALID_DEVICES:
        raise typer.BadParameter(
            f"Invalid --device. Use one of: {', '.join(sorted(VALID_DEVICES))}"
        )

    if compute_type not in VALID_COMPUTE:
        raise typer.BadParameter(
            f"Invalid --compute_type. Use one of: {', '.join(sorted(VALID_COMPUTE))}"
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

    if export_speaker_wavs_flag and not diarize:
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

    exporters = {
        "txt": lambda: export_txt(transcript.segments, output_dir / "transcript.txt"),
        "srt": lambda: export_srt(transcript.segments, output_dir / "transcript.srt"),
        "vtt": lambda: export_vtt(transcript.segments, output_dir / "transcript.vtt"),
        "json": lambda: export_json(transcript, output_dir / "transcript.json"),
        "md": lambda: export_md(transcript, output_dir / "transcript.md", source=input_audio),
    }
    for fmt in exports:
        if fmt == "rttm":
            if result.rttm:
                export_rttm(result.rttm, output_dir / "diarization.rttm")
            continue
        exporters[fmt]()

    if export_speaker_wavs_flag:
        cache_dir = output_dir / "cache"
        wav_source = result.formatted_audio
        if wav_source is None:
            if not ffmpeg_available():
                raise typer.BadParameter(
                    "ffmpeg not found. Install ffmpeg or enable --format_audio."
                )
            wav_source = ensure_wav_16k_mono(result.source_audio, cache_dir)
        speakers_dir = output_dir / "speakers"
        export_speaker_wavs_fn(transcript.segments, wav_source, speakers_dir)

    LOGGER.info("Outputs written to: %s", output_dir)


if __name__ == "__main__":
    app()
