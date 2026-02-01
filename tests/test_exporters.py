from pathlib import Path

from whisperx_diarize.exporters import export_json, export_srt, export_txt, export_vtt
from whisperx_diarize.types import Segment, Transcript, Word


def _sample_transcript():
    segments = [
        Segment(
            id=0,
            start=0.0,
            end=1.2,
            text="Hola mundo",
            speaker="SPEAKER_00",
            words=[Word(start=0.0, end=0.5, text="Hola"), Word(start=0.6, end=1.2, text="mundo")],
        ),
        Segment(
            id=1,
            start=1.2,
            end=2.0,
            text="Adios",
            speaker="SPEAKER_01",
            words=[],
        ),
    ]
    return Transcript(language="es", segments=segments, metadata={"model": "dummy"})


def test_exporters(tmp_path: Path):
    transcript = _sample_transcript()

    txt_path = tmp_path / "transcript.txt"
    srt_path = tmp_path / "transcript.srt"
    vtt_path = tmp_path / "transcript.vtt"
    json_path = tmp_path / "transcript.json"

    export_txt(transcript.segments, txt_path)
    export_srt(transcript.segments, srt_path)
    export_vtt(transcript.segments, vtt_path)
    export_json(transcript, json_path)

    assert txt_path.exists()
    assert srt_path.exists()
    assert vtt_path.exists()
    assert json_path.exists()

    txt = txt_path.read_text(encoding="utf-8")
    assert "SPEAKER_00" in txt

    srt = srt_path.read_text(encoding="utf-8")
    assert "SPEAKER_01" in srt

    vtt = vtt_path.read_text(encoding="utf-8")
    assert "WEBVTT" in vtt

    json_text = json_path.read_text(encoding="utf-8")
    assert "\"segments\"" in json_text