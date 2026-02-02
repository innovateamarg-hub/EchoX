# EchoX

EchoX transcribe audio en español con WhisperX y agrega diarizacion por hablante con pyannote. El objetivo es:
- transcripcion con timestamps
- etiquetas de hablante
- exportacion a varios formatos
- opcion para generar un WAV por hablante con silencios (pistas alineadas)

> Nota: con un solo microfono, la diarizacion no separa voces perfectamente. Hay solapamientos y errores normales.

## Requisitos
- Python 3.9+
- ffmpeg (recomendado para --format_audio y export de WAV por hablante)
- GPU opcional (CUDA) para acelerar

## Estructura del proyecto
```
whisperx_diarize/
  cli.py
  types.py
  pipeline/
    __init__.py
    runner.py
  exporters/
    __init__.py
  utils/
    audio.py
    __init__.py
tests/
pyproject.toml
README.md
```

## Instalacion
Usa siempre un venv (.venv) para evitar instalar dependencias globales o en la raiz.
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
python -m pip install -e .
```

Si queres correr tests:
```bash
python -m pip install -e .[test]
pytest
```

## Hugging Face token
La diarizacion requiere un token de Hugging Face y aceptar los terminos del modelo en el hub.
- Opcion 1: variable de entorno
```bash
set HF_TOKEN=tu_token
```
- Opcion 2: pasar --hf_token en el CLI

## Uso
Comando basico:
```bash
whisperx_diarize input_audio --language es --output_dir salida --model large-v2 --diarize --min_speakers 3 --max_speakers 3
```

Otros ejemplos:
```bash
# Exportar txt, srt y json
whisperx_diarize ./ejemplo.wav --language es --diarize --min_speakers 3 --max_speakers 3 --output_dir out --export txt,srt,json --export_speaker_wavs

# Modo asistido (wizard) por terminal
whisperx_diarize --wizard

# Normalizar audio a WAV mono 16kHz antes de correr
whisperx_diarize ./input.mp3 --format_audio --output_dir out
```

Tambien podes usar el alias `echox`:
```bash
echox ./ejemplo.wav --language es --export md
```

## Formatos de salida
- transcript.txt: texto con [SPEAKER_00]
- transcript.srt y transcript.vtt: subtitulos con speaker por cue
- transcript.json: metadatos + segments + words (si hay)
- transcript.md: resumen en Markdown con segmentos y speakers
- diarization.rttm: RTTM estandar (solo si --diarize)
- speakers/*.wav: un WAV por hablante con silencios (si --export_speaker_wavs)

## Ejemplo de salida (SRT)
```srt
1
00:00:00,000 --> 00:00:01,200
SPEAKER_00: Hola mundo

2
00:00:01,200 --> 00:00:02,000
SPEAKER_01: Adios
```

## Ejemplo de salida (JSON)
```json
{
  "language": "es",
  "metadata": {
    "model": "large-v2",
    "device": "cuda",
    "compute_type": "float16",
    "aligned": true,
    "diarize": true
  },
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 1.2,
      "text": "Hola mundo",
      "speaker": "SPEAKER_00",
      "words": [
        {"start": 0.0, "end": 0.5, "text": "Hola"},
        {"start": 0.6, "end": 1.2, "text": "mundo"}
      ]
    }
  ]
}
```

## Troubleshooting
- Error por token: asegurate de setear HF_TOKEN o pasar --hf_token, y aceptar terminos en Hugging Face.
- Error por ffmpeg: instala ffmpeg o evita --format_audio y --export_speaker_wavs.
- GPU no detectada: usa --device cpu y --compute_type int8 o float32.
- Versiones: WhisperX y pyannote pueden cambiar APIs; EchoX intenta adaptarse, pero revisa la seccion de compatibilidad en el codigo si algo falla.

## Compatibilidad y notas
- EchoX usa introspeccion para adaptarse a funciones disponibles en WhisperX/pyannote.
- Si la alineacion falla, continua con los timestamps base.
- La exportacion de WAV por hablante requiere audio mono 16kHz; EchoX convierte con ffmpeg cuando hace falta.
