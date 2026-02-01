FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY whisperx_diarize /app/whisperx_diarize

RUN pip install --no-cache-dir -e .

# Example:
# docker run --rm -e HF_TOKEN=... -v $PWD:/data echox whisperx_diarize /data/input.wav --diarize

ENTRYPOINT ["whisperx_diarize"]