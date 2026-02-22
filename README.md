# song2songMT

Translates songs across languages while preserving musical structure. Separates vocal tracks from instrumentals using Demucs, runs the lyrics through neural MT, then synthesizes translated vocals with a TTS model and remixes with the original instrumental.

## Setup

```bash
uv venv && source .venv/bin/activate
uv sync
```

Requires `ffmpeg` installed on the system.

## Usage

```bash
bash run_song.sh <input_audio> <target_language>
# or
python main.py --input songs/<song>.mp3 --tgt_lang es
```

## Pipeline

1. **Separation** — Demucs splits audio into vocals + instrumental
2. **Transcription** — ASR extracts lyrics from vocals
3. **Translation** — Neural MT translates lyrics to target language
4. **Synthesis** — TTS generates translated vocals
5. **Remix** — Translated vocals mixed back with instrumental
