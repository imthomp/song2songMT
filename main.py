import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

from transformers import pipeline
from demucs.separate import main as demucs_main
import ffmpeg

###############################################
# 1. VOCAL SEPARATION (DEMUCS)
###############################################
def separate_vocals(input_audio, output_dir):
    """
    Runs Demucs to separate vocals from accompaniment.
    Produces:
        output_dir/vocals.wav
        output_dir/accompaniment.wav
    """
    logging.info("[1/5] Separating vocals with Demucs")
    print(f"  Input: {input_audio}")

    args = [
        "-n", "mdx_extra_q",
        "--two-stems",
        "vocals",
        "-o",
        str(output_dir),
        str(input_audio),
    ]

    print("  Running Demucs...")
    demucs_main(args)
    print("  Demucs complete.")

    # Demucs creates an output subfolder per run; find the most recent
    folders = sorted(Path(output_dir).glob("*/"), key=os.path.getmtime)
    if not folders:
        raise RuntimeError("Demucs did not produce an output folder")

    demucs_folder = folders[-1]
    print(f"  Found output folder: {demucs_folder.name}")

    vocal_file = demucs_folder / "vocals.wav"
    accomp_file = demucs_folder / "no_vocals.wav"

    print(f"  Vocals: {vocal_file}")
    print(f"  Instrumental: {accomp_file}")
    return str(vocal_file), str(accomp_file)


###############################################
# 2. SPEECH RECOGNITION (WHISPER large-v3)
###############################################
def transcribe_audio(vocal_wav):
    """
    Uses HuggingFace Whisper large-v3 for ASR.
    """
    logging.info("[2/5] Running Whisper ASR")
    print(f"  Input: {vocal_wav}")

    # instantiate pipeline lazily to avoid reloading the model repeatedly
    global _WHISPER_PIPELINE
    try:
        _WHISPER_PIPELINE
        print("  Using cached Whisper model.")
    except NameError:
        print("  Loading Whisper large-v3 model (first time, may take a moment)...")
        _WHISPER_PIPELINE = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-large-v3",
        )
        print("  Whisper model loaded.")

    print("  Transcribing audio...")
    result = _WHISPER_PIPELINE(str(vocal_wav))
    transcript = result.get("text", "")
    print(f"  Transcription complete.")

    logging.debug("Transcribed lyrics: %s", transcript)
    print(f"  Text: {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    return transcript


###############################################
# 3. MACHINE TRANSLATION (M2M100)
###############################################
def translate_text(text, src="en", tgt="es"):
    """
    Uses Facebook M2M100 for multilingual MT.
    """
    logging.info("[3/5] Translating lyrics")
    print(f"  Source language: {src}, Target language: {tgt}")

    global _TRANSLATOR_PIPELINE
    try:
        _TRANSLATOR_PIPELINE
        print("  Using cached translator model.")
    except NameError:
        print("  Loading M2M100 translation model (first time, may take a moment)...")
        _TRANSLATOR_PIPELINE = pipeline(
            "translation",
            model="facebook/m2m100_418M",
        )
        print("  Translator model loaded.")

    print("  Translating text...")
    translated = _TRANSLATOR_PIPELINE(text, src_lang=src, tgt_lang=tgt)[0].get(
        "translation_text", ""
    )
    print("  Translation complete.")

    logging.debug("Translation: %s", translated)
    print(f"  Translated text: {translated[:100]}{'...' if len(translated) > 100 else ''}")
    return translated


###############################################
# 4. TEXT-TO-SPEECH (XTTS - Coqui)
###############################################
def tts_generate(text, output_wav, language="es"):
    """
    Uses Coqui TTS XTTS (multilingual) to synthesize translated vocals.
    """
    logging.info("[4/5] Synthesizing translated vocals")
    print(f"  Language: {language}")
    print(f"  Output: {output_wav}")

    # Lazy-load Coqui TTS model
    global _TTS_MODEL
    try:
        _TTS_MODEL
        print("  Using cached TTS model.")
    except NameError:
        print("  Loading Coqui XTTS v2 model (first time, may take a moment)...")
        from TTS.api import TTS

        _TTS_MODEL = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        print("  TTS model loaded.")

    print("  Generating speech...")
    _TTS_MODEL.tts_to_file(
        text=text,
        file_path=str(output_wav),
        speaker="female-en-5",
        language=language,
    )
    print("  Speech generation complete.")

    return str(output_wav)


###############################################
# 5. MIX NEW VOCALS BACK INTO INSTRUMENTAL
###############################################
def mix_vocals_and_instrumental(vocals, instrumental, output_path):
    """
    Simple overlay mix using ffmpeg.
    (For more advanced mixing, adjust gain or alignment.)
    """
    logging.info("[5/5] Mixing translated vocals with instrumental")
    print(f"  Vocals: {vocals}")
    print(f"  Instrumental: {instrumental}")
    print(f"  Output: {output_path}")

    # Ensure inputs exist
    if not Path(vocals).exists():
        raise FileNotFoundError(f"Vocals file not found: {vocals}")
    if not Path(instrumental).exists():
        raise FileNotFoundError(f"Instrumental file not found: {instrumental}")

    print("  Mixing audio with ffmpeg...")
    (
        ffmpeg
        .input(instrumental)
        .filter("volume", 1.0)
        .overlay(ffmpeg.input(vocals), shortest=1)
        .output(output_path)
        .overwrite_output()
        .run(quiet=True)
    )
    print("  Mixing complete.")

    logging.info("Final translated song saved to: %s", output_path)
    print(f"  Final output: {output_path}")
    return str(output_path)


###############################################
# MAIN PIPELINE
###############################################
def translate_song(
    input_audio,
    src_lang="en",
    tgt_lang="es",
    output_name="translated_song.wav"
):
    print(f"\n{'='*60}")
    print(f"Song Translation Pipeline")
    print(f"{'='*60}")
    print(f"Input: {input_audio}")
    print(f"Source language: {src_lang}")
    print(f"Target language: {tgt_lang}")
    print(f"Output: {output_name}")
    print(f"{'='*60}\n")

    input_path = Path(input_audio)
    if not input_path.exists():
        raise FileNotFoundError(f"Input audio file not found: {input_audio}")

    with TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print(f"Temporary directory: {tmp}\n")

        # Step 1: Vocal Separation
        vocal_wav, instrumental_wav = separate_vocals(input_path, tmp)
        print()

        # Step 2: ASR
        transcript = transcribe_audio(vocal_wav)
        print()

        # Step 3: MT
        translated_lyrics = translate_text(transcript, src=src_lang, tgt=tgt_lang)
        print()

        # Step 4: TTS
        translated_vocals = tmp / "vocals_translated.wav"
        tts_generate(translated_lyrics, translated_vocals, language=tgt_lang)
        print()

        # Step 5: Mix
        final = mix_vocals_and_instrumental(
            str(translated_vocals), str(instrumental_wav), str(output_name)
        )
        print()

        return final


###############################################
# RUN DIRECTLY
###############################################
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Song translation pipeline")
    parser.add_argument("audio", help="Path to input audio file")
    parser.add_argument("--src", default="en", help="Source language")
    parser.add_argument("--tgt", default="es", help="Target language")
    parser.add_argument("--out", default="translated_song.wav", help="Output filename")

    args = parser.parse_args()

    try:
        result = translate_song(
            args.audio, src_lang=args.src, tgt_lang=args.tgt, output_name=args.out
        )
        print(f"\n{'='*60}")
        print(f"SUCCESS! Pipeline complete.")
        print(f"Output saved to: {result}")
        print(f"{'='*60}\n")
        logging.info("Pipeline finished. Output: %s", result)
    except Exception as exc:
        print(f"\n{'='*60}")
        print(f"ERROR: Pipeline failed.")
        print(f"{'='*60}\n")
        logging.exception("Pipeline failed: %s", exc)
        raise
