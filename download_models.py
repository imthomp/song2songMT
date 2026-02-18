import os
from demucs.pretrained import get_model
from transformers import pipeline
from TTS.api import TTS

print("--- 1. Downloading Demucs (htdemucs) ---")
# This triggers the download to ~/.cache/torch/hub/checkpoints/
get_model('htdemucs') 
get_model('mdx_extra_q')

print("\n--- 2. Downloading Whisper (large-v3) ---")
# This triggers download to ~/.cache/huggingface/
pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

print("\n--- 3. Downloading M2M100 (Translator) ---")
pipeline("translation", model="facebook/m2m100_418M")

print("\n--- 4. Downloading Coqui TTS (XTTS v2) ---")
# This triggers download to ~/.local/share/tts/
TTS("tts_models/multilingual/multi-dataset/xtts_v2")

print("\n--- SUCCESS: All models are cached. ---")