#!/bin/bash
#SBATCH --partition=cs
#SBATCH --qos=cs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# If SLURM_JOB_ID is not set, then we are on the login node.
# We need to submit the job to SLURM.
if [ -z "$SLURM_JOB_ID" ]; then
    # 1. Get the input file from the command line
    INPUT_FILE="$1"
    if [ -z "$INPUT_FILE" ]; then
        echo "Usage: $0 <input_audio_file>"
        exit 1
    fi

    # 2. Extract just the name (e.g., "songs/input/firstnoel.mp3" -> "firstnoel")
    # This removes the folder path and the extension
    SONG_NAME=$(basename "$INPUT_FILE" | cut -d. -f1)

    # 3. Define the Log file and Output Audio names
    LOG_FILE="songs/output/${SONG_NAME}.out"
    AUDIO_OUT="songs/output/${SONG_NAME}.mp3"

    echo "Submitting job for: $SONG_NAME"
    echo "Log file will be:   $LOG_FILE"

    # 4. Submit to Slurm
    # We pass -o to name the log file dynamically
    # We pass the arguments explicitly to this script
    sbatch -o "$LOG_FILE" --job-name="${SONG_NAME}" "$0" "$INPUT_FILE" en es "$AUDIO_OUT"

    exit 0
fi

# If we are here, we are on a compute node.
echo "Job started on $(hostname) at $(date)"

# Load modules
module load python/3.11
module load ffmpeg # Uncomment if ffmpeg is not found later

echo "--- FFMPEG DEBUG ---"
echo "which ffmpeg:"
which ffmpeg
echo "ffmpeg -version:"
ffmpeg -version
echo "--- END FFMPEG DEBUG ---"

# Activate environment
VENV_DIR=".venv"
source "$VENV_DIR/bin/activate"

# Assign Arguments
INPUT_AUDIO="$1"
SRC_LANG="${2:-en}"
TGT_LANG="${3:-es}"
OUT_FILE="${4:-translated_song.wav}"

# Debugging Output
echo "----------------------------------------"
echo "Processing: $INPUT_AUDIO"
echo "Source: $SRC_LANG | Target: $TGT_LANG"
echo "Output: $OUT_FILE"
echo "----------------------------------------"

# Run
python main.py "$INPUT_AUDIO" --src "$SRC_LANG" --tgt "$TGT_LANG" --out "$OUT_FILE"

echo "Job completed at $(date)"