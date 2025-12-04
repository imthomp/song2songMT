#!/bin/bash
#SBATCH --job-name=song_translation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=job_%j.log
#SBATCH --error=job_%j.err

# Load required modules
module load cuda/12.1
module load python/3.11

# Create and activate virtual environment (if needed)
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python main.py "$1" --src "${2:-en}" --tgt "${3:-es}" --out "${4:-translated_song.wav}"

echo "Job completed at $(date)"