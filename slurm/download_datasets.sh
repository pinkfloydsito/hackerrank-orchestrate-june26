#!/bin/bash
#SBATCH --job-name=download_datasets
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/download_%j.out
#SBATCH --error=logs/download_%j.err

set -euo pipefail

echo "=== Dataset Download Job Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Installing roboflow..."
pip install -q roboflow python-dotenv

echo "Starting download..."
python download_datasets.py

echo "=== Download Complete ==="
echo "Date: $(date)"

# List downloaded files
echo "Downloaded files:"
find data/raw -type f 2>/dev/null | head -50 || echo "No files found in data/raw/"
