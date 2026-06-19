#!/bin/bash
#SBATCH --job-name=train_yolo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/train_yolo_%j.out
#SBATCH --error=logs/train_yolo_%j.err

set -euo pipefail

echo "=== YOLO Training Job Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Preparing YOLO dataset..."
python scripts/prepare_yolo_dataset.py

echo "Starting YOLO training..."
python scripts/train_yolo.py

echo "=== YOLO Training Job Complete ==="
echo "Date: $(date)"
