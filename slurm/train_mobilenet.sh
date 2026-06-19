#!/bin/bash
#SBATCH --job-name=train_mobilenet
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100"
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -euo pipefail

echo "=== Training Job Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Running tests before training..."
python -m pytest tests/ -x --tb=short

echo "Starting training..."
python scripts/train_mobilenet.py

echo "=== Training Job Complete ==="
echo "Date: $(date)"
