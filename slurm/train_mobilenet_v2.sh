#!/bin/bash
#SBATCH --job-name=train_mobilenet_v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.err

set -euo pipefail

echo "=== Training V2 Job Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Running tests before training..."
python -m pytest tests/ -x --tb=short

echo "Starting improved training..."
python scripts/train_mobilenet_v2.py

echo "=== Training V2 Job Complete ==="
echo "Date: $(date)"
