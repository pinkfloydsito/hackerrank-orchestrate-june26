#!/bin/bash
#SBATCH --job-name=infer_ensemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/infer_ensemble_%j.out
#SBATCH --error=logs/infer_ensemble_%j.err

set -euo pipefail

echo "=== Ensemble Inference Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Running tests before inference..."
python -m pytest tests/ -x --tb=short

echo "Starting ensemble inference on test claims..."
python scripts/run_ensemble_inference.py --dataset test --output outputs/output.csv

echo "=== Ensemble Inference Complete ==="
echo "Date: $(date)"
