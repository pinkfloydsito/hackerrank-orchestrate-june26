#!/bin/bash
#SBATCH --job-name=infer_sample
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/infer_sample_%j.out
#SBATCH --error=logs/infer_sample_%j.err

set -euo pipefail

echo "=== Inference Job Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Running 3-stage pipeline inference on sample claims..."
python scripts/run_inference.py --dataset sample --output outputs/sample_predictions.csv

echo "=== Evaluating against ground truth ==="
python scripts/evaluate.py --predictions outputs/sample_predictions.csv --ground-truth dataset/sample_claims.csv --output outputs/evaluation_report.md

echo "=== Inference Job Complete ==="
echo "Date: $(date)"
