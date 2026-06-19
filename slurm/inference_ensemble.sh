#!/bin/bash
#SBATCH --job-name=infer_ensemble
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/infer_ensemble_%j.out
#SBATCH --error=logs/infer_ensemble_%j.err

set -euo pipefail

echo "=== Ensemble Inference Job Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Running full ensemble inference (YOLO + MobileNet + VLM)..."
python scripts/run_ensemble_inference.py --dataset sample --output outputs/sample_predictions_ensemble.csv

echo "=== Evaluating ensemble results ==="
python scripts/evaluate.py \
    --predictions outputs/sample_predictions_ensemble.csv \
    --ground-truth dataset/sample_claims.csv \
    --output outputs/evaluation_report_ensemble.md

echo "=== Ensemble Inference Job Complete ==="
echo "Date: $(date)"
