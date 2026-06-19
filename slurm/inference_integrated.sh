#!/bin/bash
#SBATCH --job-name=infer_integrated
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/infer_integrated_%j.out
#SBATCH --error=logs/infer_integrated_%j.err

set -euo pipefail

echo "=== Integrated Inference Job Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Running integrated inference with localization + classifier..."
python scripts/run_integrated_inference.py --dataset sample --output outputs/sample_predictions_integrated.csv

echo "=== Evaluating integrated results ==="
python scripts/evaluate.py \
    --predictions outputs/sample_predictions_integrated.csv \
    --ground-truth dataset/sample_claims.csv \
    --output outputs/evaluation_report_integrated.md

echo "=== Inference Job Complete ==="
echo "Date: $(date)"
