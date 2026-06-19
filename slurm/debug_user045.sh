#!/bin/bash
#SBATCH --job-name=debug_user045
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodelist=falcon1,falcon2
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/debug_user045_%j.out
#SBATCH --error=logs/debug_user045_%j.err

set -euo pipefail

echo "=== Debug user_045 Started ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd ~/hackathon/hackerrank-orchestrate-june26 || exit 1
source .venv/bin/activate

echo "Running debug on user_045..."
python scripts/debug_claim.py --user_id user_045 --dataset test

echo "=== Debug Complete ==="
echo "Date: $(date)"
