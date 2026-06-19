#!/bin/bash
set -euo pipefail

LOGDIR="logs"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "  Multi-Modal Evidence Review Pipeline"
echo "  $(date)"
echo "=========================================="

# Step 1: Run tests
echo "[0/5] Running tests..."
python -m pytest tests/ -x --tb=short || {
    echo "ERROR: Tests failed. Fix before proceeding."
    exit 1
}

# Step 2: Download (skip if already done)
if [ ! -d "data/raw/car" ] || [ ! -d "data/raw/laptop" ] || [ ! -d "data/raw/package" ]; then
    echo "[1/5] Downloading datasets..."
    python scripts/download_datasets.py
else
    echo "[1/5] Download skipped (data/raw/ exists)"
fi

# Step 3: Preprocess (skip if already done)
if [ ! -f "data/processed/train.json" ]; then
    echo "[2/5] Preprocessing..."
    python scripts/preprocess.py
else
    echo "[2/5] Preprocess skipped (train.json exists)"
fi

# Step 4: Train (submit SLURM job)
echo "[3/5] Submitting training job..."
if [ ! -f "models/checkpoints/best_mobilenet.pt" ]; then
    JOBID=$(sbatch --parsable slurm/train_mobilenet.sh)
    echo "Training job submitted: $JOBID"
    echo "Waiting for job $JOBID..."
    while squeue -j "$JOBID" 2>/dev/null | grep -q "$JOBID"; do
        sleep 30
    done
    echo "Training job $JOBID completed"
    
    # Check if model was created
    if [ ! -f "models/checkpoints/best_mobilenet.pt" ]; then
        echo "ERROR: Training failed - no checkpoint found"
        exit 1
    fi
else
    echo "[3/5] Training skipped (checkpoint exists)"
fi

# Step 5: Evaluate on sample claims
echo "[4/5] Running inference on sample claims..."
python scripts/run_inference.py --dataset sample --output outputs/sample_predictions.csv

echo "[5/5] Evaluating..."
python scripts/evaluate.py --predictions outputs/sample_predictions.csv --ground-truth dataset/sample_claims.csv

echo ""
echo "=========================================="
echo "  Pipeline Complete!"
echo "  Check outputs/ for results"
echo "=========================================="