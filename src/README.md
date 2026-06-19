# HackerRank Orchestrate - Build Agent

## System Status

### Inference Jobs (Running)
- **Job 66739663** (V100 32GB, falcon1/2): Processing test claims - ~15-20s per claim
- **Job 66739622** (V100 16GB, falcon3): Processing test claims - OOM risk

### Training Jobs (Completed)
- **MobileNet Option A** (Job 66739540): ✅ Complete - Val loss 2.24, Issue 80.8%, Part 54.0%
- **YOLO11-nano** (Job 66739342): ✅ Complete - Best mAP50 0.448
- **MobileNet v2** (Job 66739261): ✅ Complete - Val loss 7.09 (worse than Option A)

### Code Status
- **All tests**: 86 passing (1 skipped - missing image locally)
- **Git**: Pushed to main (commit 5c746c4)
- **Research folder**: Created with strategy, pros/cons, requirements
- **Evaluation report**: Created in evaluation/evaluation_report.md

## What to Do Next

1. **Wait for inference job 66739663 to complete** (~30-40 min remaining)
2. **Copy outputs/output.csv from Tartu** to local repo
3. **Create code.zip** for submission (exclude .venv, node_modules, build artifacts)
4. **Verify output.csv** has 44 rows with exact required columns
5. **Upload chat transcript** from $HOME/hackerrank_orchestrate/log.txt

## How to Check Job Status

```bash
# Check all jobs
ssh tartu 'squeue -u andressebastian1'

# Check 32GB inference progress
ssh tartu 'tail -30 ~/hackathon/hackerrank-orchestrate-june26/logs/infer_ensemble_v32_66739663.out'

# Count processed claims
ssh tartu 'grep -c "Claim user_" ~/hackathon/hackerrank-orchestrate-june26/logs/infer_ensemble_v32_66739663.out'
```

## Repository Structure

```
src/hackerrank_orchestrate/
├── perception.py              # Qwen VLM wrapper
├── evidence_evaluator.py     # Deterministic rules
├── adjudicator.py            # Claim status decisions
├── yolo_detector.py         # YOLO11-nano detection
├── classifier_integration.py # MobileNet cross-check
├── localization.py          # Damage region cropping
├── data/                    # Dataset loaders
│   ├── dataset_loader.py   # PyTorch datasets
│   └── preprocessor.py      # Label normalization
├── models/                  # Model definitions
│   └── mobilenet_classifier.py  # MobileNetV3-Large
└── utils/                   # Logger, helpers

scripts/
├── run_ensemble_inference.py  # Full pipeline
├── train_mobilenet.py      # Option A training
├── train_yolo.py          # YOLO training
└── prepare_yolo_dataset.py # Data preprocessing

slurm/
├── inference_ensemble_v32.sh  # V100 32GB inference
├── train_mobilenet_optA.sh  # Option A training
└── train_yolo.sh           # YOLO training

evaluation/
└── evaluation_report.md     # Operational analysis

research/
├── STRATEGY.md            # Architecture overview
├── PROS_CONS.md          # Advantages/limitations
└── HACKATHON_REQUIREMENTS.md  # Requirements checklist
```

## Key Design Decisions

1. **Three-stage pipeline**: VLM (perception) → Rules (evidence) → Rules (adjudication)
2. **AdamW > SGD**: Simple cross entropy outperformed weighted loss + label smoothing
3. **Light augmentation**: RandomHorizontalFlip + mild ColorJitter > heavy augmentation
4. **Confidence thresholds**: 0.35 supported, 0.5 mismatch, 0.2 evidence gate
5. **Wrong object → contradicted**: Clear visual contradiction is not "not enough info"

## Submission Checklist

- [ ] output.csv generated (44 rows, exact columns)
- [ ] code.zip created (exclude .venv, build artifacts)
- [ ] evaluation/ folder included
- [ ] README.md updated
- [ ] Chat transcript uploaded
- [ ] All tests passing
