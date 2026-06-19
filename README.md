# Multi-Modal Evidence Review System

## Overview

This system verifies damage claims using a **three-stage pipeline** that separates visual perception from policy decisions:

1. **Perception**: Qwen 2.5-VL-7B-Instruct extracts visual facts from images
2. **Evidence Evaluation**: Deterministic rules assess evidence sufficiency
3. **Adjudication**: Rule engine makes final claim decisions

**Ensemble components**: YOLO11-nano (damage detection) + MobileNetV3-Large (multi-task classification) cross-check VLM outputs.

## Architecture

```
Claims CSV → Image Loading → YOLO Detection → Image Cropping → VLM Perception
                                    ↓                                         ↓
                              MobileNet Classifier ← Cross-check ← Label Normalization
                                    ↓                                         ↓
                              Evidence Evaluator → Adjudicator → Output CSV
```

## Key Features

- **Three-stage separation**: VLM is a "camera" (describes only), rules are the "judge" (decides)
- **Embedding-based label normalization**: Maps VLM descriptions to canonical labels using `intfloat/multilingual-e5-small`
- **Confidence-based ensemble**: Cross-checks VLM + YOLO + MobileNet predictions
- **Zero API cost**: All models are open-source (Qwen, YOLO, MobileNet)
- **Deterministic decisions**: Every claim_status has a traceable reason

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (16GB+ for training, 32GB+ recommended for inference)
- SLURM cluster access (for HPC training)

### Installation

```bash
git clone <repo-url>
cd hackerrank-orchestrate-june26
pip install -e ".[dev]"
```

### Running Inference

```bash
# Local (requires GPU with 32GB+ memory)
python scripts/run_ensemble_inference.py --dataset test --output outputs/output.csv

# SLURM (recommended for HPC)
sbatch slurm/inference_ensemble_v32.sh
```

### Running Training

```bash
# MobileNet (Option A - AdamW + light augmentation)
sbatch slurm/train_mobilenet_optA.sh

# YOLO damage detection
sbatch slurm/train_yolo.sh
```

## Repository Structure

```
.
├── src/hackerrank_orchestrate/          # Core pipeline
│   ├── perception.py                    # Qwen VLM wrapper
│   ├── evidence_evaluator.py          # Deterministic rules
│   ├── adjudicator.py                  # Claim status decisions
│   ├── yolo_detector.py              # YO11-nano detection
│   ├── classifier_integration.py     # MobileNet cross-check
│   ├── localization.py               # Damage region cropping
│   ├── data/                           # Dataset loaders
│   └── models/                         # Model definitions
├── scripts/                             # Training & inference scripts
│   ├── run_ensemble_inference.py       # Full pipeline
│   ├── train_mobilenet.py            # MobileNet training (Option A)
│   ├── train_yolo.py                 # YOLO training
│   └── prepare_yolo_dataset.py      # Data preprocessing
├── slurm/                               # SLURM job scripts
├── tests/                               # 86 tests covering all components
├── evaluation/                          # Evaluation report
├── research/                           # Strategy documentation
└── dataset/                            # Input data (claims, images, history)
```

## Evaluation

### Running Tests

```bash
pytest tests/ -x
```

86 tests covering:
- Configuration and data loading
- Label normalization
- MobileNet forward pass and training
- Three-stage pipeline integration
- Inference and parsing

### Evaluation Report

See `evaluation/evaluation_report.md` for:
- Model call counts and token usage
- Cost analysis ($0 - all open-source)
- Latency metrics (~15-20s per claim)
- Performance benchmarks

## Performance

### Training Results

| Model | Val Loss | Issue Acc | Part Acc | Damage Acc |
|-------|----------|-----------|----------|------------|
| MobileNet v1 (AdamW, no aug) | 2.56 | 81.8% | 58.8% | 96.7% |
| MobileNet v2 (SGD, heavy aug) | 7.09 | 27.2% | 37.6% | 95.9% |
| MobileNet Option A (AdamW, light aug) | 2.24 | 80.8% | 54.0% | 97.2% |
| YOLO11-nano | mAP50: 0.448 | mAP50-95: 0.279 | - | - |

### Key Insight

**AdamW + simple loss outperformed SGD + weighted loss + label smoothing** (2.24 vs 7.09 val loss, 80.8% vs 27.2% issue accuracy). Heavy augmentation (RandomResizedCrop, RandomErasing) degraded performance. Light augmentation (RandomHorizontalFlip + mild ColorJitter) was optimal.

## Design Decisions

1. **Separate perception from policy**: VLM describes only visual facts; rules make decisions
2. **Confidence thresholds**: Lowered from 0.6 to 0.35-0.5 to reduce false "not_enough_information"
3. **Wrong object → contradicted**: Not "not_enough_information" - clear visual contradiction
4. **Ensemble priority**: YOLO detection > VLM reasoning > MobileNet classification > Evidence rules

## Hackathon Requirements

✅ Reads `dataset/claims.csv` and produces `output.csv`  
✅ Includes `evaluation/` folder with evaluation report  
✅ Uses `dataset/sample_claims.csv` for development evaluation  
✅ Operational analysis covering cost, latency, and rate limits  
✅ Deterministic, auditable decisions with traceable reasoning  

## License

MIT
