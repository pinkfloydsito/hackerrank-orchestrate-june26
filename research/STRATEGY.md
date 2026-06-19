# Multi-Modal Evidence Review System - Strategy

## Architecture Overview

Our system uses a **three-stage pipeline** designed to separate perception from policy:

### Stage 1: Perception (VLM Only)
- **Qwen 2.5-VL-7B-Instruct** (4-bit quantized) processes images
- Task: Describe visual facts only (damage type, location, severity, image quality)
- No policy decisions - the model is a "camera, not a judge"
- Outputs structured JSON with 7 fields: valid_image, visible_issue, object_part, severity, confidence, supporting_image_ids, observations, risk_flags
- **Embedding-based label normalization** maps VLM descriptions to canonical labels using `intfloat/multilingual-e5-small`

### Stage 2: Evidence Evaluation (Deterministic Rules)
- Applies business logic to VLM findings
- Checks: claim mismatch, severity exaggeration, evidence sufficiency, user history risk
- **Confidence-gated decisions**: Low confidence (< 0.2) triggers manual review
- Outputs: evidence_standard_met (true/false), risk_flags, evidence_standard_met_reason

### Stage 3: Adjudication (Rule Engine)
- Makes final claim_status decision based on evidence evaluation
- Rules:
  - Wrong object → **contradicted** (not "not_enough_information")
  - Claim mismatch (confidence > 0.5) → **contradicted**
  - No damage visible (confidence > 0.35) → **not_enough_information** or **contradicted** depending on claim strength
  - Supported evidence (confidence > 0.35) → **supported**
  - Unknown issue with high confidence → **contradicted**
  - Unknown issue with low confidence → **not_enough_information**

## Ensemble Components

### YOLO11-nano (Damage Detection)
- Trained on 2992 annotated images from Roboflow datasets (car, laptop, package)
- Provides automatic damage localization when preprocessed bboxes are unavailable
- Classification: dent, scratch, crack, water_damage, etc.
- Used as a first opinion for cropping and cross-checking VLM

### MobileNetV3-Large (Multi-Task Classifier)
- 4-task head: object_type (3 classes), issue_type (12 classes), object_part (27 classes), has_damage (binary)
- **Architecture decision**: AdamW optimizer + simple MultiTaskLoss (cross entropy) worked best
- SGD + weighted loss + label smoothing actually **degraded** performance (27% vs 82% issue accuracy)
- Serves as a "second opinion" cross-check with VLM
- Confidence-based ensemble: trust classifier > 0.85, trust VLM if classifier < 0.6

## Key Design Decisions

### 1. Separate Perception from Policy
- VLM describes only visual facts
- Evidence evaluator applies deterministic business rules
- Adjudicator makes final policy decisions
- Prevents VLM hallucination from making claim decisions

### 2. Embedding-Based Label Normalization
- VLM outputs descriptive text (e.g., "big dent on front")
- Normalizer maps to canonical labels using cosine similarity of embeddings
- Handles synonyms and variations automatically

### 3. Confidence Thresholds
- **Evidence evaluator gate**: 0.2 (lowered from 0.3 to reduce false rejections)
- **Mismatch threshold**: 0.5 (lowered from 0.6)
- **No-damage threshold**: 0.35 (lowered from 0.6)
- **Supported threshold**: 0.35 (lowered from 0.4)

### 4. Three-Stage Training
1. **YOLO training**: 150 epochs on COCO-formatted annotations
2. **MobileNet training**: 30 epochs with early stopping (patience 7)
3. **VLM inference**: No training required (zero-shot with structured prompting)

## Data Flow

```
Claims CSV → Image Loading → YOLO Detection → Image Cropping → VLM Perception
                                    ↓                                         ↓
                              MobileNet Classifier ← Cross-check ← Label Normalization
                                    ↓                                         ↓
                              Evidence Evaluator → Adjudicator → Output CSV
```

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| VLM hallucination | Three-stage separation; VLM only describes, never decides |
| OOM on 16GB V100 | Process images one at a time; use 4-bit quantization; fallback to 32GB GPU |
| Class imbalance | WeightedRandomSampler (training); class weights (optional, but proved harmful) |
| Overfitting | Dropout 0.3, early stopping, light augmentation (RandomHorizontalFlip + mild ColorJitter) |
| Label mismatch | Embedding-based normalization with 5-smooth similarity |

## Performance

### Training Results
- **MobileNet v1 (AdamW, no aug)**: Val loss 2.56, Issue 81.8%, Part 58.8%, Damage 96.7%
- **MobileNet v2 (SGD, heavy aug, weighted)**: Val loss 7.09, Issue 27.2%, Part 37.6%, Damage 95.9%
- **MobileNet Option A (AdamW, light aug)**: Val loss 2.24, Issue 80.8%, Part 54.0%, Damage 97.2%
- **YOLO11-nano**: Best mAP50 0.448, mAP50-95 0.279

### Inference
- ~15-20 seconds per claim (VLM dominates)
- 44 claims in ~15-20 minutes
- GPU memory: ~15-20GB peak (VLM 7B 4-bit + YOLO + MobileNet + intermediate tensors)
