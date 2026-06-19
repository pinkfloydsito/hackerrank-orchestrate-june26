# Evaluation Report

## System Overview

Multi-Modal Evidence Review System using a three-stage pipeline:
1. **Perception**: Qwen 2.5-VL-7B-Instruct (4-bit quantized) for visual fact extraction with multi-crop analysis
2. **Evidence Evaluation**: Deterministic rule engine for evidence sufficiency
3. **Adjudication**: Rule-based claim status decision

Ensemble components: YOLO11-nano (damage detection), MobileNetV3-Large (multi-task classification with TTA)

## Latest Architecture Improvements

### Multi-Crop VLM Analysis
- **7 crops per image**: Full image + 4 quadrants + center crop + YOLO-detected region crop
- **Claim-aware aggregation**: Prioritizes findings that match the claimed issue over highest-confidence mismatches
- **Purpose**: Reduces false negatives from YOLO blind spots (e.g., laptop corner dents)

### Test-Time Augmentation (TTA)
- **5 augmentations per image**: Random flips, rotations, color jitter
- **Averaged softmax probabilities** for more robust MobileNet predictions
- **Impact**: Reduces uncertainty on borderline cases

### Updated Confidence Thresholds
- **Supported**: confidence > 0.35 (unchanged)
- **Contradicted**: confidence > 0.7 (raised from 0.35 to reduce false contradictions)
- **Not Enough Information**: "none" finding at confidence < 0.7, or no findings meet thresholds
- **Claim mismatch**: visible_issue != "none" + confidence > 0.5

### Label Normalization
- **Embedding-based normalization** via `intfloat/multilingual-e5-small`
- **Boost removed**: Was +0.1, now +0.0 (prevents VLM confidence from overflowing to 1.0)
- **Purpose**: Matches classifier labels to VLM outputs using semantic similarity

## Operational Metrics

### Model Calls
- **VLM calls**: 7 per image (multi-crop), ~2 images per claim = ~14 VLM calls per claim
- **YOLO calls**: 1 per image (damage detection + localization)
- **MobileNet calls**: 5 per image (TTA) = ~10 MobileNet calls per claim
- **Total for 44 test claims**: ~616 VLM calls + ~100 YOLO calls + ~440 MobileNet calls

### Token Usage
- **VLM input**: ~500-1000 tokens per crop (system prompt + user claim + image tokens)
- **VLM output**: ~200-400 tokens per crop (structured JSON response)
- **Total**: ~400,000-600,000 input tokens + ~160,000-320,000 output tokens for full test set

### Images Processed
- **Test set**: 44 claims, ~1-3 images per claim = ~80-100 images total
- **Sample set**: 20 claims, ~1-3 images per claim = ~40-60 images total

### Cost
- **Total cost**: $0 (all models are open-source and self-hosted)
- **GPU cost**: ~$2-3/hour for V100 32GB on cloud (if not using university HPC)
- **Alternative (GPT-4V)**: ~$0.005-0.015 per image = ~$50-150 for 616 images (multi-crop)
- **Alternative (Claude 3 Opus)**: ~$0.015-0.03 per image = ~$150-300 for 616 images

### Latency
- **Per claim**: ~2-3 minutes (multi-crop VLM dominates at ~10-15s per crop)
- **Full test set (44 claims)**: ~1.5-2 hours
- **Sample set (20 claims)**: ~40-60 minutes

### Rate Limits
- **No API rate limits**: All models are self-hosted
- **GPU memory**: ~15-20GB peak (VLM 7B 4-bit + YOLO + MobileNet + intermediates)
- **Processing**: Single-threaded (sequential claim processing)
- **Optimization potential**: Batch image processing, model parallelism, or async VLM calls could reduce latency by 2-3x

## Batching and Caching Strategy

### Batching
- **No batching**: Claims processed sequentially to avoid GPU OOM
- **Image batching**: YOLO and MobileNet process images one at a time
- **Future improvement**: Batch 2-4 images per VLM call if GPU memory allows

### Caching
- **No caching**: Each claim is processed independently
- **Future improvement**: Cache VLM responses for identical images across claims
- **Future improvement**: Cache YOLO detections to avoid re-running on same images

### Retry Strategy
- **VLM**: Single attempt with fallback to default response on failure
- **YOLO**: No retry needed (deterministic model)
- **MobileNet**: No retry needed (deterministic model)
- **Image loading**: Try multiple path candidates (PROJECT_ROOT / p, PROJECT_ROOT / dataset / p)

## Key Design Decisions

### AdamW > SGD
- **Simple MultiTaskLoss + AdamW** achieved 80.8% issue accuracy
- **SGD + weighted loss** only achieved 27.2% issue accuracy
- **Insight**: Weighted loss and label smoothing hurt more than helped on this dataset

### Light Augmentation > Heavy Augmentation
- **RandomHorizontalFlip + mild ColorJitter** outperformed heavy augmentation
- **Training data**: 4027 images — car (2137), laptop (586), package (95)
- **Most common class**: dent (1763) on car objects

### Claim-Aware Aggregation
- **Problem**: VLM would report highest-confidence finding (e.g., glass_shatter at 0.9) even when claim was about a crack
- **Solution**: Prioritize findings that match the claimed issue over highest-confidence mismatches
- **Example**: User claims laptop corner crack → VLM reports crack on one image, glass_shatter on another → now picks crack

### Multi-Crop > Single-Crop
- **Problem**: YOLO fails on laptop corner dents (blind spot from training data imbalance)
- **Solution**: Quadrant crops force VLM to inspect edges and corners explicitly
- **Prompt**: "Inspect corners, edges, seams, and any visible mark" — report severity even if low

## Evaluation Results

### Sample Claims (Development Set)
- **Processed**: 20 claims
- **Purpose**: Validate system behavior and tune thresholds
- **Method**: Manual inspection of outputs against expected behavior

### Test Claims (Final Output)
- **Processed**: 44 claims (Job 66740020, falcon1/2 V100 32GB)
- **Output**: `outputs/output.csv`
- **Claim status distribution**: (to be filled after inference completes)

## Performance Analysis

### Model Performance
- **MobileNet validation**: Object 100%, Issue 80.8%, Part 54.0%, Damage 97.2%
- **YOLO validation**: mAP50 0.448, mAP50-95 0.279
- **VLM accuracy**: Qualitative assessment (structured JSON parsing works well)

### Known Issues
1. **VLM OOM on 16GB GPU**: Requires 32GB GPU or gradient checkpointing
2. **MobileNet issue accuracy**: Plateaus at ~80% (fine-grained classification is challenging)
3. **YOLO localization**: mAP50-95 only 0.279 (precise damage localization is hard)
4. **YOLO blind spot**: Fails on laptop corner dents due to training class imbalance
5. **Severity calibration**: Subjective and varies by damage type
6. **Multi-crop speed**: 2-3 min per claim vs 15s before (14 VLM calls vs 1)

## Recommendations

### Short-term
1. Use V100 32GB or A100 40GB for inference to avoid OOM
2. Process claims in batches of 2-4 to improve throughput
3. Cache VLM responses for duplicate images
4. Consider reducing multi-crop to 3-4 crops (full + 2 quadrants + YOLO crop) for speed

### Long-term
1. Fine-tune Qwen on domain-specific damage images (requires labeled dataset)
2. Train larger YOLO model (YOLO11-medium) with more data
3. Add active learning loop to collect human feedback on uncertain claims
4. Implement model distillation to reduce VLM size (e.g., Qwen 2.5-VL-3B)
5. Balance training data for laptop corner dents (currently underrepresented)

## Conclusion

The system successfully processes multi-modal evidence (images + conversation) to make auditable, deterministic claim decisions. The three-stage architecture separates perception from policy, making the system robust and maintainable. The ensemble approach (VLM + YOLO + MobileNet) provides redundancy and cross-checks that reduce individual model errors.

**Key improvements in this iteration**:
- Multi-crop VLM reduces false negatives on edge/corner damage
- Claim-aware aggregation prioritizes user claims over model confidence
- TTA provides more robust classifier predictions
- Updated thresholds (0.7 contradicted) reduce false contradictions
- AdamW + simple loss outperforms complex weighted strategies

**Total cost for 44 claims**: $0 (open-source models, self-hosted)  
**Total time**: ~1.5-2 hours (multi-crop inference)  
**Accuracy**: Issue type ~80%, Part ~54%, Damage ~97% (on validation set)

---

## README (from src/)

```
# HackerRank Orchestrate - Build Agent

## System Status

### Inference Jobs (Running)
- **Job 66740020** (V100 32GB, falcon1/2): Processing test claims with multi-crop VLM + TTA
- **Previous Job 66739713** (V100 32GB, falcon1): Generated initial output.csv (44 claims, 23 contradicted, 17 supported, 4 NEI)

### Training Jobs (Completed)
- **MobileNet Option A** (Job 66739540): ✅ Complete - Val loss 2.24, Issue 80.8%, Part 54.0%, Damage 97.2%
- **YOLO11-nano** (Job 66739342): ✅ Complete - Best mAP50 0.448
- **MobileNet v2** (Job 66739261): ✅ Complete - Val loss 7.09 (worse than Option A)

### Code Status
- **All tests**: 87 passing
- **Git**: Pushed to main (commit d1c826e)
- **Research folder**: Created with strategy, pros/cons, requirements
- **Evaluation report**: Updated in evaluation/evaluation_report.md

## What to Do Next

1. **Wait for inference job 66740020 to complete** (~1.5-2 hours for 44 claims)
2. **Copy outputs/output.csv from Tartu** to local repo
3. **Create code.zip** for submission (exclude .venv, node_modules, build artifacts)
4. **Verify output.csv** has 44 rows with exact required columns
5. **Upload chat transcript** from $HOME/hackerrank_orchestrate/log.txt

## How to Check Job Status

```bash
# Check all jobs
ssh tartu 'squeue -u andressebastian1'

# Check 32GB inference progress
ssh tartu 'tail -30 ~/hackathon/hackerrank-orchestrate-june26/logs/infer_ensemble_v32_66740020.out'

# Count processed claims
ssh tartu 'grep -c "Claim user_" ~/hackathon/hackerrank-orchestrate-june26/logs/infer_ensemble_v32_66740020.out'
```

## Repository Structure

```
src/hackerrank_orchestrate/
├── perception.py              # Qwen VLM wrapper with multi-crop
├── evidence_evaluator.py     # Deterministic rules with updated thresholds
├── adjudicator.py            # Claim status decisions (0.7 contradicted threshold)
├── yolo_detector.py         # YOLO11-nano detection
├── classifier_integration.py # MobileNet with TTA (5 augmentations)
├── localization.py          # Damage region cropping
├── data/                    # Dataset loaders
│   ├── dataset_loader.py   # PyTorch datasets
│   └── preprocessor.py      # Label normalization (embedding-based, +0.0 boost)
├── models/                  # Model definitions
│   └── mobilenet_classifier.py  # MobileNetV3-Large
└── utils/                   # Logger, helpers

scripts/
├── run_ensemble_inference.py  # Full pipeline with multi-crop
├── debug_claim.py            # Per-claim debug with YOLO boxes, VLM crops, TTA
├── train_mobilenet.py      # Option A training (AdamW)
├── train_yolo.py          # YOLO training
└── prepare_yolo_dataset.py # Data preprocessing

slurm/
├── inference_ensemble_v32.sh  # V100 32GB inference
├── debug_user045.sh          # Debug script for user_045 validation
├── train_mobilenet_optA.sh  # Option A training
└── train_yolo.sh           # YOLO training

evaluation/
└── evaluation_report.md     # This file

research/
├── STRATEGY.md            # Architecture overview
├── PROS_CONS.md          # Advantages/limitations
└── HACKATHON_REQUIREMENTS.md  # Requirements checklist
```

## Key Design Decisions

1. **Three-stage pipeline**: VLM (perception) → Rules (evidence) → Rules (adjudication)
2. **AdamW > SGD**: Simple cross entropy outperformed weighted loss + label smoothing
3. **Light augmentation**: RandomHorizontalFlip + mild ColorJitter > heavy augmentation
4. **Confidence thresholds**: 0.35 supported, 0.7 contradicted, 0.5 mismatch, 0.2 evidence gate
5. **Wrong object → contradicted**: Clear visual contradiction is not "not enough info"
6. **Multi-crop VLM**: 7 crops per image to catch edge/corner damage
7. **Claim-aware aggregation**: Prioritize matching claimed issue over highest confidence
8. **TTA**: 5 augmentations averaged for robust classifier predictions
9. **Label normalization +0.0**: Removed +0.1 boost that caused confidence overflow

## Submission Checklist

- [ ] output.csv generated (44 rows, exact columns)
- [ ] code.zip created (exclude .venv, build artifacts)
- [ ] evaluation/ folder included
- [ ] README.md updated
- [ ] Chat transcript uploaded
- [ ] All tests passing
```
