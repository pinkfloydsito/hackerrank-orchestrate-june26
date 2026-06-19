# Pros and Cons Analysis

## Advantages of Our Approach

### 1. Deterministic Policy Layer
- **Pro**: Business rules are explicit, auditable, and reproducible
- **Pro**: Easy to adjust thresholds without retraining models
- **Pro**: No "black box" decisions - every claim_status has a traceable reason
- **Pro**: Compliance-friendly for insurance/fraud detection use cases

### 2. Model-Agnostic Perception
- **Pro**: VLM can be swapped (Qwen, GPT-4V, Claude) without changing policy logic
- **Pro**: Structured JSON output makes parsing robust
- **Pro**: Embedding-based normalization handles VLM label variability

### 3. Ensemble Redundancy
- **Pro**: YOLO + MobileNet + VLM provide cross-checks (not single point of failure)
- **Pro**: Confidence-based arbitration handles model disagreement gracefully
- **Pro**: When VLM is uncertain, YOLO/MobileNet can provide hints

### 4. Cost Efficiency
- **Pro**: Qwen 2.5-VL-7B is open-source (no API costs)
- **Pro**: 4-bit quantization fits on consumer/entry-level GPUs (V100 16GB)
- **Pro**: YOLO11-nano and MobileNet are tiny models (fast inference)
- **Pro**: ~$0 cost for inference vs. GPT-4V at $0.005-0.015/image

### 5. Training Data Efficiency
- **Pro**: YOLO trained on 2992 annotated images (small dataset)
- **Pro**: MobileNet trained on 2818 images (transfer learning from ImageNet)
- **Pro**: VLM is zero-shot (no training required)

### 6. Modular Architecture
- **Pro**: Each stage can be developed, tested, and deployed independently
- **Pro**: Easy to add new rules or new models without touching other stages
- **Pro**: 86 tests covering all components (config, data loader, inference, label normalization, integration, MobileNet, three-stage pipeline)

## Disadvantages and Limitations

### 1. VLM Dependency
- **Con**: VLM is the bottleneck (~15-20 seconds per claim)
- **Con**: GPU memory requirements are high (15-20GB peak with 7B model)
- **Con**: OOM errors on 16GB GPUs require workarounds (gradient checkpointing, single-image processing, or 32GB GPU)
- **Con**: VLM can still hallucinate visual facts (mitigated by ensemble)

### 2. Rule Maintenance
- **Con**: Deterministic rules require manual tuning for edge cases
- **Con**: New damage types or new fraud patterns require rule updates
- **Con**: Threshold tuning is heuristic (0.2, 0.35, 0.5) - may not generalize to all datasets

### 3. MobileNet Limitations
- **Con**: Issue type accuracy plateaus at ~80% (fine-grained classification is hard)
- **Con**: Object part accuracy ~54-59% (many similar-looking parts)
- **Con**: Heavy augmentation (RandomResizedCrop, RandomErasing) actually degraded performance
- **Con**: SGD + class weights + label smoothing were counterproductive (27% vs 82% issue accuracy)

### 4. YOLO Limitations
- **Con**: mAP50-95 only 0.279 (localization is imprecise for small damage)
- **Con**: Trained on limited dataset (2992 images, 98 test)
- **Con**: Class imbalance (dent 1763 vs scratch 30 samples)
- **Con**: Not all damage types are equally detectable

### 5. Dataset Gaps
- **Con**: No ground truth labels for test claims (44 rows) - cannot evaluate locally
- **Con**: Sample claims (20 rows) are too small for reliable evaluation
- **Con**: User history is sparse (only past claim counts, no behavioral features)

### 6. Edge Cases
- **Con**: Multiple images per claim - which image is "supporting"? (VLM doesn't always populate supporting_image_ids)
- **Con**: Severity calibration is subjective (low vs medium vs high)
- **Con**: "Unknown" issue handling is tricky - when to say NEI vs contradicted?

## Comparison with Alternatives

| Approach | Pros | Cons | Our Choice |
|----------|------|------|------------|
| **End-to-end LLM** (GPT-4V decides everything) | Simple, no code | Expensive, non-deterministic, hard to audit | ❌ Rejected |
| **Pure rules + CV** (no VLM) | Fast, cheap | Misses nuanced damage, can't read conversation | ❌ Rejected |
| **VLM + LLM** (VLM describes, LLM decides) | Flexible, handles text | Expensive, slower, LLM can hallucinate policy | ⚠️ Partial (VLM + rules) |
| **Our approach: VLM + Rules + Ensemble** | Auditable, fast, cheap, robust | Requires tuning, multiple models | ✅ Chosen |

## What We Learned

1. **AdamW > SGD for transfer learning**: SGD with Nesterov momentum performed worse on this small dataset (27% vs 82% issue accuracy)
2. **Simple > complex for loss**: Plain cross entropy outperformed weighted loss + label smoothing (2.24 vs 7.09 val loss)
3. **Light augmentation > heavy**: RandomHorizontalFlip + mild ColorJitter was better than RandomResizedCrop + RandomErasing
4. **Ensemble > single model**: VLM + YOLO + MobileNet cross-checks reduced individual model errors
5. **Separation of concerns > end-to-end**: Keeping VLM as "camera" and rules as "judge" improved reliability
