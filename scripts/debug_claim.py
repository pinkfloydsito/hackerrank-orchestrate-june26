#!/usr/bin/env python3
"""Debug script for single-claim visual inspection.

Generates debug images with bounding boxes, crops, and model outputs.
"""

import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import pandas as pd

from hackerrank_orchestrate.perception import QwenPerception, VisualFindings
from hackerrank_orchestrate.localization import DamageLocalizer
from hackerrank_orchestrate.yolo_detector import YOLODetector
from hackerrank_orchestrate.classifier_integration import ClassifierSecondOpinion
from hackerrank_orchestrate.evidence_evaluator import evaluate_evidence, _extract_claimed_issue
from hackerrank_orchestrate.adjudicator import adjudicate
from hackerrank_orchestrate.utils.logger import setup_logger
from hackerrank_orchestrate.config import PROJECT_ROOT, SAMPLE_CLAIMS_PATH, TEST_CLAIMS_PATH

logger = setup_logger(__name__)


def draw_yolo_boxes(image: Image.Image, yolo_results, output_path: Path):
    """Draw YOLO bounding boxes on image and save."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for r in yolo_results:
        if len(r.boxes) == 0:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)
        
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            label = f"class:{cls} conf:{conf:.2f}"
            draw.text((x1, y1 - 20), label, fill="red", font=font)
    
    img.save(output_path)
    logger.info(f"Saved YOLO debug image to {output_path}")


def debug_claim(
    user_id: str,
    claims_path: Path = TEST_CLAIMS_PATH,
    output_dir: Path = Path("debug"),
) -> None:
    """Debug a single claim with full visual output."""
    
    claims_df = pd.read_csv(claims_path)
    claim_row = claims_df[claims_df["user_id"] == user_id]
    
    if claim_row.empty:
        logger.error(f"Claim {user_id} not found in {claims_path}")
        return
    
    row = claim_row.iloc[0]
    
    output_dir = output_dir / user_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Debugging claim {user_id}: {row['claim_object']}")
    logger.info(f"Claim: {row['user_claim'][:100]}...")
    
    # Initialize components
    yolo = YOLODetector(conf_threshold=0.3)
    localizer = DamageLocalizer(padding=0.2)
    classifier = ClassifierSecondOpinion()
    qwen = QwenPerception()
    
    image_paths = row["image_paths"].split(";")
    
    debug_results = {
        "user_id": user_id,
        "claim_object": row["claim_object"],
        "user_claim": row["user_claim"],
        "images": [],
    }
    
    vlm_best_findings = []  # Store VisualFindings objects for aggregation
    
    for img_idx, p in enumerate(image_paths):
        img_path = PROJECT_ROOT / p
        if not img_path.exists():
            img_path = PROJECT_ROOT / "dataset" / p
        
        if not img_path.exists():
            logger.warning(f"Image not found: {p}")
            continue
        
        img = Image.open(img_path).convert("RGB")
        img_info = {
            "path": str(img_path),
            "size": img.size,
            "yolo": {},
            "vlm": {},
            "classifier": {},
        }
        
        # --- YOLO Detection ---
        yolo_results = yolo.model(img, verbose=False)
        
        # Save YOLO debug image with boxes
        yolo_debug_path = output_dir / f"img_{img_idx+1}_yolo_boxes.jpg"
        draw_yolo_boxes(img, yolo_results, yolo_debug_path)
        
        # Get YOLO class and crop
        yolo_cls = yolo.detect_and_classify(img)
        crop, bbox = yolo.get_crop(img, padding=0.2)
        
        if crop is None:
            crop = img  # fallback to full image
        
        # Save YOLO crop
        crop.save(output_dir / f"img_{img_idx+1}_yolo_crop.jpg")
        
        img_info["yolo"] = {
            "detected_class": yolo_cls[0] if yolo_cls else None,
            "detected_conf": yolo_cls[1] if yolo_cls else None,
            "bbox": bbox,
        }
        
        # --- VLM on original + crops ---
        # VLM on original image
        vlm_orig = qwen._predict_single_image(img, row["user_claim"], row["claim_object"], img_idx)
        
        # VLM on YOLO crop
        vlm_crop = qwen._predict_single_image(crop, row["user_claim"], row["claim_object"], img_idx)
        
        # VLM on multi-crop (quadrants)
        if img.width > 512 or img.height > 512:
            crops = qwen._generate_crops(img)
            crop_findings = []
            for c_idx, c in enumerate(crops):
                c.save(output_dir / f"img_{img_idx+1}_crop_{c_idx}.jpg")
                try:
                    cf = qwen._predict_single_image(c, row["user_claim"], row["claim_object"], img_idx)
                    crop_findings.append(cf)
                except Exception as e:
                    logger.warning(f"Crop {c_idx} failed: {e}")
            
            best_crop = qwen._select_best_crop_finding(crop_findings) if crop_findings else vlm_orig
        else:
            best_crop = vlm_crop
        
        img_info["vlm"] = {
            "original": vlm_orig.model_dump(),
            "yolo_crop": vlm_crop.model_dump(),
            "best_crop": best_crop.model_dump(),
        }
        
        # Store VisualFindings object for aggregation
        vlm_best_findings.append(best_crop)
        
        # --- Classifier with TTA ---
        clf_pred = classifier.predict(img, use_tta=True)
        img_info["classifier"] = clf_pred
        
        debug_results["images"].append(img_info)
        
        logger.info(f"  Image {img_idx+1}: YOLO={yolo_cls}, VLM_orig={vlm_orig.visible_issue}, "
                    f"VLM_crop={vlm_crop.visible_issue}, VLM_best={best_crop.visible_issue}, "
                    f"CLF={clf_pred['issue_type']} (conf={clf_pred['confidence_issue']:.2f})")
    
    # Save debug JSON report
    with open(output_dir / "debug_report.json", "w") as f:
        json.dump(debug_results, f, indent=2, default=str)
    
    logger.info(f"Debug artifacts saved to {output_dir}")
    
    # Also run full pipeline for comparison
    logger.info("\n--- Full Pipeline Output ---")
    
    # Use best_crop findings
    aggregated = qwen._aggregate_findings(vlm_best_findings)
    
    if qwen.use_label_normalization and aggregated.visible_issue != "unknown":
        aggregated, _ = qwen._normalize_labels(aggregated, row["claim_object"])
    
    logger.info(f"Aggregated VLM: issue={aggregated.visible_issue}, part={aggregated.object_part}, "
                f"conf={aggregated.confidence:.2f}, severity={aggregated.severity}")
    
    # Evidence evaluation
    evidence = evaluate_evidence(
        findings=aggregated,
        claim_text=row["user_claim"],
        claim_object=row["claim_object"],
        user_history={},
        evidence_requirements="",
    )
    
    # Adjudication
    decision = adjudicate(
        findings=aggregated,
        evidence=evidence,
        claim_text=row["user_claim"],
    )
    
    logger.info(f"Evidence: standard_met={evidence.evidence_standard_met}, flags={evidence.risk_flags}")
    logger.info(f"Decision: {decision.claim_status}")
    logger.info(f"Justification: {decision.claim_status_justification}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", default="user_045", help="User ID to debug")
    parser.add_argument("--dataset", choices=["sample", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("debug"))
    args = parser.parse_args()
    
    claims_path = TEST_CLAIMS_PATH if args.dataset == "test" else SAMPLE_CLAIMS_PATH
    debug_claim(args.user_id, claims_path, args.output)
