#!/usr/bin/env python3
"""Integrated inference pipeline with localization, VLM, and classifier ensemble.

Stages:
1. Localization: Crop damage regions using preprocessed bounding boxes
2. Perception (VLM): Extract visual facts from cropped images
3. Classifier: MobileNet provides second opinion on damage type
4. Cross-check: Ensemble VLM + classifier predictions
5. Evidence Evaluation: Deterministic rules for evidence sufficiency
6. Adjudication: Rule-based claim status decision
"""

import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from hackerrank_orchestrate.perception import QwenPerception, VisualFindings
from hackerrank_orchestrate.localization import DamageLocalizer
from hackerrank_orchestrate.classifier_integration import ClassifierSecondOpinion
from hackerrank_orchestrate.evidence_evaluator import evaluate_evidence
from hackerrank_orchestrate.adjudicator import adjudicate
from hackerrank_orchestrate.utils.logger import setup_logger
from hackerrank_orchestrate.config import (
    SAMPLE_CLAIMS_PATH, TEST_CLAIMS_PATH, USER_HISTORY_PATH,
    EVIDENCE_REQUIREMENTS_PATH, OUTPUTS_DIR, PROJECT_ROOT
)

logger = setup_logger(__name__)


def load_user_history() -> dict:
    df = pd.read_csv(USER_HISTORY_PATH)
    return {row["user_id"]: row.to_dict() for _, row in df.iterrows()}


def load_evidence_requirements() -> str:
    df = pd.read_csv(EVIDENCE_REQUIREMENTS_PATH)
    return df.to_string(index=False)


def resolve_image_path(p: str) -> Path:
    """Resolve image path relative to project root."""
    p = p.strip()
    if not p:
        return None
    path_candidates = [
        PROJECT_ROOT / p,
        PROJECT_ROOT / "dataset" / p,
    ]
    for path_obj in path_candidates:
        if path_obj.exists():
            return path_obj
    return None


def run_integrated_inference(input_path: Path, output_path: Path, use_classifier: bool = True, use_localization: bool = True) -> None:
    """Run integrated inference pipeline."""
    logger.info(f"Loading claims from {input_path}")
    claims_df = pd.read_csv(input_path)
    user_history = load_user_history()
    evidence_rules = load_evidence_requirements()
    
    # Initialize components
    logger.info("Initializing pipeline components...")
    qwen = QwenPerception()
    localizer = DamageLocalizer(padding=0.2) if use_localization else None
    classifier = ClassifierSecondOpinion() if use_classifier else None
    
    results = []
    for idx, row in tqdm(claims_df.iterrows(), total=len(claims_df)):
        claim = {
            "user_id": row["user_id"],
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
            "user_history": user_history.get(row["user_id"], {}),
            "evidence_rules": evidence_rules,
        }
        
        try:
            # --- Stage 1: Localization (crop damage regions) ---
            image_paths = claim["image_paths"].split(";")
            images = []
            crops = []
            for p in image_paths:
                path_obj = resolve_image_path(p)
                if path_obj is None:
                    continue
                
                if use_localization and localizer is not None:
                    crop, bbox = localizer.get_crop(str(path_obj))
                    crops.append(crop)
                    images.append(crop)  # Send cropped image to VLM
                else:
                    img = Image.open(path_obj).convert("RGB")
                    images.append(img)
                    crops.append(img)
            
            if not images:
                logger.warning(f"No valid images for claim {row['user_id']}")
                results.append(_fallback_result(claim))
                continue
            
            # --- Stage 2: Perception (VLM on cropped images) ---
            findings = qwen.predict(images, claim["user_claim"], claim["claim_object"])
            
            # --- Stage 3: Classifier Second Opinion (on cropped images) ---
            if use_classifier and classifier is not None and crops:
                # Run classifier on each cropped image
                clf_predictions = []
                for crop in crops:
                    try:
                        pred = classifier.predict(crop)
                        clf_predictions.append(pred)
                    except Exception as e:
                        logger.error(f"Classifier error: {e}")
                
                if clf_predictions:
                    # Aggregate classifier predictions (take highest confidence)
                    best_clf = max(clf_predictions, key=lambda x: x["confidence_issue"])
                    
                    # Cross-check with VLM
                    final_issue, final_part, ensemble_conf, reason = classifier.cross_check(
                        findings.visible_issue,
                        findings.object_part,
                        crops[0],  # Use first crop for cross-check
                        confidence_threshold=0.75,
                    )
                    
                    # Update findings with ensemble decision
                    findings.visible_issue = final_issue
                    findings.object_part = final_part
                    findings.confidence = ensemble_conf
                    
                    # Calibrate severity based on classifier
                    if findings.visible_issue != "none" and findings.visible_issue != "unknown":
                        findings.severity = classifier.calibrate_severity(
                            ensemble_conf, 
                            best_clf["has_damage"]
                        )
                    
                    logger.info(f"Claim {row['user_id']}: {reason}")
            
            # --- Stage 4: Evidence Evaluation ---
            evidence = evaluate_evidence(
                findings=findings,
                claim_text=claim["user_claim"],
                claim_object=claim["claim_object"],
                user_history=claim["user_history"],
                evidence_requirements=claim["evidence_rules"],
            )
            
            # --- Stage 5: Adjudication ---
            decision = adjudicate(
                findings=findings,
                evidence=evidence,
                claim_text=claim["user_claim"],
            )
            
            # --- Format output ---
            result = {
                "user_id": row["user_id"],
                "image_paths": row["image_paths"],
                "user_claim": row["user_claim"],
                "claim_object": row["claim_object"],
                "evidence_standard_met": str(evidence.evidence_standard_met).lower(),
                "evidence_standard_met_reason": evidence.evidence_standard_met_reason,
                "risk_flags": ";".join(evidence.risk_flags) if evidence.risk_flags else "none",
                "issue_type": findings.visible_issue,
                "object_part": findings.object_part,
                "claim_status": decision.claim_status,
                "claim_status_justification": decision.claim_status_justification,
                "supporting_image_ids": ";".join(findings.supporting_image_ids) if findings.supporting_image_ids else "none",
                "valid_image": str(findings.valid_image).lower(),
                "severity": findings.severity,
            }
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error processing claim {row['user_id']}: {e}")
            results.append(_fallback_result(claim))
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")
    
    # Summary
    status_counts = results_df["claim_status"].value_counts()
    logger.info(f"Claim status distribution: {status_counts.to_dict()}")


def _fallback_result(claim: dict) -> dict:
    """Generate fallback result for failed claims."""
    return {
        "user_id": claim["user_id"],
        "image_paths": claim["image_paths"],
        "user_claim": claim["user_claim"],
        "claim_object": claim["claim_object"],
        "evidence_standard_met": "false",
        "evidence_standard_met_reason": "Image processing failed",
        "risk_flags": "manual_review_required",
        "issue_type": "unknown",
        "object_part": "unknown",
        "claim_status": "not_enough_information",
        "claim_status_justification": "Failed to process images for this claim.",
        "supporting_image_ids": "none",
        "valid_image": "false",
        "severity": "unknown",
    }


def main():
    parser = argparse.ArgumentParser(description="Run integrated inference pipeline")
    parser.add_argument("--dataset", choices=["sample", "test"], default="sample",
                        help="Which dataset to run inference on")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: auto-generated)")
    parser.add_argument("--no-classifier", action="store_true",
                        help="Disable MobileNet classifier second opinion")
    parser.add_argument("--no-localization", action="store_true",
                        help="Disable damage localization cropping")
    args = parser.parse_args()
    
    if args.dataset == "sample":
        input_path = SAMPLE_CLAIMS_PATH
        output_path = args.output or (OUTPUTS_DIR / "sample_predictions_integrated.csv")
    else:
        input_path = TEST_CLAIMS_PATH
        output_path = args.output or (OUTPUTS_DIR / "output_integrated.csv")
    
    run_integrated_inference(
        input_path=input_path,
        output_path=output_path,
        use_classifier=not args.no_classifier,
        use_localization=not args.no_localization,
    )


if __name__ == "__main__":
    main()
