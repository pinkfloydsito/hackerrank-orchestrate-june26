#!/usr/bin/env python3
"""Run three-stage pipeline inference on sample/test claims.

Stages:
1. Perception (Qwen VLM) - visual fact extraction
2. Evidence Evaluation (deterministic rules) - evidence sufficiency
3. Adjudication (rule engine) - claim status decision
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from hackerrank_orchestrate.perception import QwenPerception
from hackerrank_orchestrate.evidence_evaluator import evaluate_evidence
from hackerrank_orchestrate.adjudicator import adjudicate
from hackerrank_orchestrate.utils.logger import setup_logger
from hackerrank_orchestrate.config import (
    SAMPLE_CLAIMS_PATH, TEST_CLAIMS_PATH, USER_HISTORY_PATH,
    EVIDENCE_REQUIREMENTS_PATH, OUTPUTS_DIR
)

logger = setup_logger(__name__)


def load_user_history() -> dict:
    df = pd.read_csv(USER_HISTORY_PATH)
    return {row["user_id"]: row.to_dict() for _, row in df.iterrows()}


def load_evidence_requirements() -> str:
    df = pd.read_csv(EVIDENCE_REQUIREMENTS_PATH)
    return df.to_string(index=False)


def run_inference(input_path: Path, output_path: Path) -> None:
    logger.info(f"Loading claims from {input_path}")
    claims_df = pd.read_csv(input_path)
    user_history = load_user_history()
    evidence_rules = load_evidence_requirements()

    logger.info("Initializing Qwen perception model...")
    qwen = QwenPerception()

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
            # --- Stage 1: Perception (Qwen VLM) ---
            perceptions = qwen.batch_predict([claim])
            findings = perceptions[0]

            # --- Stage 2: Evidence Evaluation (deterministic rules) ---
            evidence = evaluate_evidence(
                findings=findings,
                claim_text=claim["user_claim"],
                claim_object=claim["claim_object"],
                user_history=claim["user_history"],
                evidence_requirements=claim["evidence_rules"],
            )

            # --- Stage 3: Adjudication (rule engine) ---
            decision = adjudicate(
                findings=findings,
                evidence=evidence,
                claim_text=claim["user_claim"],
            )

            # --- Format output to match expected CSV schema ---
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
            # Default fallback
            results.append({
                "user_id": row["user_id"],
                "image_paths": row["image_paths"],
                "user_claim": row["user_claim"],
                "claim_object": row["claim_object"],
                "evidence_standard_met": "false",
                "evidence_standard_met_reason": f"Pipeline error: {str(e)}",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "Pipeline failed to process this claim.",
                "supporting_image_ids": "none",
                "valid_image": "false",
                "severity": "unknown",
            })

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")
    logger.info(f"Processed {len(results_df)} claims")

    # Summary stats
    status_counts = results_df["claim_status"].value_counts()
    logger.info(f"Claim status distribution: {status_counts.to_dict()}")


def main():
    parser = argparse.ArgumentParser(description="Run three-stage pipeline on claims")
    parser.add_argument("--dataset", choices=["sample", "test"], default="sample",
                        help="Which dataset to run inference on")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: auto-generated)")
    args = parser.parse_args()

    if args.dataset == "sample":
        input_path = SAMPLE_CLAIMS_PATH
        output_path = args.output or (OUTPUTS_DIR / "sample_predictions.csv")
    else:
        input_path = TEST_CLAIMS_PATH
        output_path = args.output or (OUTPUTS_DIR / "output.csv")

    run_inference(input_path, output_path)


if __name__ == "__main__":
    main()
