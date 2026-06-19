#!/usr/bin/env python3
"""Run Qwen VLM inference on sample/test claims."""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from hackerrank_orchestrate.models.qwen_inference import QwenInference
from hackerrank_orchestrate.utils.logger import setup_logger
from hackerrank_orchestrate.config import SAMPLE_CLAIMS_PATH, TEST_CLAIMS_PATH, USER_HISTORY_PATH, EVIDENCE_REQUIREMENTS_PATH, OUTPUTS_DIR

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

    logger.info("Initializing Qwen model...")
    qwen = QwenInference()

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
            result = qwen.batch_predict([claim])[0]
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing claim {row['user_id']}: {e}")
            results.append(qwen._default_response())

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved results to {output_path}")


if __name__ == "__main__":
    # Run on sample claims first
    run_inference(SAMPLE_CLAIMS_PATH, OUTPUTS_DIR / "sample_predictions.csv")
    # Run on test claims
    run_inference(TEST_CLAIMS_PATH, OUTPUTS_DIR / "output.csv")
