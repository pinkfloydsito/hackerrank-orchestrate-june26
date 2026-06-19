#!/usr/bin/env python3
"""Evaluate predictions on sample claims against ground truth."""

import argparse
import pandas as pd
from pathlib import Path
from hackerrank_orchestrate.evaluation.metrics import Evaluator
from hackerrank_orchestrate.config import SAMPLE_CLAIMS_PATH, OUTPUTS_DIR
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against ground truth")
    parser.add_argument("--predictions", type=Path, default=OUTPUTS_DIR / "sample_predictions.csv",
                        help="Path to predictions CSV")
    parser.add_argument("--ground-truth", type=Path, default=SAMPLE_CLAIMS_PATH,
                        help="Path to ground truth CSV")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "evaluation_report.md",
                        help="Path to write evaluation report")
    args = parser.parse_args()

    logger.info("Loading ground truth and predictions...")
    gt_df = pd.read_csv(args.ground_truth)
    pred_df = pd.read_csv(args.predictions)

    evaluator = Evaluator()
    metrics = evaluator.evaluate(pred_df.to_dict('records'), gt_df.to_dict('records'))

    evaluator.generate_report(metrics, args.output)

    logger.info("Evaluation complete.")
    logger.info(f"Overall Score: {metrics['overall_score']:.3f}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
