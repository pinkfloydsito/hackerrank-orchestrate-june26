#!/usr/bin/env python3
"""Evaluate predictions on sample claims against ground truth."""

import pandas as pd
from hackerrank_orchestrate.evaluation.metrics import Evaluator
from hackerrank_orchestrate.config import SAMPLE_CLAIMS_PATH, OUTPUTS_DIR
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    logger.info("Loading ground truth and predictions...")
    gt_df = pd.read_csv(SAMPLE_CLAIMS_PATH)
    pred_df = pd.read_csv(OUTPUTS_DIR / "sample_predictions.csv")

    evaluator = Evaluator()
    metrics = evaluator.evaluate(pred_df.to_dict('records'), gt_df.to_dict('records'))

    evaluator.generate_report(metrics, OUTPUTS_DIR / "evaluation_report.md")

    logger.info("Evaluation complete.")
    logger.info(f"Overall Score: {metrics['overall_score']:.3f}")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.3f}")


if __name__ == "__main__":
    main()
