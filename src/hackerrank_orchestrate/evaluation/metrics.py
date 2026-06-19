import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from hackerrank_orchestrate.config import SAMPLE_CLAIMS_PATH, OUTPUTS_DIR
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)

class Evaluator:
    def __init__(self):
        self.results = []

    def evaluate(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict[str, float]:
        metrics = {}
        
        # Convert to DataFrames for easier comparison
        pred_df = pd.DataFrame(predictions)
        gt_df = pd.DataFrame(ground_truth)
        
        # Ensure columns match
        required_cols = [
            "claim_status", "issue_type", "object_part", "evidence_standard_met",
            "valid_image", "severity", "risk_flags"
        ]
        
        for col in required_cols:
            if col in pred_df.columns and col in gt_df.columns:
                pred_vals = pred_df[col].astype(str)
                gt_vals = gt_df[col].astype(str)
                
                if col == "claim_status":
                    metrics["claim_status_acc"] = accuracy_score(gt_vals, pred_vals)
                elif col == "issue_type":
                    metrics["issue_type_acc"] = accuracy_score(gt_vals, pred_vals)
                elif col == "object_part":
                    metrics["object_part_acc"] = accuracy_score(gt_vals, pred_vals)
                elif col in ["evidence_standard_met", "valid_image"]:
                    metrics[f"{col}_acc"] = accuracy_score(gt_vals.str.lower(), pred_vals.str.lower())
                elif col == "severity":
                    metrics["severity_acc"] = accuracy_score(gt_vals, pred_vals)
                elif col == "risk_flags":
                    metrics["risk_flags_f1"] = self._compute_risk_flags_f1(gt_vals, pred_vals)
        
        # Overall accuracy (weighted average)
        metrics["overall_score"] = sum(
            metrics.get(k, 0) for k in ["claim_status_acc", "issue_type_acc", "object_part_acc", "evidence_standard_met_acc", "valid_image_acc"]
        ) / 5.0
        
        return metrics
    
    def _compute_risk_flags_f1(self, gt_series: pd.Series, pred_series: pd.Series) -> float:
        """Compute F1 score for multi-label risk flags."""
        from sklearn.preprocessing import MultiLabelBinarizer
        
        def parse_flags(s):
            if pd.isna(s) or s == "none":
                return []
            return [f.strip() for f in str(s).split(";")]
        
        gt_flags = gt_series.apply(parse_flags)
        pred_flags = pred_series.apply(parse_flags)
        
        mlb = MultiLabelBinarizer()
        gt_binary = mlb.fit_transform(gt_flags)
        pred_binary = mlb.transform(pred_flags)
        
        return f1_score(gt_binary, pred_binary, average="macro", zero_division=0)
    
    def generate_report(self, metrics: Dict, output_path: Path) -> None:
        report = f"""# Evaluation Report

## Metrics
- Claim Status Accuracy: {metrics.get('claim_status_acc', 0):.3f}
- Issue Type Accuracy: {metrics.get('issue_type_acc', 0):.3f}
- Object Part Accuracy: {metrics.get('object_part_acc', 0):.3f}
- Evidence Standard Met Accuracy: {metrics.get('evidence_standard_met_acc', 0):.3f}
- Valid Image Accuracy: {metrics.get('valid_image_acc', 0):.3f}
- Severity Accuracy: {metrics.get('severity_acc', 0):.3f}
- Risk Flags F1: {metrics.get('risk_flags_f1', 0):.3f}
- Overall Score: {metrics.get('overall_score', 0):.3f}

## Operational Analysis
- Model: Qwen2.5-VL-7B-Instruct
- Training: MobileNetV3-Large multi-task classifier
- Total samples evaluated: {len(self.results) if self.results else 'N/A'}
"""
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")


def main():
    """Run evaluation on sample claims."""
    from hackerrank_orchestrate.models.qwen_inference import QwenInference
    from hackerrank_orchestrate.data.dataset_loader import DamageDataset
    
    # Load sample claims
    claims_df = pd.read_csv(SAMPLE_CLAIMS_PATH)
    
    # Initialize Qwen
    qwen = QwenInference()
    
    # Process claims
    predictions = []
    for _, row in claims_df.iterrows():
        claim = {
            "user_id": row["user_id"],
            "image_paths": row["image_paths"],
            "user_claim": row["user_claim"],
            "claim_object": row["claim_object"],
            "user_history": {},  # Load from user_history.csv if needed
            "evidence_rules": "",  # Load from evidence_requirements.csv if needed
        }
        predictions.append(qwen.predict(claim))
    
    # Evaluate
    evaluator = Evaluator()
    metrics = evaluator.evaluate(predictions, claims_df.to_dict('records'))
    
    # Save report
    evaluator.generate_report(metrics, OUTPUTS_DIR / "evaluation_report.md")
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUTPUTS_DIR / "sample_predictions.csv", index=False)
    
    logger.info(f"Evaluation complete. Overall score: {metrics['overall_score']:.3f}")


if __name__ == "__main__":
    main()
