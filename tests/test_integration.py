"""Integration tests for the full three-stage pipeline."""

import pytest
from pathlib import Path

from hackerrank_orchestrate.perception import VisualFindings
from hackerrank_orchestrate.evidence_evaluator import evaluate_evidence
from hackerrank_orchestrate.adjudicator import adjudicate


class TestPipeline:
    def test_raw_data_exists(self):
        for obj_type in ["car", "laptop", "package"]:
            assert Path(f"data/raw/{obj_type}").exists(), f"Missing raw data: {obj_type}"

    def test_processed_data_exists(self):
        assert Path("data/processed/train.json").exists(), "Missing processed train data"
        assert Path("data/processed/val.json").exists(), "Missing processed val data"
        assert Path("data/processed/test.json").exists(), "Missing processed test data"

    def test_checkpoint_directory(self):
        Path("models/checkpoints").mkdir(parents=True, exist_ok=True)
        assert Path("models/checkpoints").exists()

    def test_output_directory(self):
        Path("outputs").mkdir(parents=True, exist_ok=True)
        assert Path("outputs").exists()

    def test_three_stage_pipeline_end_to_end(self):
        """Test the full three-stage pipeline on a synthetic claim."""
        # Stage 1: Perception
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="rear_bumper",
            severity="medium",
            confidence=0.8,
            supporting_image_ids=["img_1"],
            observations=["clear dent on rear bumper"],
            risk_flags=[],
        )

        # Stage 2: Evidence Evaluation
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="The back of the car has a dent now. It was not there before.",
            claim_object="car",
            user_history={"history_flags": "none"},
            evidence_requirements="",
        )
        assert evidence.evidence_standard_met is True
        assert "rear_bumper" in evidence.evidence_standard_met_reason

        # Stage 3: Adjudication
        decision = adjudicate(findings, evidence, "The back of the car has a dent")
        assert decision.claim_status == "supported"
        assert "directly show" in decision.claim_status_justification

    def test_pipeline_with_risk_flags(self):
        """Test pipeline when user has risk history."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="scratch",
            object_part="rear_bumper",
            severity="low",
            confidence=0.8,
            supporting_image_ids=["img_1"],
            observations=["minor scratch on rear bumper"],
            risk_flags=[],
        )

        evidence = evaluate_evidence(
            findings=findings,
            claim_text="The back looks pretty bad to me, so I uploaded both photos.",
            claim_object="car",
            user_history={"history_flags": "user_history_risk"},
            evidence_requirements="",
        )
        assert "user_history_risk" in evidence.risk_flags
        assert "claim_mismatch" in evidence.risk_flags  # exaggeration detected

        decision = adjudicate(findings, evidence, "The back looks pretty bad")
        assert decision.claim_status == "contradicted"  # mismatch + exaggeration

    def test_pipeline_insufficient_evidence(self):
        """Test pipeline when images are not usable."""
        findings = VisualFindings(
            valid_image=False,
            visible_issue="unknown",
            object_part="unknown",
            severity="unknown",
            confidence=0.0,
            supporting_image_ids=[],
            observations=[],
            risk_flags=["blurry_image"],
        )

        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a dent on my hood",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert evidence.evidence_standard_met is False

        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "not_enough_information"

    def test_pipeline_wrong_object(self):
        """Test pipeline when images show wrong object."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.7,
            supporting_image_ids=["img_1"],
            observations=["dent visible"],
            risk_flags=["wrong_object"],
        )

        evidence = evaluate_evidence(
            findings=findings,
            claim_text="My car has a dent",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert evidence.evidence_standard_met is False
        assert "different object" in evidence.evidence_standard_met_reason

    def test_sample_claims_csv_exists(self):
        assert Path("dataset/sample_claims.csv").exists(), "Missing sample claims CSV"

    def test_user_history_csv_exists(self):
        assert Path("dataset/user_history.csv").exists(), "Missing user history CSV"

    def test_evidence_requirements_csv_exists(self):
        assert Path("dataset/evidence_requirements.csv").exists(), "Missing evidence requirements CSV"
