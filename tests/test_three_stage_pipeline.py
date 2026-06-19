"""Tests for the three-stage pipeline: perception, evidence evaluation, adjudication."""

import pytest
from pydantic import ValidationError

from hackerrank_orchestrate.perception import VisualFindings, QwenPerception
from hackerrank_orchestrate.evidence_evaluator import evaluate_evidence, EvidenceEvaluation, _extract_claimed_issue
from hackerrank_orchestrate.adjudicator import adjudicate, ClaimDecision


class TestVisualFindings:
    def test_valid_perception(self):
        data = {
            "valid_image": True,
            "visible_issue": "dent",
            "object_part": "hood",
            "severity": "medium",
            "confidence": 0.8,
            "supporting_image_ids": ["img_1"],
            "observations": ["visible dent on hood"],
            "risk_flags": [],
        }
        findings = VisualFindings(**data)
        assert findings.valid_image is True
        assert findings.visible_issue == "dent"
        assert findings.confidence == 0.8
        assert findings.supporting_image_ids == ["img_1"]

    def test_invalid_issue_type_defaults(self):
        data = {
            "valid_image": True,
            "visible_issue": "invalid_type",
            "object_part": "hood",
            "severity": "medium",
        }
        findings = VisualFindings(**data)
        assert findings.visible_issue == "unknown"

    def test_invalid_severity_defaults(self):
        data = {
            "valid_image": True,
            "visible_issue": "dent",
            "object_part": "hood",
            "severity": "extreme",
        }
        findings = VisualFindings(**data)
        assert findings.severity == "unknown"

    def test_string_risk_flags_parsing(self):
        """Test that string risk flags are validated."""
        data = {
            "valid_image": True,
            "visible_issue": "dent",
            "object_part": "hood",
            "severity": "medium",
            "risk_flags": ["blurry_image", "wrong_angle", "invalid_flag"],
        }
        findings = VisualFindings(**data)
        # invalid_flag should be skipped by validator
        assert "blurry_image" in findings.risk_flags
        assert "wrong_angle" in findings.risk_flags
        assert "invalid_flag" not in findings.risk_flags

    def test_confidence_bounds(self):
        """Test confidence is constrained to 0-1."""
        with pytest.raises(ValidationError):
            VisualFindings(
                valid_image=True,
                visible_issue="dent",
                object_part="hood",
                severity="medium",
                confidence=1.5,
            )

    def test_default_values(self):
        """Test that optional fields have sensible defaults."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
        )
        assert findings.confidence == 0.5
        assert findings.supporting_image_ids == []
        assert findings.observations == []
        assert findings.risk_flags == []


class TestEvidenceEvaluation:
    def test_valid_image_sufficient_evidence(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
            supporting_image_ids=["img_1"],
            observations=["clear dent on hood"],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a dent on my hood",
            claim_object="car",
            user_history={"history_flags": "none"},
            evidence_requirements="",
        )
        assert evidence.evidence_standard_met is True
        assert "hood" in evidence.evidence_standard_met_reason
        assert "dent" in evidence.evidence_standard_met_reason

    def test_invalid_image_insufficient(self):
        findings = VisualFindings(
            valid_image=False,
            visible_issue="unknown",
            object_part="unknown",
            severity="unknown",
            confidence=0.0,
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a dent on my hood",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert evidence.evidence_standard_met is False
        assert "not usable" in evidence.evidence_standard_met_reason

    def test_low_confidence_rejected(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.15,  # Below new 0.2 threshold
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a dent on my hood",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert evidence.evidence_standard_met is False
        assert "confidence is very low" in evidence.evidence_standard_met_reason

    def test_low_confidence_but_not_rejected(self):
        """Confidence 0.25 is above the new 0.2 threshold, so should pass."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.25,
            supporting_image_ids=["img_1"],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a dent on my hood",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert evidence.evidence_standard_met is True

    def test_wrong_object_rejected(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
            risk_flags=["wrong_object"],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a dent on my hood",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert evidence.evidence_standard_met is False
        assert "different object" in evidence.evidence_standard_met_reason

    def test_user_history_risk_added(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
            supporting_image_ids=["img_1"],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a dent on my hood",
            claim_object="car",
            user_history={"history_flags": "user_history_risk"},
            evidence_requirements="",
        )
        assert "user_history_risk" in evidence.risk_flags

    def test_claim_mismatch_detected(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="scratch",
            object_part="hood",
            severity="low",
            confidence=0.8,
            supporting_image_ids=["img_1"],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a big dent on my hood",  # claimed dent, visible scratch
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert "claim_mismatch" in evidence.risk_flags

    def test_no_supporting_images_advisory(self):
        """Supporting image check is now advisory, not a hard gate."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
            supporting_image_ids=[],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is a dent on my hood",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        # Should pass through now (advisory only)
        assert evidence.evidence_standard_met is True
        assert "No specific image" in evidence.evidence_standard_met_reason

    def test_exaggeration_detected(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="scratch",
            object_part="rear_bumper",
            severity="low",
            confidence=0.8,
            supporting_image_ids=["img_1"],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="The back looks pretty bad to me",  # exaggerated language
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert "claim_mismatch" in evidence.risk_flags
        assert "exaggerated" in evidence.evidence_standard_met_reason or "severe" in evidence.evidence_standard_met_reason

    def test_exaggeration_with_severe_keyword(self):
        """Test that 'severe' keyword triggers exaggeration detection."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="scratch",
            object_part="hood",
            severity="low",
            confidence=0.8,
            supporting_image_ids=["img_1"],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is severe damage on my hood",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert "claim_mismatch" in evidence.risk_flags

    def test_no_exaggeration_when_severity_matches(self):
        """Medium severity with 'severe' language should not trigger exaggeration."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
            supporting_image_ids=["img_1"],
        )
        evidence = evaluate_evidence(
            findings=findings,
            claim_text="There is severe damage on my hood",
            claim_object="car",
            user_history={},
            evidence_requirements="",
        )
        assert "claim_mismatch" not in evidence.risk_flags


class TestAdjudication:
    def test_insufficient_evidence(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=False,
            evidence_standard_met_reason="Image is blurry",
            risk_flags=["blurry_image"],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "not_enough_information"
        assert "not provide sufficient evidence" in decision.claim_status_justification

    def test_supported_claim(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
            supporting_image_ids=["img_1"],
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible and the dent can be verified",
            risk_flags=[],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "supported"
        assert "directly show" in decision.claim_status_justification

    def test_contradicted_mismatch(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="scratch",
            object_part="hood",
            severity="low",
            confidence=0.8,
            supporting_image_ids=["img_1"],
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible",
            risk_flags=["claim_mismatch"],
        )
        decision = adjudicate(findings, evidence, "There is a big dent on my hood")
        assert decision.claim_status == "contradicted"
        assert "contradicts" in decision.claim_status_justification

    def test_contradicted_mismatch_lower_confidence(self):
        """Claim mismatch now triggers at confidence > 0.5 (was 0.6)."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="scratch",
            object_part="hood",
            severity="low",
            confidence=0.55,
            supporting_image_ids=["img_1"],
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible",
            risk_flags=["claim_mismatch"],
        )
        decision = adjudicate(findings, evidence, "There is a big dent on my hood")
        assert decision.claim_status == "contradicted"

    def test_contradicted_no_damage(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="none",
            object_part="hood",
            severity="none",
            confidence=0.9,
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible",
            risk_flags=[],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "contradicted"
        assert "shows no damage" in decision.claim_status_justification

    def test_contradicted_no_damage_lower_threshold(self):
        """No damage now triggers at confidence > 0.35 (was 0.6)."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="none",
            object_part="hood",
            severity="none",
            confidence=0.4,
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible",
            risk_flags=[],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "contradicted"

    def test_contradicted_wrong_object(self):
        """Wrong object should now return contradicted, not not_enough_information."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=False,
            evidence_standard_met_reason="The images appear to show a different object than claimed.",
            risk_flags=["wrong_object"],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "contradicted"
        assert "different object" in decision.claim_status_justification

    def test_user_history_risk_but_supported(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.8,
            supporting_image_ids=["img_1"],
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible and the dent can be verified",
            risk_flags=["user_history_risk"],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "supported"
        assert "User history adds risk context" in decision.claim_status_justification

    def test_low_confidence_not_enough_info(self):
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.3,  # Below new 0.35 threshold
            supporting_image_ids=["img_1"],
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible",
            risk_flags=[],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "not_enough_information"
        assert "confidence is too low" in decision.claim_status_justification

    def test_supported_with_lower_confidence(self):
        """Supported now triggers at confidence > 0.35 (was 0.40)."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="dent",
            object_part="hood",
            severity="medium",
            confidence=0.38,
            supporting_image_ids=["img_1"],
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible and the dent can be verified",
            risk_flags=[],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "supported"

    def test_unknown_issue_not_enough_info(self):
        """Unknown issue with low confidence should return not_enough_information."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="unknown",
            object_part="hood",
            severity="unknown",
            confidence=0.4,
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible",
            risk_flags=[],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "not_enough_information"

    def test_unknown_issue_contradicted(self):
        """Unknown issue with high confidence should return contradicted."""
        findings = VisualFindings(
            valid_image=True,
            visible_issue="unknown",
            object_part="hood",
            severity="unknown",
            confidence=0.7,
        )
        evidence = EvidenceEvaluation(
            evidence_standard_met=True,
            evidence_standard_met_reason="The hood is visible",
            risk_flags=[],
        )
        decision = adjudicate(findings, evidence, "There is a dent on my hood")
        assert decision.claim_status == "contradicted"


class TestClaimedIssueExtraction:
    def test_extract_dent(self):
        assert _extract_claimed_issue("There is a dent on my hood") == "dent"

    def test_extract_scratch(self):
        assert _extract_claimed_issue("My car got scratched") == "scratch"

    def test_extract_crack(self):
        assert _extract_claimed_issue("The screen is cracked") == "crack"

    def test_extract_water_damage(self):
        assert _extract_claimed_issue("Water damage on the floor") == "water_damage"

    def test_no_issue_found(self):
        assert _extract_claimed_issue("Hello, I have a problem") == ""


class TestQwenPerceptionMock:
    """Test QwenPerception methods without loading the model."""

    def test_extract_json_valid(self):
        qwen = QwenPerception.__new__(QwenPerception)
        text = 'Some text {"key": "value"} more text'
        result = qwen._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_no_json(self):
        qwen = QwenPerception.__new__(QwenPerception)
        text = "No JSON here at all"
        result = qwen._extract_json(text)
        assert result is None

    def test_extract_json_multiple(self):
        qwen = QwenPerception.__new__(QwenPerception)
        text = '{"a": 1} {"b": 2}'
        result = qwen._extract_json(text)
        assert result == {"a": 1}

    def test_parse_response_with_string_lists(self):
        """Test parsing when model returns string instead of list."""
        qwen = QwenPerception.__new__(QwenPerception)
        text = '{"valid_image": true, "visible_issue": "dent", "object_part": "hood", "severity": "medium", "supporting_image_ids": "img_1;img_2", "risk_flags": "blurry_image", "observations": "dent on hood", "confidence": 0.8}'
        result = qwen._parse_response(text)
        assert result.supporting_image_ids == ["img_1", "img_2"]
        assert result.risk_flags == ["blurry_image"]
        assert result.observations == ["dent on hood"]

    def test_default_response(self):
        qwen = QwenPerception.__new__(QwenPerception)
        defaults = qwen._default_response()
        assert defaults["valid_image"] is False
        assert defaults["visible_issue"] == "unknown"
        assert defaults["confidence"] == 0.0
