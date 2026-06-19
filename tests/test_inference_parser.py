"""Tests for Qwen inference JSON parsing and response handling."""

import pytest
import json
from pydantic import ValidationError

from hackerrank_orchestrate.models.qwen_inference import QwenInference, ClaimAnalysisResponse


class TestClaimAnalysisResponse:
    def test_valid_response(self):
        data = {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "Image shows clear dent",
            "risk_flags": "none",
            "issue_type": "dent",
            "object_part": "hood",
            "claim_status": "supported",
            "claim_status_justification": "The image clearly shows a dent on the hood",
            "supporting_image_ids": "img_1",
            "valid_image": True,
            "severity": "medium",
        }
        response = ClaimAnalysisResponse(**data)
        assert response.evidence_standard_met is True
        assert response.issue_type == "dent"
        assert response.claim_status == "supported"

    def test_invalid_issue_type(self):
        data = {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "test",
            "risk_flags": "none",
            "issue_type": "invalid_type",
            "object_part": "hood",
            "claim_status": "supported",
            "claim_status_justification": "test",
            "supporting_image_ids": "none",
            "valid_image": True,
            "severity": "medium",
        }
        # Our validator is forgiving and defaults to 'unknown'
        response = ClaimAnalysisResponse(**data)
        assert response.issue_type == "unknown"

    def test_invalid_claim_status(self):
        data = {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "test",
            "risk_flags": "none",
            "issue_type": "dent",
            "object_part": "hood",
            "claim_status": "maybe",
            "claim_status_justification": "test",
            "supporting_image_ids": "none",
            "valid_image": True,
            "severity": "medium",
        }
        # Our validator is forgiving and defaults to 'not_enough_information'
        response = ClaimAnalysisResponse(**data)
        assert response.claim_status == "not_enough_information"

    def test_missing_field(self):
        data = {
            "evidence_standard_met": True,
            "issue_type": "dent",
            # missing other required fields
        }
        # Missing required fields still raise ValidationError
        with pytest.raises(ValidationError):
            ClaimAnalysisResponse(**data)


class TestQwenInference:
    def test_default_response(self):
        qwen = QwenInference.__new__(QwenInference)
        response = qwen._default_response()
        assert response["claim_status"] == "not_enough_information"
        assert response["issue_type"] == "unknown"
        assert response["valid_image"] is False

    def test_extract_json_valid(self):
        qwen = QwenInference.__new__(QwenInference)
        text = 'Some text {"key": "value"} more text'
        result = qwen._extract_json(text)
        assert result == {"key": "value"}

    def test_extract_json_no_json(self):
        qwen = QwenInference.__new__(QwenInference)
        text = "No JSON here at all"
        result = qwen._extract_json(text)
        assert result is None

    def test_extract_json_multiple(self):
        qwen = QwenInference.__new__(QwenInference)
        text = '{"a": 1} {"b": 2}'
        result = qwen._extract_json(text)
        # Should extract first valid JSON
        assert result == {"a": 1}
