"""Tests for Qwen perception JSON parsing and response handling."""

import pytest
import json
from pydantic import ValidationError

from hackerrank_orchestrate.perception import QwenPerception, VisualFindings


class TestVisualFindings:
    def test_valid_response(self):
        data = {
            "valid_image": True,
            "visible_issue": "dent",
            "object_part": "hood",
            "severity": "medium",
            "confidence": 0.8,
            "supporting_image_ids": ["img_1"],
            "observations": ["clear dent on hood"],
            "risk_flags": [],
        }
        response = VisualFindings(**data)
        assert response.valid_image is True
        assert response.visible_issue == "dent"
        assert response.severity == "medium"

    def test_invalid_issue_type(self):
        data = {
            "valid_image": True,
            "visible_issue": "invalid_type",
            "object_part": "hood",
            "severity": "medium",
        }
        # Our validator is forgiving and defaults to 'unknown'
        response = VisualFindings(**data)
        assert response.visible_issue == "unknown"

    def test_invalid_severity(self):
        data = {
            "valid_image": True,
            "visible_issue": "dent",
            "object_part": "hood",
            "severity": "extreme",
        }
        # Our validator is forgiving and defaults to 'unknown'
        response = VisualFindings(**data)
        assert response.severity == "unknown"

    def test_missing_field(self):
        data = {
            "valid_image": True,
            "visible_issue": "dent",
            # missing other required fields
        }
        # Missing required fields still raise ValidationError
        with pytest.raises(ValidationError):
            VisualFindings(**data)


class TestQwenPerception:
    def test_default_response(self):
        qwen = QwenPerception.__new__(QwenPerception)
        response = qwen._default_response()
        assert response["valid_image"] is False
        assert response["visible_issue"] == "unknown"
        assert response["confidence"] == 0.0

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
        # Should extract first valid JSON
        assert result == {"a": 1}

    def test_parse_response_with_string_fields(self):
        """Test parsing when model returns strings instead of lists."""
        qwen = QwenPerception.__new__(QwenPerception)
        text = json.dumps({
            "valid_image": True,
            "visible_issue": "dent",
            "object_part": "hood",
            "severity": "medium",
            "confidence": 0.8,
            "supporting_image_ids": "img_1;img_2",
            "risk_flags": "blurry_image",
            "observations": "dent on hood",
        })
        result = qwen._parse_response(text)
        assert result.supporting_image_ids == ["img_1", "img_2"]
        assert result.risk_flags == ["blurry_image"]
        assert result.observations == ["dent on hood"]
