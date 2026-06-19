"""Tests for configuration and constants."""

import pytest
from pathlib import Path

from hackerrank_orchestrate.config import (
    PROJECT_ROOT,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    CHECKPOINTS_DIR,
    OUTPUTS_DIR,
    OBJECT_TYPES,
    ISSUE_TYPES,
    OBJECT_PARTS,
    RISK_FLAGS,
    CLAIM_STATUS,
    SEVERITY,
    IMAGE_SIZE,
    BATCH_SIZE,
)


class TestPaths:
    def test_project_root_exists(self):
        assert PROJECT_ROOT.exists(), f"Project root does not exist: {PROJECT_ROOT}"

    def test_data_dirs_exist(self):
        for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CHECKPOINTS_DIR, OUTPUTS_DIR]:
            assert d.exists(), f"Directory does not exist: {d}"

    def test_dataset_paths_exist(self):
        from hackerrank_orchestrate.config import (
            EVIDENCE_REQUIREMENTS_PATH,
            USER_HISTORY_PATH,
            SAMPLE_CLAIMS_PATH,
            TEST_CLAIMS_PATH,
        )
        assert EVIDENCE_REQUIREMENTS_PATH.exists()
        assert USER_HISTORY_PATH.exists()
        assert SAMPLE_CLAIMS_PATH.exists()
        assert TEST_CLAIMS_PATH.exists()


class TestEnums:
    def test_object_types(self):
        assert set(OBJECT_TYPES) == {"car", "laptop", "package"}

    def test_issue_types(self):
        assert len(ISSUE_TYPES) == 12
        assert all(isinstance(t, str) for t in ISSUE_TYPES)
        assert "unknown" in ISSUE_TYPES
        assert "none" in ISSUE_TYPES

    def test_object_parts(self):
        for obj_type in OBJECT_TYPES:
            assert obj_type in OBJECT_PARTS
            assert "unknown" in OBJECT_PARTS[obj_type]

    def test_risk_flags(self):
        assert len(RISK_FLAGS) == 14
        assert "none" in RISK_FLAGS
        assert "manual_review_required" in RISK_FLAGS

    def test_claim_status(self):
        assert set(CLAIM_STATUS) == {"supported", "contradicted", "not_enough_information"}

    def test_severity(self):
        assert set(SEVERITY) == {"none", "low", "medium", "high", "unknown"}

    def test_image_size(self):
        assert isinstance(IMAGE_SIZE, int)
        assert IMAGE_SIZE > 0

    def test_batch_size(self):
        assert isinstance(BATCH_SIZE, int)
        assert BATCH_SIZE > 0
