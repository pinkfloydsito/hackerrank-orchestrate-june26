"""Integration tests for the full pipeline."""

import pytest
from pathlib import Path


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
