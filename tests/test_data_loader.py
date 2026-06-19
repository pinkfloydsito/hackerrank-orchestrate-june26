"""Tests for data loading and dataset creation."""

import pytest
import torch
from pathlib import Path

from hackerrank_orchestrate.data.dataset_loader import DamageDataset, create_dataloader
from hackerrank_orchestrate.config import IMAGE_SIZE, BATCH_SIZE


class TestDamageDataset:
    def test_from_split(self):
        # This test requires preprocessed data to exist
        if not Path("data/processed/train.json").exists():
            pytest.skip("Preprocessed data not available")
        
        dataset = DamageDataset.from_split("train")
        assert len(dataset) > 0
        
        img, labels = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert isinstance(labels, dict)
        assert "object_type" in labels
        assert "issue_type" in labels
        assert "object_part" in labels
        assert "has_damage" in labels

    def test_empty_records(self):
        empty_dataset = DamageDataset([])
        assert len(empty_dataset) == 0

    def test_single_record(self):
        record = {
            "image_id": 0,
            "image_path": "data/processed/images/car/000000.jpg",
            "object_type": "car",
            "issue_type": "dent",
            "object_part": "hood",
            "has_damage": True,
        }
        if not Path(record["image_path"]).exists():
            pytest.skip("Image file not available")
        
        dataset = DamageDataset([record])
        img, labels = dataset[0]
        assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert labels["object_type"] == 0  # car


class TestDataloader:
    def test_create_dataloader(self):
        if not Path("data/processed/train.json").exists():
            pytest.skip("Preprocessed data not available")
        
        dataset = DamageDataset.from_split("train")
        if len(dataset) == 0:
            pytest.skip("Dataset is empty")
        
        loader = create_dataloader(dataset, batch_size=min(4, len(dataset)), shuffle=True)
        batch = next(iter(loader))
        images, labels = batch
        
        assert images.shape[0] <= 4
        assert images.shape[1:] == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert isinstance(labels, dict)
