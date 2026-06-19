"""PyTorch dataset and dataloader for training with augmentation and class balancing."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from hackerrank_orchestrate.config import IMAGE_SIZE, OBJECT_PARTS, OBJECT_TYPES, PROCESSED_DATA_DIR
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


def _build_label_encoders() -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
    """Build label encoders for object type, issue type, and object part."""
    object_type_encoder = {t: i for i, t in enumerate(OBJECT_TYPES)}
    issue_types = ["dent", "scratch", "crack", "glass_shatter", "broken_part",
                   "missing_part", "torn_packaging", "crushed_packaging",
                   "water_damage", "stain", "none", "unknown"]
    issue_type_encoder = {t: i for i, t in enumerate(issue_types)}
    
    # Build object part encoder (combined across all object types)
    all_parts = set()
    for parts in OBJECT_PARTS.values():
        all_parts.update(parts)
    object_part_encoder = {p: i for i, p in enumerate(sorted(all_parts))}
    
    return {
        "object_type": object_type_encoder,
        "issue_type": issue_type_encoder,
        "object_part": object_part_encoder,
    }, issue_type_encoder


LABEL_ENCODERS, _ = _build_label_encoders()


class DamageDataset(Dataset):
    """PyTorch dataset for multi-task damage classification with augmentation."""

    def __init__(
        self,
        records: List[Dict[str, Any]],
        transform: Optional[transforms.Compose] = None,
        training: bool = False,
    ) -> None:
        self.records = records
        self.training = training
        self.transform = transform or self._get_transform(training)

    @staticmethod
    def _get_transform(training: bool = False) -> transforms.Compose:
        """Get augmentation pipeline."""
        if training:
            return transforms.Compose([
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.1),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        record = self.records[idx]
        
        img = Image.open(record["image_path"]).convert("RGB")
        img_tensor = self.transform(img)

        labels = {
            "object_type": torch.tensor(
                LABEL_ENCODERS["object_type"][record["object_type"]], dtype=torch.long
            ),
            "issue_type": torch.tensor(
                LABEL_ENCODERS["issue_type"].get(record["issue_type"], 0), dtype=torch.long
            ),
            "object_part": torch.tensor(
                LABEL_ENCODERS["object_part"].get(record["object_part"], 0), dtype=torch.long
            ),
            "has_damage": torch.tensor(1.0 if record["has_damage"] else 0.0, dtype=torch.float32),
        }

        return img_tensor, labels

    @classmethod
    def from_split(cls, split_name: str, training: bool = False) -> "DamageDataset":
        """Load dataset from a processed split file."""
        split_path = PROCESSED_DATA_DIR / f"{split_name}.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        with open(split_path, "r") as f:
            records = json.load(f)

        logger.info(f"Loaded {len(records)} records from {split_name}")
        return cls(records, training=training)

    def compute_class_weights(self, label_key: str = "issue_type") -> torch.Tensor:
        """Compute inverse frequency weights for a label key."""
        labels = [LABEL_ENCODERS[label_key].get(r.get(label_key, "unknown"), 0) for r in self.records]
        counts = np.bincount(labels, minlength=len(LABEL_ENCODERS[label_key]))
        # Inverse frequency weighting
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * len(weights)  # normalize
        return torch.tensor(weights, dtype=torch.float32)

    def get_balanced_sampler(self, label_key: str = "object_type") -> WeightedRandomSampler:
        """Create a weighted sampler that balances classes."""
        labels = [LABEL_ENCODERS[label_key].get(r.get(label_key, "unknown"), 0) for r in self.records]
        counts = np.bincount(labels, minlength=len(LABEL_ENCODERS[label_key]))
        # Sample weight for each example
        sample_weights = 1.0 / (counts[labels] + 1e-6)
        return WeightedRandomSampler(sample_weights, len(self.records), replacement=True)


def create_dataloader(
    dataset: DamageDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    use_balanced_sampling: bool = False,
) -> torch.utils.data.DataLoader:
    """Create a PyTorch DataLoader."""
    sampler = None
    if use_balanced_sampling and dataset.training:
        sampler = dataset.get_balanced_sampler("issue_type")
        shuffle = False  # Cannot use shuffle with sampler
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
