"""Localization module for cropping damage regions from images."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from hackerrank_orchestrate.config import PROCESSED_DATA_DIR
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


class DamageLocalizer:
    """Localize damage regions using preprocessed bounding boxes."""

    def __init__(self, padding: float = 0.2) -> None:
        self.padding = padding
        self._records = self._load_all_records()

    def _load_all_records(self) -> List[Dict]:
        """Load all preprocessed records with bounding boxes."""
        all_records = []
        for split_file in ["train.json", "val.json", "test.json"]:
            split_path = PROCESSED_DATA_DIR / split_file
            if split_path.exists():
                with open(split_path, "r") as f:
                    records = json.load(f)
                    all_records.extend(records)
        logger.info(f"Loaded {len(all_records)} records with bounding boxes")
        return all_records

    def find_record(self, image_path: str) -> Optional[Dict]:
        """Find matching record for an image path."""
        # Normalize path for comparison
        target = Path(image_path).name
        for record in self._records:
            if Path(record["image_path"]).name == target:
                return record
        return None

    def get_crop(self, image_path: str, padding: Optional[float] = None) -> Tuple[Image.Image, Optional[List[float]]]:
        """Crop image to damage region with padding.
        
        Returns:
            Tuple of (cropped_image, bbox_or_none)
        """
        pad = padding if padding is not None else self.padding
        
        record = self.find_record(image_path)
        if record is None or record.get("bbox") is None:
            logger.warning(f"No bounding box found for {image_path}, returning full image")
            return Image.open(image_path).convert("RGB"), None
        
        bbox = record["bbox"]  # [x1, y1, x2, y2]
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        
        # Apply padding
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1
        
        pad_x = box_w * pad
        pad_y = box_h * pad
        
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(width, int(x2 + pad_x))
        y2 = min(height, int(y2 + pad_y))
        
        cropped = img.crop((x1, y1, x2, y2))
        logger.info(f"Cropped {image_path} from ({width}x{height}) to ({cropped.size})")
        
        return cropped, [x1, y1, x2, y2]

    def get_crops_for_claim(self, image_paths: List[str]) -> List[Tuple[Image.Image, Optional[List[float]]]]:
        """Get cropped images for all paths in a claim."""
        results = []
        for path in image_paths:
            try:
                crop, bbox = self.get_crop(path)
                results.append((crop, bbox))
            except Exception as e:
                logger.error(f"Failed to crop {path}: {e}")
                # Fallback to full image
                try:
                    img = Image.open(path).convert("RGB")
                    results.append((img, None))
                except Exception:
                    logger.error(f"Failed to load {path}")
        return results
