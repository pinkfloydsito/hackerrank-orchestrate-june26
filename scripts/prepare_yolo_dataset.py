#!/usr/bin/env python3
"""Prepare dataset for YOLO object detection training.

Converts preprocessed COCO-format annotations to YOLO format:
- COCO: [x1, y1, x2, y2] absolute pixels
- YOLO: [class_id, x_center, y_center, width, height] normalized 0-1

Creates YOLO dataset structure:
  yolo_data/
  ├── images/
  │   ├── train/
  │   ├── val/
  │   └── test/
  ├── labels/
  │   ├── train/
  │   ├── val/
  │   └── test/
  └── data.yaml
"""

import json
import shutil
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

from hackerrank_orchestrate.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from hackerrank_orchestrate.data.dataset_loader import LABEL_ENCODERS
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)

YOLO_DATA_DIR = PROJECT_ROOT / "yolo_data"
ISSUE_TYPES = [
    "dent", "scratch", "crack", "glass_shatter", "broken_part",
    "missing_part", "torn_packaging", "crushed_packaging",
    "water_damage", "stain", "none", "unknown",
]


def convert_coco_to_yolo(
    bbox: List[float], 
    img_width: int, 
    img_height: int
) -> Tuple[float, float, float, float]:
    """Convert COCO [x1, y1, x2, y2] to YOLO [x_center, y_center, w, h] normalized."""
    x1, y1, x2, y2 = bbox
    
    # Ensure coordinates are within image bounds
    x1 = max(0, min(x1, img_width))
    y1 = max(0, min(y1, img_height))
    x2 = max(0, min(x2, img_width))
    y2 = max(0, min(y2, img_height))
    
    # Convert to YOLO format
    w = x2 - x1
    h = y2 - y1
    x_center = x1 + w / 2
    y_center = y1 + h / 2
    
    # Normalize
    x_center_norm = x_center / img_width
    y_center_norm = y_center / img_height
    w_norm = w / img_width
    h_norm = h / img_height
    
    # Clamp to [0, 1]
    x_center_norm = max(0.0, min(1.0, x_center_norm))
    y_center_norm = max(0.0, min(1.0, y_center_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    
    return x_center_norm, y_center_norm, w_norm, h_norm


def load_records() -> List[Dict]:
    """Load all preprocessed records."""
    all_records = []
    for split_file in ["train.json", "val.json", "test.json"]:
        split_path = PROCESSED_DATA_DIR / split_file
        if split_path.exists():
            with open(split_path, "r") as f:
                records = json.load(f)
                all_records.extend(records)
            logger.info(f"Loaded {len(records)} records from {split_file}")
    return all_records


def group_by_image(records: List[Dict]) -> Dict[str, List[Dict]]:
    """Group records by image path."""
    images = defaultdict(list)
    for record in records:
        if record.get("bbox") is not None and record.get("has_damage", False):
            images[record["image_path"]].append(record)
    return images


def create_yolo_dataset():
    """Create YOLO-formatted dataset."""
    logger.info("Creating YOLO dataset...")
    
    # Load records
    records = load_records()
    images = group_by_image(records)
    
    logger.info(f"Total images with annotations: {len(images)}")
    
    # Create directory structure
    for split in ["train", "val", "test"]:
        (YOLO_DATA_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    
    # Track class distribution
    class_counts = defaultdict(int)
    
    # Split name mapping (data uses "valid", YOLO expects "val")
    split_mapping = {"train": "train", "valid": "val", "val": "val", "test": "test", "sample": "val"}
    
    # Process each image
    for img_path, annotations in images.items():
        # Determine split from first annotation, mapping to YOLO format
        raw_split = annotations[0].get("split", "train")
        split = split_mapping.get(raw_split, "train")
        
        # Resolve image path
        src_path = PROJECT_ROOT / img_path
        if not src_path.exists():
            # Try with dataset prefix
            src_path = PROJECT_ROOT / "dataset" / img_path
        
        if not src_path.exists():
            logger.warning(f"Image not found: {img_path}")
            continue
        
        # Copy image to YOLO directory
        dst_img_path = YOLO_DATA_DIR / "images" / split / src_path.name
        shutil.copy2(src_path, dst_img_path)
        
        # Create label file
        label_path = YOLO_DATA_DIR / "labels" / split / f"{src_path.stem}.txt"
        
        with open(label_path, "w") as f:
            for ann in annotations:
                bbox = ann.get("bbox")
                if bbox is None:
                    continue
                
                issue_type = ann.get("issue_type", "unknown")
                if issue_type == "none" or issue_type == "unknown":
                    continue
                
                class_id = ISSUE_TYPES.index(issue_type) if issue_type in ISSUE_TYPES else 11
                class_counts[class_id] += 1
                
                # Convert bbox
                img_width = ann.get("width", 224)
                img_height = ann.get("height", 224)
                x_center, y_center, w, h = convert_coco_to_yolo(bbox, img_width, img_height)
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
    
    # Create data.yaml
    yaml_path = YOLO_DATA_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_DATA_DIR.absolute()}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n\n")
        f.write(f"nc: {len(ISSUE_TYPES)}\n")
        f.write(f"names: {ISSUE_TYPES}\n")
    
    logger.info(f"YOLO dataset created at {YOLO_DATA_DIR}")
    logger.info(f"Class distribution: {dict(class_counts)}")
    
    # Count images per split
    for split in ["train", "val", "test"]:
        img_count = len(list((YOLO_DATA_DIR / "images" / split).glob("*")))
        logger.info(f"  {split}: {img_count} images")


if __name__ == "__main__":
    create_yolo_dataset()
    logger.info("Dataset preparation complete!")
