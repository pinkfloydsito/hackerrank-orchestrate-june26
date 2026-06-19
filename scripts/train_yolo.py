#!/usr/bin/env python3
"""Train YOLO object detection model for damage localization.

Uses ultralytics YOLOv8-nano or YOLO11-nano for fast training and inference.
"""

import sys
from pathlib import Path
from ultralytics import YOLO
from hackerrank_orchestrate.config import PROJECT_ROOT
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)

YOLO_DATA_DIR = PROJECT_ROOT / "yolo_data"
DATA_YAML = YOLO_DATA_DIR / "data.yaml"


def train_yolo():
    """Train YOLO model for damage detection."""
    if not DATA_YAML.exists():
        logger.error(f"Dataset not found at {DATA_YAML}. Run prepare_yolo_dataset.py first.")
        sys.exit(1)
    
    logger.info("Starting YOLO training...")
    
    # Use YOLO11-nano (fastest, smallest model)
    # Alternative: YOLOv8-nano if YOLO11 not available
    model = YOLO("yolo11n.pt")
    
    # Training configuration optimized for our dataset
    results = model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=640,
        batch=32,
        patience=20,
        save=True,
        project=str(PROJECT_ROOT / "models"),
        name="yolo_damage_detector",
        exist_ok=True,
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=5.0,
        warmup_momentum=0.5,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        auto_augment="randaugment",
        erasing=0.1,
        crop_fraction=0.5,
    )
    
    logger.info(f"Training complete. Best model: {results.best}")
    logger.info(f"Final metrics: mAP50={results.results_dict.get('metrics/mAP50(B)', 0):.3f}")
    
    return results


if __name__ == "__main__":
    train_yolo()
