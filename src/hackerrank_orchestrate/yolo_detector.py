"""YOLO-based damage detection for automatic localization.

This module provides YOLO-based object detection to automatically find
damage regions in images, replacing the preprocessed bounding box dependency.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

from hackerrank_orchestrate.config import PROJECT_ROOT
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


class YOLODetector:
    """YOLO-based damage detector for automatic localization."""

    def __init__(self, model_path: Optional[Path] = None, conf_threshold: float = 0.3):
        self.conf_threshold = conf_threshold
        
        if model_path is not None:
            self.model_path = model_path
        else:
            # Try to find trained model
            default_path = PROJECT_ROOT / "models" / "yolo_damage_detector" / "weights" / "best.pt"
            if default_path.exists():
                self.model_path = default_path
            else:
                # Use pre-trained YOLO as fallback (won't detect damage, but will detect objects)
                logger.warning("No trained YOLO model found. Using pre-trained YOLOv8n as fallback.")
                self.model_path = "yolo11n.pt"
        
        self._load_model()

    def _load_model(self) -> None:
        """Load YOLO model."""
        try:
            self.model = YOLO(str(self.model_path))
            logger.info(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            # Fallback to pre-trained model
            self.model = YOLO("yolo11n.pt")
            logger.info("Using pre-trained YOLO model as fallback")

    def detect(self, image: Image.Image) -> List[Tuple[int, float, float, float, float, float]]:
        """Detect damage regions in an image.
        
        Returns:
            List of (class_id, confidence, x_center, y_center, width, height) in normalized coordinates
        """
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x_center, y_center, w, h = box.xywhn[0].tolist()
                detections.append((cls_id, conf, x_center, y_center, w, h))
        
        return detections

    def get_crop(
        self, 
        image: Image.Image, 
        padding: float = 0.2
    ) -> Tuple[Image.Image, Optional[List[float]]]:
        """Get cropped image around the highest-confidence detection.
        
        Returns:
            Tuple of (cropped_image, bbox_or_none)
        """
        detections = self.detect(image)
        
        if not detections:
            logger.warning("No damage detected in image, returning full image")
            return image, None
        
        # Sort by confidence and take highest
        detections.sort(key=lambda x: x[1], reverse=True)
        cls_id, conf, x_center, y_center, w, h = detections[0]
        
        # Convert normalized to absolute coordinates
        img_width, img_height = image.size
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        w_abs = w * img_width
        h_abs = h * img_height
        
        # Calculate box with padding
        pad_x = w_abs * padding
        pad_y = h_abs * padding
        
        x1 = max(0, int(x_center_abs - w_abs / 2 - pad_x))
        y1 = max(0, int(y_center_abs - h_abs / 2 - pad_y))
        x2 = min(img_width, int(x_center_abs + w_abs / 2 + pad_x))
        y2 = min(img_height, int(y_center_abs + h_abs / 2 + pad_y))
        
        cropped = image.crop((x1, y1, x2, y2))
        logger.info(f"Cropped image to detection: class={cls_id}, conf={conf:.3f}, size=({cropped.size})")
        
        return cropped, [x1, y1, x2, y2]

    def get_all_crops(
        self, 
        image: Image.Image, 
        padding: float = 0.2,
        max_crops: int = 3
    ) -> List[Tuple[Image.Image, List[float]]]:
        """Get all cropped regions around detections.
        
        Returns:
            List of (cropped_image, bbox) tuples
        """
        detections = self.detect(image)
        
        if not detections:
            return [(image, None)]
        
        # Sort by confidence and take top-N
        detections.sort(key=lambda x: x[1], reverse=True)
        detections = detections[:max_crops]
        
        img_width, img_height = image.size
        crops = []
        
        for cls_id, conf, x_center, y_center, w, h in detections:
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            w_abs = w * img_width
            h_abs = h * img_height
            
            pad_x = w_abs * padding
            pad_y = h_abs * padding
            
            x1 = max(0, int(x_center_abs - w_abs / 2 - pad_x))
            y1 = max(0, int(y_center_abs - h_abs / 2 - pad_y))
            x2 = min(img_width, int(x_center_abs + w_abs / 2 + pad_x))
            y2 = min(img_height, int(y_center_abs + h_abs / 2 + pad_y))
            
            cropped = image.crop((x1, y1, x2, y2))
            crops.append((cropped, [x1, y1, x2, y2]))
        
        return crops

    def detect_and_classify(self, image: Image.Image) -> Optional[Tuple[str, float]]:
        """Detect damage and return the most likely damage type.
        
        Returns:
            Tuple of (issue_type, confidence) or None if no detection
        """
        from hackerrank_orchestrate.config import ISSUE_TYPES
        
        detections = self.detect(image)
        if not detections:
            return None
        
        detections.sort(key=lambda x: x[1], reverse=True)
        cls_id, conf, *_ = detections[0]
        
        if cls_id < len(ISSUE_TYPES):
            issue_type = ISSUE_TYPES[cls_id]
        else:
            issue_type = "unknown"
        
        return issue_type, conf
