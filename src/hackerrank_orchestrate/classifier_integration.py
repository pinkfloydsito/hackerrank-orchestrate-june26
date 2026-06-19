"""MobileNet classifier integration for second-opinion predictions."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms

from hackerrank_orchestrate.config import IMAGE_SIZE, CHECKPOINTS_DIR
from hackerrank_orchestrate.data.dataset_loader import LABEL_ENCODERS
from hackerrank_orchestrate.models.mobilenet_classifier import MobileNetMultiTask
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


class ClassifierSecondOpinion:
    """MobileNet multi-task classifier for damage prediction."""

    def __init__(self, checkpoint_path: Optional[Path] = None, device: str = "cuda"):
        self.device = device
        self.checkpoint_path = checkpoint_path or CHECKPOINTS_DIR / "best_mobilenet.pt"
        
        # Build label decoders (reverse of encoders)
        self.issue_type_decoder = {v: k for k, v in LABEL_ENCODERS["issue_type"].items()}
        self.object_part_decoder = {v: k for k, v in LABEL_ENCODERS["object_part"].items()}
        self.object_type_decoder = {v: k for k, v in LABEL_ENCODERS["object_type"].items()}
        
        self._load_model()
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _load_model(self) -> None:
        """Load the trained MobileNet model."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.checkpoint_path}")
        
        num_object_types = len(LABEL_ENCODERS["object_type"])
        num_issue_types = len(LABEL_ENCODERS["issue_type"])
        num_object_parts = len(LABEL_ENCODERS["object_part"])
        
        self.model = MobileNetMultiTask(
            num_object_types=num_object_types,
            num_issue_types=num_issue_types,
            num_object_parts=num_object_parts,
        )
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Loaded MobileNet classifier from {self.checkpoint_path}")

    def predict(self, image: Image.Image) -> Dict[str, any]:
        """Run classifier on a single image.
        
        Returns:
            Dict with keys: object_type, issue_type, object_part, has_damage, 
                           confidence_issue, confidence_part, all_issue_probs
        """
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
        
        # Decode predictions
        object_type_idx = outputs["object_type"].argmax(dim=1).item()
        issue_type_idx = outputs["issue_type"].argmax(dim=1).item()
        object_part_idx = outputs["object_part"].argmax(dim=1).item()
        has_damage_prob = torch.sigmoid(outputs["has_damage"]).item()
        
        # Get confidence scores (probabilities)
        issue_probs = torch.softmax(outputs["issue_type"], dim=1)[0]
        object_part_probs = torch.softmax(outputs["object_part"], dim=1)[0]
        
        issue_confidence = issue_probs[issue_type_idx].item()
        part_confidence = object_part_probs[object_part_idx].item()
        
        return {
            "object_type": self.object_type_decoder[object_type_idx],
            "issue_type": self.issue_type_decoder[issue_type_idx],
            "object_part": self.object_part_decoder[object_part_idx],
            "has_damage": has_damage_prob > 0.5,
            "has_damage_prob": has_damage_prob,
            "confidence_issue": issue_confidence,
            "confidence_part": part_confidence,
            "all_issue_probs": {self.issue_type_decoder[i]: p.item() for i, p in enumerate(issue_probs)},
        }

    def cross_check(
        self,
        vlm_issue: str,
        vlm_part: str,
        image: Image.Image,
        confidence_threshold: float = 0.75,
    ) -> Tuple[str, str, float, str]:
        """Cross-check VLM prediction with classifier.
        
        Returns:
            Tuple of (final_issue, final_part, confidence, decision_reason)
        """
        classifier_pred = self.predict(image)
        clf_issue = classifier_pred["issue_type"]
        clf_part = classifier_pred["object_part"]
        clf_conf = classifier_pred["confidence_issue"]
        
        # Decision logic
        if vlm_issue == clf_issue and vlm_part == clf_part:
            # Agreement - boost confidence
            return clf_issue, clf_part, max(clf_conf, 0.9), "VLM and classifier agree"
        
        if clf_conf > 0.85:
            # Classifier is very confident - trust it over VLM
            return clf_issue, clf_part, clf_conf, f"Classifier override (conf={clf_conf:.2f})"
        
        if clf_conf < 0.60:
            # Classifier is uncertain - trust VLM with reasonable confidence
            return vlm_issue, vlm_part, 0.75, f"Classifier uncertain (conf={clf_conf:.2f}), using VLM"
        
        # Moderate disagreement - use VLM with medium confidence
        return vlm_issue, vlm_part, 0.70, f"Disagreement: VLM={vlm_issue}, CLF={clf_issue} (conf={clf_conf:.2f})"

    def calibrate_severity(self, confidence: float, has_damage: bool) -> str:
        """Calibrate severity based on classifier confidence."""
        if not has_damage:
            return "none"
        if confidence > 0.85:
            return "high"
        elif confidence > 0.65:
            return "medium"
        elif confidence > 0.40:
            return "low"
        return "unknown"
