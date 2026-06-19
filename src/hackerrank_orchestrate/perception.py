"""Perception stage: Visual Fact Extraction using Qwen VLM.

This module contains only the Qwen inference wrapper with a minimal schema.
The model's job is to describe what it sees, not make policy decisions.
"""

import torch
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from hackerrank_orchestrate.config import (
    QWEN_MODEL_NAME, QWEN_FALLBACK_MODEL, QWEN_MAX_NEW_TOKENS, OUTPUTS_DIR,
    ISSUE_TYPES, SEVERITY, RISK_FLAGS, OBJECT_PARTS
)
from hackerrank_orchestrate.data.preprocessor import LabelNormalizer
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


class LabelCalibration(BaseModel):
    """Calibration data from label normalization."""
    normalized_issue: str = ""
    normalized_part: str = ""
    issue_similarity: float = 0.0
    part_similarity: float = 0.0
    calibration_boost: float = 0.0



class VisualFindings(BaseModel):
    """Perception-only schema. The model describes visual facts only.

    No policy decisions, no claim adjudication, no evidence sufficiency judgment.
    """

    valid_image: bool = Field(
        description="Whether the image set is usable for visual analysis (not blurry, not completely wrong object)"
    )
    visible_issue: str = Field(
        description="What damage/issue is visible: one of the canonical issue types or 'none'/'unknown'"
    )
    object_part: str = Field(
        description="Which part of the object shows the issue"
    )
    severity: str = Field(
        description="Severity of visible issue: none, low, medium, high, or unknown"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Model confidence in its visual assessment (0-1)"
    )
    supporting_image_ids: List[str] = Field(
        default_factory=list,
        description="List of image IDs that show the visible issue"
    )
    observations: List[str] = Field(
        default_factory=list,
        description="List of raw visual observations (e.g., 'visible dent on hood', 'image is blurry')"
    )
    risk_flags: List[str] = Field(
        default_factory=list,
        description="Visual risk flags only: blurry_image, wrong_object, wrong_angle, damage_not_visible, low_light_or_glare, cropped_or_obstructed, possible_manipulation, non_original_image"
    )

    @field_validator("visible_issue")
    @classmethod
    def validate_visible_issue(cls, v):
        if v not in ISSUE_TYPES:
            logger.warning(f"Invalid visible_issue '{v}', defaulting to 'unknown'")
            return "unknown"
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        if v not in SEVERITY:
            logger.warning(f"Invalid severity '{v}', defaulting to 'unknown'")
            return "unknown"
        return v

    @field_validator("risk_flags")
    @classmethod
    def validate_risk_flags(cls, v):
        valid_visual_flags = {
            "blurry_image", "cropped_or_obstructed", "low_light_or_glare",
            "wrong_angle", "wrong_object", "damage_not_visible",
            "possible_manipulation", "non_original_image", "text_instruction_present",
            "none"
        }
        validated = []
        for flag in v:
            if flag in valid_visual_flags or flag in RISK_FLAGS:
                validated.append(flag)
            else:
                logger.warning(f"Invalid risk flag '{flag}', skipping")
        return validated


class QwenPerception:
    """Qwen 2.5 VL inference wrapper for visual perception only."""

    def __init__(self, model_name: str = QWEN_MODEL_NAME, device: str = "cuda", use_label_normalization: bool = True):
        self.device = device
        self.model_name = model_name
        self.use_label_normalization = use_label_normalization
        self.label_normalizer = LabelNormalizer() if use_label_normalization else None
        self._load_model()

    def _load_model(self) -> None:
        """Load the Qwen model with appropriate model class for VL models."""
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            
            # Try to load as a vision-language model first
            # Qwen2-VL models are not AutoModelForCausalLM, they need special handling
            self._try_load_vl_model(bnb_config)
            
            logger.info(f"Loaded Qwen model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            logger.info(f"Trying fallback model: {QWEN_FALLBACK_MODEL}")
            self.model_name = QWEN_FALLBACK_MODEL
            self._load_model()

    def _try_load_vl_model(self, bnb_config) -> None:
        """Try multiple model classes for VL models."""
        errors = []
        
        # Try 1: AutoModelForVision2Seq (if available in transformers)
        try:
            from transformers import AutoModelForVision2Seq
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            logger.info("Loaded model using AutoModelForVision2Seq")
            return
        except Exception as e:
            errors.append(f"AutoModelForVision2Seq: {e}")
        
        # Try 2: Qwen2VLForConditionalGeneration (specific to Qwen2-VL)
        try:
            from transformers import Qwen2VLForConditionalGeneration
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            logger.info("Loaded model using Qwen2VLForConditionalGeneration")
            return
        except Exception as e:
            errors.append(f"Qwen2VLForConditionalGeneration: {e}")
        
        # Try 3: AutoModel (generic, should work for any model)
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer if hasattr(self.processor, 'tokenizer') else self.processor
            logger.info("Loaded model using AutoModel")
            return
        except Exception as e:
            errors.append(f"AutoModel: {e}")
        
        # Try 4: AutoModelForCausalLM (for text-only models)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            logger.info("Loaded model using AutoModelForCausalLM")
            return
        except Exception as e:
            errors.append(f"AutoModelForCausalLM: {e}")
        
        # All attempts failed
        raise Exception(f"All model loading attempts failed: {'; '.join(errors)}")

    def _build_system_prompt(self) -> str:
        return """You are a visual inspector. Your ONLY job is to describe what you see in the images.

You must NOT make policy decisions, evaluate evidence sufficiency, or adjudicate claims.
You are a camera, not a judge.

Describe:
1. Is the image usable? (clear, right object, right angle)
2. What damage or issue is visible?
3. Where on the object is it located?
4. How severe does it appear?
5. Which image(s) show the issue?
6. Any visual quality problems (blur, wrong angle, wrong object, etc.)?

CRITICAL INSTRUCTIONS FOR LABELING:
- Use EXACT canonical labels from the lists below, NOT descriptive phrases
- For object parts, use one of: front_bumper, rear_bumper, door, hood, windshield, side_mirror, headlight, taillight, fender, quarter_panel, body, screen, keyboard, trackpad, hinge, lid, corner, port, base, box, package_corner, package_side, seal, label, contents, item, unknown
- For issue types, use one of: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown
- For severity, use one of: none, low, medium, high, unknown

CRITICAL: INSPECTION GUIDELINES FOR SUBTLE DAMAGE:
- Carefully inspect corners, edges, seams, hinges, ports, and all joints.
- Look for any mark, indentation, discoloration, scratch, or deformation — even very small ones.
- A small dent, tiny scratch, or slight corner deformation IS visible damage. Do NOT classify it as "none".
- If you see ANY mark or anomaly at all, report it as the appropriate issue type (dent, scratch, crack, etc.) with severity "low".
- Only use "none" if the object is completely pristine and undamaged with zero visible marks.
- If you are unsure whether a mark is damage or normal wear, report it as the most likely issue type with lower confidence (0.4-0.6).

SEVERITY RUBRIC (be precise):
- none: COMPLETELY pristine. Zero visible marks, scratches, dents, or discoloration. Object looks brand new.
- low: Very minor damage. Small dent (<1cm), hairline scratch, tiny chip, slight corner deformation, minor discoloration. Visible upon close inspection but not immediately obvious.
- medium: Noticeable damage. Clearly visible at a glance. Dent >1cm, crack <5cm, visible scratch across panel, torn seal, bent corner.
- high: Significant damage. Immediately obvious. Major functional impact or structural damage. Shattered glass, large crack >5cm, crushed corner, broken hinge, missing part, severe dent.
- unknown: Cannot determine severity from the image (e.g., image too far, too dark, or angle hides the damage extent).

CONFIDENCE GUIDANCE:
- 0.9-1.0: Absolutely certain. Damage is clear, well-lit, and unambiguous.
- 0.7-0.9: Confident. Damage is visible but could be minor or ambiguous.
- 0.5-0.7: Moderate. Damage may be present but not entirely clear.
- 0.3-0.5: Uncertain. Some indication but not confident.
- 0.0-0.3: Very uncertain. Guessing.

Output a single JSON object with these exact keys:
{
  "valid_image": true or false,
  "visible_issue": "one of the canonical issue types listed above",
  "object_part": "one of the canonical part labels listed above",
  "severity": "none, low, medium, high, or unknown",
  "confidence": 0.0 to 1.0,
  "supporting_image_ids": ["img_1", "img_2"] or [],
  "observations": ["short observation 1", "short observation 2"],
  "risk_flags": ["blurry_image", "wrong_angle", "damage_not_visible"] or []
}

Important: Only describe visual facts. Do NOT say if the claim is supported or contradicted."""

    def _build_user_content(self, claim_text: str, claim_object: str) -> str:
        # Extract claimed issue and part from the conversation for focus guidance
        from hackerrank_orchestrate.evidence_evaluator import _extract_claimed_issue
        claimed_issue = _extract_claimed_issue(claim_text)
        
        # Try to extract claimed part from conversation
        claim_lower = claim_text.lower()
        part_keywords = {
            "corner": ["corner"], "screen": ["screen"], "keyboard": ["keyboard"],
            "trackpad": ["trackpad"], "hinge": ["hinge"], "lid": ["lid"],
            "door": ["door"], "hood": ["hood"], "windshield": ["windshield"],
            "headlight": ["headlight"], "taillight": ["taillight"],
            "fender": ["fender"], "bumper": ["bumper"], "front_bumper": ["front bumper"],
            "rear_bumper": ["rear bumper"], "side_mirror": ["side mirror"],
            "body": ["body"], "port": ["port"], "base": ["base"],
            "box": ["box"], "package_corner": ["package corner", "corner of package"],
            "package_side": ["package side", "side of package"], "seal": ["seal"],
            "label": ["label"], "contents": ["contents"],
        }
        claimed_parts = []
        for part, keywords in part_keywords.items():
            if any(kw in claim_lower for kw in keywords):
                claimed_parts.append(part)
        
        focus_instruction = ""
        if claimed_issue or claimed_parts:
            focus_parts = []
            if claimed_issue:
                focus_parts.append(f"looking specifically for: {claimed_issue}")
            if claimed_parts:
                focus_parts.append(f"on these parts: {', '.join(claimed_parts)}")
            focus_instruction = f"\n\nFOCUS INSTRUCTION: The user claims damage ({' | '.join(focus_parts)}). Carefully inspect these specific areas in the images. Even small marks or subtle deformations in these areas should be reported."
        
        return f"""Claim Conversation (for context only, do not use to override what you see):
{claim_text}

Claimed Object: {claim_object}{focus_instruction}

Describe what you see in the images. Return only the JSON object."""

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text response."""
        json_pattern = r'\{[\s\S]*?\}'
        matches = re.findall(json_pattern, text)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        return None

    def _parse_response(self, text: str) -> VisualFindings:
        """Parse response text into VisualFindings."""
        json_data = self._extract_json(text)

        if json_data is None:
            logger.error(f"Failed to parse JSON from response: {text[:200]}...")
            return VisualFindings(**self._default_response())

        # Normalize fields
        if isinstance(json_data.get("supporting_image_ids"), str):
            ids = json_data["supporting_image_ids"]
            if ids == "none" or ids == "":
                json_data["supporting_image_ids"] = []
            else:
                json_data["supporting_image_ids"] = [x.strip() for x in ids.split(";")]

        if isinstance(json_data.get("risk_flags"), str):
            flags = json_data["risk_flags"]
            if flags == "none" or flags == "":
                json_data["risk_flags"] = []
            else:
                json_data["risk_flags"] = [x.strip() for x in flags.split(";")]

        if isinstance(json_data.get("observations"), str):
            obs = json_data["observations"]
            if obs == "none" or obs == "":
                json_data["observations"] = []
            else:
                json_data["observations"] = [x.strip() for x in obs.split(";")]

        try:
            return VisualFindings(**json_data)
        except Exception as e:
            logger.error(f"Failed to validate response: {e}")
            defaults = self._default_response()
            defaults.update({k: v for k, v in json_data.items() if k in defaults})
            return VisualFindings(**defaults)

    def _default_response(self) -> Dict[str, Any]:
        return {
            "valid_image": False,
            "visible_issue": "unknown",
            "object_part": "unknown",
            "severity": "unknown",
            "confidence": 0.0,
            "supporting_image_ids": [],
            "observations": ["Failed to parse model response"],
            "risk_flags": [],
        }

    def _normalize_labels(self, finding: VisualFindings, claim_object: str) -> Tuple[VisualFindings, LabelCalibration]:
        """Normalize object_part and issue_type to canonical labels using embeddings.

        Also computes a calibration boost based on similarity scores.
        """
        calibration = LabelCalibration()

        if self.label_normalizer is None:
            calibration.normalized_issue = finding.visible_issue
            calibration.normalized_part = finding.object_part
            return finding, calibration

        try:
            # Normalize visible_issue using the raw label
            normalized_issue, _ = self.label_normalizer.normalize(
                finding.visible_issue, claim_object
            )
            # Normalize object_part using the raw label
            _, normalized_part = self.label_normalizer.normalize(
                finding.object_part, claim_object
            )

            # Compute similarity scores (approximate from normalization internals)
            issue_embedding = self.label_normalizer.encoder.encode(
                [finding.visible_issue], convert_to_numpy=True, show_progress_bar=False
            )
            part_embedding = self.label_normalizer.encoder.encode(
                [finding.object_part], convert_to_numpy=True, show_progress_bar=False
            )

            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np

            issue_sims = cosine_similarity(
                issue_embedding, self.label_normalizer.issue_embeddings
            )[0]
            part_sims = cosine_similarity(
                part_embedding, self.label_normalizer.part_embeddings.get(claim_object, self.label_normalizer.part_embeddings.get("car", np.array([])))
            )[0] if self.label_normalizer.part_embeddings else np.array([0.0])

            issue_sim = float(np.max(issue_sims)) if len(issue_sims) > 0 else 0.0
            part_sim = float(np.max(part_sims)) if len(part_sims) > 0 else 0.0

            calibration.issue_similarity = issue_sim
            calibration.part_similarity = part_sim
            calibration.normalized_issue = normalized_issue
            calibration.normalized_part = normalized_part

            # Calibration boost based on similarity - reduced to avoid over-confidence
            avg_sim = (issue_sim + part_sim) / 2
            if avg_sim > 0.8:
                calibration.calibration_boost = 0.0
            elif avg_sim > 0.5:
                calibration.calibration_boost = 0.0
            else:
                calibration.calibration_boost = -0.05

            # Update finding with normalized labels
            finding = VisualFindings(
                valid_image=finding.valid_image,
                visible_issue=normalized_issue,
                object_part=normalized_part,
                severity=finding.severity,
                confidence=min(1.0, max(0.0, finding.confidence + calibration.calibration_boost)),
                supporting_image_ids=finding.supporting_image_ids,
                observations=finding.observations,
                risk_flags=finding.risk_flags,
            )

            logger.info(
                f"Normalized labels: issue={finding.visible_issue} (sim={issue_sim:.3f}), "
                f"part={finding.object_part} (sim={part_sim:.3f}), "
                f"boost={calibration.calibration_boost:+.3f}"
            )

        except Exception as e:
            logger.warning(f"Label normalization failed: {e}, using original labels")
            calibration.normalized_issue = finding.visible_issue
            calibration.normalized_part = finding.object_part

        return finding, calibration

    def predict(self, images: List[Image.Image], claim_text: str, claim_object: str, use_multi_crop: bool = True) -> VisualFindings:
        """Run perception inference on a single claim with images.
        
        Processes images one at a time to avoid OOM on GPUs with limited memory.
        Applies label normalization and confidence calibration after prediction.
        
        If use_multi_crop is True and an image is large enough, generates
        quadrant crops and inspects each region carefully.
        """
        if not images:
            return VisualFindings(**self._default_response())
        
        # Process images one at a time to avoid OOM
        all_findings = []
        for idx, img in enumerate(images):
            try:
                if use_multi_crop and (img.width > 512 or img.height > 512):
                    # Generate multiple crops for detailed inspection
                    crops = self._generate_crops(img)
                    crop_findings = []
                    for crop_idx, crop in enumerate(crops):
                        try:
                            finding = self._predict_single_image(crop, claim_text, claim_object, idx)
                            crop_findings.append(finding)
                        except Exception as e:
                            logger.warning(f"Crop {crop_idx} for image {idx} failed: {e}")
                    
                    # Select best finding from crops
                    best_finding = self._select_best_crop_finding(crop_findings)
                    all_findings.append(best_finding)
                else:
                    finding = self._predict_single_image(img, claim_text, claim_object, idx)
                    all_findings.append(finding)
            except Exception as e:
                logger.error(f"Error processing image {idx}: {e}")
                all_findings.append(VisualFindings(
                    valid_image=False,
                    visible_issue="unknown",
                    object_part="unknown",
                    severity="unknown",
                    confidence=0.0,
                    supporting_image_ids=[],
                    observations=[f"Image {idx} failed: {str(e)}"],
                    risk_flags=["damage_not_visible"],
                ))
        
        # Aggregate findings from all images
        aggregated = self._aggregate_findings(all_findings)

        # Apply label normalization and confidence calibration
        if self.use_label_normalization and aggregated.visible_issue != "unknown":
            aggregated, calibration = self._normalize_labels(aggregated, claim_object)
            logger.info(
                f"Post-normalization: issue={aggregated.visible_issue}, "
                f"part={aggregated.object_part}, conf={aggregated.confidence:.3f}"
            )

        return aggregated
    
    def _generate_crops(self, image: Image.Image, min_size: int = 256) -> List[Image.Image]:
        """Generate multiple crops from an image for detailed inspection."""
        crops = [image]  # Full image first
        
        w, h = image.size
        if w < min_size or h < min_size:
            return crops
        
        # Generate quadrant crops
        # Top-left, top-right, bottom-left, bottom-right
        half_w = w // 2
        half_h = h // 2
        
        quadrants = [
            (0, 0, half_w, half_h),          # top-left
            (half_w, 0, w, half_h),          # top-right
            (0, half_h, half_w, h),          # bottom-left
            (half_w, half_h, w, h),          # bottom-right
        ]
        
        for (x1, y1, x2, y2) in quadrants:
            crop = image.crop((x1, y1, x2, y2))
            if crop.width >= min_size and crop.height >= min_size:
                crops.append(crop)
        
        # Center crop (20% padding from center)
        pad_w = w // 5
        pad_h = h // 5
        center_crop = image.crop((pad_w, pad_h, w - pad_w, h - pad_h))
        if center_crop.width >= min_size and center_crop.height >= min_size:
            crops.append(center_crop)
        
        return crops
    
    def _select_best_crop_finding(self, findings: List[VisualFindings]) -> VisualFindings:
        """Select the best finding from multiple crops.
        
        Priority:
        1. Findings with visible damage (not none/unknown) and highest confidence
        2. Findings with valid_image=True and highest confidence
        3. Any finding with highest confidence
        """
        if not findings:
            return VisualFindings(**self._default_response())
        
        # First, try findings with actual damage
        damage_findings = [f for f in findings if f.visible_issue not in ("none", "unknown") and f.confidence > 0.3]
        if damage_findings:
            return max(damage_findings, key=lambda f: f.confidence)
        
        # Then, try findings with valid images
        valid_findings = [f for f in findings if f.valid_image and f.confidence > 0.0]
        if valid_findings:
            return max(valid_findings, key=lambda f: f.confidence)
        
        # Fallback: return the finding with highest confidence
        return max(findings, key=lambda f: f.confidence)

    def _predict_single_image(self, image: Image.Image, claim_text: str, claim_object: str, image_idx: int) -> VisualFindings:
        """Run perception on a single image."""
        system_prompt = self._build_system_prompt()
        user_content = self._build_user_content(claim_text, claim_object)

        # Build messages with single image
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_content},
                {"type": "image", "image": image}
            ]}
        ]

        # For VL models, use processor to handle images
        if hasattr(self, 'processor') and self.processor is not None:
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        finding = self._parse_response(response_text)
        # Add image ID to supporting_image_ids if it found something
        if finding.valid_image and finding.visible_issue != "unknown":
            img_id = f"img_{image_idx + 1}"
            if img_id not in finding.supporting_image_ids:
                finding.supporting_image_ids.append(img_id)
        
        return finding

    def _aggregate_findings(self, findings: List[VisualFindings]) -> VisualFindings:
        """Aggregate findings from multiple images into a single finding."""
        if not findings:
            return VisualFindings(**self._default_response())
        
        # Check if any image is valid
        any_valid = any(f.valid_image for f in findings)
        if not any_valid:
            return VisualFindings(
                valid_image=False,
                visible_issue="unknown",
                object_part="unknown",
                severity="unknown",
                confidence=0.0,
                supporting_image_ids=[],
                observations=["All images failed validation"],
                risk_flags=["damage_not_visible"],
            )
        
        # Find the best finding (highest confidence, valid image)
        valid_findings = [f for f in findings if f.valid_image]
        if not valid_findings:
            return findings[0]  # Return first finding if none are valid
        
        # Find the finding with highest confidence that has a visible issue
        findings_with_issue = [f for f in valid_findings if f.visible_issue != "unknown"]
        if findings_with_issue:
            best_finding = max(findings_with_issue, key=lambda f: f.confidence)
        else:
            best_finding = valid_findings[0]
        
        # Collect all observations and risk flags
        all_observations = []
        all_risk_flags = set()
        all_supporting_ids = []
        
        for f in findings:
            all_observations.extend(f.observations)
            all_risk_flags.update(f.risk_flags)
            all_supporting_ids.extend(f.supporting_image_ids)
        
        # Use best finding as base but merge observations
        return VisualFindings(
            valid_image=best_finding.valid_image,
            visible_issue=best_finding.visible_issue,
            object_part=best_finding.object_part,
            severity=best_finding.severity,
            confidence=best_finding.confidence,
            supporting_image_ids=list(dict.fromkeys(all_supporting_ids)),  # Remove duplicates, preserve order
            observations=list(dict.fromkeys(all_observations)),
            risk_flags=list(all_risk_flags),
        )

    def batch_predict(self, claims: List[Dict]) -> List[VisualFindings]:
        """Run perception inference on multiple claims with label normalization."""
        results = []
        for claim in claims:
            try:
                image_paths = claim.get("image_paths", "").split(";")
                images = []
                for p in image_paths:
                    p = p.strip()
                    if not p:
                        continue
                    # Resolve path - try with dataset prefix first, then as-is
                    from hackerrank_orchestrate.config import PROJECT_ROOT
                    path_candidates = [
                        PROJECT_ROOT / p,
                        PROJECT_ROOT / "dataset" / p,
                    ]
                    for path_obj in path_candidates:
                        if path_obj.exists():
                            images.append(Image.open(path_obj).convert("RGB"))
                            break

                if not images:
                    logger.warning(
                        f"No valid images for claim {claim.get('user_id', 'unknown')}"
                    )
                    results.append(VisualFindings(**self._default_response()))
                    continue

                result = self.predict(
                    images,
                    claim.get("user_claim", ""),
                    claim.get("claim_object", ""),
                )
                results.append(result)

            except Exception as e:
                logger.error(
                    f"Error processing claim {claim.get('user_id', 'unknown')}: {e}"
                )
                results.append(VisualFindings(**self._default_response()))

        return results
