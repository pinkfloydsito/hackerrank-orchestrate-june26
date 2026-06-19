import torch
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
from pydantic import BaseModel, Field, field_validator
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from hackerrank_orchestrate.config import QWEN_MODEL_NAME, QWEN_FALLBACK_MODEL, QWEN_MAX_NEW_TOKENS, OUTPUTS_DIR
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


class ClaimAnalysisResponse(BaseModel):
    """Structured response from the VLM for claim analysis."""
    evidence_standard_met: bool = Field(description="Whether image set is sufficient to evaluate")
    evidence_standard_met_reason: str = Field(description="Short reason for evidence decision")
    risk_flags: str = Field(description="Semicolon-separated risk flags or 'none'")
    issue_type: str = Field(description="Type of visible issue")
    object_part: str = Field(description="Relevant object part")
    claim_status: str = Field(description="Final decision: supported, contradicted, or not_enough_information")
    claim_status_justification: str = Field(description="Concise image-grounded explanation")
    supporting_image_ids: str = Field(description="Semicolon-separated image IDs or 'none'")
    valid_image: bool = Field(description="Whether image set is usable for automated review")
    severity: str = Field(description="none, low, medium, high, or unknown")

    @field_validator("issue_type")
    @classmethod
    def validate_issue_type(cls, v):
        valid_types = [
            "dent", "scratch", "crack", "glass_shatter", "broken_part",
            "missing_part", "torn_packaging", "crushed_packaging",
            "water_damage", "stain", "none", "unknown",
        ]
        if v not in valid_types:
            logger.warning(f"Invalid issue_type '{v}', defaulting to 'unknown'")
            return "unknown"
        return v

    @field_validator("claim_status")
    @classmethod
    def validate_claim_status(cls, v):
        valid_statuses = ["supported", "contradicted", "not_enough_information"]
        if v not in valid_statuses:
            logger.warning(f"Invalid claim_status '{v}', defaulting to 'not_enough_information'")
            return "not_enough_information"
        return v

    @field_validator("severity")
    @classmethod
    def validate_severity(cls, v):
        valid_severities = ["none", "low", "medium", "high", "unknown"]
        if v not in valid_severities:
            logger.warning(f"Invalid severity '{v}', defaulting to 'unknown'")
            return "unknown"
        return v


class QwenInference:
    """Qwen 2.5 VL inference wrapper with structured outputs."""

    def __init__(self, model_name: str = QWEN_MODEL_NAME, device: str = "cuda"):
        self.device = device
        self.model_name = model_name
        self._load_model()

    def _load_model(self) -> None:
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            logger.info(f"Loaded Qwen model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            logger.info(f"Trying fallback model: {QWEN_FALLBACK_MODEL}")
            self.model_name = QWEN_FALLBACK_MODEL
            self._load_model()

    def _build_system_prompt(self) -> str:
        return """You are a multi-modal evidence review system. Analyze the provided images and claim conversation to determine if the damage claim is supported, contradicted, or lacks sufficient information.

Rules:
- The images are the primary source of truth.
- User history adds risk context but should not override clear visual evidence.
- If the image clearly shows the claimed damage on the correct object part, the claim is supported.
- If the image shows no damage or contradicts the claim, the claim is contradicted.
- If the image is unclear, wrong object, or insufficient, return not_enough_information.

Output a single JSON object with these exact keys:
{
  "evidence_standard_met": true or false,
  "evidence_standard_met_reason": "short reason",
  "risk_flags": "flag1;flag2 or none",
  "issue_type": "one of: dent, scratch, crack, glass_shatter, broken_part, missing_part, torn_packaging, crushed_packaging, water_damage, stain, none, unknown",
  "object_part": "relevant part",
  "claim_status": "supported, contradicted, or not_enough_information",
  "claim_status_justification": "concise image-grounded explanation",
  "supporting_image_ids": "img_1;img_2 or none",
  "valid_image": true or false,
  "severity": "none, low, medium, high, or unknown"
}"""

    def _build_user_content(self, claim_text: str, user_history: Dict, evidence_rules: str) -> str:
        return f"""Claim Conversation:
{claim_text}

User History:
{json.dumps(user_history, indent=2)}

Evidence Requirements:
{evidence_rules}

Analyze the images and return only the JSON object."""

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON object from text response."""
        # Try to find JSON block between curly braces
        json_pattern = r'\{[\s\S]*?\}'
        matches = re.findall(json_pattern, text)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None

    def _parse_response(self, text: str) -> ClaimAnalysisResponse:
        """Parse response text into structured ClaimAnalysisResponse."""
        json_data = self._extract_json(text)
        
        if json_data is None:
            logger.error(f"Failed to parse JSON from response: {text[:200]}...")
            return ClaimAnalysisResponse(**self._default_response())
        
        try:
            return ClaimAnalysisResponse(**json_data)
        except Exception as e:
            logger.error(f"Failed to validate response: {e}")
            # Try to fill in defaults for missing fields
            defaults = self._default_response()
            defaults.update(json_data)
            return ClaimAnalysisResponse(**defaults)

    def _default_response(self) -> Dict[str, Any]:
        return {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "Failed to parse model response",
            "risk_flags": "manual_review_required",
            "issue_type": "unknown",
            "object_part": "unknown",
            "claim_status": "not_enough_information",
            "claim_status_justification": "Model failed to produce valid output",
            "supporting_image_ids": "none",
            "valid_image": False,
            "severity": "unknown",
        }

    def predict(self, images: List[Image.Image], claim_text: str, user_history: Dict, evidence_rules: str) -> ClaimAnalysisResponse:
        """Run inference on a single claim with images."""
        system_prompt = self._build_system_prompt()
        user_content = self._build_user_content(claim_text, user_history, evidence_rules)
        
        # Build messages with proper Qwen 2.5 VL chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "text", "text": user_content},
                *[{"type": "image", "image": img} for img in images]
            ]}
        ]
        
        # Apply chat template and generate
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
                do_sample=False,  # Deterministic for reproducibility
                temperature=None,
                top_p=None,
            )
        
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return self._parse_response(response_text)

    def batch_predict(self, claims: List[Dict]) -> List[ClaimAnalysisResponse]:
        """Run inference on multiple claims."""
        results = []
        for claim in claims:
            try:
                # Parse image paths
                image_paths = claim.get("image_paths", "").split(";")
                images = []
                for p in image_paths:
                    p = p.strip()
                    if p and Path(p).exists():
                        images.append(Image.open(p).convert("RGB"))
                
                if not images:
                    logger.warning(f"No valid images for claim {claim.get('user_id', 'unknown')}")
                    default = self._default_response()
                    default.update({
                        "user_id": claim.get("user_id", ""),
                        "image_paths": claim.get("image_paths", ""),
                        "user_claim": claim.get("user_claim", ""),
                        "claim_object": claim.get("claim_object", ""),
                    })
                    results.append(ClaimAnalysisResponse(**default))
                    continue
                
                result = self.predict(
                    images,
                    claim.get("user_claim", ""),
                    claim.get("user_history", {}),
                    claim.get("evidence_rules", ""),
                )
                
                # Add input fields to response
                result_dict = result.dict()
                result_dict.update({
                    "user_id": claim.get("user_id", ""),
                    "image_paths": claim.get("image_paths", ""),
                    "user_claim": claim.get("user_claim", ""),
                    "claim_object": claim.get("claim_object", ""),
                })
                results.append(ClaimAnalysisResponse(**result_dict))
                
            except Exception as e:
                logger.error(f"Error processing claim {claim.get('user_id', 'unknown')}: {e}")
                default = self._default_response()
                default.update({
                    "user_id": claim.get("user_id", ""),
                    "image_paths": claim.get("image_paths", ""),
                    "user_claim": claim.get("user_claim", ""),
                    "claim_object": claim.get("claim_object", ""),
                })
                results.append(ClaimAnalysisResponse(**default))
        
        return results
