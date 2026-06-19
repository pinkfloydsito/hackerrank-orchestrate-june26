import torch
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from hackerrank_orchestrate.config import QWEN_MODEL_NAME, QWEN_FALLBACK_MODEL, QWEN_MAX_NEW_TOKENS, OUTPUTS_DIR
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)

class QwenInference:
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

    def build_prompt(self, images: List[Image.Image], claim_text: str, user_history: Dict, evidence_rules: str) -> str:
        system_prompt = """You are a multi-modal evidence review system. Analyze the provided images and claim conversation to determine if the damage claim is supported, contradicted, or lacks sufficient information.

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
        
        user_content = f"Claim Conversation:\n{claim_text}\n\nUser History:\n{json.dumps(user_history, indent=2)}\n\nEvidence Requirements:\n{evidence_rules}\n\nAnalyze the images and return only the JSON object."
        
        return system_prompt, user_content

    def predict(self, images: List[Image.Image], claim_text: str, user_history: Dict, evidence_rules: str) -> Dict[str, Any]:
        system_prompt, user_content = self.build_prompt(images, claim_text, user_history, evidence_rules)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        # Add images to the message
        for img in images:
            messages.append({"role": "user", "content": img})
        
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS)
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            json_str = response[response.find("{"):response.rfind("}")+1]
            result = json.loads(json_str)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response: {response}")
            result = self._default_response()
        
        return result

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

    def batch_predict(self, claims: List[Dict]) -> List[Dict]:
        results = []
        for claim in claims:
            try:
                images = [Image.open(p).convert("RGB") for p in claim["image_paths"].split(";")]
                result = self.predict(
                    images,
                    claim["user_claim"],
                    claim.get("user_history", {}),
                    claim.get("evidence_rules", ""),
                )
                result["user_id"] = claim["user_id"]
                result["image_paths"] = claim["image_paths"]
                result["user_claim"] = claim["user_claim"]
                result["claim_object"] = claim["claim_object"]
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing claim {claim['user_id']}: {e}")
                result = self._default_response()
                result.update({
                    "user_id": claim["user_id"],
                    "image_paths": claim["image_paths"],
                    "user_claim": claim["user_claim"],
                    "claim_object": claim["claim_object"],
                })
                results.append(result)
        return results
