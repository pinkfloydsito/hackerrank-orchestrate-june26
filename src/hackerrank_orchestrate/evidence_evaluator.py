"""Evidence Evaluation stage: Deterministic rule-based evidence assessment.

This module derives `evidence_standard_met` and `evidence_standard_met_reason`
from visual findings + claim context. No model inference here — pure logic.
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from hackerrank_orchestrate.config import RISK_FLAGS
from hackerrank_orchestrate.perception import VisualFindings
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class EvidenceEvaluation:
    """Result of evidence evaluation."""

    evidence_standard_met: bool
    evidence_standard_met_reason: str
    risk_flags: List[str] = field(default_factory=list)


def evaluate_evidence(
    findings: VisualFindings,
    claim_text: str,
    claim_object: str,
    user_history: Dict[str, Any],
    evidence_requirements: str,
) -> EvidenceEvaluation:
    """Evaluate whether visual evidence meets the standard for the claim.

    This is a deterministic function — no stochastic models, no hallucination.
    All decisions are based on explicit rules applied to perception-stage facts.
    """
    risk_flags = list(findings.risk_flags)  # Start with visual risks
    reasons = []

    # --- Rule 1: Image quality gate ---
    if not findings.valid_image:
        reasons.append("The submitted image set is not usable for automated review.")
        if "blurry_image" not in risk_flags:
            risk_flags.append("blurry_image")
        return EvidenceEvaluation(
            evidence_standard_met=False,
            evidence_standard_met_reason=" ".join(reasons),
            risk_flags=risk_flags,
        )

    # --- Rule 2: Visual confidence gate ---
    if findings.confidence < 0.2:
        reasons.append(
            f"Model confidence is very low ({findings.confidence:.2f}), so visual assessment is uncertain."
        )
        if "damage_not_visible" not in risk_flags:
            risk_flags.append("damage_not_visible")
        return EvidenceEvaluation(
            evidence_standard_met=False,
            evidence_standard_met_reason=" ".join(reasons),
            risk_flags=risk_flags,
        )

    # --- Rule 3: Object/part visibility gate ---
    if findings.object_part == "unknown":
        reasons.append("The relevant object part cannot be identified from the images.")
        if "damage_not_visible" not in risk_flags:
            risk_flags.append("damage_not_visible")
        return EvidenceEvaluation(
            evidence_standard_met=False,
            evidence_standard_met_reason=" ".join(reasons),
            risk_flags=risk_flags,
        )

    # --- Rule 4: Wrong object check ---
    if "wrong_object" in findings.risk_flags:
        reasons.append("The images appear to show a different object than claimed.")
        return EvidenceEvaluation(
            evidence_standard_met=False,
            evidence_standard_met_reason=" ".join(reasons),
            risk_flags=risk_flags,
        )

    # --- Rule 5: Evidence sufficiency based on issue type ---
    if findings.visible_issue == "unknown":
        if findings.severity == "none":
            reasons.append("No visible damage is detected in the images.")
        else:
            reasons.append("Damage may be present but the specific issue type cannot be determined.")
        return EvidenceEvaluation(
            evidence_standard_met=False,
            evidence_standard_met_reason=" ".join(reasons),
            risk_flags=risk_flags,
        )

    # --- Rule 6: Supporting image check (advisory, not a hard gate) ---
    if not findings.supporting_image_ids:
        reasons.append("No specific image clearly shows the claimed issue, but findings are otherwise clear.")

    # --- If we got here, evidence is sufficient ---
    reasons.append(
        f"The {findings.object_part} is visible and the {findings.visible_issue} can be verified from the submitted image(s)."
    )

    # --- Rule 7: Claim mismatch detection ---
    # Only flag mismatch when:
    # 1. We DO see damage, but it's a DIFFERENT type than claimed (e.g., claimed dent, see crack)
    # 2. We see the claimed part is clearly intact and the claim says it's damaged (visible_issue == "none" with high confidence)
    # Do NOT flag mismatch when we simply can't verify the damage (low confidence, bad angle, etc.)
    claimed_issue = _extract_claimed_issue(claim_text)
    if claimed_issue and claimed_issue != findings.visible_issue:
        if findings.visible_issue == "none" and findings.confidence > 0.7:
            # High confidence that nothing is visible → claim is contradicted
            risk_flags.append("claim_mismatch")
            reasons.append(
                f"The claimed issue ({claimed_issue}) is not visible in the images; the {findings.object_part} appears intact."
            )
        elif findings.visible_issue != "none" and findings.confidence > 0.5:
            # We see damage, but it's a different type than claimed
            risk_flags.append("claim_mismatch")
            reasons.append(
                f"The claimed issue ({claimed_issue}) differs from the visible issue ({findings.visible_issue})."
            )

    # --- Rule 8: User history risk ---
    history_flags = user_history.get("history_flags", "none")
    if history_flags and history_flags != "none":
        risk_flags.append(history_flags)
        reasons.append(f"User history flags: {history_flags}.")

    # --- Rule 9: Severity vs claim exaggeration ---
    exaggeration_patterns = [
        "pretty bad", "badly crushed", "severe", "totally destroyed",
        "completely damaged", "heavily damaged", "major damage",
        "terrible", "extreme", "very bad", "completely destroyed",
        "shattered", "ruined", "wrecked",
    ]
    if findings.severity in ("low", "none") and any(p in claim_text.lower() for p in exaggeration_patterns):
        risk_flags.append("claim_mismatch")
        reasons.append("Claim language suggests severe damage but visible damage is minimal or absent.")

    # --- Rule 10: Manual review for non-original images ---
    if "non_original_image" in findings.risk_flags or "possible_manipulation" in findings.risk_flags:
        risk_flags.append("manual_review_required")
        reasons.append("Image authenticity concerns detected.")

    # --- Rule 11: Wrong angle / damage not visible ---
    if "wrong_angle" in findings.risk_flags or "damage_not_visible" in findings.risk_flags:
        if findings.confidence < 0.5:
            return EvidenceEvaluation(
                evidence_standard_met=False,
                evidence_standard_met_reason=" ".join(reasons),
                risk_flags=risk_flags,
            )
        else:
            reasons.append("Some images have suboptimal angles but the damage is still visible in others.")

    # Deduplicate risk flags
    risk_flags = list(dict.fromkeys(risk_flags))  # preserves order, removes duplicates

    # Remove visual-only flags that are not in the final schema if needed
    # (but we keep them all since the Evaluator handles them)

    return EvidenceEvaluation(
        evidence_standard_met=True,
        evidence_standard_met_reason=" ".join(reasons),
        risk_flags=risk_flags,
    )


def _extract_claimed_issue(claim_text: str) -> str:
    """Extract the claimed issue type from the user claim text.

    Lightweight keyword matching. No heavy NLP needed.
    """
    claim_lower = claim_text.lower()

    # Priority order matters
    issue_keywords = {
        "dent": ["dent", "dented"],
        "scratch": ["scratch", "scratched", "scrape"],
        "crack": ["crack", "cracked", "shattered"],
        "glass_shatter": ["glass", "shattered glass", "broken glass"],
        "broken_part": ["broken", "damaged part", "part damaged"],
        "missing_part": ["missing", "lost part"],
        "torn_packaging": ["torn", "ripped packaging"],
        "crushed_packaging": ["crushed", "flattened package"],
        "water_damage": ["water", "water damage", "wet"],
        "stain": ["stain", "stained", "mark"],
    }

    for issue, keywords in issue_keywords.items():
        for kw in keywords:
            if kw in claim_lower:
                return issue

    return ""


