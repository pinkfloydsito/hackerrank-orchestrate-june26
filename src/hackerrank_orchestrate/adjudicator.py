"""Adjudication stage: Simple rule-based claim status decision.

This module derives `claim_status` and `claim_status_justification`
from evidence evaluation + visual findings. No model inference here.
"""

from typing import Dict, List
from dataclasses import dataclass
from hackerrank_orchestrate.perception import VisualFindings
from hackerrank_orchestrate.evidence_evaluator import EvidenceEvaluation, _extract_claimed_issue
from hackerrank_orchestrate.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ClaimDecision:
    """Final claim decision."""

    claim_status: str
    claim_status_justification: str


def adjudicate(
    findings: VisualFindings,
    evidence: EvidenceEvaluation,
    claim_text: str,
) -> ClaimDecision:
    """Adjudicate the claim based on evidence and visual findings.

    Deterministic rule engine. No stochastic models.
    """
    risk_flags = evidence.risk_flags

    # --- Rule 0: Wrong object → contradicted ---
    if "wrong_object" in risk_flags and findings.confidence > 0.4:
        justification = (
            f"The submitted image(s) show a different object than the claimed "
            f"{findings.object_part}. This contradicts the claim."
        )
        return ClaimDecision(
            claim_status="contradicted",
            claim_status_justification=justification.strip(),
        )

    # --- Rule 1: Evidence not sufficient → not_enough_information ---
    if not evidence.evidence_standard_met:
        justification = (
            f"The submitted images do not provide sufficient evidence to evaluate the claim. "
            f"{evidence.evidence_standard_met_reason}"
        )
        return ClaimDecision(
            claim_status="not_enough_information",
            claim_status_justification=justification.strip(),
        )

    # --- Rule 2: Claim mismatch + high confidence → contradicted ---
    if "claim_mismatch" in risk_flags and findings.confidence > 0.5:
        justification = (
            f"The images show {findings.visible_issue} on the {findings.object_part}, "
            f"but the user claimed something different. The visual evidence contradicts the claim."
        )
        if "user_history_risk" in risk_flags:
            justification += " User history also shows prior rejected or exaggerated claims."
        return ClaimDecision(
            claim_status="contradicted",
            claim_status_justification=justification.strip(),
        )

    # --- Rule 3: No visible damage but claim says damage → contradicted ---
    if findings.visible_issue == "none" and findings.confidence > 0.35:
        justification = (
            f"The {findings.object_part} is visible and shows no damage, "
            f"contradicting the claim."
        )
        return ClaimDecision(
            claim_status="contradicted",
            claim_status_justification=justification.strip(),
        )

    # --- Rule 4: Valid evidence, matching issue, good confidence → supported ---
    if findings.confidence > 0.35 and findings.visible_issue != "none" and findings.visible_issue != "unknown":
        justification = (
            f"The submitted image(s) directly show {findings.visible_issue} on the {findings.object_part}. "
        )
        if "user_history_risk" in risk_flags:
            justification += "User history adds risk context but the visual evidence is clear."
        else:
            justification += "The visual evidence supports the claim."
        return ClaimDecision(
            claim_status="supported",
            claim_status_justification=justification.strip(),
        )

    # --- Rule 5: Low confidence but something visible → not_enough_information ---
    if findings.visible_issue not in ("none", "unknown") and findings.confidence <= 0.35:
        justification = (
            f"The images may show {findings.visible_issue} but the confidence is too low "
            f"({findings.confidence:.2f}) to make a definitive decision."
        )
        return ClaimDecision(
            claim_status="not_enough_information",
            claim_status_justification=justification.strip(),
        )

    # --- Rule 6: Unknown issue with low confidence → not_enough_information ---
    if findings.visible_issue == "unknown" and findings.confidence <= 0.5:
        justification = (
            f"The visual evidence is ambiguous. {evidence.evidence_standard_met_reason}"
        )
        return ClaimDecision(
            claim_status="not_enough_information",
            claim_status_justification=justification.strip(),
        )

    # --- Rule 7: Unknown issue but claim says damage → contradicted ---
    if findings.visible_issue == "unknown" and findings.confidence > 0.5:
        claimed_issue = _extract_claimed_issue(claim_text)
        if claimed_issue:
            justification = (
                f"The images do not clearly show {claimed_issue} despite adequate visibility. "
                f"The claim is contradicted by visual evidence."
            )
        else:
            justification = (
                f"The images do not show the claimed damage clearly. "
                f"The claim is contradicted by visual evidence."
            )
        return ClaimDecision(
            claim_status="contradicted",
            claim_status_justification=justification.strip(),
        )

    # --- Fallback ---
    justification = (
        f"The visual evidence is ambiguous. {evidence.evidence_standard_met_reason}"
    )
    return ClaimDecision(
        claim_status="not_enough_information",
        claim_status_justification=justification.strip(),
    )



