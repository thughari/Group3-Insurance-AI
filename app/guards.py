import re
from dataclasses import dataclass


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""


# Pattern → refusal message for prohibited requests
BLOCK_PATTERNS = {
    # Safety-critical: final decisions & guaranteed quotes
    "final underwriting decision": "I cannot provide final underwriting decisions. All assessments are indicative and subject to review by a licensed underwriter.",
    "guaranteed premium": "I can only provide indicative premium estimates. Final premium is determined after full underwriting review.",
    # Medical advice
    "medical diagnosis": "I cannot provide medical advice or diagnosis. Please consult a qualified medical professional.",
    "diagnose me": "I cannot provide medical advice or diagnosis. Please consult a qualified medical professional.",
    "prescribe me": "I cannot provide medical advice or prescribe medication.",
    "what medicine should": "I cannot provide medical advice or prescribe medication.",
}

# Prompt injection patterns (regex)
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)",
    r"ignore\s+all\s+instructions",
    r"you\s+are\s+now\s+(?:a|an|the)\b",
    r"disregard\s+(all\s+)?(previous|prior|your)\s+(instructions|rules|guidelines)",
    r"pretend\s+you\s+are",
    r"reveal\s+(your|the)\s+(system|initial)\s+prompt",
    r"show\s+me\s+your\s+(system|initial)\s+prompt",
    r"what\s+is\s+your\s+system\s+prompt",
    r"act\s+as\s+(?:a|an)\s+(?:different|new)\b",
    r"override\s+(your|all)\s+(safety|rules|instructions)",
    r"jailbreak",
]

# PHI / PII leakage patterns
PHI_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",          # SSN format
    r"\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b",  # Credit card format
    r"\b\d{12}\b",                       # Aadhaar format
    r"social\s+security\s+number",
    r"\bssn\b",
    r"\baadhaar\b",
    r"credit\s+card\s+number",
    r"bank\s+account\s+number",
    r"my\s+(?:password|pin)\s+is",
]


def apply_guardrails(text: str) -> GuardResult:
    lower = text.lower()

    # 1. Check direct block patterns
    for pattern, reason in BLOCK_PATTERNS.items():
        if pattern in lower:
            return GuardResult(blocked=True, reason=reason)

    # 2. Check prompt injection
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower):
            return GuardResult(
                blocked=True,
                reason="I cannot process this request. It appears to contain instructions that would compromise my safety guidelines.",
            )

    # 3. Check PHI / PII leakage
    for pattern in PHI_PATTERNS:
        if re.search(pattern, lower):
            return GuardResult(
                blocked=True,
                reason="For your security, please do not share sensitive personal information such as SSN, Aadhaar, credit card numbers, or bank details in this chat.",
            )

    return GuardResult(blocked=False)
