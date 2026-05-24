import re
from dataclasses import dataclass
from app.cache import guardrail_cache
from llm_guard.input_scanners import Anonymize, PromptInjection
from llm_guard import scan_prompt as _sp


@dataclass
class GuardResult:
    blocked: bool
    reason: str = ""


# ── Tier 1: Domain-specific fast regex ──────────────────────────────────────
# These are insurance-domain rules that no ML model can know better than us.
BLOCK_PATTERNS = {
    "final underwriting decision": (
        "I cannot provide final underwriting decisions. "
        "All assessments are indicative and subject to review by a licensed underwriter."
    ),
    "guaranteed premium": (
        "I can only provide indicative premium estimates. "
        "Final premium is determined after full underwriting review."
    ),
    "medical diagnosis": (
        "I cannot provide medical advice or diagnosis. "
        "Please consult a qualified medical professional."
    ),
    "diagnose me": (
        "I cannot provide medical advice or diagnosis. "
        "Please consult a qualified medical professional."
    ),
    "prescribe me": "I cannot provide medical advice or prescribe medication.",
    "what medicine should": "I cannot provide medical advice or prescribe medication.",
}


# ── Tier 2 & 3: LLM Guard scanners ─────────────────────────────────────────
# Loaded once at startup. Models are downloaded from HuggingFace on first run
# and cached locally for subsequent runs.
_SCANNERS_READY = False
_anonymize_scanner = None
_injection_scanner = None
_scan_prompt = None

def _init_llm_guard():
    """Lazy-initialize LLM Guard scanners. Called once at first guardrail check."""
    global _SCANNERS_READY, _anonymize_scanner, _injection_scanner, _scan_prompt
    if _SCANNERS_READY:
        return

    try:

        # Anonymize: Detects PII (SSN, Credit Cards, Phone, Email) using Presidio + spaCy
        # entity_types=None means detect ALL supported entity types
        _anonymize_scanner = Anonymize(
            preamble="",
            allowed_names=[],
            hidden_names=[],
            recognizer_conf=None,
            language="en",
        )

        # PromptInjection: ML classifier fine-tuned specifically for injection detection
        # Uses deepset/deberta-v3-base-injection model from HuggingFace
        _injection_scanner = PromptInjection(threshold=0.75)

        _scan_prompt = _sp
        _SCANNERS_READY = True
        print("✅ LLM Guard scanners initialized (Anonymize + PromptInjection).")

    except Exception as e:
        print(f"⚠️  LLM Guard scanners could not be initialized: {e}. Falling back to regex.")
        _SCANNERS_READY = True  # Mark as attempted so we don't retry every request


def _run_llm_guard(text: str) -> GuardResult:
    """Run LLM Guard Anonymize + PromptInjection scanners."""
    if not _scan_prompt or not _anonymize_scanner or not _injection_scanner:
        return GuardResult(blocked=False)

    try:
        _, results_valid, _ = _scan_prompt(
            [_anonymize_scanner, _injection_scanner],
            text,
        )

        # results_valid is a dict: {scanner_name: True (safe) / False (blocked)}
        if not results_valid.get("Anonymize", True):
            return GuardResult(
                blocked=True,
                reason=(
                    "For your security, please do not share sensitive personal "
                    "information such as SSNs, credit card numbers, or contact "
                    "details in this chat."
                ),
            )

        if not results_valid.get("PromptInjection", True):
            return GuardResult(
                blocked=True,
                reason=(
                    "I cannot process this request. It appears to contain "
                    "instructions that would compromise my safety guidelines."
                ),
            )

    except Exception as e:
        print(f"LLM Guard scan error: {e}")

    return GuardResult(blocked=False)


def apply_guardrails(text: str) -> GuardResult:
    """
    3-Tier guardrail pipeline:
      Tier 1 — Fast domain regex   (instant, insurance-specific)
      Tier 2 — LLM Guard Anonymize (Presidio NER-based PII detection)
      Tier 3 — LLM Guard PromptInjection (ML classifier)
    All results are cached per-query.
    """
    # Cache check — guardrails are deterministic for the same input
    cached = guardrail_cache.get(text)
    if cached is not None:
        return cached

    lower = text.lower()

    # ── Tier 1: Fast domain-specific keyword blocks ──────────────────────────
    for pattern, reason in BLOCK_PATTERNS.items():
        if pattern in lower:
            result = GuardResult(blocked=True, reason=reason)
            guardrail_cache.set(text, result)
            return result

    # ── Tiers 2 & 3: LLM Guard (PII + Injection) ────────────────────────────
    _init_llm_guard()  # No-op after first call
    result = _run_llm_guard(text)

    guardrail_cache.set(text, result)
    return result
