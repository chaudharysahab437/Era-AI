"""Privacy filters for the memory engine.

Two different rules apply:

* SECRETS (API keys, passwords, tokens, private keys...) are NEVER stored --
  not automatically, and not even on an explicit save request.
* SENSITIVE personal data (health, financial, government IDs) is never stored
  *automatically*. It can be stored only through an explicit save with
  ``allow_sensitive=True``; it is then masked in API responses unless the
  caller asks to reveal it, and is not sent to the AI model unless the user
  opts that memory in.

Nothing in this module logs or returns the offending text -- only category
names -- so a detection can never leak the value it detected.
"""

from __future__ import annotations

import re
from typing import List

from .text_utils import split_sentences

# ---------------------------------------------------------------- secrets ----
_SECRET_PATTERNS = [
    ("api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}")),
    ("api_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")),
    ("api_key", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("api_key", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")),
    ("api_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("token", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{5,}")),
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("token", re.compile(r"\bbearer\s+[A-Za-z0-9_\-\.=]{16,}", re.I)),
    (
        "credential",
        re.compile(
            r"\b(?:password|passwd|pwd|passcode|pass\s?phrase|api[ _\-]?key|secret(?:\s?key)?|access[ _\-]?key|auth(?:entication)?[ _\-]?token|token|pin|otp)\b\s*(?:is|are|was|=|:)\s*\S{3,}",
            re.I,
        ),
    ),
    ("credential", re.compile(r"\b(?:key|token|secret|password)\b[^\n]{0,24}?[A-Za-z0-9_\-]{28,}", re.I)),
    ("token", re.compile(r"\b[A-Fa-f0-9]{32,}\b")),
]

# -------------------------------------------------------------- sensitive ----
_HEALTH = re.compile(
    r"\b(diagnos\w*|prescri\w*|medicat\w*|medicine|symptom\w*|surgery|surgical|therap(?:y|ist)|cancer|diabet\w*|"
    r"asthma|hypertension|depress\w*|anxiety|bipolar|schizophren\w*|hiv|aids|blood\s+(?:pressure|sugar|group|type|test)|"
    r"allerg\w*|chronic|mental\s+health|pregnan\w*|disorder|adhd|autis\w*|migraine|infection|epilep\w*|"
    r"medical\s+(?:history|condition|record|report)|lab\s+results?|vaccin\w*|std|sti)\b",
    re.I,
)
_FINANCIAL = re.compile(
    r"\b(bank\s+account|account\s+(?:number|no\b)|ifsc|upi(?:\s+id)?|credit\s+card|debit\s+card|card\s+number|cvv|cvc|"
    r"salary|income|net\s+worth|loan|emi|mortgage|debts?|savings|bank\s+balance|ctc|iban|swift|routing\s+number|"
    r"tax\s+return|investments?|portfolio)\b",
    re.I,
)
_GOV_ID_WORDS = re.compile(
    r"\b(aadhaar|aadhar|pan\s+(?:card|number|no\b)|ssn|social\s+security|passport(?:\s+(?:no|number))?|"
    r"driving\s+licen[cs]e|driver'?s\s+licen[cs]e|voter\s+id|national\s+id|tax\s+id|gstin)\b",
    re.I,
)
_GOV_ID_PATTERNS = [
    re.compile(r"(?<!\d)(?<!\d\s)\d{4}\s\d{4}\s\d{4}(?!\s?\d)"),  # Aadhaar-style grouping (not inside a longer digit run, e.g. a card)
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),  # Indian PAN
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN
]
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ \-]?){13,19}\b")


def _luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        alt = not alt
    return total % 10 == 0


def _has_card_number(text: str) -> bool:
    for m in _CARD_CANDIDATE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


def find_secrets(text: str) -> List[str]:
    """Return the *kinds* of secrets present (never the secrets themselves)."""
    kinds: List[str] = []
    for kind, pat in _SECRET_PATTERNS:
        if pat.search(text or "") and kind not in kinds:
            kinds.append(kind)
    return kinds


def classify_sensitive(text: str) -> List[str]:
    """Return sensitive categories present: health / financial / government_id."""
    text = text or ""
    cats: List[str] = []
    if _HEALTH.search(text):
        cats.append("health")
    if _FINANCIAL.search(text) or _has_card_number(text):
        cats.append("financial")
    if _GOV_ID_WORDS.search(text) or any(p.search(text) for p in _GOV_ID_PATTERNS):
        cats.append("government_id")
    return cats


def scrub(text: str) -> str:
    """Sentence-level scrub for text kept in the short-lived turn buffer:
    any sentence with a secret or sensitive data is replaced by a placeholder,
    so raw conversation retention can never silently keep either."""
    out = []
    for sent in split_sentences(text):
        if find_secrets(sent):
            out.append("[secret removed]")
        elif classify_sensitive(sent):
            out.append("[sensitive content not retained]")
        else:
            out.append(sent)
    return " ".join(out)
