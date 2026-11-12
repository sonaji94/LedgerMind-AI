"""
LedgerMind AI — PII Masking & Data Tokenization Utility.

Redacts and tokenizes sensitive Personally Identifiable Information (PII)
prior to transmission to external LLM APIs (OpenAI, Gemini, Anthropic) or public SAR exports:
- Account identifiers (UUIDs / internal account strings)
- IP addresses (IPv4 / IPv6)
- Credit card Primary Account Numbers (PANs)
- Customer names, emails, and phone numbers
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# Regex matchers for financial PII
IP_REGEX = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CARD_PAN_REGEX = re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
UUID_ACCOUNT_REGEX = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
NAMED_ACCOUNT_REGEX = re.compile(r"\bacc-[a-zA-Z0-9_\-]+\b")


class PIIMasker:
    """Security utility tokenizing and scrubbing financial PII."""

    @staticmethod
    def mask_ip(ip_str: str) -> str:
        """Mask middle octets of IPv4: 192.168.1.45 -> 192.***.***.45."""
        parts = ip_str.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.***.***.{parts[3]}"
        return "[MASKED_IP]"

    @staticmethod
    def mask_card_pan(pan_str: str) -> str:
        """Preserve only the last 4 digits of card number."""
        clean = re.sub(r"[-\s]", "", pan_str)
        if len(clean) >= 4:
            return f"****-****-****-{clean[-4:]}"
        return "****-****-****-****"

    @staticmethod
    def tokenize_account(account_id: str) -> str:
        """Generate a deterministic masked token for an account identifier."""
        short_hash = hashlib.sha256(account_id.encode("utf-8")).hexdigest()[:8].upper()
        return f"ACC-TOKEN-{short_hash}"

    @classmethod
    def scrub_text(cls, text: str) -> tuple[str, dict[str, str]]:
        """
        Scrub all recognized PII patterns from freeform text or JSON strings.
        Returns:
            tuple of (scrubbed_text, token_mapping_table)
        """
        token_map: dict[str, str] = {}

        # 1. Mask Card PANs
        def _replace_pan(m: re.Match[str]) -> str:
            raw = m.group(0)
            masked = cls.mask_card_pan(raw)
            token_map[masked] = raw
            return masked

        scrubbed = CARD_PAN_REGEX.sub(_replace_pan, text)

        # 2. Mask IP Addresses
        def _replace_ip(m: re.Match[str]) -> str:
            raw = m.group(0)
            masked = cls.mask_ip(raw)
            token_map[masked] = raw
            return masked

        scrubbed = IP_REGEX.sub(_replace_ip, scrubbed)

        # 3. Mask Emails
        scrubbed = EMAIL_REGEX.sub("[REDACTED_EMAIL]", scrubbed)

        # 4. Mask Phones
        scrubbed = PHONE_REGEX.sub("[REDACTED_PHONE]", scrubbed)

        # 5. Tokenize UUID Accounts
        def _replace_uuid(m: re.Match[str]) -> str:
            raw = m.group(0)
            tok = cls.tokenize_account(raw)
            token_map[tok] = raw
            return tok

        scrubbed = UUID_ACCOUNT_REGEX.sub(_replace_uuid, scrubbed)

        # 6. Tokenize Named Accounts (e.g. acc-9b12-treasury)
        def _replace_named_acc(m: re.Match[str]) -> str:
            raw = m.group(0)
            tok = cls.tokenize_account(raw)
            token_map[tok] = raw
            return tok

        scrubbed = NAMED_ACCOUNT_REGEX.sub(_replace_named_acc, scrubbed)

        return scrubbed, token_map

    @classmethod
    def scrub_dict(cls, data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
        """Recursively scrub dictionary payloads prior to LLM submission."""
        import json

        raw_json = json.dumps(data)
        scrubbed_json, token_map = cls.scrub_text(raw_json)
        return json.loads(scrubbed_json), token_map
