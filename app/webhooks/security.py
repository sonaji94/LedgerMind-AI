"""
Razorpay Webhook Signature Verification.

Enforces HMAC-SHA256 constant-time signature verification on raw request bytes.
Prevents webhook forgery and tampering attacks.
"""

from __future__ import annotations

import hashlib
import hmac


def verify_razorpay_signature(
    raw_body: bytes,
    signature: str | None,
    secret: str,
) -> bool:
    """
    Validate that the incoming HTTP request payload matches the Razorpay HMAC-SHA256 signature.

    Args:
        raw_body: Exact raw HTTP payload bytes (before JSON parsing).
        signature: Hex digest passed in the 'X-Razorpay-Signature' header.
        secret: Shared secret configured in the Razorpay Webhooks dashboard.

    Returns:
        True if signature is valid and authentic; False otherwise.
    """
    if not signature or not secret:
        return False

    try:
        computed_digest = hmac.new(
            key=secret.encode("utf-8"),
            msg=raw_body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed_digest, signature.strip())
    except Exception:
        return False
