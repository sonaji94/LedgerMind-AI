"""
Unit tests for the production deployment verification script (scripts/verify_live_deployment.py).
"""

from __future__ import annotations

import hashlib
import hmac

from scripts.verify_live_deployment import sign_payload


def test_sign_payload_rfc2104_hmac_sha256() -> None:
    secret = "rzp_whsec_test_secret_12345"
    payload = b'{"event":"payment.captured","payload":{"payment":{"entity":{"id":"pay_123"}}}}'

    computed_sig = sign_payload(payload, secret)
    expected_sig = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()

    assert computed_sig == expected_sig
    assert len(computed_sig) == 64  # SHA256 hex digest length


def test_sign_payload_differs_on_tampered_payload() -> None:
    secret = "rzp_whsec_test_secret_12345"
    payload1 = b'{"amount":1000}'
    payload2 = b'{"amount":2000}'

    sig1 = sign_payload(payload1, secret)
    sig2 = sign_payload(payload2, secret)

    assert sig1 != sig2
