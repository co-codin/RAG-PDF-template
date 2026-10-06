"""API-key verification (framework-agnostic; wired into FastAPI in api/dependencies.py)."""

from __future__ import annotations

import hmac

from pydantic import SecretStr


def verify_api_key(expected: SecretStr | None, provided: str | None) -> bool:
    """True when auth is disabled (no key configured) or the provided key matches.
    Uses a constant-time comparison."""
    if expected is None:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.encode(), expected.get_secret_value().encode())
