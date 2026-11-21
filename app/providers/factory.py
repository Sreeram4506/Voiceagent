"""The telephony provider for this process.

Plivo is built once per set of credentials and reused, so every dial goes over the same pooled HTTPS
connection instead of opening (and leaking) a new client per call.
"""

import os
from functools import lru_cache

from app.config import get_settings
from app.providers.base import TelephonyProvider
from app.providers.mock import MockTelephony
from app.providers.plivo import PlivoTelephony


def get_telephony() -> TelephonyProvider:
    """Plivo when credentials are configured; otherwise the offline mock."""
    if os.getenv("TESTING") == "1":
        return MockTelephony()
    auth_id = os.getenv("PLIVO_AUTH_ID")
    if not auth_id:
        return MockTelephony()
    return _plivo(auth_id, os.environ["PLIVO_AUTH_TOKEN"], get_settings().public_url)


@lru_cache(maxsize=4)
def _plivo(auth_id: str, auth_token: str, public_url: str) -> PlivoTelephony:
    return PlivoTelephony(auth_id, auth_token, public_url=public_url)


def reset_telephony() -> None:
    """Forget the shared client, e.g. after credentials change in tests."""
    _plivo.cache_clear()
