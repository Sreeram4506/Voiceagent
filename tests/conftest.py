import os
import tempfile
import time

# Must be set before app modules read settings.
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["TESTING"] = "1"
os.environ["VA_DATABASE_URL"] = f"sqlite:///{_tmp.name}"
os.environ.pop("EXOTEL_ACCOUNT_SID", None)
os.environ.pop("PLIVO_AUTH_ID", None)
os.environ.pop("PLIVO_AUTH_TOKEN", None)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.db import Base, SessionLocal, engine, init_db  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    Base.metadata.drop_all(engine)
    init_db()
    yield


@pytest.fixture(autouse=True)
def no_dialer_threads(monkeypatch):
    """Tests drive the auto-dialer one pass at a time; a background dialer thread would race them.

    Returns the tenant ids the app asked to start a dialer thread for.
    """
    from app.services import autodialer

    spawned: list[int] = []
    monkeypatch.setattr(autodialer, "_spawn_worker", spawned.append)
    return spawned


EVENTUALLY_TIMEOUT_SECONDS = 5.0
EVENTUALLY_POLL_SECONDS = 0.02


def _eventually(check, timeout: float = EVENTUALLY_TIMEOUT_SECONDS):
    """Poll check() until it returns something truthy; return that, or the last falsy result on timeout.

    The test client doesn't wait for a websocket handler to finish after the socket closes, so a call
    is saved a moment after the `with` block ends, just as it is after a real hang-up.
    """
    deadline = time.monotonic() + timeout
    result = check()
    while not result and time.monotonic() < deadline:
        time.sleep(EVENTUALLY_POLL_SECONDS)
        result = check()
    return result


@pytest.fixture
def eventually():
    return _eventually


@pytest.fixture
def session():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture
def client():
    from app.main import app

    with TestClient(app) as c:
        yield c
