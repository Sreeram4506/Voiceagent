import os
import tempfile

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
