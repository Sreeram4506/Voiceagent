from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings


class Base(DeclarativeBase):
    pass


def _make_engine(url: str):
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    pool_args = {}
    if not url.startswith("sqlite"):
        pool_args = {"pool_pre_ping": True, "pool_recycle": 300}
    return create_engine(url, connect_args=connect_args, future=True, **pool_args)


_url = get_settings().database_url
print(f"[DB] Using database: {_url.split('@')[-1] if '@' in _url else _url}")
engine = _make_engine(_url)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def init_db() -> None:
    from app import models  # noqa: F401  (register tables)

    Base.metadata.create_all(engine)
    _ensure_columns()


def _ensure_columns() -> None:
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    tables = inspector.get_table_names()
    # Columns added after a table was first created: create_all does not add them to an existing table.
    added = (("leads", "campaign_id", "INTEGER"), ("calls", "recording_file", "VARCHAR(100)"))
    for table, column, kind in added:
        if table not in tables or column in {c["name"] for c in inspector.get_columns(table)}:
            continue
        exists = "" if engine.dialect.name == "sqlite" else "IF NOT EXISTS "
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {exists}{column} {kind}"))


def get_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
