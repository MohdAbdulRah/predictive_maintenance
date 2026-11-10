from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from config import load_config, ROOT
from database.models import Base

_engine = None
_Session = None


def get_engine(url: str | None = None):
    global _engine, _Session
    if _engine is None or url is not None:
        url = url or load_config()["database"]["url"]
        if url.startswith("sqlite:///") and not url.startswith("sqlite:////") and ":memory:" not in url:
            rel = url.replace("sqlite:///", "")
            p = (ROOT / rel).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            url = f"sqlite:///{p}"
        kw = {"future": True}
        if url.startswith("sqlite") and ":memory:" in url:
            from sqlalchemy.pool import StaticPool
            kw.update(connect_args={"check_same_thread": False}, poolclass=StaticPool)
        _engine = create_engine(url, **kw)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        Base.metadata.create_all(_engine)
    return _engine


@contextmanager
def session_scope():
    get_engine()
    s = _Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()
