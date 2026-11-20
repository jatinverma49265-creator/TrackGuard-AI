from functools import lru_cache
from typing import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session

from src.api.settings import Settings, get_settings


@lru_cache(maxsize=4)
def _engine_for(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True, future=True)


def get_db(settings: Settings = Depends(get_settings)) -> Generator[Session, None, None]:
    if settings.database_url is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="database_not_configured",
        )
    engine = _engine_for(settings.database_url.get_secret_value())
    with Session(engine) as session:
        yield session
