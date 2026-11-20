from dataclasses import dataclass
from typing import Generator
from uuid import UUID, uuid4

from fastapi import Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from src.api.auth import AuthenticatedUser, get_current_user
from src.api.settings import Settings, get_settings
from src.api.storage import StorageConfigurationError, SupabaseStorageClient
from src.db.models import Tenant, User
from src.db.session import get_db


@dataclass(frozen=True)
class TenantContext:
    user_id: UUID
    tenant_id: UUID
    email: str | None


def get_tenant_context(
    principal: AuthenticatedUser = Depends(get_current_user),
    session: Session = Depends(get_db),
) -> TenantContext:
    """Provision and resolve the authenticated user's personal tenant."""
    user_values = {"id": principal.user_id, "email": principal.email}
    session.execute(
        pg_insert(User)
        .values(**user_values)
        .on_conflict_do_update(index_elements=[User.id], set_={"email": principal.email})
    )
    session.execute(
        pg_insert(Tenant)
        .values(id=uuid4(), owner_user_id=principal.user_id, name="Personal workspace")
        .on_conflict_do_nothing(index_elements=[Tenant.owner_user_id])
    )
    session.commit()

    user = session.get(User, principal.user_id)
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="account_inactive")
    tenant = session.scalar(select(Tenant).where(Tenant.owner_user_id == principal.user_id))
    if tenant is None:
        raise HTTPException(status_code=503, detail="tenant_unavailable")
    return TenantContext(user_id=principal.user_id, tenant_id=tenant.id, email=user.email)


def get_storage_client(
    settings: Settings = Depends(get_settings),
) -> Generator[SupabaseStorageClient, None, None]:
    try:
        client = SupabaseStorageClient(settings)
    except StorageConfigurationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="storage_not_configured",
        ) from exc
    try:
        yield client
    finally:
        client.close()
