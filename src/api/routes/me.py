from fastapi import APIRouter, Depends

from src.api.dependencies import TenantContext, get_tenant_context
from src.api.schemas import MeResponse


router = APIRouter(tags=["account"])


@router.get("/me", response_model=MeResponse)
def get_me(context: TenantContext = Depends(get_tenant_context)) -> MeResponse:
    return MeResponse(
        user_id=context.user_id,
        email=context.email,
        tenant_id=context.tenant_id,
    )
