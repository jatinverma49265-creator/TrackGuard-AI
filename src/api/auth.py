"""Supabase access-token verification with bounded JWKS caching."""

from dataclasses import dataclass
from functools import lru_cache
from uuid import UUID

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import InvalidTokenError, PyJWKClient, PyJWKClientConnectionError, PyJWKClientError

from src.api.settings import Settings, get_settings


bearer_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    user_id: UUID
    email: str | None = None


class AuthenticationUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=8)
def _jwks_client(url: str, cache_seconds: int, refresh_cooldown_seconds: int) -> PyJWKClient:
    return PyJWKClient(
        url,
        cache_jwk_set=True,
        lifespan=cache_seconds,
        cache_keys=False,
        cooldown_duration=refresh_cooldown_seconds,
        timeout=5,
    )


class SupabaseTokenVerifier:
    def __init__(self, settings: Settings, jwks_client: PyJWKClient | None = None) -> None:
        if settings.issuer is None or settings.jwks_url is None:
            raise AuthenticationUnavailable("Supabase Auth is not configured")
        self.settings = settings
        self.jwks_client = jwks_client or _jwks_client(
            settings.jwks_url,
            settings.jwks_cache_seconds,
            settings.jwks_refresh_cooldown_seconds,
        )

    def verify(self, token: str) -> AuthenticatedUser:
        try:
            signing_key = self.jwks_client.get_signing_key_from_jwt(token)
            algorithm = signing_key.algorithm_name
            if algorithm not in self.settings.jwt_algorithms:
                raise InvalidTokenError("Unsupported signing algorithm")
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.settings.jwt_algorithms,
                issuer=self.settings.issuer,
                audience=self.settings.jwt_audience,
                leeway=self.settings.jwt_clock_skew_seconds,
                options={"require": ["sub", "exp", "iss", "aud"]},
            )
            user_id = UUID(claims["sub"])
        except PyJWKClientConnectionError as exc:
            raise AuthenticationUnavailable("Could not refresh Supabase signing keys") from exc
        except (PyJWKClientError, InvalidTokenError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid access token") from exc

        email = claims.get("email")
        return AuthenticatedUser(user_id=user_id, email=email if isinstance(email, str) else None)


def get_token_verifier(settings: Settings = Depends(get_settings)) -> SupabaseTokenVerifier:
    try:
        return SupabaseTokenVerifier(settings)
    except AuthenticationUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="authentication_not_configured",
        ) from exc


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> AuthenticatedUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="authentication_required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        verifier = SupabaseTokenVerifier(settings)
        return verifier.verify(credentials.credentials)
    except AuthenticationUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="authentication_temporarily_unavailable",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid_access_token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc
