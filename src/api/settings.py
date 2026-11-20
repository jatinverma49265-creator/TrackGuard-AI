"""Environment-backed API settings. Secret values are never checked into source."""

from functools import lru_cache
from typing import Optional

from pydantic import AnyHttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TRACKGUARD_", extra="ignore")

    database_url: Optional[SecretStr] = None
    migration_database_url: Optional[SecretStr] = None
    supabase_url: Optional[AnyHttpUrl] = None
    supabase_storage_secret_key: Optional[SecretStr] = None
    jwt_audience: str = "authenticated"
    jwt_algorithms: list[str] = ["ES256", "RS256", "EdDSA"]
    jwks_cache_seconds: int = 300
    jwks_refresh_cooldown_seconds: int = 10
    jwt_clock_skew_seconds: int = 5

    @field_validator("jwks_cache_seconds")
    @classmethod
    def validate_jwks_cache_seconds(cls, value: int) -> int:
        if not 1 <= value <= 900:
            raise ValueError("JWKS cache duration must be between 1 and 900 seconds")
        return value

    @field_validator("jwks_refresh_cooldown_seconds")
    @classmethod
    def validate_jwks_refresh_cooldown_seconds(cls, value: int) -> int:
        if not 0 <= value <= 60:
            raise ValueError("JWKS refresh cooldown must be between 0 and 60 seconds")
        return value

    @field_validator("jwt_clock_skew_seconds")
    @classmethod
    def validate_jwt_clock_skew_seconds(cls, value: int) -> int:
        if not 0 <= value <= 60:
            raise ValueError("JWT clock skew must be between 0 and 60 seconds")
        return value

    @property
    def issuer(self) -> Optional[str]:
        if self.supabase_url is None:
            return None
        return f"{str(self.supabase_url).rstrip('/')}/auth/v1"

    @property
    def jwks_url(self) -> Optional[str]:
        issuer = self.issuer
        return f"{issuer}/.well-known/jwks.json" if issuer else None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
