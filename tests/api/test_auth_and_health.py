from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import uuid4

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from src.api.auth import SupabaseTokenVerifier, get_current_user
from src.api.main import create_app
from src.api.settings import Settings


class FakeJwksClient:
    def __init__(self, key):
        self.key = key

    def get_signing_key_from_jwt(self, token):
        return SimpleNamespace(key=self.key, algorithm_name="RS256")


@pytest.fixture
def signing_keys():
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, private_key.public_key()


def make_token(private_key, **overrides):
    now = datetime.now(timezone.utc)
    claims = {
        "sub": str(uuid4()),
        "iss": "https://trackguard-test.supabase.co/auth/v1",
        "aud": "authenticated",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    }
    claims.update(overrides)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key"})


def verifier(public_key):
    settings = Settings(supabase_url="https://trackguard-test.supabase.co")
    return SupabaseTokenVerifier(settings, FakeJwksClient(public_key))


def test_verifies_signature_issuer_audience_expiry_and_uuid_subject(signing_keys):
    private_key, public_key = signing_keys
    verified = verifier(public_key).verify(make_token(private_key, sub=str(uuid4()), email="driver@example.test"))
    assert verified.email == "driver@example.test"

    with pytest.raises(ValueError):
        verifier(public_key).verify(make_token(private_key, iss="https://wrong.example/auth/v1"))
    with pytest.raises(ValueError):
        verifier(public_key).verify(make_token(private_key, aud="other"))
    with pytest.raises(ValueError):
        verifier(public_key).verify(
            make_token(private_key, exp=datetime.now(timezone.utc) - timedelta(minutes=1))
        )
    with pytest.raises(ValueError):
        verifier(public_key).verify(make_token(private_key, sub="not-a-uuid"))


def test_rejects_token_signed_by_another_key(signing_keys):
    _, public_key = signing_keys
    other_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with pytest.raises(ValueError):
        verifier(public_key).verify(make_token(other_key))


def test_health_is_minimal_and_public_and_missing_bearer_is_401(monkeypatch):
    monkeypatch.delenv("TRACKGUARD_SUPABASE_URL", raising=False)
    client = TestClient(create_app())

    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert client.get("/me").status_code == 401


def test_readiness_is_separate_and_sanitized_without_database(monkeypatch):
    monkeypatch.delenv("TRACKGUARD_DATABASE_URL", raising=False)
    response = TestClient(create_app()).get("/ready")
    assert response.status_code == 503
    assert response.json() == {"detail": "database_not_configured"}
    assert "password" not in response.text.lower()

