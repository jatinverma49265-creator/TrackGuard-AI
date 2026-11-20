"""Small Supabase Storage REST client for private video uploads."""

from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlsplit

import httpx

from src.api.settings import Settings


VIDEO_BUCKET = "trackguard-videos"
SIGNED_UPLOAD_URL_LIFETIME_SECONDS = 2 * 60 * 60
STORAGE_REQUEST_TIMEOUT_SECONDS = 15.0


class StorageConfigurationError(RuntimeError):
    """Raised when server-side Storage settings are missing or unsafe."""


class StorageUnavailable(RuntimeError):
    """Raised for Storage transport, HTTP, or response failures."""


class StorageMetadataInvalid(RuntimeError):
    """Raised when Storage returns incomplete object metadata."""


@dataclass(frozen=True)
class StoredObjectInfo:
    size_bytes: int
    content_type: str


class SupabaseStorageClient:
    def __init__(self, settings: Settings, http_client: httpx.Client | None = None) -> None:
        if settings.supabase_url is None or settings.supabase_storage_secret_key is None:
            raise StorageConfigurationError("Storage is not configured")

        base_url = str(settings.supabase_url).rstrip("/")
        if urlsplit(base_url).scheme.lower() != "https":
            raise StorageConfigurationError("Storage requires HTTPS")

        self._api_base = f"{base_url}/storage/v1"
        self._secret = settings.supabase_storage_secret_key.get_secret_value()
        self._http = http_client or httpx.Client(timeout=STORAGE_REQUEST_TIMEOUT_SECONDS)
        self._owns_http_client = http_client is None

    def close(self) -> None:
        if self._owns_http_client:
            self._http.close()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "apikey": self._secret,
            "Accept": "application/json",
        }

    def create_signed_upload_url(self, object_key: str) -> str:
        encoded_path = quote(f"{VIDEO_BUCKET}/{object_key}", safe="/")
        endpoint = f"{self._api_base}/object/upload/sign/{encoded_path}"
        try:
            response = self._http.post(
                endpoint,
                json={},
                headers=self._headers,
                timeout=STORAGE_REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code >= 400:
                raise StorageUnavailable("Storage could not create an upload URL")
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise StorageUnavailable("Storage could not create an upload URL") from None

        response_url = payload.get("url") if isinstance(payload, dict) else None
        if not isinstance(response_url, str) or not response_url:
            raise StorageUnavailable("Storage returned an invalid upload URL")

        signed_url = (
            response_url
            if response_url.startswith(("https://", "http://"))
            else urljoin(f"{self._api_base}/", response_url.lstrip("/"))
        )
        actual = urlsplit(signed_url)
        base = urlsplit(self._api_base)
        expected_path = f"/storage/v1/object/upload/sign/{VIDEO_BUCKET}/{object_key}"
        if (
            actual.scheme != "https"
            or actual.hostname != base.hostname
            or actual.port != base.port
            or unquote(actual.path) != expected_path
            or not actual.query
            or actual.username
            or actual.password
        ):
            raise StorageUnavailable("Storage returned an invalid upload URL")
        return signed_url

    def get_object_info(self, object_key: str) -> StoredObjectInfo | None:
        encoded_path = quote(f"{VIDEO_BUCKET}/{object_key}", safe="/")
        endpoint = f"{self._api_base}/object/info/{encoded_path}"
        try:
            response = self._http.get(
                endpoint,
                headers=self._headers,
                timeout=STORAGE_REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 404:
                return None
            if response.status_code >= 400:
                raise StorageUnavailable("Storage could not verify the uploaded object")
            payload: Any = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise StorageUnavailable("Storage could not verify the uploaded object") from None

        if not isinstance(payload, dict):
            raise StorageMetadataInvalid("Storage returned invalid object metadata")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        size_value = metadata.get("size", payload.get("size"))
        content_type = (
            metadata.get("mimetype")
            or metadata.get("contentType")
            or payload.get("content_type")
            or payload.get("mimetype")
        )
        try:
            if isinstance(size_value, bool):
                raise ValueError
            size_bytes = int(size_value)
        except (TypeError, ValueError):
            raise StorageMetadataInvalid("Storage returned invalid object metadata") from None
        if size_bytes < 0 or not isinstance(content_type, str) or not content_type.strip():
            raise StorageMetadataInvalid("Storage returned invalid object metadata")
        return StoredObjectInfo(size_bytes=size_bytes, content_type=content_type.split(";", 1)[0].strip().lower())
