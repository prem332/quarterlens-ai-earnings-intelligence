import os
import logging
from functools import lru_cache
from typing import Optional

from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError, HttpResponseError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _kv_name(name: str) -> str:
    """Convert any casing convention to KV hyphen format: AZURE_SEARCH_KEY → AZURE-SEARCH-KEY"""
    return name.replace("_", "-").upper()


def _env_name(name: str) -> str:
    """Convert to .env underscore format: AZURE-SEARCH-KEY → AZURE_SEARCH_KEY"""
    return name.replace("-", "_").upper()


class KeyVaultClient:
    """
    Wraps Azure Key Vault secret resolution with .env fallback.
    Instantiate once and reuse; SecretClient is thread-safe.
    """

    def __init__(self, vault_url: Optional[str] = None):
        self._vault_url = vault_url or os.getenv("AZURE_KEY_VAULT_URL")
        self._client: Optional[SecretClient] = None

        if self._vault_url:
            try:
                credential = DefaultAzureCredential()
                self._client = SecretClient(
                    vault_url=self._vault_url, credential=credential
                )
                logger.info("KeyVaultClient: connected to %s", self._vault_url)
            except Exception as exc:
                logger.warning(
                    "KeyVaultClient: failed to initialize SecretClient — will use .env fallback only. Error: %s",
                    exc,
                )
        else:
            logger.warning(
                "KeyVaultClient: AZURE_KEY_VAULT_URL not set — using .env fallback only."
            )

    def get_secret(self, name: str) -> str:
        """
        Resolve a secret by name.

        Resolution order:
          1. Azure Key Vault (if client is available)
          2. Environment variable / .env

        Args:
            name: Secret name in any format (hyphens or underscores, any case).
                  e.g. "AZURE-SEARCH-ADMIN-KEY" or "AZURE_SEARCH_ADMIN_KEY"

        Returns:
            Secret value as a string.

        Raises:
            ValueError: if the secret cannot be found in either source.
        """
        kv_key = _kv_name(name)
        env_key = _env_name(name)

        # 1. Try Key Vault
        if self._client is not None:
            try:
                secret = self._client.get_secret(kv_key)
                logger.debug("KeyVaultClient: resolved '%s' from Key Vault", kv_key)
                return secret.value
            except ResourceNotFoundError:
                logger.warning(
                    "KeyVaultClient: '%s' not found in Key Vault — falling back to .env",
                    kv_key,
                )
            except HttpResponseError as exc:
                logger.warning(
                    "KeyVaultClient: HTTP error fetching '%s' — falling back to .env. Error: %s",
                    kv_key,
                    exc,
                )

        # 2. Fall back to .env / environment
        value = os.getenv(env_key)
        if value is not None:
            logger.debug("KeyVaultClient: resolved '%s' from environment", env_key)
            return value

        raise ValueError(
            f"Secret '{name}' not found in Key Vault ('{kv_key}') "
            f"or environment ('{env_key}'). "
            "Ensure it is set in Key Vault or your .env file."
        )

    @lru_cache(maxsize=64)
    def get_secret_cached(self, name: str) -> str:
        """
        Cached variant — suitable for secrets read repeatedly at startup
        (endpoints, deployment names). Do NOT use for secrets that rotate.
        Cache is process-scoped and clears on restart.
        """
        return self.get_secret(name)

kv = KeyVaultClient()