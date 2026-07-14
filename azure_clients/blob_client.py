import logging
from typing import Optional

from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

from azure_clients.key_vault_client import kv

logger = logging.getLogger(__name__)


class BlobClient:
    """
    Thin wrapper around Azure BlobServiceClient.
    Instantiate once; BlobServiceClient is thread-safe.
    """

    def __init__(self):
        connection_string = kv.get_secret("AZURE-BLOB-CONNECTION-STRING")
        self._service: BlobServiceClient = BlobServiceClient.from_connection_string(
            connection_string
        )
        logger.info("BlobClient: connected to Blob Storage")

    def upload_blob(
        self,
        container: str,
        blob_path: str,
        data: bytes,
        overwrite: bool = True,
    ) -> None:
        """
        Upload bytes to a blob.

        Args:
            container: Container name (e.g. "raw-documents").
            blob_path: Blob path within the container (e.g. "AAPL/10-Q/2024-Q1.htm").
            data:      Raw bytes to upload.
            overwrite: If True, replaces existing blob. Default True.
        """
        blob = self._service.get_blob_client(container=container, blob=blob_path)
        blob.upload_blob(data, overwrite=overwrite)
        logger.debug("BlobClient: uploaded %s/%s (%d bytes)", container, blob_path, len(data))

    def download_blob(self, container: str, blob_path: str) -> bytes:
        """
        Download a blob as bytes.

        Args:
            container: Container name.
            blob_path: Blob path within the container.

        Returns:
            Blob contents as bytes.

        Raises:
            FileNotFoundError: if the blob does not exist.
        """
        blob = self._service.get_blob_client(container=container, blob=blob_path)
        try:
            stream = blob.download_blob()
            data = stream.readall()
            logger.debug("BlobClient: downloaded %s/%s (%d bytes)", container, blob_path, len(data))
            return data
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Blob not found: {container}/{blob_path}")

    def blob_exists(self, container: str, blob_path: str) -> bool:
        """Check whether a blob exists without downloading it."""
        blob = self._service.get_blob_client(container=container, blob=blob_path)
        return blob.exists()

    def list_blobs(self, container: str, prefix: Optional[str] = None) -> list[str]:
        """
        List blob paths in a container, optionally filtered by prefix.

        Args:
            container: Container name.
            prefix:    Path prefix filter (e.g. "AAPL/10-Q/").

        Returns:
            List of blob path strings.
        """
        container_client = self._service.get_container_client(container)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return [b.name for b in blobs]


# Module-level singleton
blob = BlobClient()